#include "rt.h"

#include "cpu_moe.h"
#include "fakequant.h"
#include "fp8_gemv.h"
#include "kernels.h"

#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

template <class T>
static T* dalloc(size_t n) {
  void* p;
  CHECK(hipMalloc(&p, n * sizeof(T)));
  CHECK(hipMemset(p, 0, n * sizeof(T)));
  return (T*)p;
}

// ------------------------------------------------------------------ rotary frequencies
// precompute_freqs_cis, in the reference's fp32 op order

static std::vector<float> rope_freqs(int dim, int original_seq_len, float base, float factor, float beta_fast,
                                     float beta_slow) {
  std::vector<float> f(dim / 2);
  for (int i = 0; i < dim / 2; i++) f[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
  if (original_seq_len > 0) {
    auto corrected = [&](double rot) { return dim * log(original_seq_len / (rot * 2 * M_PI)) / (2 * log((double)base)); };
    const int low = std::max((int)floor(corrected(beta_fast)), 0);
    const int high = std::min((int)ceil(corrected(beta_slow)), dim - 1);
    for (int i = 0; i < dim / 2; i++) {
      const float ramp = std::min(1.0f, std::max(0.0f, ((float)i - (float)low) / std::max((float)(high - low), 1e-3f)));
      const float smooth = 1.0f - ramp;
      f[i] = f[i] / factor * (1.0f - smooth) + f[i] * smooth;
    }
  }
  return f;
}

// ------------------------------------------------------------------ construction

Runtime::Runtime(const Model& model, RtOptions options, const std::string& fp8_path)
    : m(model), hp(model.hp), opt(options) {
  CHECK(hipSetDevice(opt.gpu));
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // rotary frequencies: window-only layers at rope_base, compressed layers at
  // compress_rope_base with YaRN (as Engine::rope)
  const auto f_win = rope_freqs(hp.rope_dim, 0, hp.rope_base, 1, 0, 0);
  const auto f_cmp = rope_freqs(hp.rope_dim, hp.yarn_orig_ctx, hp.compress_rope_base, hp.yarn_factor, hp.yarn_beta_fast,
                                hp.yarn_beta_slow);
  const int RD2 = hp.rope_dim / 2;
  freqs_buf = dalloc<float>(2 * RD2);
  CHECK(hipMemcpy(freqs_buf, f_win.data(), RD2 * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(freqs_buf + RD2, f_cmp.data(), RD2 * 4, hipMemcpyHostToDevice));

  // the fp8 weights: the whole data section of fp8pack's GGUF in one device buffer
  gguf_init_params gp = {true, nullptr};
  gguf_context* g = gguf_init_from_file(fp8_path.c_str(), gp);
  if (!g) throw std::runtime_error("cannot read " + fp8_path);
  {
    int fd = open(fp8_path.c_str(), O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    const size_t off = gguf_get_data_offset(g), n = st.st_size - off;
    void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(hipMalloc(&fp8_buf, n));
    CHECK(hipMemcpy(fp8_buf, (char*)map + off, n, hipMemcpyHostToDevice));
    munmap(map, st.st_size);
    close(fd);
    fprintf(stderr, "# runtime: %s of fp8 weights on GPU %d\n", fmt_bytes((double)n).c_str(), opt.gpu);
  }
  auto fp8w = [&](int L, const char* name) {
    Fp8W w;
    const std::string n = "blk." + std::to_string(L) + "." + name + ".weight";
    const int64_t id = gguf_find_tensor(g, n.c_str()), ids = gguf_find_tensor(g, (n + ".scale").c_str());
    if (id < 0) return w;
    if (ids < 0) throw std::runtime_error("no scale for " + n);
    w.q = (uint8_t*)fp8_buf + gguf_get_tensor_offset(g, id);
    w.s = (uint8_t*)fp8_buf + gguf_get_tensor_offset(g, ids);
    w.K = (int)gguf_get_tensor_ne(g, id)[0];
    w.N = (int)gguf_get_tensor_ne(g, id)[1];
    return w;
  };
  // the small weights: device copies of the host tensors (as stored, or converted to f32)
  size_t small_bytes = 0;
  auto upload = [&](const void* src, size_t n) {
    void* p;
    CHECK(hipMalloc(&p, n));
    CHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice));
    owned.push_back(p);
    small_bytes += n;
    return p;
  };
  auto f32 = [&](const ggml_tensor* t) -> const float* {
    if (!t) return nullptr;
    if (t->type == GGML_TYPE_F32) return (const float*)upload(t->data, ggml_nbytes(t));
    std::vector<float> v(ggml_nelements(t));  // the reference applies these in fp32
    ggml_get_type_traits(t->type)->to_float(t->data, v.data(), v.size());
    return (const float*)upload(v.data(), v.size() * 4);
  };
  auto bf16 = [&](const ggml_tensor* t) -> const uint16_t* {
    if (!t) return nullptr;
    if (t->type != GGML_TYPE_BF16) throw std::runtime_error(std::string("expected bf16: ") + t->name);
    return (const uint16_t*)upload(t->data, ggml_nbytes(t));
  };

  W.resize(hp.n_layer);
  for (int L = 0; L < hp.n_layer; L++) {
    const Layer& l = m.layers[L];
    if (!l.loaded) continue;
    LayerW& w = W[L];
    w.q_a = fp8w(L, "attn_q_a");
    w.q_b = fp8w(L, "attn_q_b");
    w.kv = fp8w(L, "attn_kv");
    w.out_a = fp8w(L, "attn_output_a");
    w.out_b = fp8w(L, "attn_output_b");
    w.idx_q_b = fp8w(L, "indexer.attn_q_b");
    w.gate_sh = fp8w(L, "ffn_gate_shexp");
    w.up_sh = fp8w(L, "ffn_up_shexp");
    w.down_sh = fp8w(L, "ffn_down_shexp");
    w.engram_kv = fp8w(L, "engram_kv");
    if (!w.q_a.q || !w.out_b.q || !w.down_sh.q) throw std::runtime_error("fp8 file lacks layer " + std::to_string(L));
    w.attn_norm = f32(l.attn_norm);
    w.q_a_norm = f32(l.q_a_norm);
    w.kv_norm = f32(l.kv_norm);
    w.sinks = f32(l.sinks);
    w.ffn_norm = f32(l.ffn_norm);
    w.gate_inp = f32(l.gate_inp);
    w.gate_bias = f32(l.gate_bias);
    w.hc_attn_fn = f32(l.hc_attn_fn);
    w.hc_attn_base = f32(l.hc_attn_base);
    w.hc_attn_scale = f32(l.hc_attn_scale);
    w.hc_ffn_fn = f32(l.hc_ffn_fn);
    w.hc_ffn_base = f32(l.hc_ffn_base);
    w.hc_ffn_scale = f32(l.hc_ffn_scale);
    w.cmp_kv = f32(l.cmp_kv);  // bf16 in the file; the compressor pools in fp32
    w.cmp_gate = f32(l.cmp_gate);
    w.cmp_norm = f32(l.cmp_norm);
    w.idx_k = f32(l.idx_k);
    w.idx_k_norm = f32(l.idx_k_norm);
    w.idx_proj = bf16(l.idx_proj);
    w.engram_gate = f32(l.engram_gate);
    w.freqs = freqs_buf + (l.kind.ratio ? RD2 : 0);
  }
  gguf_free(g);
  out_w = bf16(m.output);
  out_norm = f32(m.output_norm);
  fprintf(stderr, "# runtime: %s of small weights on GPU %d\n", fmt_bytes((double)small_bytes).c_str(), opt.gpu);
  cpu_moe.reset(new CpuMoeKernel(hp.n_embd, hp.n_ff_exp, hp.n_expert_used, opt.n_threads));

  const int E = hp.n_embd, HC = hp.hc_mult, HD = hp.head_dim, NH = hp.n_head;
  // per-token buffers hold a whole chunk: [kChunk][...]
  const int C = kChunk;
  h = dalloc<float>((size_t)C * E * HC);
  h2 = dalloc<float>((size_t)C * E * HC);
  pre = dalloc<float>(C * HC);
  a_pre = dalloc<float>(C * HC);
  a_post = dalloc<float>(C * HC);
  a_comb = dalloc<float>(C * HC * HC);
  f_pre = dalloc<float>(C * HC);
  f_post = dalloc<float>(C * HC);
  f_comb = dalloc<float>(C * HC * HC);
  mixes = dalloc<float>(C * (2 + HC) * HC);
  x = dalloc<float>((size_t)C * E);
  t1 = dalloc<float>((size_t)C * hp.q_lora);
  qr = dalloc<float>((size_t)C * hp.q_lora);
  q = dalloc<float>((size_t)C * NH * HD);
  kvv = dalloc<float>((size_t)C * HD);
  o = dalloc<float>((size_t)C * NH * HD);
  oa = dalloc<float>((size_t)C * hp.o_groups * hp.o_lora);
  y = dalloc<float>((size_t)C * E);
  ysh = dalloc<float>((size_t)C * E);
  yr = dalloc<float>((size_t)C * E);
  if (opt.canon) {
    canon_rows = dalloc<float>((size_t)(kMoeMaxDies + 1) * C * hp.n_expert_used * E);
    canon_where = dalloc<int>((size_t)C * kMoeIdStride);
  }
  logits_d = dalloc<float>((size_t)C * hp.n_vocab);
  partial = dalloc<float>((size_t)kSplitMax * std::max(hp.n_vocab, 32768));
  A = {dalloc<uint8_t>((size_t)C * NH * HD), dalloc<float>((size_t)C * NH * HD / 32)};
  B = {dalloc<uint8_t>((size_t)C * NH * HD), dalloc<float>((size_t)C * NH * HD / 32)};
  cmp_kvn = dalloc<float>(HD);
  cmp_sc = dalloc<float>(HD);
  lat = dalloc<float>(HD);
  ik = dalloc<float>(hp.idx_dim);
  iq = dalloc<float>(hp.idx_heads * hp.idx_dim);
  iw = dalloc<float>(hp.idx_heads);
  iscore = dalloc<float>(kTopkMaxN);
  gate = dalloc<float>((size_t)C * hp.n_ff_exp);
  up = dalloc<float>((size_t)C * hp.n_ff_exp);
  act = dalloc<float>((size_t)C * hp.n_ff_exp);
  router_logits = dalloc<float>((size_t)C * hp.n_expert);
  {  // device + pinned host staging of the routed experts' inputs, one contiguous block each
    moe_lists_bytes = (size_t)C * kMoeListStride * sizeof(MoeList);
    const size_t bytes = moe_block_bytes(C);
    char* d = dalloc<char>(bytes);
    char* hm;
    CHECK(hipHostMalloc((void**)&hm, bytes, 0));
    d_moe = {(MoeList*)d, (float*)(d + moe_lists_bytes)};
    h_moe = {(MoeList*)hm, (float*)(hm + moe_lists_bytes)};
  }
  rw = dalloc<float>((size_t)C * kMoeIdStride);
  rids = dalloc<int>((size_t)C * kMoeIdStride);
  att_scratch = dalloc<float>((size_t)((hp.window + hp.idx_topk + 31) / 32) * NH * (HD + 2));
  erows = dalloc<float>((size_t)hp.engram_layers.size() * C * hp.engram_cols() * hp.engram_dim);  // [slot][token][rows]
  ekv = dalloc<float>((size_t)C * E * (HC + 1));
  {
    std::vector<int> v(opt.max_ctx);
    for (int i = 0; i < (int)v.size(); i++) v[i] = i;
    iota = dalloc<int>(v.size());
    CHECK(hipMemcpy(iota, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  }
  for (int L : hp.index_source_layers)
    if (L < hp.n_layer && m.layers[L].loaded) topk_buf[L] = dalloc<int>((size_t)C * hp.idx_topk);
  // the sequence state, zeroed (masked-out slots must hold finite values: 0 x NaN would leak)
  S.resize(hp.n_layer);
  for (int L = 0; L < hp.n_layer; L++) {
    const LayerKind& k = m.layers[L].kind;
    if (!m.layers[L].loaded) continue;
    S[L].win = dalloc<float>((size_t)hp.window * HD);
    if (k.kv_source) {
      const int r = k.ratio;
      if (r > 1) {
        S[L].cmp_kv = dalloc<float>((size_t)r * HD);
        S[L].cmp_score = dalloc<float>((size_t)r * HD);
      }
      S[L].comp = dalloc<float>((size_t)opt.max_ctx / r * HD);
      S[L].idx_k = dalloc<float>((size_t)opt.max_ctx / r * hp.idx_dim);
    }
  }
  CHECK(hipHostMalloc(&h_yr, (size_t)C * (opt.canon ? hp.n_expert_used : 1) * E * 4, 0));  // canon: the CPU's rows
  CHECK(hipHostMalloc(&h_erows, hp.engram_layers.size() * C * hp.engram_cols() * hp.engram_dim * 4, 0));
  if (opt.max_ctx > kTopkMaxN) throw std::runtime_error("runtime: --max-ctx above the top-k kernel's limit");
  CHECK(hipEventCreateWithFlags(&ev_staged, hipEventDisableTiming));

  if (!opt.dspark.empty()) dspark.reset(new DSpark(*this, opt.dspark));  // before the pools take the rest
  setup_experts();
}

std::vector<int32_t> Runtime::verify(const std::vector<int32_t>& chunk) {
  const int M = (int)chunk.size(), HD = hp.head_dim, WIN = hp.window;
  if (M < 1 || M > DSpark::kMaxChunk) throw std::runtime_error("verify: chunk of " + std::to_string(M));
  if (!ring_backup) ring_backup = dalloc<float>((size_t)hp.n_layer * DSpark::kMaxChunk * HD);
  verify_p0 = n_past_;
  verify_m = M;
  // the ring slots these positions will overwrite
  for (int L = 0; L < hp.n_layer; L++)
    if (S[L].win)
      for (int i = 0; i < M; i++)
        CHECK(hipMemcpyAsync(ring_backup + ((size_t)L * DSpark::kMaxChunk + i) * HD, S[L].win + (size_t)((verify_p0 + i) % WIN) * HD,
                             HD * 4, hipMemcpyDeviceToDevice, s));
  for (int L = 0; L < hp.n_layer; L++) {
    if (!S[L].cmp_kv) continue;  // ratio > 1 KV sources: the partial group before the chunk
    const int r = m.layers[L].kind.ratio;
    if (!cmp_rows.count(L)) {
      cmp_rows[L] = dalloc<float>((size_t)DSpark::kMaxChunk * 2 * HD);
      cmp_backup[L] = dalloc<float>((size_t)2 * r * HD);
    }
    CHECK(hipMemcpyAsync(cmp_backup[L], S[L].cmp_kv, (size_t)r * HD * 4, hipMemcpyDeviceToDevice, s));
    CHECK(hipMemcpyAsync(cmp_backup[L] + (size_t)r * HD, S[L].cmp_score, (size_t)r * HD * 4, hipMemcpyDeviceToDevice, s));
  }
  verifying = true;
  std::vector<int32_t> next = forward_chunk(chunk);
  verifying = false;
  return next;
}

// M tokens at positions n_past .. in one forward: every token computed exactly as decode()
// would compute it alone (batched kernels are bit-identical per row), the greedy next token
// after each. The sequence advances by M.
std::vector<int32_t> Runtime::forward_chunk(const std::vector<int32_t>& toks) {
  const int M = (int)toks.size(), E = hp.n_embd, HC = hp.hc_mult;
  const int64_t V = hp.n_vocab;
  if (M < 1 || M > kChunk) throw std::runtime_error("forward_chunk: " + std::to_string(M) + " tokens");
  if (n_past_ + M > opt.max_ctx) throw std::runtime_error("runtime: chunk past --max-ctx");
  auto t0 = Clock::now();
  pos = n_past_;
  for (auto& tc : topk_cur) tc.clear();
  last = Times{};
  marks.clear();
  mark("engram.fetch");
  fetch_engram_rows(toks);
  std::vector<float> emb((size_t)M * E), one_hot((size_t)M * HC, 0.0f);
  for (int t = 0; t < M; t++) {
    const ggml_bf16_t* row = (const ggml_bf16_t*)((const char*)m.tok_embd->data + (size_t)toks[t] * m.tok_embd->nb[1]);
    for (int i = 0; i < E; i++) emb[(size_t)t * E + i] = ggml_bf16_to_fp32(row[i]);
    one_hot[(size_t)t * HC] = 1.0f;
  }
  CHECK(hipMemcpyAsync(x, emb.data(), emb.size() * 4, hipMemcpyHostToDevice, s));
  CHECK(hipMemcpyAsync(pre, one_hot.data(), one_hot.size() * 4, hipMemcpyHostToDevice, s));
  for (int t = 0; t < M; t++) k_repeat(x + (size_t)t * E, E, HC, h + (size_t)t * E * HC, s);
  for (int L = 0; L < hp.n_layer; L++) block(L, nullptr, M);
  mark("head");
  k_hc_pre(h, pre, E, x, s, M);
  k_rms_norm(x, out_norm, x, E, M, hp.rms_eps, s);
  k_gemv_bf16_m(out_w, (int)V, E, x, M, logits_d, partial, s);
  std::vector<float> lg((size_t)M * V);
  mark("end");
  CHECK(hipMemcpyAsync(lg.data(), logits_d, lg.size() * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipStreamSynchronize(s));
  std::vector<int32_t> next(M);
  for (int t = 0; t < M; t++)
    next[t] = (int32_t)(std::max_element(lg.begin() + (size_t)t * V, lg.begin() + (size_t)(t + 1) * V) - (lg.begin() + (size_t)t * V));
  if (route_log) write_route_records(toks.data(), M);
  n_past_ += M;
  tokens_.insert(tokens_.end(), toks.begin(), toks.end());
  if (profile) {
    last_groups.clear();
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      float ms;
      CHECK(hipEventElapsedTime(&ms, marks[i].second, marks[i + 1].second));
      last_groups[marks[i].first] += ms;
    }
  }
  last.total_ms = ms_since(t0);
  return next;
}

void Runtime::commit(int n_keep) {
  const int HD = hp.head_dim, WIN = hp.window, M = verify_m;
  if (n_keep < 1 || n_keep > M || n_past_ != verify_p0 + M) throw std::runtime_error("commit without a matching verify");
  for (int L = 0; L < hp.n_layer; L++)
    if (S[L].win)
      for (int i = n_keep; i < M; i++)
        CHECK(hipMemcpyAsync(S[L].win + (size_t)((verify_p0 + i) % WIN) * HD, ring_backup + ((size_t)L * DSpark::kMaxChunk + i) * HD,
                             HD * 4, hipMemcpyDeviceToDevice, s));
  // compressor partial groups: each slot from the last kept position writing it, else as before
  for (auto& [L, rows] : cmp_rows) {
    const int r = m.layers[L].kind.ratio;
    for (int slot = 0; slot < r; slot++) {
      int last = -1;
      for (int i = 0; i < n_keep; i++)
        if ((verify_p0 + i) % r == slot) last = i;
      const float* kv = last >= 0 ? rows + (size_t)last * 2 * HD : cmp_backup[L] + (size_t)slot * HD;
      const float* sc = last >= 0 ? rows + (size_t)last * 2 * HD + HD : cmp_backup[L] + (size_t)(r + slot) * HD;
      CHECK(hipMemcpyAsync(S[L].cmp_kv + (size_t)slot * HD, kv, HD * 4, hipMemcpyDeviceToDevice, s));
      CHECK(hipMemcpyAsync(S[L].cmp_score + (size_t)slot * HD, sc, HD * 4, hipMemcpyDeviceToDevice, s));
    }
  }
  n_past_ = verify_p0 + n_keep;
  tokens_.resize(n_past_);
  if (dspark) {
    const size_t nh = dspark->target_layers.size() * hp.n_embd;
    for (int i = 0; i < n_keep; i++) dspark->seed(verify_p0 + i, dspark->hid_chunk + (size_t)i * nh);
    CHECK(hipMemcpyAsync(dspark->main_hidden, dspark->hid_chunk + (size_t)(n_keep - 1) * nh, nh * 4, hipMemcpyDeviceToDevice, s));
  }
  CHECK(hipStreamSynchronize(s));
  verify_m = 0;
}

DraftResult Runtime::draft(int32_t anchor, bool keep_logits) {
  if (!dspark) throw std::runtime_error("no DSpark draft loaded");
  return dspark->draft(anchor, n_past_ - 1, keep_logits);
}

// The placement's top pairs into the dies' free memory (after everything else is allocated),
// one loader thread per die, and the [layer][expert] table the dispatch kernel reads.
void Runtime::setup_experts() {
  if (opt.placement.empty()) return;
  const int E = hp.n_embd, F = hp.n_ff_exp, X = hp.n_expert, home = opt.gpu;
  if ((int)opt.moe_dies.size() > kMoeMaxDies) throw std::runtime_error("at most 4 dies for experts");
  dies.resize(opt.moe_dies.size());
  for (size_t i = 0; i < dies.size(); i++) {  // per-die buffers first, then measure what is left
    Die& d = dies[i];
    d.dev = opt.moe_dies[i];
    CHECK(hipSetDevice(d.dev));
    CHECK(hipMalloc(&d.scratch, opt.canon ? fp4_moe_canon_rows_scratch_bytes(E, F, kMoeMaxK, kChunk)
                                          : fp4_moe_scratch_bytes(E, F, kMoeMaxK, kChunk)));
    const size_t y_rows = (size_t)kChunk * (opt.canon ? kMoeMaxK : 1);  // canon: one row per expert
    if (d.dev != home) {
      CHECK(hipStreamCreateWithFlags(&d.st, hipStreamNonBlocking));
      CHECK(hipEventCreateWithFlags(&d.done, hipEventDisableTiming));
      CHECK(hipMalloc(&d.d_in, moe_block_bytes(kChunk)));
      CHECK(hipMalloc(&d.d_y, y_rows * E * 4));
      CHECK(hipHostMalloc(&d.h_y, y_rows * E * 4, 0));
    }
    CHECK(hipSetDevice(home));
    d.y_home = dalloc<float>((size_t)kChunk * E);
  }
  CHECK(hipStreamCreateWithFlags(&s_local, hipStreamNonBlocking));
  CHECK(hipEventCreateWithFlags(&ev_dispatched, hipEventDisableTiming));
  CHECK(hipEventCreateWithFlags(&ev_local, hipEventDisableTiming));
  std::vector<int> cap;
  const int spare = opt.load_test >= 0 ? kLoadTestSlots : 0;
  for (const Die& d : dies) cap.push_back(std::max(0, pool_slots_that_fit(d.dev, E, F, (size_t)(opt.reserve_gb * 1e9)) - spare));
  std::vector<bool> loaded(hp.n_layer);
  for (int L = 0; L < hp.n_layer; L++) loaded[L] = m.layers[L].loaded;
  placement = place_experts(read_placement(opt.placement), hp.n_layer, X, cap, loaded);
  auto t0 = Clock::now();
  std::vector<std::thread> th;
  std::vector<std::string> errors(dies.size());
  for (size_t i = 0; i < dies.size(); i++)
    th.emplace_back([&, i] {
      try {
        Die& d = dies[i];
        d.pool = pool_create(d.dev, placement.used[i] + spare, E, F);
        CHECK(hipSetDevice(d.dev));
        hipStream_t ls;
        CHECK(hipStreamCreate(&ls));
        for (int L = 0; L < hp.n_layer; L++)
          for (int e = 0; e < X; e++) {
            if (placement.at(L, e) != (int)i) continue;
            const Layer& l = m.layers[L];
            auto ex = [&](const ggml_tensor* t) { return (const char*)t->data + (size_t)e * t->nb[2]; };
            pool_load(d.pool, placement.slot[(size_t)L * X + e], ex(l.gate_exps), ex(l.up_exps), ex(l.down_exps), ls);
          }
        CHECK(hipStreamSynchronize(ls));
        CHECK(hipStreamDestroy(ls));
      } catch (const std::exception& ex) {
        errors[i] = ex.what();
      }
    });
  for (auto& t : th) t.join();
  for (auto& e : errors) if (!e.empty()) throw std::runtime_error("expert upload: " + e);
  CHECK(hipSetDevice(home));
  std::vector<int> table((size_t)hp.n_layer * X);
  size_t n_gpu = 0;
  for (size_t k = 0; k < table.size(); k++) {
    table[k] = placement.die[k] < 0 ? -1 : (placement.die[k] << 16 | placement.slot[k]);
    n_gpu += placement.die[k] >= 0;
  }
  d_table = dalloc<int>(table.size());
  CHECK(hipMemcpy(d_table, table.data(), table.size() * 4, hipMemcpyHostToDevice));
  fprintf(stderr, "# runtime: %zu experts (%.1f%% of the loaded layers' pairs) on GPUs", n_gpu,
          100.0 * n_gpu / (std::count(loaded.begin(), loaded.end(), true) * X));
  for (size_t i = 0; i < dies.size(); i++) fprintf(stderr, "%s%d: %d", i ? ", " : " (", dies[i].dev, placement.used[i]);
  fprintf(stderr, "), loaded in %.1f s\n", ms_since(t0) / 1000);
  if (opt.load_test >= 0) {
    std::vector<ExpertPool*> pools;
    for (Die& d : dies) pools.push_back(&d.pool);
    loader.reset(new ExpertLoader(m, pools, !opt.load_pinned));
    if (!opt.load_pinned) loader->wait_direct();  // measure the direct path only
    if (opt.load_test > 0) loader->start_test(kLoadTestSlots, opt.load_test);
    fprintf(stderr, "# load test: %d spare slots per die, %.0f loads/s (%s)\n", kLoadTestSlots, opt.load_test,
            loader->direct_ready() ? "direct" : "pinned");
    CHECK(hipSetDevice(home));
  }
}

Runtime::~Runtime() {
  (void)hipStreamSynchronize(s);
  if (loader) {
    fprintf(stderr, "# load test: %ld loads\n", loader->loads());
    loader.reset();
  }
  dspark.reset();
  cpu_moe.reset();
  for (Die& d : dies) {
    (void)hipSetDevice(d.dev);
    if (d.st) (void)hipStreamSynchronize(d.st);
    pool_free(d.pool);
    (void)hipSetDevice(d.dev);
    (void)hipFree(d.scratch);
    if (d.st) {
      (void)hipFree(d.d_in);
      (void)hipFree(d.d_y);
      (void)hipHostFree(d.h_y);
      (void)hipEventDestroy(d.done);
      (void)hipStreamDestroy(d.st);
    }
    (void)hipSetDevice(opt.gpu);
    (void)hipFree(d.y_home);
  }
  (void)hipSetDevice(opt.gpu);
  if (d_table) (void)hipFree(d_table);
  if (s_local) {
    (void)hipStreamSynchronize(s_local);
    (void)hipStreamDestroy(s_local);
    (void)hipEventDestroy(ev_dispatched);
    (void)hipEventDestroy(ev_local);
  }
  (void)hipEventDestroy(ev_staged);
  if (ring_backup) (void)hipFree(ring_backup);
  for (auto& [L, p] : cmp_rows) (void)hipFree(p);
  for (auto& [L, p] : cmp_backup) (void)hipFree(p);
  if (route_log) fclose(route_log);
  if (d_route) (void)hipFree(d_route);
  for (void* p : owned) (void)hipFree(p);
  for (const LayerState& st : S)
    for (float* p : {st.win, st.cmp_kv, st.cmp_score, st.comp, st.idx_k}) if (p) (void)hipFree(p);
  for (void* p : {(void*)h, (void*)h2, (void*)pre, (void*)a_pre, (void*)a_post, (void*)a_comb, (void*)f_pre, (void*)f_post,
                  (void*)f_comb, (void*)mixes, (void*)x, (void*)t1, (void*)qr, (void*)q, (void*)kvv, (void*)o, (void*)oa,
                  (void*)y, (void*)ysh, (void*)yr, (void*)canon_rows, (void*)canon_where, (void*)logits_d, (void*)partial, (void*)A.q, (void*)A.s, (void*)B.q, (void*)B.s, (void*)cmp_kvn,
                  (void*)cmp_sc, (void*)lat, (void*)ik, (void*)iq, (void*)iw, (void*)iscore, (void*)gate, (void*)up,
                  (void*)act, (void*)router_logits, (void*)d_moe.lists, (void*)rw, (void*)rids, (void*)att_scratch, (void*)erows, (void*)ekv, (void*)iota,
                  fp8_buf, (void*)freqs_buf})
    (void)hipFree(p);
  for (auto& [L, p] : topk_buf) (void)hipFree(p);
  for (void* p : {(void*)h_moe.lists, (void*)h_yr, (void*)h_erows}) (void)hipHostFree(p);
  for (hipEvent_t e : ev_pool) (void)hipEventDestroy(e);
  (void)hipStreamDestroy(s);
}

// ------------------------------------------------------------------ pieces

// act_quant an activation into a code set (once per distinct input)
void Runtime::quant(const float* xin, int n, Codes c, int M) { fp8_quant_act(xin, c.q, c.s, M, n, s); }

// the exact fp8 GEMV on a quantized input (rows_per_group: block-diagonal, attn_output_a)
void Runtime::gemv8(const Fp8W& w, Codes c, float* yout, int rows_per_group, int M) {
  fp8_gemv(w.q, w.s, w.N, w.K, rows_per_group ? rows_per_group : w.N, Fp8Act{c.q, c.s}, M, yout, s);
}

void Runtime::mark(const char* group) {
  if (!profile) return;
  if (marks.size() == ev_pool.size()) {
    hipEvent_t e;
    CHECK(hipEventCreate(&e));
    ev_pool.push_back(e);
  }
  hipEvent_t e = ev_pool[marks.size()];
  CHECK(hipEventRecord(e, s));
  marks.emplace_back(group, e);
}

void Runtime::tap(Taps* taps, const char* name, const float* d, std::vector<int64_t> ne) {
  if (!taps) return;
  Tap t;
  t.ne = ne;
  int64_t n = 1;
  for (auto v : ne) n *= v;
  t.f.resize(n);
  CHECK(hipMemcpyAsync(t.f.data(), d, n * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipStreamSynchronize(s));
  (*taps)[name] = std::move(t);
}

// The engram rows depend only on the tokens, so every engram layer's rows are gathered at the
// start of the step (host: hash, read the mmap'd mxfp8 rows, convert) and go to the GPU in one
// copy; the layers then only compute. h_erows is rewritten next step, after end_step's sync.
void Runtime::fetch_engram_rows(const std::vector<int32_t>& chunk) {
  const int n_cols = hp.engram_cols(), D = hp.engram_dim, n_slots = (int)hp.engram_layers.size();
  const size_t n = (size_t)n_cols * D;  // one token's rows of one slot
  const ggml_tensor* tab = m.engram_table;  // host mmap, mxfp8
  const auto to_float = ggml_get_type_traits(tab->type)->to_float;
  std::vector<int32_t> hs = tokens_;  // the history up to and including each chunk token
  for (size_t t = 0; t < chunk.size(); t++) {
    hs.push_back(chunk[t]);
    for (int e = 0; e < n_slots; e++) {
      const std::vector<int32_t> rows = engram_rows(hp, hs, e, pos + (int)t);
      for (int c = 0; c < n_cols; c++)
        to_float((const char*)tab->data + (size_t)rows[c] * tab->nb[1], h_erows + ((size_t)e * kChunk + t) * n + (size_t)c * D, D);
    }
  }
  CHECK(hipMemcpyAsync(erows, h_erows, (size_t)n_slots * kChunk * n * 4, hipMemcpyHostToDevice, s));
}

void Runtime::engram(int L, int M) {
  const int n = hp.engram_cols() * hp.engram_dim, E = hp.n_embd, HC = hp.hc_mult;
  quant(erows + (size_t)m.layers[L].kind.engram * kChunk * n, n, B, M);
  gemv8(W[L].engram_kv, B, ekv, 0, M);
  for (int t = 0; t < M; t++)
    k_engram(h + (size_t)t * E * HC, ekv + (size_t)t * E * (HC + 1), W[L].engram_gate, HC, E, hp.rms_eps, s);
}

// Attention for the chunk's M tokens at positions pos ..: the projections batched (one weight
// read), then per token in position order the window K/V, the compressor, the indexer and the
// attention itself (each token sees exactly the state it would alone), then the output
// projections batched.
void Runtime::attention(int L, const float* xin, float* out, int M) {
  const LayerW& w = W[L];
  const int HD = hp.head_dim, NH = hp.n_head, E = hp.n_embd;
  const float eps = hp.rms_eps;
  const int p0 = pos;

  mark("attn.q");
  quant(xin, E, A, M);  // x feeds q_a and kv
  gemv8(w.q_a, A, t1, 0, M);
  k_rms_norm(t1, w.q_a_norm, qr, hp.q_lora, M, eps, s);
  quant(qr, hp.q_lora, B, M);  // qr feeds q_b and the indexer's q_b
  gemv8(w.q_b, B, q, 0, M);
  mark("attn.kv");
  gemv8(w.kv, A, kvv, 0, M);
  for (int t = 0; t < M; t++) {
    pos = p0 + t;
    attn_core(L, t, xin + (size_t)t * E, q + (size_t)t * NH * HD, kvv + (size_t)t * HD,
              Codes{B.q + (size_t)t * hp.q_lora, B.s + (size_t)t * hp.q_lora / 32}, o + (size_t)t * NH * HD);
  }
  pos = p0;
  mark("attn.out");
  quant(o, NH * HD, B, M);
  gemv8(w.out_a, B, oa, hp.o_lora, M);  // block-diagonal over the 8 groups
  quant(oa, hp.o_groups * hp.o_lora, B, M);
  gemv8(w.out_b, B, out, 0, M);
}

// one token (chunk index t, at position pos): rope of q, its window K/V into the ring, the
// compressor and indexer, the attention, and the inverse rope of its output
void Runtime::attn_core(int L, int t, const float* xin, float* q_t, const float* kvv_t, Codes qr_t, float* o_t) {
  const LayerW& w = W[L];
  const LayerKind& k = m.layers[L].kind;
  const int HD = hp.head_dim, NH = hp.n_head, RD = hp.rope_dim, WIN = hp.window, E = hp.n_embd;
  const float eps = hp.rms_eps;
  std::map<int, const int*>& tcur = topk_cur[t];
  k_rope(q_t, NH, HD, RD, w.freqs, pos, false, s);
  float* ring = S[L].win;
  k_kv_finish(kvv_t, w.kv_norm, HD, RD, w.freqs, pos, eps, ring + (size_t)(pos % WIN) * HD, s);

  const int* topk = nullptr;
  int K = 0;
  const float* comp = nullptr;
  if (k.ratio) {
    const int r = k.ratio;
    mark("attn.compressor");
    if (k.kv_source) {  // Compressor.forward at decode: one token into the group; pool when complete
      k_gemv_f32(w.cmp_kv, HD, E, xin, cmp_kvn, partial, s);
      const float* latent = nullptr;
      if (r > 1) {
        k_gemv_f32(w.cmp_gate, HD, E, xin, cmp_sc, partial, s);
        float* kv_state = S[L].cmp_kv;
        float* sc_state = S[L].cmp_score;
        CHECK(hipMemcpyAsync(kv_state + (size_t)(pos % r) * HD, cmp_kvn, HD * 4, hipMemcpyDeviceToDevice, s));
        CHECK(hipMemcpyAsync(sc_state + (size_t)(pos % r) * HD, cmp_sc, HD * 4, hipMemcpyDeviceToDevice, s));
        if (verifying) {  // this chunk position's row, for commit
          float* row = cmp_rows.at(L) + (size_t)(pos - verify_p0) * 2 * HD;
          CHECK(hipMemcpyAsync(row, cmp_kvn, HD * 4, hipMemcpyDeviceToDevice, s));
          CHECK(hipMemcpyAsync(row + HD, cmp_sc, HD * 4, hipMemcpyDeviceToDevice, s));
        }
        if ((pos + 1) % r == 0) {
          k_compress_pool(kv_state, sc_state, r, HD, lat, s);
          latent = lat;
        }
      } else {
        CHECK(hipMemcpyAsync(lat, cmp_kvn, HD * 4, hipMemcpyDeviceToDevice, s));
        latent = lat;
      }
      if (latent) {
        const int gi = pos / r, lpos = gi * r;  // a latent stands for its group's first token
        k_rms_norm(lat, w.cmp_norm, lat, HD, 1, eps, s);
        // index keys from the RoPE-free latent first
        k_gemv_f32(w.idx_k, hp.idx_dim, HD, lat, ik, partial, s);
        k_rms_norm(ik, w.idx_k_norm, ik, hp.idx_dim, 1, eps, s);
        k_rope(ik, 1, hp.idx_dim, RD, w.freqs, lpos, false, s);
        k_fq4(ik, hp.idx_dim, false, s);
        float* idx_cache = S[L].idx_k;
        CHECK(hipMemcpyAsync(idx_cache + (size_t)gi * hp.idx_dim, ik, hp.idx_dim * 4, hipMemcpyDeviceToDevice, s));
        k_rope(lat, 1, HD, RD, w.freqs, lpos, false, s);
        k_fq4(lat, HD, true, s);
        float* comp_cache = S[L].comp;
        CHECK(hipMemcpyAsync(comp_cache + (size_t)gi * HD, lat, HD * 4, hipMemcpyDeviceToDevice, s));
      }
    }
    const int n_comp = (pos + 1) / r;
    K = std::min(hp.idx_topk, n_comp);
    comp = S[k.kv_from].comp;
    mark("attn.indexer");
    if (k.index_source && K > 0) {  // Indexer.forward, one query
      gemv8(w.idx_q_b, qr_t, iq);
      k_rope(iq, hp.idx_heads, hp.idx_dim, RD, w.freqs, pos, false, s);
      k_fq4(iq, hp.idx_heads * hp.idx_dim, false, s);
      k_gemv_bf16(w.idx_proj, hp.idx_heads, E, xin, iw, partial, s);
      const float* keys = S[k.kv_from].idx_k;
      k_index_scores(iq, hp.idx_heads, hp.idx_dim, keys, n_comp, iw, 1.0f / sqrtf((float)hp.idx_dim) / sqrtf((float)hp.idx_heads),
                     iscore, s);
      if (K == n_comp) {
        tcur[L] = iota;  // every compressed position is picked
      } else {
        int* buf = topk_buf.at(L) + (size_t)t * hp.idx_topk;
        k_topk(iscore, n_comp, K, buf, s);
        tcur[L] = buf;
      }
    }
    if (K > 0) topk = tcur.at(k.idx_from);
  }
  mark("attn.core");
  k_attn_decode(q_t, NH, HD, ring, WIN, pos, comp, topk, K, w.sinks, 1.0f / sqrtf((float)HD), o_t, att_scratch, s);
  k_rope(o_t, NH, HD, RD, w.freqs, pos, true, s);  // take the query's rotation back out
}

void Runtime::moe(int L, float* out, int M, Taps* taps) {
  const LayerW& w = W[L];
  const int E = hp.n_embd, F = hp.n_ff_exp, KE = hp.n_expert_used;
  const float* xin = d_moe.x;  // [M][E]
  mark("ffn.router");
  k_gemv_f32_m(w.gate_inp, hp.n_expert, E, xin, M, router_logits, partial, s);
  for (int t = 0; t < M; t++)
    k_router(router_logits + (size_t)t * hp.n_expert, w.gate_bias, hp.n_expert, KE, hp.expert_weights_norm,
             hp.expert_weights_scale, rids + t * kMoeIdStride, rw + t * kMoeIdStride, s);
  if (route_log)  // ids and weights next to each other per token and layer: [ids 8 | weights 8]
    for (int t = 0; t < M; t++) {
      int* dst = d_route + ((size_t)t * hp.n_layer + L) * 16;
      CHECK(hipMemcpyAsync(dst, rids + t * kMoeIdStride, KE * 4, hipMemcpyDeviceToDevice, s));
      CHECK(hipMemcpyAsync(dst + 8, rw + t * kMoeIdStride, KE * 4, hipMemcpyDeviceToDevice, s));
    }
  // split every token's experts by where they live, and stage the lists + x to the host in one
  // copy (the CPU and the other dies read it there)
  const int n_dies = (int)dies.size();
  mark("ffn.dispatch");
  k_moe_dispatch(rids, rw, KE, d_table ? d_table + (size_t)L * hp.n_expert : nullptr, n_dies, d_moe.lists, M, s, canon_where);
  if (n_dies) CHECK(hipEventRecord(ev_dispatched, s));
  const size_t block = moe_block_bytes(M);
  CHECK(hipMemcpyAsync(h_moe.lists, d_moe.lists, moe_lists_bytes, hipMemcpyDeviceToHost, s));
  CHECK(hipMemcpyAsync(h_moe.x, d_moe.x, (size_t)M * E * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipEventRecord(ev_staged, s));
  constexpr int LS = kMoeListStride * (int)(sizeof(MoeList) / 4);  // a token's lists, in ints
  auto sel = [&](const MoeList* lists, int die) {  // die `die`'s list of every token
    return ExpertSel{lists[die].ids, lists[die].w, &lists[die].count, LS};
  };
  auto place_rows = [&](int p) { return canon_rows + (size_t)p * kChunk * KE * E; };  // canon
  bool local = false;
  for (int i = 0; i < n_dies; i++) {
    Die& d = dies[i];
    if (!d.st) {  // the home die: its lists on the second stream, next to the shared expert
      local = true;
      CHECK(hipStreamWaitEvent(s_local, ev_dispatched, 0));
      if (opt.canon)
        fp4_moe_canon_rows(d.pool.view, xin, sel(d_moe.lists, i), KE, M, hp.swiglu_clamp[L], place_rows(i), d.scratch, s_local);
      else
        fp4_moe_chunk(d.pool.view, xin, sel(d_moe.lists, i), KE, M, hp.swiglu_clamp[L], d.y_home, d.scratch, s_local);
      CHECK(hipEventRecord(ev_local, s_local));
      continue;
    }
    // another die: starts when the block is in host memory, no host involvement
    CHECK(hipSetDevice(d.dev));
    CHECK(hipStreamWaitEvent(d.st, ev_staged, 0));
    CHECK(hipMemcpyAsync(d.d_in, h_moe.lists, block, hipMemcpyHostToDevice, d.st));
    const float* dx = (const float*)(d.d_in + moe_lists_bytes);
    if (opt.canon) {  // the copy back waits for the counts (below)
      fp4_moe_canon_rows(d.pool.view, dx, sel((const MoeList*)d.d_in, i), KE, M, hp.swiglu_clamp[L], d.d_y, d.scratch, d.st);
    } else {
      fp4_moe_chunk(d.pool.view, dx, sel((const MoeList*)d.d_in, i), KE, M, hp.swiglu_clamp[L], d.d_y, d.scratch, d.st);
      CHECK(hipMemcpyAsync(d.h_y, d.d_y, (size_t)M * E * 4, hipMemcpyDeviceToHost, d.st));
      CHECK(hipEventRecord(d.done, d.st));
    }
    CHECK(hipSetDevice(opt.gpu));
  }
  mark("ffn.shared");
  quant(xin, E, A, M);
  gemv8(w.gate_sh, A, gate, 0, M);
  gemv8(w.up_sh, A, up, 0, M);
  k_swiglu(gate, up, hp.swiglu_clamp_sh[L], M * F, act, s);
  quant(act, F, B, M);
  gemv8(w.down_sh, B, ysh, 0, M);
  // the host: the lists (counts for the statistics) and the CPU's misses, token by token
  auto t0 = Clock::now();
  CHECK(hipEventSynchronize(ev_staged));
  last.wait_ms += ms_since(t0);
  // the place's rows of this chunk (canon): the counts are known now
  std::vector<int> n_rows(n_dies + 1, 0);
  for (int t = 0; t < M; t++) {
    const MoeList* lt = h_moe.lists + (size_t)t * kMoeListStride;
    for (int i = 0; i <= n_dies; i++) n_rows[i] += lt[i].count;
    for (int i = 0; i < n_dies; i++) {
      last.gpu_experts += lt[i].count;
      total_gpu_experts += lt[i].count;
    }
    last.cpu_experts += lt[n_dies].count;
    total_cpu_experts += lt[n_dies].count;
  }
  if (opt.canon)
    for (int i = 0; i < n_dies; i++) {
      Die& d = dies[i];
      if (!d.st) continue;
      CHECK(hipSetDevice(d.dev));
      if (n_rows[i]) CHECK(hipMemcpyAsync(d.h_y, d.d_y, (size_t)n_rows[i] * E * 4, hipMemcpyDeviceToHost, d.st));
      CHECK(hipEventRecord(d.done, d.st));
      CHECK(hipSetDevice(opt.gpu));
    }
  // the host: the CPU's misses, token by token
  std::vector<char> has_cpu(M, 0);
  bool any_cpu = false;
  int cpu_row = 0;
  for (int t = 0; t < M; t++) {
    const MoeList& lc = h_moe.lists[(size_t)t * kMoeListStride + n_dies];
    if (lc.count == 0) continue;
    if (!any_cpu) mark("ffn.wait_for_cpu_experts");
    any_cpu = has_cpu[t] = true;
    auto t1c = Clock::now();
    const Layer& hl = m.layers[L];  // host tensors: the experts live in the mmap
    const CpuExperts ce{hl.gate_exps, hl.up_exps, hl.down_exps, hp.swiglu_clamp[L]};
    if (opt.canon) {
      cpu_moe->forward_canon_rows(ce, h_moe.x + (size_t)t * E, lc.ids, lc.w, lc.count, h_yr + (size_t)cpu_row * E);
      cpu_row += lc.count;
    } else {
      cpu_moe->forward(ce, h_moe.x + (size_t)t * E, lc.ids, lc.w, lc.count, h_yr + (size_t)t * E);
    }
    last.cpu_moe_ms += ms_since(t1c);
  }
  if (any_cpu) {
    if (opt.canon) CHECK(hipMemcpyAsync(place_rows(n_dies), h_yr, (size_t)cpu_row * E * 4, hipMemcpyHostToDevice, s));
    else CHECK(hipMemcpyAsync(yr, h_yr, (size_t)M * E * 4, hipMemcpyHostToDevice, s));
  }
  // the dies' results, waited for in the stream
  if (n_dies) mark("ffn.wait_for_gpu_experts");
  if (local) CHECK(hipStreamWaitEvent(s, ev_local, 0));
  for (int i = 0; i < n_dies; i++)
    if (dies[i].st) {
      CHECK(hipStreamWaitEvent(s, dies[i].done, 0));
      if (opt.canon) {
        if (n_rows[i]) CHECK(hipMemcpyAsync(place_rows(i), dies[i].h_y, (size_t)n_rows[i] * E * 4, hipMemcpyHostToDevice, s));
      } else {
        CHECK(hipMemcpyAsync(dies[i].y_home, dies[i].h_y, (size_t)M * E * 4, hipMemcpyHostToDevice, s));
      }
    }
  mark("ffn.add");
  if (opt.canon) {
    k_moe_combine(ysh, canon_rows, kChunk * KE, d_moe.lists, canon_where, KE, n_dies, M, out, E, s);
  } else {
    for (int t = 0; t < M; t++) {  // fixed order: dies, then the CPU (if this token had misses)
      const float* parts[kMoeMaxDies + 1] = {};
      for (int i = 0; i < n_dies; i++) parts[i] = dies[i].y_home + (size_t)t * E;
      if (has_cpu[t]) parts[n_dies] = yr + (size_t)t * E;
      k_add_parts(ysh + (size_t)t * E, parts, n_dies + 1, out + (size_t)t * E, E, s);
    }
  }
  if (taps) {
    tap(taps, "router_w", rw, {KE, 1});  // (synchronizes the stream)
    Tap ti;
    ti.ne = {KE, 1};
    ti.i.resize(KE);
    CHECK(hipMemcpy(ti.i.data(), rids, KE * 4, hipMemcpyDeviceToHost));
    (*taps)["router_i"] = ti;
  }
}

void Runtime::block(int L, Taps* taps, int M) {
  const LayerW& w = W[L];
  const int E = hp.n_embd, HC = hp.hc_mult, MIX = (2 + HC) * HC;
  const float eps = hp.rms_eps;
  auto ht = [&](int t) { return h + (size_t)t * E * HC; };
  mark("engram");
  if (m.layers[L].kind.engram >= 0) engram(L, M);
  tap(taps, "h_in", h, {E, HC, 1});
  if (dspark) {  // the draft reads this layer's input stream, averaged over the hc copies
    const int d = dspark->target_index(L);
    const size_t nh = dspark->target_layers.size() * E;
    if (d >= 0)
      for (int t = 0; t < M; t++)  // while verifying, per chunk position (only kept ones seed the draft)
        k_hc_pre(ht(t), dspark->quarter, E, (verifying ? dspark->hid_chunk + t * nh : dspark->main_hidden) + (size_t)d * E, s);
  }
  mark("hc");
  // each sublayer's mixes give the next sublayer's input mix (see Engine::block)
  k_gemv_f32_m(w.hc_attn_fn, MIX, E * HC, h, M, mixes, partial, s);
  k_hc_split(h, E * HC, mixes, w.hc_attn_scale, w.hc_attn_base, eps, hp.hc_eps, hp.hc_iters, a_pre, a_post, a_comb, s, M);
  k_hc_pre_norm(h, pre, w.attn_norm, E, eps, x, s, M);
  tap(taps, "attn_in", x, {E, 1});
  attention(L, x, y, M);
  tap(taps, "attn_out", y, {E, 1});
  mark("hc");
  k_hc_post(y, h, a_post, a_comb, E, h2, s, M);
  std::swap(h, h2);
  k_gemv_f32_m(w.hc_ffn_fn, MIX, E * HC, h, M, mixes, partial, s);
  k_hc_split(h, E * HC, mixes, w.hc_ffn_scale, w.hc_ffn_base, eps, hp.hc_eps, hp.hc_iters, f_pre, f_post, f_comb, s, M);
  k_hc_pre_norm(h, a_pre, w.ffn_norm, E, eps, d_moe.x, s, M);
  tap(taps, "ffn_in", d_moe.x, {E, 1});
  moe(L, y, M, taps);
  tap(taps, "ffn_out", y, {E, 1});
  mark("hc");
  k_hc_post(y, h, f_post, f_comb, E, h2, s, M);
  std::swap(h, h2);
  CHECK(hipMemcpyAsync(pre, f_pre, (size_t)M * HC * 4, hipMemcpyDeviceToDevice, s));
  tap(taps, "h_out", h, {E, HC, 1});
  tap(taps, "pre_out", pre, {HC, 1});
  if (taps && m.layers[L].kind.index_source && topk_cur[0].count(L)) {
    const int n_comp = (pos + 1) / m.layers[L].kind.ratio, K = std::min(hp.idx_topk, n_comp);
    Tap tp;
    tp.ne = {K, 1};
    tp.i.resize(K);
    CHECK(hipMemcpyAsync(tp.i.data(), topk_cur[0][L], K * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    (*taps)["topk"] = tp;
  }
}

// ------------------------------------------------------------------ entry points

void Runtime::begin_step(int32_t tok) {
  // the compressed caches and the top-k hold max_ctx positions
  if (n_past_ >= opt.max_ctx) throw std::runtime_error("runtime: position " + std::to_string(n_past_) + " is past --max-ctx " + std::to_string(opt.max_ctx));
  pos = n_past_;
  token = tok;
  hist = tokens_;
  hist.push_back(tok);
  for (auto& tc : topk_cur) tc.clear();
  last = Times{};
  marks.clear();
  mark("engram.fetch");
  fetch_engram_rows({tok});
}

// Routing log format (little endian):
//   header: "DSRT" magic, int32 version = 2, int32 n_layer, int32 k
//   per token forwarded: int32 pos, int32 token, int32 is_prompt, int32 step (the forward: a
//   verify chunk logs its tokens with one step, rejected drafts included; a position logged again
//   later was rolled back, the last record of a position is the committed one), int16
//   ids[n_layer][k], float w[n_layer][k]
// Layers that are not loaded log ids -1. Version 1 had no step field.
void Runtime::log_routing(const std::string& path) {
  route_log = fopen(path.c_str(), "wb");
  if (!route_log) throw std::runtime_error("cannot write " + path);
  const int32_t hdr[3] = {2, hp.n_layer, hp.n_expert_used};
  fwrite("DSRT", 1, 4, route_log);
  fwrite(hdr, 4, 3, route_log);
  d_route = dalloc<int>((size_t)kChunk * hp.n_layer * 16);
}

// the M tokens of the forward that started at position `pos`
void Runtime::write_route_records(const int32_t* toks, int M) {
  const int nl = hp.n_layer, k = hp.n_expert_used;
  std::vector<int32_t> buf((size_t)M * nl * 16);
  CHECK(hipMemcpy(buf.data(), d_route, buf.size() * 4, hipMemcpyDeviceToHost));
  for (int t = 0; t < M; t++) {
    const int32_t head[4] = {pos + t, toks[t], in_prefill ? 1 : 0, route_step};
    std::vector<int16_t> ids((size_t)nl * k, -1);
    std::vector<float> w((size_t)nl * k, 0.0f);
    for (int L = 0; L < nl; L++) {
      if (!m.layers[L].loaded) continue;
      const int32_t* r = buf.data() + ((size_t)t * nl + L) * 16;
      for (int j = 0; j < k; j++) {
        ids[L * k + j] = (int16_t)r[j];
        memcpy(&w[L * k + j], &r[8 + j], 4);
      }
    }
    fwrite(head, 4, 4, route_log);
    fwrite(ids.data(), 2, ids.size(), route_log);
    fwrite(w.data(), 4, w.size(), route_log);
  }
  route_step++;
}

void Runtime::end_step() {
  CHECK(hipStreamSynchronize(s));
  if (route_log) write_route_records(&token, 1);
  n_past_++;
  tokens_.push_back(token);
}

Taps Runtime::run_layer(int L, const std::vector<float>& h_in, const std::vector<float>& pre_in) {
  Taps taps;
  CHECK(hipMemcpyAsync(h, h_in.data(), h_in.size() * 4, hipMemcpyHostToDevice, s));
  CHECK(hipMemcpyAsync(pre, pre_in.data(), pre_in.size() * 4, hipMemcpyHostToDevice, s));
  block(L, &taps);
  CHECK(hipStreamSynchronize(s));
  return taps;
}

std::vector<float> Runtime::prefill(const std::vector<int32_t>& tokens) {
  std::vector<float> logits;
  in_prefill = true;
  for (size_t i = 0; i < tokens.size(); i++) logits = decode(tokens[i], i + 1 == tokens.size());
  in_prefill = false;
  return logits;
}

std::vector<float> Runtime::decode(int32_t tok, bool want_logits) {
  auto t0 = Clock::now();
  begin_step(tok);
  const int E = hp.n_embd, HC = hp.hc_mult;
  // embedding (bf16 row of the host table) into hc copies; the first block's pre-mix is copy 0
  std::vector<float> emb(E), one_hot(HC, 0.0f);
  const ggml_bf16_t* row = (const ggml_bf16_t*)((const char*)m.tok_embd->data + (size_t)tok * m.tok_embd->nb[1]);
  for (int i = 0; i < E; i++) emb[i] = ggml_bf16_to_fp32(row[i]);
  one_hot[0] = 1.0f;
  CHECK(hipMemcpyAsync(x, emb.data(), E * 4, hipMemcpyHostToDevice, s));
  CHECK(hipMemcpyAsync(pre, one_hot.data(), HC * 4, hipMemcpyHostToDevice, s));
  k_repeat(x, E, HC, h, s);
  for (int L = 0; L < hp.n_layer; L++) block(L, nullptr);
  if (dspark) dspark->seed(pos);  // this position's K/V into the draft's window rings
  if (!want_logits) {
    end_step();
    last.total_ms = ms_since(t0);
    return {};
  }
  // head: collapse with the last pre-mix, norm, logits
  mark("head");
  k_hc_pre(h, pre, E, x, s);
  k_rms_norm(x, out_norm, x, E, 1, hp.rms_eps, s);
  k_gemv_bf16(out_w, hp.n_vocab, E, x, logits_d, partial, s);
  std::vector<float> logits(hp.n_vocab);
  mark("end");
  CHECK(hipMemcpyAsync(logits.data(), logits_d, logits.size() * 4, hipMemcpyDeviceToHost, s));
  end_step();
  if (profile) {
    last_groups.clear();
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      float ms;
      CHECK(hipEventElapsedTime(&ms, marks[i].second, marks[i + 1].second));
      last_groups[marks[i].first] += ms;
    }
  }
  last.total_ms = ms_since(t0);
  return logits;
}
