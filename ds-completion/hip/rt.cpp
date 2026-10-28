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
  h = dalloc<float>(E * HC);
  h2 = dalloc<float>(E * HC);
  pre = dalloc<float>(HC);
  a_pre = dalloc<float>(HC);
  a_post = dalloc<float>(HC);
  a_comb = dalloc<float>(HC * HC);
  f_pre = dalloc<float>(HC);
  f_post = dalloc<float>(HC);
  f_comb = dalloc<float>(HC * HC);
  mixes = dalloc<float>((2 + HC) * HC);
  x = dalloc<float>(E);
  t1 = dalloc<float>(hp.q_lora);
  qr = dalloc<float>(hp.q_lora);
  q = dalloc<float>(NH * HD);
  kvv = dalloc<float>(HD);
  o = dalloc<float>(NH * HD);
  oa = dalloc<float>(hp.o_groups * hp.o_lora);
  y = dalloc<float>(E);
  ysh = dalloc<float>(E);
  yr = dalloc<float>(E);
  logits_d = dalloc<float>(hp.n_vocab);
  partial = dalloc<float>((size_t)kSplitMax * std::max(hp.n_vocab, 32768));
  A = {dalloc<uint8_t>(NH * HD), dalloc<float>(NH * HD / 32)};
  B = {dalloc<uint8_t>(NH * HD), dalloc<float>(NH * HD / 32)};
  cmp_kvn = dalloc<float>(HD);
  cmp_sc = dalloc<float>(HD);
  lat = dalloc<float>(HD);
  ik = dalloc<float>(hp.idx_dim);
  iq = dalloc<float>(hp.idx_heads * hp.idx_dim);
  iw = dalloc<float>(hp.idx_heads);
  iscore = dalloc<float>(kTopkMaxN);
  gate = dalloc<float>(hp.n_ff_exp);
  up = dalloc<float>(hp.n_ff_exp);
  act = dalloc<float>(hp.n_ff_exp);
  router_logits = dalloc<float>(hp.n_expert);
  {  // device + pinned host staging of the routed experts' inputs, one contiguous block each
    moe_block_bytes = (size_t)E * 4 + 8 * 4 + 8 * 4 + (kMoeMaxDies + 1) * sizeof(MoeList);
    char* d = dalloc<char>(moe_block_bytes);
    char* hm;
    CHECK(hipHostMalloc((void**)&hm, moe_block_bytes, 0));
    d_moe = {(float*)d, (int*)(d + E * 4), (float*)(d + E * 4 + 32), (MoeList*)(d + E * 4 + 64)};
    h_moe = {(float*)hm, (int*)(hm + E * 4), (float*)(hm + E * 4 + 32), (MoeList*)(hm + E * 4 + 64)};
  }
  rw = d_moe.w;
  rids = d_moe.ids;
  att_scratch = dalloc<float>((size_t)((hp.window + hp.idx_topk + 31) / 32) * NH * (HD + 2));
  erows = dalloc<float>(hp.engram_layers.size() * hp.engram_cols() * hp.engram_dim);  // [engram slot][rows]
  ekv = dalloc<float>((size_t)E * (HC + 1));
  {
    std::vector<int> v(opt.max_ctx);
    for (int i = 0; i < (int)v.size(); i++) v[i] = i;
    iota = dalloc<int>(v.size());
    CHECK(hipMemcpy(iota, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  }
  for (int L : hp.index_source_layers)
    if (L < hp.n_layer && m.layers[L].loaded) topk_buf[L] = dalloc<int>(hp.idx_topk);
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
  CHECK(hipHostMalloc(&h_yr, E * 4, 0));
  CHECK(hipHostMalloc(&h_erows, hp.engram_layers.size() * hp.engram_cols() * hp.engram_dim * 4, 0));
  if (opt.max_ctx > kTopkMaxN) throw std::runtime_error("runtime: --max-ctx above the top-k kernel's limit");
  CHECK(hipEventCreateWithFlags(&ev_staged, hipEventDisableTiming));

  if (!opt.dspark.empty()) dspark.reset(new DSpark(*this, opt.dspark));  // before the pools take the rest
  setup_experts();
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
    CHECK(hipMalloc(&d.scratch, fp4_moe_scratch_bytes(E, F, kMoeMaxK)));
    if (d.dev != home) {
      CHECK(hipStreamCreateWithFlags(&d.st, hipStreamNonBlocking));
      CHECK(hipEventCreateWithFlags(&d.done, hipEventDisableTiming));
      CHECK(hipMalloc(&d.d_in, moe_block_bytes));
      CHECK(hipMalloc(&d.d_y, E * 4));
      CHECK(hipHostMalloc(&d.h_y, E * 4, 0));
    }
    CHECK(hipSetDevice(home));
    d.y_home = dalloc<float>(E);
  }
  CHECK(hipStreamCreateWithFlags(&s_local, hipStreamNonBlocking));
  CHECK(hipEventCreateWithFlags(&ev_dispatched, hipEventDisableTiming));
  CHECK(hipEventCreateWithFlags(&ev_local, hipEventDisableTiming));
  std::vector<int> cap;
  for (const Die& d : dies) cap.push_back(pool_slots_that_fit(d.dev, E, F, (size_t)(opt.reserve_gb * 1e9)));
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
        d.pool = pool_create(d.dev, placement.used[i], E, F);
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
}

Runtime::~Runtime() {
  (void)hipStreamSynchronize(s);
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
  if (route_log) fclose(route_log);
  if (d_route) (void)hipFree(d_route);
  for (void* p : owned) (void)hipFree(p);
  for (const LayerState& st : S)
    for (float* p : {st.win, st.cmp_kv, st.cmp_score, st.comp, st.idx_k}) if (p) (void)hipFree(p);
  for (void* p : {(void*)h, (void*)h2, (void*)pre, (void*)a_pre, (void*)a_post, (void*)a_comb, (void*)f_pre, (void*)f_post,
                  (void*)f_comb, (void*)mixes, (void*)x, (void*)t1, (void*)qr, (void*)q, (void*)kvv, (void*)o, (void*)oa,
                  (void*)y, (void*)ysh, (void*)yr, (void*)logits_d, (void*)partial, (void*)A.q, (void*)A.s, (void*)B.q, (void*)B.s, (void*)cmp_kvn,
                  (void*)cmp_sc, (void*)lat, (void*)ik, (void*)iq, (void*)iw, (void*)iscore, (void*)gate, (void*)up,
                  (void*)act, (void*)router_logits, (void*)d_moe.x, (void*)att_scratch, (void*)erows, (void*)ekv, (void*)iota,
                  fp8_buf, (void*)freqs_buf})
    (void)hipFree(p);
  for (auto& [L, p] : topk_buf) (void)hipFree(p);
  for (void* p : {(void*)h_moe.x, (void*)h_yr, (void*)h_erows}) (void)hipHostFree(p);
  for (hipEvent_t e : ev_pool) (void)hipEventDestroy(e);
  (void)hipStreamDestroy(s);
}

// ------------------------------------------------------------------ pieces

// act_quant an activation into a code set (once per distinct input)
void Runtime::quant(const float* xin, int n, Codes c) { fp8_quant_act(xin, c.q, c.s, 1, n, s); }

// the exact fp8 GEMV on a quantized input (rows_per_group: block-diagonal, attn_output_a)
void Runtime::gemv8(const Fp8W& w, Codes c, float* yout, int rows_per_group) {
  fp8_gemv(w.q, w.s, w.N, w.K, rows_per_group ? rows_per_group : w.N, Fp8Act{c.q, c.s}, 1, yout, s);
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
void Runtime::fetch_engram_rows() {
  const int n_cols = hp.engram_cols(), D = hp.engram_dim, n_slots = (int)hp.engram_layers.size();
  const ggml_tensor* tab = m.engram_table;  // host mmap, mxfp8
  const auto to_float = ggml_get_type_traits(tab->type)->to_float;
  for (int e = 0; e < n_slots; e++) {
    const std::vector<int32_t> rows = engram_rows(hp, hist, e, pos);
    for (int c = 0; c < n_cols; c++)
      to_float((const char*)tab->data + (size_t)rows[c] * tab->nb[1], h_erows + ((size_t)e * n_cols + c) * D, D);
  }
  CHECK(hipMemcpyAsync(erows, h_erows, (size_t)n_slots * n_cols * D * 4, hipMemcpyHostToDevice, s));
}

void Runtime::engram(int L) {
  const int n = hp.engram_cols() * hp.engram_dim;
  quant(erows + (size_t)m.layers[L].kind.engram * n, n, B);
  gemv8(W[L].engram_kv, B, ekv);
  k_engram(h, ekv, W[L].engram_gate, hp.hc_mult, hp.n_embd, hp.rms_eps, s);
}

void Runtime::attention(int L, const float* xin, float* out) {
  const LayerW& w = W[L];
  const LayerKind& k = m.layers[L].kind;
  const int HD = hp.head_dim, NH = hp.n_head, RD = hp.rope_dim, WIN = hp.window, E = hp.n_embd;
  const float eps = hp.rms_eps;

  mark("attn.q");
  quant(xin, E, A);  // x feeds q_a and kv
  gemv8(w.q_a, A, t1);
  k_rms_norm(t1, w.q_a_norm, qr, hp.q_lora, 1, eps, s);
  quant(qr, hp.q_lora, B);  // qr feeds q_b and the indexer's q_b
  gemv8(w.q_b, B, q);
  k_rope(q, NH, HD, RD, w.freqs, pos, false, s);
  mark("attn.kv");
  gemv8(w.kv, A, kvv);
  float* ring = S[L].win;
  k_kv_finish(kvv, w.kv_norm, HD, RD, w.freqs, pos, eps, ring + (size_t)(pos % WIN) * HD, s);

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
      gemv8(w.idx_q_b, B, iq);
      k_rope(iq, hp.idx_heads, hp.idx_dim, RD, w.freqs, pos, false, s);
      k_fq4(iq, hp.idx_heads * hp.idx_dim, false, s);
      k_gemv_bf16(w.idx_proj, hp.idx_heads, E, xin, iw, partial, s);
      const float* keys = S[k.kv_from].idx_k;
      k_index_scores(iq, hp.idx_heads, hp.idx_dim, keys, n_comp, iw, 1.0f / sqrtf((float)hp.idx_dim) / sqrtf((float)hp.idx_heads),
                     iscore, s);
      if (K == n_comp) {
        topk_cur[L] = iota;  // every compressed position is picked
      } else {
        k_topk(iscore, n_comp, K, topk_buf.at(L), s);
        topk_cur[L] = topk_buf.at(L);
      }
    }
    if (K > 0) topk = topk_cur.at(k.idx_from);
  }
  mark("attn.core");
  k_attn_decode(q, NH, HD, ring, WIN, pos, comp, topk, K, w.sinks, 1.0f / sqrtf((float)HD), o, att_scratch, s);
  k_rope(o, NH, HD, RD, w.freqs, pos, true, s);  // take the query's rotation back out
  mark("attn.out");
  quant(o, NH * HD, B);
  gemv8(w.out_a, B, oa, hp.o_lora);  // block-diagonal over the 8 groups
  quant(oa, hp.o_groups * hp.o_lora, B);
  gemv8(w.out_b, B, out);
}

void Runtime::moe(int L, const float* xin, float* out, Taps* taps) {
  const LayerW& w = W[L];
  const int E = hp.n_embd, F = hp.n_ff_exp, KE = hp.n_expert_used;
  mark("ffn.router");
  k_gemv_f32(w.gate_inp, hp.n_expert, E, xin, router_logits, partial, s);
  k_router(router_logits, w.gate_bias, hp.n_expert, KE, hp.expert_weights_norm, hp.expert_weights_scale, rids, rw, s);
  if (route_log) {  // ids and weights next to each other per layer: [ids 8 | weights 8]
    CHECK(hipMemcpyAsync(d_route + L * 16, rids, KE * 4, hipMemcpyDeviceToDevice, s));
    CHECK(hipMemcpyAsync(d_route + L * 16 + 8, rw, KE * 4, hipMemcpyDeviceToDevice, s));
  }
  if (xin != d_moe.x) throw std::runtime_error("moe input must be staged in d_moe.x");
  // split the chosen experts by where they live, and stage x + router outputs + lists to the
  // host in one copy (the CPU and the other dies read it there)
  const int n_dies = (int)dies.size();
  mark("ffn.dispatch");
  k_moe_dispatch(rids, rw, KE, d_table ? d_table + (size_t)L * hp.n_expert : nullptr, n_dies, d_moe.lists, s);
  if (n_dies) CHECK(hipEventRecord(ev_dispatched, s));
  CHECK(hipMemcpyAsync(h_moe.x, d_moe.x, moe_block_bytes, hipMemcpyDeviceToHost, s));
  CHECK(hipEventRecord(ev_staged, s));
  const float* parts[kMoeMaxDies + 1] = {};
  bool local = false;
  for (int i = 0; i < n_dies; i++) {
    Die& d = dies[i];
    parts[i] = d.y_home;
    if (!d.st) {  // the home die: its list on the second stream, next to the shared expert
      local = true;
      CHECK(hipStreamWaitEvent(s_local, ev_dispatched, 0));
      const MoeList* dl = d_moe.lists + i;
      fp4_moe(d.pool.view, d_moe.x, dl->ids, dl->w, KE, hp.swiglu_clamp[L], d.y_home, d.scratch, s_local, &dl->count);
      CHECK(hipEventRecord(ev_local, s_local));
      continue;
    }
    // another die: starts when the block is in host memory, no host involvement
    CHECK(hipSetDevice(d.dev));
    CHECK(hipStreamWaitEvent(d.st, ev_staged, 0));
    CHECK(hipMemcpyAsync(d.d_in, h_moe.x, moe_block_bytes, hipMemcpyHostToDevice, d.st));
    const MoeList* dl = (const MoeList*)(d.d_in + E * 4 + 64) + i;
    fp4_moe(d.pool.view, (const float*)d.d_in, dl->ids, dl->w, KE, hp.swiglu_clamp[L], d.d_y, d.scratch, d.st, &dl->count);
    CHECK(hipMemcpyAsync(d.h_y, d.d_y, E * 4, hipMemcpyDeviceToHost, d.st));
    CHECK(hipEventRecord(d.done, d.st));
    CHECK(hipSetDevice(opt.gpu));
  }
  mark("ffn.shared");
  quant(xin, E, A);
  gemv8(w.gate_sh, A, gate);
  gemv8(w.up_sh, A, up);
  k_swiglu(gate, up, hp.swiglu_clamp_sh[L], F, act, s);
  quant(act, F, B);
  gemv8(w.down_sh, B, ysh);
  // the host: the lists (counts for the statistics) and the CPU's misses
  auto t0 = Clock::now();
  CHECK(hipEventSynchronize(ev_staged));
  last.wait_ms += ms_since(t0);
  const MoeList& cpu_list = h_moe.lists[n_dies];
  for (int i = 0; i < n_dies; i++) {
    last.gpu_experts += h_moe.lists[i].count;
    total_gpu_experts += h_moe.lists[i].count;
  }
  last.cpu_experts += cpu_list.count;
  total_cpu_experts += cpu_list.count;
  if (cpu_list.count > 0) {
    mark("ffn.wait_for_cpu_experts");
    auto t1c = Clock::now();
    const Layer& hl = m.layers[L];  // host tensors: the experts live in the mmap
    cpu_moe->forward(CpuExperts{hl.gate_exps, hl.up_exps, hl.down_exps, hp.swiglu_clamp[L]}, h_moe.x, cpu_list.ids, cpu_list.w,
                     cpu_list.count, h_yr);
    last.cpu_moe_ms += ms_since(t1c);
    CHECK(hipMemcpyAsync(yr, h_yr, E * 4, hipMemcpyHostToDevice, s));
    parts[n_dies] = yr;
  }
  // the dies' results, waited for in the stream
  if (n_dies) mark("ffn.wait_for_gpu_experts");
  if (local) CHECK(hipStreamWaitEvent(s, ev_local, 0));
  for (int i = 0; i < n_dies; i++)
    if (dies[i].st) {
      CHECK(hipStreamWaitEvent(s, dies[i].done, 0));
      CHECK(hipMemcpyAsync(dies[i].y_home, dies[i].h_y, E * 4, hipMemcpyHostToDevice, s));
    }
  mark("ffn.add");
  k_add_parts(ysh, parts, n_dies + 1, out, E, s);  // fixed order: dies, then the CPU
  if (taps) {
    tap(taps, "router_w", rw, {KE, 1});  // (synchronizes the stream)
    Tap ti;
    ti.ne = {KE, 1};
    ti.i.resize(KE);
    CHECK(hipMemcpy(ti.i.data(), rids, KE * 4, hipMemcpyDeviceToHost));
    (*taps)["router_i"] = ti;
  }
}

void Runtime::block(int L, Taps* taps) {
  const LayerW& w = W[L];
  const int E = hp.n_embd, HC = hp.hc_mult;
  const float eps = hp.rms_eps;
  mark("engram");
  if (m.layers[L].kind.engram >= 0) engram(L);
  tap(taps, "h_in", h, {E, HC, 1});
  if (dspark) {  // the draft reads this layer's input stream, averaged over the hc copies
    const int t = dspark->target_index(L);
    if (t >= 0) k_hc_pre(h, dspark->quarter, E, dspark->main_hidden + (size_t)t * E, s);
  }
  mark("hc");
  // each sublayer's mixes give the next sublayer's input mix (see Engine::block)
  k_gemv_f32(w.hc_attn_fn, (2 + HC) * HC, E * HC, h, mixes, partial, s);
  k_hc_split(h, E * HC, mixes, w.hc_attn_scale, w.hc_attn_base, eps, hp.hc_eps, hp.hc_iters, a_pre, a_post, a_comb, s);
  k_hc_pre_norm(h, pre, w.attn_norm, E, eps, x, s);
  tap(taps, "attn_in", x, {E, 1});
  attention(L, x, y);
  tap(taps, "attn_out", y, {E, 1});
  mark("hc");
  k_hc_post(y, h, a_post, a_comb, E, h2, s);
  std::swap(h, h2);
  k_gemv_f32(w.hc_ffn_fn, (2 + HC) * HC, E * HC, h, mixes, partial, s);
  k_hc_split(h, E * HC, mixes, w.hc_ffn_scale, w.hc_ffn_base, eps, hp.hc_eps, hp.hc_iters, f_pre, f_post, f_comb, s);
  k_hc_pre_norm(h, a_pre, w.ffn_norm, E, eps, d_moe.x, s);
  tap(taps, "ffn_in", d_moe.x, {E, 1});
  moe(L, d_moe.x, y, taps);
  tap(taps, "ffn_out", y, {E, 1});
  mark("hc");
  k_hc_post(y, h, f_post, f_comb, E, h2, s);
  std::swap(h, h2);
  CHECK(hipMemcpyAsync(pre, f_pre, HC * 4, hipMemcpyDeviceToDevice, s));
  tap(taps, "h_out", h, {E, HC, 1});
  tap(taps, "pre_out", pre, {HC, 1});
  if (taps && m.layers[L].kind.index_source && topk_cur.count(L)) {
    const int n_comp = (pos + 1) / m.layers[L].kind.ratio, K = std::min(hp.idx_topk, n_comp);
    Tap t;
    t.ne = {K, 1};
    t.i.resize(K);
    CHECK(hipMemcpyAsync(t.i.data(), topk_cur[L], K * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    (*taps)["topk"] = t;
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
  topk_cur.clear();
  last = Times{};
  marks.clear();
  mark("engram.fetch");
  fetch_engram_rows();
}

// Routing log format (little endian):
//   header: "DSRT" magic, int32 version = 1, int32 n_layer, int32 k
//   per token: int32 pos, int32 token, int32 is_prompt, int16 ids[n_layer][k], float w[n_layer][k]
// Layers that are not loaded log ids -1.
void Runtime::log_routing(const std::string& path) {
  route_log = fopen(path.c_str(), "wb");
  if (!route_log) throw std::runtime_error("cannot write " + path);
  const int32_t hdr[3] = {1, hp.n_layer, hp.n_expert_used};
  fwrite("DSRT", 1, 4, route_log);
  fwrite(hdr, 4, 3, route_log);
  d_route = dalloc<int>((size_t)hp.n_layer * 16);
}

void Runtime::write_route_record() {
  const int nl = hp.n_layer, k = hp.n_expert_used;
  std::vector<int32_t> buf((size_t)nl * 16);
  CHECK(hipMemcpy(buf.data(), d_route, buf.size() * 4, hipMemcpyDeviceToHost));
  const int32_t head[3] = {pos, token, in_prefill ? 1 : 0};
  std::vector<int16_t> ids((size_t)nl * k, -1);
  std::vector<float> w((size_t)nl * k, 0.0f);
  for (int L = 0; L < nl; L++) {
    if (!m.layers[L].loaded) continue;
    for (int j = 0; j < k; j++) {
      ids[L * k + j] = (int16_t)buf[L * 16 + j];
      memcpy(&w[L * k + j], &buf[L * 16 + 8 + j], 4);
    }
  }
  fwrite(head, 4, 3, route_log);
  fwrite(ids.data(), 2, ids.size(), route_log);
  fwrite(w.data(), 4, w.size(), route_log);
}

void Runtime::end_step() {
  CHECK(hipStreamSynchronize(s));
  if (route_log) write_route_record();
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
