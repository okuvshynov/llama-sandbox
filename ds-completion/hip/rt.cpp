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
#include <set>
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
  mem_devs = {opt.gpu};
  if (!opt.placement.empty())
    for (int d : opt.moe_dies)
      if (std::find(mem_devs.begin(), mem_devs.end(), d) == mem_devs.end()) mem_devs.push_back(d);
  for (int d : mem_devs) {  // create every device's context now, so it is counted first
    CHECK(hipSetDevice(d));
    CHECK(hipFree(nullptr));
  }
  CHECK(hipSetDevice(opt.gpu));
  mem_mark("context, other processes");

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
  mem_mark("dense fp8 weights");
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
  mem_mark("small weights (rotary, embeddings, head, norms, ...)");
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
  logits_d = dalloc<float>((size_t)DSpark::kMaxChunk * hp.n_vocab);  // only verify chunks get logits
  partial = dalloc<float>((size_t)kSplitMax * std::max(hp.n_vocab, 32768));
  A = {dalloc<uint8_t>((size_t)C * NH * HD), dalloc<float>((size_t)C * NH * HD / 32)};
  B = {dalloc<uint8_t>((size_t)C * NH * HD), dalloc<float>((size_t)C * NH * HD / 32)};
  cmp_kvn = dalloc<float>(HD);
  cmp_sc = dalloc<float>(HD);
  lat = dalloc<float>(HD);
  ik = dalloc<float>(hp.idx_dim);
  iq = dalloc<float>(hp.idx_heads * hp.idx_dim);
  iw = dalloc<float>(hp.idx_heads);
  iscore = dalloc<float>(opt.max_ctx);  // one indexer's scores: up to max_ctx compressed positions (ratio 1)
  if (hp.cand_source >= 0) {
    const size_t nb = (size_t)opt.max_ctx / hp.cand_block + 1;
    cand_keep = dalloc<uint8_t>((size_t)C * nb);
    cand_bscore = dalloc<float>(nb);
    cand_bidx = dalloc<int>(hp.cand_topk_blocks);
    for (int L = 0; L < hp.n_layer; L++)  // consumers mask positions in the source's coordinates
      if (m.layers[L].loaded && m.layers[L].kind.uses_cands && hp.compress_ratios[L] != hp.compress_ratios[hp.cand_source])
        throw std::runtime_error("runtime: a candidate consumer with another compression ratio than its source");
  }
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
  att_scratch = dalloc<float>((size_t)kAttnMaxTok * ((hp.window + hp.idx_topk + 31) / 32) * NH * (HD + 2));  // step 68
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
  attn_check = getenv("DS_ATTN_CHECK") != nullptr;  // README step 68
  if (const char* e = getenv("DS_ROUTE_AHEAD")) {  // README step 63
    ra_k = std::max(0, std::min(8, atoi(e)));
    ra_logits = dalloc<float>((size_t)hp.n_layer * (ra_k + 1) * 8 * hp.n_expert);
    ra_act = dalloc<int>((size_t)hp.n_layer * 8 * kMoeIdStride);
    ra_valid.assign((size_t)hp.n_layer * (ra_k + 1), 0);
    ra.assign(ra_k + 1, RaCell{});
    ra_freq.assign(hp.n_layer, {});
    ra_res.assign(hp.n_layer, std::vector<char>(hp.n_expert, 0));
    ra_cnt.assign(hp.n_layer, std::vector<uint16_t>(hp.n_expert, 0));
    ra_tl.assign(hp.n_layer, {});
    if (const char* p = getenv("DS_ROUTE_LOG")) {
      ra_log = fopen(p, "wb");
      if (!ra_log) throw std::runtime_error(std::string("cannot write ") + p);
      const int32_t hdr[4] = {0x48415244 /* "DRAH" */, hp.n_layer, hp.n_expert, ra_k};
      fwrite(hdr, 4, 4, ra_log);
    }
  }
  mem_mark("activations and scratch (sized for 512-token chunks)");
  // the sequence state, zeroed (masked-out slots must hold finite values: 0 x NaN would leak)
  S.resize(hp.n_layer);
  // the window ring: the window plus a chunk, so a chunk writes no position its tokens still need
  // and a rejected draft's writes land on positions no window reads again (README step 66)
  ring_n = hp.window + kChunk;
  for (int L = 0; L < hp.n_layer; L++) {
    const LayerKind& k = m.layers[L].kind;
    if (!m.layers[L].loaded) continue;
    S[L].win = dalloc<float>((size_t)ring_n * HD);
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
  CHECK(hipEventCreateWithFlags(&ev_staged, hipEventDisableTiming));

  mem_mark("sequence state: window, compressed KV, indexer keys (--max-ctx)");
  if (!opt.dspark.empty()) dspark.reset(new DSpark(*this, opt.dspark));  // before the pools take the rest
  mem_mark("draft (DSpark)");
  setup_experts();

  // the per-token buffers, for addressing half a chunk (set_half); h / h2 are kept per half
  {
    const int KE = hp.n_expert_used, rows = opt.canon ? KE : 1;
    auto add = [&](void* p, size_t bytes) { tok_ptrs.push_back({(void**)p, bytes}); };
    for (float** p : {&pre, &a_pre, &a_post, &f_pre, &f_post}) add(p, (size_t)HC * 4);
    for (float** p : {&a_comb, &f_comb}) add(p, (size_t)HC * HC * 4);
    add(&mixes, (size_t)(2 + HC) * HC * 4);
    for (float** p : {&x, &y, &ysh, &yr}) add(p, (size_t)E * 4);
    for (float** p : {&t1, &qr}) add(p, (size_t)hp.q_lora * 4);
    for (float** p : {&q, &o}) add(p, (size_t)NH * HD * 4);
    add(&kvv, (size_t)HD * 4);
    add(&oa, (size_t)hp.o_groups * hp.o_lora * 4);
    for (float** p : {&gate, &up, &act}) add(p, (size_t)hp.n_ff_exp * 4);
    add(&router_logits, (size_t)hp.n_expert * 4);
    add(&rw, (size_t)kMoeIdStride * 4);
    add(&rids, (size_t)kMoeIdStride * 4);
    add(&ekv, (size_t)E * (HC + 1) * 4);
    add(&erows, (size_t)hp.engram_cols() * hp.engram_dim * 4);
    add(&d_moe.lists, (size_t)kMoeListStride * sizeof(MoeList));
    add(&h_moe.lists, (size_t)kMoeListStride * sizeof(MoeList));
    add(&d_moe.x, (size_t)E * 4);
    add(&h_moe.x, (size_t)E * 4);
    add(&h_yr, (size_t)rows * E * 4);
    add(&canon_rows, (size_t)KE * E * 4);  // null without --canon (skipped)
    add(&canon_where, (size_t)kMoeIdStride * 4);
    add(&d_route, (size_t)hp.n_layer * 16 * 4);  // null unless --log-routing
    for (Die& d : dies) {
      add(&d.y_home, (size_t)E * 4);
      add(&d.d_y, (size_t)rows * E * 4);
      add(&d.h_y, (size_t)rows * E * 4);
    }
    if (dspark) add(&dspark->hid_chunk, dspark->target_layers.size() * (size_t)E * 4);
    // half 0 uses the events above, half 1 its own
    half_ev[0] = {ev_staged, ev_dispatched, ev_local, {}};
    half_ev[1] = {nullptr, nullptr, nullptr, {}};
    auto ev = [&]() { hipEvent_t e; CHECK(hipEventCreateWithFlags(&e, hipEventDisableTiming)); return e; };
    half_ev[1].staged = ev();
    if (ev_dispatched) half_ev[1].dispatched = ev();
    if (ev_local) half_ev[1].local = ev();
    for (Die& d : dies) {
      CHECK(hipSetDevice(d.dev));
      half_ev[0].done.push_back(d.done);
      half_ev[1].done.push_back(d.done ? ev() : nullptr);
    }
    CHECK(hipSetDevice(opt.gpu));
  }
  mem_mark("cache and loader staging, the rest");
  mem_report();
}

// the lookahead study (README step 63): per lookahead j, over every (token, layer) whose
// prediction was made: the chosen experts among the top 6 / 12 / 24 of the earlier layer's
// scores, and the chosen CPU experts among the top 1 / 3 / 6 non-resident experts by those scores
// (what a prefetch would fetch) and by the cache's use counts (step 52's guess)
void Runtime::ra_step_end() {
  bool any = false;
  for (char v : ra_valid) any |= v;
  if (!any) return;
  const int X = hp.n_expert, K1 = ra_k + 1;
  CHECK(hipStreamSynchronize(s));
  std::vector<float> lg((size_t)hp.n_layer * K1 * 8 * X);
  std::vector<int> act((size_t)hp.n_layer * 8 * kMoeIdStride);
  CHECK(hipMemcpy(lg.data(), ra_logits, lg.size() * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(act.data(), ra_act, act.size() * 4, hipMemcpyDeviceToHost));
  std::vector<int> order(X);
  std::vector<float> sc(X);
  for (int L = 0; L < hp.n_layer; L++) {
    if (!ra_valid[(size_t)L * K1]) continue;  // the layer itself ran this step (j = 0 is always made)
    const float* bias = (const float*)m.layers[L].gate_bias->data;
    const std::vector<int>& freq = ra_freq[L];  // taken before this step's routing was counted
    for (int j = 0; j < K1; j++) {
      if (!ra_valid[(size_t)L * K1 + j]) continue;
      RaCell& c = ra[j];
      for (int t = 0; t < ra_m; t++) {
        const float* l = lg.data() + (((size_t)L * K1 + j) * 8 + t) * X;
        for (int e = 0; e < X; e++) {
          const float x = l[e];
          sc[e] = sqrtf(x > 20.0f ? x : log1pf(expf(x))) + bias[e];
          order[e] = e;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] != sc[b] ? sc[a] > sc[b] : a < b; });
        std::vector<int> nonres;  // the top non-resident by these scores
        for (int i = 0; i < X && (int)nonres.size() < 6; i++)
          if (!ra_res[L][order[i]]) nonres.push_back(order[i]);
        const int* a = act.data() + ((size_t)L * 8 + t) * kMoeIdStride;
        for (int k = 0; k < hp.n_expert_used; k++) {
          const int e = a[k];
          int rank = 0;
          while (rank < 24 && order[rank] != e) rank++;
          c.exp++;
          c.in6 += rank < 6;
          c.in12 += rank < 12;
          c.in24 += rank < 24;
          if (!ra_res[L][e]) {
            c.cpu++;
            for (int i = 0; i < (int)nonres.size(); i++)
              if (nonres[i] == e) { c.p1 += i < 1; c.p3 += i < 3; c.p6 += i < 6; }
            for (int i = 0; i < (int)freq.size(); i++)
              if (freq[i] == e) { c.f1 += i < 1; c.f3 += i < 3; c.f6 += i < 6; }
          }
        }
      }
    }
  }
  if (ra_log) {  // one record per step: M, the end time, then per layer (README step 63's log)
    const auto tend = Clock::now();
    auto t0 = tend;
    for (int L = 0; L < hp.n_layer; L++)
      if (ra_valid[(size_t)L * K1] && ra_tl[L] < t0) t0 = ra_tl[L];
    auto ms = [&](Clock::time_point a) { return (float)std::chrono::duration<double, std::milli>(a - t0).count(); };
    const int32_t m32 = ra_m;
    const float te = ms(tend);
    fwrite(&m32, 4, 1, ra_log);
    fwrite(&te, 4, 1, ra_log);
    constexpr int TOP = 64;
    std::vector<int16_t> ids(TOP);
    std::vector<float> scs(TOP);
    for (int L = 0; L < hp.n_layer; L++) {
      const uint8_t valid = ra_valid[(size_t)L * K1];
      fwrite(&valid, 1, 1, ra_log);
      if (!valid) continue;
      const float tl = ms(ra_tl[L]);
      fwrite(&tl, 4, 1, ra_log);
      for (int t = 0; t < ra_m; t++)
        for (int k = 0; k < 6; k++) {
          const int16_t e = (int16_t)act[((size_t)L * 8 + t) * kMoeIdStride + k];
          fwrite(&e, 2, 1, ra_log);
        }
      fwrite(ra_res[L].data(), 1, X, ra_log);
      fwrite(ra_cnt[L].data(), 2, X, ra_log);
      const float* bias = (const float*)m.layers[L].gate_bias->data;
      for (int j = 1; j < K1; j++) {  // the guess made j layers earlier (absent: ids -1)
        const bool have = ra_valid[(size_t)L * K1 + j];
        for (int t = 0; t < ra_m; t++) {
          if (have) {
            const float* l = lg.data() + (((size_t)L * K1 + j) * 8 + t) * X;
            for (int e = 0; e < X; e++) {
              sc[e] = sqrtf(l[e] > 20.0f ? l[e] : log1pf(expf(l[e]))) + bias[e];
              order[e] = e;
            }
            std::partial_sort(order.begin(), order.begin() + TOP, order.end(), [&](int a, int b) { return sc[a] != sc[b] ? sc[a] > sc[b] : a < b; });
            for (int i = 0; i < TOP; i++) ids[i] = (int16_t)order[i], scs[i] = sc[order[i]];
          } else {
            std::fill(ids.begin(), ids.end(), -1);
            std::fill(scs.begin(), scs.end(), 0.0f);
          }
          fwrite(ids.data(), 2, TOP, ra_log);
          fwrite(scs.data(), 4, TOP, ra_log);
        }
      }
    }
  }
  std::fill(ra_valid.begin(), ra_valid.end(), 0);
}

void Runtime::ra_report() const {
  if (ra.empty() || !ra[0].exp) return;
  fprintf(stderr, "# router lookahead (README step 63): layer L+j's router on layer L's input, generation\n");
  fprintf(stderr, "#   j   chosen experts in the guess's top 6 / 12 / 24 |   CPU experts in the guess's top 1 / 3 / 6 non-resident | cache's top 1 / 3 / 6\n");
  for (size_t j = 0; j < ra.size(); j++) {
    const RaCell& c = ra[j];
    if (!c.exp) continue;
    const double e = (double)c.exp, u = std::max(1.0, (double)c.cpu);
    fprintf(stderr, "#   %zu   %5.1f%% %5.1f%% %5.1f%%  (%ld)              |   %5.1f%% %5.1f%% %5.1f%%  (%ld)                        | %5.1f%% %5.1f%% %5.1f%%\n", j,
            100 * c.in6 / e, 100 * c.in12 / e, 100 * c.in24 / e, c.exp, 100 * c.p1 / u, 100 * c.p3 / u, 100 * c.p6 / u, c.cpu,
            100 * c.f1 / u, 100 * c.f3 / u, 100 * c.f6 / u);
  }
}

void Runtime::mem_mark(const char* phase) {
  std::vector<size_t> used;
  int cur;
  CHECK(hipGetDevice(&cur));
  if (mem_total.empty()) mem_total.assign(mem_devs.size(), 0);
  for (size_t i = 0; i < mem_devs.size(); i++) {
    size_t fr, tot;
    CHECK(hipSetDevice(mem_devs[i]));
    CHECK(hipMemGetInfo(&fr, &tot));
    used.push_back(tot - fr);
    mem_total[i] = tot;
  }
  CHECK(hipSetDevice(cur));
  mem_snaps.push_back({phase, used});
}

// one line per phase: GB per device (and, for the expert dies, what it would hold in expert slots)
void Runtime::mem_report() const {
  if (mem_snaps.empty()) return;
  const double slot = 3.0 * hp.n_embd * hp.n_ff_exp / 32 * 17;  // an expert slot: gate + up + down in mxfp4
  fprintf(stderr, "# GPU memory (GB; the driver's used memory after each startup phase; expert slot = %.1f MB):\n", slot / 1e6);
  fprintf(stderr, "#   %-62s", "phase");
  for (int d : mem_devs) fprintf(stderr, "  GPU %d", d);
  fprintf(stderr, "\n");
  for (size_t p = 0; p < mem_snaps.size(); p++) {
    fprintf(stderr, "#   %-62s", mem_snaps[p].first.c_str());
    for (size_t i = 0; i < mem_devs.size(); i++) {
      const double b = (double)mem_snaps[p].second[i] - (p ? (double)mem_snaps[p - 1].second[i] : 0.0);
      fprintf(stderr, " %6.2f", b / 1e9);
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "#   %-62s", "free");
  for (size_t i = 0; i < mem_devs.size(); i++) fprintf(stderr, " %6.2f", (double)(mem_total[i] - mem_snaps.back().second[i]) / 1e9);
  fprintf(stderr, "\n#   %-62s", "total");
  for (size_t i = 0; i < mem_devs.size(); i++) fprintf(stderr, " %6.2f", (double)mem_total[i] / 1e9);
  fprintf(stderr, "\n");
}

// every per-token buffer from token tok_base to token t0 (null buffers stay null)
void Runtime::shift_tokens(int t0) {
  const ptrdiff_t dt = t0 - tok_base;
  for (const TokPtr& tp : tok_ptrs)
    if (*tp.p) *tp.p = (char*)*tp.p + dt * (ptrdiff_t)tp.bytes;
  tok_base = t0;
}

void Runtime::set_half(int i, int t0) {
  const size_t hb = (size_t)hp.n_embd * hp.hc_mult;
  if (cur_half >= 0) {  // keep the current half's stream pair (unshifted)
    half_h[cur_half] = h - tok_base * hb;
    half_h2[cur_half] = h2 - tok_base * hb;
  }
  shift_tokens(i < 0 ? 0 : t0);
  if (i >= 0) {
    h = half_h[i] + tok_base * hb;
    h2 = half_h2[i] + tok_base * hb;
    ev_staged = half_ev[i].staged;
    ev_dispatched = half_ev[i].dispatched;
    ev_local = half_ev[i].local;
    for (size_t d = 0; d < dies.size(); d++) dies[d].done = half_ev[i].done[d];
  } else {
    h = half_h[0];  // both halves end every layer on the same buffer (two swaps per block)
    h2 = half_h2[0];
    ev_staged = half_ev[0].staged;
    ev_dispatched = half_ev[0].dispatched;
    ev_local = half_ev[0].local;
    for (size_t d = 0; d < dies.size(); d++) dies[d].done = half_ev[0].done[d];
  }
  cur_half = i;
}

std::vector<int32_t> Runtime::verify(const std::vector<int32_t>& chunk, bool head) {
  const int M = (int)chunk.size(), HD = hp.head_dim;
  if (M < 1 || M > DSpark::kMaxChunk) throw std::runtime_error("verify: chunk of " + std::to_string(M));
  verify_p0 = n_past_;
  verify_m = M;
  // (the window ring needs no backup: a rejected draft's slots are written again, by the token
  // that takes its position, before any window reads them, and they overwrite nothing still read)
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
  const auto t_v = Clock::now();
  std::vector<int32_t> next = forward_chunk(chunk, head);
  trace_host(("verify " + std::to_string(M) + " tokens").c_str(), t_v, Clock::now(), 3);
  verifying = false;
  return next;
}

// M tokens at positions n_past .. in one forward: every token computed exactly as decode()
// would compute it alone (batched kernels are bit-identical per row), the greedy next token
// after each. The sequence advances by M.
std::vector<int32_t> Runtime::forward_chunk(const std::vector<int32_t>& toks, bool head) {
  const int M = (int)toks.size(), E = hp.n_embd, HC = hp.hc_mult;
  const int64_t V = hp.n_vocab;
  if (M < 1 || M > kChunk || (head && M > DSpark::kMaxChunk)) throw std::runtime_error("forward_chunk: " + std::to_string(M) + " tokens");
  if (n_past_ + M > opt.max_ctx) throw std::runtime_error("runtime: chunk past --max-ctx");
  auto t0 = Clock::now();
  pos = n_past_;
  for (auto& tc : topk_cur) tc.clear();
  std::fill(cand_cur.begin(), cand_cur.end(), nullptr);
  last = Times{};
  marks.clear();
  cur_layer = -1;
  trace_step_begin();
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
  chunked = true;
  if (!head && opt.prefill_overlap && M >= 2) {
    // two halves A, B: pre(L, A) pre(L, B) post(L, A) pre(L + 1, A) post(L, B) pre(L + 1, B) ...
    // The host computes A's CPU experts while the GPU runs B's attention, and B's while the GPU
    // runs A's next layer. Causal order holds: B's attention at L follows A's at L.
    const int n0 = M / 2, n[2] = {n0, M - n0}, t0[2] = {0, n0};
    half_h[0] = half_h[1] = h;
    half_h2[0] = half_h2[1] = h2;
    auto pre_h = [&](int L, int i) { set_half(i, t0[i]); pos = n_past_ + t0[i]; block_pre(L, nullptr, n[i]); };
    auto post_h = [&](int L, int i) { set_half(i, t0[i]); pos = n_past_ + t0[i]; block_post(L, nullptr, n[i]); };
    pre_h(0, 0);
    pre_h(0, 1);
    for (int L = 0; L < hp.n_layer; L++) {
      post_h(L, 0);
      if (L + 1 < hp.n_layer) pre_h(L + 1, 0);
      post_h(L, 1);
      if (L + 1 < hp.n_layer) pre_h(L + 1, 1);
    }
    set_half(-1, 0);
    pos = n_past_;
  } else {
    for (int L = 0; L < hp.n_layer; L++) block(L, nullptr, M);
  }
  cur_layer = -1;
  chunked = false;
  std::vector<int32_t> next;
  if (head) {
    mark("head");
    k_hc_pre(h, pre, E, x, s, M);
    k_rms_norm(x, out_norm, x, E, M, hp.rms_eps, s);
    k_gemv_bf16_m(out_w, (int)V, E, x, M, logits_d, partial, s);
    std::vector<float> lg((size_t)M * V);
    mark("end");
    CHECK(hipMemcpyAsync(lg.data(), logits_d, lg.size() * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    next.resize(M);
    for (int t = 0; t < M; t++)
      next[t] = (int32_t)(std::max_element(lg.begin() + (size_t)t * V, lg.begin() + (size_t)(t + 1) * V) - (lg.begin() + (size_t)t * V));
  } else {
    mark("end");
    CHECK(hipStreamSynchronize(s));
  }
  if (route_log) write_route_records(toks.data(), M);
  if (cache) cache->step(M);  // a step boundary: the forward's GPU work has finished
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
  if (profile) fold_misses();
  trace_step_end();
  if (ra_logits) ra_step_end();
  last.total_ms = ms_since(t0);
  return next;
}

void Runtime::commit(int n_keep) {
  const int HD = hp.head_dim, M = verify_m;
  if (n_keep < 1 || n_keep > M || n_past_ != verify_p0 + M) throw std::runtime_error("commit without a matching verify");
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
  const auto t0 = Clock::now();
  DraftResult r = dspark->draft(anchor, n_past_ - 1, keep_logits);
  trace_host("draft", t0, Clock::now(), 3);
  return r;
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
  mem_mark("expert path buffers (per die: scratch, staging)");
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
  mem_mark("expert slots");
  CHECK(hipSetDevice(home));
  std::vector<int>& table = h_table;
  table.assign((size_t)hp.n_layer * X, -1);
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
  if (opt.cache_loads > 0) {
    if (opt.load_test >= 0) throw std::runtime_error("--load-test and the expert cache are exclusive");
    std::vector<ExpertPool*> pools;
    for (Die& d : dies) pools.push_back(&d.pool);
    cache.reset(new ExpertCache(m, pools, h_table, d_table, X, {opt.cache_loads, opt.cache_period, opt.cache_window, opt.cache_wait, opt.cache_min}));
    fprintf(stderr, "# expert cache: up to %.0f loads/s, rebalance every %d tokens over the last %d%s\n", opt.cache_loads,
            opt.cache_period, opt.cache_window, opt.canon ? "" : " (without --canon the output depends on load timing)");
    CHECK(hipSetDevice(home));
  }
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
  trace_write();
  ra_report();
  if (attn_check)
    fprintf(stderr, "# attention check (README step 68): %ld launches, %ld tokens, %ld differ from one-token launches\n",
            attn_check_calls, attn_check_tokens, attn_check_bad);
  if (ra_log) fclose(ra_log);
  if (loader) {
    fprintf(stderr, "# load test: %ld loads\n", loader->loads());
    loader.reset();
  }
  if (cache) {
    const ExpertCache::Stats cs = cache->stats();
    fprintf(stderr, "# expert cache: %ld rebalances, %ld swaps planned, %ld loads started, %ld published\n", cs.rebalances,
            cs.swaps_planned, cs.loads_started, cs.loads_published);
    cache.reset();
  }
  if (profile) {
    long tot = 0;
    for (auto& row : miss_hist) for (auto& c : row) tot += c.n;
    if (tot) {
      static const char* mn[4] = {"1", "2-3", "4-6", "7+"};
      static const char* dn[7] = {"0", "1", "2", "3", "4", "5-7", "8+"};
      fprintf(stderr, "# CPU experts per layer (generation): tokens in the forward, distinct CPU experts: layers, mean pairs, "
              "CPU ms, the GPU's wait ms (mean per layer)\n");
      for (int a = 0; a < 4; a++)
        for (int b = 0; b < 7; b++) {
          const MissCell& c = miss_hist[a][b];
          if (!c.n) continue;
          fprintf(stderr, "#   %3s tokens, %3s experts: %6ld layers (%4.1f%%), %5.2f pairs, CPU %6.3f ms, GPU waits %6.3f ms\n", mn[a], dn[b],
                  c.n, 100.0 * c.n / tot, (double)c.pairs / c.n, c.cpu_ms / c.n, c.wait_ms / c.n);
        }
    }
  }
  if (profile) {
    long tot = 0;
    for (auto& p : pred_hist) tot += p.layers;
    if (tot) {
      static const char* mn[4] = {"1", "2-3", "4-6", "7+"};
      fprintf(stderr, "# prefetch prediction (generation): tokens in the forward: layers, with CPU experts; the guess among them: "
              "cache's top-1 non-resident, any of its top 3, previous forward's top CPU expert; guesses on layers without any\n");
      for (int a = 0; a < 4; a++) {
        const PredCell& p = pred_hist[a];
        if (!p.layers) continue;
        const double w = std::max(1L, p.with_cpu);
        fprintf(stderr, "#   %3s tokens: %6ld layers, %6ld with CPU experts (%4.1f%%); top-1 %4.1f%%, top-3 %4.1f%%, previous %4.1f%%; "
                "wasted on %4.1f%% of layers\n", mn[a], p.layers, p.with_cpu, 100.0 * p.with_cpu / p.layers, 100.0 * p.freq1 / w,
                100.0 * p.freq3 / w, 100.0 * p.prev1 / w, 100.0 * p.wasted / p.layers);
      }
    }
  }
  if (profile && cpu_moe && cpu_moe->bstats.calls) {
    const auto& b = cpu_moe->bstats;
    fprintf(stderr, "# CPU experts, batched: %ld calls, %ld pairs over %ld experts (by tokens: 1 %ld, 2-3 %ld, 4-7 %ld, 8-15 %ld, 16+ %ld); "
            "ms: prep %.0f inputs %.0f gate_up %.0f act %.0f down %.0f sum %.0f (%.3f ms per pair)\n",
            b.calls, b.pairs, b.experts, b.by_tokens[0], b.by_tokens[1], b.by_tokens[2], b.by_tokens[3], b.by_tokens[4], b.prep,
            b.inputs, b.gate_up, b.act, b.down, b.sum, (b.prep + b.inputs + b.gate_up + b.act + b.down + b.sum) / b.pairs);
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
                  (void*)cmp_sc, (void*)lat, (void*)ik, (void*)iq, (void*)iw, (void*)iscore, (void*)cand_keep, (void*)cand_bscore, (void*)cand_bidx, (void*)gate, (void*)up,
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
  // at most 8 inputs per call; every row's result is the same at any M (fp8_gemv.h)
  const int rpg = rows_per_group ? rows_per_group : w.N, Kt = w.K * (w.N / rpg);
  if (M > 8) {  // prompt chunks: the many-input kernel (the same bits as one input at a time)
    fp8_gemm(w.q, w.s, w.N, w.K, rpg, Fp8Act{c.q, c.s}, M, yout, s);
    return;
  }
  for (int m0 = 0; m0 < M; m0 += 8)
    fp8_gemv(w.q, w.s, w.N, w.K, rpg, Fp8Act{c.q + (size_t)m0 * Kt, c.s + (size_t)m0 * Kt / 32}, std::min(8, M - m0),
             yout + (size_t)m0 * w.N, s);
}

// profile: each layer's GPU wait for the CPU (its "ffn.wait_for_cpu_experts" interval) next to
// what the CPU had to do there, folded into the histogram (call once the step's marks are done)
void Runtime::fold_misses() {
  for (const MissRec& r : miss_recs) {
    float w = 0;
    if (r.mark >= 0 && r.mark + 1 < (int)marks.size()) CHECK(hipEventElapsedTime(&w, marks[r.mark].second, marks[r.mark + 1].second));
    const int mb = r.M == 1 ? 0 : r.M <= 3 ? 1 : r.M <= 6 ? 2 : 3;
    const int db = r.distinct <= 4 ? r.distinct : r.distinct <= 7 ? 5 : 6;
    MissCell& c = miss_hist[mb][db];
    c.n++;
    c.pairs += r.pairs;
    c.cpu_ms += r.cpu_ms;
    c.wait_ms += w;
  }
  miss_recs.clear();
}

// ------------------------------------------------------------------ trace (README step 51)

void Runtime::set_trace(const std::string& path) {
  trace_path = path;
  profile = true;
  trace_epoch = Clock::now();
  CHECK(hipEventCreate(&tr_staged));
  die_trace.resize(dies.size());
  for (size_t i = 0; i < dies.size(); i++) {
    DieTrace& t = die_trace[i];
    t.st = dies[i].st ? dies[i].st : s_local;
    CHECK(hipSetDevice(dies[i].dev));
    CHECK(hipEventCreate(&t.base));
  }
  CHECK(hipSetDevice(opt.gpu));
}

void Runtime::trace_host(const char* name, Clock::time_point a, Clock::time_point b, int tid, const std::string& args) {
  if (trace_path.empty()) return;
  const double ts = std::chrono::duration<double, std::micro>(a - trace_epoch).count();
  trace_evs.push_back({name, tid, ts, std::chrono::duration<double, std::micro>(b - a).count(), args});
}

hipEvent_t Runtime::trace_ev(int die) {
  DieTrace& t = die_trace[die];
  if (t.used == t.pool.size()) {
    hipEvent_t e;
    CHECK(hipEventCreate(&e));
    t.pool.push_back(e);
  }
  return t.pool[t.used++];
}

// at a step's start: a reference event on every die's expert stream. The other devices' streams
// are idle then, so the host waits for the event and reads its clock (exact to the wake-up); the
// home die's second stream shares die 0's clock and is placed from die 0's anchor at the end.
void Runtime::trace_step_begin() {
  if (trace_path.empty()) return;
  step_anchor_us = 1e300;
  for (size_t i = 0; i < die_trace.size(); i++) {
    DieTrace& t = die_trace[i];
    t.used = 0;
    t.spans.clear();
    CHECK(hipSetDevice(dies[i].dev));
    CHECK(hipEventRecord(t.base, t.st));
  }
  for (size_t i = 0; i < die_trace.size(); i++) {
    if (!dies[i].st) continue;
    CHECK(hipEventSynchronize(die_trace[i].base));
    die_trace[i].base_us = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  }
  CHECK(hipSetDevice(opt.gpu));
}

// at a step's end (its events have completed): die 0's groups per layer, the dies' expert spans
void Runtime::trace_step_end() {
  if (trace_path.empty() || marks.empty()) return;
  // each group ends where the next begins, so a layer's span (its groups in a row; the prompt's
  // two halves interleave, giving two per layer) contains them exactly and Perfetto nests them
  const double t0 = step_anchor_us < 1e299 ? step_anchor_us : step_t0_us;
  std::vector<double> at(marks.size());
  for (size_t i = 0; i < marks.size(); i++) {
    float a;
    CHECK(hipEventElapsedTime(&a, marks[0].second, marks[i].second));
    at[i] = t0 + a * 1000.0;
  }
  for (size_t i = 0; i < die_trace.size(); i++)
    if (!dies[i].st) {
      float a;
      CHECK(hipEventElapsedTime(&a, marks[0].second, die_trace[i].base));
      die_trace[i].base_us = t0 + a * 1000.0;
    }
  if (step_anchor_us < 1e299)  // generation (anchored clocks): the issue-to-start lags
    for (size_t i = 0; i + 1 < marks.size(); i++) {
      const double dur = at[i + 1] - at[i], lag = at[i + 1] - mark_host[i + 1];
      gen_gpu_us += dur;
      const int b = lag < 0 ? 0 : lag < 30 ? 1 : lag < 100 ? 2 : lag < 300 ? 3 : lag < 1000 ? 4 : 5;
      lag_hist[b]++;
      if (lag < 30) {
        hb_total_us += dur;
        hb_by_next[marks[i + 1].first] += dur;
      }
    }
  for (size_t i = 0; i + 1 < marks.size(); i++) {
    const int L = mark_layer[i];
    if (L >= 0 && (i == 0 || mark_layer[i - 1] != L)) {
      size_t j = i;
      while (j + 1 < marks.size() && mark_layer[j] == L) j++;
      trace_evs.push_back({"layer " + std::to_string(L), 2, at[i], at[j] - at[i], ""});
    }
    std::string name = marks[i].first;
    if (L >= 0) name += " L" + std::to_string(L);
    trace_evs.push_back({name, 2, at[i], at[i + 1] - at[i], ""});
  }
  for (size_t i = 0; i < die_trace.size(); i++) {
    DieTrace& t = die_trace[i];
    CHECK(hipSetDevice(dies[i].dev));
    for (const DieTrace::Span& sp : t.spans) {
      float a, d;
      CHECK(hipEventSynchronize(sp.b));
      CHECK(hipEventElapsedTime(&a, t.base, sp.a));
      CHECK(hipEventElapsedTime(&d, sp.a, sp.b));
      trace_evs.push_back({"experts L" + std::to_string(sp.L), 10 + (int)i, t.base_us + a * 1000.0, d * 1000.0, sp.args});
    }
  }
  CHECK(hipSetDevice(opt.gpu));
}

void Runtime::trace_write() {
  if (trace_path.empty()) return;
  FILE* f = fopen(trace_path.c_str(), "w");
  if (!f) return;
  fprintf(f, "{\"traceEvents\":[\n");
  auto meta = [&](int tid, const std::string& name, bool first = false) {
    fprintf(f, "%s{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"args\":{\"name\":\"%s\"}}", first ? "" : ",\n", tid, name.c_str());
  };
  meta(1, "host thread", true);
  meta(3, "steps (host)");
  meta(2, "GPU " + std::to_string(opt.gpu) + " main stream (dense part)");
  for (size_t i = 0; i < dies.size(); i++)
    meta(10 + (int)i, "GPU " + std::to_string(dies[i].dev) + (dies[i].st ? " experts" : " experts (second stream)"));
  // start and end rounded to 0.1 us each, so a span that ends where its parent ends still nests
  for (const TraceEv& e : trace_evs) {
    const double a = std::round(e.ts * 10) / 10, b = std::round((e.ts + e.dur) * 10) / 10;
    fprintf(f, ",\n{\"name\":\"%s\",\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"ts\":%.1f,\"dur\":%.1f%s%s%s}", e.name.c_str(), e.tid, a,
            b - a, e.args.empty() ? "" : ",\"args\":{", e.args.c_str(), e.args.empty() ? "" : "}");
  }
  fprintf(f, "\n]}\n");
  fclose(f);
  fprintf(stderr, "# trace: %zu events in %s (open in ui.perfetto.dev)\n", trace_evs.size(), trace_path.c_str());
  if (gen_gpu_us > 0) {  // README step 65
    fprintf(stderr, "# GPU 0 issue-to-start lag per group (generation): <0 %ld, 0-30 us %ld, 30-100 %ld, 100-300 %ld, 300-1000 %ld, >1 ms %ld\n",
            lag_hist[0], lag_hist[1], lag_hist[2], lag_hist[3], lag_hist[4], lag_hist[5]);
    fprintf(stderr, "# host-bound estimate: %.1f of %.1f ms on GPU 0's stream (%.1f%%) in groups whose next group started within 30 us of being issued; by that group:",
            hb_total_us / 1e3, gen_gpu_us / 1e3, 100 * hb_total_us / gen_gpu_us);
    std::vector<std::pair<double, std::string>> v;
    for (auto& [k, x] : hb_by_next) v.push_back({x, k});
    std::sort(v.rbegin(), v.rend());
    for (size_t i = 0; i < v.size() && i < 8; i++) fprintf(stderr, " %s %.1f", v[i].second.c_str(), v[i].first / 1e3);
    fprintf(stderr, " ms\n");
  }
}

void Runtime::mark(const char* group) {
  if (!profile) return;
  if (marks.size() == ev_pool.size()) {
    hipEvent_t e;
    CHECK(hipEventCreate(&e));
    ev_pool.push_back(e);
  }
  hipEvent_t e = ev_pool[marks.size()];
  if (marks.empty()) step_t0_us = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  CHECK(hipEventRecord(e, s));
  marks.emplace_back(group, e);
  if (mark_layer.size() < marks.size()) mark_layer.resize(marks.size());
  mark_layer[marks.size() - 1] = cur_layer;
  if (!trace_path.empty()) {
    if (mark_host.size() < marks.size()) mark_host.resize(marks.size());
    mark_host[marks.size() - 1] = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
  }
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
  // only this forward's rows of each slot (README step 48: copying all kChunk rows cost a decode
  // step ~2 ms once kChunk grew to 512)
  for (int e = 0; e < n_slots; e++)
    CHECK(hipMemcpyAsync(erows + (size_t)e * kChunk * n, h_erows + (size_t)e * kChunk * n, chunk.size() * n * 4,
                         hipMemcpyHostToDevice, s));
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
  // README step 67: in phases. (A) every token's rope of q and its window KV into the ring (the
  // ring holds window + a chunk: no token's write touches a position another still needs);
  // (B) token by token, the compressor and the indexer (a group completed mid-chunk makes a
  // compressed position for the later tokens only: each token's count and top-k are its own);
  // (C) each token's attention; (D) the inverse rope of every output.
  const LayerKind& kind = m.layers[L].kind;
  const int RD = hp.rope_dim, WIN = hp.window;
  k_rope_m(q, M, NH, HD, RD, w.freqs, p0, false, s);
  k_kv_finish_m(kvv, M, w.kv_norm, HD, RD, w.freqs, p0, eps, S[L].win, ring_n, s);
  att_K.assign(M, 0);
  att_topk.assign(M, nullptr);
  const float* comp = kind.ratio ? S[kind.kv_from].comp : nullptr;
  for (int t = 0; t < M; t++) {
    pos = p0 + t;
    attn_prep(L, tok_base + t, xin + (size_t)t * E, Codes{B.q + (size_t)t * hp.q_lora, B.s + (size_t)t * hp.q_lora / 32},
              att_K[t], att_topk[t]);
  }
  mark("attn.core");
  const float scale = 1.0f / sqrtf((float)HD);
  for (int t0 = 0; t0 < M; t0 += kAttnMaxTok) {  // README step 68: up to 8 tokens per launch
    const int n = std::min(kAttnMaxTok, M - t0);
    k_attn_decode_m(q + (size_t)t0 * NH * HD, n, NH, HD, S[L].win, WIN, p0 + t0, comp, att_K.data() + t0,
                    att_topk.data() + t0, w.sinks, scale, o + (size_t)t0 * NH * HD, att_scratch, s, ring_n);
    if (attn_check) {  // each token again alone, compared bit for bit (DS_ATTN_CHECK)
      if (!attn_check_buf) attn_check_buf = dalloc<float>((size_t)kAttnMaxTok * NH * HD);
      for (int i = 0; i < n; i++)
        k_attn_decode(q + (size_t)(t0 + i) * NH * HD, NH, HD, S[L].win, WIN, p0 + t0 + i, comp, att_topk[t0 + i],
                      att_K[t0 + i], w.sinks, scale, attn_check_buf + (size_t)i * NH * HD, att_scratch, s, ring_n);
      std::vector<float> a((size_t)n * NH * HD), b(a.size());
      CHECK(hipMemcpyAsync(a.data(), o + (size_t)t0 * NH * HD, a.size() * 4, hipMemcpyDeviceToHost, s));
      CHECK(hipMemcpyAsync(b.data(), attn_check_buf, b.size() * 4, hipMemcpyDeviceToHost, s));
      CHECK(hipStreamSynchronize(s));
      attn_check_calls++;
      attn_check_tokens += n;
      for (int i = 0; i < n; i++) attn_check_bad += memcmp(a.data() + (size_t)i * NH * HD, b.data() + (size_t)i * NH * HD, (size_t)NH * HD * 4) != 0;
    }
  }
  k_rope_m(o, M, NH, HD, RD, w.freqs, p0, true, s);  // take the queries' rotation back out
  pos = p0;
  mark("attn.out");
  quant(o, NH * HD, B, M);
  gemv8(w.out_a, B, oa, hp.o_lora, M);  // block-diagonal over the 8 groups
  quant(oa, hp.o_groups * hp.o_lora, B, M);
  gemv8(w.out_b, B, out, 0, M);
}

// one token (chunk index t, at position pos): the compressor and the indexer; K and topk: the
// compressed keys its attention takes (README step 67: the rope, the ring and the attention run
// for the whole chunk in attention())
void Runtime::attn_prep(int L, int t, const float* xin, Codes qr_t, int& K, const int*& topk) {
  const LayerW& w = W[L];
  const LayerKind& k = m.layers[L].kind;
  const int HD = hp.head_dim, RD = hp.rope_dim, E = hp.n_embd;
  const float eps = hp.rms_eps;
  std::map<int, const int*>& tcur = topk_cur[t];
  topk = nullptr;
  K = 0;
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
    mark("attn.indexer");
    if (k.index_source && K > 0) {  // Indexer.forward, one query
      gemv8(w.idx_q_b, qr_t, iq);
      k_rope(iq, hp.idx_heads, hp.idx_dim, RD, w.freqs, pos, false, s);
      k_fq4(iq, hp.idx_heads * hp.idx_dim, false, s);
      k_gemv_bf16(w.idx_proj, hp.idx_heads, E, xin, iw, partial, s);
      const float* keys = S[k.kv_from].idx_k;
      k_index_scores(iq, hp.idx_heads, hp.idx_dim, keys, n_comp, iw, 1.0f / sqrtf((float)hp.idx_dim) / sqrtf((float)hp.idx_heads),
                     iscore, s);
      // two-level top-k beyond cand_topk_blocks blocks: the source picks candidate blocks from its
      // own scores, later indexers score only inside them (a no-op below 16,384 positions)
      if (k.cand_source) {
        cand_cur[t] = nullptr;
        if ((n_comp + hp.cand_block - 1) / hp.cand_block > hp.cand_topk_blocks) {
          uint8_t* keep = cand_keep + (size_t)t * (opt.max_ctx / hp.cand_block + 1);
          k_candidate_blocks(iscore, n_comp, hp.cand_block, hp.cand_topk_blocks, cand_bscore, cand_bidx, keep, s);
          cand_cur[t] = keep;
        }
      } else if (k.uses_cands && cand_cur[t]) {
        k_mask_candidates(iscore, n_comp, hp.cand_block, cand_cur[t], s);
      }
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
}

void Runtime::moe(int L, float* out, int M, Taps* taps) {
  moe_launch(L, M);
  moe_finish(L, out, M, taps);
}

// the GPU part: router, dispatch, staging to the host, the dies' experts, the shared expert
void Runtime::moe_launch(int L, int M) {
  const LayerW& w = W[L];
  const int E = hp.n_embd, F = hp.n_ff_exp, KE = hp.n_expert_used;
  const float* xin = d_moe.x;  // [M][E]
  mark("ffn.router");
  k_gemv_f32_m(w.gate_inp, hp.n_expert, E, xin, M, router_logits, partial, s);
  k_router(router_logits, w.gate_bias, hp.n_expert, KE, hp.expert_weights_norm, hp.expert_weights_scale, rids, rw, s, M,
           kMoeIdStride);
  if (ra_k >= 0 && ra_logits && !in_prefill && M <= 8) {  // the lookahead study: this layer's x through later routers
    ra_m = M;
    ra_tl[L] = Clock::now();
    ra_freq[L] = cache ? cache->top_nonresident(L, 6) : std::vector<int>();
    for (int e = 0; e < hp.n_expert; e++) {  // the table in force for this forward
      ra_res[L][e] = !h_table.empty() && h_table[(size_t)L * hp.n_expert + e] >= 0;
      ra_cnt[L][e] = cache ? (uint16_t)std::min(65535, cache->uses(L, e)) : 0;
    }
    CHECK(hipMemcpyAsync(ra_act + (size_t)L * 8 * kMoeIdStride, rids, (size_t)M * kMoeIdStride * 4, hipMemcpyDeviceToDevice, s));
    for (int j = 0; j <= ra_k && L + j < hp.n_layer; j++) {
      const int Lt = L + j;
      if (!m.layers[Lt].loaded || !W[Lt].gate_inp) continue;
      k_gemv_f32_m(W[Lt].gate_inp, hp.n_expert, E, xin, M,
                   ra_logits + ((size_t)Lt * (ra_k + 1) + j) * 8 * hp.n_expert, partial, s);
      ra_valid[(size_t)Lt * (ra_k + 1) + j] = 1;
    }
  }
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
  const size_t lists_bytes = (size_t)M * kMoeListStride * sizeof(MoeList);  // this forward's tokens only
  CHECK(hipMemcpyAsync(h_moe.lists, d_moe.lists, lists_bytes, hipMemcpyDeviceToHost, s));
  CHECK(hipMemcpyAsync(h_moe.x, d_moe.x, (size_t)M * E * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipEventRecord(ev_staged, s));
  if (!trace_path.empty() && !in_prefill) CHECK(hipEventRecord(tr_staged, s));
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
      hipEvent_t tr_a = nullptr;
      if (!trace_path.empty()) CHECK(hipEventRecord(tr_a = trace_ev(i), s_local));
      if (opt.canon)
        fp4_moe_canon_rows(d.pool.view, xin, sel(d_moe.lists, i), KE, M, hp.swiglu_clamp[L], place_rows(i), d.scratch, s_local);
      else
        fp4_moe_chunk(d.pool.view, xin, sel(d_moe.lists, i), KE, M, hp.swiglu_clamp[L], d.y_home, d.scratch, s_local);
      CHECK(hipEventRecord(ev_local, s_local));
      if (tr_a) {
        hipEvent_t tr_b = trace_ev(i);
        CHECK(hipEventRecord(tr_b, s_local));
        die_trace[i].spans.push_back({L, tr_a, tr_b, ""});
      }
      continue;
    }
    // another die: starts when the block is in host memory, no host involvement
    CHECK(hipSetDevice(d.dev));
    CHECK(hipStreamWaitEvent(d.st, ev_staged, 0));
    hipEvent_t tr_a = nullptr;
    if (!trace_path.empty()) CHECK(hipEventRecord(tr_a = trace_ev(i), d.st));
    CHECK(hipMemcpyAsync(d.d_in, h_moe.lists, lists_bytes, hipMemcpyHostToDevice, d.st));
    CHECK(hipMemcpyAsync(d.d_in + moe_lists_bytes, h_moe.x, (size_t)M * E * 4, hipMemcpyHostToDevice, d.st));
    const float* dx = (const float*)(d.d_in + moe_lists_bytes);
    if (opt.canon) {  // the copy back waits for the counts (below)
      fp4_moe_canon_rows(d.pool.view, dx, sel((const MoeList*)d.d_in, i), KE, M, hp.swiglu_clamp[L], d.d_y, d.scratch, d.st);
    } else {
      fp4_moe_chunk(d.pool.view, dx, sel((const MoeList*)d.d_in, i), KE, M, hp.swiglu_clamp[L], d.d_y, d.scratch, d.st);
      CHECK(hipMemcpyAsync(d.h_y, d.d_y, (size_t)M * E * 4, hipMemcpyDeviceToHost, d.st));
      CHECK(hipEventRecord(d.done, d.st));
    }
    if (tr_a) {  // copy in and the kernel (fast: and the copy back)
      hipEvent_t tr_b = trace_ev(i);
      CHECK(hipEventRecord(tr_b, d.st));
      die_trace[i].spans.push_back({L, tr_a, tr_b, ""});
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
}

// the host part: the CPU's experts, the dies' results back, the sum
void Runtime::moe_finish(int L, float* out, int M, Taps* taps) {
  const int E = hp.n_embd, KE = hp.n_expert_used, n_dies = (int)dies.size();
  auto place_rows = [&](int p) { return canon_rows + (size_t)p * kChunk * KE * E; };  // canon
  bool local = false;
  for (const Die& d : dies) local |= !d.st;
  // the host: the lists (counts for the statistics) and the CPU's misses
  auto t0 = Clock::now();
  CHECK(hipEventSynchronize(ev_staged));
  last.wait_ms += ms_since(t0);
  trace_host("wait for the staged block", t0, Clock::now(), 1);
  if (!trace_path.empty() && !in_prefill && !marks.empty()) {  // die 0's clock vs the host's (above)
    CHECK(hipEventSynchronize(tr_staged));
    const double now = std::chrono::duration<double, std::micro>(Clock::now() - trace_epoch).count();
    float el;
    CHECK(hipEventElapsedTime(&el, marks[0].second, tr_staged));
    step_anchor_us = std::min(step_anchor_us, now - el * 1000.0);
  }
  // each die's experts this layer (slots: one per expert), for its span; generation only (the
  // prompt's two halves interleave, so the latest span may be the other half's)
  if (!trace_path.empty() && !in_prefill) {
    for (int i = 0; i < n_dies; i++) {
      if (die_trace[i].spans.empty() || die_trace[i].spans.back().L != L) continue;
      std::set<int> slots;
      int pairs = 0;
      for (int t = 0; t < M; t++) {
        const MoeList& l = h_moe.lists[(size_t)t * kMoeListStride + i];
        for (int j = 0; j < l.count; j++) slots.insert(l.ids[j]);
        pairs += l.count;
      }
      die_trace[i].spans.back().args = "\"distinct experts\":" + std::to_string(slots.size()) +
                                       ",\"pairs\":" + std::to_string(pairs) + ",\"tokens\":" + std::to_string(M);
    }
  }
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
  if (cache) cache->observe(L, h_moe.lists, M, n_dies);
  if (opt.canon)
    for (int i = 0; i < n_dies; i++) {
      Die& d = dies[i];
      if (!d.st) continue;
      CHECK(hipSetDevice(d.dev));
      if (n_rows[i]) CHECK(hipMemcpyAsync(d.h_y, d.d_y, (size_t)n_rows[i] * E * 4, hipMemcpyDeviceToHost, d.st));
      CHECK(hipEventRecord(d.done, d.st));
      CHECK(hipSetDevice(opt.gpu));
    }
  // the host: the CPU's misses; a chunk at once (each expert's rows read once for all its
  // tokens, the same bits as token by token), a single token directly
  std::vector<char> has_cpu(M, 0);
  bool any_cpu = false;
  int cpu_row = 0;
  const size_t mk0 = marks.size();  // profile: the next mark is this layer's wait for the CPU
  const auto t_cpu0 = Clock::now();
  if (M > 1) {
    std::vector<CpuMoeKernel::CpuRoute> routes(M);
    for (int t = 0; t < M; t++) {
      const MoeList& lc = h_moe.lists[(size_t)t * kMoeListStride + n_dies];
      routes[t] = {lc.ids, lc.w, lc.count};
      has_cpu[t] = lc.count > 0;
      any_cpu |= lc.count > 0;
      cpu_row += lc.count;
    }
    if (any_cpu) {
      mark("ffn.wait_for_cpu_experts");
      auto t1c = Clock::now();
      const Layer& hl = m.layers[L];
      const CpuExperts ce{hl.gate_exps, hl.up_exps, hl.down_exps, hp.swiglu_clamp[L]};
      cpu_moe->forward_batch(ce, h_moe.x, M, routes.data(), opt.canon, h_yr);
      last.cpu_moe_ms += ms_since(t1c);
    }
  }
  for (int t = 0; t < M && M == 1; t++) {
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
  if (profile && !in_prefill) {
    std::set<int> distinct;
    int pairs = 0;
    for (int t = 0; t < M; t++) {
      const MoeList& lc = h_moe.lists[(size_t)t * kMoeListStride + n_dies];
      for (int j = 0; j < lc.count; j++) distinct.insert(lc.ids[j]);
      pairs += lc.count;
    }
    miss_recs.push_back({M, (int)distinct.size(), pairs, any_cpu ? ms_since(t_cpu0) : 0.0, any_cpu ? (int)mk0 : -1});
    if (cache) {  // the prefetch prediction (README step 52)
      if (prev_top.empty()) prev_top.assign(hp.n_layer, -1);
      const std::vector<int> f3 = cache->top_nonresident(L, 3);
      PredCell& pc = pred_hist[M == 1 ? 0 : M <= 3 ? 1 : M <= 6 ? 2 : 3];
      pc.layers++;
      if (distinct.empty()) {
        pc.wasted++;
      } else {
        pc.with_cpu++;
        pc.freq1 += !f3.empty() && distinct.count(f3[0]);
        bool any3 = false;
        for (int e : f3) any3 |= distinct.count(e) > 0;
        pc.freq3 += any3;
        pc.prev1 += prev_top[L] >= 0 && distinct.count(prev_top[L]);
      }
      std::map<int, int> uses;  // this forward's most used CPU expert of the layer, for the next
      for (int t = 0; t < M; t++) {
        const MoeList& lc = h_moe.lists[(size_t)t * kMoeListStride + n_dies];
        for (int j = 0; j < lc.count; j++) uses[lc.ids[j]]++;
      }
      int best = -1, bn = 0;
      for (auto& [e, n] : uses)
        if (n > bn) best = e, bn = n;
      prev_top[L] = best;
    }
    if (any_cpu)
      trace_host(("CPU experts L" + std::to_string(L)).c_str(), t_cpu0, Clock::now(), 1,
                 "\"distinct experts\":" + std::to_string(distinct.size()) + ",\"pairs\":" + std::to_string(pairs) +
                 ",\"tokens\":" + std::to_string(M));
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
  block_pre(L, taps, M);
  block_post(L, taps, M);
}

// up to the routed experts started (GPU work only: the host does not wait)
void Runtime::block_pre(int L, Taps* taps, int M) {
  cur_layer = L;
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
      for (int t = 0; t < M; t++)  // in a chunk, per position (verify: only kept ones seed the draft)
        k_hc_pre(ht(t), dspark->quarter, E, (chunked ? dspark->hid_chunk + t * nh : dspark->main_hidden) + (size_t)d * E, s);
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
  moe_launch(L, M);
}

// from the CPU's experts (the host computes them) to the block output
void Runtime::block_post(int L, Taps* taps, int M) {
  cur_layer = L;
  const int E = hp.n_embd, HC = hp.hc_mult;
  moe_finish(L, y, M, taps);
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
  std::fill(cand_cur.begin(), cand_cur.end(), nullptr);
  last = Times{};
  marks.clear();
  cur_layer = -1;
  trace_step_begin();
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
  if (cache) cache->step(1);  // a step boundary: no GPU work reads a pool now
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
  if (tokens.empty()) return {};
  in_prefill = true;
  // chunks through forward_chunk: the dense projections are bit-identical per token at any M
  // (README step 20), the attention part runs token by token, so the window ring and the
  // compressor state end up exactly as after decode() per token
  Times sum;
  std::map<std::string, double> groups;
  int chunks = 0;
  for (size_t i = 0; i + 1 < tokens.size();) {
    const size_t n = std::min<size_t>(std::clamp(opt.prefill_chunk, 1, kChunk), tokens.size() - 1 - i);
    const int p0 = n_past_;
    forward_chunk(std::vector<int32_t>(tokens.begin() + i, tokens.begin() + i + n), false);
    if (dspark) {  // every position of the chunk into the draft's rings, as decode() does per token
      const size_t nh = dspark->target_layers.size() * hp.n_embd;
      for (size_t j = 0; j < n; j++) dspark->seed(p0 + (int)j, dspark->hid_chunk + j * nh);
      CHECK(hipMemcpyAsync(dspark->main_hidden, dspark->hid_chunk + (n - 1) * nh, nh * 4, hipMemcpyDeviceToDevice, s));
    }
    i += n;
    if (profile) {  // the prompt's chunks summed: where prompt processing goes
      sum.total_ms += last.total_ms, sum.cpu_moe_ms += last.cpu_moe_ms, sum.wait_ms += last.wait_ms;
      sum.gpu_experts += last.gpu_experts, sum.cpu_experts += last.cpu_experts;
      for (auto& [g, ms] : last_groups) groups[g] += ms;
      chunks++;
    }
  }
  if (profile && chunks) {
    fprintf(stderr, "# prompt profile: %d chunks of up to %d, %.0f ms (cpu experts %.0f, waiting for the GPU %.0f; expert-tokens gpu %d cpu %d); GPU timeline:",
            chunks, std::clamp(opt.prefill_chunk, 1, kChunk), sum.total_ms, sum.cpu_moe_ms, sum.wait_ms, sum.gpu_experts, sum.cpu_experts);
    for (auto& [g, ms] : groups) fprintf(stderr, " %s %.0f", g.c_str(), ms);
    fprintf(stderr, "\n");
  }
  std::vector<float> logits = decode(tokens.back(), true);
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
  cur_layer = -1;
  if (dspark) dspark->seed(pos);  // this position's K/V into the draft's window rings
  if (!want_logits) {
    end_step();
    last.total_ms = ms_since(t0);
    trace_host("decode (no logits)", t0, Clock::now(), 3);
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
  if (profile) fold_misses();
  trace_step_end();
  if (ra_logits) ra_step_end();
  last.total_ms = ms_since(t0);
  trace_host("decode", t0, Clock::now(), 3);
  return logits;
}
