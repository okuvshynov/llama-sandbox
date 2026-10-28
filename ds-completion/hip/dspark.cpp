#include "dspark.h"

#include "fakequant.h"
#include "kernels.h"
#include "rt.h"

#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <cmath>
#include <cstring>
#include <stdexcept>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

static float bf16f(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

void* DSpark::upload(const void* src, size_t n) {
  void* p;
  CHECK(hipMalloc(&p, n));
  if (src) CHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice));
  else CHECK(hipMemset(p, 0, n));
  owned.push_back(p);
  return p;
}

DSpark::Fp8 DSpark::fp8(const void* bf16, int N, int K, const std::string& name) {
  std::vector<uint8_t> q((size_t)N * K), sc((size_t)(N / 32) * (K / 32));
  if (fp8_repack_bf16((const uint16_t*)bf16, N, K, q.data(), sc.data()) != 0)
    throw std::runtime_error("dspark: " + name + " is not exact fp8 in 32x32 blocks");
  Fp8 w;
  w.N = N;
  w.K = K;
  w.q = (uint8_t*)upload(q.data(), q.size());
  w.s = (uint8_t*)upload(sc.data(), sc.size());
  return w;
}

DSpark::DSpark(Runtime& r, const std::string& path) : rt(r) {
  const Hparams& hp = rt.hp;
  const int E = hp.n_embd, HC = hp.hc_mult, HD = hp.head_dim, NH = hp.n_head, F = hp.n_ff_exp;
  if (!fp8_codec_selftest()) throw std::runtime_error("dspark: e4m3 codec mismatch");
  ggml_context* ctx = nullptr;
  gguf_init_params gp = {true, &ctx};
  gguf_context* g = gguf_init_from_file(path.c_str(), gp);
  if (!g) throw std::runtime_error("cannot read " + path);
  auto key_i = [&](const char* k, int def) {
    const int64_t id = gguf_find_key(g, k);
    return id < 0 ? def : (int)gguf_get_val_u32(g, id);
  };
  if (std::string(gguf_get_val_str(g, gguf_find_key(g, "general.architecture"))) != "dflash")
    throw std::runtime_error(path + ": not a dflash (DSpark) GGUF");
  block_size = key_i("dflash.block_size", 5);
  n_expert = key_i("dflash.expert_count", 128);
  n_used = key_i("dflash.expert_used_count", 3);
  {
    const int64_t id = gguf_find_key(g, "dflash.expert_weights_scale");
    if (id >= 0) route_scale = gguf_get_val_f32(g, id);
    const int64_t idn = gguf_find_key(g, "dflash.expert_weights_norm");
    if (idn >= 0) route_norm = gguf_get_val_bool(g, idn);
    const int64_t idt = gguf_find_key(g, "dflash.target_layers");
    if (idt < 0) throw std::runtime_error("dspark: no dflash.target_layers");
    for (size_t i = 0; i < gguf_get_arr_n(g, idt); i++) target_layers.push_back(((const int32_t*)gguf_get_arr_data(g, idt))[i]);
    const int64_t idm = gguf_find_key(g, "tokenizer.ggml.mask_token_id");
    if (idm >= 0) noise_id = (int32_t)gguf_get_val_u32(g, idm);
  }
  const int n_layer = key_i("dflash.block_count", 3);
  std::vector<float> clamp(n_layer, 10.0f), clamp_sh(n_layer, 10.0f);
  for (auto [k, v] : {std::make_pair("dflash.swiglu_clamp_exp", &clamp), std::make_pair("dflash.swiglu_clamp_shexp", &clamp_sh)}) {
    const int64_t id = gguf_find_key(g, k);
    if (id >= 0)
      for (int i = 0; i < n_layer && i < (int)gguf_get_arr_n(g, id); i++) (*v)[i] = ((const float*)gguf_get_arr_data(g, id))[i];
  }

  // tensors: mmap the file, data pointers per tensor
  const int fd = open(path.c_str(), O_RDONLY);
  struct stat st;
  fstat(fd, &st);
  void* map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  const size_t data0 = gguf_get_data_offset(g);
  auto T = [&](const std::string& name, ggml_type type, std::vector<int64_t> ne) -> const void* {
    const int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) throw std::runtime_error("dspark: missing tensor " + name);
    const ggml_tensor* t = ggml_get_tensor(ctx, name.c_str());
    if (t->type != type) throw std::runtime_error("dspark: " + name + " has type " + ggml_type_name(t->type));
    ne.resize(4, 1);
    for (int d = 0; d < 4; d++)
      if (t->ne[d] != ne[d]) throw std::runtime_error("dspark: " + name + " has an unexpected shape");
    return (const char*)map + data0 + gguf_get_tensor_offset(g, id);
  };
  auto f32 = [&](const std::string& name, std::vector<int64_t> ne) {
    int64_t n = 1;
    for (auto v : ne) n *= v;
    return (const float*)upload(T(name, GGML_TYPE_F32, ne), n * 4);
  };

  CHECK(hipSetDevice(rt.opt.gpu));
  hipStream_t up_s;
  CHECK(hipStreamCreate(&up_s));
  L.resize(n_layer);
  size_t bytes = 0;
  for (int i = 0; i < n_layer; i++) {
    LayerW& w = L[i];
    const std::string p = "blk." + std::to_string(i) + ".";
    auto f8 = [&](const char* n, int N, int K) { return fp8(T(p + n + ".weight", GGML_TYPE_BF16, {K, N}), N, K, p + n); };
    w.q_a = f8("attn_q_a", hp.q_lora, E);
    w.q_b = f8("attn_q_b", NH * HD, hp.q_lora);
    w.kv = f8("attn_kv", HD, E);
    w.out_a = f8("attn_output_a", hp.o_groups * hp.o_lora, NH * HD / hp.o_groups);
    w.out_b = f8("attn_output_b", E, hp.o_groups * hp.o_lora);
    w.gate_sh = f8("ffn_gate_shexp", F, E);
    w.up_sh = f8("ffn_up_shexp", F, E);
    w.down_sh = f8("ffn_down_shexp", E, F);
    w.attn_norm = f32(p + "attn_norm.weight", {E});
    w.q_a_norm = f32(p + "attn_q_a_norm.weight", {hp.q_lora});
    w.kv_norm = f32(p + "attn_kv_a_norm.weight", {HD});
    w.sinks = f32(p + "attn_sinks.weight", {NH});
    w.ffn_norm = f32(p + "ffn_norm.weight", {E});
    w.gate_inp = f32(p + "ffn_gate_inp.weight", {E, n_expert});
    w.gate_bias = f32(p + "exp_probs_b.bias", {n_expert});
    const int mix = (2 + HC) * HC;
    w.hc_attn_fn = f32(p + "hc_attn_fn.weight", {E * HC, mix});
    w.hc_attn_base = f32(p + "hc_attn_base.weight", {mix});
    w.hc_attn_scale = f32(p + "hc_attn_scale.weight", {3});
    w.hc_ffn_fn = f32(p + "hc_ffn_fn.weight", {E * HC, mix});
    w.hc_ffn_base = f32(p + "hc_ffn_base.weight", {mix});
    w.hc_ffn_scale = f32(p + "hc_ffn_scale.weight", {3});
    w.ex = {fp4_upload(T(p + "ffn_gate_exps.weight", GGML_TYPE_MXFP4, {E, F, n_expert}), E, F, n_expert, up_s),
            fp4_upload(T(p + "ffn_up_exps.weight", GGML_TYPE_MXFP4, {E, F, n_expert}), E, F, n_expert, up_s),
            fp4_upload(T(p + "ffn_down_exps.weight", GGML_TYPE_MXFP4, {F, E, n_expert}), F, E, n_expert, up_s)};
    bytes += (size_t)3 * n_expert * F * E / 32 * 17;
    w.clamp = clamp[i];
    w.clamp_sh = clamp_sh[i];
  }
  CHECK(hipStreamDestroy(up_s));
  const int n_t = (int)target_layers.size();
  fc = fp8(T("fc.weight", GGML_TYPE_BF16, {(int64_t)E * n_t, E}), E, E * n_t, "fc");
  enc_norm = f32("enc.output_norm.weight", {E});
  out_norm = f32("output_norm.weight", {E});
  {
    const ggml_tensor* t = ggml_get_tensor(ctx, "markov_w1.weight");
    if (!t) throw std::runtime_error("dspark: no markov_w1");
    rank = (int)t->ne[0];
  }
  const int64_t V = hp.n_vocab;
  const uint16_t* w1 = (const uint16_t*)T("markov_w1.weight", GGML_TYPE_BF16, {rank, V});
  markov_w1.assign(w1, w1 + (size_t)V * rank);
  markov_w2 = (const uint16_t*)upload(T("markov_w2.weight", GGML_TYPE_BF16, {rank, V}), (size_t)V * rank * 2);
  const float* cp = (const float*)T("conf_proj.weight", GGML_TYPE_F32, {E + rank, 1});
  conf_proj.assign(cp, cp + E + rank);
  munmap(map, st.st_size);
  gguf_free(g);
  ggml_free(ctx);

  // scratch and state
  auto dz = [&](size_t n) { return (float*)upload(nullptr, n * 4); };
  const int B = block_size;
  main_hidden = dz((size_t)n_t * E);
  hid_chunk = dz((size_t)kMaxChunk * n_t * E);
  {
    std::vector<float> qv(HC, 1.0f / HC);
    quarter = (const float*)upload(qv.data(), HC * 4);
  }
  main_x = dz(E);
  hblk = dz((size_t)B * HC * E);
  h2 = dz((size_t)B * HC * E);
  pre = dz((size_t)B * HC);
  a_pre = dz((size_t)B * HC);
  a_post = dz((size_t)B * HC);
  a_comb = dz((size_t)B * HC * HC);
  f_post = dz((size_t)B * HC);
  f_comb = dz((size_t)B * HC * HC);
  mixes = dz((size_t)B * (2 + HC) * HC);
  xa = dz((size_t)B * E);
  t1 = dz((size_t)B * hp.q_lora);
  qr = dz((size_t)B * hp.q_lora);
  qblk = dz((size_t)B * NH * HD);
  kvv = dz((size_t)B * HD);
  kvblk = dz((size_t)B * HD);
  o = dz((size_t)B * NH * HD);
  oa = dz((size_t)B * hp.o_groups * hp.o_lora);
  y = dz((size_t)B * E);
  ysh = dz((size_t)B * E);
  yr = dz((size_t)B * E);
  gate = dz((size_t)B * F);
  up = dz((size_t)B * F);
  act = dz((size_t)B * F);
  rlog = dz((size_t)B * n_expert);
  rw = dz((size_t)B * 8);
  xfin = dz((size_t)B * E);
  xn = dz((size_t)B * E);
  logits = dz((size_t)B * V);
  mbias = dz(V);
  memb = dz((size_t)B * rank);
  rids = (int*)dz((size_t)B * 8);
  d_ids = (int*)dz(B + 1);
  markov_w1_dev = (const uint16_t*)upload(markov_w1.data(), markov_w1.size() * 2);
  {
    std::vector<int> v(B);
    for (int i = 0; i < B; i++) v[i] = i;
    iota5 = (int*)upload(v.data(), B * 4);
  }
  for (int i = 0; i < n_layer; i++) rings.push_back(dz((size_t)hp.window * HD));
  CHECK(hipMalloc(&moe_scratch, fp4_moe_scratch_bytes(E, F, n_used, B)));
  owned.push_back(moe_scratch);
  fprintf(stderr, "# dspark: %d layers, block %d, %d experts top-%d, targets", n_layer, block_size, n_expert, n_used);
  for (int t : target_layers) fprintf(stderr, " %d", t);
  fprintf(stderr, "; %.1f GB on GPU %d\n", (bytes + (size_t)n_layer * 165e6) / 1e9, rt.opt.gpu);
}

DSpark::~DSpark() {
  if (hidden_log) fclose(hidden_log);
  for (LayerW& w : L) {
    fp4_free(w.ex.gate);
    fp4_free(w.ex.up);
    fp4_free(w.ex.down);
  }
  for (void* p : owned) (void)hipFree(p);
}

int DSpark::target_index(int Lm) const {
  for (size_t i = 0; i < target_layers.size(); i++)
    if (target_layers[i] == Lm) return (int)i;
  return -1;
}

// main_x from this main position's target hiddens, then each draft layer's window K/V for it
void DSpark::seed(int pos, const float* hidden) {
  if (!hidden) hidden = main_hidden;
  const Hparams& hp = rt.hp;
  const int E = hp.n_embd, HD = hp.head_dim, RD = hp.rope_dim;
  hipStream_t s = rt.s;
  auto g8 = [&](const Fp8& w, Runtime::Codes c, float* out) { rt.gemv8(Runtime::Fp8W{w.q, w.s, w.N, w.K}, c, out); };
  if (hidden_log) {
    std::vector<float> v(target_layers.size() * E);
    CHECK(hipMemcpyAsync(v.data(), hidden, v.size() * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    fwrite(v.data(), 4, v.size(), hidden_log);
  }
  rt.quant(hidden, (int)target_layers.size() * E, rt.A);
  g8(fc, rt.A, main_x);
  k_rms_norm(main_x, enc_norm, main_x, E, 1, hp.rms_eps, s);
  rt.quant(main_x, E, rt.A);
  for (size_t l = 0; l < L.size(); l++) {
    g8(L[l].kv, rt.A, kvv);
    k_kv_finish(kvv, L[l].kv_norm, HD, RD, rt.freqs_buf, pos, hp.rms_eps, rings[l] + (size_t)(pos % hp.window) * HD, s);
  }
}

// The block's positions go through each layer together: projections, mixes, experts and the
// head as multi-token calls (one weight read; every kernel bit-identical per row to its
// single-row form, so the drafts are the same as one position at a time). Attention per
// position, after every position's K/V is written. The Markov step runs on the GPU (row gather,
// bias GEMV, argmax), left to right; one host sync at the end.
DraftResult DSpark::draft(int32_t anchor, int pos, bool keep_logits) {
  const Hparams& hp = rt.hp;
  const int E = hp.n_embd, HC = hp.hc_mult, HD = hp.head_dim, NH = hp.n_head, RD = hp.rope_dim, F = hp.n_ff_exp;
  const int B = block_size, MIX = (2 + HC) * HC;
  const int64_t V = hp.n_vocab;
  const float eps = hp.rms_eps;
  hipStream_t s = rt.s;
  const float* freqs = rt.freqs_buf;  // window-only layers: rope_base, no YaRN
  auto g8 = [&](const Fp8& w, Runtime::Codes c, float* out, int rpg, int M) {
    rt.gemv8(Runtime::Fp8W{w.q, w.s, w.N, w.K}, c, out, rpg, M);
  };
  // DS_DRAFT_PROFILE=1: GPU time per part of the draft, summed over the run, printed at exit
  static const bool prof = getenv("DS_DRAFT_PROFILE") != nullptr;
  static std::map<std::string, double> prof_ms;
  static int prof_n = 0;
  static std::vector<std::pair<std::string, hipEvent_t>> ev;
  size_t ev_i = 0;
  auto mark = [&](const char* name) {
    if (!prof) return;
    if (ev_i == ev.size()) {
      hipEvent_t e;
      CHECK(hipEventCreate(&e));
      ev.emplace_back(name, e);
    }
    ev[ev_i].first = name;
    CHECK(hipEventRecord(ev[ev_i].second, s));
    ev_i++;
  };
  mark("embed");
  // the block's input: [anchor, noise, ...] embedded, hc copies, identity pre-mix
  {
    std::vector<float> emb((size_t)B * E), one_hot((size_t)B * HC, 0.0f);
    std::vector<int32_t> ids(B + 1, noise_id);
    ids[0] = anchor;
    for (int j = 0; j < B; j++) {
      const ggml_bf16_t* row = (const ggml_bf16_t*)((const char*)rt.m.tok_embd->data + (size_t)ids[j] * rt.m.tok_embd->nb[1]);
      for (int i = 0; i < E; i++) emb[(size_t)j * E + i] = ggml_bf16_to_fp32(row[i]);
      one_hot[(size_t)j * HC] = 1.0f;
    }
    CHECK(hipMemcpyAsync(xa, emb.data(), emb.size() * 4, hipMemcpyHostToDevice, s));
    CHECK(hipMemcpyAsync(pre, one_hot.data(), one_hot.size() * 4, hipMemcpyHostToDevice, s));
    CHECK(hipMemcpyAsync(d_ids, ids.data(), 4, hipMemcpyHostToDevice, s));  // the anchor; the rest are drafted
    for (int j = 0; j < B; j++) k_repeat(xa + (size_t)j * E, E, HC, hblk + (size_t)j * HC * E, s);
  }
  std::vector<float> dbg;  // debug_log: [layer][attn_in, attn_out, ffn_in, ffn_out][block][E]
  auto keep = [&](size_t l, int what, const float* d) {
    if (!debug_log) return;
    dbg.resize(L.size() * 4 * B * E);
    CHECK(hipMemcpyAsync(dbg.data() + (l * 4 + what) * B * E, d, (size_t)B * E * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
  };

  for (size_t l = 0; l < L.size(); l++) {
    const LayerW& w = L[l];
    mark("hc");
    // attention: mixes, then q and K/V for every block position (it attends to itself both ways)
    k_gemv_f32_m(w.hc_attn_fn, MIX, E * HC, hblk, B, mixes, rt.partial, s);
    k_hc_split(hblk, E * HC, mixes, w.hc_attn_scale, w.hc_attn_base, eps, hp.hc_eps, hp.hc_iters, a_pre, a_post, a_comb, s, B);
    k_hc_pre_norm(hblk, pre, w.attn_norm, E, eps, xa, s, B);
    keep(l, 0, xa);
    mark("attn.proj");
    rt.quant(xa, E, rt.A, B);
    g8(w.q_a, rt.A, t1, 0, B);
    k_rms_norm(t1, w.q_a_norm, qr, hp.q_lora, B, eps, s);
    rt.quant(qr, hp.q_lora, rt.B, B);
    g8(w.q_b, rt.B, qblk, 0, B);
    g8(w.kv, rt.A, kvv, 0, B);
    for (int j = 0; j < B; j++) {
      const int pj = pos + 1 + j;
      k_rope(qblk + (size_t)j * NH * HD, NH, HD, RD, freqs, pj, false, s);
      k_kv_finish(kvv + (size_t)j * HD, w.kv_norm, HD, RD, freqs, pj, eps, kvblk + (size_t)j * HD, s);
    }
    mark("attn.core");
    for (int j = 0; j < B; j++) {
      // window ring (main positions <= pos) + all block positions
      float* oj = o + (size_t)j * NH * HD;
      k_attn_decode(qblk + (size_t)j * NH * HD, NH, HD, rings[l], hp.window, pos, kvblk, iota5, B, w.sinks,
                    1.0f / sqrtf((float)HD), oj, rt.att_scratch, s);
      k_rope(oj, NH, HD, RD, freqs, pos + 1 + j, true, s);
    }
    mark("attn.out");
    rt.quant(o, NH * HD, rt.B, B);
    g8(w.out_a, rt.B, oa, hp.o_lora, B);
    rt.quant(oa, hp.o_groups * hp.o_lora, rt.B, B);
    g8(w.out_b, rt.B, y, 0, B);
    keep(l, 1, y);
    k_hc_post(y, hblk, a_post, a_comb, E, h2, s, B);
    std::swap(hblk, h2);
    // FFN: routed (3 of 128, on this die) + shared, every block position at once
    mark("hc");
    k_gemv_f32_m(w.hc_ffn_fn, MIX, E * HC, hblk, B, mixes, rt.partial, s);
    // pre <- ffn_pre: the next layer's attention mix
    k_hc_split(hblk, E * HC, mixes, w.hc_ffn_scale, w.hc_ffn_base, eps, hp.hc_eps, hp.hc_iters, pre, f_post, f_comb, s, B);
    k_hc_pre_norm(hblk, a_pre, w.ffn_norm, E, eps, xa, s, B);
    keep(l, 2, xa);
    mark("ffn.routed");
    k_gemv_f32_m(w.gate_inp, n_expert, E, xa, B, rlog, rt.partial, s);
    for (int j = 0; j < B; j++)
      k_router(rlog + (size_t)j * n_expert, w.gate_bias, n_expert, n_used, route_norm, route_scale, rids + j * 8, rw + j * 8, s);
    fp4_moe_chunk(w.ex, xa, ExpertSel{rids, rw, nullptr, 8}, n_used, B, w.clamp, yr, moe_scratch, s);
    mark("ffn.shared");
    rt.quant(xa, E, rt.A, B);
    g8(w.gate_sh, rt.A, gate, 0, B);
    g8(w.up_sh, rt.A, up, 0, B);
    k_swiglu(gate, up, w.clamp_sh, B * F, act, s);
    rt.quant(act, F, rt.B, B);
    g8(w.down_sh, rt.B, ysh, 0, B);
    for (int j = 0; j < B; j++) {
      const float* parts[1] = {yr + (size_t)j * E};
      k_add_parts(ysh + (size_t)j * E, parts, 1, y + (size_t)j * E, E, s);
    }
    keep(l, 3, y);
    k_hc_post(y, hblk, f_post, f_comb, E, h2, s, B);
    std::swap(hblk, h2);
  }
  if (debug_log) {
    fwrite(dbg.data(), 4, dbg.size(), debug_log);
    fclose(debug_log);
    debug_log = nullptr;
  }
  mark("head");
  // head: collapse, the draft's norm, the main model's head (one read for the whole block)
  k_hc_pre(hblk, pre, E, xfin, s, B);
  k_rms_norm(xfin, out_norm, xn, E, B, eps, s);
  k_gemv_bf16_m(rt.out_w, (int)V, E, xn, B, logits, rt.partial, s);
  // Markov head, left to right on the GPU: bias from the previous token's rank-256 embedding,
  // then greedy
  mark("markov");
  for (int i = 0; i < B; i++) {
    float* lg = logits + (size_t)i * V;
    k_gather_bf16_row(markov_w1_dev, d_ids + i, rank, memb + (size_t)i * rank, s);
    k_gemv_bf16(markov_w2, (int)V, rank, memb + (size_t)i * rank, mbias, rt.partial, s);
    const float* parts[1] = {mbias};
    k_add_parts(lg, parts, 1, lg, (int)V, s);
    k_argmax(lg, (int)V, d_ids + i + 1, s);
  }
  mark("copy");
  DraftResult r;
  r.ids.resize(B + 1);
  std::vector<float> xf((size_t)B * E), emb((size_t)B * rank);
  CHECK(hipMemcpyAsync(r.ids.data(), d_ids, (B + 1) * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipMemcpyAsync(xf.data(), xfin, xf.size() * 4, hipMemcpyDeviceToHost, s));
  CHECK(hipMemcpyAsync(emb.data(), memb, emb.size() * 4, hipMemcpyDeviceToHost, s));
  if (keep_logits) {
    r.logits.resize((size_t)B * V);
    CHECK(hipMemcpyAsync(r.logits.data(), logits, r.logits.size() * 4, hipMemcpyDeviceToHost, s));
  }
  mark("end");
  CHECK(hipStreamSynchronize(s));
  if (prof) {
    for (size_t i = 0; i + 1 < ev_i; i++) {
      float ms;
      CHECK(hipEventElapsedTime(&ms, ev[i].second, ev[i + 1].second));
      prof_ms[ev[i].first] += ms;
    }
    if (++prof_n % 20 == 0) {
      fprintf(stderr, "\n# draft GPU time per call (%d calls):", prof_n);
      for (auto& [k, v] : prof_ms) fprintf(stderr, " %s %.2f", k.c_str(), v / prof_n);
      fprintf(stderr, "\n");
    }
  }
  r.conf.resize(B);
  for (int i = 0; i < B; i++) {
    float c = 0.0f;
    for (int k = 0; k < E; k++) c += conf_proj[k] * xf[(size_t)i * E + k];
    for (int k = 0; k < rank; k++) c += conf_proj[E + k] * emb[(size_t)i * rank + k];
    r.conf[i] = c;
  }
  return r;
}
