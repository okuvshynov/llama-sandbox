#include "forward.h"

#include "fakequant.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

static std::runtime_error err(const std::string& s) { return std::runtime_error(s); }

// ================================================================== host-side plan for one chunk
//
// Everything that depends only on positions, computed once per chunk of tokens at [p0, p0 + T).

// One compression ratio's bookkeeping (all layers with that ratio share it).
struct RatioPlan {
  int r = 0;
  int n_comp = 0;  // compressed positions that exist after this chunk: (p0 + T) / r
  int K = 0;       // positions each query attends to: min(index_topk, n_comp)
  // what query i may see: compressed position c is visible iff c < (pos_i + 1) / r
  std::vector<ggml_fp16_t> idx_mask;  // [n_comp, T] additive, for the indexer
  std::vector<float> vis_mask;        // [1, n_comp, T] additive, gathered at the selected positions
  // the compressor: groups of r tokens that complete inside this chunk
  int n_g = 0;
  std::vector<int32_t> gather;      // [r * n_g] member rows in concat(state [r], new [T])
  std::vector<int32_t> lat_pos;     // [n_g] a latent stands for its group's first token: g * r
  std::vector<int64_t> cache_rows;  // [n_g] g
  // the trailing incomplete group's members from this chunk go to state slots p % r
  int tail_src = 0;                 // first source row (in concat(state, new)) of those members
  std::vector<int64_t> tail_slots;  // [n_tail]
};

struct Plan {
  int p0 = 0, T = 0;
  std::vector<int32_t> tokens, pos;
  std::vector<float> win_mask;      // [W + T, T]: ring slots, then the chunk's own keys
  int win_src = 0;                  // first row (in concat(ring, new)) written back to the ring
  std::vector<int64_t> win_slots;   // [min(T, W)] ring slots for those rows
  std::map<int, RatioPlan> ratio;
  std::vector<std::vector<int32_t>> engram_ids;  // [engram slot][T * n_cols] rows of the table
  std::vector<float> one_hot_pre;   // [hc, T]: the first block's pre-mix (copy 0)
};

static Plan make_plan(const Hparams& hp, const State& st, const std::vector<int32_t>& toks, int max_ctx) {
  Plan p;
  p.p0 = st.n_past;
  p.T = (int)toks.size();
  p.tokens = toks;
  const int p0 = p.p0, T = p.T, W = hp.window;
  for (int i = 0; i < T; i++) p.pos.push_back(p0 + i);

  // window: ring slot s holds the newest position < p0 that is congruent to s (mod W)
  const float NEG = -INFINITY;
  p.win_mask.assign((size_t)(W + T) * T, NEG);
  for (int i = 0; i < T; i++) {
    const int q = p0 + i;
    for (int s = 0; s < W; s++) {
      if (s > p0 - 1) continue;  // never written
      const int ps = s + W * ((p0 - 1 - s) / W);
      if (ps > q - W) p.win_mask[(size_t)i * (W + T) + s] = 0.0f;
    }
    for (int j = 0; j < T; j++) {
      const int kp = p0 + j;
      if (kp <= q && kp > q - W) p.win_mask[(size_t)i * (W + T) + W + j] = 0.0f;
    }
  }
  const int n_write = std::min(T, W);
  p.win_src = W + T - n_write;
  for (int k = 0; k < n_write; k++) p.win_slots.push_back((p0 + T - n_write + k) % W);

  // compressed attention, per ratio in use
  for (int L = 0; L < hp.n_layer; L++) {
    const int r = hp.compress_ratios[L];
    if (r == 0 || p.ratio.count(r)) continue;
    RatioPlan& rp = p.ratio[r];
    rp.r = r;
    rp.n_comp = (p0 + T) / r;
    if (rp.n_comp > max_ctx / r) throw err("context longer than --max-ctx");
    if (r == hp.compress_ratios[hp.cand_source] && (rp.n_comp + hp.cand_block - 1) / hp.cand_block > hp.cand_topk_blocks)
      throw err("candidate-block selection (> 16k tokens) is not implemented yet");
    rp.K = std::min(hp.idx_topk, rp.n_comp);
    rp.idx_mask.assign((size_t)rp.n_comp * T, ggml_fp32_to_fp16(NEG));
    rp.vis_mask.assign((size_t)rp.n_comp * T, NEG);
    for (int i = 0; i < T; i++) {
      const int n_vis = (p0 + i + 1) / r;
      for (int c = 0; c < n_vis; c++) {
        rp.idx_mask[(size_t)i * rp.n_comp + c] = ggml_fp32_to_fp16(0.0f);
        rp.vis_mask[(size_t)i * rp.n_comp + c] = 0.0f;
      }
    }
    const int g_lo = p0 / r, g_hi = (p0 + T) / r - 1;
    rp.n_g = std::max(0, g_hi - g_lo + 1);
    for (int g = g_lo; g <= g_hi; g++) {
      for (int k = 0; k < r && r > 1; k++) {
        const int pp = g * r + k;
        rp.gather.push_back(pp < p0 ? pp % r : r + (pp - p0));
      }
      rp.lat_pos.push_back(g * r);
      rp.cache_rows.push_back(g);
    }
    const int tail0 = std::max((g_hi + 1) * r, p0);
    rp.tail_src = r + (tail0 - p0);
    for (int pp = tail0; pp < p0 + T && r > 1; pp++) rp.tail_slots.push_back(pp % r);
  }

  // engram hashes (engram.py NgramHashState.forward): the n-grams ending at each position
  std::vector<int32_t> hist = st.tokens;
  hist.insert(hist.end(), toks.begin(), toks.end());
  const int n_gram = hp.engram_ngram, n_head = hp.engram_heads, n_cols = hp.engram_cols();
  const uint64_t pad = (uint64_t)hp.engram_token_map[hp.engram_pad];
  for (size_t e = 0; e < hp.engram_layers.size(); e++) {
    std::vector<int32_t> ids((size_t)T * n_cols);
    const uint64_t* mult = hp.engram_mult.data() + e * n_gram;
    const uint64_t* primes = hp.engram_primes.data() + e * n_cols;
    const uint64_t* offs = hp.engram_offsets.data() + e * n_cols;
    for (int i = 0; i < T; i++) {
      const int q = p0 + i;
      uint64_t tok[8];
      for (int s = 0; s < n_gram; s++)  // once a look-back hits the start, every longer one is padded
        tok[s] = q - s < 0 ? pad : (uint64_t)hp.engram_token_map[hist[q - s]];
      uint64_t rolling = tok[0] * mult[0];
      for (int o = 1; o < n_gram; o++) {
        rolling ^= tok[o] * mult[o];
        for (int h = 0; h < n_head; h++) {
          const int col = (o - 1) * n_head + h;
          ids[(size_t)i * n_cols + col] = (int32_t)(rolling % primes[col] + offs[col]);
        }
      }
    }
    p.engram_ids.push_back(std::move(ids));
  }
  p.one_hot_pre.assign((size_t)hp.hc_mult * T, 0.0f);
  for (int i = 0; i < T; i++) p.one_hot_pre[(size_t)i * hp.hc_mult] = 1.0f;
  return p;
}

// ================================================================== one ggml graph

struct Engine::Graph {
  ggml_context* ctx = nullptr;     // op tensors (allocated by gallocr)
  ggml_context* ctx_in = nullptr;  // input tensors (allocated in their own buffer, then filled)
  ggml_cgraph* gf = nullptr;
  ggml_backend_buffer_t in_buf = nullptr;
  ggml_gallocr_t alloc = nullptr;
  std::vector<std::function<void()>> fill;               // input setters, run after allocation
  std::map<std::string, ggml_tensor*> inputs;             // memo: one tensor per input name
  std::vector<std::pair<std::string, ggml_tensor*>> taps;
  bool want_taps = false;
  std::map<int, ggml_tensor*> comp_cur, idx_k_cur;  // [src] cache tensor after this graph's writes
  std::map<int, ggml_tensor*> topk;                 // [index source] its selection this chunk

  Graph() {
    ggml_init_params p = {ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false), nullptr, true};
    ctx = ggml_init(p);
    ggml_init_params pi = {ggml_tensor_overhead() * 256, nullptr, true};
    ctx_in = ggml_init(pi);
    gf = ggml_new_graph_custom(ctx, 65536, false);
  }
  ~Graph() {
    if (alloc) ggml_gallocr_free(alloc);
    if (in_buf) ggml_backend_buffer_free(in_buf);
    ggml_free(ctx_in);
    ggml_free(ctx);
  }

  template <class T>
  ggml_tensor* input(const std::string& name, ggml_type type, std::vector<int64_t> ne, const std::vector<T>* data) {
    auto it = inputs.find(name);
    if (it != inputs.end()) return it->second;
    ne.resize(4, 1);
    ggml_tensor* t = ggml_new_tensor_4d(ctx_in, type, ne[0], ne[1], ne[2], ne[3]);
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
    if (ggml_nbytes(t) != data->size() * sizeof(T)) throw err("input size mismatch: " + name);
    fill.push_back([t, data] { ggml_backend_tensor_set(t, data->data(), 0, ggml_nbytes(t)); });
    inputs[name] = t;
    return t;
  }

  void tap(const std::string& name, ggml_tensor* t) {
    if (!want_taps) return;
    if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    taps.emplace_back(name, t);
  }
};

void Engine::compute(Graph& g) {
  if (!g.inputs.empty()) {
    g.in_buf = ggml_backend_alloc_ctx_tensors(g.ctx_in, cpu);
    for (auto& f : g.fill) f();
  }
  g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
  if (!ggml_gallocr_alloc_graph(g.alloc, g.gf)) throw err("graph allocation failed");
  ggml_backend_cpu_set_n_threads(cpu, opt.n_threads);
  if (ggml_backend_graph_compute(cpu, g.gf) != GGML_STATUS_SUCCESS) throw err("graph compute failed");
}

// ================================================================== engine

Engine::Engine(const Model& m_, Options o) : m(m_), hp(m_.hp), opt(o) {
  cpu = ggml_backend_cpu_init();
  if (!cpu) throw err("cannot init the CPU backend");
  // the sequence state, for the layers that are loaded
  ggml_init_params p = {ggml_tensor_overhead() * 512, nullptr, true};
  st.ctx = ggml_init(p);
  st.win.assign(hp.n_layer, nullptr);
  for (int L = 0; L < hp.n_layer; L++) {
    const Layer& l = m.layers[L];
    if (!l.loaded) continue;
    st.win[L] = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, hp.head_dim, hp.window);
    if (l.kind.kv_source) {
      const int r = l.kind.ratio;
      if (r > 1) {
        st.cmp_kv[L] = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, hp.head_dim, r);
        st.cmp_score[L] = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, hp.head_dim, r);
      }
      st.comp[L] = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, hp.head_dim, opt.max_ctx / r);
      st.idx_k[L] = ggml_new_tensor_2d(st.ctx, GGML_TYPE_F32, hp.idx_dim, opt.max_ctx / r);
    }
  }
  st.buf = ggml_backend_alloc_ctx_tensors(st.ctx, cpu);
  // zero: masked-out slots must hold finite values (0 * NaN would leak through P @ V)
  ggml_backend_buffer_clear(st.buf, 0);

  // compressor and indexer-key weights in f32 (a few MB per KV-source layer)
  ggml_init_params pf = {ggml_tensor_overhead() * 64, nullptr, true};
  f32_ctx = ggml_init(pf);
  std::vector<ggml_tensor*> src;
  for (const Layer& l : m.layers)
    if (l.loaded)
      for (ggml_tensor* t : {l.cmp_kv, l.cmp_gate, l.idx_k})
        if (t) src.push_back(t);
  for (ggml_tensor* t : src) f32w[t] = ggml_new_tensor_2d(f32_ctx, GGML_TYPE_F32, t->ne[0], t->ne[1]);
  if (!src.empty()) {
    f32_buf = ggml_backend_alloc_ctx_tensors(f32_ctx, cpu);
    for (ggml_tensor* t : src) {
      std::vector<float> v(ggml_nelements(t));
      ggml_get_type_traits(t->type)->to_float(t->data, v.data(), ggml_nelements(t));
      ggml_backend_tensor_set(f32w[t], v.data(), 0, v.size() * sizeof(float));
    }
  }
}

Engine::~Engine() {
  delete plan;
  if (f32_buf) ggml_backend_buffer_free(f32_buf);
  if (f32_ctx) ggml_free(f32_ctx);
  if (st.buf) ggml_backend_buffer_free(st.buf);
  if (st.ctx) ggml_free(st.ctx);
  ggml_backend_free(cpu);
}

void Engine::begin_chunk(const std::vector<int32_t>& tokens) {
  delete plan;
  topk_host.clear();
  plan = new Plan(make_plan(hp, st, tokens, opt.max_ctx));
}

void Engine::end_chunk() {
  st.n_past += plan->T;
  st.tokens.insert(st.tokens.end(), plan->tokens.begin(), plan->tokens.end());
  delete plan;
  plan = nullptr;
}

// ------------------------------------------------------------------ small helpers

ggml_tensor* Engine::fq8(Graph& g, ggml_tensor* x) {
  if (!opt.exact_quant) return x;
  return fq_fp8(g.ctx, ggml_is_contiguous(x) ? x : ggml_cont(g.ctx, x));
}

ggml_tensor* Engine::rms(Graph& g, ggml_tensor* x, ggml_tensor* w) {
  return ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, hp.rms_eps), w);
}

// RoPE on the last rope_dim dims (adjacent pairs, as the reference's view_as_complex).
// Compressed-attention layers rotate at compress_rope_base with YaRN; window-only layers at
// rope_base without. The reference applies no YaRN magnitude correction, so attn_factor cancels
// ggml's built-in 1 + 0.1 ln(1/freq_scale).
ggml_tensor* Engine::rope(Graph& g, ggml_tensor* x, ggml_tensor* pos, int L, int n_offs, bool inverse) {
  const bool comp = hp.compress_ratios[L] != 0;
  const float base = comp ? hp.compress_rope_base : hp.rope_base;
  const float fscale = comp ? 1.0f / hp.yarn_factor : 1.0f;
  const float ext = comp ? 1.0f : 0.0f;
  const float attn = comp ? 1.0f / (1.0f + 0.1f * logf(1.0f / fscale)) : 1.0f;
  const int n_orig = comp ? hp.yarn_orig_ctx : 0;
  const float bf = comp ? hp.yarn_beta_fast : 0.0f, bs = comp ? hp.yarn_beta_slow : 0.0f;
  auto f = inverse ? ggml_rope_ext_back : ggml_rope_ext;
  ggml_tensor* r = f(g.ctx, x, pos, nullptr, hp.rope_dim, GGML_ROPE_TYPE_NORMAL, n_orig, base, fscale, ext, attn, bf, bs);
  return ggml_rope_set_offset(r, n_offs);
}

// ------------------------------------------------------------------ engram (model.py Engram)

ggml_tensor* Engine::engram(Graph& g, int L, ggml_tensor* h) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  const int e = l.kind.engram, T = plan->T, E = hp.n_embd, HC = hp.hc_mult;
  const int n_cols = hp.engram_cols(), D = hp.engram_dim;
  ggml_tensor* ids = g.input("engram_ids." + std::to_string(e), GGML_TYPE_I32, {(int64_t)n_cols * T},
                             &plan->engram_ids[e]);
  // the table rows are fp8 x 2^k already, but act_quant may still move tiny blocks (amax < 1e-4)
  ggml_tensor* rows = ggml_reshape_2d(c, ggml_get_rows(c, m.engram_table, ids), (int64_t)n_cols * D, T);
  ggml_tensor* kv = ggml_mul_mat(c, l.engram_kv, fq8(g, rows));  // [E * (HC + 1), T]
  ggml_tensor* key = ggml_cont(c, ggml_view_3d(c, kv, E, HC, T, E * sizeof(float), kv->nb[1], 0));
  ggml_tensor* val = ggml_view_3d(c, kv, E, 1, T, E * sizeof(float), kv->nb[1], (size_t)HC * E * sizeof(float));
  // gate = sigmoid(signed sqrt(normalized <h, w * key>)), per token and hc copy
  ggml_tensor* dot = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, h, hp.rms_eps), ggml_rms_norm(c, key, hp.rms_eps)),
                              l.engram_gate);
  dot = ggml_scale(c, ggml_sum_rows(c, dot), 1.0f / sqrtf((float)E));  // [1, HC, T]
  ggml_tensor* mag = ggml_sqrt(c, ggml_clamp(c, ggml_abs(c, dot), 1e-6f, INFINITY));
  ggml_tensor* sign = ggml_scale_bias(c, ggml_step(c, ggml_neg(c, dot)), -2.0f, 1.0f);  // copysign: sign(0) = +1
  ggml_tensor* gate = ggml_sigmoid(c, ggml_mul(c, mag, sign));
  return ggml_add(c, h, ggml_mul(c, ggml_repeat_4d(c, ggml_cont(c, val), E, HC, T, 1), gate));
}

// ------------------------------------------------------------------ hyper-connections (Block.hc_mixes)

void Engine::hc_mixes(Graph& g, ggml_tensor* h, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                      ggml_tensor** pre, ggml_tensor** post, ggml_tensor** comb) {
  ggml_context* c = g.ctx;
  const int HC = hp.hc_mult, T = plan->T;
  // one RMS statistic over the whole flattened hc * n_embd stream per token
  ggml_tensor* flat = ggml_reshape_2d(c, h, (int64_t)hp.n_embd * HC, T);
  ggml_tensor* mixes = ggml_mul_mat(c, fn, ggml_rms_norm(c, flat, hp.rms_eps));  // [(2 + HC) * HC, T]
  auto part = [&](int k) { return ggml_view_2d(c, mixes, HC, T, mixes->nb[1], (size_t)k * HC * sizeof(float)); };
  auto s = [&](int k) { return ggml_view_1d(c, scale, 1, (size_t)k * sizeof(float)); };
  auto b = [&](int k) { return ggml_view_1d(c, base, HC, (size_t)k * HC * sizeof(float)); };
  *pre = ggml_scale_bias(c, ggml_sigmoid(c, ggml_add(c, ggml_mul(c, part(0), s(0)), b(0))), 1.0f, hp.hc_eps);
  *post = ggml_scale(c, ggml_sigmoid(c, ggml_add(c, ggml_mul(c, part(1), s(1)), b(1))), 2.0f);
  *comb = ggml_dsv4_hc_comb(c, mixes, scale, base, hp.hc_eps, hp.hc_iters);  // Sinkhorn, [dst, src, T]
}

// ------------------------------------------------------------------ attention (model.py Attention)

// Compressor + index keys for a KV-source layer; returns its compressed cache (after any writes).
ggml_tensor* Engine::compressor(Graph& g, int L, ggml_tensor* x) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  const int r = l.kind.ratio, HD = hp.head_dim, IDX = hp.idx_dim, RD = hp.rope_dim;
  const RatioPlan& rp = plan->ratio.at(r);
  const std::string sr = std::to_string(r);
  ggml_tensor* kvn = ggml_mul_mat(c, w32(l.cmp_kv), x);  // [HD, T] (fp32 linear, no activation quant)
  ggml_tensor* lat = nullptr;
  if (r == 1) {
    lat = kvn;  // a group of one token: no pooling
  } else {
    ggml_tensor* scn = ggml_mul_mat(c, w32(l.cmp_gate), x);
    ggml_tensor* src_kv = ggml_concat(c, st.cmp_kv.at(L), kvn, 1);  // [HD, r + T]
    ggml_tensor* src_sc = ggml_concat(c, st.cmp_score.at(L), scn, 1);
    if (rp.n_g > 0) {
      ggml_tensor* gi = g.input("cmp_gather." + sr, GGML_TYPE_I32, {(int64_t)rp.gather.size()}, &rp.gather);
      auto members = [&](ggml_tensor* src) {  // [HD, r * n_g] -> [r, HD, n_g]
        ggml_tensor* t = ggml_reshape_3d(c, ggml_get_rows(c, src, gi), HD, r, rp.n_g);
        return ggml_cont(c, ggml_permute(c, t, 1, 0, 2, 3));
      };
      // softmax over the group's members, per dimension, then the weighted sum
      ggml_tensor* pooled = ggml_sum_rows(c, ggml_mul(c, members(src_kv), ggml_soft_max(c, members(src_sc))));
      lat = ggml_reshape_2d(c, pooled, HD, rp.n_g);
    }
    if (!rp.tail_slots.empty()) {  // members of the still-incomplete group wait in the state
      ggml_tensor* slots = g.input("cmp_tail." + sr, GGML_TYPE_I64, {(int64_t)rp.tail_slots.size()}, &rp.tail_slots);
      const int n = (int)rp.tail_slots.size();
      for (auto [state, src] : {std::pair{st.cmp_kv.at(L), src_kv}, std::pair{st.cmp_score.at(L), src_sc}}) {
        ggml_tensor* rows = ggml_view_2d(c, src, HD, n, src->nb[1], (size_t)rp.tail_src * src->nb[1]);
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, state, rows, slots));
      }
    }
  }
  if (rp.n_g == 0) return st.comp.at(L);
  lat = rms(g, lat, l.cmp_norm);
  ggml_tensor* lpos = g.input("lat_pos." + sr, GGML_TYPE_I32, {rp.n_g}, &rp.lat_pos);
  ggml_tensor* rows = g.input("cache_rows." + sr, GGML_TYPE_I64, {rp.n_g}, &rp.cache_rows);
  // index keys come from the RoPE-free latent, so they are made first
  ggml_tensor* ik = rms(g, ggml_mul_mat(c, w32(l.idx_k), lat), l.idx_k_norm);  // [IDX, n_g]
  ik = rope(g, ggml_reshape_3d(c, ik, IDX, 1, rp.n_g), lpos, L, IDX - RD);
  ik = ggml_reshape_2d(c, ik, IDX, rp.n_g);
  if (opt.exact_quant) ik = fq_fp4(c, ik);
  g.idx_k_cur[L] = ggml_set_rows(c, st.idx_k.at(L), ik, rows);
  ggml_tensor* kl = rope(g, ggml_reshape_3d(c, lat, HD, 1, rp.n_g), lpos, L, HD - RD);
  kl = ggml_reshape_2d(c, kl, HD, rp.n_g);
  if (opt.exact_quant) kl = fq_fp4_e4(c, kl);
  g.comp_cur[L] = ggml_set_rows(c, st.comp.at(L), kl, rows);
  return g.comp_cur[L];
}

// The indexer: which compressed positions each query attends to. [K, T] i32 (unordered).
ggml_tensor* Engine::indexer(Graph& g, int L, ggml_tensor* x, ggml_tensor* qr) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  const int r = l.kind.ratio, IDX = hp.idx_dim, IH = hp.idx_heads, RD = hp.rope_dim, T = plan->T;
  const RatioPlan& rp = plan->ratio.at(r);
  if (rp.K == 0) return nullptr;
  ggml_tensor* pos = g.input("pos", GGML_TYPE_I32, {T}, &plan->pos);
  ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, l.idx_q_b, fq8(g, qr)), IDX, IH, T);
  q = rope(g, q, pos, L, IDX - RD);
  if (opt.exact_quant) q = fq_fp4(c, ggml_cont(c, q));
  ggml_tensor* w = ggml_scale(c, ggml_mul_mat(c, l.idx_proj, x), 1.0f / sqrtf((float)IDX) / sqrtf((float)IH));
  ggml_tensor* keys = g.idx_k_cur.count(l.kind.kv_from) ? g.idx_k_cur[l.kind.kv_from] : st.idx_k.at(l.kind.kv_from);
  keys = ggml_view_3d(c, keys, IDX, 1, rp.n_comp, keys->nb[1], keys->nb[1], 0);
  ggml_tensor* mask = g.input("idx_mask." + std::to_string(r), GGML_TYPE_F16, {rp.n_comp, T}, &rp.idx_mask);
  ggml_tensor* score = ggml_lightning_indexer(c, ggml_cont(c, q), keys, w, mask);  // [n_comp, T]
  // (layers after the candidate source would only score inside its 2048 candidate blocks;
  // below 16k tokens that is every block, see make_plan)
  ggml_tensor* idx = ggml_top_k(c, score, rp.K);
  g.tap("topk", idx);
  return idx;
}

ggml_tensor* Engine::attention(Graph& g, int L, ggml_tensor* x) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  const LayerKind& k = l.kind;
  const int T = plan->T, HD = hp.head_dim, NH = hp.n_head, W = hp.window, RD = hp.rope_dim;
  ggml_tensor* pos = g.input("pos", GGML_TYPE_I32, {T}, &plan->pos);

  ggml_tensor* qr = rms(g, ggml_mul_mat(c, l.q_a, fq8(g, x)), l.q_a_norm);  // [q_lora, T]
  ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, l.q_b, fq8(g, qr)), HD, NH, T);
  q = rope(g, q, pos, L, HD - RD);
  ggml_tensor* kv = rms(g, ggml_mul_mat(c, l.kv, fq8(g, x)), l.kv_norm);  // one shared KV head
  kv = rope(g, ggml_reshape_3d(c, kv, HD, 1, T), pos, L, HD - RD);
  kv = fq8(g, ggml_reshape_2d(c, kv, HD, T));

  // sliding window: the ring (positions before this chunk) followed by the chunk's own keys
  ggml_tensor* Kw = ggml_concat(c, st.win[L], kv, 1);  // [HD, W + T]
  {  // write back the newest W keys; taken from Kw so the write cannot run before the read
    ggml_tensor* slots = g.input("win_slots", GGML_TYPE_I64, {(int64_t)plan->win_slots.size()}, &plan->win_slots);
    ggml_tensor* rows = ggml_view_2d(c, Kw, HD, (int64_t)plan->win_slots.size(), Kw->nb[1], (size_t)plan->win_src * Kw->nb[1]);
    ggml_build_forward_expand(g.gf, ggml_set_rows(c, st.win[L], rows, slots));
  }
  ggml_tensor* scores = ggml_mul_mat(c, Kw, ggml_permute(c, q, 0, 2, 1, 3));  // [W + T, T, NH]
  ggml_tensor* mask = g.input("win_mask", GGML_TYPE_F32, {W + T, T}, &plan->win_mask);

  // compressed KV: the K positions the indexer picked for each query
  ggml_tensor* Kc = nullptr;
  int K = 0;
  if (k.ratio) {
    ggml_tensor* cache = k.kv_source ? compressor(g, L, x)
                         : g.comp_cur.count(k.kv_from) ? g.comp_cur[k.kv_from] : st.comp.at(k.kv_from);
    const RatioPlan& rp = plan->ratio.at(k.ratio);
    ggml_tensor* idx = nullptr;
    if (k.index_source) {
      idx = g.topk[L] = indexer(g, L, x, qr);
    } else if (g.topk.count(k.idx_from)) {
      idx = g.topk[k.idx_from];
    } else if (rp.K > 0) {  // the source ran in an earlier graph of this chunk
      const auto& v = topk_host.at(k.idx_from);
      idx = g.input("topk." + std::to_string(k.idx_from), GGML_TYPE_I32, {rp.K, T}, &v);
    }
    if (idx) {
      K = rp.K;
      ggml_tensor* flat = ggml_reshape_1d(c, idx, (int64_t)K * T);
      Kc = ggml_reshape_3d(c, ggml_get_rows(c, cache, flat), HD, K, T);  // [HD, K, T]
      ggml_tensor* sc = ggml_mul_mat(c, Kc, q);                          // [K, NH, T]
      scores = ggml_concat(c, scores, ggml_cont(c, ggml_permute(c, sc, 0, 2, 1, 3)), 0);
      ggml_tensor* vis = g.input("vis_mask." + std::to_string(k.ratio), GGML_TYPE_F32, {1, rp.n_comp, T}, &rp.vis_mask);
      ggml_tensor* mc = ggml_reshape_2d(c, ggml_get_rows(c, vis, idx), K, T);  // positions not yet visible: -inf
      mask = ggml_concat(c, mask, mc, 0);
    }
  }
  // one softmax over window + compressed keys, with a per-head sink logit in the denominator
  ggml_tensor* p = ggml_soft_max_ext(c, scores, mask, 1.0f / sqrtf((float)HD), 0.0f);  // [W + T + K, T, NH]
  ggml_soft_max_add_sinks(p, l.sinks);
  const int NW = W + T;
  ggml_tensor* pw = ggml_view_3d(c, p, NW, T, NH, p->nb[1], p->nb[2], 0);
  ggml_tensor* o = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, Kw)), pw);  // [HD, T, NH]
  o = ggml_permute(c, o, 0, 2, 1, 3);                                           // [HD, NH, T]
  if (Kc) {
    ggml_tensor* pc = ggml_view_3d(c, p, K, T, NH, p->nb[1], p->nb[2], (size_t)NW * sizeof(float));
    pc = ggml_cont(c, ggml_permute(c, pc, 0, 2, 1, 3));                         // [K, NH, T]
    o = ggml_add(c, o, ggml_mul_mat(c, ggml_cont(c, ggml_permute(c, Kc, 1, 0, 2, 3)), pc));
  }
  o = rope(g, ggml_cont(c, o), pos, L, HD - RD, /*inverse*/ true);  // take the query's rotation back out

  // grouped low-rank output: group j of 8 heads -> o_lora, then all groups -> n_embd
  const int G = hp.o_groups, GD = HD * NH / G;
  o = fq8(g, ggml_reshape_3d(c, o, GD, G, T));
  ggml_tensor* wa = ggml_view_3d(c, l.out_a, GD, hp.o_lora, G, l.out_a->nb[1], l.out_a->nb[1] * hp.o_lora, 0);
  ggml_tensor* oa = ggml_mul_mat(c, wa, ggml_permute(c, o, 0, 2, 1, 3));  // [o_lora, T, G]
  oa = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, oa, 0, 2, 1, 3)), (int64_t)hp.o_lora * G, T);
  return ggml_mul_mat(c, l.out_b, fq8(g, oa));
}

// ------------------------------------------------------------------ MoE (model.py Gate, Expert, MoE)

ggml_tensor* Engine::expert_ffn(Graph& g, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, ggml_tensor* x,
                                float limit) {
  ggml_context* c = g.ctx;
  ggml_tensor* xq = fq8(g, x);
  ggml_tensor* a = ggml_swiglu_clamp(c, ggml_mul_mat(c, gate, xq), ggml_mul_mat(c, up, xq), limit);
  return ggml_mul_mat(c, down, fq8(g, a));
}

ggml_tensor* Engine::moe(Graph& g, int L, ggml_tensor* x) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  const int T = plan->T, E = hp.n_embd, X = hp.n_expert, K = hp.n_expert_used;
  // routing: sqrt(softplus) scores; the bias picks the experts, the raw scores weight them
  ggml_tensor* scores = ggml_sqrt(c, ggml_softplus(c, ggml_mul_mat(c, l.gate_inp, x)));  // [X, T]
  ggml_tensor* ids = ggml_argsort_top_k(c, ggml_add(c, scores, l.gate_bias), K);      // [K, T]
  ggml_tensor* w = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, scores, 1, X, T), ids), K, T);
  if (hp.expert_weights_norm) w = ggml_div(c, w, ggml_scale_bias(c, ggml_sum_rows(c, w), 1.0f, 1e-20f));
  w = ggml_scale(c, w, hp.expert_weights_scale);
  g.tap("router_w", w);
  g.tap("router_i", ids);

  ggml_tensor* xq = ggml_reshape_3d(c, fq8(g, x), E, 1, T);
  ggml_tensor* gate = ggml_mul_mat_id(c, l.gate_exps, xq, ids);  // [F, K, T]
  ggml_tensor* up = ggml_mul_mat_id(c, l.up_exps, xq, ids);
  ggml_tensor* a = ggml_swiglu_clamp(c, gate, up, hp.swiglu_clamp[L]);
  a = ggml_mul(c, a, ggml_reshape_3d(c, w, 1, K, T));  // the reference weights before w2
  ggml_tensor* d = ggml_mul_mat_id(c, l.down_exps, fq8(g, a), ids);  // [E, K, T]
  ggml_tensor* y = nullptr;
  for (int j = 0; j < K; j++) {
    ggml_tensor* dj = ggml_view_2d(c, d, E, T, d->nb[2], (size_t)j * d->nb[1]);
    y = y ? ggml_add(c, y, dj) : dj;
  }
  return ggml_add(c, y, expert_ffn(g, l.gate_sh, l.up_sh, l.down_sh, x, hp.swiglu_clamp_sh[L]));
}

// ------------------------------------------------------------------ one block (model.py Block.forward)

void Engine::block(Graph& g, int L, ggml_tensor** hp_, ggml_tensor** pre_) {
  ggml_context* c = g.ctx;
  const Layer& l = m.layers[L];
  ggml_tensor* h = *hp_;
  ggml_tensor *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb;
  // each sublayer's mixes give the *next* sublayer's input mix: attention uses the pre-mix the
  // previous block's FFN made, the FFN uses the one this attention made
  hc_mixes(g, h, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, &a_pre, &a_post, &a_comb);
  ggml_tensor* x = rms(g, ggml_dsv4_hc_pre(c, h, *pre_), l.attn_norm);
  g.tap("attn_in", x);
  x = attention(g, L, x);
  g.tap("attn_out", x);
  h = ggml_dsv4_hc_post(c, x, h, a_post, a_comb);

  hc_mixes(g, h, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, &f_pre, &f_post, &f_comb);
  x = rms(g, ggml_dsv4_hc_pre(c, h, a_pre), l.ffn_norm);
  g.tap("ffn_in", x);
  x = moe(g, L, x);
  g.tap("ffn_out", x);
  h = ggml_dsv4_hc_post(c, x, h, f_post, f_comb);
  g.tap("h_out", h);
  g.tap("pre_out", f_pre);
  *hp_ = h;
  *pre_ = f_pre;
}

// ------------------------------------------------------------------ entry points

std::vector<float> Engine::forward(const std::vector<int32_t>& tokens) {
  begin_chunk(tokens);
  Graph g;
  ggml_context* c = g.ctx;
  const int T = plan->T, E = hp.n_embd, HC = hp.hc_mult;
  ggml_tensor* tok = g.input("tokens", GGML_TYPE_I32, {T}, &plan->tokens);
  ggml_tensor* h = ggml_get_rows(c, m.tok_embd, tok);  // [E, T]
  h = ggml_repeat_4d(c, ggml_reshape_3d(c, h, E, 1, T), E, HC, T, 1);  // hc copies of the embedding
  ggml_tensor* pre = g.input("one_hot_pre", GGML_TYPE_F32, {HC, T}, &plan->one_hot_pre);
  for (int L = 0; L < hp.n_layer; L++) {
    if (!m.layers[L].loaded) throw err("forward needs all layers loaded");
    if (m.layers[L].kind.engram >= 0) h = engram(g, L, h);
    block(g, L, &h, &pre);
  }
  ggml_tensor* x = ggml_dsv4_hc_pre(c, h, pre);  // collapse with the last block's FFN pre-mix
  x = ggml_view_2d(c, x, E, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);  // generation needs the last position
  ggml_tensor* logits = ggml_mul_mat(c, m.output, rms(g, x, m.output_norm));
  ggml_set_output(logits);
  ggml_build_forward_expand(g.gf, logits);
  compute(g);
  std::vector<float> out(hp.n_vocab);
  ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
  end_chunk();
  return out;
}

Taps Engine::run_layer(int L, const std::vector<float>& h_in, const std::vector<float>& pre_in) {
  if (!plan) throw err("run_layer outside begin_chunk/end_chunk");
  Graph g;
  g.want_taps = true;
  const int T = plan->T, E = hp.n_embd, HC = hp.hc_mult;
  ggml_tensor* h = g.input("h_in", GGML_TYPE_F32, {E, HC, T}, &h_in);
  ggml_tensor* pre = g.input("pre_in", GGML_TYPE_F32, {HC, T}, &pre_in);
  if (m.layers[L].kind.engram >= 0) h = engram(g, L, h);
  g.tap("h_in", h);
  block(g, L, &h, &pre);
  if (g.topk.count(L) && g.topk[L]) ggml_set_output(g.topk[L]);
  compute(g);
  if (g.topk.count(L) && g.topk[L]) {
    auto& v = topk_host[L];
    v.resize(ggml_nelements(g.topk[L]));
    ggml_backend_tensor_get(g.topk[L], v.data(), 0, ggml_nbytes(g.topk[L]));
  }
  Taps taps;
  for (auto& [name, t] : g.taps) {
    Tap tp;
    for (int d = 0; d < ggml_n_dims(t); d++) tp.ne.push_back(t->ne[d]);
    if (t->type == GGML_TYPE_I32) {
      tp.i.resize(ggml_nelements(t));
      ggml_backend_tensor_get(t, tp.i.data(), 0, ggml_nbytes(t));
    } else {
      tp.f.resize(ggml_nelements(t));
      ggml_backend_tensor_get(t, tp.f.data(), 0, ggml_nbytes(t));
    }
    taps[name] = std::move(tp);
  }
  return taps;
}
