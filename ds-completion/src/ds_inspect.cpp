// ds-inspect: load the model (or some layers) through the real loader and show what it found.
//
//   ds-inspect -m <first shard .gguf> [--layers 0,1,2] [--touch]
//
// Prints the hyper-parameters, one line per layer with its attention wiring (which layer's
// compressed KV and top-k it reads) and its bytes, and the bytes a decode token reads.
// Loading itself is the schema check: every tensor of the selected layers is verified for type
// and shape, and every tensor in the file must be known. --touch reads every byte of the
// selected layers once (a checksum), to time page-cache reads through the mappings.

#include "model.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <stdexcept>

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    char* end;
    long a = strtol(p, &end, 10), b = a;
    if (*end == '-') b = strtol(end + 1, &end, 10);
    for (long x = a; x <= b; x++) v.push_back((int)x);
    p = *end == ',' ? end + 1 : end;
  }
  return v;
}

static size_t nb(const ggml_tensor* t) { return t ? ggml_nbytes(t) : 0; }

static std::string join(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
  return s;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string path;
  std::vector<int> sel;
  bool touch = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
    else if (!strcmp(argv[i], "--layers") && i + 1 < argc) sel = parse_list(argv[++i]);
    else if (!strcmp(argv[i], "--touch")) touch = true;
    else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (path.empty()) {
    fprintf(stderr, "usage: ds-inspect -m <model-00001-of-000NN.gguf> [--layers 0,1,2-5] [--touch]\n");
    return 2;
  }

  Model m;
  auto t0 = std::chrono::steady_clock::now();
  try {
    m.load(path, sel);
  } catch (const std::exception& e) {
    fprintf(stderr, "load failed: %s\n", e.what());
    return 1;
  }
  double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const Hparams& h = m.hp;
  printf("# %zu shards, %zu tensors, loaded in %.2f s; schema check passed for %s\n", m.shard_paths.size(),
         m.n_tensors(), t_load, sel.empty() ? "all layers" : ("layers " + join(sel)).c_str());

  printf("\nmodel      layers %d, embd %d, vocab %d, hc copies %d (sinkhorn %d iters, eps %g)\n", h.n_layer,
         h.n_embd, h.n_vocab, h.hc_mult, h.hc_iters, h.hc_eps);
  printf("attention  %d heads x %d (rope %d), q lora %d, out %d groups x lora %d, window %d, rms eps %g\n",
         h.n_head, h.head_dim, h.rope_dim, h.q_lora, h.o_groups, h.o_lora, h.window, h.rms_eps);
  printf("rope       base %g (window-only layers), compressed %g + YaRN x%g over %d (beta %g/%g)\n",
         h.rope_base, h.compress_rope_base, h.yarn_factor, h.yarn_orig_ctx, h.yarn_beta_fast, h.yarn_beta_slow);
  printf("indexer    %d heads x %d, top %d; candidates from layer %d: top %d blocks of %d\n", h.idx_heads,
         h.idx_dim, h.idx_topk, h.cand_source, h.cand_topk_blocks, h.cand_block);
  printf("moe        %d experts, top %d, ff %d, gating func %d, weights norm %d x %g, swiglu clamp %g\n",
         h.n_expert, h.n_expert_used, h.n_ff_exp, h.gating_func, h.expert_weights_norm, h.expert_weights_scale,
         h.swiglu_clamp[0]);
  printf("engram     layers %s, n-grams 2..%d x %d heads = %d rows of %d per token, pad %d -> %d, table %" PRId64
         " rows\n", join(h.engram_layers).c_str(), h.engram_ngram, h.engram_heads, h.engram_cols(), h.engram_dim,
         h.engram_pad, h.engram_token_map[h.engram_pad], m.engram_table->ne[1]);

  printf("\n%-3s %-5s %-6s %-6s %-30s %10s %10s\n", "L", "ratio", "kv<-", "idx<-", "role", "dense", "experts");
  size_t dense_all = 0, exp_all = 0, shared_all = 0;
  int n_loaded = 0;
  for (int L = 0; L < h.n_layer; L++) {
    const Layer& l = m.layers[L];
    if (!l.loaded) continue;
    n_loaded++;
    const LayerKind& k = l.kind;
    std::string role;
    if (k.kv_source) role += "kv-src ";
    if (k.index_source) role += k.cand_source ? "idx-src+cands " : k.uses_cands ? "idx-src(in cands) " : "idx-src ";
    if (k.engram >= 0) role += "engram ";
    if (k.ratio == 0) role += "window-only ";
    size_t experts = nb(l.gate_exps) + nb(l.up_exps) + nb(l.down_exps);
    size_t shared = nb(l.gate_sh) + nb(l.up_sh) + nb(l.down_sh);
    size_t dense = 0;
    for (const ggml_tensor* t : {l.attn_norm, l.q_a, l.q_a_norm, l.q_b, l.kv, l.kv_norm, l.out_a, l.out_b, l.sinks,
                                 l.cmp_kv, l.cmp_gate, l.cmp_norm, l.idx_k, l.idx_k_norm, l.idx_q_b, l.idx_proj,
                                 l.ffn_norm, l.gate_inp, l.gate_bias, l.gate_bias_vl, l.hc_attn_fn, l.hc_attn_base,
                                 l.hc_attn_scale, l.hc_ffn_fn, l.hc_ffn_base, l.hc_ffn_scale, l.engram_kv,
                                 l.engram_gate})
      dense += nb(t);
    dense += shared;
    dense_all += dense;
    exp_all += experts;
    shared_all += shared;
    char kv[8] = "-", idx[8] = "-";
    if (k.ratio) { snprintf(kv, sizeof kv, "%d", k.kv_from); snprintf(idx, sizeof idx, "%d", k.idx_from); }
    printf("%-3d %-5d %-6s %-6s %-30s %10s %10s\n", L, k.ratio, kv, idx, role.c_str(), fmt_bytes(dense).c_str(),
           fmt_bytes(experts).c_str());
  }
  const size_t head = nb(m.output) + nb(m.output_norm);
  printf("\nloaded %d layers: dense %s (of which shared experts %s), routed experts %s; head %s, embd %s,"
         " engram table %s\n", n_loaded, fmt_bytes(dense_all).c_str(), fmt_bytes(shared_all).c_str(),
         fmt_bytes(exp_all).c_str(), fmt_bytes(head).c_str(), fmt_bytes(nb(m.tok_embd)).c_str(),
         fmt_bytes(nb(m.engram_table)).c_str());
  const double per_tok = dense_all + exp_all * (double)h.n_expert_used / h.n_expert + head;
  printf("decode reads per token (these layers + head): %s = dense %s + %d of %d experts %s + head %s\n",
         fmt_bytes(per_tok).c_str(), fmt_bytes(dense_all).c_str(), h.n_expert_used, h.n_expert,
         fmt_bytes(exp_all * (double)h.n_expert_used / h.n_expert).c_str(), fmt_bytes(head).c_str());

  if (touch) {
    std::vector<const ggml_tensor*> ts;
    for (const Layer& l : m.layers)
      if (l.loaded)
        for (const ggml_tensor* t : {l.q_a, l.q_b, l.kv, l.out_a, l.out_b, l.gate_exps, l.up_exps, l.down_exps,
                                     l.gate_sh, l.up_sh, l.down_sh, l.hc_attn_fn, l.hc_ffn_fn, l.gate_inp,
                                     l.engram_kv, l.cmp_kv, l.cmp_gate, l.idx_q_b, l.idx_proj})
          if (t) ts.push_back(t);
    size_t total = 0;
    uint64_t sum = 0;
    auto t1 = std::chrono::steady_clock::now();
    for (const ggml_tensor* t : ts) {
      const uint64_t* p = (const uint64_t*)t->data;
      const size_t n = ggml_nbytes(t) / 8;
      uint64_t s = 0;
#pragma omp parallel for reduction(^ : s) schedule(static)
      for (size_t i = 0; i < n; i++) s ^= p[i];
      sum ^= s;
      total += ggml_nbytes(t);
    }
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    printf("touch: read %s in %.2f s = %.1f GB/s (%d threads), xor %016" PRIx64 "\n", fmt_bytes(total).c_str(), dt,
           total / dt / 1e9, omp_get_max_threads(), sum);
  }
  return 0;
}
