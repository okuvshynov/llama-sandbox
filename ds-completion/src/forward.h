// The DeepSeek-V4.1 forward pass as ggml graphs (CPU backend, one sequence).
//
// Structure follows inference/model.py one to one; the names of the pieces (hc_mixes, hc_pre,
// compressor, indexer, ...) are the reference's. What differs is only how state is kept:
//
//   - The sequence's state (window KV rings, compressor partial groups, compressed KV and index
//     keys) lives in persistent tensors (State). The host knows every position, so all the
//     position bookkeeping (which ring slot holds which position, which compression groups
//     complete in this chunk, what each query may see) is computed on the host into small input
//     tensors (Plan); the graph only does arithmetic and gathers.
//   - Tokens can be fed in chunks of any size at any position (the reference only prefills at 0).
//
// Layer-level testing: run_layer() executes one block on given inputs and returns the
// reference's named intermediates (the same names ds_ref.py records).
#pragma once

#include "model.h"

#include <map>
#include <string>
#include <vector>

struct Options {
  bool exact_quant = true;  // apply the reference's fp8/fp4 fake quantization (fakequant.h)
  int max_ctx = 4096;       // capacity of the compressed caches
  int n_threads = 16;
};

// Everything a later step needs from earlier tokens.
struct State {
  int n_past = 0;
  std::vector<int32_t> tokens;  // all tokens so far (the engram hashes look 3 back)
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  std::vector<ggml_tensor*> win;                    // [L] window KV ring [head_dim, window]
  std::map<int, ggml_tensor*> cmp_kv, cmp_score;    // [src] compressor partial group [head_dim, ratio] (ratio > 1)
  std::map<int, ggml_tensor*> comp;                 // [src] compressed KV [head_dim, max_ctx / ratio]
  std::map<int, ggml_tensor*> idx_k;                // [src] index keys [idx_dim, max_ctx / ratio]
};

// One named intermediate copied out of a graph (for tests and dumps).
struct Tap {
  std::vector<int64_t> ne;  // ggml order
  std::vector<float> f;     // f32 values, or
  std::vector<int32_t> i;   // i32 values
};
using Taps = std::map<std::string, Tap>;

// The engram table rows (absolute, n_cols of them) for the token at position q of `hist`
// (all tokens so far, including q), for engram slot e.
std::vector<int32_t> engram_rows(const Hparams& hp, const std::vector<int32_t>& hist, int e, int q);

struct Plan;     // host-side position bookkeeping for one chunk (forward.cpp)
struct GraphIn;  // the per-graph input tensors (forward.cpp)

// The reference's forward on the CPU (the GPU path is hip/rt.h, which does not use this).
class Engine {
 public:
  Engine(const Model& m, Options opt);
  ~Engine();

  // Full forward of `tokens` at positions [n_past, n_past + n): updates the state and returns
  // the logits of the last token.
  std::vector<float> forward(const std::vector<int32_t>& tokens);

  // Teacher-forced single block, for tests: begin_chunk() once per chunk, then run_layer() for
  // layers in increasing order (consumer layers read what their source layer wrote), then
  // end_chunk(). h_in is [n_embd * hc, n] (for engram layers: the stream *before* the engram),
  // pre_in [hc, n]. Taps returned: h_in (after engram), attn_in, attn_out, ffn_in, ffn_out,
  // router_w, router_i, topk (index-source layers), h_out, pre_out.
  void begin_chunk(const std::vector<int32_t>& tokens);
  Taps run_layer(int L, const std::vector<float>& h_in, const std::vector<float>& pre_in);
  void end_chunk();

  State st;

 private:
  struct Graph;
  const Model& m;
  const Hparams& hp;
  Options opt;
  ggml_backend_t cpu;
  std::vector<Layer> lay;  // m.layers
  Plan* plan = nullptr;  // current chunk
  // f32 copies of the small bf16 weights the reference applies in fp32 (the compressor pools in
  // fp32; ggml would round its input to bf16 for a bf16 weight, which the fp4 rounding of the
  // compressed KV then amplifies). Indexed by the original tensor.
  ggml_context* f32_ctx = nullptr;
  ggml_backend_buffer_t f32_buf = nullptr;
  std::map<const ggml_tensor*, ggml_tensor*> f32w;
  ggml_tensor* w32(ggml_tensor* t) const { auto it = f32w.find(t); return it == f32w.end() ? t : it->second; }
  // index-source selections of the current chunk, for consumer layers built in later graphs
  // (run_layer builds one graph per layer; forward() keeps them inside its one graph)
  std::map<int, std::vector<int32_t>> topk_host;

  // graph building, one function per reference module
  ggml_tensor* fq8(Graph& g, ggml_tensor* x);
  ggml_tensor* rms(Graph& g, ggml_tensor* x, ggml_tensor* w);
  ggml_tensor* rope(Graph& g, ggml_tensor* x, ggml_tensor* pos, int L, int n_offs, bool inverse = false);
  ggml_tensor* engram(Graph& g, int L, ggml_tensor* h);
  void hc_mixes(Graph& g, ggml_tensor* h, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                ggml_tensor** pre, ggml_tensor** post, ggml_tensor** comb);
  ggml_tensor* attention(Graph& g, int L, ggml_tensor* x);
  ggml_tensor* compressor(Graph& g, int L, ggml_tensor* x);
  ggml_tensor* indexer(Graph& g, int L, ggml_tensor* x, ggml_tensor* qr);
  ggml_tensor* moe(Graph& g, int L, ggml_tensor* x);
  ggml_tensor* expert_ffn(Graph& g, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* down, ggml_tensor* x, float limit);
  void block(Graph& g, int L, ggml_tensor** h, ggml_tensor** pre);
  void compute(Graph& g);
};
