// The DeepSeek-V4.1-Flash weights, memory-mapped from the split GGUF.
//
// Nothing is copied: every shard is mmap'd read-only and each tensor's data points into the
// mapping (the 474 GB model lives in the page cache, so "loading" is just page-table work).
// Each shard is wrapped in a ggml CPU buffer, so the tensors can be used directly in CPU graphs.
//
// The loader is strict on purpose: every tensor the model needs is checked for its exact type and
// shape, and every tensor in the file must be claimed by the schema. A different conversion or
// a different model fails here, not as wrong numbers three steps later.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

struct Hparams {
  int n_layer, n_embd, n_vocab;
  // attention
  int n_head, head_dim, rope_dim, q_lora, o_groups, o_lora, window;
  float rms_eps, rope_base, compress_rope_base;
  float yarn_factor, yarn_beta_fast, yarn_beta_slow;
  int yarn_orig_ctx;
  std::vector<int> compress_ratios;  // per layer (the file also lists the 3 MTP layers; we keep n_layer)
  std::vector<int> kv_source_layers, index_source_layers;
  int cand_source, cand_topk_blocks, cand_block;
  int idx_heads, idx_dim, idx_topk;
  // MoE
  int n_expert, n_expert_used, n_ff_exp, gating_func;
  float expert_weights_scale;
  bool expert_weights_norm;
  std::vector<float> swiglu_clamp, swiglu_clamp_sh;  // per layer
  // hyper-connections
  int hc_mult, hc_iters;
  float hc_eps;
  // engram
  std::vector<int> engram_layers;
  int engram_ngram, engram_heads, engram_dim, engram_pad;
  std::vector<uint64_t> engram_mult;     // [engram layer][ngram]       hash multipliers
  std::vector<uint64_t> engram_offsets;  // [engram layer][hash column] first row in the table
  std::vector<uint64_t> engram_primes;   // [engram layer][hash column] bucket count (prime)
  std::vector<int32_t> engram_token_map; // token id -> compressed id

  int engram_cols() const { return (engram_ngram - 1) * engram_heads; }  // rows fetched per token
};

// How a layer's attention is wired. Derived from the hparams, not from which tensors exist
// (the loader then checks that the tensors agree with it).
struct LayerKind {
  int ratio = 0;            // 0: sliding window only; r: also attends to KV compressed r:1
  bool kv_source = false;   // builds the compressed KV (+ index keys) that later layers reuse
  bool index_source = false;// runs its own indexer (top-k over compressed positions)
  bool cand_source = false; // its indexer also picks the candidate blocks for later indexers
  bool uses_cands = false;  // its indexer only scores inside the candidate blocks
  int kv_from = -1;         // layer whose compressed KV it reads (itself if kv_source)
  int idx_from = -1;        // layer whose top-k it uses (itself if index_source)
  int engram = -1;          // engram slot (index into Hparams::engram_layers), -1 if none
};

struct Layer {
  LayerKind kind;
  ggml_tensor *attn_norm, *q_a, *q_a_norm, *q_b, *kv, *kv_norm, *out_a, *out_b, *sinks;
  ggml_tensor *cmp_kv = nullptr, *cmp_gate = nullptr, *cmp_norm = nullptr;  // kv sources (gate: ratio > 1)
  ggml_tensor *idx_k = nullptr, *idx_k_norm = nullptr;                       // kv sources
  ggml_tensor *idx_q_b = nullptr, *idx_proj = nullptr;                       // index sources
  ggml_tensor *ffn_norm, *gate_inp, *gate_bias, *gate_bias_vl;
  ggml_tensor *gate_exps, *up_exps, *down_exps;  // [n_embd, n_ff, n_expert] mxfp4 (down: [n_ff, n_embd, n_expert])
  ggml_tensor *gate_sh, *up_sh, *down_sh;
  ggml_tensor *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
  ggml_tensor *engram_kv = nullptr, *engram_gate = nullptr;
  bool loaded = false;
};

struct Model {
  Hparams hp;
  std::vector<Layer> layers;
  ggml_tensor *tok_embd, *output, *output_norm, *engram_table;

  // Opens all shards next to `first_shard` (…-00001-of-000NN.gguf) and binds the tensors of the
  // layers in `layers` (empty = all). Global tensors are always bound. Throws on any mismatch.
  void load(const std::string& first_shard, const std::vector<int>& layers = {});
  // Maps the routed experts of the loaded layers (except `skip`) into this process's page
  // tables up front, with n_threads in parallel (MADV_POPULATE_READ: pages not yet in the page
  // cache are read from disk). Without it, every expert's first use in a run pays ~4k minor
  // page faults inside the decode step. Returns the bytes covered.
  size_t map_experts(int n_threads, const std::set<int>& skip = {});
  ~Model();

  ggml_tensor* get(const std::string& name) const;  // any tensor in the file, nullptr if absent
  size_t n_tensors() const { return by_name.size(); }
  std::vector<std::string> shard_paths;

 private:
  struct Shard {
    int fd = -1;
    void* map = nullptr;
    size_t size = 0;
    struct gguf_context* g = nullptr;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
  };
  std::vector<Shard> shards;
  std::unordered_map<std::string, ggml_tensor*> by_name;
};

std::string fmt_bytes(double b);
