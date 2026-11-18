// The DSpark draft head of DeepSeek-V4.1-Flash-1017 (inference/model.py: DSparkBlock,
// DSparkAttention, DSparkMarkovHead, DSparkConfidenceHead, Transformer.forward_spec), on the
// runtime's GPU. Weights: the DSpark GGUF (arch "dflash"); its dense linears are exact fp8 in
// bf16 and are re-blocked to fp8 at load (fp8_repack_bf16), so they run through the same exact
// fp8 GEMV as the main model. Its 3 x 128 routed experts have the main model's shapes and type:
// they live in slots of the home die's expert pool when there is one (pool_slots, load_experts;
// README step 123), else in arrays of their own on the home die.
//
// How it drafts (model.py):
//   - The main forward hands over its stream at the *inputs* of layers 37, 38, 39, averaged over
//     the hc copies (main_hidden, [3][n_embd]); fc + norm turn it into main_x.
//   - seed(pos), after every main token (prompt included): each draft layer's own wkv/kv_norm of
//     main_x, at the main position, into that layer's 128-slot window ring.
//   - draft(anchor, pos), after a main step at pos produced `anchor`: a block of block_size
//     tokens [anchor, noise, noise, ...] at positions pos+1.. runs through the 3 draft layers.
//     Each block position attends to the window ring (main positions <= pos) and to all block
//     positions (both directions). Then the main head (on the draft's own final norm), plus the
//     Markov head's bias from the previous token, left to right, greedy: block_size drafts.
//     The confidence head scores each position.
// Numerics as the main runtime: fp32, fake quantization where the reference has it.
#pragma once

#include "expert_pool.h"
#include "fp4_moe.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

class Runtime;

struct DraftResult {
  std::vector<int32_t> ids;   // [block_size + 1]: the anchor, then the drafts
  std::vector<float> conf;    // [block_size] confidence logits
  std::vector<float> logits;  // [block_size][n_vocab] (Markov bias included), if asked for
};

class DSpark {
 public:
  // experts_in_pool: the routed experts are not uploaded here; the runtime then gives them
  // pool_slots() slots of the home die's pool with load_experts() (before the first draft)
  DSpark(Runtime& rt, const std::string& gguf_path, bool experts_in_pool = false);
  ~DSpark();
  // every expert, or with DS_DRAFT_KEEP only the kept ones (README step 125)
  int pool_slots() const;
  // the experts into slots from base on, layer after layer (current device = pool.dev)
  void load_experts(ExpertPool& pool, int base);

  int block_size = 5;
  int32_t noise_id = 128799;
  std::vector<int> target_layers;  // main layers whose input stream feeds the draft
  float* main_hidden = nullptr;    // device [n_targets][n_embd], written by the main forward
  const float* quarter = nullptr;  // device [hc]: 1/hc each (the mean over the hc copies)
  int target_index(int L) const;   // its slot in main_hidden, or -1

  // the window K/V of main position pos from `hidden` (default: main_hidden, this step's)
  void seed(int pos, const float* hidden = nullptr);
  float* hid_chunk = nullptr;      // device [Runtime::kChunk][n_targets][n_embd]: a forward chunk's hiddens
  static constexpr int kMaxChunk = 8;
  std::vector<float> in_emb, in_one_hot;  // draft()'s input block, on the host (kept for its asynchronous copies)
  std::vector<int32_t> in_ids;
  FILE* hidden_log = nullptr;
  FILE* debug_log = nullptr;   // debugging: the next draft's per-layer attn/ffn in/out [layer][4][block][n_embd], then closed  // debugging: every seeded position's main_hidden, f32 [n_targets][n_embd]
  DraftResult draft(int32_t anchor, int pos, bool keep_logits);
  // Runtime::mark / rewind: the window rings saved, or put back
  void mark_rings(bool restore, int slot);

 private:
  struct Fp8 { uint8_t* q = nullptr; uint8_t* s = nullptr; int N = 0, K = 0, stride = 0; };
  struct LayerW {
    Fp8 q_a, q_b, kv, out_a, out_b, gate_sh, up_sh, down_sh;
    const float *attn_norm, *q_a_norm, *kv_norm, *sinks, *ffn_norm, *gate_inp, *gate_bias;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    GpuExperts ex;
    const int* slot_of = nullptr;  // experts left out: expert id -> its slot among the layer's (else ids are slots)
    float clamp, clamp_sh;
  };
  Runtime& rt;
  int n_expert = 128, n_used = 3;
  FILE* route_log = nullptr;          // DS_DRAFT_ROUTE_LOG (README step 87)
  std::vector<int32_t> route_buf;    // [layer][block][8]
  float route_scale = 1.5f;
  bool route_norm = true;
  std::vector<LayerW> L;
  // experts in a pool: until load_experts, each layer's gate / up / down tensors in the mapped file
  bool in_pool = false;
  // DS_DRAFT_KEEP in a pool: per layer the kept experts, only they get slots (empty: every expert has one)
  std::vector<std::vector<int>> kept;
  struct ExpertSrc { const void *gate, *up, *down; };
  std::vector<ExpertSrc> ex_src;
  void* file_map = nullptr;
  size_t file_bytes = 0;
  Fp8 fc;
  const float *enc_norm = nullptr, *out_norm = nullptr;
  const uint16_t* markov_w2 = nullptr;      // device [n_vocab][rank] bf16
  std::vector<uint16_t> markov_w1;          // host [n_vocab][rank] bf16 (the embedding)
  std::vector<float> conf_proj;             // host [n_embd + rank]
  int rank = 256;
  std::vector<void*> owned;
  // device scratch
  float *main_x, *hblk, *h2, *pre, *a_pre, *a_post, *a_comb, *f_post, *f_comb, *mixes;
  float *xa, *t1, *qr, *qblk, *kvv, *kvblk, *o, *oa, *y, *ysh, *yr, *gate, *up, *act, *rlog, *rw, *xfin, *xn;
  float *logits, *mbias, *memb;
  int *rids, *iota5, *d_ids;  // d_ids: the block's ids on the device [block + 1]
  const uint16_t* markov_w1_dev = nullptr;  // device copy of the Markov embedding
  std::vector<float*> rings;  // [layer] window ring [window][head_dim]
  std::vector<float*> rings_mark[2];  // their copies at Runtime::mark, per slot (Runtime::kMarks)
  void* moe_scratch = nullptr;

  void* upload(const void* src, size_t n);
  Fp8 fp8(const void* bf16, int N, int K, const std::string& name);
};
