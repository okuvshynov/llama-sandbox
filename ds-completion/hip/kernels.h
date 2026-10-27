// Device kernels of the single-die decode runtime (rt.cpp). One token per step (M = 1).
//
// Every kernel mirrors one piece of inference/model.py (named in its comment) with the
// reference's numerics: fp32 math, and the fp8/fp4 fake quantization at the same points. All
// pointers are device pointers; all tensors f32 unless noted. Launches go on the given stream.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

// ---- norms, fake quantization ----

// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w   (w may be null), `rows` rows of n
void k_rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, hipStream_t s);
// in place: act_quant(x, 32, ue8m0, inplace) - e4m3 values, power-of-two scale per 32
void k_fq8(float* x, int n, hipStream_t s);
// in place: fp4_act_quant(x, 32) (e8m0 scales) or (x, 16, e4m3 scales)
void k_fq4(float* x, int n, bool e4m3_scale_16, hipStream_t s);

// ---- matrix-vector (M = 1) ----

// y[N] = W[N][K] . x[K], W row-major f32 / bf16. `partial` needs N * kSplitMax floats.
constexpr int kSplitMax = 64;
void k_gemv_f32(const float* W, int N, int K, const float* x, float* y, float* partial, hipStream_t s);
void k_gemv_bf16(const uint16_t* W, int N, int K, const float* x, float* y, float* partial, hipStream_t s);

// ---- rope (apply_rotary_emb) ----

// In place, on the last rope_dim dims of each of n_heads rows of head_dim, at one position.
// freqs: [rope_dim / 2] per-pair frequencies (precomputed on the host exactly as
// precompute_freqs_cis). inverse: rotate by -angle (the attention output's de-rotation).
void k_rope(float* x, int n_heads, int head_dim, int rope_dim, const float* freqs, int pos, bool inverse, hipStream_t s);

// ---- hyper-connections (Block.hc_mixes / hc_pre / hc_post, hc_split_sinkhorn) ----

// mixes[24] = hc_fn . h_flat (from k_gemv_f32, *not* yet scaled by the RMS statistic); this
// computes rsqrt(mean(h_flat^2) + eps) from h, then pre [hc], post [hc], comb [hc][hc] (src, dst).
void k_hc_split(const float* h, int n_flat, const float* mixes, const float* scale, const float* base, float norm_eps,
                float hc_eps, int iters, float* pre, float* post, float* comb, hipStream_t s);
// x = rms(sum_j pre[j] h[j]) * w   (h: [hc][n])
void k_hc_pre_norm(const float* h, const float* pre, const float* w, int n, float eps, float* x, hipStream_t s);
// x = sum_j pre[j] h[j] without the norm (the final head collapse uses k_rms_norm after)
void k_hc_pre(const float* h, const float* pre, int n, float* x, hipStream_t s);
// h_out[k] = x * post[k] + sum_j comb[j][k] * h[j]
void k_hc_post(const float* x, const float* h, const float* post, const float* comb, int n, float* h_out, hipStream_t s);

// ---- attention ----

// Decode attention of one query (all heads) over the window ring and the selected compressed
// positions (sparse_attn): keys are the ring slots whose position is in (pos - window, pos],
// then comp[topk[i]] for i < K. One softmax over all, with a per-head sink in the denominator.
// scratch: (window + K + 31) / 32 * n_head * (head_dim + 2) floats
void k_attn_decode(const float* q, int n_head, int head_dim, const float* ring, int window, int pos, const float* comp,
                   const int* topk, int K, const float* sinks, float scale, float* o, float* scratch, hipStream_t s);

// the window KV of one token: slot = fq8(rope(rms(kv) * w)) (rope on the last rope_dim dims)
void k_kv_finish(const float* kv, const float* w, int dim, int rope_dim, const float* freqs, int pos, float eps,
                 float* slot, hipStream_t s);

// ---- indexer ----

// score[c] = sum_h relu(q[h] . keys[c]) * w[h] * wscale for c < n (Indexer.forward, one query)
void k_index_scores(const float* q, int n_head, int dim, const float* keys, int n, const float* w, float wscale,
                    float* score, hipStream_t s);
// the indices of the k largest of score[0..n) (any order); n <= kTopkMaxN
constexpr int kTopkMaxN = 4096;
void k_topk(const float* score, int n, int k, int* idx, hipStream_t s);

// ---- compressor (Compressor.forward, decode) ----

// pooled[d] = sum_i kv_state[i][d] * softmax_i(score_state[i][d]), i < ratio
void k_compress_pool(const float* kv_state, const float* score_state, int ratio, int dim, float* pooled, hipStream_t s);

// ---- MoE (Gate, Expert) ----

// scores = sqrt(softplus(logits)); ids = top-k of scores + bias; w = scores[ids] (normalized
// with + 1e-20 if norm), times route_scale. Single block, n_expert <= 1024, k <= 8.
void k_router(const float* logits, const float* bias, int n_expert, int k, bool norm, float route_scale, int* ids,
              float* w, hipStream_t s);
// out = silu(min(gate, lim)) * clamp(up, -lim, lim) * (weight ? *weight : 1)
void k_swiglu(const float* gate, const float* up, float lim, int n, float* out, hipStream_t s);

// ---- engram (Engram.forward), h updated in place ----

// kv: [(hc + 1) * n] = hc keys then the value; gate_w: [hc][n] (q_weight * k_weight)
void k_engram(float* h, const float* kv, const float* gate_w, int hc, int n, float eps, hipStream_t s);

// ---- misc ----

void k_add(const float* a, const float* b, float* y, int n, hipStream_t s);

// Routed experts split by where they live (expert_pool.h placement). One list per die, plus one
// for the CPU; within a list the router's order is kept.
constexpr int kMoeMaxDies = 4, kMoeMaxK = 6;
struct MoeList {
  int count;
  int ids[kMoeMaxK];    // slots on a die; expert ids for the CPU
  float w[kMoeMaxK];    // routing weights
};
// table: [n_expert] of this layer, die << 16 | slot, or -1 (CPU). lists[0..n_dies) are the
// dies, lists[n_dies] the CPU.
void k_moe_dispatch(const int* ids, const float* w, int k, const int* table, int n_dies, MoeList* lists, hipStream_t s);
// y = a + parts[0] + ... + parts[n - 1] in this order (null parts skipped)
void k_add_parts(const float* a, const float* const* parts, int n, float* y, int len, hipStream_t s);
// y[j][i] = x[i] for j < copies (the embedding into hc copies)
void k_repeat(const float* x, int n, int copies, float* y, hipStream_t s);
