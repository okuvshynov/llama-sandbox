// The own decode runtime on one die (PLAN.md section 3, option 2): one token per step through
// our HIP kernels (kernels.h, fp8_gemv.h), with the routed experts on the CPU.
//
// It shares the ggml Engine's state (window rings, compressor groups, compressed KV, index
// keys: the same GPU tensors), so prefill can run through Engine::forward and decode here,
// and every layer can be checked teacher-forced against the reference like the Engine is.
// Routed experts: src/cpu_moe.h (AVX-512, exact fp4 x fp8, weights read in place from the mmap).
// Weights: the fp8 linears from fp8pack's file (exact fp8 x fp8 GEMV); the small f32/bf16
// tensors (norms, hc, router, compressor, indexer projections, head) are the Engine's own GPU
// copies. Numerics are the reference's: fp32 everywhere, fake quantization where it has it.
#pragma once

#include "forward.h"

#include <hip/hip_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

class CpuMoeKernel;  // src/cpu_moe.h: the routed experts on the CPU, exact fp4 x fp8

class Runtime {
 public:
  // eng must be in GPU mode; fp8_path is fp8pack's output for the same model
  Runtime(Engine& eng, const std::string& fp8_path);
  ~Runtime();

  // one decode step for `token` at position eng.st.n_past: returns the logits
  std::vector<float> decode(int32_t token);

  // teacher-forced single layer (tests): begin_step(token), then run_layer for L = 0.. in
  // order (consumers read what their source layer wrote), then end_step(). Inputs and taps as
  // Engine::run_layer (h_in is the stream before the engram on engram layers).
  void begin_step(int32_t token);
  Taps run_layer(int L, const std::vector<float>& h_in, const std::vector<float>& pre_in);
  void end_step();

  // time spent per decode step (host wall clock), split at the CPU-expert waits
  struct Times { double total_ms = 0, cpu_moe_ms = 0, wait_ms = 0; } last;
  // profile = true: GPU events between groups of launches; per group, the GPU time of the
  // last step (gaps included: a group charged with idle time was waiting on the host)
  bool profile = false;
  std::map<std::string, double> last_groups;

 private:
  struct Fp8W { uint8_t* q = nullptr; uint8_t* s = nullptr; int N = 0, K = 0; };
  struct LayerW {
    Fp8W q_a, q_b, kv, out_a, out_b, idx_q_b, gate_sh, up_sh, down_sh, engram_kv;
    const float *attn_norm, *q_a_norm, *kv_norm, *sinks, *ffn_norm, *gate_inp, *gate_bias;
    const float *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    const float *cmp_kv = nullptr, *cmp_gate = nullptr, *cmp_norm = nullptr, *idx_k = nullptr, *idx_k_norm = nullptr;
    const uint16_t* idx_proj = nullptr;  // bf16
    const float* engram_gate = nullptr;
    const float* freqs;  // [rope_dim / 2] this layer's rotary frequencies
  };

  Engine& eng;
  const Hparams& hp;
  hipStream_t s;
  std::vector<LayerW> W;
  void* fp8_buf = nullptr;  // all fp8 weights + scales
  float* freqs_buf = nullptr;
  std::unique_ptr<CpuMoeKernel> cpu_moe;

  // device scratch (one token)
  float *h, *h2, *pre, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *mixes;
  float *x, *t1, *qr, *q, *kvv, *o, *oa, *y, *ysh, *yr, *logits_d, *partial;
  // activation codes + scales for the GEMVs, two sets: A holds a layer input (x) that several
  // GEMVs read, B the intermediate ones (qr, o, oa, act, engram rows); up to 32768 values each
  struct Codes { uint8_t* q; float* s; };
  Codes A, B;
  float *cmp_kvn, *cmp_sc, *lat, *ik, *iq, *iw, *iscore, *gate, *up, *act, *router_logits, *rw, *erows, *ekv;
  int *rids, *iota;
  std::map<int, int*> topk_buf;        // [index source] its selection buffer
  std::map<int, const int*> topk_cur;  // [index source] this step's selection (topk_buf or iota)
  // pinned host staging for the CPU experts and the engram rows
  // the CPU experts' inputs, staged together so one copy moves them: x [n_embd], ids [8], w [8]
  struct MoeIn { float* x; int* ids; float* w; } d_moe, h_moe;
  float *h_yr, *h_erows, *att_scratch;
  hipEvent_t ev;

  int pos = 0;  // position of the current step
  int32_t token = 0;
  std::vector<int32_t> hist;  // tokens including the current one (engram)

  void quant(const float* xin, int n, Codes c);
  void gemv8(const Fp8W& w, Codes c, float* yout, int rows_per_group = 0);
  void attention(int L, const float* xin, float* out);
  void moe(int L, const float* xin, float* out, Taps* taps);
  void block(int L, Taps* taps);
  void engram(int L);
  void tap(Taps* taps, const char* name, const float* d, std::vector<int64_t> ne);
  void mark(const char* group);
  std::vector<std::pair<const char*, hipEvent_t>> marks;
  std::vector<hipEvent_t> ev_pool;
};
