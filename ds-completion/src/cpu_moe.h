// The routed experts of one layer for one token on the CPU (PLAN.md A3), in the reference's
// numerics: y = sum_j down_j(fq8(w_j * swiglu_clamp(gate_j(fq8 x), up_j(fq8 x)))).
//
// Weights are read in place from the GGUF mmap (mxfp4: per 32 values one e8m0 scale byte and
// 16 bytes, low nibbles = values 0..15, high nibbles = 16..31). The fp4 codes are decoded with
// a 16-entry float table (one AVX-512 vpermps per 16 values); every fp4 x fp8 product is exact
// in fp32, so this is the reference's fp8_gemm/fp4_gemm up to the order of the fp32 additions.
// (ggml's CPU kernel instead re-quantizes the activations to int8, ~1e-2 off.)
#pragma once

#include "canon_dot.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

struct CpuExperts {
  // one layer's [K, N, n_expert] mxfp4 tensors (ggml layout: ne0 = K = input dim)
  const ggml_tensor *gate, *up, *down;
  float limit;  // swiglu clamp
};

class CpuMoeKernel {
 public:
  CpuMoeKernel(int n_embd, int n_ff, int k_max, int n_threads);
  // x [n_embd] (f32, not yet quantized), ids/w [k] -> y [n_embd]
  void forward(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* y);
  // the same in the placement-invariant numerics (canon_dot.h; PLAN.md "Adaptive expert cache"):
  // every expert's output row [k][E] (bit-identical to fp4_moe_canon_rows' on a GPU), or their sum
  // in expert order
  void forward_canon_rows(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* rows);
  void forward_canon(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* y);
  // A prompt chunk's experts at once: token t's routed experts are routes[t] (x: [T][n_embd]).
  // Grouped by expert, so every expert's rows are read once for all its tokens; per (token,
  // expert) the arithmetic is forward()'s (canon: forward_canon_rows'), so the results are
  // bit-identical. out: canon, every pair's row [in token order, then list order][E]; else per
  // token its sum [T][E] (tokens without experts: untouched).
  struct CpuRoute { const int* ids; const float* w; int count; };
  void forward_batch(const CpuExperts& L, const float* x, int T, const CpuRoute* routes, bool canon, float* out);
  // control for the benchmark: the same tasks and rows, only reading the bytes
  uint64_t touch(const CpuExperts& L, const int* ids, int k);

 private:
  int E, F, n_threads;
  std::vector<float> xq, gate_out, up_out, act_q, act_l, down_out;  // scratch (act_l: lane order)
  CanonAct x_c;
  std::vector<CanonAct> act_c;
  std::vector<float> b_gate, b_up, b_xl, b_act, b_down;  // forward_batch scratch
  std::vector<CanonAct> b_xc, b_ac;
};
