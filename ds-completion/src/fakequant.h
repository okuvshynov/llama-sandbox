// The reference's fp8 / fp4 "fake quantization" (quantize, then dequantize back to float): the
// scalar rules and the host functions. No ggml here (README step 172): the GPU runtime, fp8pack
// and the kernels' benchmarks use this half. The same roundings as ggml custom ops, for the CPU
// reference path's graphs, are in fakequant_graph.h.
//
// The model was trained with these roundings in place, so the reference (inference/kernel.py)
// applies them at fixed points; matching them is what makes our per-layer outputs comparable
// to the reference bit for bit-ish instead of to ~1e-2.
#pragma once

#include <cmath>
#include <cstdint>

// host fake quantization of a plain array (no graph): act_quant(x, 32, ue8m0, inplace)
void fq_fp8_host(const float* x, float* y, int64_t n);
// one block of 32: q[j] = the e4m3 values, returns the scale (y = q * scale)
float fq_fp8_split(const float* xb, float* q);

// the scalar rules, exposed for the unit test
float fq_e4m3_round(float v);  // nearest e4m3 (fn) value, ties to even, saturating at +-448
float fq_e2m1_round(float v);  // nearest e2m1 value, ties to even, input clamped to +-6

// Re-block a bf16 weight [N rows][K] that is an exact expansion of fp8 into e4m3 codes [N][K]
// and one e8m0 exponent per 32x32 block [N/32][K/32] (value = code * 2^(e - 127)), with
// s = 2^ceil(log2(amax / 448)) per block (fp8pack.cpp, ds_ref.py fp8_blocks). Returns how many
// values the codes do not reproduce exactly (0 = lossless). N and K must be multiples of 32.
// The e4m3 codec used by fp8_repack_bf16 agrees with fq_e4m3_round on every code.
bool fp8_codec_selftest();
int64_t fp8_repack_bf16(const uint16_t* w, int64_t N, int64_t K, uint8_t* codes, uint8_t* scales);

// 2^ceil(log2(v)) for positive normal v (the kernels' fast_round_scale bit trick); the scale rule
// of both halves
inline float fq_pow2_ceil(float v) {
  int e;
  const float m = frexpf(v, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}
