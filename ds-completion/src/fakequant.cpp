#include "fakequant.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>

// Rounding to a value grid with spacing q: nearbyintf uses the current mode, which is
// round-to-nearest-even unless someone changed it - the same tie rule as the hardware casts.
static inline float round_to(float v, float q) { return nearbyintf(v / q) * q; }

float fq_e4m3_round(float v) {
  // e4m3fn: bias 7, 3 mantissa bits, normals from 2^-6, subnormals step 2^-9, max 448
  const float a = std::min(fabsf(v), 448.0f);
  if (a == 0.0f) return copysignf(0.0f, v);
  int e;
  frexpf(a, &e);              // a = m * 2^e, m in [0.5, 1) -> binade 2^(e-1)
  const int binade = std::max(e - 1, -6);
  const float r = std::min(round_to(a, ldexpf(1.0f, binade - 3)), 448.0f);
  return copysignf(r, v);
}

float fq_e2m1_round(float v) {
  // e2m1: {0, 0.5, 1, 1.5, 2, 3, 4, 6}; spacing 0.5 below 2, 1 in [2, 4), 2 in [4, 6]
  const float a = std::min(fabsf(v), 6.0f);
  const float q = a < 2.0f ? 0.5f : a < 4.0f ? 1.0f : 2.0f;
  return copysignf(round_to(a, q), v);
}

// 2^ceil(log2(v)) for positive normal v (the kernels' fast_round_scale bit trick)
static inline float pow2_ceil(float v) {
  int e;
  const float m = frexpf(v, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

namespace {
enum class Kind { FP8, FP4, FP4_E4 };

template <Kind K>
void fq_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void*) {
  constexpr int B = K == Kind::FP4_E4 ? 16 : 32;
  GGML_ASSERT(a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(dst));
  GGML_ASSERT(a->ne[0] % B == 0);
  const int64_t n_blocks = ggml_nelements(a) / B;
  const int64_t per = (n_blocks + nth - 1) / nth;
  const int64_t b0 = per * ith, b1 = std::min(n_blocks, b0 + per);
  const float* x = (const float*)a->data;
  float* y = (float*)dst->data;
  for (int64_t b = b0; b < b1; b++) {
    const float* xb = x + b * B;
    float* yb = y + b * B;
    float amax = 0.0f;
    for (int j = 0; j < B; j++) amax = std::max(amax, fabsf(xb[j]));
    float s;
    if constexpr (K == Kind::FP8) {
      s = pow2_ceil(std::max(amax, 1e-4f) * (1.0f / 448.0f));
      for (int j = 0; j < B; j++) yb[j] = fq_e4m3_round(std::clamp(xb[j] / s, -448.0f, 448.0f)) * s;
    } else {
      if constexpr (K == Kind::FP4) s = pow2_ceil(std::max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
      else s = fq_e4m3_round(std::max(amax, 6.0f * 0x1p-9f) / 6.0f);
      for (int j = 0; j < B; j++) yb[j] = fq_e2m1_round(std::clamp(xb[j] / s, -6.0f, 6.0f)) * s;
    }
  }
}
}  // namespace

ggml_tensor* fq_fp8(ggml_context* ctx, ggml_tensor* x) {
  return ggml_map_custom1(ctx, x, fq_op<Kind::FP8>, GGML_N_TASKS_MAX, nullptr);
}
ggml_tensor* fq_fp4(ggml_context* ctx, ggml_tensor* x) {
  return ggml_map_custom1(ctx, x, fq_op<Kind::FP4>, GGML_N_TASKS_MAX, nullptr);
}
ggml_tensor* fq_fp4_e4(ggml_context* ctx, ggml_tensor* x) {
  return ggml_map_custom1(ctx, x, fq_op<Kind::FP4_E4>, GGML_N_TASKS_MAX, nullptr);
}

float fq_fp8_split(const float* xb, float* q) {
  float amax = 0.0f;
  for (int j = 0; j < 32; j++) amax = std::max(amax, fabsf(xb[j]));
  const float s = pow2_ceil(std::max(amax, 1e-4f) * (1.0f / 448.0f));
  for (int j = 0; j < 32; j++) q[j] = fq_e4m3_round(std::clamp(xb[j] / s, -448.0f, 448.0f));
  return s;
}

void fq_fp8_host(const float* x, float* y, int64_t n) {
  for (int64_t b = 0; b < n / 32; b++) {
    float q[32];
    const float s = fq_fp8_split(x + b * 32, q);
    for (int j = 0; j < 32; j++) y[b * 32 + j] = q[j] * s;
  }
}

// e4m3fn byte for a value that is exactly representable (asserted by the caller's round trip)
static uint8_t e4m3_encode(float v) {
  uint8_t s = std::signbit(v) ? 0x80 : 0;
  float a = fabsf(v);
  if (a == 0.0f) return s;
  int e;
  float m = frexpf(a, &e);  // a = m * 2^e, m in [0.5, 1)
  int E = e - 1 + 7;        // biased exponent for 1.f * 2^(e-1)
  if (E <= 0) {             // subnormal: a = mant * 2^-9
    int mant = (int)lrintf(a * 512.0f);
    return s | (uint8_t)mant;
  }
  int mant = (int)lrintf((m * 2.0f - 1.0f) * 8.0f);
  return s | (uint8_t)(E << 3) | (uint8_t)mant;
}

static float e4m3_decode(uint8_t c) {
  const float sign = (c & 0x80) ? -1.0f : 1.0f;
  const int E = (c >> 3) & 0xF, M = c & 7;
  if (E == 0) return sign * ldexpf((float)M, -9);
  return sign * ldexpf(1.0f + M / 8.0f, E - 7);
}

static float bf16_value(uint16_t b) {
  const uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int64_t fp8_repack_bf16(const uint16_t* w, int64_t N, int64_t K, uint8_t* codes, uint8_t* scales) {
  int64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic)
  for (int64_t bn = 0; bn < N / 32; bn++) {
    for (int64_t bk = 0; bk < K / 32; bk++) {
      float amax = 0;
      for (int r = 0; r < 32; r++)
        for (int c = 0; c < 32; c++) amax = fmaxf(amax, fabsf(bf16_value(w[(bn * 32 + r) * K + bk * 32 + c])));
      int e = 0;  // scale exponent: 2^e
      if (amax > 0) {
        int ex;
        const float mm = frexpf(amax / 448.0f, &ex);
        e = mm == 0.5f ? ex - 1 : ex;  // ceil(log2)
      }
      const float sc = ldexpf(1.0f, e);
      scales[bn * (K / 32) + bk] = (uint8_t)(e + 127);
      for (int r = 0; r < 32; r++)
        for (int c = 0; c < 32; c++) {
          const int64_t i = (bn * 32 + r) * K + bk * 32 + c;
          const float v = bf16_value(w[i]);
          const uint8_t code = e4m3_encode(fq_e4m3_round(v / sc));
          codes[i] = code;
          if (e4m3_decode(code) * sc != v) bad++;
        }
    }
  }
  return bad;
}

bool fp8_codec_selftest() {
  for (int c = 0; c < 256; c++) {
    if ((c & 0x7F) == 0x7F) continue;  // NaN codes
    const float v = e4m3_decode((uint8_t)c);
    if (fq_e4m3_round(v) != v || (v != 0 && e4m3_encode(v) != c)) return false;
  }
  return true;
}
