#include "fakequant_graph.h"

#include "fakequant.h"

#include <algorithm>
#include <cmath>

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
      s = fq_pow2_ceil(std::max(amax, 1e-4f) * (1.0f / 448.0f));
      for (int j = 0; j < B; j++) yb[j] = fq_e4m3_round(std::clamp(xb[j] / s, -448.0f, 448.0f)) * s;
    } else {
      if constexpr (K == Kind::FP4) s = fq_pow2_ceil(std::max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
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
