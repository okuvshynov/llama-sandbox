#include "fp4_moe.h"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
  } while (0)

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------ upload

// ggml block_mxfp4 (scale byte + 16 code bytes) -> separate aligned codes and scales
__global__ void unpack_blocks_kernel(const uint8_t* src, size_t n_blocks, uint8_t* codes, uint8_t* scales) {
  const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  const uint8_t* s = src + b * 17;
  scales[b] = s[0];
  for (int i = 0; i < 16; i++) codes[b * 16 + i] = s[1 + i];
}

void fp4_unpack(const uint8_t* blocks, size_t n_blocks, uint8_t* codes, uint8_t* scales, hipStream_t s) {
  unpack_blocks_kernel<<<(unsigned)((n_blocks + 255) / 256), 256, 0, s>>>(blocks, n_blocks, codes, scales);
}

GpuMatrix fp4_upload(const void* src, int K, int N, int n_expert, hipStream_t s) {
  GpuMatrix m;
  m.N = N;
  m.K = K;
  const size_t blocks_per_expert = (size_t)N * K / 32;
  m.codes_stride = blocks_per_expert * 16;
  m.scales_stride = blocks_per_expert;
  CHECK(hipMalloc(&m.codes, blocks_per_expert * n_expert * 16));
  CHECK(hipMalloc(&m.scales, blocks_per_expert * n_expert));
  // through a staging buffer, 32 experts at a time
  const int chunk = 32;
  uint8_t* stage;
  CHECK(hipMalloc(&stage, blocks_per_expert * chunk * 17));
  for (int e0 = 0; e0 < n_expert; e0 += chunk) {
    const int ne = std::min(chunk, n_expert - e0);
    const size_t nb = blocks_per_expert * ne;
    CHECK(hipMemcpyAsync(stage, (const uint8_t*)src + blocks_per_expert * e0 * 17, nb * 17, hipMemcpyHostToDevice, s));
    fp4_unpack(stage, nb, m.codes + blocks_per_expert * e0 * 16, m.scales + blocks_per_expert * e0, s);
  }
  CHECK(hipStreamSynchronize(s));
  CHECK(hipFree(stage));
  return m;
}

void fp4_free(GpuMatrix& m) {
  (void)hipFree(m.codes);
  (void)hipFree(m.scales);
  m.codes = m.scales = nullptr;
}

// ------------------------------------------------------------------ helpers

__device__ inline float e4m3_round(float v) {
  const float a = fminf(fabsf(v), 448.0f);
  if (a == 0.0f) return copysignf(0.0f, v);
  int e;
  frexpf(a, &e);
  const int binade = max(e - 1, -6);
  const float q = ldexpf(1.0f, binade - 3);
  return copysignf(fminf(rintf(a / q) * q, 448.0f), v);
}

__device__ inline float pow2_ceil(float v) {
  int e;
  const float m = frexpf(v, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

__device__ inline uint8_t e4m3_code(float v) {
  const uint16_t h = __half_as_ushort(__float2half(v * (1.0f / 256.0f)));
  return (uint8_t)(((h >> 7) & 0x7F) | ((h >> 8) & 0x80));
}

// act_quant(x, 32, ue8m0) into paired order: element j of a block goes to 2j (j < 16) or
// 2(j - 16) + 1, so that byte j of a weight block meets (x_j, x_{j+16}) as one fp16 pair
__global__ void quant_paired_kernel(const float* x, uint8_t* q, float* s, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float v = i < n ? x[i] : 0.0f;
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  const float sc = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  if (i < n) {
    const int j = i & 31;
    q[(i & ~31) + (j < 16 ? 2 * j : 2 * (j - 16) + 1)] = e4m3_code(e4m3_round(fminf(fmaxf(v / sc, -448.0f), 448.0f)));
    if (j == 0) s[i / 32] = sc;
  }
}

// 4 e4m3 codes -> half2 (c0, c1), (c2, c3), each the value x 2^-8 (as in fp8_gemv.hip)
__device__ inline void e4m3x4_to_half2x2(uint32_t v, uint32_t& h01, uint32_t& h23) {
  const uint32_t t01 = __builtin_amdgcn_perm(0, v, 0x010c000c);
  const uint32_t t23 = __builtin_amdgcn_perm(0, v, 0x030c020c);
  h01 = ((t01 >> 1) & 0x3F803F80u) | (t01 & 0x80008000u);
  h23 = ((t23 >> 1) & 0x3F803F80u) | (t23 & 0x80008000u);
}

// one byte of e2m1 codes (low nibble = element j, high = element j + 16) -> half2, each x 2^-14
__device__ inline uint32_t e2m1x2_to_half2(uint32_t b) {
  return ((b & 7u) << 9) | ((b & 8u) << 12) | ((b & 0x70u) << 21) | ((b & 0x80u) << 24);
}

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, a), __builtin_bit_cast(half2_t, b), c, false);
}

// ------------------------------------------------------------------ the expert GEMV

// grid (row blocks, k): expert ids[blockIdx.y]'s matrix times activation set blockIdx.y (or the
// shared one when x_stride == 0). LPR lanes per row, each lane one 32-block per step.
template <int LPR>
__global__ void __launch_bounds__(256) fp4_gemv_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                                                       int N, int K, size_t codes_stride, size_t scales_stride,
                                                       const int* __restrict__ ids, const int* __restrict__ count,
                                                       const uint8_t* __restrict__ xq, const float* __restrict__ xs,
                                                       int x_stride, float* __restrict__ y) {
  const int j = blockIdx.y;
  if (count && j >= *count) return;  // the whole block: no barriers below
  const int e = ids[j];
  const int lane = threadIdx.x & 63;
  const int row = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int l = lane % LPR, nb = K / 32;
  const bool live = row < N;
  const int r = live ? row : N - 1;
  const uint8_t* wc = codes + (size_t)e * codes_stride + (size_t)r * (K / 2);
  const uint8_t* ws = scales + (size_t)e * scales_stride + (size_t)r * nb;
  const uint8_t* xr = xq + (size_t)j * x_stride;
  const float* xsr = xs + (size_t)j * (x_stride / 32);
  float acc = 0.0f;
  for (int b = l; b < nb; b += LPR) {
    const uint4 w = *(const uint4*)(wc + (size_t)b * 16);
    const uint4 x0 = *(const uint4*)(xr + (size_t)b * 32), x1 = *(const uint4*)(xr + (size_t)b * 32 + 16);
    uint32_t hx[16];
    e4m3x4_to_half2x2(x0.x, hx[0], hx[1]);
    e4m3x4_to_half2x2(x0.y, hx[2], hx[3]);
    e4m3x4_to_half2x2(x0.z, hx[4], hx[5]);
    e4m3x4_to_half2x2(x0.w, hx[6], hx[7]);
    e4m3x4_to_half2x2(x1.x, hx[8], hx[9]);
    e4m3x4_to_half2x2(x1.y, hx[10], hx[11]);
    e4m3x4_to_half2x2(x1.z, hx[12], hx[13]);
    e4m3x4_to_half2x2(x1.w, hx[14], hx[15]);
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    float p = 0.0f;
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int t = 0; t < 4; t++) p = dot2(e2m1x2_to_half2((wv[q] >> (8 * t)) & 0xFF), hx[4 * q + t], p);
    // 2^(e - 127) x 2^22 (undoing 2^-14 x 2^-8) x the activation scale
    const float wscale = __uint_as_float((uint32_t)ws[b] << 23) * 4194304.0f;
    acc += p * (wscale * xsr[b]);
  }
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
  if (live && l == 0) y[(size_t)j * N + row] = acc;
}

static void fp4_gemv(const GpuMatrix& m, const int* ids, const int* count, int k, const uint8_t* xq, const float* xs,
                     int x_stride, float* y, hipStream_t s) {
  const int nb = m.K / 32;
  if (nb >= 128) {
    constexpr int LPR = 16;
    fp4_gemv_kernel<LPR><<<dim3(cdiv(m.N, 4 * 64 / LPR), k), 256, 0, s>>>(m.codes, m.scales, m.N, m.K, m.codes_stride,
                                                                          m.scales_stride, ids, count, xq, xs, x_stride, y);
  } else {
    constexpr int LPR = 8;
    fp4_gemv_kernel<LPR><<<dim3(cdiv(m.N, 4 * 64 / LPR), k), 256, 0, s>>>(m.codes, m.scales, m.N, m.K, m.codes_stride,
                                                                          m.scales_stride, ids, count, xq, xs, x_stride, y);
  }
}

// ------------------------------------------------------------------ the block

__global__ void moe_act_kernel(const float* gate, const float* up, const float* w, float lim, int F, int n,
                               const int* count, float* act) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (count && i / F >= *count) { act[i] = 0.0f; return; }  // unused expert: finite, never read
  const float g = fminf(gate[i], lim), u = fminf(fmaxf(up[i], -lim), lim);
  act[i] = w[i / F] * (g / (1.0f + expf(-g)) * u);  // the routing weight before w2, as the reference
}

__global__ void expert_sum_kernel(const float* d, int k, const int* count, int E, float* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;
  if (count) k = *count;
  float v = 0.0f;
  for (int j = 0; j < k; j++) v += d[(size_t)j * E + i];  // expert order, as the CPU kernel
  y[i] = v;
}

size_t fp4_moe_scratch_bytes(int E, int F, int k) {
  return (size_t)E + (size_t)E / 32 * 4 + (size_t)k * F * 4 * 3 + (size_t)k * F + (size_t)k * F / 32 * 4 +
         (size_t)k * E * 4 + 256;
}

void fp4_moe(const GpuExperts& ex, const float* x, const int* ids, const float* w, int k, float limit, float* y,
             void* scratch, hipStream_t s, const int* count) {
  const int E = ex.gate.K, F = ex.gate.N;
  char* p = (char*)scratch;
  auto take = [&](size_t bytes) { char* r = p; p += (bytes + 63) / 64 * 64; return r; };
  uint8_t* xq = (uint8_t*)take(E);
  float* xs = (float*)take(E / 32 * 4);
  float* g = (float*)take((size_t)k * F * 4);
  float* u = (float*)take((size_t)k * F * 4);
  float* a = (float*)take((size_t)k * F * 4);
  uint8_t* aq = (uint8_t*)take((size_t)k * F);
  float* asc = (float*)take((size_t)k * F / 32 * 4);
  float* d = (float*)take((size_t)k * E * 4);
  quant_paired_kernel<<<cdiv(E, 256), 256, 0, s>>>(x, xq, xs, E);
  fp4_gemv(ex.gate, ids, count, k, xq, xs, 0, g, s);
  fp4_gemv(ex.up, ids, count, k, xq, xs, 0, u, s);
  moe_act_kernel<<<cdiv(k * F, 256), 256, 0, s>>>(g, u, w, limit, F, k * F, count, a);
  quant_paired_kernel<<<cdiv(k * F, 256), 256, 0, s>>>(a, aq, asc, k * F);  // per expert's 32-blocks: unused ones harmless
  fp4_gemv(ex.down, ids, count, k, aq, asc, F, d, s);
  expert_sum_kernel<<<cdiv(E, 256), 256, 0, s>>>(d, k, count, E, y);
}
