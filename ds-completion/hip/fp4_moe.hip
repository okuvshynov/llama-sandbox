#include "fp4_moe.h"

#include "canon_math.h"

#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstring>
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
  uint8_t *stage, *host;  // through pinned host memory, not straight from the mmap (expert_pool.h)
  CHECK(hipMalloc(&stage, blocks_per_expert * chunk * 17));
  CHECK(hipHostMalloc(&host, blocks_per_expert * chunk * 17, 0));
  for (int e0 = 0; e0 < n_expert; e0 += chunk) {
    const int ne = std::min(chunk, n_expert - e0);
    const size_t nb = blocks_per_expert * ne;
    CHECK(hipStreamSynchronize(s));  // the previous chunk's copy out of `host` is done
    memcpy(host, (const uint8_t*)src + blocks_per_expert * e0 * 17, nb * 17);
    CHECK(hipMemcpyAsync(stage, host, nb * 17, hipMemcpyHostToDevice, s));
    fp4_unpack(stage, nb, m.codes + blocks_per_expert * e0 * 16, m.scales + blocks_per_expert * e0, s);
  }
  CHECK(hipStreamSynchronize(s));
  CHECK(hipFree(stage));
  CHECK(hipHostFree(host));
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
// 2(j - 16) + 1, so that byte j of a weight block meets (x_j, x_{j+16}) as one fp16 pair.
// Kept as fp8 codes (half the bytes of fp16: the GEMVs read activations from cache once per
// row) and converted to fp16 in the GEMV.
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

// 4 bytes of e2m1 codes (byte t: low nibble = element j, high = element j + 16) -> 4 half2,
// each value x 2^-14: fp16 bits ((c & 7) << 9) | ((c & 8) << 12). Only the high byte of each
// half is non-zero: (c & 7) << 1 | (c & 8) << 4. Built for all 4 bytes at once, then placed by
// byte permutes.
__device__ inline void e2m1x8_to_half2x4(uint32_t v, uint32_t h[4]) {
  const uint32_t lo = ((v & 0x07070707u) << 1) | ((v & 0x08080808u) << 4);  // low nibbles' high bytes
  const uint32_t hi = ((v >> 3) & 0x0E0E0E0Eu) | (v & 0x80808080u);         // high nibbles' high bytes
  // result bytes [0, lo.t, 0, hi.t]: perm(a = hi, b = lo): selectors 0-3 pick b, 4-7 pick a, 0x0c = 0
  h[0] = __builtin_amdgcn_perm(hi, lo, 0x040c000cu);
  h[1] = __builtin_amdgcn_perm(hi, lo, 0x050c010cu);
  h[2] = __builtin_amdgcn_perm(hi, lo, 0x060c020cu);
  h[3] = __builtin_amdgcn_perm(hi, lo, 0x070c030cu);
}

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, a), __builtin_bit_cast(half2_t, b), c, false);
}

// ------------------------------------------------------------------ the expert GEMVs
//
// Three launches after the input quantization (fp4_moe below):
//   gate_up   gate and up rows of every expert in one grid (blockIdx.y = 2 j + matrix)
//   act_quant swiglu x routing weight, then act_quant into paired order (one pass)
//   down_sum  down rows of every expert, summed over the experts in a fixed order (j = 0..)
// Weights are decoded 4 bytes at a time (e2m1x8_to_half2x4); activations stay fp8 codes and
// are converted per row (fp16 activations doubled the cache traffic and were slower).

// one 32-block: 16 code bytes (32 e2m1) against 32 paired fp8 activations (byte 4q + t meets
// activation pair 4q + t)
__device__ inline float block_dot(uint4 w, uint4 x0, uint4 x1) {
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
  for (int q = 0; q < 4; q++) {
    uint32_t hw[4];
    e2m1x8_to_half2x4(wv[q], hw);
#pragma unroll
    for (int t = 0; t < 4; t++) p = dot2(hw[t], hx[4 * q + t], p);
  }
  return p;
}

// a row's dot product over its K / 32 blocks, lanes l, l + LPR, ... of LPR, reduced over them
template <int LPR>
__device__ inline float row_dot(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ ws, int nb,
                                const uint8_t* __restrict__ xr, const float* __restrict__ xsr, int l) {
  float acc = 0.0f;
  for (int b = l; b < nb; b += LPR) {
    const uint4 w = *(const uint4*)(wc + (size_t)b * 16);
    const uint4 x0 = *(const uint4*)(xr + (size_t)b * 32), x1 = *(const uint4*)(xr + (size_t)b * 32 + 16);
    // 2^(e - 127) x 2^22 (undoing 2^-14 x 2^-8) x the activation scale
    const float wscale = __uint_as_float((uint32_t)ws[b] << 23) * 4194304.0f;
    acc += block_dot(w, x0, x1) * (wscale * xsr[b]);
  }
#pragma unroll
  for (int o = LPR / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
  return acc;
}

// Token t of a chunk (blockIdx.z / .y below) uses sel.ids/w/count + t * sel.stride: one launch
// for every token's experts, each token computed exactly as alone.
template <int LPR>
__global__ void __launch_bounds__(256) gate_up_kernel(GpuMatrix gate, GpuMatrix up, ExpertSel sel, int k,
                                                      const uint8_t* __restrict__ xq, const float* __restrict__ xs,
                                                      float* __restrict__ gu) {
  const int j = blockIdx.y >> 1, t = blockIdx.z;
  const int* count = sel.count ? sel.count + (size_t)t * sel.stride : nullptr;
  if (count && j >= *count) return;  // the whole block: no barriers below
  const GpuMatrix& m = (blockIdx.y & 1) ? up : gate;
  const int N = m.N, nb = m.K / 32, lane = threadIdx.x & 63, l = lane % LPR, E = m.K;
  const int row = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int r = row < N ? row : N - 1;
  const int e = sel.ids[(size_t)t * sel.stride + j];
  const float v = row_dot<LPR>(m.codes + (size_t)e * m.codes_stride + (size_t)r * (m.K / 2),
                               m.scales + (size_t)e * m.scales_stride + (size_t)r * nb, nb, xq + (size_t)t * E,
                               xs + (size_t)t * (E / 32), l);
  if (row < N && l == 0) gu[((size_t)t * 2 * k + blockIdx.y) * N + row] = v;
}

// act = w_j * silu(min(gate, lim)) * clamp(up, +-lim), then act_quant(32, ue8m0) into paired
// order (as quant_paired_kernel); experts past the count write zeros (never read)
__global__ void act_quant_kernel(const float* __restrict__ gu, ExpertSel sel, int k, float lim, int F, int n,
                                 uint8_t* __restrict__ q, float* __restrict__ sc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = i / (k * F), i2 = i - t * k * F, j = i2 / F, f = i2 - j * F;
  const int* count = sel.count ? sel.count + (size_t)t * sel.stride : nullptr;
  float v = 0.0f;
  if (i < n && (!count || j < *count)) {
    const float* g2 = gu + (size_t)t * 2 * k * F;
    const float g = fminf(g2[(size_t)(2 * j) * F + f], lim), u = fminf(fmaxf(g2[(size_t)(2 * j + 1) * F + f], -lim), lim);
    v = sel.w[(size_t)t * sel.stride + j] * (g / (1.0f + expf(-g)) * u);  // the routing weight before w2, as the reference
  }
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  const float s = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  if (i < n) {
    const int b = i & 31;
    q[(i & ~31) + (b < 16 ? 2 * b : 2 * (b - 16) + 1)] = e4m3_code(e4m3_round(fminf(fmaxf(v / s, -448.0f), 448.0f)));
    if (b == 0) sc[i / 32] = s;
  }
}

template <int LPR>
__global__ void __launch_bounds__(256) down_sum_kernel(GpuMatrix down, ExpertSel sel, int k,
                                                       const uint8_t* __restrict__ aq, const float* __restrict__ as,
                                                       float* __restrict__ y) {
  const int N = down.N, F = down.K, nb = F / 32, lane = threadIdx.x & 63, l = lane % LPR, t = blockIdx.y;
  const int row = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int r = row < N ? row : N - 1;
  const int n = sel.count ? sel.count[(size_t)t * sel.stride] : k;
  const uint8_t* aqt = aq + (size_t)t * k * F;
  const float* ast = as + (size_t)t * k * nb;
  float v = 0.0f;
  for (int j = 0; j < n; j++) {  // expert order, as the CPU kernel
    const int e = sel.ids[(size_t)t * sel.stride + j];
    v += row_dot<LPR>(down.codes + (size_t)e * down.codes_stride + (size_t)r * (F / 2),
                      down.scales + (size_t)e * down.scales_stride + (size_t)r * nb, nb, aqt + (size_t)j * F,
                      ast + (size_t)j * nb, l);
  }
  if (row < N && l == 0) y[(size_t)t * N + row] = v;
}

// lanes per row, from a sweep of 8/16/32/64 x 8/16/32/64 (bench-expert-pool, sclk 2): gate/up
// 32, down 16 are best for 1-3 experts (72/109/150 us) and within 3% of the best for 6
constexpr int kLprGateUp = 32, kLprDown = 16;

size_t fp4_moe_scratch_bytes(int E, int F, int k, int M) {
  return (size_t)M * ((size_t)E + (size_t)E / 32 * 4 + (size_t)k * F * 4 * 2 + (size_t)k * F + (size_t)k * F / 32 * 4) + 256 * 5;
}

void fp4_moe_chunk(const GpuExperts& ex, const float* x, ExpertSel sel, int k, int M, float limit, float* y, void* scratch,
                   hipStream_t s) {
  const int E = ex.gate.K, F = ex.gate.N;
  char* p = (char*)scratch;
  auto take = [&](size_t bytes) { char* r = p; p += (bytes + 255) / 256 * 256; return r; };
  uint8_t* xq = (uint8_t*)take((size_t)M * E);
  float* xs = (float*)take((size_t)M * E / 32 * 4);
  float* gu = (float*)take((size_t)M * k * F * 4 * 2);
  uint8_t* aq = (uint8_t*)take((size_t)M * k * F);
  float* asc = (float*)take((size_t)M * k * F / 32 * 4);
  quant_paired_kernel<<<cdiv(M * E, 256), 256, 0, s>>>(x, xq, xs, M * E);  // per 32-block: rows never mix
  // grids sized for k (the counts, if any, are only known on the device)
  constexpr int GU = kLprGateUp, DN = kLprDown;
  gate_up_kernel<GU><<<dim3(cdiv(F, 4 * 64 / GU), 2 * k, M), 256, 0, s>>>(ex.gate, ex.up, sel, k, xq, xs, gu);
  act_quant_kernel<<<cdiv(M * k * F, 256), 256, 0, s>>>(gu, sel, k, limit, F, M * k * F, aq, asc);
  down_sum_kernel<DN><<<dim3(cdiv(E, 4 * 64 / DN), M), 256, 0, s>>>(ex.down, sel, k, aq, asc, y);
}

// ------------------------------------------------------------------ placement-invariant (canon)
//
// canon_dot.h's specification: exact integer block sums (v_dot4_i32_i8: weights 2 w as signed
// bytes, activations as three signed 7-bit digits of 2^9 x), float(S) times the block's scale,
// 16 lanes per row (lane l: blocks l, l + 16, ...), then xor shuffles 8, 4, 2, 1. __fmul_rn /
// __fadd_rn keep the compiler from fusing what the CPU computes as a multiply and an add.

// act_quant per 32-block, then 2^9 x split into digits (natural element order) and c = s 2^-10
__global__ void canon_quant_kernel(const float* x, int8_t* d0, int8_t* d1, int8_t* d2, float* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float v = i < n ? x[i] : 0.0f;
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  const float sc = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  if (i < n) {
    const int x9 = (int)(e4m3_round(fminf(fmaxf(v / sc, -448.0f), 448.0f)) * 512.0f);
    const int m = abs(x9), sg = x9 < 0 ? -1 : 1;
    d0[i] = (int8_t)(sg * (m & 127));
    d1[i] = (int8_t)(sg * ((m >> 7) & 127));
    d2[i] = (int8_t)(sg * (m >> 14));
    if ((i & 31) == 0) c[i / 32] = sc * 0x1p-10f;
  }
}

// 4 e2m1 codes (one per byte, low nibble) -> 4 signed bytes 2 w
__device__ inline int canon_w4(uint32_t n) {
  const uint32_t sel = n & 0x07070707u;
  const uint32_t pos = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);  // 0 1 2 3 4 6 8 12
  const uint32_t neg = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);  // their negatives
  const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
  return (int)((neg & m) | (pos & ~m));
}

// one block's 32 activations as digits: [digit][low 16 / high 16 elements][4 x 4 bytes]
struct CanonX { int v[3][2][4]; };
__device__ inline CanonX canon_load(const int8_t* d0, const int8_t* d1, const int8_t* d2) {
  CanonX x;
  const int8_t* d[3] = {d0, d1, d2};
#pragma unroll
  for (int k = 0; k < 3; k++)
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int4 t = *(const int4*)(d[k] + 16 * h);
      x.v[k][h][0] = t.x; x.v[k][h][1] = t.y; x.v[k][h][2] = t.z; x.v[k][h][3] = t.w;
    }
  return x;
}

// one block's exact sum: 16 code bytes against the digits of its 32 activations
__device__ inline int canon_block(uint4 w, const CanonX& x) {
  const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
  const int* a0 = x.v[0][0]; const int* b0 = x.v[0][1];
  const int* a1 = x.v[1][0]; const int* b1 = x.v[1][1];
  const int* a2 = x.v[2][0]; const int* b2 = x.v[2][1];
  int P0 = 0, P1 = 0, P2 = 0;
#pragma unroll
  for (int g = 0; g < 4; g++) {
    const int wl = canon_w4(wv[g] & 0x0F0F0F0Fu), wh = canon_w4((wv[g] >> 4) & 0x0F0F0F0Fu);
    P0 = __builtin_amdgcn_sdot4(wl, a0[g], P0, false);
    P0 = __builtin_amdgcn_sdot4(wh, b0[g], P0, false);
    P1 = __builtin_amdgcn_sdot4(wl, a1[g], P1, false);
    P1 = __builtin_amdgcn_sdot4(wh, b1[g], P1, false);
    P2 = __builtin_amdgcn_sdot4(wl, a2[g], P2, false);
    P2 = __builtin_amdgcn_sdot4(wh, b2[g], P2, false);
  }
  return P0 + (P1 << 7) + (P2 << 14);
}

// R consecutive rows (codes stride K/2, scales stride nb), lane l (of 16 per row): blocks l,
// l + 16, ...; then the xor tree. The R rows share each activation load; every row's
// arithmetic is the one-row specification.
template <int R>
__device__ inline void canon_rows(const uint8_t* __restrict__ wc, const uint8_t* __restrict__ ws, int nb,
                                  const int8_t* d0, const int8_t* d1, const int8_t* d2, const float* c, int l,
                                  float* out) {
  float acc[R];
#pragma unroll
  for (int i = 0; i < R; i++) acc[i] = 0.0f;
  for (int b = l; b < nb; b += 16) {
    const CanonX x = canon_load(d0 + b * 32, d1 + b * 32, d2 + b * 32);
#pragma unroll
    for (int i = 0; i < R; i++) {
      const int S = canon_block(*(const uint4*)(wc + (size_t)i * nb * 16 + (size_t)b * 16), x);
      const float scale = __fmul_rn(__uint_as_float((uint32_t)ws[(size_t)i * nb + b] << 23), c[b]);
      acc[i] = __fadd_rn(acc[i], __fmul_rn((float)S, scale));
    }
  }
#pragma unroll
  for (int i = 0; i < R; i++) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) acc[i] = __fadd_rn(acc[i], __shfl_xor(acc[i], o, 16));
    out[i] = acc[i];
  }
}
constexpr int kCanonR = 2;  // rows per lane group

struct CanonActDev { int8_t *d0, *d1, *d2; float* c; };

// Chunks: token t = blockIdx.z uses sel's entries j < count_t (expert ids / slots, weights). Its
// rows land at (base_t + j), base_t = the counts of the tokens before it: the rows are compact.
__device__ inline int canon_count(const ExpertSel& sel, int k, int t) { return sel.count ? sel.count[t * sel.stride] : k; }
__device__ inline int canon_base(const ExpertSel& sel, int k, int t) {
  int b = 0;
  for (int u = 0; u < t; u++) b += canon_count(sel, k, u);
  return b;
}

__global__ void __launch_bounds__(256) canon_gate_up_kernel(GpuMatrix gate, GpuMatrix up, ExpertSel sel, int k, CanonActDev x,
                                                            float* __restrict__ gu) {
  const int j = blockIdx.y >> 1, t = blockIdx.z;
  if (j >= canon_count(sel, k, t)) return;
  const GpuMatrix& m = (blockIdx.y & 1) ? up : gate;
  const int N = m.N, nb = m.K / 32, lane = threadIdx.x & 63, l = lane % 16;
  const int row = ((blockIdx.x * 4 + (threadIdx.x >> 6)) * 4 + lane / 16) * kCanonR;  // N % kCanonR == 0
  const int r = row < N ? row : N - kCanonR;
  const int e = sel.ids[t * sel.stride + j];
  const size_t xo = (size_t)t * m.K;
  float v[kCanonR];
  canon_rows<kCanonR>(m.codes + (size_t)e * m.codes_stride + (size_t)r * (m.K / 2),
                      m.scales + (size_t)e * m.scales_stride + (size_t)r * nb, nb, x.d0 + xo, x.d1 + xo, x.d2 + xo,
                      x.c + xo / 32, l, v);
  if (row < N && l == 0)
    for (int i = 0; i < kCanonR; i++) gu[((size_t)t * 2 * k + blockIdx.y) * N + row + i] = v[i];
}

// w_j * silu(min(gate, lim)) * clamp(up, +-lim) (canon_math.h), then act_quant into digits;
// i over [M][k][F] (unused entries are computed from garbage and never read)
__global__ void canon_act_kernel(const float* __restrict__ gu, ExpertSel sel, int k, float lim, int F, int n, CanonActDev a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tj = i / F, f = i - tj * F, t = tj / k, j = tj - t * k;
  float v = 0.0f;
  if (i < n && j < canon_count(sel, k, t)) {
    const float g = fminf(gu[(size_t)(2 * tj) * F + f], lim), u = fminf(fmaxf(gu[(size_t)(2 * tj + 1) * F + f], -lim), lim);
    v = __fmul_rn(sel.w[t * sel.stride + j], __fmul_rn(canon_silu(g), u));
  }
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  const float sc = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  if (i < n) {
    const int x9 = (int)(e4m3_round(fminf(fmaxf(v / sc, -448.0f), 448.0f)) * 512.0f);
    const int m = abs(x9), sg = x9 < 0 ? -1 : 1;
    a.d0[i] = (int8_t)(sg * (m & 127));
    a.d1[i] = (int8_t)(sg * ((m >> 7) & 127));
    a.d2[i] = (int8_t)(sg * (m >> 14));
    if ((i & 31) == 0) a.c[i / 32] = sc * 0x1p-10f;
  }
}

// one expert's down rows: rows[base_t + j] (no sum)
__global__ void __launch_bounds__(256) canon_down_kernel(GpuMatrix down, ExpertSel sel, int k, CanonActDev a,
                                                         float* __restrict__ rows) {
  const int j = blockIdx.y, t = blockIdx.z;
  if (j >= canon_count(sel, k, t)) return;
  const int N = down.N, F = down.K, nb = F / 32, lane = threadIdx.x & 63, l = lane % 16;
  const int row = ((blockIdx.x * 4 + (threadIdx.x >> 6)) * 4 + lane / 16) * kCanonR;
  const int r = row < N ? row : N - kCanonR;
  const int e = sel.ids[t * sel.stride + j];
  const size_t ao = ((size_t)t * k + j) * F;
  float v[kCanonR];
  canon_rows<kCanonR>(down.codes + (size_t)e * down.codes_stride + (size_t)r * (F / 2),
                      down.scales + (size_t)e * down.scales_stride + (size_t)r * nb, nb, a.d0 + ao, a.d1 + ao, a.d2 + ao,
                      a.c + ao / 32, l, v);
  if (row < N && l == 0) {
    float* y = rows + (size_t)(canon_base(sel, k, t) + j) * N;
    for (int i = 0; i < kCanonR; i++) y[row + i] = v[i];
  }
}

// y = ((0 + rows[0]) + rows[1]) + ... (expert order, as CpuMoeKernel::forward_canon)
__global__ void canon_sum_kernel(const float* rows, int k, int E, float* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= E) return;
  float v = 0.0f;
  for (int j = 0; j < k; j++) v = __fadd_rn(v, rows[(size_t)j * E + i]);
  y[i] = v;
}

size_t fp4_moe_canon_rows_scratch_bytes(int E, int F, int k, int M) {
  const size_t mk = (size_t)M * k;
  return (size_t)M * E * 3 + (size_t)M * E / 32 * 4 + mk * F * 4 * 2 + mk * F * 3 + mk * F / 32 * 4 + 256 * 9;
}

size_t fp4_moe_canon_scratch_bytes(int E, int F, int k) {
  return ((size_t)k * E * 4 + 255) / 256 * 256 + fp4_moe_canon_rows_scratch_bytes(E, F, k, 1);
}

void fp4_moe_canon_rows(const GpuExperts& ex, const float* x, ExpertSel sel, int k, int M, float limit, float* rows,
                        void* scratch, hipStream_t s) {
  const int E = ex.gate.K, F = ex.gate.N;
  char* p = (char*)scratch;
  auto take = [&](size_t bytes) { char* r = p; p += (bytes + 255) / 256 * 256; return r; };
  const size_t mk = (size_t)M * k;
  CanonActDev xa = {(int8_t*)take((size_t)M * E), (int8_t*)take((size_t)M * E), (int8_t*)take((size_t)M * E),
                    (float*)take((size_t)M * E / 32 * 4)};
  float* gu = (float*)take(mk * F * 4 * 2);
  CanonActDev aa = {(int8_t*)take(mk * F), (int8_t*)take(mk * F), (int8_t*)take(mk * F), (float*)take(mk * F / 32 * 4)};
  canon_quant_kernel<<<cdiv(M * E, 256), 256, 0, s>>>(x, xa.d0, xa.d1, xa.d2, xa.c, M * E);
  canon_gate_up_kernel<<<dim3(cdiv(F, 16 * kCanonR), 2 * k, M), 256, 0, s>>>(ex.gate, ex.up, sel, k, xa, gu);
  canon_act_kernel<<<cdiv((int)(mk * F), 256), 256, 0, s>>>(gu, sel, k, limit, F, (int)(mk * F), aa);
  canon_down_kernel<<<dim3(cdiv(E, 16 * kCanonR), k, M), 256, 0, s>>>(ex.down, sel, k, aa, rows);
}

void fp4_moe_canon(const GpuExperts& ex, const float* x, const int* ids, const float* w, int k, float limit, float* y,
                   void* scratch, hipStream_t s) {
  const int E = ex.gate.K;
  float* rows = (float*)scratch;  // [k][E], then the rows call's scratch
  char* rest = (char*)scratch + ((size_t)k * E * 4 + 255) / 256 * 256;
  fp4_moe_canon_rows(ex, x, ExpertSel{ids, w, nullptr, 0}, k, 1, limit, rows, rest, s);
  canon_sum_kernel<<<cdiv(E, 256), 256, 0, s>>>(rows, k, E, y);
}

void fp4_moe(const GpuExperts& ex, const float* x, const int* ids, const float* w, int k, float limit, float* y,
             void* scratch, hipStream_t s, const int* count) {
  fp4_moe_chunk(ex, x, ExpertSel{ids, w, count, 0}, k, 1, limit, y, scratch, s);
}
