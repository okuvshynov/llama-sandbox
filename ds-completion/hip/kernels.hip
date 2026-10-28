#include "kernels.h"

#include <stdexcept>
#include <string>

#include <hip/hip_fp16.h>

#include <cfloat>

// ================================================================== helpers

__device__ inline float warp_sum(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
  return v;
}

__device__ inline float warp_max(float v) {
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 64));
  return v;
}

// sum over the block (blockDim.x a multiple of 64, <= 1024); every thread gets the result
__device__ inline float block_sum(float v) {
  __shared__ float red[16];
  __shared__ float total;
  v = warp_sum(v);
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  if (l == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0;
    for (int i = 0; i < (int)(blockDim.x >> 6); i++) t += red[i];
    total = t;
  }
  __syncthreads();
  const float r = total;
  __syncthreads();
  return r;
}

__device__ inline float block_max(float v) {
  __shared__ float red[16];
  __shared__ float total;
  v = warp_max(v);
  const int w = threadIdx.x >> 6, l = threadIdx.x & 63;
  if (l == 0) red[w] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = -FLT_MAX;
    for (int i = 0; i < (int)(blockDim.x >> 6); i++) t = fmaxf(t, red[i]);
    total = t;
  }
  __syncthreads();
  const float r = total;
  __syncthreads();
  return r;
}

__device__ inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// the fake-quant rounding rules, identical to src/fakequant.cpp (ties to even, saturating)
__device__ inline float e4m3_round(float v) {
  const float a = fminf(fabsf(v), 448.0f);
  if (a == 0.0f) return copysignf(0.0f, v);
  int e;
  frexpf(a, &e);
  const int binade = max(e - 1, -6);
  const float q = ldexpf(1.0f, binade - 3);
  return copysignf(fminf(rintf(a / q) * q, 448.0f), v);
}

__device__ inline float e2m1_round(float v) {
  const float a = fminf(fabsf(v), 6.0f);
  const float q = a < 2.0f ? 0.5f : a < 4.0f ? 1.0f : 2.0f;
  return copysignf(rintf(a / q) * q, v);
}

__device__ inline float pow2_ceil(float v) {
  int e;
  const float m = frexpf(v, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ================================================================== norms, fake quantization

__global__ void rms_norm_kernel(const float* x, const float* w, float* y, int n, float eps) {
  const float* xr = x + (size_t)blockIdx.x * n;
  float* yr = y + (size_t)blockIdx.x * n;
  float ss = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
  const float r = rsqrtf(block_sum(ss) / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
}

void k_rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, hipStream_t s) {
  rms_norm_kernel<<<rows, 256, 0, s>>>(x, w, y, n, eps);
}

template <int B, bool E4M3_SCALE>
__global__ void fq4_kernel(float* x, int n_blocks) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_blocks) return;
  float* xb = x + (size_t)b * B;
  float amax = 0;
  for (int j = 0; j < B; j++) amax = fmaxf(amax, fabsf(xb[j]));
  float sc;
  if (E4M3_SCALE) sc = e4m3_round(fmaxf(amax, 6.0f * 0x1p-9f) / 6.0f);
  else sc = pow2_ceil(fmaxf(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
  for (int j = 0; j < B; j++) xb[j] = e2m1_round(fminf(fmaxf(xb[j] / sc, -6.0f), 6.0f)) * sc;
}

void k_fq4(float* x, int n, bool e4m3_scale_16, hipStream_t s) {
  if (e4m3_scale_16) fq4_kernel<16, true><<<cdiv(n / 16, 256), 256, 0, s>>>(x, n / 16);
  else fq4_kernel<32, false><<<cdiv(n / 32, 256), 256, 0, s>>>(x, n / 32);
}

// ================================================================== matrix-vector

// one wave per (row, K chunk); partial[split][row], then a reduction over the splits
template <typename T>
__device__ inline float load_dot(const T* w, const float* x, int k);

template <>
__device__ inline float load_dot<float>(const float* w, const float* x, int k) {
  const float4 a = *(const float4*)(w + k), b = *(const float4*)(x + k);
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template <>
__device__ inline float load_dot<uint16_t>(const uint16_t* w, const float* x, int k) {  // 8 bf16
  const uint4 a = *(const uint4*)(w + k);
  const float4 b0 = *(const float4*)(x + k), b1 = *(const float4*)(x + k + 4);
  auto lo = [](uint32_t v) { return __uint_as_float(v << 16); };
  auto hi = [](uint32_t v) { return __uint_as_float(v & 0xFFFF0000u); };
  return lo(a.x) * b0.x + hi(a.x) * b0.y + lo(a.y) * b0.z + hi(a.y) * b0.w + lo(a.z) * b1.x + hi(a.z) * b1.y +
         lo(a.w) * b1.z + hi(a.w) * b1.w;
}

template <typename T, int VEC>
__global__ void gemv_partial_kernel(const T* W, int N, int K, int chunk, const float* x, float* partial) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
  if (row >= N) return;  // whole waves exit together
  const int k0 = blockIdx.y * chunk, k1 = min(K, k0 + chunk);
  const T* wr = W + (size_t)row * K;
  float acc = 0;
  for (int k = k0 + lane * VEC; k < k1; k += 64 * VEC) acc += load_dot<T>(wr, x, k);
  acc = warp_sum(acc);
  if (lane == 0) partial[(size_t)blockIdx.y * N + row] = acc;
}

__global__ void split_sum_kernel(const float* partial, int N, int splits, float* y) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float v = 0;
  for (int s = 0; s < splits; s++) v += partial[(size_t)s * N + n];
  y[n] = v;
}

template <typename T, int VEC>
static void gemv(const T* W, int N, int K, const float* x, float* y, float* partial, hipStream_t s) {
  // enough (row, chunk) waves to fill the GPU; chunks a multiple of one wave step
  int splits = 1;
  while (splits < kSplitMax && (int64_t)N * splits < 2048 && K / (splits * 2) >= 64 * VEC * 2) splits *= 2;
  const int step = 64 * VEC, chunk = cdiv(cdiv(K, splits), step) * step;
  splits = cdiv(K, chunk);
  gemv_partial_kernel<T, VEC><<<dim3(cdiv(N, 4), splits), 256, 0, s>>>(W, N, K, chunk, x, partial);
  split_sum_kernel<<<cdiv(N, 256), 256, 0, s>>>(partial, N, splits, y);
}

// M inputs: the same rows, splits and lane order as gemv_partial_kernel, one accumulator per input
template <typename T, int VEC, int M>
__global__ void gemv_partial_m_kernel(const T* W, int N, int K, int chunk, const float* x, float* partial) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 6), lane = threadIdx.x & 63;
  if (row >= N) return;
  const int k0 = blockIdx.y * chunk, k1 = min(K, k0 + chunk);
  const T* wr = W + (size_t)row * K;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0;
  for (int k = k0 + lane * VEC; k < k1; k += 64 * VEC)
#pragma unroll
    for (int m = 0; m < M; m++) acc[m] += load_dot<T>(wr, x + (size_t)m * K, k);
#pragma unroll
  for (int m = 0; m < M; m++) {
    const float v = warp_sum(acc[m]);
    if (lane == 0) partial[((size_t)m * gridDim.y + blockIdx.y) * N + row] = v;
  }
}

__global__ void split_sum_m_kernel(const float* partial, int N, int splits, float* y) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (n >= N) return;
  float v = 0;
  for (int s = 0; s < splits; s++) v += partial[((size_t)m * splits + s) * N + n];
  y[(size_t)m * N + n] = v;
}

template <typename T, int VEC>
static void gemv_m(const T* W, int N, int K, const float* x, int M, float* y, float* partial, hipStream_t s) {
  int splits = 1;  // exactly as gemv()
  while (splits < kSplitMax && (int64_t)N * splits < 2048 && K / (splits * 2) >= 64 * VEC * 2) splits *= 2;
  const int step = 64 * VEC, chunk = cdiv(cdiv(K, splits), step) * step;
  splits = cdiv(K, chunk);
  const dim3 grid(cdiv(N, 4), splits);
  switch (M) {
#define CASE(mm) case mm: gemv_partial_m_kernel<T, VEC, mm><<<grid, 256, 0, s>>>(W, N, K, chunk, x, partial); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    default: throw std::runtime_error("gemv_m: M = " + std::to_string(M));
  }
  split_sum_m_kernel<<<dim3(cdiv(N, 256), M), 256, 0, s>>>(partial, N, splits, y);
}

void k_gemv_f32_m(const float* W, int N, int K, const float* x, int M, float* y, float* partial, hipStream_t s) {
  gemv_m<float, 4>(W, N, K, x, M, y, partial, s);
}
void k_gemv_bf16_m(const uint16_t* W, int N, int K, const float* x, int M, float* y, float* partial, hipStream_t s) {
  gemv_m<uint16_t, 8>(W, N, K, x, M, y, partial, s);
}

void k_gemv_f32(const float* W, int N, int K, const float* x, float* y, float* partial, hipStream_t s) {
  gemv<float, 4>(W, N, K, x, y, partial, s);
}
void k_gemv_bf16(const uint16_t* W, int N, int K, const float* x, float* y, float* partial, hipStream_t s) {
  gemv<uint16_t, 8>(W, N, K, x, y, partial, s);
}

// ================================================================== rope

__global__ void rope_kernel(float* x, int n_heads, int head_dim, int rope_dim, const float* freqs, int pos, bool inverse) {
  const int half = rope_dim / 2;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_heads * half) return;
  const int h = t / half, i = t % half;
  float* p = x + (size_t)h * head_dim + (head_dim - rope_dim) + 2 * i;
  const float ang = (float)pos * freqs[i];
  const float c = cosf(ang), sn = inverse ? -sinf(ang) : sinf(ang);
  const float a = p[0], b = p[1];
  p[0] = a * c - b * sn;
  p[1] = a * sn + b * c;
}

void k_rope(float* x, int n_heads, int head_dim, int rope_dim, const float* freqs, int pos, bool inverse, hipStream_t s) {
  const int n = n_heads * rope_dim / 2;
  rope_kernel<<<cdiv(n, 256), 256, 0, s>>>(x, n_heads, head_dim, rope_dim, freqs, pos, inverse);
}

// ================================================================== hyper-connections

// The 4x4 Sinkhorn runs on 16 lanes of wave 0 (lane = j * 4 + k: row j = source copy, column
// k = destination copy); row sums are shuffles over lanes j*4 .. j*4+3, column sums over k, k+4, ..
__global__ void hc_split_kernel(const float* h, int n_flat, const float* mixes, const float* scale, const float* base,
                                float norm_eps, float hc_eps, int iters, float* pre, float* post, float* comb) {
  constexpr int HC = 4;
  float ss = 0;
  for (int i = threadIdx.x; i < n_flat; i += blockDim.x) ss += h[i] * h[i];
  const float r = rsqrtf(block_sum(ss) / n_flat + norm_eps);
  const int t = threadIdx.x;
  if (t >= 64) return;
  if (t < HC) pre[t] = sigmoidf(mixes[t] * r * scale[0] + base[t]) + hc_eps;
  else if (t < 2 * HC) post[t - HC] = 2.0f * sigmoidf(mixes[t] * r * scale[1] + base[t]);
  const int e = t & 15;  // 16 lanes per matrix entry; lanes 16-63 compute copies and do not store
  float c = mixes[2 * HC + e] * r * scale[2] + base[2 * HC + e];
  auto row_sum = [&](float v) { v += __shfl_xor(v, 1, 64); v += __shfl_xor(v, 2, 64); return v; };
  auto col_sum = [&](float v) { v += __shfl_xor(v, 4, 64); v += __shfl_xor(v, 8, 64); return v; };
  float mx = fmaxf(c, __shfl_xor(c, 1, 64));
  mx = fmaxf(mx, __shfl_xor(mx, 2, 64));
  c = expf(c - mx);
  c = c / row_sum(c) + hc_eps;     // softmax over the destination, + eps
  c = c / (col_sum(c) + hc_eps);   // then normalize over the source
  for (int it = 1; it < iters; it++) {
    c = c / (row_sum(c) + hc_eps);
    c = c / (col_sum(c) + hc_eps);
  }
  if (t < 16) comb[e] = c;
}

void k_hc_split(const float* h, int n_flat, const float* mixes, const float* scale, const float* base, float norm_eps,
                float hc_eps, int iters, float* pre, float* post, float* comb, hipStream_t s) {
  hc_split_kernel<<<1, 1024, 0, s>>>(h, n_flat, mixes, scale, base, norm_eps, hc_eps, iters, pre, post, comb);
}

__global__ void hc_pre_kernel(const float* h, const float* pre, const float* w, int n, float eps, float* x, bool norm) {
  __shared__ float buf[8192];
  float ss = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = 0;
    for (int j = 0; j < 4; j++) v += pre[j] * h[(size_t)j * n + i];
    buf[i] = v;
    ss += v * v;
  }
  const float r = norm ? rsqrtf(block_sum(ss) / n + eps) : 1.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] = buf[i] * r * (w ? w[i] : 1.0f);
}

void k_hc_pre_norm(const float* h, const float* pre, const float* w, int n, float eps, float* x, hipStream_t s) {
  hc_pre_kernel<<<1, 1024, 0, s>>>(h, pre, w, n, eps, x, true);
}
void k_hc_pre(const float* h, const float* pre, int n, float* x, hipStream_t s) {
  hc_pre_kernel<<<1, 1024, 0, s>>>(h, pre, nullptr, n, 0.0f, x, false);
}

__global__ void hc_post_kernel(const float* x, const float* h, const float* post, const float* comb, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int k = 0; k < 4; k++) {
    float v = x[i] * post[k];
    for (int j = 0; j < 4; j++) v += comb[j * 4 + k] * h[(size_t)j * n + i];
    out[(size_t)k * n + i] = v;
  }
}

void k_hc_post(const float* x, const float* h, const float* post, const float* comb, int n, float* h_out, hipStream_t s) {
  hc_post_kernel<<<cdiv(n, 256), 256, 0, s>>>(x, h, post, comb, n, h_out);
}

// ================================================================== attention

// Split-K decode attention ("flash-decoding"). Block (g, c) takes HPB heads and key chunk c
// (CHUNK keys of: the window ring slots, then the compressed rows the indexer picked); it
// writes its unnormalized partial output with the chunk's max and sum. The combine kernel
// merges the chunks and adds the per-head sink to the denominator (sparse_attn).
constexpr int ATT_HPB = 8, ATT_CHUNK = 32;

__device__ inline const float* att_key(int j, const float* ring, int window, int pos, const float* comp, const int* topk,
                                       int head_dim, bool& valid) {
  if (j < window) {
    const int ps = pos - (((pos - j) % window) + window) % window;  // the position slot j holds
    valid = ps >= 0;
    return ring + (size_t)j * head_dim;
  }
  valid = true;
  return comp + (size_t)topk[j - window] * head_dim;
}

// head_dim == 512. Thread t of 256 makes the score of head t / 32 against key t % 32 of the
// chunk: the keys pass through shared memory in 64-dim tiles (coalesced float4 loads, a padded
// row stride so the 32 keys of a head read different banks), q is in shared memory once. No
// cross-lane sums in the score pass; the softmax per head is one 32-lane group's shuffles.
// P @ V: each thread owns two dims (float2 per key), 8 keys in flight.
__global__ void __launch_bounds__(256) attn_partial_kernel(const float* q, int head_dim, const float* ring, int window,
                                                           int pos, const float* comp, const int* topk, int nk,
                                                           float scale, float* part_o, float* part_ml) {
  constexpr int D = 512, DC = 64, VB = 8;
  __shared__ float qs[ATT_HPB][D];
  __shared__ float kt[ATT_CHUNK][DC + 1];
  __shared__ float sc[ATT_HPB][ATT_CHUNK];
  __shared__ const float* kp[ATT_CHUNK];
  __shared__ bool kvalid[ATT_CHUNK];
  const int h0 = blockIdx.x * ATT_HPB, c = blockIdx.y, j0 = c * ATT_CHUNK;
  const int n = min(ATT_CHUNK, nk - j0);
  for (int i = threadIdx.x; i < ATT_HPB * D / 4; i += blockDim.x)
    ((float4*)&qs[0][0])[i] = ((const float4*)(q + (size_t)h0 * D))[i];
  if (threadIdx.x < ATT_CHUNK) {
    bool v = false;
    const float* p = ring;
    if ((int)threadIdx.x < n) p = att_key(j0 + threadIdx.x, ring, window, pos, comp, topk, D, v);
    kp[threadIdx.x] = p;
    kvalid[threadIdx.x] = v;
  }
  __syncthreads();
  const int hh = threadIdx.x / ATT_CHUNK, jj = threadIdx.x % ATT_CHUNK;
  float dot = 0.0f;
  for (int d0 = 0; d0 < D; d0 += DC) {
    // the 32 x 64 key tile: 512 float4, two per thread
    for (int i = threadIdx.x; i < ATT_CHUNK * DC / 4; i += blockDim.x) {
      const int k = i / (DC / 4), d4 = i % (DC / 4);
      const float4 v = ((const float4*)(kp[k] + d0))[d4];
      kt[k][4 * d4 + 0] = v.x; kt[k][4 * d4 + 1] = v.y; kt[k][4 * d4 + 2] = v.z; kt[k][4 * d4 + 3] = v.w;
    }
    __syncthreads();
#pragma unroll 16
    for (int d = 0; d < DC; d++) dot += qs[hh][d0 + d] * kt[jj][d];
    __syncthreads();
  }
  // softmax pieces per head: the 32 threads of head hh are one aligned 32-lane group
  float v = jj < n && kvalid[jj] ? dot * scale : -INFINITY;
  float mx = v;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
  mx = fmaxf(mx, -1e30f);  // the reference kernel's finite floor
  const float e = expf(v - mx);
  float sum = e;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
  sc[hh][jj] = e;
  if (jj == 0) {
    part_ml[((size_t)c * gridDim.x * ATT_HPB + h0 + hh) * 2 + 0] = mx;
    part_ml[((size_t)c * gridDim.x * ATT_HPB + h0 + hh) * 2 + 1] = sum;
  }
  __syncthreads();
  const int d2 = threadIdx.x;  // dims 2 d2, 2 d2 + 1 (256 threads x 2 = 512)
  float2 acc[ATT_HPB];
#pragma unroll
  for (int h = 0; h < ATT_HPB; h++) acc[h] = make_float2(0.0f, 0.0f);
  for (int jb = 0; jb < n; jb += VB) {
    float2 kv[VB];
#pragma unroll
    for (int u = 0; u < VB; u++) kv[u] = ((const float2*)kp[min(jb + u, n - 1)])[d2];
#pragma unroll
    for (int u = 0; u < VB; u++) {
      if (jb + u >= n) break;
#pragma unroll
      for (int h = 0; h < ATT_HPB; h++) {
        const float p = sc[h][jb + u];
        acc[h].x += p * kv[u].x;
        acc[h].y += p * kv[u].y;
      }
    }
  }
#pragma unroll
  for (int h = 0; h < ATT_HPB; h++)
    ((float2*)(part_o + ((size_t)c * gridDim.x * ATT_HPB + h0 + h) * D))[d2] = acc[h];
}

__global__ void attn_combine_kernel(const float* part_o, const float* part_ml, int n_chunks, int n_head, int head_dim,
                                    const float* sinks, float* o) {
  const int h = blockIdx.x;
  float M = -1e30f;  // the reference kernel's finite floor
  for (int c = 0; c < n_chunks; c++) M = fmaxf(M, part_ml[((size_t)c * n_head + h) * 2]);
  float den = expf(sinks[h] - M);
  for (int c = 0; c < n_chunks; c++)
    den += part_ml[((size_t)c * n_head + h) * 2 + 1] * expf(part_ml[((size_t)c * n_head + h) * 2] - M);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0;
    for (int c = 0; c < n_chunks; c++)
      acc += part_o[((size_t)c * n_head + h) * head_dim + d] * expf(part_ml[((size_t)c * n_head + h) * 2] - M);
    o[(size_t)h * head_dim + d] = acc / den;
  }
}

void k_attn_decode(const float* q, int n_head, int head_dim, const float* ring, int window, int pos, const float* comp,
                   const int* topk, int K, const float* sinks, float scale, float* o, float* scratch, hipStream_t s) {
  const int nk = window + K, n_chunks = cdiv(nk, ATT_CHUNK);
  float* part_o = scratch;                                         // [n_chunks][n_head][head_dim]
  float* part_ml = scratch + (size_t)n_chunks * n_head * head_dim;  // [n_chunks][n_head][2]
  if (head_dim != 512) abort();  // the partial kernel is written for the model's head_dim
  attn_partial_kernel<<<dim3(n_head / ATT_HPB, n_chunks), 256, 0, s>>>(q, head_dim, ring, window, pos, comp, topk, nk,
                                                                       scale, part_o, part_ml);
  attn_combine_kernel<<<n_head, 256, 0, s>>>(part_o, part_ml, n_chunks, n_head, head_dim, sinks, o);
}

// ---- the window KV of one token: rms * w, rope on the tail, fp8 fake-quant, into its ring slot
__global__ void kv_finish_kernel(const float* kv, const float* w, int dim, int rope_dim, const float* freqs, int pos,
                                 float eps, float* slot) {
  __shared__ float v[1024];
  float ss = 0;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += kv[i] * kv[i];
  const float r = rsqrtf(block_sum(ss) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) v[i] = kv[i] * r * w[i];
  __syncthreads();
  const int half = rope_dim / 2, base = dim - rope_dim;
  if (threadIdx.x < half) {
    const int i = threadIdx.x;
    const float ang = (float)pos * freqs[i], c = cosf(ang), sn = sinf(ang);
    const float a = v[base + 2 * i], b = v[base + 2 * i + 1];
    v[base + 2 * i] = a * c - b * sn;
    v[base + 2 * i + 1] = a * sn + b * c;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {  // 32 consecutive threads = one fp8 block
    const float x = v[i];
    float amax = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    const float sc = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    slot[i] = e4m3_round(fminf(fmaxf(x / sc, -448.0f), 448.0f)) * sc;
  }
}

void k_kv_finish(const float* kv, const float* w, int dim, int rope_dim, const float* freqs, int pos, float eps,
                 float* slot, hipStream_t s) {
  kv_finish_kernel<<<1, 512, 0, s>>>(kv, w, dim, rope_dim, freqs, pos, eps, slot);
}

// ================================================================== indexer

// A block scores 64 keys: the tile of keys goes to shared memory (coalesced), q is stored
// transposed (qT[dim][head]) so that 32 lanes = 32 heads read consecutive words; lane h of a
// 32-lane group makes head h's dot product for one key, then the group sums relu(.) * w.
__global__ void index_scores_kernel(const float* q, int n_head, int dim, const float* keys, int n, const float* w,
                                    float wscale, float* score) {
  constexpr int TILE = 64, DIM = 128, HEADS = 32;
  __shared__ float qT[DIM][HEADS + 1];
  __shared__ float kt[TILE][DIM + 1];
  const int c0 = blockIdx.x * TILE;
  for (int i = threadIdx.x; i < HEADS * DIM; i += blockDim.x) {
    const int hh = i / DIM, d = i % DIM;
    qT[d][hh] = hh < n_head ? q[hh * DIM + d] : 0.0f;
  }
  for (int i = threadIdx.x; i < TILE * DIM; i += blockDim.x) {
    const int k = i / DIM, d = i % DIM;
    kt[k][d] = c0 + k < n ? keys[(size_t)(c0 + k) * DIM + d] : 0.0f;
  }
  __syncthreads();
  const int h = threadIdx.x & 31, grp = threadIdx.x >> 5, n_grp = blockDim.x >> 5;
  const float wh = h < n_head ? w[h] * wscale : 0.0f;
  for (int k = grp; k < TILE; k += n_grp) {
    float d = 0;
#pragma unroll 16
    for (int i = 0; i < DIM; i++) d += qT[i][h] * kt[k][i];
    float v = fmaxf(d, 0.0f) * wh;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if (h == 0 && c0 + k < n) score[c0 + k] = v;
  }
}

void k_index_scores(const float* q, int n_head, int dim, const float* keys, int n, const float* w, float wscale,
                    float* score, hipStream_t s) {
  if (dim != 128 || n_head > 32) abort();  // written for the model's indexer shape
  index_scores_kernel<<<cdiv(n, 64), 256, 0, s>>>(q, n_head, dim, keys, n, w, wscale, score);
}

// bitonic sort of (score, index) pairs, descending, in shared memory; then the first k indices
__global__ void topk_kernel(const float* score, int n, int k, int* idx) {
  __shared__ float sv[kTopkMaxN];
  __shared__ int si[kTopkMaxN];
  int P = 1;
  while (P < n) P <<= 1;
  for (int i = threadIdx.x; i < P; i += blockDim.x) {
    sv[i] = i < n ? score[i] : -INFINITY;
    si[i] = i;
  }
  __syncthreads();
  for (int size = 2; size <= P; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const float a = sv[i], b = sv[j];
          if (desc ? a < b : a > b) {
            sv[i] = b; sv[j] = a;
            const int t = si[i]; si[i] = si[j]; si[j] = t;
          }
        }
      }
      __syncthreads();
    }
  for (int i = threadIdx.x; i < k; i += blockDim.x) idx[i] = si[i];
}

void k_topk(const float* score, int n, int k, int* idx, hipStream_t s) { topk_kernel<<<1, 1024, 0, s>>>(score, n, k, idx); }

// ================================================================== compressor

__global__ void compress_pool_kernel(const float* kv, const float* sc, int ratio, int dim, float* out) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  float mx = -INFINITY;
  for (int i = 0; i < ratio; i++) mx = fmaxf(mx, sc[(size_t)i * dim + d]);
  float num = 0, den = 0;
  for (int i = 0; i < ratio; i++) {
    const float p = expf(sc[(size_t)i * dim + d] - mx);
    num += kv[(size_t)i * dim + d] * p;
    den += p;
  }
  out[d] = num / den;
}

void k_compress_pool(const float* kv_state, const float* score_state, int ratio, int dim, float* pooled, hipStream_t s) {
  compress_pool_kernel<<<cdiv(dim, 256), 256, 0, s>>>(kv_state, score_state, ratio, dim, pooled);
}

// ================================================================== MoE

// one wave: every lane holds n/64 experts; K rounds of wave argmax (ties to the lower index).
// K and the per-lane count are compile-time so every array stays in registers.
template <int K>
__global__ void router_kernel(const float* logits, const float* bias, int n, bool norm, float route_scale, int* ids,
                              float* w) {
  constexpr int PER = 16;  // n <= 1024
  const int lane = threadIdx.x;
  float sc[PER], sel[PER];
#pragma unroll
  for (int i = 0; i < PER; i++) {
    const int e = lane + 64 * i;
    const float x = e < n ? logits[e] : 0.0f;
    sc[i] = sqrtf(x > 20.0f ? x : log1pf(expf(x)));  // torch softplus (threshold 20)
    sel[i] = e < n ? sc[i] + bias[e] : -INFINITY;
  }
  float total = 0.0f, my_w = 0.0f;
  int my_id = 0;
#pragma unroll
  for (int r = 0; r < K; r++) {
    float best = -INFINITY, bsc = 0.0f;
    int bi = n;
#pragma unroll
    for (int i = 0; i < PER; i++)
      if (sel[i] > best) { best = sel[i]; bi = lane + 64 * i; bsc = sc[i]; }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) {
      const float ob = __shfl_xor(best, o, 64), os = __shfl_xor(bsc, o, 64);
      const int oi = __shfl_xor(bi, o, 64);
      if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; bsc = os; }
    }
#pragma unroll
    for (int i = 0; i < PER; i++)
      if (lane + 64 * i == bi) sel[i] = -INFINITY;  // the owner retires it
    if (lane == r) { my_id = bi; my_w = bsc; }       // lane r keeps pick r
    total += bsc;
  }
  if (lane < K) {
    float v = my_w;
    if (norm && K > 1) v /= total + 1e-20f;
    ids[lane] = my_id;
    w[lane] = v * route_scale;
  }
}

void k_router(const float* logits, const float* bias, int n_expert, int k, bool norm, float route_scale, int* ids,
              float* w, hipStream_t s) {
  switch (k) {
    case 3: router_kernel<3><<<1, 64, 0, s>>>(logits, bias, n_expert, norm, route_scale, ids, w); break;  // DSpark draft
    case 6: router_kernel<6><<<1, 64, 0, s>>>(logits, bias, n_expert, norm, route_scale, ids, w); break;
    case 8: router_kernel<8><<<1, 64, 0, s>>>(logits, bias, n_expert, norm, route_scale, ids, w); break;
    default: throw std::runtime_error("k_router: no instance for k = " + std::to_string(k));
  }
}

__global__ void swiglu_kernel(const float* gate, const float* up, float lim, int n, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = fminf(gate[i], lim), u = fminf(fmaxf(up[i], -lim), lim);
  out[i] = g / (1.0f + expf(-g)) * u;
}

void k_swiglu(const float* gate, const float* up, float lim, int n, float* out, hipStream_t s) {
  swiglu_kernel<<<cdiv(n, 256), 256, 0, s>>>(gate, up, lim, n, out);
}

// ================================================================== engram

// one block per hc copy j: gate_j = sigmoid(signed sqrt(<h_j, w_j * key_j> / (rms h_j rms key_j sqrt n)))
__global__ void engram_kernel(float* h, const float* kv, const float* gate_w, int hc, int n, float eps) {
  const int j = blockIdx.x;
  float* hj = h + (size_t)j * n;
  const float* key = kv + (size_t)j * n;
  const float* val = kv + (size_t)hc * n;
  const float* gw = gate_w + (size_t)j * n;
  float sh = 0, sk = 0, dot = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    sh += hj[i] * hj[i];
    sk += key[i] * key[i];
    dot += hj[i] * gw[i] * key[i];
  }
  sh = block_sum(sh);
  sk = block_sum(sk);
  dot = block_sum(dot);
  const float rstd = rsqrtf(sh / n + eps) * rsqrtf(sk / n + eps);
  const float d = dot * rstd * rsqrtf((float)n);
  const float g = sigmoidf(copysignf(sqrtf(fmaxf(fabsf(d), 1e-6f)), d));
  for (int i = threadIdx.x; i < n; i += blockDim.x) hj[i] += g * val[i];
}

void k_engram(float* h, const float* kv, const float* gate_w, int hc, int n, float eps, hipStream_t s) {
  engram_kernel<<<hc, 1024, 0, s>>>(h, kv, gate_w, hc, n, eps);
}

// ================================================================== misc


// one thread per token: k is 6
__global__ void moe_dispatch_kernel(const int* ids, const float* w, int k, const int* table, int n_dies, MoeList* lists) {
  const int t = threadIdx.x;
  ids += t * kMoeIdStride;
  w += t * kMoeIdStride;
  lists += t * kMoeListStride;
  for (int d = 0; d <= n_dies; d++) lists[d].count = 0;
  for (int j = 0; j < k; j++) {
    const int code = table ? table[ids[j]] : -1;
    MoeList& l = lists[code < 0 ? n_dies : code >> 16];
    l.ids[l.count] = code < 0 ? ids[j] : (code & 0xFFFF);
    l.w[l.count] = w[j];
    l.count++;
  }
}

void k_moe_dispatch(const int* ids, const float* w, int k, const int* table, int n_dies, MoeList* lists, int M, hipStream_t s) {
  moe_dispatch_kernel<<<1, M, 0, s>>>(ids, w, k, table, n_dies, lists);
}

struct Parts { const float* p[kMoeMaxDies + 1]; };

__global__ void add_parts_kernel(const float* a, Parts parts, int n, float* y, int len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= len) return;
  float v = a[i];
  for (int j = 0; j < n; j++) if (parts.p[j]) v += parts.p[j][i];
  y[i] = v;
}

void k_add_parts(const float* a, const float* const* parts, int n, float* y, int len, hipStream_t s) {
  Parts p = {};
  for (int j = 0; j < n; j++) p.p[j] = parts[j];
  add_parts_kernel<<<cdiv(len, 256), 256, 0, s>>>(a, p, n, y, len);
}

__global__ void repeat_kernel(const float* x, int n, int copies, float* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int j = 0; j < copies; j++) y[(size_t)j * n + i] = x[i];
}
void k_repeat(const float* x, int n, int copies, float* y, hipStream_t s) {
  repeat_kernel<<<cdiv(n, 256), 256, 0, s>>>(x, n, copies, y);
}
