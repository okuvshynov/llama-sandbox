#include "fp8_gemv.h"

#include <algorithm>

// ---------------------------------------------------------------- activation fake-quant

// nearest e4m3fn value (ties to even), |v| <= 448 (same rule as src/fakequant.cpp)
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

// an exactly representable e4m3 value -> its code: fp16(v * 2^-8) has the code in bits 14:7
__device__ inline uint8_t e4m3_code(float v) {
  const uint16_t h = __half_as_ushort(__float2half(v * (1.0f / 256.0f)));
  return (uint8_t)(((h >> 7) & 0x7F) | ((h >> 8) & 0x80));
}

// one thread per value, 32 lanes per block of the reference's act_quant(x, 32, "ue8m0"):
// the block's absmax by shuffles, then every lane rounds its own value
__global__ void quant_act_kernel(const float* x, uint8_t* q, float* s, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const float v = i < n ? x[i] : 0.0f;
  float amax = fabsf(v);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  const float sc = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
  if (i < n) {
    q[i] = e4m3_code(e4m3_round(fminf(fmaxf(v / sc, -448.0f), 448.0f)));
    if ((i & 31) == 0) s[i / 32] = sc;
  }
}

void fp8_quant_act(const float* x, uint8_t* q, float* s, int M, int K_total, hipStream_t stream) {
  const int n = M * K_total;
  quant_act_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, q, s, n);
}

// ---------------------------------------------------------------- GEMV

// 4 e4m3 codes -> two half2 = (code0, code1), (code2, code3), each the e4m3 value * 2^-8.
// v_perm places code i in the high byte of its 16-bit half (bits 15:8); the sign is then in
// place and the 7 magnitude bits are one position too high: shift those right by one.
__device__ inline void e4m3x4_to_half2x2(uint32_t v, uint32_t& h01, uint32_t& h23) {
  const uint32_t t01 = __builtin_amdgcn_perm(0, v, 0x010c000c);  // bytes: [0, c0, 0, c1]
  const uint32_t t23 = __builtin_amdgcn_perm(0, v, 0x030c020c);  // bytes: [0, c2, 0, c3]
  h01 = ((t01 >> 1) & 0x3F803F80u) | (t01 & 0x80008000u);
  h23 = ((t23 >> 1) & 0x3F803F80u) | (t23 & 0x80008000u);
}

__device__ inline void decode16(const uint4 v, uint32_t* h) {
  e4m3x4_to_half2x2(v.x, h[0], h[1]);
  e4m3x4_to_half2x2(v.y, h[2], h[3]);
  e4m3x4_to_half2x2(v.z, h[4], h[5]);
  e4m3x4_to_half2x2(v.w, h[6], h[7]);
}

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
  return __builtin_amdgcn_fdot2(__builtin_bit_cast(half2_t, a), __builtin_bit_cast(half2_t, b), c, false);
}

// A wave covers 64 / LPR row sets of R consecutive rows; LPR lanes split K for each set, each
// lane stepping 16 codes at a time. R divides 32, so the R rows share one weight scale row.
template <int LPR, int R, int M>
__global__ void __launch_bounds__(256) fp8_gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                       int N, int K, int rpg, const uint8_t* __restrict__ xq,
                                                       const float* __restrict__ xs, float* __restrict__ y) {
  const int lane = threadIdx.x & 63;
  const int set = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int row0 = set * R;
  const int l = lane % LPR;
  const bool live = row0 < N;
  const int r0 = live ? row0 : N - R;  // dead lanes still take part in the shuffles
  const int g = r0 / rpg;              // R rows never straddle a group (rpg is a multiple of 32)
  const int K_total = K * (N / rpg);
  const uint8_t* wr = w + (size_t)r0 * K;
  const uint8_t* sr = ws + (size_t)(r0 / 32) * (K / 32);
  const uint8_t* xr = xq + (size_t)g * K;
  const float* xsr = xs + (size_t)g * (K / 32);
  float acc[R][M];
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[i][m] = 0.0f;

  for (int k = l * 16; k < K; k += LPR * 16) {
    uint32_t hx[M][8];
    float sx[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
      decode16(*(const uint4*)(xr + (size_t)m * K_total + k), hx[m]);
      sx[m] = xsr[(size_t)m * (K_total / 32) + k / 32];
    }
    const float wscale = __uint_as_float((uint32_t)sr[k / 32] << 23) * 65536.0f;  // 2^(e-127), times the 2^16 undone
#pragma unroll
    for (int i = 0; i < R; i++) {
      uint32_t hw[8];
      decode16(*(const uint4*)(wr + (size_t)i * K + k), hw);
#pragma unroll
      for (int m = 0; m < M; m++) {
        float p = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) p = dot2(hw[j], hx[m][j], p);
        acc[i][m] += p * (wscale * sx[m]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < M; m++) {
      float v = acc[i][m];
#pragma unroll
      for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
      if (live && l == 0) y[(size_t)m * N + row0 + i] = v;
    }
}

// Several inputs (M >= 2). A one-input call is bound by memory; with M inputs the old loop
// was bound by arithmetic, and mostly by decoding: every row set decoded all M inputs again
// (~240 of ~350 instructions per step at M = 6). Here the block decodes each tile of the inputs
// once, into shared memory as fp16 (low and high halves of every 16 codes in two arrays, so a
// lane's 16-byte reads are consecutive across lanes); a lane then decodes its R weight rows and
// reuses each input's fp16 values for all of them.
// Same work split, same per-lane order of k steps and the same per-element expression as
// fp8_gemv_kernel, so every row's result is bit-identical to a one-input call. A block's rows
// lie in one group.
template <int LPR, int R, int M>
__global__ void __launch_bounds__(256) fp8_gemv_lds_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                           int N, int K, int rpg, const uint8_t* __restrict__ xq,
                                                           const float* __restrict__ xs, float* __restrict__ y) {
  constexpr int KT = 1024, ROWS = 4 * (64 / LPR) * R;
  __shared__ uint4 xlo[M][KT / 16], xhi[M][KT / 16];  // fp16 of codes 0-7 / 8-15 of each 16
  __shared__ float st[M][KT / 32];
  const int lane = threadIdx.x & 63;
  const int set = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int row0 = set * R;
  const int l = lane % LPR;
  const bool live = row0 < N;
  const int r0 = live ? row0 : N - R;  // dead lanes still load tiles and take part in the shuffles
  const int g = (blockIdx.x * ROWS) / rpg;
  const int K_total = K * (N / rpg);
  const uint8_t* wr = w + (size_t)r0 * K;
  const uint8_t* sr = ws + (size_t)(r0 / 32) * (K / 32);
  const uint8_t* xg = xq + (size_t)g * K;
  const float* xsg = xs + (size_t)g * (K / 32);
  float acc[R][M];
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[i][m] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += KT) {
    const int kn = min(KT, K - k0);  // a multiple of LPR * 16, as K and KT are
    __syncthreads();                 // the previous tile is consumed
    for (int i = threadIdx.x; i < M * (kn / 16); i += blockDim.x) {
      const int m = i / (kn / 16), j = i % (kn / 16);
      uint32_t h[8];
      decode16(*(const uint4*)(xg + (size_t)m * K_total + k0 + j * 16), h);
      xlo[m][j] = make_uint4(h[0], h[1], h[2], h[3]);
      xhi[m][j] = make_uint4(h[4], h[5], h[6], h[7]);
    }
    for (int i = threadIdx.x; i < M * (kn / 32); i += blockDim.x) {
      const int m = i / (kn / 32), j = i % (kn / 32);
      st[m][j] = xsg[(size_t)m * (K_total / 32) + k0 / 32 + j];
    }
    __syncthreads();
    for (int k = k0 + l * 16; k < k0 + kn; k += LPR * 16) {  // this lane's steps, in order
      const float wscale = __uint_as_float((uint32_t)sr[k / 32] << 23) * 65536.0f;
      uint32_t hw[R][8];
#pragma unroll
      for (int i = 0; i < R; i++) decode16(*(const uint4*)(wr + (size_t)i * K + k), hw[i]);
      const int j = (k - k0) / 16;
#pragma unroll
      for (int m = 0; m < M; m++) {
        const uint4 a = xlo[m][j], b = xhi[m][j];
        const uint32_t hx[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        const float sc = wscale * st[m][(k - k0) / 32];
#pragma unroll
        for (int i = 0; i < R; i++) {
          float p = 0.0f;
#pragma unroll
          for (int e = 0; e < 8; e++) p = dot2(hw[i][e], hx[e], p);
          acc[i][m] += p * sc;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < M; m++) {
      float v = acc[i][m];
#pragma unroll
      for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
      if (live && l == 0) y[(size_t)m * N + row0 + i] = v;
    }
}

// Many inputs (prompt chunks): the same per-element arithmetic as fp8_gemv_kernel for the same
// lanes-per-row split, so every result is bit-identical to a one-input fp8_gemv call. A block
// takes MT inputs (blockIdx.y picks the tile), decodes their K tile into shared memory once, and
// every lane reuses each input's fp16 values for its R weight rows and each decoded weight row
// for the MT inputs. A lane takes the steps that are its own in the specification (k = 16 l mod
// 16 LPR), in increasing order, whatever the K tile.
template <int LPR, int R, int MT>
__global__ void __launch_bounds__(256) fp8_gemm_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                                       int N, int K, int rpg, int M, const uint8_t* __restrict__ xq,
                                                       const float* __restrict__ xs, float* __restrict__ y) {
  // K tile: at least one lane stride, so every lane has a step in every tile
  constexpr int S = LPR * 16, KT = S > 512 ? S : 512, ROWS = 4 * (64 / LPR) * R;
  __shared__ uint4 xlo[MT][KT / 16], xhi[MT][KT / 16];
  __shared__ float st[MT][KT / 32];
  const int lane = threadIdx.x & 63;
  const int set = (blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6)) * (64 / LPR) + lane / LPR;
  const int row0 = set * R;
  const int l = lane % LPR;
  const bool live = row0 < N;
  const int r0 = live ? row0 : N - R;
  const int g = (blockIdx.x * ROWS) / rpg;
  const int K_total = K * (N / rpg);
  const int m0 = blockIdx.y * MT, mc = min(MT, M - m0);
  const uint8_t* wr = w + (size_t)r0 * K;
  const uint8_t* sr = ws + (size_t)(r0 / 32) * (K / 32);
  const uint8_t* xg = xq + (size_t)m0 * K_total + (size_t)g * K;
  const float* xsg = xs + (size_t)m0 * (K_total / 32) + (size_t)g * (K / 32);
  float acc[R][MT];
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < MT; m++) acc[i][m] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += KT) {
    const int kn = min(KT, K - k0);
    __syncthreads();
    for (int i = threadIdx.x; i < MT * (kn / 16); i += blockDim.x) {
      const int m = i / (kn / 16), j = i % (kn / 16);
      uint32_t hh[8] = {};
      if (m < mc) decode16(*(const uint4*)(xg + (size_t)m * K_total + k0 + j * 16), hh);
      xlo[m][j] = make_uint4(hh[0], hh[1], hh[2], hh[3]);
      xhi[m][j] = make_uint4(hh[4], hh[5], hh[6], hh[7]);
    }
    for (int i = threadIdx.x; i < MT * (kn / 32); i += blockDim.x) {
      const int m = i / (kn / 32), j = i % (kn / 32);
      st[m][j] = m < mc ? xsg[(size_t)m * (K_total / 32) + k0 / 32 + j] : 0.0f;
    }
    __syncthreads();
    const int first = k0 + (((l * 16 - k0) % S) + S) % S;  // this lane's first step in the tile
    for (int k = first; k < k0 + kn; k += S) {
      const float wscale = __uint_as_float((uint32_t)sr[k / 32] << 23) * 65536.0f;
      uint32_t hw[R][8];
#pragma unroll
      for (int i = 0; i < R; i++) decode16(*(const uint4*)(wr + (size_t)i * K + k), hw[i]);
      const int j = (k - k0) / 16;
#pragma unroll
      for (int m = 0; m < MT; m++) {
        const uint4 a = xlo[m][j], b = xhi[m][j];
        const uint32_t hx[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        const float sc = wscale * st[m][(k - k0) / 32];
#pragma unroll
        for (int i = 0; i < R; i++) {
          float p = 0.0f;
#pragma unroll
          for (int e = 0; e < 8; e++) p = dot2(hw[i][e], hx[e], p);
          acc[i][m] += p * sc;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R; i++)
#pragma unroll
    for (int m = 0; m < MT; m++) {
      float v = acc[i][m];
#pragma unroll
      for (int o = LPR / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
      if (live && l == 0 && m < mc) y[(size_t)(m0 + m) * N + row0 + i] = v;
    }
}

template <int LPR, int R>
static void launch_r(const uint8_t* w, const uint8_t* ws, int N, int K, int rpg, Fp8Act x, int M, float* y,
                     hipStream_t stream) {
  const int rows_per_block = 4 * (64 / LPR) * R;  // 4 waves
  const dim3 grid((N + rows_per_block - 1) / rows_per_block), block(256);
  // several inputs, rows split over 8 or 16 lanes: through shared memory (results/bench-fp8-multi.txt;
  // with 64 lanes per row a block covers too few rows to pay for decoding the tiles)
  if (M >= 2 && LPR <= 16 && rpg % rows_per_block == 0) {
    switch (M) {
#define CASE(m) case m: fp8_gemv_lds_kernel<LPR, R, m><<<grid, block, 0, stream>>>(w, ws, N, K, rpg, x.q, x.s, y); break;
      CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
    }
    return;
  }
  switch (M) {
#define CASE(m) case m: fp8_gemv_kernel<LPR, R, m><<<grid, block, 0, stream>>>(w, ws, N, K, rpg, x.q, x.s, y); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#undef CASE
  }
}

template <int LPR>
static void launch_lpr(const uint8_t* w, const uint8_t* ws, int N, int K, int rpg, Fp8Act x, int M, float* y,
                       hipStream_t stream, int r) {
  switch (r) {
    case 8: launch_r<LPR, 8>(w, ws, N, K, rpg, x, M, y, stream); break;
    case 4: launch_r<LPR, 4>(w, ws, N, K, rpg, x, M, y, stream); break;
    case 2: launch_r<LPR, 2>(w, ws, N, K, rpg, x, M, y, stream); break;
    default: launch_r<LPR, 1>(w, ws, N, K, rpg, x, M, y, stream); break;
  }
}

// the lanes-per-row split for a shape: it fixes the order of every output's fp32 additions
int fp8_lpr(int N, int K) {
  // from the sweep in results/bench-fp8-sweep.txt: at most ~10 16-byte steps per lane; all 64
  // lanes once K >= 4096
  int lpr = 8;
  while (lpr < 64 && K / (lpr * 16) > 10) lpr *= 2;
  if (K >= 4096) lpr = 64;
  // then keep >= 1024 waves in flight (16 per CU) when the rows are few
  while (lpr < 64 && (int64_t)N * lpr / 64 < 1024 && K % (2 * lpr * 16) == 0) lpr *= 2;
  return lpr;
}

template <int LPR, int R, int MT>
static bool launch_gemm(const uint8_t* w, const uint8_t* ws, int N, int K, int rpg, Fp8Act x, int M, float* y,
                        hipStream_t stream) {
  const int rows_per_block = 4 * (64 / LPR) * R;
  if (rpg % rows_per_block) return false;  // a block's rows must lie in one group
  const dim3 grid((N + rows_per_block - 1) / rows_per_block, (M + MT - 1) / MT), block(256);
  fp8_gemm_kernel<LPR, R, MT><<<grid, block, 0, stream>>>(w, ws, N, K, rpg, M, x.q, x.s, y);
  return true;
}

template <int LPR>
static bool gemm_lpr(const uint8_t* w, const uint8_t* ws, int N, int K, int rpg, Fp8Act x, int M, float* y,
                     hipStream_t stream, int r, int mt) {
#define G(R_, MT_) if (r == R_ && mt == MT_) return launch_gemm<LPR, R_, MT_>(w, ws, N, K, rpg, x, M, y, stream);
  G(2, 8) G(2, 16) G(4, 8) G(4, 16) G(1, 16)
#undef G
  return false;
}

void fp8_gemm_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int r, int mt) {
  const int lpr = fp8_lpr(N, K);
  // built-in choice (results/prefill/bench-fp8-gemm.txt): 2 rows x 8 inputs; 8-lane rows (q_b)
  // gain nothing over the GEMV and stay with it
  if (r == 0 && mt == 0 && lpr == 8) r = -1;
  if (r == 0) r = 2;
  if (mt == 0) mt = 8;
  bool ok = false;
  switch (r < 0 ? 0 : lpr) {
    case 0: break;
    case 64: ok = gemm_lpr<64>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r, mt); break;
    case 32: ok = gemm_lpr<32>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r, mt); break;
    case 16: ok = gemm_lpr<16>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r, mt); break;
    default: ok = gemm_lpr<8>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r, mt); break;
  }
  if (!ok)  // no instance for this split: slices of 8 through the GEMV (the same bits)
    for (int m0 = 0; m0 < M; m0 += 8)
      fp8_gemv(w, w_scale, N, K, rows_per_group, Fp8Act{x.q + (size_t)m0 * K * (N / rows_per_group), x.s + (size_t)m0 * K * (N / rows_per_group) / 32},
               std::min(8, M - m0), y + (size_t)m0 * N, stream);
}

void fp8_gemm(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M, float* y,
              hipStream_t stream) {
  fp8_gemm_cfg(w, w_scale, N, K, rows_per_group, x, M, y, stream, 0, 0);
}

void fp8_gemv_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int lpr, int r) {
  if (lpr == 0) lpr = fp8_lpr(N, K);
  // several inputs: two rows per lane group; four for 8 inputs over 64-lane rows (sweeps at M = 6, 8)
  if (r == 0) r = M == 1 ? 1 : (M >= 8 && lpr == 64 && N >= 2048) ? 4 : 2;
  switch (lpr) {
    case 64: launch_lpr<64>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r); break;
    case 32: launch_lpr<32>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r); break;
    case 16: launch_lpr<16>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r); break;
    default: launch_lpr<8>(w, w_scale, N, K, rows_per_group, x, M, y, stream, r); break;
  }
}

void fp8_gemv(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M, float* y,
              hipStream_t stream) {
  fp8_gemv_cfg(w, w_scale, N, K, rows_per_group, x, M, y, stream, 0, 0);
}
