#include "canon_dot.h"

#include "fakequant.h"

#include <immintrin.h>

#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int QB = 32, BLK = 17;

// 2 * e2m1 value by code (bit 3 = sign)
constexpr int kW2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

inline float e8m0(uint8_t e) {
  const uint32_t b = (uint32_t)e << 23;
  float f;
  memcpy(&f, &b, 4);
  return f;
}

}  // namespace

void canon_act(const float* x, int n, CanonAct& a) {
  canon_act_size(n, a);
  canon_act_blocks(x, 0, a.nb, a);
}

void canon_act_size(int n, CanonAct& a) {
  a.nb = n / QB;
  a.d0.resize(n);
  a.d1.resize(n);
  a.d2.resize(n);
  a.x9.resize(n);
  a.corr.resize(a.nb);
  a.c.resize(a.nb);
}

// the specification (README step 80: canon_act_blocks below must give the same bytes)
void canon_act_blocks_ref(const float* x, int b0, int b1, CanonAct& a) {
  for (int b = b0; b < b1; b++) {
    float q[QB];
    const float s = fq_fp8_split(x + b * QB, q);
    a.c[b] = s * 0x1p-10f;
    int32_t sum = 0;
    for (int j = 0; j < QB; j++) {
      const int i = b * QB + j;
      const int32_t v = (int32_t)(q[j] * 512.0f);  // exact: e4m3 values are multiples of 2^-9
      const int32_t m = v < 0 ? -v : v, sg = v < 0 ? -1 : 1;
      a.x9[i] = v;
      a.d0[i] = (int8_t)(sg * (m & 127));
      a.d1[i] = (int8_t)(sg * ((m >> 7) & 127));
      a.d2[i] = (int8_t)(sg * (m >> 14));
      sum += v;
    }
    a.corr[b] = 12 * sum;
  }
}

// AVX-512, the same bytes as canon_act_blocks_ref (README step 80): per block of 32 (two registers)
//   s   = pow2_ceil(max(amax, 1e-4) / 448): the next power of two at or above (exponent bits)
//   y   = x / s, as x * (1 / s): s is a power of two, so both give the correctly rounded x 2^-k
//   q   = e4m3 round of y: step 2^(binade - 3), binade = max(floor(log2 |y|), -6), ties to even
//   v   = q * 512 (exact), its three signed 7-bit digits, corr = 12 sum v, c = s 2^-10
void canon_act_blocks(const float* x, int b0, int b1, CanonAct& a) {
  const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
  const __m512 lim = _mm512_set1_ps(448.0f), nlim = _mm512_set1_ps(-448.0f);
  const __m512i m127 = _mm512_set1_epi32(127), emin = _mm512_set1_epi32(-6);
  for (int b = b0; b < b1; b++) {
    const float* xb = x + (size_t)b * QB;
    const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
    // pow2_ceil of a positive normal t: t itself if its mantissa is zero, else the next binade
    const float t = std::max(amax, 1e-4f) * (1.0f / 448.0f);
    uint32_t tb;
    memcpy(&tb, &t, 4);
    const uint32_t sb = (tb & 0x007FFFFFu) ? ((tb & 0x7F800000u) + 0x00800000u) : tb;
    float sc, inv;
    memcpy(&sc, &sb, 4);
    const uint32_t ib = (uint32_t)(254 - (int)(sb >> 23)) << 23;  // 1 / s, exact
    memcpy(&inv, &ib, 4);
    a.c[b] = sc * 0x1p-10f;
    __m512i vsum = _mm512_setzero_si512();
    for (int h = 0; h < 2; h++) {
      const __m512 y = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(h ? x1 : x0, _mm512_set1_ps(inv)), nlim), lim);
      const __m512 ay = _mm512_and_ps(y, absmask);
      // binade from the exponent bits (subnormals and zero give <= -127, clamped to -6)
      const __m512i ex = _mm512_max_epi32(_mm512_sub_epi32(_mm512_srli_epi32(_mm512_castps_si512(ay), 23), m127), emin);
      // y / step and step: 2^(3 - binade) and 2^(binade - 3) as exponent bits
      const __m512 up = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_sub_epi32(_mm512_set1_epi32(130), ex), 23));
      const __m512 step = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ex, _mm512_set1_epi32(124)), 23));
      __m512 r = _mm512_mul_ps(_mm512_roundscale_ps(_mm512_mul_ps(ay, up), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), step);
      r = _mm512_min_ps(r, lim);
      const __m512 q = _mm512_or_ps(r, _mm512_andnot_ps(absmask, y));  // the sign of y (copysign)
      const __m512i v = _mm512_cvttps_epi32(_mm512_mul_ps(q, _mm512_set1_ps(512.0f)));  // exact
      const __m512i m = _mm512_abs_epi32(v);
      const __mmask16 neg = _mm512_cmplt_epi32_mask(v, _mm512_setzero_si512());
      auto sgn = [&](__m512i d) { return _mm512_mask_sub_epi32(d, neg, _mm512_setzero_si512(), d); };
      const __m512i e0 = sgn(_mm512_and_si512(m, m127)), e1 = sgn(_mm512_and_si512(_mm512_srli_epi32(m, 7), m127)),
                    e2 = sgn(_mm512_srli_epi32(m, 14));
      const size_t i = (size_t)b * QB + 16 * h;
      _mm512_storeu_si512(a.x9.data() + i, v);
      _mm_storeu_si128((__m128i*)(a.d0.data() + i), _mm512_cvtepi32_epi8(e0));
      _mm_storeu_si128((__m128i*)(a.d1.data() + i), _mm512_cvtepi32_epi8(e1));
      _mm_storeu_si128((__m128i*)(a.d2.data() + i), _mm512_cvtepi32_epi8(e2));
      vsum = _mm512_add_epi32(vsum, v);
    }
    a.corr[b] = 12 * _mm512_reduce_add_epi32(vsum);
  }
}

float canon_dot_ref(const uint8_t* row, const CanonAct& a) {
  float acc[16] = {};
  for (int b = 0; b < a.nb; b++) {
    const uint8_t* blk = row + (size_t)b * BLK;
    int64_t S = 0;
    for (int j = 0; j < 16; j++) {
      S += (int64_t)kW2[blk[1 + j] & 15] * a.x9[b * QB + j];
      S += (int64_t)kW2[blk[1 + j] >> 4] * a.x9[b * QB + 16 + j];
    }
    const float v = (float)(int32_t)S * (e8m0(blk[0]) * a.c[b]);
    acc[b % 16] += v;
  }
  for (int o = 8; o > 0; o >>= 1) {
    float t[16];
    for (int l = 0; l < 16; l++) t[l] = acc[l] + acc[l ^ o];
    memcpy(acc, t, sizeof acc);
  }
  return acc[0];
}

// R rows at a time, 16 blocks per step. Two blocks per register: 128-bit lanes [low nibbles of
// b, high nibbles of b, low of b + 1, high of b + 1], decoded to 2 w + 12 (unsigned bytes) by a
// table; the activation digits are in natural order, so byte i of both registers is the same
// element (loaded once for the R rows). vpdpbusd sums 4 products per 32-bit lane: a block's sum
// is spread over 8 lanes, collected by an integer transpose-reduce of 8 registers (order-free:
// integers). Rows are prefetched into L2 16 KB ahead, as in cpu_moe.cpp.
namespace {

constexpr int kPrefetch = 16384;

struct Consts {
  __m512i lut, sh, low4, gidx, unperm, xor_idx[4];
  Consts() {
    lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    sh = _mm512_setr_epi64(0, 0, 4, 4, 0, 0, 4, 4);
    low4 = _mm512_set1_epi8(0x0F);
    gidx = _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(BLK));
    // the transpose-reduce leaves blocks in the order 0 2 4 6 1 3 5 7 8 10 12 14 9 11 13 15
    unperm = _mm512_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15);
    for (int i = 0, o = 8; o > 0; i++, o >>= 1) {
      alignas(64) int32_t t[16];
      for (int l = 0; l < 16; l++) t[l] = l ^ o;
      xor_idx[i] = _mm512_load_si512(t);
    }
  }
};

inline __m512i reduce8(const __m512i* T) {
  __m512i U[4], V[2];
  for (int k = 0; k < 4; k++)
    U[k] = _mm512_add_epi32(_mm512_unpacklo_epi32(T[2 * k], T[2 * k + 1]), _mm512_unpackhi_epi32(T[2 * k], T[2 * k + 1]));
  for (int k = 0; k < 2; k++)
    V[k] = _mm512_add_epi32(_mm512_unpacklo_epi64(U[2 * k], U[2 * k + 1]), _mm512_unpackhi_epi64(U[2 * k], U[2 * k + 1]));
  return _mm512_add_epi32(_mm512_shuffle_i32x4(V[0], V[1], _MM_SHUFFLE(2, 0, 2, 0)),
                          _mm512_shuffle_i32x4(V[0], V[1], _MM_SHUFFLE(3, 1, 3, 1)));
}

template <int R>
inline void dot_rows_r(const Consts& K, const uint8_t* rows, size_t row_bytes, const CanonAct& a, float* y) {
  const int nb = a.nb;
  __m512 acc[R];
  for (int i = 0; i < R; i++) acc[i] = _mm512_setzero_ps();
  for (int b0 = 0; b0 < nb; b0 += 16) {
    // the last group may have fewer than 16 blocks (an even number: down has 72): its other
    // lanes are left alone, as in the specification
    const int nk = nb - b0 >= 16 ? 8 : (nb - b0) / 2;
    const __mmask16 live = nk == 8 ? (__mmask16)0xFFFF : (__mmask16)((1u << (2 * nk)) - 1);
    __m512i T[R][8];
    for (int k = 0; k < 8; k++) {
      if (k >= nk) {
        for (int i = 0; i < R; i++) T[i][k] = _mm512_setzero_si512();
        continue;
      }
      const int b = b0 + 2 * k;
      const size_t off = (size_t)b * QB;
      const __m512i x2 = _mm512_loadu_si512(a.d2.data() + off), x1 = _mm512_loadu_si512(a.d1.data() + off),
                    x0 = _mm512_loadu_si512(a.d0.data() + off);
      for (int i = 0; i < R; i++) {
        const uint8_t* row = rows + i * row_bytes;
        if ((k & 1) == 0) _mm_prefetch((const char*)(row + (size_t)b * BLK + kPrefetch), _MM_HINT_T1);
        const __m128i c0 = _mm_loadu_si128((const __m128i*)(row + (size_t)b * BLK + 1));
        const __m128i c1 = _mm_loadu_si128((const __m128i*)(row + (size_t)(b + 1) * BLK + 1));
        const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_broadcastsi128_si256(c0)),
                                             _mm256_broadcastsi128_si256(c1), 1);
        const __m512i w = _mm512_shuffle_epi8(K.lut, _mm512_and_si512(_mm512_srlv_epi64(v, K.sh), K.low4));
        __m512i t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x2);
        t = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, x1);
        T[i][k] = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, x0);
      }
    }
    const __m512i corr = _mm512_maskz_loadu_epi32(live, a.corr.data() + b0);
    const __m512 c = _mm512_maskz_loadu_ps(live, a.c.data() + b0);
    for (int i = 0; i < R; i++) {
      const uint8_t* row = rows + i * row_bytes;
      const __m512i S = _mm512_sub_epi32(_mm512_permutexvar_epi32(K.unperm, reduce8(T[i])), corr);
      // the 16 scale bytes (17 apart); the gather reads 4 bytes each, so the last one reads 3
      // bytes past its block (inside the row's next block or the next row)
      const __m512i e = _mm512_and_si512(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(), live, K.gidx, row + (size_t)b0 * BLK, 1),
                                         _mm512_set1_epi32(0xFF));
      const __m512 scale = _mm512_mul_ps(_mm512_castsi512_ps(_mm512_slli_epi32(e, 23)), c);
      acc[i] = _mm512_mask_add_ps(acc[i], live, acc[i], _mm512_mul_ps(_mm512_cvtepi32_ps(S), scale));
    }
  }
  for (int i = 0; i < R; i++) {
    __m512 v = acc[i];
    for (int j = 0; j < 4; j++) v = _mm512_add_ps(v, _mm512_permutexvar_ps(K.xor_idx[j], v));
    y[i] = _mm512_cvtss_f32(v);
  }
}

}  // namespace

void canon_dot_rows(const uint8_t* rows, size_t row_bytes, const CanonAct& a, int r0, int r1, float* y) {
  static const Consts K;
  int r = r0;
  // 4 rows at a time (bench-cpu-moe, 16 threads: 1 row +13%, 2 rows +9%, 4 rows +4% over cpu_moe's
  // kernel)
  for (; r + 4 <= r1; r += 4) dot_rows_r<4>(K, rows + (size_t)r * row_bytes, row_bytes, a, y + r);
  for (; r < r1; r++) dot_rows_r<1>(K, rows + (size_t)r * row_bytes, row_bytes, a, y + r);
}

// R rows against nx activations: per 16-block group the rows' decoded weights go to a small L1
// buffer (they do not fit in registers next to the R x 8 partial sums), and each activation's
// digits are loaded once for the R rows. Per (row, activation) exactly dot_rows_r's steps.
template <int R>
inline void dot_rows_tile(const Consts& K, const uint8_t* rows, size_t row_bytes, const CanonAct* const* a, int nx,
                          float* const* y, int r, __m512* acc) {
  const int nb = a[0]->nb;
  alignas(64) __m512i wb[R][8];
  for (int j = 0; j < nx * R; j++) acc[j] = _mm512_setzero_ps();
  for (int b0 = 0; b0 < nb; b0 += 16) {
    const int nk = nb - b0 >= 16 ? 8 : (nb - b0) / 2;
    const __mmask16 live = nk == 8 ? (__mmask16)0xFFFF : (__mmask16)((1u << (2 * nk)) - 1);
    __m512 escale[R];
    for (int i = 0; i < R; i++) {
      const uint8_t* row = rows + i * row_bytes;
      for (int k = 0; k < nk; k++) {
        const int b = b0 + 2 * k;
        if ((k & 1) == 0) _mm_prefetch((const char*)(row + (size_t)b * BLK + kPrefetch), _MM_HINT_T1);
        const __m128i c0 = _mm_loadu_si128((const __m128i*)(row + (size_t)b * BLK + 1));
        const __m128i c1 = _mm_loadu_si128((const __m128i*)(row + (size_t)(b + 1) * BLK + 1));
        const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_broadcastsi128_si256(c0)),
                                             _mm256_broadcastsi128_si256(c1), 1);
        wb[i][k] = _mm512_shuffle_epi8(K.lut, _mm512_and_si512(_mm512_srlv_epi64(v, K.sh), K.low4));
      }
      const __m512i e = _mm512_and_si512(_mm512_mask_i32gather_epi32(_mm512_setzero_si512(), live, K.gidx, row + (size_t)b0 * BLK, 1),
                                         _mm512_set1_epi32(0xFF));
      escale[i] = _mm512_castsi512_ps(_mm512_slli_epi32(e, 23));
    }
    for (int j = 0; j < nx; j++) {
      const CanonAct& aj = *a[j];
      __m512i T[R][8];
      for (int k = 0; k < 8; k++) {
        if (k >= nk) {
          for (int i = 0; i < R; i++) T[i][k] = _mm512_setzero_si512();
          continue;
        }
        const size_t off = (size_t)(b0 + 2 * k) * QB;
        const __m512i d2 = _mm512_loadu_si512(aj.d2.data() + off), d1 = _mm512_loadu_si512(aj.d1.data() + off),
                      d0 = _mm512_loadu_si512(aj.d0.data() + off);
        for (int i = 0; i < R; i++) {
          const __m512i w = _mm512_load_si512(&wb[i][k]);
          __m512i t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, d2);
          t = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, d1);
          T[i][k] = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, d0);
        }
      }
      const __m512i corr = _mm512_maskz_loadu_epi32(live, aj.corr.data() + b0);
      const __m512 c = _mm512_maskz_loadu_ps(live, aj.c.data() + b0);
      for (int i = 0; i < R; i++) {
        const __m512i S = _mm512_sub_epi32(_mm512_permutexvar_epi32(K.unperm, reduce8(T[i])), corr);
        __m512& ac = acc[j * R + i];
        ac = _mm512_mask_add_ps(ac, live, ac, _mm512_mul_ps(_mm512_cvtepi32_ps(S), _mm512_mul_ps(escale[i], c)));
      }
    }
  }
  for (int j = 0; j < nx; j++)
    for (int i = 0; i < R; i++) {
      __m512 v = acc[j * R + i];
      for (int s = 0; s < 4; s++) v = _mm512_add_ps(v, _mm512_permutexvar_ps(K.xor_idx[s], v));
      y[j][r + i] = _mm512_cvtss_f32(v);
    }
}

void canon_dot_rows_multi(const uint8_t* rows, size_t row_bytes, const CanonAct* const* a, int nx, int r0, int r1, float* const* y) {
  static const Consts K;
  // per thread, 64-byte aligned (a std::vector of __m512 does not keep the type's alignment)
  thread_local struct Acc {
    __m512* p = nullptr;
    size_t n = 0;
    ~Acc() { free(p); }
  } acc;
  if (acc.n < (size_t)nx * 2) {
    free(acc.p);
    acc.n = (size_t)nx * 2;
    acc.p = (__m512*)aligned_alloc(64, acc.n * sizeof(__m512));
  }
  // activations in blocks of 8 (their digits for a 16-block group then stay in L1: ~12 KB)
  constexpr int XB = 8;
  int r = r0;
  for (; r + 2 <= r1; r += 2)
    for (int j = 0; j < nx; j += XB) dot_rows_tile<2>(K, rows + (size_t)r * row_bytes, row_bytes, a + j, std::min(XB, nx - j), y + j, r, acc.p);
  for (; r < r1; r++)
    for (int j = 0; j < nx; j += XB) dot_rows_tile<1>(K, rows + (size_t)r * row_bytes, row_bytes, a + j, std::min(XB, nx - j), y + j, r, acc.p);
}
