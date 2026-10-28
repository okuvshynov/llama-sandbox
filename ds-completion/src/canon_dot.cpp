#include "canon_dot.h"

#include "fakequant.h"

#include <immintrin.h>

#include <cmath>
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
  a.nb = n / QB;
  a.d0.resize(n);
  a.d1.resize(n);
  a.d2.resize(n);
  a.x9.resize(n);
  a.corr.resize(a.nb);
  a.c.resize(a.nb);
  for (int b = 0; b < a.nb; b++) {
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
