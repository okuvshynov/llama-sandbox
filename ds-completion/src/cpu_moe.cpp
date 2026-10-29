#include "cpu_moe.h"

#include "canon_math.h"
#include "fakequant.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int QB = 32, BLK = 17;  // mxfp4: 32 values in 1 scale byte + 16 code bytes

// e8m0 byte -> 2^(e - 127) (e = 0 gives 0: 2^-127 is not a normal float; no weight uses it)
struct ScaleTable {
  float v[256];
  ScaleTable() { for (int e = 0; e < 256; e++) { uint32_t b = (uint32_t)e << 23; memcpy(&v[e], &b, 4); } }
};
const ScaleTable kScale;

// Lane d of a 16-lane vector takes code c(d) = 4 (d % 4) + d / 4 of a block half: the 16 code
// bytes are broadcast to all four 128-bit lanes (a load, no shuffle port), shifted right per lane
// by 8 (d / 4) (+ 4 for the high nibbles), and vpermps only reads the low 4 bits of an index.
// The activations are stored in the same lane order (permute_blocks), so the dot is unchanged.
inline int lane_code(int d) { return 4 * (d % 4) + d / 4; }

// R consecutive rows (row stride row_bytes) dotted with x (in lane order), for nb blocks of 32
// Software prefetch into L2, this many bytes ahead of each row's position: with the static
// schedule a thread's next task is the adjacent block of rows, so 16 KB ahead is the rows it
// needs next. Measured (results/bench-cpu-moe-*.txt): 79 -> 67 ms per token at 16 threads,
// 69 -> 66 at 32; L1 prefetch, other distances and SMT helper threads did not help.
constexpr int kPrefetch = 16384;

template <int R>
inline void dot_rows(const uint8_t* rows, size_t row_bytes, int nb, const float* x, float* out) {
  // e2m1 values in code order (bit 3 = sign)
  const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f,
                                    -3.0f, -4.0f, -6.0f);
  const __m512i sh_lo = _mm512_setr_epi32(0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 16, 24, 24, 24, 24);
  const __m512i sh_hi = _mm512_add_epi32(sh_lo, _mm512_set1_epi32(4));
  __m512 acc[R];
  for (int i = 0; i < R; i++) acc[i] = _mm512_setzero_ps();
  for (int b = 0; b < nb; b++) {
    const __m512 xl = _mm512_loadu_ps(x + b * QB), xh = _mm512_loadu_ps(x + b * QB + 16);
    if ((b & 3) == 0)  // one prefetch per 64 bytes (~4 blocks) of every row
      for (int i = 0; i < R; i++) _mm_prefetch((const char*)(rows + i * row_bytes + (size_t)b * BLK + kPrefetch), _MM_HINT_T1);
#pragma GCC unroll 8
    for (int i = 0; i < R; i++) {
      const uint8_t* blk = rows + i * row_bytes + (size_t)b * BLK;
      const __m512i q = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(blk + 1)));
      const __m512 wl = _mm512_permutexvar_ps(_mm512_srlv_epi32(q, sh_lo), lut);
      const __m512 wh = _mm512_permutexvar_ps(_mm512_srlv_epi32(q, sh_hi), lut);
      const __m512 t = _mm512_fmadd_ps(wh, xh, _mm512_mul_ps(wl, xl));  // exact products, fp32 sums
      acc[i] = _mm512_fmadd_ps(t, _mm512_set1_ps(kScale.v[blk[0]]), acc[i]);
    }
  }
  for (int i = 0; i < R; i++) out[i] = _mm512_reduce_add_ps(acc[i]);
}

// x (n values, blocks of 32) into the kernel's lane order: [low half permuted, high half permuted]
inline void permute_blocks(const float* x, float* y, int n) {
  for (int b = 0; b < n / QB; b++)
    for (int d = 0; d < 16; d++) {
      y[b * QB + d] = x[b * QB + lane_code(d)];
      y[b * QB + 16 + d] = x[b * QB + 16 + lane_code(d)];
    }
}

// rows [r0, r1) of one expert's matrix (K = nb * 32 inputs) against x
template <int R>
inline void dot_range_r(const uint8_t* base, size_t row_bytes, int nb, const float* x, int r0, int r1, float* y) {
  int r = r0;
  for (; r + R <= r1; r += R) dot_rows<R>(base + (size_t)r * row_bytes, row_bytes, nb, x, y + r);
  for (; r < r1; r++) dot_rows<1>(base + (size_t)r * row_bytes, row_bytes, nb, x, y + r);
}

// 4 rows at a time (1, 2 and 8 measured no better)
inline void dot_range(const uint8_t* base, size_t row_bytes, int nb, const float* x, int r0, int r1, float* y) {
  dot_range_r<4>(base, row_bytes, nb, x, r0, r1, y);
}

const uint8_t* expert_base(const ggml_tensor* t, int e) { return (const uint8_t*)t->data + (size_t)e * t->nb[2]; }

}  // namespace

CpuMoeKernel::CpuMoeKernel(int n_embd, int n_ff, int k_max, int nt) : E(n_embd), F(n_ff), n_threads(nt) {
  xq.resize(2 * E);
  gate_out.resize((size_t)k_max * F);
  up_out.resize((size_t)k_max * F);
  act_q.resize((size_t)k_max * F);
  act_l.resize((size_t)k_max * F);
  down_out.resize((size_t)k_max * E);
  act_c.resize(k_max);
}

void CpuMoeKernel::forward_canon(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* y) {
  forward_canon_rows(L, x, ids, w, k, down_out.data());
  for (int i = 0; i < E; i++) {
    float v = 0.0f;
    for (int j = 0; j < k; j++) v += down_out[(size_t)j * E + i];
    y[i] = v;
  }
}

void CpuMoeKernel::forward_canon_rows(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* rows) {
  constexpr int CH = 32;  // rows per task
  constexpr int AB = 8;   // activation blocks (of 32 values) per task
  canon_act_size(E, x_c);
  for (int j = 0; j < k; j++) canon_act_size(F, act_c[j]);
  const size_t gu_row = L.gate->nb[1], d_row = L.down->nb[1];
  const int gu_tasks = k * 2 * (F / CH), d_tasks = k * (E / CH);
  const int x_tasks = E / QB / AB, a_per = F / QB / AB, a_tasks = k * a_per;  // E / 32 and F / 32 are multiples of 8
#pragma omp parallel num_threads(n_threads)
  {
    // the input's digits, a few blocks per task (one thread for all of it cost ~8% at 1 expert)
#pragma omp for schedule(static)
    for (int t = 0; t < x_tasks; t++) canon_act_blocks(x, t * AB, (t + 1) * AB, x_c);
#pragma omp for schedule(static)
    for (int t = 0; t < gu_tasks; t++) {
      const int per = F / CH, j = t / (2 * per), m = (t / per) % 2, c = t % per;
      const ggml_tensor* W = m ? L.up : L.gate;
      float* out = (m ? up_out.data() : gate_out.data()) + (size_t)j * F;
      canon_dot_rows(expert_base(W, ids[j]), gu_row, x_c, c * CH, (c + 1) * CH, out);
    }
    // the activation and its digits, a few blocks of one expert per task
#pragma omp for schedule(static)
    for (int t = 0; t < a_tasks; t++) {
      const int j = t / a_per, b0 = (t % a_per) * AB;
      float* a = act_q.data() + (size_t)j * F;
      for (int i = b0 * QB; i < (b0 + AB) * QB; i++) {
        const float g = std::min(gate_out[(size_t)j * F + i], L.limit);
        const float u = std::clamp(up_out[(size_t)j * F + i], -L.limit, L.limit);
        a[i] = w[j] * (canon_silu(g) * u);
      }
      canon_act_blocks(a, b0, b0 + AB, act_c[j]);
    }
#pragma omp for schedule(static)
    for (int t = 0; t < d_tasks; t++) {
      const int per = E / CH, j = t / per, c = t % per;
      canon_dot_rows(expert_base(L.down, ids[j]), d_row, act_c[j], c * CH, (c + 1) * CH, rows + (size_t)j * E);
    }
  }
}

void CpuMoeKernel::forward(const CpuExperts& L, const float* x, const int* ids, const float* w, int k, float* y) {
  constexpr int CH = 32;  // rows per task
  fq_fp8_host(x, xq.data(), E);
  permute_blocks(xq.data(), xq.data() + E, E);  // the second half of xq holds the lane-ordered copy
  const float* xl = xq.data() + E;
  const size_t gu_row = L.gate->nb[1], d_row = L.down->nb[1];
  const int nb_e = E / QB, nb_f = F / QB;
  const int gu_tasks = k * 2 * (F / CH), d_tasks = k * (E / CH);
#pragma omp parallel num_threads(n_threads)
  {
    // phase 1: gate and up rows of every routed expert
#pragma omp for schedule(static)
    for (int t = 0; t < gu_tasks; t++) {
      const int per = F / CH, j = t / (2 * per), m = (t / per) % 2, c = t % per;
      const ggml_tensor* W = m ? L.up : L.gate;
      float* out = (m ? up_out.data() : gate_out.data()) + (size_t)j * F;
      dot_range(expert_base(W, ids[j]), gu_row, nb_e, xl, c * CH, (c + 1) * CH, out);
    }
    // swiglu_clamp, the routing weight (before w2, as the reference), then fq8 for w2's input
#pragma omp for schedule(static)
    for (int j = 0; j < k; j++) {
      float* a = act_q.data() + (size_t)j * F;
      for (int i = 0; i < F; i++) {
        const float g = std::min(gate_out[(size_t)j * F + i], L.limit);
        const float u = std::clamp(up_out[(size_t)j * F + i], -L.limit, L.limit);
        a[i] = w[j] * (g / (1.0f + expf(-g)) * u);
      }
      fq_fp8_host(a, a, F);
      permute_blocks(a, act_l.data() + (size_t)j * F, F);
    }
    // phase 2: down rows of every routed expert
#pragma omp for schedule(static)
    for (int t = 0; t < d_tasks; t++) {
      const int per = E / CH, j = t / per, c = t % per;
      dot_range(expert_base(L.down, ids[j]), d_row, nb_f, act_l.data() + (size_t)j * F, c * CH, (c + 1) * CH,
                down_out.data() + (size_t)j * E);
    }
    // the sum over experts, in expert order
#pragma omp for schedule(static)
    for (int i = 0; i < E; i++) {
      float v = 0.0f;
      for (int j = 0; j < k; j++) v += down_out[(size_t)j * E + i];
      y[i] = v;
    }
  }
}

uint64_t CpuMoeKernel::touch(const CpuExperts& L, const int* ids, int k) {
  constexpr int CH = 32;
  const size_t gu_row = L.gate->nb[1], d_row = L.down->nb[1];
  const int gu_tasks = k * 2 * (F / CH), d_tasks = k * (E / CH);
  uint64_t total = 0;
#pragma omp parallel num_threads(n_threads) reduction(^ : total)
  {
    __m512i acc = _mm512_setzero_si512();
    auto rd = [&](const uint8_t* p, size_t n) {
      for (size_t i = 0; i + 64 <= n; i += 64) acc = _mm512_xor_si512(acc, _mm512_loadu_si512(p + i));
    };
#pragma omp for schedule(static)
    for (int t = 0; t < gu_tasks; t++) {
      const int per = F / CH, j = t / (2 * per), m = (t / per) % 2, c = t % per;
      rd(expert_base(m ? L.up : L.gate, ids[j]) + (size_t)c * CH * gu_row, CH * gu_row);
    }
#pragma omp for schedule(static)
    for (int t = 0; t < d_tasks; t++) {
      const int per = E / CH, j = t / per, c = t % per;
      rd(expert_base(L.down, ids[j]) + (size_t)c * CH * d_row, CH * d_row);
    }
    total ^= (uint64_t)_mm512_reduce_add_epi64(acc);
  }
  return total;
}
