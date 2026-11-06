// bench-fp4-decode: what the CPU's expert kernel costs per weight once memory is out of the way
// (README step 78). Each thread works on its own 64 rows (K = 5120, the gate / up shape: ~174 KB,
// resident in its L2), over and over, so every variant is a compute ceiling:
//   canon      canon_dot_rows, today's canonical kernel (mxfp4 rows of 17-byte blocks, one token)
//   fast       the fast kernel's dot_rows<4> (copied from cpu_moe.cpp: fp32 products, lane order)
//   decode17   the canonical kernel's decode alone, today's layout: two 17-byte blocks -> 64 int8
//              (2 w + 12) per register (loads, broadcast, insert, shift, and, pshufb)
//   decode64   decode alone in a CPU layout: a row's codes contiguous (64-byte aligned), scales
//              apart; one 64-byte load -> 128 int8 (and, shift, and, two pshufb)
//   arith      the canonical arithmetic alone: 3 vpdpbusd + 2 shifts per 64 weights, the
//              transpose-reduce and the scaling per 16 blocks, weights from a register (no loads)
// Reports G weights/s per core and per clock (the clock measured with a chain of dependent adds),
// at 1 and at 16 threads, and what that rate would make one expert (35.4M weights) on 16 threads.
//   bin/bench-fp4-decode [threads] [K]   (K: 5120, gate / up rows; 2304, down rows)
//   bin/bench-fp4-decode call [threads]   a whole one-token call as forward_batch runs it: gate +
//     up in 32-row tasks, then down in 32-row tasks (canon_dot_rows), on whole experts in today's
//     layout; warm (the same expert every call: L2 + L3) and cold (24 experts in turn, 450 MB:
//     DRAM); the dynamic schedule (the runtime's) and static (contiguous rows per thread).
//     bin/bench-fp4-decode call 16 4k: the experts on 4 KB pages (the model's mapping), else 2 MB
//     bin/bench-fp4-decode call 16 2m 1500: an idle gap of 1500 us before every call (the runtime's GPU
//     part; -1500: the master spins through it instead of sleeping); 'late' is when the last thread entered the call's parallel region, after the call began
#include "canon_dot.h"

#include <immintrin.h>
#include <omp.h>

#include <chrono>
#include <sys/mman.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int QB = 32, BLK = 17, ROWS = 64;
int K = 5120, NB = 5120 / 32;  // the row length: 5120 (gate, up) or 2304 (down)
constexpr double kExpertWeights = 3.0 * 2304 * 5120;  // gate + up + down
volatile uint64_t g_sink;

// GHz of the calling thread: 2^28 dependent 1-cycle adds
double measure_ghz() {
  uint64_t x = 0;
  const auto t0 = Clock::now();
  for (int i = 0; i < (1 << 28); i++) asm volatile("add $1, %0" : "+r"(x));
  const double s = std::chrono::duration<double>(Clock::now() - t0).count();
  g_sink = x;
  return (1 << 28) / s / 1e9;
}

// ---- fast: cpu_moe.cpp's dot_rows<4>, unchanged (without its prefetch: the rows are in L2)
const float kScaleV[2] = {1.0f, 0.5f};
inline void fast_rows4(const uint8_t* rows, size_t row_bytes, int nb, const float* x, float* out) {
  const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f,
                                    -4.0f, -6.0f);
  const __m512i sh_lo = _mm512_setr_epi32(0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 16, 24, 24, 24, 24);
  const __m512i sh_hi = _mm512_add_epi32(sh_lo, _mm512_set1_epi32(4));
  __m512 acc[4];
  for (int i = 0; i < 4; i++) acc[i] = _mm512_setzero_ps();
  for (int b = 0; b < nb; b++) {
    const __m512 xl = _mm512_loadu_ps(x + b * QB), xh = _mm512_loadu_ps(x + b * QB + 16);
#pragma GCC unroll 4
    for (int i = 0; i < 4; i++) {
      const uint8_t* blk = rows + i * row_bytes + (size_t)b * BLK;
      const __m512i q = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)(blk + 1)));
      const __m512 wl = _mm512_permutexvar_ps(_mm512_srlv_epi32(q, sh_lo), lut);
      const __m512 wh = _mm512_permutexvar_ps(_mm512_srlv_epi32(q, sh_hi), lut);
      const __m512 t = _mm512_fmadd_ps(wh, xh, _mm512_mul_ps(wl, xl));
      acc[i] = _mm512_fmadd_ps(t, _mm512_set1_ps(kScaleV[blk[0] & 1]), acc[i]);
    }
  }
  for (int i = 0; i < 4; i++) out[i] = _mm512_reduce_add_ps(acc[i]);
}

// ---- decode17: two blocks of today's layout per register, as canon_dot.cpp decodes them
uint64_t decode17(const uint8_t* rows, int nrows) {
  const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
  const __m512i sh = _mm512_setr_epi64(0, 0, 4, 4, 0, 0, 4, 4), low4 = _mm512_set1_epi8(0x0F);
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
  for (int r = 0; r < nrows; r++) {
    const uint8_t* row = rows + (size_t)r * NB * BLK;
    for (int b = 0; b < NB; b += 2) {
      const __m128i c0 = _mm_loadu_si128((const __m128i*)(row + (size_t)b * BLK + 1));
      const __m128i c1 = _mm_loadu_si128((const __m128i*)(row + (size_t)(b + 1) * BLK + 1));
      const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_broadcastsi128_si256(c0)), _mm256_broadcastsi128_si256(c1), 1);
      const __m512i w = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srlv_epi64(v, sh), low4));
      if (b & 2) acc1 = _mm512_add_epi8(acc1, w); else acc0 = _mm512_add_epi8(acc0, w);
    }
  }
  return (uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi8(acc0, acc1));
}

// ---- decode64: a CPU layout, a row's 2560 code bytes contiguous; byte j of a 64-byte group holds
// element j in its low nibble and element j + 64 in its high one
uint64_t decode64(const uint8_t* codes, int nrows) {
  const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
  const __m512i low4 = _mm512_set1_epi8(0x0F);
  __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
  for (int r = 0; r < nrows; r++) {
    const uint8_t* row = codes + (size_t)r * K / 2;
    for (int g = 0; g < K / 128; g++) {
      const __m512i v = _mm512_load_si512(row + (size_t)g * 64);
      acc0 = _mm512_add_epi8(acc0, _mm512_shuffle_epi8(lut, _mm512_and_si512(v, low4)));
      acc1 = _mm512_add_epi8(acc1, _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v, 4), low4)));
    }
  }
  return (uint64_t)_mm512_reduce_add_epi64(_mm512_add_epi8(acc0, acc1));
}

// ---- arith: the canonical kernel's arithmetic for R = 4 rows with no weight loads: the weights
// are a register stepped by one add per 64 (so the compiler cannot hoist the products)
inline __m512i reduce8(const __m512i* T) {
  __m512i U[4], V[2];
  for (int k = 0; k < 4; k++)
    U[k] = _mm512_add_epi32(_mm512_unpacklo_epi32(T[2 * k], T[2 * k + 1]), _mm512_unpackhi_epi32(T[2 * k], T[2 * k + 1]));
  for (int k = 0; k < 2; k++)
    V[k] = _mm512_add_epi32(_mm512_unpacklo_epi64(U[2 * k], U[2 * k + 1]), _mm512_unpackhi_epi64(U[2 * k], U[2 * k + 1]));
  return _mm512_add_epi32(_mm512_shuffle_i32x4(V[0], V[1], _MM_SHUFFLE(2, 0, 2, 0)), _mm512_shuffle_i32x4(V[0], V[1], _MM_SHUFFLE(3, 1, 3, 1)));
}
float arith4(const int8_t* d0, const int8_t* d1, const int8_t* d2, int nrows) {
  __m512 tot = _mm512_setzero_ps();
  const __m512 c = _mm512_set1_ps(0x1p-20f);
  for (int r = 0; r < nrows; r += 4) {
    __m512 acc[4];
    __m512i w = _mm512_set1_epi8((char)(13 + r));
    for (int i = 0; i < 4; i++) acc[i] = _mm512_setzero_ps();
    for (int b0 = 0; b0 < NB; b0 += 16) {
      __m512i T[4][8];
      const int nk = NB - b0 >= 16 ? 8 : (NB - b0) / 2;  // down: 72 blocks, the last group half full
      for (int k = 0; k < 8; k++) {
        if (k >= nk) { for (int i = 0; i < 4; i++) T[i][k] = _mm512_setzero_si512(); continue; }
        const size_t off = (size_t)(b0 + 2 * k) * QB;
        const __m512i x2 = _mm512_loadu_si512(d2 + off), x1 = _mm512_loadu_si512(d1 + off), x0 = _mm512_loadu_si512(d0 + off);
        for (int i = 0; i < 4; i++) {
          w = _mm512_add_epi8(w, _mm512_set1_epi8(1));
          __m512i t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x2);
          t = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, x1);
          T[i][k] = _mm512_dpbusd_epi32(_mm512_slli_epi32(t, 7), w, x0);
        }
      }
      for (int i = 0; i < 4; i++)
        acc[i] = _mm512_add_ps(acc[i], _mm512_mul_ps(_mm512_cvtepi32_ps(reduce8(T[i])), c));
    }
    for (int i = 0; i < 4; i++) tot = _mm512_add_ps(tot, acc[i]);
  }
  return _mm512_reduce_add_ps(tot);
}

}  // namespace

int call_main(int T, bool small_pages, int gap_us);
int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "call"))
    return call_main(argc > 2 ? atoi(argv[2]) : 16, argc > 3 && !strcmp(argv[3], "4k"), argc > 4 ? atoi(argv[4]) : 0);
  const int T = argc > 1 ? atoi(argv[1]) : 16;
  if (argc > 2) { K = atoi(argv[2]); NB = K / QB; }
  if (K % 128) { fprintf(stderr, "K must be a multiple of 128\n"); return 1; }
  printf("# bench-fp4-decode: per thread %d rows x K = %d (%zu KB in today's layout, in L2), repeated\n", ROWS, K,
         (size_t)ROWS * NB * BLK / 1024);
  struct Buf {
    std::vector<uint8_t> rows17;  // today's layout
    uint8_t* codes = nullptr;     // CPU layout, 64-byte aligned
    CanonAct act;
    std::vector<float> xf;
  };
  const int maxT = std::max(T, 1);
  std::vector<Buf> buf(maxT);
  for (int t = 0; t < maxT; t++) {
    std::mt19937 rng(17 + t);
    Buf& b = buf[t];
    b.rows17.resize((size_t)ROWS * NB * BLK + 64);
    for (size_t i = 0; i < b.rows17.size(); i++) b.rows17[i] = (uint8_t)rng();
    for (int r = 0; r < ROWS; r++)
      for (int k = 0; k < NB; k++) b.rows17[((size_t)r * NB + k) * BLK] = (uint8_t)(118 + rng() % 8);  // sane scales
    b.codes = (uint8_t*)aligned_alloc(64, (size_t)ROWS * K / 2);
    for (size_t i = 0; i < (size_t)ROWS * K / 2; i++) b.codes[i] = (uint8_t)rng();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    b.xf.resize(K);
    for (auto& v : b.xf) v = nd(rng);
    canon_act(b.xf.data(), K, b.act);
  }
  const char* names[] = {"canon", "fast", "decode17", "decode64", "arith"};
  for (int nt : {1, T}) {
    std::vector<double> ghz(nt);
#pragma omp parallel num_threads(nt)
    ghz[omp_get_thread_num()] = measure_ghz();
    double g = 0;
    for (double v : ghz) g += v / nt;
    printf("\n## %d thread%s, clock %.2f GHz (dependent adds, while all %d run)\n", nt, nt > 1 ? "s" : "", g, nt);
    printf("%-9s %9s %9s %16s\n", "variant", "Gw/s/core", "w/clock", "ms/expert @16");
    for (int v = 0; v < 5; v++) {
      const int reps = v == 0 || v == 1 ? 60 : v == 4 ? 60 : 300;
      double secs = 0;
#pragma omp parallel num_threads(nt) reduction(max : secs)
      {
        Buf& b = buf[omp_get_thread_num()];
        float y[ROWS];
        uint64_t s = 0;
        for (int pass = 0; pass < 2; pass++) {  // the first warms the caches
          const auto t0 = Clock::now();
          for (int r = 0; r < reps; r++) {
            switch (v) {
              case 0: canon_dot_rows(b.rows17.data(), (size_t)NB * BLK, b.act, 0, ROWS, y); s += (uint64_t)y[r % ROWS]; break;
              case 1:
                for (int i = 0; i < ROWS; i += 4) fast_rows4(b.rows17.data() + (size_t)i * NB * BLK, (size_t)NB * BLK, NB, b.xf.data(), y + i);
                s += (uint64_t)y[r % ROWS];
                break;
              case 2: s += decode17(b.rows17.data(), ROWS); break;
              case 3: s += decode64(b.codes, ROWS); break;
              case 4: s += (uint64_t)arith4(b.act.d0.data(), b.act.d1.data(), b.act.d2.data(), ROWS); break;
            }
          }
          if (pass) secs = std::chrono::duration<double>(Clock::now() - t0).count();
        }
        g_sink = s;
      }
      const double w = (double)ROWS * K * (v == 0 || v == 1 || v == 4 ? 60 : 300);  // per thread
      const double per_core = w / secs / 1e9;
      printf("%-9s %9.2f %9.2f %16.3f\n", names[v], per_core, per_core / g, kExpertWeights / (per_core * 1e9 * 16) * 1e3);
    }
  }
  printf("\n# ms/expert @16: one expert (gate + up + down, %.1fM weights) as if all its rows ran at this K's per-core rate on 16 cores\n",
         kExpertWeights / 1e6);
  return 0;
}

// ---------------------------------------------------------------- call: a whole one-token call
int call_main(int T, bool small_pages, int gap_us) {
  constexpr int E = 5120, F = 2304, CH = 32, NE = 24;
  const size_t gu_row = (size_t)E / QB * BLK, d_row = (size_t)F / QB * BLK;
  const size_t gu_bytes = (size_t)F * gu_row, d_bytes = (size_t)E * d_row;
  struct Ex { uint8_t *gate, *up, *down; };
  std::vector<Ex> ex(NE);
  std::mt19937 rng(5);
  auto fill = [&](size_t n) {
    const size_t len = (n + 64 + (2u << 20) - 1) / (2u << 20) * (2u << 20);
    uint8_t* p = (uint8_t*)mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(p, len, small_pages ? MADV_NOHUGEPAGE : MADV_HUGEPAGE);
    for (size_t i = 0; i < n + 64; i++) p[i] = (uint8_t)rng();
    for (size_t i = 0; i < n; i += BLK) p[i] = (uint8_t)(118 + rng() % 8);
    return p;
  };
  for (auto& e : ex) e = {fill(gu_bytes), fill(gu_bytes), fill(d_bytes)};
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(E), a(F);
  for (auto& v : x) v = nd(rng);
  for (auto& v : a) v = nd(rng);
  CanonAct xc, ac;
  canon_act(x.data(), E, xc);
  canon_act(a.data(), F, ac);
  std::vector<float> yg(F), yu(F), yd(E);
  const double bytes = 2.0 * gu_bytes + d_bytes;
  printf("# bench-fp4-decode call: one token, one expert per call (%.1f MB), %d threads, 32-row tasks, %s pages; ms per call\n", bytes / 1e6, T,
         small_pages ? "4 KB" : "2 MB");
  printf("# (phases timed by the master thread: gate + up, then down, each ending at the team's barrier)\n");
  printf("# idle gap before each call: %d us%s\n", gap_us < 0 ? -gap_us : gap_us, gap_us < 0 ? " (the master spinning)" : gap_us ? " (the master sleeping)" : "");
  printf("%-8s %-6s %8s %8s %8s %8s %8s\n", "schedule", "data", "call", "gate_up", "down", "GB/s", "late");
  for (int sched = 0; sched < 2; sched++) {
    omp_set_schedule(sched ? omp_sched_static : omp_sched_dynamic, 0);
    for (int hot = 1; hot >= 0; hot--) {
      const int N = 400;
      double t_gu = 0, t_d = 0, t_all = 0, t_late = 0;
      std::vector<double> entry(T);
      for (int c = -8; c < N; c++) {
        const Ex& e = ex[hot ? 0 : (c + 8) % NE];
        if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        if (gap_us < 0) {  // the same gap, the master spinning instead of sleeping
          const double until = omp_get_wtime() - gap_us * 1e-6;
          while (omp_get_wtime() < until) {}
        }
        double a1 = 0, a2 = 0;
        const double start = omp_get_wtime();
#pragma omp parallel num_threads(T)
        {
          entry[omp_get_thread_num()] = omp_get_wtime();
#pragma omp for schedule(runtime)
          for (int task = 0; task < 2 * (F / CH); task++) {
            const int m = task / (F / CH), r = (task % (F / CH)) * CH;
            canon_dot_rows(m ? e.up : e.gate, gu_row, xc, r, r + CH, m ? yu.data() : yg.data());
          }
#pragma omp master
          a1 = omp_get_wtime();
#pragma omp for schedule(runtime)
          for (int task = 0; task < E / CH; task++) canon_dot_rows(e.down, d_row, ac, task * CH, task * CH + CH, yd.data());
#pragma omp master
          a2 = omp_get_wtime();
        }
        if (c >= 0) {
          t_gu += a1 - start; t_d += a2 - a1; t_all += a2 - start;  // from the call's start (the wake-up counts)
          double last = start;
          for (double v : entry) last = std::max(last, v);
          t_late += last - start;
        }
      }
      printf("%-8s %-6s %8.3f %8.3f %8.3f %8.0f %8.3f\n", sched ? "static" : "dynamic", hot ? "warm" : "cold", t_all / N * 1e3, t_gu / N * 1e3,
             t_d / N * 1e3, bytes / (t_all / N) / 1e9, t_late / N * 1e3);
    }
  }
  g_sink = (uint64_t)(yg[3] + yu[5] + yd[7]);
  return 0;
}
