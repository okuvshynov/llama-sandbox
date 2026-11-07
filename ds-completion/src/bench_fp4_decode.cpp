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
//   bin/bench-fp4-decode act   canon_act_blocks (AVX-512) against canon_act_blocks_ref byte for byte on
//     mixed inputs (README step 80), and both timed on one thread for 160 and 72 blocks
//   bin/bench-fp4-decode call [threads]   a whole one-token call as forward_batch runs it: gate +
//     up in 32-row tasks, then down in 32-row tasks (canon_dot_rows), on whole experts in today's
//     layout; warm (the same expert every call: L2 + L3) and cold (24 experts in turn, 450 MB:
//     DRAM); the dynamic schedule (the runtime's) and static (contiguous rows per thread).
//     bin/bench-fp4-decode call 16 4k: the experts on 4 KB pages (the model's mapping), else 2 MB
//     bin/bench-fp4-decode call 16 2m 1500: an idle gap of 1500 us before every call (the runtime's GPU
//     bin/bench-fp4-decode call 16 2m 0 once|rewrite|copy N: only the dynamic warm case, N calls, with the
//     activations written once (as above), rewritten by the team every call (as forward_batch does),
//     or rewritten and then copied by each thread into its own buffer, or rewritten by one thread only
//     (single: no two writers share a line), or rewritten by the team in 16-block chunks of the
//     64-byte aligned arrays (aligned16: every thread whole lines of its own) (README step 79: under perf stat),
//     or rewritten by one worker, not the master (worker: its core spun through the gap; README step 81)
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

int call_main(int T, bool small_pages, int gap_us, int act, int calls);
int act_main();
int main(int argc, char** argv) {
  if (argc > 1 && !strcmp(argv[1], "act")) return act_main();
  if (argc > 1 && !strcmp(argv[1], "call"))
    return call_main(argc > 2 ? atoi(argv[2]) : 16, argc > 3 && !strcmp(argv[3], "4k"), argc > 4 ? atoi(argv[4]) : 0,
                     argc > 5 ? (!strcmp(argv[5], "once") ? 0 : !strcmp(argv[5], "rewrite") ? 1 : !strcmp(argv[5], "copy") ? 2 : !strcmp(argv[5], "single") ? 3 : !strcmp(argv[5], "aligned16") ? 4 : !strcmp(argv[5], "worker") ? 5 : -1) : -1,
                     argc > 6 ? atoi(argv[6]) : 400);
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
// what canon_dot_rows reads of an activation (not x9): dst sized for the same n
static void copy_act(const CanonAct& src, CanonAct& dst) {
  memcpy(dst.d0.data(), src.d0.data(), src.d0.size());
  memcpy(dst.d1.data(), src.d1.data(), src.d1.size());
  memcpy(dst.d2.data(), src.d2.data(), src.d2.size());
  memcpy(dst.corr.data(), src.corr.data(), src.corr.size() * 4);
  memcpy(dst.c.data(), src.c.data(), src.c.size() * 4);
}

int call_main(int T, bool small_pages, int gap_us, int act, int calls) {
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
  std::vector<CanonAct> pxc(T), pac(T);  // per-thread copies (act = copy)
  for (int i = 0; i < T; i++) { canon_act_size(E, pxc[i]); canon_act_size(F, pac[i]); }
  std::vector<float> yg(F), yu(F), yd(E);
  const double bytes = 2.0 * gu_bytes + d_bytes;
  printf("# bench-fp4-decode call: one token, one expert per call (%.1f MB), %d threads, 32-row tasks, %s pages; ms per call\n", bytes / 1e6, T,
         small_pages ? "4 KB" : "2 MB");
  printf("# (phases timed by the master thread: gate + up, then down, each ending at the team's barrier)\n");
  printf("# idle gap before each call: %d us%s\n", gap_us < 0 ? -gap_us : gap_us, gap_us < 0 ? " (the master spinning)" : gap_us ? " (the master sleeping)" : "");
  printf("%-8s %-6s %8s %8s %8s %8s %8s\n", "schedule", "data", "call", "gate_up", "down", "GB/s", "late");
  auto mis = [](const CanonAct& a) {
    return (int)(((uintptr_t)a.d0.data() | (uintptr_t)a.d1.data() | (uintptr_t)a.d2.data() | (uintptr_t)a.corr.data() | (uintptr_t)a.c.data()) % 64);
  };
  if (act >= 0) printf("# arrays 64-byte aligned: %s\n", mis(xc) == 0 && mis(ac) == 0 ? "yes" : "NO");
  if (act >= 0) printf("# activations: %s; %d calls\n", act == 0 ? "written once" : act == 1 ? "rewritten every call" : act == 2 ? "rewritten, then per-thread copies" : act == 3 ? "rewritten by one thread" : act == 4 ? "rewritten in 16-block chunks (whole lines per thread)" : "rewritten by one worker (not the master)", calls);
  for (int sched = 0; sched < (act >= 0 ? 1 : 2); sched++) {
    omp_set_schedule(sched ? omp_sched_static : omp_sched_dynamic, 0);
    for (int hot = 1; hot >= (act >= 0 ? 1 : 0); hot--) {
      const int N = act >= 0 ? calls : 400;
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
          const int tid = omp_get_thread_num();
          entry[tid] = omp_get_wtime();
          const CanonAct *xa = &xc, *aa = &ac;
          if (act == 1 || act == 2) {
#pragma omp for schedule(static)
            for (int b0 = 0; b0 < E / QB; b0 += 8) canon_act_blocks(x.data(), b0, b0 + 8, xc);
          }
          if (act == 3) {
#pragma omp single
            canon_act_blocks(x.data(), 0, E / QB, xc);
          }
          if (act == 4) {
#pragma omp for schedule(static)
            for (int b0 = 0; b0 < E / QB; b0 += 16) canon_act_blocks(x.data(), b0, std::min(b0 + 16, E / QB), xc);
          }
          if (act == 5) {
            if (tid == std::min(1, T - 1)) canon_act_blocks(x.data(), 0, E / QB, xc);
#pragma omp barrier
          }
          if (act == 2) { copy_act(xc, pxc[tid]); xa = &pxc[tid]; }
#pragma omp for schedule(runtime)
          for (int task = 0; task < 2 * (F / CH); task++) {
            const int m = task / (F / CH), r = (task % (F / CH)) * CH;
            canon_dot_rows(m ? e.up : e.gate, gu_row, *xa, r, r + CH, m ? yu.data() : yg.data());
          }
          if (act == 1 || act == 2) {
#pragma omp for schedule(static)
            for (int b0 = 0; b0 < F / QB; b0 += 8) canon_act_blocks(a.data(), b0, b0 + 8, ac);
          }
          if (act == 3) {
#pragma omp single
            canon_act_blocks(a.data(), 0, F / QB, ac);
          }
          if (act == 4) {
#pragma omp for schedule(static)
            for (int b0 = 0; b0 < F / QB; b0 += 16) canon_act_blocks(a.data(), b0, std::min(b0 + 16, F / QB), ac);
          }
          if (act == 5) {
            if (tid == std::min(1, T - 1)) canon_act_blocks(a.data(), 0, F / QB, ac);
#pragma omp barrier
          }
          if (act == 2) { copy_act(ac, pac[tid]); aa = &pac[tid]; }
#pragma omp master
          a1 = omp_get_wtime();
#pragma omp for schedule(runtime)
          for (int task = 0; task < E / CH; task++) canon_dot_rows(e.down, d_row, *aa, task * CH, task * CH + CH, yd.data());
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

// ---------------------------------------------------------------- act: the activation digits
int act_main() {
  std::mt19937 rng(80);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::uniform_int_distribution<int> kind(0, 9), ex(-40, 40);
  const int NB = 1 << 21;  // blocks checked
  std::vector<float> x((size_t)NB * QB);
  for (int b = 0; b < NB; b++) {
    const int k = kind(rng);  // per block: a magnitude regime
    const float scale = ldexpf(1.0f, ex(rng));
    for (int j = 0; j < QB; j++) {
      float v = u(rng) * scale;
      const int r = (int)(rng() % 16);
      if (k == 0 && r < 4) v = 0.0f;                                  // zeros
      if (k == 1) v = ldexpf(u(rng), -140);                            // subnormal inputs
      if (k == 2 && r < 8) v = (float)((int)(rng() % 17) * 2 + 1) * ldexpf(1.0f, ex(rng) / 4 - 4);  // odd multiples: ties
      if (k == 3) v = u(rng) * 1e-6f;                                  // amax below 1e-4
      if (k == 4 && r == 0) v = -v;
      if (k == 5) v = u(rng) * 3e4f;                                   // amax above 448
      if (k == 6 && j == 0) v = 448.0f * scale;                        // a power-of-two-exact amax
      x[(size_t)b * QB + j] = v;
    }
  }
  CanonAct a, r;
  canon_act_size(NB * QB, a);
  canon_act_size(NB * QB, r);
  canon_act_blocks(x.data(), 0, NB, a);
  canon_act_blocks_ref(x.data(), 0, NB, r);
  long bad[6] = {};
  for (size_t i = 0; i < (size_t)NB * QB; i++) {
    bad[0] += a.d0[i] != r.d0[i];
    bad[1] += a.d1[i] != r.d1[i];
    bad[2] += a.d2[i] != r.d2[i];
    bad[3] += a.x9[i] != r.x9[i];
  }
  for (int b = 0; b < NB; b++) {
    bad[4] += a.corr[b] != r.corr[b];
    bad[5] += memcmp(&a.c[b], &r.c[b], 4) != 0;
  }
  printf("# canon_act_blocks (AVX-512) vs canon_act_blocks_ref, %d blocks of mixed inputs: differing d0 %ld, d1 %ld, d2 %ld, x9 %ld, corr %ld, c %ld\n",
         NB, bad[0], bad[1], bad[2], bad[3], bad[4], bad[5]);
  // speed: one thread, the input of gate / up (160 blocks) and of down (72)
  for (int nb : {160, 72}) {
    CanonAct t;
    canon_act_size(nb * QB, t);
    double best[2] = {1e9, 1e9};
    for (int rep = 0; rep < 5; rep++)
      for (int v = 0; v < 2; v++) {
        const int n = 2000;
        const auto t0 = Clock::now();
        for (int i = 0; i < n; i++) (v ? canon_act_blocks_ref : canon_act_blocks)(x.data() + (size_t)(i % 64) * nb * QB, 0, nb, t);
        best[v] = std::min(best[v], std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / n);
      }
    printf("%d blocks (%d values), one thread: AVX-512 %.2f us, scalar %.2f us (x%.1f)\n", nb, nb * QB, best[0], best[1], best[1] / best[0]);
  }
  g_sink = (uint64_t)a.corr[7];
  return 0;
}
