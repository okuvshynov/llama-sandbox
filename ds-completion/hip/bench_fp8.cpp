// bench-fp8: the fp8 GEMV (fp8_gemv.hip) on real weights from dense-fp8.gguf (fp8pack output).
//
//   bench-fp8 <dense-fp8.gguf> [--dev D] [--layer L] [--ms 1,2,4,8] [--reps 200]
//
// Per tensor and batch size M:
//   - the device activation quantizer must equal the host one (src/fakequant.cpp rules) bit for bit;
//   - y must match a double-precision host result of the same fp8 x fp8 math (expect ~1e-7: only
//     the order of fp32 additions differs);
//   - time per call (events, back to back) and weight GB/s against the ~880 GB/s HBM ceiling.
// --bitwise: instead, for M = 2..8, every token's result of one M-token call must equal a
// single-token call on that token bit for bit (batched verification relies on it).
// --gemm: the many-input kernel (fp8_gemm, prompt chunks) for each M of --ms: every token's
// result against a one-token fp8_gemv call, bit for bit; time against fp8_gemv in slices of 8;
// with --sweep every (rows per lane group, inputs per tile) instance.

#include "fp8_gemv.h"
#include "fakequant.h"
#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

static float pow2_ceil(float v) {
  int e;
  const float m = frexpf(v, &e);
  return ldexpf(1.0f, m == 0.5f ? e - 1 : e);
}

static float e4m3_decode(uint8_t c) {
  const float sign = (c & 0x80) ? -1.0f : 1.0f;
  const int E = (c >> 3) & 0xF, M = c & 7;
  return E == 0 ? sign * ldexpf((float)M, -9) : sign * ldexpf(1.0f + M / 8.0f, E - 7);
}

struct Shape { const char* name; int rpg_div; };  // rpg = N / rpg_div (groups)

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2) { fprintf(stderr, "usage: bench-fp8 <dense-fp8.gguf> [--dev D] [--layer L] [--ms 1,2,4,8] [--reps N]\n"); return 2; }
  int dev = 0, layer = 2, reps = 200;
  bool sweep = false, bitwise = false, gemm = false;
  std::vector<int> ms = {1, 2, 4, 8};
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--dev") dev = atoi(argv[++i]);
    else if (a == "--layer") layer = atoi(argv[++i]);
    else if (a == "--reps") reps = atoi(argv[++i]);
    else if (a == "--sweep") sweep = true;
    else if (a == "--bitwise") bitwise = true;
    else if (a == "--gemm") gemm = true;
    else if (a == "--ms") { ms.clear(); for (char* p = argv[++i]; *p;) { ms.push_back((int)strtol(p, &p, 10)); if (*p == ',') p++; } }
  }
  CHECK(hipSetDevice(dev));
  gguf_init_params gp = {true, nullptr};
  gguf_context* g = gguf_init_from_file(argv[1], gp);
  if (!g) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  FILE* f = fopen(argv[1], "rb");
  auto read = [&](const std::string& name, std::vector<uint8_t>& out, int64_t* ne) {
    int64_t id = gguf_find_tensor(g, name.c_str());
    if (id < 0) return false;
    const size_t n = gguf_get_tensor_size(g, id);
    out.resize(n);
    fseeko(f, (off_t)(gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)), SEEK_SET);
    if (fread(out.data(), 1, n, f) != n) return false;
    if (ne) { ne[0] = gguf_get_tensor_ne(g, id)[0]; ne[1] = gguf_get_tensor_ne(g, id)[1]; }
    return true;
  };

  const std::vector<Shape> shapes = {
      {"attn_q_a", 1}, {"attn_q_b", 1}, {"attn_kv", 1}, {"attn_output_a", 8}, {"attn_output_b", 1},
      {"indexer.attn_q_b", 1}, {"ffn_gate_shexp", 1}, {"ffn_down_shexp", 1}, {"engram_kv", 1}};
  hipStream_t st;
  CHECK(hipStreamCreate(&st));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;

  printf("# %s, device %d, layer %d (engram_kv from layer 1)\n", argv[1], dev, layer);
  printf("%-18s %6s %6s %3s %10s %10s %8s %8s %10s\n", "tensor", "K", "N", "M", "quant", "relerr", "us", "GB/s", "x of 880");
  for (const Shape& sh : shapes) {
    const int L = !strcmp(sh.name, "engram_kv") ? 1 : layer;
    const std::string name = "blk." + std::to_string(L) + "." + sh.name + ".weight";
    std::vector<uint8_t> wq, wsc;
    int64_t ne[2];
    if (!read(name, wq, ne) || !read(name + ".scale", wsc, nullptr)) { printf("%-18s missing\n", sh.name); continue; }
    const int K = (int)ne[0], N = (int)ne[1], groups = sh.rpg_div, rpg = N / groups, Kt = K * groups;
    uint8_t *dw, *ds;
    CHECK(hipMalloc(&dw, wq.size()));
    CHECK(hipMalloc(&ds, wsc.size()));
    CHECK(hipMemcpy(dw, wq.data(), wq.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(ds, wsc.data(), wsc.size(), hipMemcpyHostToDevice));
    for (int M : ms) {
      std::vector<float> x((size_t)M * Kt);
      for (auto& v : x) v = nd(rng) * 0.5f;
      // host quantization (the reference rule) and device quantization must agree exactly
      std::vector<float> hq(x.size()), hs(x.size() / 32);
      for (size_t b = 0; b < hs.size(); b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
        const float s = pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
        hs[b] = s;
        for (int j = 0; j < 32; j++) hq[b * 32 + j] = fq_e4m3_round(x[b * 32 + j] / s);
      }
      float *dx, *dxs, *dy;
      uint8_t* dxq;
      CHECK(hipMalloc(&dx, x.size() * 4));
      CHECK(hipMalloc(&dxq, x.size()));
      CHECK(hipMalloc(&dxs, hs.size() * 4));
      CHECK(hipMalloc(&dy, (size_t)M * N * 4));
      CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      fp8_quant_act(dx, dxq, dxs, M, Kt, st);
      std::vector<uint8_t> gq(x.size());
      std::vector<float> gs(hs.size());
      CHECK(hipMemcpyAsync(gq.data(), dxq, gq.size(), hipMemcpyDeviceToHost, st));
      CHECK(hipMemcpyAsync(gs.data(), dxs, gs.size() * 4, hipMemcpyDeviceToHost, st));
      CHECK(hipStreamSynchronize(st));
      long qbad = 0;
      for (size_t i = 0; i < x.size(); i++) qbad += e4m3_decode(gq[i]) != hq[i];
      for (size_t i = 0; i < hs.size(); i++) qbad += gs[i] != hs[i];

      Fp8Act act = {dxq, dxs};
      if (gemm) {
        auto timed = [&](auto fn) {
          for (int i = 0; i < 3; i++) fn();
          CHECK(hipEventRecord(e0, st));
          for (int i = 0; i < 20; i++) fn();
          CHECK(hipEventRecord(e1, st));
          CHECK(hipEventSynchronize(e1));
          float t;
          CHECK(hipEventElapsedTime(&t, e0, e1));
          return t * 1000.0 / 20;
        };
        auto slices = [&]() {
          for (int m0 = 0; m0 < M; m0 += 8)
            fp8_gemv(dw, ds, N, K, rpg, Fp8Act{dxq + (size_t)m0 * Kt, dxs + (size_t)m0 * Kt / 32}, std::min(8, M - m0), dy + (size_t)m0 * N, st);
        };
        float* dy1;
        CHECK(hipMalloc(&dy1, (size_t)N * 4));
        std::vector<float> yM((size_t)M * N), y1(N);
        auto check = [&]() {  // the last fp8_gemm result in dy against one-token calls
          CHECK(hipMemcpy(yM.data(), dy, yM.size() * 4, hipMemcpyDeviceToHost));
          long diff = 0;
          for (int m = 0; m < M; m++) {
            fp8_gemv(dw, ds, N, K, rpg, Fp8Act{dxq + (size_t)m * Kt, dxs + (size_t)m * Kt / 32}, 1, dy1, st);
            CHECK(hipMemcpy(y1.data(), dy1, N * 4, hipMemcpyDeviceToHost));
            diff += memcmp(y1.data(), yM.data() + (size_t)m * N, N * 4) ? 1 : 0;
          }
          return diff;
        };
        const double us8 = timed(slices);
        printf("%-18s %6d %6d %4d lpr %2d  gemv x8: %8.1f us", sh.name, K, N, M, fp8_lpr(N, K), us8);
        // (rows per lane group, inputs per tile, blocks per CU); 0s = the built-in choice
        std::vector<std::array<int, 3>> cfgs = {{0, 0, 0}};
        if (sweep) cfgs = {{2, 8, 1}, {2, 8, 2}, {2, 16, 1}, {2, 16, 3}, {4, 8, 1}, {4, 8, 2}, {4, 16, 1}, {4, 16, 2}};
        for (const auto& cfg : cfgs) {
          const int r = cfg[0], mt = cfg[1], mb = cfg[2];
          const double us = timed([&]() { fp8_gemm_cfg(dw, ds, N, K, rpg, act, M, dy, st, r, mt, mb); });
          fp8_gemm_cfg(dw, ds, N, K, rpg, act, M, dy, st, r, mt, mb);
          const long diff = check();
          printf("  gemm %d/%d/%d: %8.1f us (%.1fx) %s", r, mt, mb, us, us8 / us, diff ? "DIFFERS" : "same");
        }
        printf("\n");
        CHECK(hipFree(dy1));
        CHECK(hipFree(dx)); CHECK(hipFree(dxq)); CHECK(hipFree(dxs)); CHECK(hipFree(dy));
        continue;
      }
      if (bitwise) {
        // every token of the M-token call against a one-token call on that token alone
        float* dy1;
        CHECK(hipMalloc(&dy1, (size_t)N * 4));
        fp8_gemv(dw, ds, N, K, rpg, act, M, dy, st);
        std::vector<float> yM((size_t)M * N), y1(N);
        CHECK(hipMemcpyAsync(yM.data(), dy, yM.size() * 4, hipMemcpyDeviceToHost, st));
        long diff = 0;
        for (int m = 0; m < M; m++) {
          Fp8Act a1 = {dxq + (size_t)m * Kt, dxs + (size_t)m * Kt / 32};
          fp8_gemv(dw, ds, N, K, rpg, a1, 1, dy1, st);
          CHECK(hipMemcpyAsync(y1.data(), dy1, N * 4, hipMemcpyDeviceToHost, st));
          CHECK(hipStreamSynchronize(st));
          diff += memcmp(y1.data(), yM.data() + (size_t)m * N, N * 4) ? 1 : 0;
          if (m == 0 && M > 1) {  // control: token 1's own result must differ from token 0's
            std::vector<float> yo(N);
            Fp8Act ao = {dxq + (size_t)Kt, dxs + (size_t)Kt / 32};
            fp8_gemv(dw, ds, N, K, rpg, ao, 1, dy1, st);
            CHECK(hipMemcpyAsync(yo.data(), dy1, N * 4, hipMemcpyDeviceToHost, st));
            CHECK(hipStreamSynchronize(st));
            if (!memcmp(yo.data(), yM.data(), N * 4)) { printf("control failed: tokens 0 and 1 give the same bits\n"); return 1; }
          }
        }
        printf("%-18s %6d %6d %3d %s\n", sh.name, K, N, M, diff ? "DIFFERS from M = 1" : "bit-identical to M = 1");
        CHECK(hipFree(dy1));
        CHECK(hipFree(dx)); CHECK(hipFree(dxq)); CHECK(hipFree(dxs)); CHECK(hipFree(dy));
        continue;
      }
      fp8_gemv(dw, ds, N, K, rpg, act, M, dy, st);
      std::vector<float> y((size_t)M * N);
      CHECK(hipMemcpyAsync(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost, st));
      CHECK(hipStreamSynchronize(st));
      double num = 0, den = 0;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          const int gi = n / rpg;
          double r = 0;
          for (int k = 0; k < K; k++) {
            const size_t xi = (size_t)m * Kt + (size_t)gi * K + k;
            const double wv = e4m3_decode(wq[(size_t)n * K + k]) * ldexp(1.0, (int)wsc[(n / 32) * (K / 32) + k / 32] - 127);
            r += wv * (double)hq[xi] * hs[xi / 32];
          }
          const double d = y[(size_t)m * N + n] - r;
          num += d * d;
          den += r * r;
        }
      // timing: the built-in choice, or (--sweep) every lanes-per-row x rows-per-wave split
      auto time_cfg = [&](int lpr, int r) {
        for (int i = 0; i < 10; i++) fp8_gemv_cfg(dw, ds, N, K, rpg, act, M, dy, st, lpr, r);
        CHECK(hipEventRecord(e0, st));
        for (int i = 0; i < reps; i++) fp8_gemv_cfg(dw, ds, N, K, rpg, act, M, dy, st, lpr, r);
        CHECK(hipEventRecord(e1, st));
        CHECK(hipEventSynchronize(e1));
        float ms_total;
        CHECK(hipEventElapsedTime(&ms_total, e0, e1));
        return ms_total * 1000.0 / reps;
      };
      const double bytes = wq.size() + wsc.size();
      if (!sweep) {
        const double us = time_cfg(0, 0), gbs = bytes / (us * 1e-6) / 1e9;
        printf("%-18s %6d %6d %3d %10s %10.2e %8.1f %8.1f %10.2f\n", sh.name, K, N, M, qbad ? "MISMATCH" : "exact",
               sqrt(num / den), us, gbs, gbs / 880.0);
      } else {
        printf("%-18s %6d %6d %3d %10s %10.2e  ", sh.name, K, N, M, qbad ? "MISMATCH" : "exact", sqrt(num / den));
        double best = 1e30;
        int bl = 0, br = 0;
        for (int lpr : {8, 16, 32, 64}) {
          if (K % (lpr * 16)) continue;
          for (int r : {1, 2, 4, 8}) {
            const double us = time_cfg(lpr, r);
            printf(" %d/%d:%.0f", lpr, r, bytes / (us * 1e-6) / 1e9);
            if (us < best) { best = us; bl = lpr; br = r; }
          }
        }
        printf("   best %d/%d %.1f us %.0f GB/s\n", bl, br, best, bytes / (best * 1e-6) / 1e9);
      }
      CHECK(hipFree(dx)); CHECK(hipFree(dxq)); CHECK(hipFree(dxs)); CHECK(hipFree(dy));
    }
    CHECK(hipFree(dw));
    CHECK(hipFree(ds));
  }
  return 0;
}
