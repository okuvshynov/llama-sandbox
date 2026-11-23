// test-fakequant: the rounding functions the runtime uses on the host (src/fakequant.cpp) against
// the reference's (ref/kernel_cpu.py), on the inputs of ref/make_fq_test.py. Must be bit-identical.
//
//   test-fakequant [dir]      dir: in.f32 and the reference's fp8.f32, fp4.f32, fp4e4.f32
//                             (default tests/fq, which is in git; README step 186). Exit code 1 if a value differs.
//
// fp8        fq_fp8_host, the function the CPU experts and the canonical sums call
// fp8 split  fq_fp8_split: its codes times its scale must be the same values
// fp4, fp4e4 the scalar rules fq_e2m1_round / fq_e4m3_round / fq_pow2_ceil, put together here as the
//            reference does (blocks of 32 with a power-of-two scale; blocks of 16 with an e4m3
//            scale). The runtime's fp4 rounding itself runs in GPU kernels (hip/kernels.hip),
//            which the logits checks cover; this keeps the rules they were written from honest.
// No ggml (README step 192): until then this tested the CPU reference path's graph ops only.
#include "fakequant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> load(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  std::vector<float> v;
  float x;
  while (fread(&x, 4, 1, f) == 1) v.push_back(x);
  fclose(f);
  return v;
}

static void fp4_blocks(const float* x, float* y, int64_t n, int B, bool e4m3_scale) {
  for (int64_t b = 0; b < n / B; b++) {
    const float* xb = x + b * B;
    float amax = 0.0f;
    for (int j = 0; j < B; j++) amax = std::max(amax, fabsf(xb[j]));
    const float s = e4m3_scale ? fq_e4m3_round(std::max(amax, 6.0f * 0x1p-9f) / 6.0f)
                               : fq_pow2_ceil(std::max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    for (int j = 0; j < B; j++) y[b * B + j] = fq_e2m1_round(std::clamp(xb[j] / s, -6.0f, 6.0f)) * s;
  }
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/fq";
  const std::vector<float> in = load((dir + "/in.f32").c_str());
  const int64_t n = (int64_t)in.size();
  if (n == 0 || n % 64) { fprintf(stderr, "%s/in.f32: %ld values, not rows of 64\n", dir.c_str(), (long)n); return 1; }
  const char* names[] = {"fp8", "fp8 split", "fp4", "fp4e4"};
  const char* files[] = {"fp8", "fp8", "fp4", "fp4e4"};
  int bad_total = 0;
  for (int k = 0; k < 4; k++) {
    const std::vector<float> want = load((dir + "/" + files[k] + ".f32").c_str());
    if ((int64_t)want.size() != n) { fprintf(stderr, "%s.f32: %zu values, in.f32 has %ld\n", files[k], want.size(), (long)n); return 1; }
    std::vector<float> y(n);
    if (k == 0) fq_fp8_host(in.data(), y.data(), n);
    else if (k == 1)
      for (int64_t b = 0; b < n / 32; b++) {
        float q[32];
        const float s = fq_fp8_split(in.data() + b * 32, q);
        for (int j = 0; j < 32; j++) y[b * 32 + j] = q[j] * s;
      }
    else fp4_blocks(in.data(), y.data(), n, k == 2 ? 32 : 16, k == 3);
    int bad = 0;
    for (int64_t i = 0; i < n; i++) {
      const float a = y[i], b = want[i];
      if (memcmp(&a, &b, 4) != 0 && !(a == 0 && b == 0)) {
        if (bad < 5) printf("  %s[%ld]: in %.9g ours %.9g ref %.9g\n", names[k], (long)i, in[i], a, b);
        bad++;
      }
    }
    printf("%-9s %ld values, %d differ\n", names[k], (long)n, bad);
    bad_total += bad;
  }
  printf(bad_total ? "FAIL\n" : "ALL BIT-EXACT\n");
  return bad_total ? 1 : 0;
}
