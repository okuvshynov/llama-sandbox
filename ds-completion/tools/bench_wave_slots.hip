// Launch time by the launch's wave count (README step 102): a kernel whose every wave reads
// `steps` KB of its own (64 lanes x 16 bytes per step) and writes one float, over grids of 64 to
// 11264 waves. A launch of 2017-2048 waves (the GPU's 2048 wave slots filled exactly) takes ~48 us
// whatever it does, an empty kernel too; multiples of 2048 above that cost a light kernel ~30 us
// more than their neighbours. With `split`, the same grids as two launches (3/8 and 5/8).
//
//   bin-hip/bench-wave-slots [threads per block = 256] [split]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); abort(); } } while (0)

__global__ void waves_kernel(const float* w, int steps, size_t region, float* y, int nreg, int b0) {
  const int wave = (blockIdx.x + b0) * (blockDim.x / 64) + (threadIdx.x >> 6), lane = threadIdx.x & 63;
  const float* p = w + (size_t)(wave % nreg) * region;
  float acc = 0;
  for (int i = 0; i < steps; i++) {
    const float4 v = *(const float4*)(p + (i * 64 + lane) * 4);
    acc += v.x + v.y + v.z + v.w;
  }
  for (int o = 32; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 64);
  if (lane == 0) y[wave] = acc;
}

int main(int argc, char** argv) {
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  const int threads = argc > 1 ? atoi(argv[1]) : 256, wpb = threads / 64;
  const bool split = argc > 2 && !strcmp(argv[2], "split");
  const size_t tot = (size_t)64 << 20;  // floats
  float *w, *y;
  CHECK(hipMalloc(&w, tot * 4));
  CHECK(hipMemset(w, 0, tot * 4));
  CHECK(hipMalloc(&y, 1 << 20));
  const int stepsv[] = {0, 1, 5, 20, 80};
  printf("# %d threads per block%s; us per launch by waves and KB read per wave\n%8s", threads,
         split ? ", each grid as two launches (3/8 + 5/8 of its blocks)" : "", "waves");
  for (int st : stepsv) printf(" %8d", st);
  printf("\n");
  std::vector<int> waves;
  for (int v = 256; v <= 11264; v += 256) waves.push_back(v);
  for (int v : {1984, 2016, 2020, 2032, 2040, 2052, 2064, 2080, 2112, 2176}) waves.push_back(v);
  std::sort(waves.begin(), waves.end());
  for (int wv : waves) {
    if (wv % wpb) continue;
    const int blocks = wv / wpb, b1 = split ? blocks * 3 / 8 : blocks;
    printf("%8d", wv);
    for (int st : stepsv) {
      const size_t region = st ? (size_t)st * 256 : 256;
      const int nreg = (int)(tot / region);
      auto run = [&] {
        waves_kernel<<<b1, threads, 0, s>>>(w, st, region, y, nreg, 0);
        if (b1 < blocks) waves_kernel<<<blocks - b1, threads, 0, s>>>(w, st, region, y, nreg, b1);
      };
      for (int i = 0; i < 20; i++) run();
      CHECK(hipStreamSynchronize(s));
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 500; i++) run();
      CHECK(hipStreamSynchronize(s));
      printf(" %8.1f", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 500);
    }
    printf("\n");
  }
  return 0;
}
