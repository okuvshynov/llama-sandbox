// Weight reads alone, by row stride and rows per wave (README step 101): the fp8 GEMV's load
// pattern without its arithmetic (64 lanes per row, 16 bytes per lane and step, 4 waves per
// block), over a random matrix whose rows sit `K + pad` bytes apart on the GPU. Rows a multiple of
// 4096 bytes apart read at a fraction of the speed, and worse with more rows per wave.
//
//   bin-hip/bench-row-stride
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); abort(); } } while (0)

// P 0: step-major (the kernel's order: per step, the R rows); 1: row-major (a row whole, then the next);
// 2: step-major, the R loads issued before any is used
template <int R, int P, int WAVES>
__global__ void __launch_bounds__(64 * WAVES) loads_kernel(const uint8_t* __restrict__ w, int N, int K, float* __restrict__ y, int ST) {
  const int lane = threadIdx.x & 63;
  const int row0 = (blockIdx.x * WAVES + (threadIdx.x >> 6)) * R;
  if (row0 >= N) return;
  const uint8_t* wr = w + (size_t)row0 * ST;
  uint32_t acc[R];
#pragma unroll
  for (int i = 0; i < R; i++) acc[i] = 0;
  if (P == 1) {
#pragma unroll
    for (int i = 0; i < R; i++)
      for (int k = lane * 16; k < K; k += 1024) {
        const uint4 v = *(const uint4*)(wr + (size_t)i * ST + k);
        acc[i] += v.x ^ v.y ^ v.z ^ v.w;
      }
  } else {
    for (int k = lane * 16; k < K; k += 1024) {
      if (P == 2) {
        uint4 v[R];
#pragma unroll
        for (int i = 0; i < R; i++) v[i] = *(const uint4*)(wr + (size_t)i * ST + k);
#pragma unroll
        for (int i = 0; i < R; i++) acc[i] += v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
      } else {
#pragma unroll
        for (int i = 0; i < R; i++) {
          const uint4 v = *(const uint4*)(wr + (size_t)i * ST + k);
          acc[i] += v.x ^ v.y ^ v.z ^ v.w;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < R; i++) {
    uint32_t v = acc[i];
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    if (lane == 0) y[row0 + i] = (float)v;
  }
}

template <int R, int P, int WAVES>
static double run(const uint8_t* w, int N, int K, float* y, hipStream_t s, int reps, int ST) {
  const dim3 grid((N + WAVES * R - 1) / (WAVES * R));
  for (int i = 0; i < 20; i++) loads_kernel<R, P, WAVES><<<grid, 64 * WAVES, 0, s>>>(w, N, K, y, ST);
  CHECK(hipStreamSynchronize(s));
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; i++) loads_kernel<R, P, WAVES><<<grid, 64 * WAVES, 0, s>>>(w, N, K, y, ST);
  CHECK(hipStreamSynchronize(s));
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}

int main() {
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  const int shapes[][2] = {{8192, 4096}, {5120, 8192}, {1280, 5120}, {32768, 1280}};
  for (auto& sh : shapes) {
    const int N = sh[0], K = sh[1];
    printf("## N = %d, K = %d (%.1f MB): us per call (GB/s of weights), step-major, 4 waves per block\n", N, K, (double)N * K / 1e6);
    printf("%-24s %14s %14s %14s %14s\n", "row stride on the GPU", "R = 1", "2", "4", "8");
    for (int pad : {0, 64, 256, 1024, 1088, 4160}) {
      const int ST = K + pad;
      std::vector<uint8_t> h((size_t)N * ST + 4096);
      for (auto& b : h) b = (uint8_t)rand();
      uint8_t* w; float* y;
      CHECK(hipMalloc(&w, h.size()));
      CHECK(hipMalloc(&y, N * 4));
      CHECK(hipMemcpy(w, h.data(), h.size(), hipMemcpyHostToDevice));
      printf("K + %-20d", pad);
      for (double t : {run<1, 0, 4>(w, N, K, y, s, 300, ST), run<2, 0, 4>(w, N, K, y, s, 300, ST), run<4, 0, 4>(w, N, K, y, s, 300, ST), run<8, 0, 4>(w, N, K, y, s, 300, ST)})
        printf(" %7.1f (%4.0f)", t, (double)N * K / t / 1e3);
      printf("\n");
      CHECK(hipFree(w)); CHECK(hipFree(y));
    }
  }
  return 0;
}
