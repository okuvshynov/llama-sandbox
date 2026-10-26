// ceiling: what one Vega II die can do, measured without ggml.
//   read bandwidth  : streaming float4 reads of a buffer, summed (so they can't be elided),
//                     at the weight size (58 MiB, >> 4 MiB L2) and at 1 GiB
//   copy bandwidth  : hipMemcpy device-to-device (read + write)
//   FMA peak        : dependent-free FP32 FMAs from registers
// hipEvent timing, median of 20 after warm-up.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_) { fprintf(stderr, "%s -> %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void read_kernel(const float4* __restrict__ p, size_t n, float* out) {
  float4 acc = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = p[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 12345.678f) *out = s;  // never true; keeps the loads alive
}

// gfx9 VOP3 takes at most one literal/SGPR operand, so two literal constants
// would cost extra moves per FMA; use a[j] = a[j]*s + a[j] with s in an SGPR.
__global__ void fma_kernel(float* out, int iters, float s_) {
  float a[8];
  for (int j = 0; j < 8; j++) a[j] = threadIdx.x * 0.001f + j;
  for (int i = 0; i < iters; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = fmaf(a[j], s_, a[j]);
  float s = 0;
  for (int j = 0; j < 8; j++) s += a[j];
  if (s == 12345.678f) *out = s;
}

template <class F> static float med_ms(F f) {
  hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
  for (int i = 0; i < 3; i++) f();
  std::vector<float> t;
  for (int i = 0; i < 20; i++) { CHECK(hipEventRecord(a)); f(); CHECK(hipEventRecord(b)); CHECK(hipEventSynchronize(b)); float ms; CHECK(hipEventElapsedTime(&ms, a, b)); t.push_back(ms); }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main(int argc, char** argv) {
  int dev = argc > 1 ? atoi(argv[1]) : 0;
  CHECK(hipSetDevice(dev));
  hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, dev));
  printf("# device %d  %s  CUs=%d  clock(max)=%d MHz\n", dev, p.gcnArchName, p.multiProcessorCount, p.clockRate / 1000);
  float* out; CHECK(hipMalloc(&out, 4));
  size_t big = 1ull << 30;
  void* buf; void* buf2; CHECK(hipMalloc(&buf, big)); CHECK(hipMalloc(&buf2, big)); CHECK(hipMemset(buf, 0, big));
  int blocks = p.multiProcessorCount * 128;  // fewer blocks under-fill HBM at tens of MiB (see grid sweep)
  // single = one launch per timed interval; x10 = ten back-to-back launches, per launch
  for (size_t bytes : {(size_t)32 << 20, (size_t)58 << 20, (size_t)256 << 20, big}) {
    size_t n = bytes / 16;
    float ms = med_ms([&] { read_kernel<<<blocks, 256>>>((const float4*)buf, n, out); });
    float ms10 = med_ms([&] { for (int r = 0; r < 10; r++) read_kernel<<<blocks, 256>>>((const float4*)buf, n, out); }) / 10;
    printf("read   %7.1f MiB  single %8.1f us %7.1f GB/s   x10 %8.1f us %7.1f GB/s\n", bytes / 1048576.0, ms * 1e3,
           bytes / (ms * 1e-3) / 1e9, ms10 * 1e3, bytes / (ms10 * 1e-3) / 1e9);
  }
  for (int bpc : {8, 16, 64, 128}) {  // grid size sensitivity at the weight size
    size_t bytes = (size_t)58 << 20, n = bytes / 16;
    float ms10 = med_ms([&] { for (int r = 0; r < 10; r++) read_kernel<<<p.multiProcessorCount * bpc, 256>>>((const float4*)buf, n, out); }) / 10;
    printf("read   %7.1f MiB  x10 grid=%3d blocks/CU %8.1f us %7.1f GB/s\n", bytes / 1048576.0, bpc, ms10 * 1e3, bytes / (ms10 * 1e-3) / 1e9);
  }
  {
    float ms = med_ms([&] { CHECK(hipMemcpyAsync(buf2, buf, big, hipMemcpyDeviceToDevice, 0)); });
    printf("copy   %7.1f MiB  %8.1f us  %7.1f GB/s (read+write)\n", big / 1048576.0, ms * 1e3, 2.0 * big / (ms * 1e-3) / 1e9);
  }
  {
    int iters = 4096, threads = 256, nb = p.multiProcessorCount * 40;
    float ms = med_ms([&] { fma_kernel<<<nb, threads>>>(out, iters, 1e-7f); });
    double flops = 2.0 * 8 * iters * (double)threads * nb;
    printf("fp32 FMA peak        %8.1f us  %7.2f TFLOPS  (=> effective %.0f MHz at 128 FLOP/CU/clk)\n", ms * 1e3,
           flops / (ms * 1e-3) / 1e12, flops / (ms * 1e-3) / (p.multiProcessorCount * 128.0) / 1e6);
  }
  return 0;
}
