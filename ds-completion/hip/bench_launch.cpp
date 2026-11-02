// bench-launch: what a kernel launch costs on this machine (README step 49): back-to-back empty and
// small kernels on one stream (host submit and completion per launch), and a launch + synchronize
// round trip.
//   bin-hip/bench-launch
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
__global__ void empty(int* p) { if (threadIdx.x == 0 && p == nullptr) p[0] = 1; }
__global__ void small(float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = y[i] * 1.0001f + 1.0f; }
int main() {
  (void)hipSetDevice(0);
  hipStream_t s;
  (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  float* y;
  (void)hipMalloc(&y, 5120 * 4 * 4);
  using C = std::chrono::steady_clock;
  auto ms = [](C::time_point a) { return std::chrono::duration<double, std::milli>(C::now() - a).count(); };
  for (int rep = 0; rep < 2; rep++) {
    const int N = 20000;
    auto t0 = C::now();
    for (int i = 0; i < N; i++) empty<<<1, 64, 0, s>>>(nullptr);
    double sub = ms(t0);
    (void)hipStreamSynchronize(s);
    double tot = ms(t0);
    printf("empty kernel x%d: host submit %.2f us each, until done %.2f us each\n", N, sub * 1000 / N, tot * 1000 / N);
    t0 = C::now();
    for (int i = 0; i < N; i++) small<<<80, 256, 0, s>>>(y, 20480);
    sub = ms(t0); (void)hipStreamSynchronize(s); tot = ms(t0);
    printf("small kernel (20K floats) x%d: host submit %.2f us each, until done %.2f us each\n", N, sub * 1000 / N, tot * 1000 / N);
    // a host round trip: launch + sync each time (what a host wait in the middle of a layer costs)
    t0 = C::now();
    for (int i = 0; i < 2000; i++) { small<<<80, 256, 0, s>>>(y, 20480); (void)hipStreamSynchronize(s); }
    printf("launch + synchronize, round trip: %.2f us each\n", ms(t0) * 1000 / 2000);
  }
  return 0;
}
