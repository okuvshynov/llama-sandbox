// scratch: what in a background load slows a die's small kernels (README steps 86, 122)?
// Foreground on die 0: 400 small kernels then a synchronize ("small"), and one kernel reading 1 GiB ("big").
// Background: a second thread and stream, as the expert loader's (two buffers in turn, blocking events), running one of several patterns.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
using C = std::chrono::steady_clock;
static double ms(C::time_point a) { return std::chrono::duration<double, std::milli>(C::now() - a).count(); }
__global__ void small(float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = y[i] * 1.0001f + 1.0f; }
__global__ void big(const float* x, float* out, size_t per) {
  const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; float s = 0;
  for (size_t i = 0; i < per; i++) s += x[t * per + i];
  out[t] = s;
}
__global__ void unpack(const unsigned char* src, unsigned char* dst, size_t n) {
  const size_t t = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 16;
  for (size_t i = t; i < t + 16 && i < n; i++) dst[i] = src[i] ^ 1;
}
static const size_t MAT = 6270976;
struct Bg { int dev; bool copy, kernel, low_prio; int pause_us; const char* name; };
int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 60;
  hipSetDevice(0);
  hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  float *y, *x, *out; hipMalloc(&y, 20480 * 4); hipMalloc(&x, 1ull << 30); hipMalloc(&out, 1048576 * 4);
  hipMemset(x, 0, 1ull << 30); hipMemset(y, 0, 20480 * 4); hipDeviceSynchronize();
  const Bg modes[] = {
      {-1, false, false, true, 0, "nothing in the background"},
      {0, true, false, true, 0, "die 0: copies only (6.27 MB host to device)"},
      {0, false, true, true, 0, "die 0: unpack kernels only"},
      {0, true, true, true, 0, "die 0: copy + unpack (the loader's load)"},
      {0, true, true, false, 0, "die 0: copy + unpack, default priority"},
      {0, true, true, true, 3000, "die 0: copy + unpack, one per 3 ms"},
      {1, true, true, true, 0, "die 1 (same card): copy + unpack"},
      {2, true, true, true, 0, "die 2 (other card): copy + unpack"},
      {-1, false, false, true, 0, "nothing again"},
  };
  printf("%-46s %9s %9s %9s %9s %10s\n", "background", "small med", "small p90", "big med", "big p90", "bg per s");
  for (const Bg& m : modes) {
    std::atomic<bool> stop{false}; std::atomic<long> n_bg{0};
    std::thread bg;
    if (m.dev >= 0) bg = std::thread([&] {
      hipSetDevice(m.dev);
      int lo, hi; hipDeviceGetStreamPriorityRange(&lo, &hi);
      hipStream_t st; hipStreamCreateWithPriority(&st, hipStreamNonBlocking, m.low_prio ? lo : 0);
      unsigned char *host[2], *dstage[2], *slot; hipEvent_t fr[2];
      hipMalloc(&slot, MAT);
      for (int b = 0; b < 2; b++) {
        hipHostMalloc(&host[b], MAT, 0); memset(host[b], b, MAT); hipMalloc(&dstage[b], MAT);
        hipEventCreateWithFlags(&fr[b], hipEventDisableTiming | hipEventBlockingSync); hipEventRecord(fr[b], st);
      }
      for (int b = 0; !stop; b ^= 1) {
        hipEventSynchronize(fr[b]);
        if (m.copy) hipMemcpyAsync(dstage[b], host[b], MAT, hipMemcpyHostToDevice, st);
        if (m.kernel) unpack<<<(unsigned)(MAT / 16 / 256 + 1), 256, 0, st>>>(dstage[b], slot, MAT);
        hipEventRecord(fr[b], st);
        n_bg++;
        if (m.pause_us) std::this_thread::sleep_for(std::chrono::microseconds(m.pause_us));
      }
      hipStreamSynchronize(st);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    hipSetDevice(0);
    std::vector<double> ts, tb; const long n0 = n_bg; auto t_all = C::now();
    for (int r = 0; r < reps; r++) {
      auto t0 = C::now();
      for (int i = 0; i < 400; i++) small<<<80, 256, 0, s>>>(y, 20480);
      hipStreamSynchronize(s); ts.push_back(ms(t0));
      t0 = C::now();
      big<<<4096, 256, 0, s>>>(x, out, (1ull << 28) / 1048576);
      hipStreamSynchronize(s); tb.push_back(ms(t0));
    }
    const double wall = ms(t_all) / 1000; const long nb = n_bg - n0;
    stop = true; if (bg.joinable()) bg.join();
    std::sort(ts.begin(), ts.end()); std::sort(tb.begin(), tb.end());
    printf("%-46s %7.2f ms %7.2f ms %7.2f ms %7.2f ms %10.0f\n", m.name, ts[reps / 2], ts[reps * 9 / 10], tb[reps / 2], tb[reps * 9 / 10], nb / wall);
    fflush(stdout);
  }
  return 0;
}
