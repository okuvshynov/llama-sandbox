// scratch: the stalls of a die's kernels while a copy lands on it (README steps 86, 122).
// Foreground on die 0: chains of 100 kernels of three sizes (tiny ~5 us, mid ~30 us, long ~80 us), a synchronize after each chain.
// Background: one pattern of copies on a second thread and stream (two buffers in turn, blocking events, as the expert loader).
// One background mode per process (argv[1]), so no mode can depend on what ran before it.
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
static double us(C::time_point a) { return std::chrono::duration<double, std::micro>(C::now() - a).count(); }
__global__ void tiny(float* y, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = y[i] * 1.0001f + 1.0f; }
// one row per thread: a dot product over K floats (a matrix read, as the runtime's small projections)
__global__ void rows(const float* w, const float* x, float* y, int R, int K, size_t stride) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= R) return;
  float s = 0; for (int k = 0; k < K; k++) s += w[(size_t)r * stride + k] * x[k];
  y[r] = s;
}
struct Bg { int dev; int dir; size_t piece; int pause_us; const char* name; };  // dir: 1 host to device, 2 device to host, 3 device to device
static const size_t MAT = 6270976;
static const Bg modes[] = {
    {-1, 0, 0, 0, "nothing"},
    {0, 1, MAT, 0, "to die 0: 6.27 MB, back to back"},
    {0, 1, MAT, 1500, "to die 0: 6.27 MB, one per 1.5 ms"},
    {0, 1, 1 << 20, 0, "to die 0: 1 MB pieces, back to back"},
    {0, 1, 1 << 18, 0, "to die 0: 256 KB pieces, back to back"},
    {0, 1, 1 << 16, 0, "to die 0: 64 KB pieces, back to back"},
    {0, 1, 1 << 20, 400, "to die 0: 1 MB per 0.4 ms (2.6 GB/s)"},
    {0, 1, 1 << 18, 100, "to die 0: 256 KB per 0.1 ms (2.6 GB/s)"},
    {1, 1, MAT, 0, "to die 1 (same card): 6.27 MB, back to back"},
    {2, 1, MAT, 0, "to die 2 (other card): 6.27 MB, back to back"},
    {0, 2, MAT, 0, "from die 0 to the host: 6.27 MB, back to back"},
    {0, 3, MAT, 0, "within die 0, device to device: 6.27 MB"},
    {0, 3, 1 << 14, 50, "within die 0: 16 KB per 50 us"},
};
int main(int argc, char** argv) {
  const int mode = argc > 1 ? atoi(argv[1]) : 0, reps = argc > 2 ? atoi(argv[2]) : 100;
  if (mode < 0) { printf("%zu\n", sizeof modes / sizeof modes[0]); return 0; }
  const Bg& m = modes[mode];
  hipSetDevice(0);
  hipStream_t s; hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  const int R = 4096, Kmax = 4096; const size_t spread = argc > 3 ? (size_t)atoll(argv[3]) << 30 : 0; const size_t wbytes = spread ? spread : (size_t)R * Kmax * 4, stride = wbytes / 4 / R;
  float *w, *x, *y, *yt; if (hipMalloc(&w, wbytes) != hipSuccess) { printf("hipMalloc failed\n"); return 1; } hipMalloc(&x, Kmax * 4); hipMalloc(&y, R * 4); hipMalloc(&yt, 20480 * 4);
  hipMemset(w, 0, wbytes); hipMemset(x, 0, Kmax * 4); hipMemset(yt, 0, 20480 * 4); hipDeviceSynchronize();
  auto chain = [&](int K) {  // us per kernel over a chain of 100
    auto t0 = C::now();
    for (int i = 0; i < 100; i++) { if (K) rows<<<(R + 255) / 256, 256, 0, s>>>(w, x, y, R, K, stride); else tiny<<<80, 256, 0, s>>>(yt, 20480); }
    hipStreamSynchronize(s); return us(t0) / 100;
  };
  auto med = [&](int K, int n) { std::vector<double> v; for (int i = 0; i < n; i++) v.push_back(chain(K)); std::sort(v.begin(), v.end()); return v[n / 2]; };
  // rows for ~30 us and ~80 us per kernel, found with nothing in the background
  int Rmid = 1, Rlong = 1;  // here: the columns per row
  for (int k = 1; k <= Kmax; k += (k < 64 ? 1 : k / 16)) { const double t = med(k, 5); if (t < 30) Rmid = k; if (t < 80) Rlong = k; else break; }
  std::atomic<bool> stop{false}; std::atomic<long> n_bg{0};
  std::thread bg;
  if (m.dev >= 0) bg = std::thread([&] {
    hipSetDevice(m.dev);
    int lo, hi; hipDeviceGetStreamPriorityRange(&lo, &hi);
    hipStream_t st; hipStreamCreateWithPriority(&st, hipStreamNonBlocking, lo);
    unsigned char *host[2], *dstage[2], *slot; hipEvent_t fr[2];
    hipMalloc(&slot, MAT);
    for (int b = 0; b < 2; b++) {
      hipHostMalloc(&host[b], MAT, 0); memset(host[b], b, MAT); hipMalloc(&dstage[b], MAT);
      hipEventCreateWithFlags(&fr[b], hipEventDisableTiming | hipEventBlockingSync); hipEventRecord(fr[b], st);
    }
    size_t off = 0;
    for (int b = 0; !stop; b ^= 1) {
      hipEventSynchronize(fr[b]);
      if (off + m.piece > MAT) off = 0;
      if (m.dir == 1) hipMemcpyAsync(dstage[b] + off, host[b] + off, m.piece, hipMemcpyHostToDevice, st);
      if (m.dir == 2) hipMemcpyAsync(host[b] + off, dstage[b] + off, m.piece, hipMemcpyDeviceToHost, st);
      if (m.dir == 3) hipMemcpyAsync(slot + off, dstage[b] + off, m.piece, hipMemcpyDeviceToDevice, st);
      hipEventRecord(fr[b], st);
      off += m.piece; n_bg++;
      if (m.pause_us) std::this_thread::sleep_for(std::chrono::microseconds(m.pause_us));
    }
    hipStreamSynchronize(st);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  hipSetDevice(0);
  const long n0 = n_bg; auto t_all = C::now();
  std::vector<double> a, b, c;
  for (int r = 0; r < reps; r++) { a.push_back(chain(0)); b.push_back(chain(Rmid)); c.push_back(chain(Rlong)); }
  const double wall = us(t_all) / 1e6; const long nb = n_bg - n0;
  stop = true; if (bg.joinable()) bg.join();
  for (auto* v : {&a, &b, &c}) std::sort(v->begin(), v->end());
  printf("%-46s tiny %5.1f us  mid %6.1f us  long %6.1f us   %6.0f copies/s = %5.2f GB/s  (4096 rows over %zu GiB; mid %d columns, long %d)\n", m.name, a[reps / 2], b[reps / 2], c[reps / 2],
         nb / wall, nb / wall * m.piece / 1e9, wbytes >> 30, Rmid, Rlong);
  return 0;
}
