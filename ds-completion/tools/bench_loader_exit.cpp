// The expert loader's shutdown alone, many times (README step 96: a rare abort at exit).
// One thread per die creates what ExpertLoader::work() holds (a low-priority stream, 2 pinned host
// buffers, 2 device buffers, 3 events), uses the stream once, idles on a condition variable, and on
// stop all threads free them at the same moment, as ~ExpertLoader made them before step 96.
//
//   bin-hip/bench-loader-exit [N = 1000] [serial]     serial: the frees one thread at a time (step 96's change)
//
// ~37 ms per iteration; an abort or a crash is the finding, a "done" line the absence of one.
#include <hip/hip_runtime.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_));                 \
      abort();                                                                \
    }                                                                         \
  } while (0)

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 1000;
  const bool serial = argc > 2 && !strcmp(argv[2], "serial");
  const size_t bytes = (size_t)2304 * 5120 / 32 * 17;  // one mxfp4 expert matrix
  int nd = 0;
  CHECK(hipGetDeviceCount(&nd));
  std::mutex free_mu;
  for (int it = 0; it < N; it++) {
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false;
    std::atomic<int> ready{0};
    std::vector<std::thread> th;
    for (int d = 0; d < nd; d++)
      th.emplace_back([&, d] {
        int lo, hi;
        CHECK(hipSetDevice(d));
        CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));
        hipStream_t s;
        void *host[2], *dev[2];
        hipEvent_t fr[2], fin;
        CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, lo));
        for (int b = 0; b < 2; b++) {
          CHECK(hipHostMalloc(&host[b], bytes, 0));
          CHECK(hipMalloc(&dev[b], bytes));
          CHECK(hipEventCreateWithFlags(&fr[b], hipEventDisableTiming | hipEventBlockingSync));
          CHECK(hipEventRecord(fr[b], s));
        }
        CHECK(hipEventCreateWithFlags(&fin, hipEventDisableTiming | hipEventBlockingSync));
        CHECK(hipMemcpyAsync(dev[0], host[0], 4096, hipMemcpyHostToDevice, s));
        CHECK(hipEventRecord(fin, s));
        CHECK(hipEventSynchronize(fin));
        ready++;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&] { return stop; });
        }
        std::unique_lock<std::mutex> fl(free_mu, std::defer_lock);
        if (serial) fl.lock();
        (void)hipEventDestroy(fin);
        (void)hipStreamSynchronize(s);
        for (int b = 0; b < 2; b++) {
          (void)hipHostFree(host[b]);
          (void)hipFree(dev[b]);
          (void)hipEventDestroy(fr[b]);
        }
        (void)hipStreamDestroy(s);
      });
    while (ready < nd) std::this_thread::yield();
    {
      std::lock_guard<std::mutex> lk(mu);
      stop = true;
    }
    cv.notify_all();
    for (auto& t : th) t.join();
    if (it % 200 == 0) {
      fprintf(stderr, "iteration %d\n", it);
      fflush(stderr);
    }
  }
  fprintf(stderr, "done: %d iterations, %d dies, %s frees\n", N, nd, serial ? "serialized" : "concurrent");
  return 0;
}
