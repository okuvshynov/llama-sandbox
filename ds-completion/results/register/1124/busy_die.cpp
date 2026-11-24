// scratch: does a lock for one die wait for that die's own work? Die 0 kept busy (a compute loop, or copies);
// the lock timed for die 0 alone and for die 1 alone, a fresh 6 GiB region of the fault-filled file each.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static double thread_sys() { rusage r; getrusage(RUSAGE_THREAD, &r); return r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }
static std::vector<hsa_agent_t> gpus;
static hsa_status_t cb(hsa_agent_t a, void*) {
  hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) gpus.push_back(a);
  return HSA_STATUS_SUCCESS;
}
static const size_t R = 6ull << 30, piece = 64ull << 20;
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  size_t off = 2ull << 20;
  hipSetDevice(0); char* big; hipMalloc(&big, 1ull << 30);
  char* pin; hipHostMalloc(&pin, 256ull << 20, 0);
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  const char* names[] = {"die 0 idle", "die 0 computing (hipMemset loop)", "die 0 copying (host-to-device loop)"};
  for (int busy = argc > 2 ? atoi(argv[2]) : 0; busy < (argc > 2 ? atoi(argv[2]) + 1 : 3); busy++) {
    std::atomic<bool> stop{false}; std::atomic<long> iters{0};
    std::thread w([&] {
      hipSetDevice(0);
      while (!stop) {
        if (busy == 1) { hipMemset(big, 1, 1ull << 30); hipDeviceSynchronize(); }
        else if (busy == 2) hipMemcpy(big, pin, 256ull << 20, hipMemcpyHostToDevice);
        else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        iters++;
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    { const long i0 = iters; std::this_thread::sleep_for(std::chrono::milliseconds(250)); printf("%-38s no lock running: the busy loop ran %ld times in 0.25 s\n", names[busy], iters - i0); }
    for (int g = 0; g < 4; g++) {
      const bool pause = g >= 2;
      char* p = base + off; off += R; madvise(p, R, MADV_POPULATE_READ);
      const long i0 = iters; const double s0 = thread_sys(); auto t0 = Clock::now(); int bad = 0; void* o;
      for (char* q = p; q < p + R; q += piece) bad += hsa_amd_memory_lock(q, piece, &gpus[(g % 2) * 2 + (argc > 3)], 1, &o) != HSA_STATUS_SUCCESS, pause ? std::this_thread::sleep_for(std::chrono::microseconds(500)) : (void)0;
      const double t = since(t0), s = thread_sys() - s0;
      printf("%-38s lock for die %d%s: %5.2f s = %5.1f GB/s (in the kernel %.2f s, waiting %.2f s); the busy loop ran %ld times meanwhile%s\n",
             names[busy], (g % 2) * 2 + (argc > 3), pause ? " with the loader's 0.5 ms pauses" : "", t, 6.44 / t, s, t - s, iters - i0, bad ? "  (FAILED)" : ""); fflush(stdout);
    }
    stop = true; w.join();
  }
  return 0;
}
