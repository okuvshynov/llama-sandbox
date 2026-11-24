// scratch: registering memory on 2 MB pages of the hugetlb pool (a memfd, shared, mapped read-only as the engine maps a shard)
//   ./reg_huge <shard> <mode>     mode: hip | hsa4 | hsa3 | hsa1 | cache-hip | cache-hsa4   (cache-*: the same on the page cache, for reference)
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static double cpu_s() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6 + r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6; }
static std::vector<hsa_agent_t> gpus;
static hsa_status_t cb(hsa_agent_t a, void*) {
  hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) gpus.push_back(a);
  return HSA_STATUS_SUCCESS;
}
int main(int argc, char** argv) {
  const std::string mode = argv[2];
  const size_t R = 6ull << 30, piece = 64ull << 20, off = 2ull << 20;
  const int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* file = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  char* p;
  if (mode.rfind("cache", 0) == 0) { p = file + off; madvise(p, R, MADV_POPULATE_READ); }
  else {
    const int mfd = memfd_create("ds-huge", MFD_HUGETLB);
    if (mfd < 0 || ftruncate(mfd, R)) { perror("memfd_create / ftruncate (the hugetlb pool?)"); return 1; }
    char* w = (char*)mmap(nullptr, R, PROT_READ | PROT_WRITE, MAP_SHARED, mfd, 0);
    if (w == MAP_FAILED) { perror("mmap of the huge pages"); return 1; }
    auto t0 = Clock::now(); std::vector<std::thread> th;
    for (int t = 0; t < 16; t++) th.emplace_back([&, t] { memcpy(w + R / 16 * t, file + off + R / 16 * t, R / 16); });
    for (auto& t : th) t.join();
    printf("# 6 GiB copied onto huge pages by 16 threads in %.2f s = %.1f GB/s\n", since(t0), 6.44 / since(t0));
    munmap(w, R);
    p = (char*)mmap(nullptr, R, PROT_READ, MAP_SHARED, mfd, 0);  // as the engine maps a shard
    if (p == MAP_FAILED) { perror("read-only mmap"); return 1; }
    if (memcmp(p, file + off, 64 << 20)) { printf("copy differs\n"); return 1; }
  }
  hipSetDevice(0); int* d0; hipMalloc(&d0, 4);
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  const double c0 = cpu_s(); auto t0 = Clock::now(); int bad = 0; void* o;
  for (char* q = p; q < p + R; q += piece) {
    if (mode == "hip" || mode == "cache-hip") bad += hipHostRegister(q, piece, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess;
    else if (mode == "hsa4" || mode == "cache-hsa4") bad += hsa_amd_memory_lock(q, piece, gpus.data(), 4, &o) != HSA_STATUS_SUCCESS;
    else if (mode == "hsa3") bad += hsa_amd_memory_lock(q, piece, gpus.data() + 1, 3, &o) != HSA_STATUS_SUCCESS;
    else if (mode == "hsa1") bad += hsa_amd_memory_lock(q, piece, gpus.data() + 1, 1, &o) != HSA_STATUS_SUCCESS;
  }
  const double t = since(t0);
  printf("%-10s register 6 GiB: %5.2f s = %6.1f GB/s (cpu %.2f s)%s; 289 GB at this rate: %.0f s\n", mode.c_str(), t, 6.44 / t, cpu_s() - c0, bad ? "  (FAILED)" : "", 289 / (6.44 / t));
  if (mode == "hip" || mode == "cache-hip") {  // HIP's direct path from the registered memory: rate and bytes
    hipSetDevice(1); hipStream_t s; hipStreamCreate(&s);
    const size_t chunk = 4ull << 20, total = 1ull << 30; char* dst; hipMalloc(&dst, 64ull << 20);
    const double c1 = cpu_s(); auto t1 = Clock::now(); int cb2 = 0;
    for (size_t x = 0; x < total; x += chunk) cb2 += hipMemcpyAsync(dst + (x % (64ull << 20)), p + x, chunk, hipMemcpyHostToDevice, s) != hipSuccess;
    hipStreamSynchronize(s); const double tc = since(t1);
    std::vector<char> back(chunk); hipMemcpy(back.data(), dst + (64ull << 20) - chunk, chunk, hipMemcpyDeviceToHost);
    printf("           copy to die 1 from it: %.2f GB/s, cpu %.2f ms per 18.8 MB, bytes %s%s\n", 1.074 / tc, (cpu_s() - c1) / (total / 18.8e6) * 1e3,
           memcmp(back.data(), p + total - chunk, chunk) ? "DIFFER" : "equal", cb2 ? " (calls FAILED)" : "");
  }
  fflush(stdout);
  if (argc > 3) sleep(atoi(argv[3]));  // hold, for a look at the cards' memory
  return 0;
}
