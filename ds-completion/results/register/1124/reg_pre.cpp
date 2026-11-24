// scratch: does one hsa_amd_memory_lock for all GPUs make the hipHostRegister after it cheap?
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <random>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double cpu_s() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6 + r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6; }
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static std::vector<hsa_agent_t> gpus;
static hsa_status_t cb(hsa_agent_t a, void*) {
  hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) gpus.push_back(a);
  return HSA_STATUS_SUCCESS;
}
static const size_t R = 6ull << 30, piece = 64ull << 20;
static hipStream_t s; static char* dst;
template <class F> static void timed(const char* name, F f) {
  const double c0 = cpu_s(); auto t0 = Clock::now(); const int bad = f(); const double t = since(t0);
  printf("  %-46s %5.2f s = %6.2f GB/s (cpu %.2f s)%s\n", name, t, 6.44 / t, cpu_s() - c0, bad ? "  (FAILED)" : ""); fflush(stdout);
}
static int hsa_lock(char* p, int n) { int bad = 0; void* o; for (char* q = p; q < p + R; q += piece) bad += hsa_amd_memory_lock(q, piece, n ? gpus.data() : nullptr, n, &o) != HSA_STATUS_SUCCESS; return bad; }
static int hip_reg(char* p) { int bad = 0; for (char* q = p; q < p + R; q += piece) bad += hipHostRegister(q, piece, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess; return bad; }
static int hip_unreg(char* p) { int bad = 0; for (char* q = p; q < p + R; q += piece) bad += hipHostUnregister(q) != hipSuccess; return bad; }
static int hsa_unlock(char* p) { int bad = 0; for (char* q = p; q < p + R; q += piece) bad += hsa_amd_memory_unlock(q) != HSA_STATUS_SUCCESS; return bad; }
static void copies(char* src) {  // 2 GiB to each of the 4 dies in 4 MB chunks, a region not copied before for each die
  const size_t chunk = 4ull << 20, total = 1ull << 30;
  for (int d = 0; d < 4; d++) {
    hipSetDevice(d);
    static hipStream_t st[4]; static char* ds[4];
    if (!st[d]) { hipStreamCreate(&st[d]); hipMalloc(&ds[d], 64ull << 20); }
    const double c0 = cpu_s(); auto t0 = Clock::now(); int bad = 0;
    for (size_t o = 0; o < total; o += chunk) bad += hipMemcpyAsync(ds[d] + (o % (64ull << 20)), src + d * total + o, chunk, hipMemcpyHostToDevice, st[d]) != hipSuccess;
    hipStreamSynchronize(st[d]);
    const double t = since(t0);
    printf("    copy to die %d: %5.2f GB/s, cpu %.2f ms per 18.8 MB%s\n", d, 1.074 / t, (cpu_s() - c0) / (total / 18.8e6) * 1e3, bad ? "  (FAILED)" : "");
  }
  hipSetDevice(0); fflush(stdout);
}
static char* anon_shuffled() {
  char* p = (char*)mmap(nullptr, R + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  p = (char*)(((uintptr_t)p + (2 << 20) - 1) & ~(uintptr_t)((2 << 20) - 1));
  madvise(p, R, MADV_NOHUGEPAGE);
  std::vector<uint32_t> idx(R >> 12);
  for (size_t i = 0; i < idx.size(); i++) idx[i] = (uint32_t)i;
  std::mt19937 g(1); std::shuffle(idx.begin(), idx.end(), g);
  for (uint32_t i : idx) p[(size_t)i << 12] = 1;
  return p;
}
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY);
  struct stat stt; fstat(fd, &stt);
  char* base = (char*)mmap(nullptr, stt.st_size, PROT_READ, MAP_SHARED, fd, 0);
  size_t off = 2ull << 20;
  auto file = [&] { char* p = base + off; off += R; madvise(p, R, MADV_POPULATE_READ); return p; };
  hipSetDevice(0); int* d0; hipMalloc(&d0, 4);
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  for (int kind = 0; kind < 2; kind++) {
    printf("# %s\n", kind ? "anonymous 4 KiB pages touched in random order (scattered)" : "page cache, the file read sequentially after boot");
    char* a = kind ? anon_shuffled() : file();
    char* b = kind ? anon_shuffled() : file();
    char* c = kind ? anon_shuffled() : file();
    printf(" A. hipHostRegister alone (today's loader)\n");
    timed("hipHostRegister", [&] { return hip_reg(a); });
    copies(a);
    timed("hipHostUnregister", [&] { return hip_unreg(a); });
    printf(" B. one hsa_amd_memory_lock for the 4 GPUs, then hipHostRegister\n");
    timed("hsa_amd_memory_lock, 4 agents", [&] { return hsa_lock(b, 4); });
    timed("hipHostRegister after it", [&] { return hip_reg(b); });
    copies(b);
    timed("hipHostUnregister", [&] { return hip_unreg(b); });
    timed("hsa_amd_memory_unlock", [&] { return hsa_unlock(b); });
    printf(" C. one hsa_amd_memory_lock for all agents (NULL, 0), then hipHostRegister\n");
    timed("hsa_amd_memory_lock, all agents", [&] { return hsa_lock(c, 0); });
    timed("hipHostRegister after it", [&] { return hip_reg(c); });
    copies(c);
    timed("hipHostUnregister", [&] { return hip_unreg(c); });
    timed("hsa_amd_memory_unlock", [&] { return hsa_unlock(c); });
  }
  return 0;
}
