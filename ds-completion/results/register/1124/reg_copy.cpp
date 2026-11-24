// scratch: (1) lock calls shaped as HIP's against one call; (2) hipMemcpyAsync from memory that is
// pageable, registered with HIP, or locked with hsa_amd_memory_lock
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
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
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  size_t off = 2ull << 20;
  auto file = [&] { char* p = base + off; off += R; madvise(p, R, MADV_POPULATE_READ); return p; };
  hipSetDevice(0); int* d0; hipMalloc(&d0, 4);  // HIP (and with it HSA) up
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  printf("# %zu GPUs; 6 GiB per case, 64 MB pieces\n", gpus.size());
  bool keep = false;
  auto lock = [&](const char* name, char* p, int mode) {
    const double c0 = cpu_s(); auto t0 = Clock::now(); int bad = 0; void* o;
    for (char* q = p; q < p + R; q += piece) {
      if (mode == 0) bad += hsa_amd_memory_lock(q, piece, gpus.data(), 4, &o) != HSA_STATUS_SUCCESS;       // one call, 4 agents
      if (mode == 1) for (int g = 0; g < 4; g++) bad += hsa_amd_memory_lock(q, piece, &gpus[g], 1, &o) != HSA_STATUS_SUCCESS;  // as HIP: 4 calls
      if (mode >= 10) bad += hsa_amd_memory_lock(q, piece, &gpus[mode - 10], 1, &o) != HSA_STATUS_SUCCESS;  // one GPU
      if (mode == 2) bad += hsa_amd_memory_lock(q, piece, gpus.data() + 1, 3, &o) != HSA_STATUS_SUCCESS;   // GPUs 1-3
      if (mode == 3) bad += hipHostRegister(q, piece, hipHostRegisterReadOnly) != hipSuccess;
    }
    const double t = since(t0);
    printf("%-52s %5.2f s = %6.2f GB/s (cpu %.2f s)%s\n", name, t, 6.44 / t, cpu_s() - c0, bad ? "  (FAILED)" : ""); fflush(stdout);
    if (!keep) for (char* q = p; q < p + R; q += piece) for (int k = 0; k < (mode == 1 ? 4 : 1); k++) hsa_amd_memory_unlock(q);
  };
  for (int g = 0; g < 4; g++) { char n[64]; snprintf(n, sizeof n, "lock: one call, GPU %d alone", g); lock(n, file(), 10 + g); }
  lock("lock: one call, 4 agents", file(), 0);
  lock("lock: one call, GPUs 1-3", file(), 2);
  lock("lock: 4 calls of 1 agent on the same piece (HIP's)", file(), 1);
  keep = true;
  char* reg = base + (2ull << 20);  lock("hipHostRegister", reg, 3);
  char* hsa = reg + R;  lock("lock: one call, 4 agents (for the copies below)", hsa, 0);
  char* plain = hsa + R;
  // copies: 2 GiB from each kind of memory to GPU 1, in 4 MB chunks, two in flight at most is not kept: all queued, then a wait
  hipSetDevice(1);
  hipStream_t s; hipStreamCreate(&s);
  const size_t chunk = 4ull << 20, total = 2ull << 30;
  char* dst; hipMalloc(&dst, 64ull << 20);
  char* pin; hipHostMalloc(&pin, 64ull << 20, 0);
  auto copy = [&](const char* name, char* src, bool via_pinned) {
    for (int rep = 0; rep < 2; rep++) {
      const double c0 = cpu_s(); auto t0 = Clock::now(); int bad = 0;
      for (size_t o = 0; o < total; o += chunk) {
        const char* from = src + o;
        if (via_pinned) { hipStreamSynchronize(s); __builtin_memcpy(pin, from, chunk); from = pin; }
        bad += hipMemcpyAsync(dst + (o % (64ull << 20)), from, chunk, hipMemcpyHostToDevice, s) != hipSuccess;
      }
      hipStreamSynchronize(s);
      const double t = since(t0);
      printf("copy %-36s rep %d  %5.2f s = %5.2f GB/s, cpu %.2f s = %.2f ms per 18.8 MB%s\n", name, rep, t, 2.147 / t, cpu_s() - c0,
             (cpu_s() - c0) / (total / 18.8e6) * 1e3, bad ? "  (FAILED)" : ""); fflush(stdout);
    }
  };
  copy("registered with HIP", reg, false);
  copy("locked with hsa_amd_memory_lock", hsa, false);
  copy("pageable (the file mapping as it is)", plain, false);
  copy("through a pinned buffer (memcpy first)", plain, true);
  // do the copied bytes arrive? compare one chunk from the hsa-locked source
  { std::vector<char> back(chunk); hipMemcpyAsync(dst, hsa + 5 * chunk, chunk, hipMemcpyHostToDevice, s); hipStreamSynchronize(s);
    hipMemcpy(back.data(), dst, chunk, hipMemcpyDeviceToHost);
    printf("# a chunk copied from the hsa-locked memory read back: %s\n", __builtin_memcmp(back.data(), hsa + 5 * chunk, chunk) ? "DIFFERS" : "equal"); }
  return 0;
}
