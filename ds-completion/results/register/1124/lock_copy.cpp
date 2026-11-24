// scratch: no registration up front. Per expert (3 x 6.27 MB): hsa_amd_memory_lock for the one die that loads it,
// then hsa_amd_memory_async_copy into memory that HIP allocated. 1 die, then 4 dies at once (a thread each).
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static double thread_cpu() { rusage r; getrusage(RUSAGE_THREAD, &r); return r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6 + r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6; }
static std::vector<hsa_agent_t> gpus; static hsa_agent_t cpu_agent;
static hsa_status_t cb(hsa_agent_t a, void*) {
  hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) gpus.push_back(a); else if (t == HSA_DEVICE_TYPE_CPU) cpu_agent = a;
  return HSA_STATUS_SUCCESS;
}
static const size_t MAT = 6270976, EXPERT = 3 * MAT;
static char* base; static char* dst[4];
struct Res { double wall = 0, cpu = 0, lock = 0; int bad = 0; bool equal = true; };
// die d loads n experts starting at file offset off; mode 0: lock + hsa copy; 1: hsa copy from memory locked before
static Res run(int d, size_t off, int n, bool lock_now) {
  Res r; hipSetDevice(d);
  hsa_signal_t sig; hsa_signal_create(0, 0, nullptr, &sig);
  const double c0 = thread_cpu(); auto t0 = Clock::now();
  for (int i = 0; i < n; i++) {
    char* e = base + off + (size_t)i * EXPERT;
    if (lock_now) {
      char* a = (char*)((uintptr_t)e & ~(uintptr_t)4095); const size_t len = (((uintptr_t)e + EXPERT + 4095) & ~(uintptr_t)4095) - (uintptr_t)a;
      void* o; auto tl = Clock::now();
      r.bad += hsa_amd_memory_lock(a, len, &gpus[d], 1, &o) != HSA_STATUS_SUCCESS;
      r.lock += since(tl);
    }
    for (int m = 0; m < 3; m++) {
      hsa_signal_store_relaxed(sig, 1);
      r.bad += hsa_amd_memory_async_copy(dst[d] + m * MAT, gpus[d], e + m * MAT, cpu_agent, MAT, 0, nullptr, sig) != HSA_STATUS_SUCCESS;
      hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    }
  }
  r.wall = since(t0); r.cpu = thread_cpu() - c0;
  std::vector<char> back(EXPERT);
  hipMemcpy(back.data(), dst[d], EXPERT, hipMemcpyDeviceToHost);
  r.equal = !memcmp(back.data(), base + off + (size_t)(n - 1) * EXPERT, EXPERT);
  hsa_signal_destroy(sig);
  return r;
}
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY); const int N = argc > 2 ? atoi(argv[2]) : 300;
  struct stat st; fstat(fd, &st);
  base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  for (int d = 0; d < 4; d++) { hipSetDevice(d); hipMalloc(&dst[d], EXPERT); }
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  const size_t span = (size_t)N * EXPERT;  // per die per case
  printf("# %d experts of %.1f MB per die per case\n", N, EXPERT / 1e6);
  auto show = [&](const char* name, std::vector<Res> rs, double wall) {
    double cpu = 0, lock = 0; int bad = 0; bool eq = true;
    for (auto& r : rs) cpu += r.cpu, lock += r.lock, bad += r.bad, eq = eq && r.equal;
    const double loads = (double)N * rs.size();
    printf("%-44s %6.0f loads/s = %5.2f GB/s; cpu %.2f ms per load, of it the lock %.2f ms (wall); bytes %s%s\n", name, loads / wall, loads * EXPERT / 1e9 / wall,
           cpu / loads * 1e3, lock / loads * 1e3, eq ? "equal" : "DIFFER", bad ? "  (calls FAILED)" : ""); fflush(stdout);
  };
  size_t off = 4096 * 3 + 1000;  // not page-aligned, as the tensors
  for (int rep = 0; rep < 2; rep++) {
    { auto t0 = Clock::now(); Res r = run(1, off, N, true); show("1 die: lock per expert + copy", {r}, since(t0)); }
    { auto t0 = Clock::now(); Res r = run(1, off, N, false); show("1 die: the same experts again, copy only", {r}, since(t0)); }
    off += span;
    std::vector<Res> rs(4); std::vector<std::thread> th; auto t0 = Clock::now();
    for (int d = 0; d < 4; d++) th.emplace_back([&, d] { rs[d] = run(d, off + d * span, N, true); });
    for (auto& t : th) t.join();
    show("4 dies at once: lock per expert + copy", rs, since(t0));
    th.clear(); t0 = Clock::now();
    for (int d = 0; d < 4; d++) th.emplace_back([&, d] { rs[d] = run(d, off + d * span, N, false); });
    for (auto& t : th) t.join();
    show("4 dies at once: again, copy only", rs, since(t0));
    off += 4 * span;
  }
  return 0;
}
