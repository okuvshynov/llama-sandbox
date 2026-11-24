// scratch: hsa_amd_memory_lock by number of GPUs and by the pages' physical layout
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double sys_s() { rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6; }
static std::vector<hsa_agent_t> gpus;
static hsa_status_t cb(hsa_agent_t a, void*) {
  hsa_device_type_t t; hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) gpus.push_back(a);
  return HSA_STATUS_SUCCESS;
}
static const size_t R = 6ull << 30;
static size_t piece = 64ull << 20;
static void run(const char* name, char* p, int n_agents) {
  const double s0 = sys_s(); auto t0 = Clock::now(); int bad = 0;
  for (char* q = p; q < p + R; q += piece) { void* o; bad += hsa_amd_memory_lock(q, piece, n_agents ? gpus.data() : nullptr, n_agents, &o) != HSA_STATUS_SUCCESS; }
  const double t = std::chrono::duration<double>(Clock::now() - t0).count(), s = sys_s() - s0;
  auto t1 = Clock::now();
  for (char* q = p; q < p + R; q += piece) hsa_amd_memory_unlock(q);
  const double tu = std::chrono::duration<double>(Clock::now() - t1).count();
  printf("%-44s %d GPU%s  lock %5.2f s = %5.2f GB/s (sys %.2f s, waiting %.2f s); unlock %.2f s%s\n", name, n_agents ? n_agents : (int)gpus.size(),
         n_agents ? " " : "+", t, 6.44 / t, s, t - s, tu, bad ? "  (FAILED)" : "");
  fflush(stdout);
}
static char* anon(bool huge, bool shuffled) {
  char* p = (char*)mmap(nullptr, R + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  p = (char*)(((uintptr_t)p + (2 << 20) - 1) & ~(uintptr_t)((2 << 20) - 1));
  madvise(p, R, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  std::vector<uint32_t> idx(R >> 12);
  for (size_t i = 0; i < idx.size(); i++) idx[i] = (uint32_t)i;
  if (shuffled) { std::mt19937 g(1); std::shuffle(idx.begin(), idx.end(), g); }
  for (uint32_t i : idx) p[(size_t)i << 12] = 1;
  return p;
}
int main(int argc, char** argv) {
  if (argc > 2) piece = (size_t)atoll(argv[2]) << 20;
  hsa_init(); hsa_iterate_agents(cb, nullptr);
  printf("# %zu GPUs; 6 GiB per case in %zu MB pieces\n", gpus.size(), piece >> 20);
  const int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  size_t off = 2ull << 20;
  auto file = [&] { char* p = base + off; off += R; madvise(p, R, MADV_POPULATE_READ); return p; };
  run("page cache (file read by cat)", file(), 1);
  run("page cache, a second region", file(), 1);
  run("page cache", file(), 2);
  run("page cache", file(), 4);
  run("page cache, all agents (as HIP)", file(), 0);
  char* a = anon(false, false);
  run("anonymous 4 KiB, touched in order", a, 1);
  run("anonymous 4 KiB, touched in order", a, 4);
  char* b = anon(false, true);
  run("anonymous 4 KiB, touched in random order", b, 1);
  run("anonymous 4 KiB, touched in random order", b, 4);
  char* c = anon(true, false);
  run("anonymous 2 MB huge pages", c, 1);
  run("anonymous 2 MB huge pages", c, 4);
  hsa_shut_down();
  return 0;
}
