// LD_PRELOAD: count and time the HSA calls HIP makes under hipHostRegister
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <dlfcn.h>
#include <chrono>
#include <cstdio>
#include <atomic>
using Clock = std::chrono::steady_clock;
static std::atomic<long> n_lock{0}, n_agents{0}, n_bytes{0}, ns_lock{0}, n_pool{0}, ns_pool{0}, b_pool{0};
extern "C" hsa_status_t hsa_amd_memory_lock(void* p, size_t size, hsa_agent_t* a, int n, void** out) {
  static auto f = (hsa_status_t(*)(void*, size_t, hsa_agent_t*, int, void**))dlsym(RTLD_NEXT, "hsa_amd_memory_lock");
  auto t0 = Clock::now(); auto r = f(p, size, a, n, out);
  ns_lock += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
  n_lock++; n_agents += n; n_bytes += size; return r;
}
extern "C" hsa_status_t hsa_amd_memory_lock_to_pool(void* p, size_t size, hsa_agent_t* a, int n, hsa_amd_memory_pool_t pool, uint32_t flags, void** out) {
  static auto f = (hsa_status_t(*)(void*, size_t, hsa_agent_t*, int, hsa_amd_memory_pool_t, uint32_t, void**))dlsym(RTLD_NEXT, "hsa_amd_memory_lock_to_pool");
  auto t0 = Clock::now(); auto r = f(p, size, a, n, pool, flags, out);
  ns_pool += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
  n_pool++; n_agents += n; b_pool += size; return r;
}
__attribute__((destructor)) static void fin() {
  fprintf(stderr, "# shim: hsa_amd_memory_lock %ld calls, %.2f GiB, %.2f s; lock_to_pool %ld calls, %.2f GiB, %.2f s; agents passed in total %ld\n",
          n_lock.load(), n_bytes / 1073741824.0, ns_lock / 1e9, n_pool.load(), b_pool / 1073741824.0, ns_pool / 1e9, n_agents.load());
}
