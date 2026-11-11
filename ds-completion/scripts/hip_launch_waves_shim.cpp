// LD_PRELOAD shim (README step 102): counts every hipLaunchKernel of a run by (kernel, waves), to
// find the launches whose wave count the GPU dispatches slowly (2017-2048 waves: ~48 us whatever
// the kernel does; bench-wave-slots). Writes "count waves kernel-offset" lines to $LAUNCHLOG at
// exit; scripts/launch_waves.py names the kernels from the binary's symbols and flags them.
//
//   echo 'hip_4.2 { global: hipLaunchKernel; };' > /tmp/v.map
//   g++ -O2 -std=c++17 -shared -fPIC -I/opt/rocm/include -Wl,--version-script=/tmp/v.map \
//       -o /tmp/launch_waves.so scripts/hip_launch_waves_shim.cpp -ldl
//   LAUNCHLOG=/tmp/launches.txt LD_PRELOAD=/tmp/launch_waves.so bin-hip/ds-completion ...
//   python3 scripts/launch_waves.py /tmp/launches.txt bin-hip/ds-completion
#define __HIP_PLATFORM_AMD__ 1
#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

static std::mutex mu;
static std::map<std::pair<const void*, long>, long> counts;

static struct AtExit {
  ~AtExit() {
    if (counts.empty()) return;  // a child process that launched nothing
    FILE* f = fopen(getenv("LAUNCHLOG") ? getenv("LAUNCHLOG") : "/dev/stderr", "w");
    if (!f) return;
    for (const auto& [k, n] : counts) {
      Dl_info di;
      const char* base = dladdr(k.first, &di) && di.dli_fbase ? (const char*)di.dli_fbase : nullptr;
      fprintf(f, "%ld %ld 0x%lx\n", n, k.second, (unsigned long)((const char*)k.first - base));
    }
    fclose(f);
  }
} at_exit;

extern "C" hipError_t hipLaunchKernel(const void* fn, dim3 grid, dim3 block, void** args, size_t shm, hipStream_t s) {
  static const auto real =
      (hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t))dlsym(RTLD_NEXT, "hipLaunchKernel");
  const long waves = (long)grid.x * grid.y * grid.z * ((block.x * block.y * block.z + 63) / 64);
  {
    std::lock_guard<std::mutex> l(mu);
    counts[{fn, waves}]++;
  }
  return real(fn, grid, block, args, shm, s);
}
