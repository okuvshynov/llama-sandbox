// LD_PRELOAD shim: counts HIP kernel launches (hipLaunchKernel: what <<<>>> compiles to), hipMemcpyAsync,
// hipEventRecord and hipStreamWaitEvent; prints the totals at exit (README step 49).
//   gcc -O2 -shared -fPIC -o /tmp/hip_count_shim.so scripts/hip_count_shim.c -ldl
//   LD_PRELOAD=/tmp/hip_count_shim.so bin-hip/ds-completion ...
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdatomic.h>
#include <stdint.h>
static atomic_long n_launch, n_memcpy, n_record, n_wait;
typedef int (*launch_t)(const void*, unsigned long long, unsigned long long, void**, size_t, void*);
// hipLaunchKernel(const void* f, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t s): dim3 is 12 bytes; pass through as raw ints
typedef struct { unsigned x, y, z; } d3;
typedef int (*launch2_t)(const void*, d3, d3, void**, size_t, void*);
int hipLaunchKernel(const void* f, d3 g, d3 b, void** a, size_t sh, void* s) {
  static launch2_t real; if (!real) real = (launch2_t)dlsym(RTLD_NEXT, "hipLaunchKernel");
  atomic_fetch_add(&n_launch, 1); return real(f, g, b, a, sh, s);
}
typedef int (*memcpy_t)(void*, const void*, size_t, int, void*);
int hipMemcpyAsync(void* d, const void* s_, size_t n, int k, void* s) {
  static memcpy_t real; if (!real) real = (memcpy_t)dlsym(RTLD_NEXT, "hipMemcpyAsync");
  atomic_fetch_add(&n_memcpy, 1); return real(d, s_, n, k, s);
}
typedef int (*rec_t)(void*, void*);
int hipEventRecord(void* e, void* s) {
  static rec_t real; if (!real) real = (rec_t)dlsym(RTLD_NEXT, "hipEventRecord");
  atomic_fetch_add(&n_record, 1); return real(e, s);
}
typedef int (*wait_t)(void*, void*, unsigned);
int hipStreamWaitEvent(void* s, void* e, unsigned f) {
  static wait_t real; if (!real) real = (wait_t)dlsym(RTLD_NEXT, "hipStreamWaitEvent");
  atomic_fetch_add(&n_wait, 1); return real(s, e, f);
}
__attribute__((destructor)) static void report(void) {
  fprintf(stderr, "# shim: %ld kernel launches, %ld hipMemcpyAsync, %ld hipEventRecord, %ld hipStreamWaitEvent\n",
          (long)n_launch, (long)n_memcpy, (long)n_record, (long)n_wait);
}
