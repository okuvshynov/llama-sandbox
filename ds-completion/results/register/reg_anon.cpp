// registration rate of anonymous memory: 4 KiB pages vs transparent huge pages; and the time to fill it from a cached file
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char** argv) {
  const size_t G = 6ull << 30, P = 64ull << 20;
  (void)hipSetDevice(0); int* d; (void)hipMalloc(&d, 4);
  int fd = open(argv[1], O_RDONLY);
  for (int huge = 0; huge < 2; huge++) {
    char* a = (char*)mmap(0, G + (2 << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    a = (char*)(((uintptr_t)a + (2 << 20) - 1) & ~(uintptr_t)((2 << 20) - 1));
    madvise(a, G, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    double t0 = now();
    std::vector<std::thread> th;
    for (int t = 0; t < 16; t++) th.emplace_back([&, t] {
      size_t lo = G / 16 * t, hi = G / 16 * (t + 1);
      for (size_t o = lo; o < hi;) { ssize_t r = pread(fd, a + o, std::min<size_t>(hi - o, 8 << 20), (10ull << 30) + o); if (r <= 0) break; o += r; }
    });
    for (auto& x : th) x.join();
    double t1 = now();
    for (size_t o = 0; o < G; o += P)
      if (hipHostRegister(a + o, P, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess) { printf("register failed\n"); return 1; }
    double t2 = now();
    for (size_t o = 0; o < G; o += P) (void)hipHostUnregister(a + o);
    printf("%-28s fill from the cached file, 16 threads %.2f s = %.1f GB/s; register %.2f s = %.2f GB/s\n",
           huge ? "anonymous, 2 MB huge pages" : "anonymous, 4 KiB pages", t1 - t0, G / 1e9 / (t1 - t0), t2 - t1, G / 1e9 / (t2 - t1));
    fflush(stdout);
    munmap(a, G);
  }
}
