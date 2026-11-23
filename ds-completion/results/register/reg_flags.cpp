// scratch: hipHostRegister flags and where its time goes (user / system)
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
using Clock = std::chrono::steady_clock;
static double cpu(int who) { rusage r; getrusage(RUSAGE_SELF, &r); const timeval& t = who ? r.ru_stime : r.ru_utime; return t.tv_sec + t.tv_usec * 1e-6; }
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  const size_t R = 6ull << 30, piece = 64ull << 20;
  hipSetDevice(0);
  int* d; hipMalloc(&d, 4);
  struct { const char* name; unsigned flags; bool populate; } cases[] = {
      {"read-only | portable (the loader's)", hipHostRegisterReadOnly | hipHostRegisterPortable, true},
      {"read-only", hipHostRegisterReadOnly, true},
      {"portable", hipHostRegisterPortable, true},
      {"default (0)", hipHostRegisterDefault, true},
  };
  size_t off = 2ull << 20;
  for (auto& c : cases) {
    char* p = base + off; off += R;
    if (c.populate && madvise(p, R, MADV_POPULATE_READ)) perror("madvise");
    const double u0 = cpu(0), s0 = cpu(1);
    auto t0 = Clock::now();
    int bad = 0;
    for (char* q = p; q < p + R; q += piece) bad += hipHostRegister(q, piece, c.flags) != hipSuccess;
    const double t = std::chrono::duration<double>(Clock::now() - t0).count();
    printf("%-38s %5.2f s = %4.2f GB/s; cpu user %.2f s, system %.2f s%s\n", c.name, t, 6.44 / t, cpu(0) - u0, cpu(1) - s0, bad ? "  (calls FAILED)" : "");
    fflush(stdout);
  }
  return 0;
}
