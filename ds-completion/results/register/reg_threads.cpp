// scratch: what does hipHostRegister of page-cache memory cost, and does it scale with threads?
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static double s_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
int main(int argc, char** argv) {
  const char* file = argv[1];
  const double gb = atof(argv[2]);          // region size per case
  const int fd = open(file, O_RDONLY);
  struct stat st; fstat(fd, &st);
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  const size_t R = (size_t)(gb * (1ull << 30));
  hipSetDevice(0);
  int* dummy; hipMalloc(&dummy, 4);  // the runtime is up
  size_t off = 2ull << 20;
  auto region = [&]() { char* p = base + off; off += R; if (off > (size_t)st.st_size) { fprintf(stderr, "file too small\n"); exit(1); } return p; };
  auto reg = [&](char* p, size_t n, size_t piece, int threads, std::vector<char*>* out) {
    std::vector<std::thread> th;
    std::vector<std::vector<char*>> done(threads);
    const size_t per = n / threads / piece * piece;
    for (int t = 0; t < threads; t++)
      th.emplace_back([&, t] {
        for (char* q = p + t * per; q < p + (t + 1) * per; q += piece) {
          if (hipHostRegister(q, piece, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess) { fprintf(stderr, "register failed\n"); exit(1); }
          done[t].push_back(q);
        }
      });
    for (auto& x : th) x.join();
    for (auto& d : done) out->insert(out->end(), d.begin(), d.end());
  };
  auto populate = [&](char* p, size_t n, int threads) {
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++)
      th.emplace_back([&, t] { if (madvise(p + t * (n / threads), n / threads, MADV_POPULATE_READ)) perror("madvise"); });
    for (auto& x : th) x.join();
  };
  struct Case { const char* name; size_t piece_mb; int threads; int populate_threads; };
  const Case cases[] = {
      {"64 MB pieces, 1 thread (as the loader)", 64, 1, 0},   {"the same again (a second region)", 64, 1, 0},
      {"1024 MB pieces, 1 thread", 1024, 1, 0},               {"64 MB pieces, 4 threads", 64, 4, 0},
      {"64 MB pieces, 16 threads", 64, 16, 0},                {"populated first by 16 threads; 64 MB, 1 thread", 64, 1, 16},
      {"populated first by 16 threads; 64 MB, 16 threads", 64, 16, 16},
  };
  std::vector<char*> all;
  for (const Case& c : cases) {
    char* p = region();
    double tp = 0;
    if (c.populate_threads) { auto t0 = Clock::now(); populate(p, R, c.populate_threads); tp = s_since(t0); }
    auto t0 = Clock::now();
    reg(p, R, c.piece_mb << 20, c.threads, &all);
    const double t = s_since(t0);
    printf("%-52s populate %5.2f s, register %6.2f s = %5.2f GB/s (%.2f GB/s with the populate)\n", c.name, tp, t, gb * 1.0737 / t, gb * 1.0737 / (t + tp));
    fflush(stdout);
  }
  auto t0 = Clock::now();
  for (char* q : all) hipHostUnregister(q);
  printf("unregister %zu pieces: %.2f s\n", all.size(), s_since(t0));
  return 0;
}
