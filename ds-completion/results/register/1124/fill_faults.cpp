// scratch: drop a file from the page cache, then bring its first G GiB back by page faults as the engine does:
// 16 threads, each reading whole 6.27 MB blocks (an expert's matrix), the blocks in random order
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
int main(int argc, char** argv) {
  const int fd = open(argv[1], O_RDONLY);
  const size_t G = (size_t)atoll(argv[2]) << 30, block = 6270976 / 4096 * 4096;
  const int T = argc > 3 ? atoi(argv[3]) : 16;
  struct stat st; fstat(fd, &st);
  if (posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)) perror("fadvise");
  char* base = (char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  std::vector<size_t> blocks;
  for (size_t o = 0; o + block <= G; o += block) blocks.push_back(o);
  std::mt19937 g(1); std::shuffle(blocks.begin(), blocks.end(), g);
  std::atomic<size_t> next{0}; std::atomic<long> sum{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (int t = 0; t < T; t++) th.emplace_back([&] {
    long s = 0;
    for (size_t i; (i = next++) < blocks.size();) for (size_t o = 0; o < block; o += 4096) s += base[blocks[i] + o];
    sum += s;
  });
  for (auto& t : th) t.join();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("# %zu blocks of %.2f MB = %.1f GiB by page faults, %d threads, %.1f s = %.2f GB/s\n", blocks.size(), block / 1e6, G / 1073741824.0, T, s, G / 1e9 / s);
  return 0;
}
