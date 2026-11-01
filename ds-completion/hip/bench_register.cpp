// bench-register: does registering host memory with HIP (hipHostRegister, the expert loader's
// direct path) stall other GPU work, and how does that depend on the piece size? (PLAN item 30)
//
//   bench-register <file> [--gb 8] [--pieces 0,4096,1024,256,64] [--dev 0]
//
// The file (a model shard) is mmap'd read-only and its pages read once (in the page cache, as in
// the runtime). Per piece size (MB; 0 = one call for the whole range), on a fresh --gb region of
// the file (pages pinned once are cheaper to pin again): a probe thread does what a runtime step
// does, back to back (a tiny kernel, a 20 KB copy to the device and one back through pinned
// memory, an event synchronize) and records each round trip, while the main thread registers the
// region in pieces of that size. Reported: the registration time, the probe's median / max round
// trip and how long it was blocked (round trips over 10 ms). Then copies of 18.8 MB (one expert):
// within a piece, across a piece boundary as two copies (one copy across two registrations is
// refused), and from unregistered (pageable) memory, in GB/s.
// --faults FILE2: also a CPU thread that reads FILE2 (another shard, in the page cache but not
// mapped yet in this process) page by page during the registration, as the CPU expert kernel
// first touches the model: its page-fault throughput and longest wait (mmap lock contention?).
// Always: a thread that maps and unmaps 1 MB of anonymous memory in a loop (what malloc does
// for large blocks: a writer of the process's memory-map lock) and its longest call.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

__global__ void tiny(int* p) { if (threadIdx.x == 0) p[0]++; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2) { fprintf(stderr, "usage: bench-register <file> [--gb 16] [--pieces 0,4096,1024,256,64] [--dev 0]\n"); return 2; }
  double gb = 8;
  int dev = 0;
  const char* fault_path = nullptr;
  std::vector<long> pieces = {0, 4096, 1024, 256, 64};
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--gb") gb = atof(argv[++i]);
    else if (a == "--dev") dev = atoi(argv[++i]);
    else if (a == "--faults") fault_path = argv[++i];
    else if (a == "--pieces") { pieces.clear(); for (char* p = argv[++i]; *p;) { pieces.push_back(strtol(p, &p, 10)); if (*p == ',') p++; } }
  }
  const int fd = open(argv[1], O_RDONLY);
  if (fd < 0) { perror(argv[1]); return 1; }
  struct stat sb;
  fstat(fd, &sb);
  const size_t len = (size_t)(gb * 1e9) & ~(size_t)4095, total = (size_t)sb.st_size & ~(size_t)4095;
  if (len * pieces.size() > total) { fprintf(stderr, "the file holds %zu regions of %.1f GB, %zu asked\n", total / len, gb, pieces.size()); return 1; }
  char* file = (char*)mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
  if (file == MAP_FAILED) { perror("mmap"); return 1; }
  volatile uint64_t sink = 0;
  for (size_t i = 0; i < len * pieces.size(); i += 4096) sink += file[i];  // into the page cache
  CHECK(hipSetDevice(dev));
  int* d;
  CHECK(hipMalloc(&d, 20480));
  char* hp;
  CHECK(hipHostMalloc((void**)&hp, 20480, 0));
  hipEvent_t ev;
  CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  printf("# %s: %.1f GB registered per run (a fresh region each), probe on device %d\n", argv[1], len / 1e9, dev);
  // the fault reader's file: one fresh slice of it per run, in the page cache, never mapped before
  char* ffile = nullptr;
  size_t flen = 0, fslice = 0;
  if (fault_path) {
    const int fd2 = open(fault_path, O_RDONLY);
    struct stat sb2;
    if (fd2 < 0 || fstat(fd2, &sb2)) { perror(fault_path); return 1; }
    flen = (size_t)sb2.st_size & ~(size_t)4095;
    fslice = (flen / pieces.size()) & ~(size_t)4095;
    std::vector<char> buf(1 << 20);
    for (off_t o = 0; o < (off_t)flen; o += buf.size()) if (pread(fd2, buf.data(), buf.size(), o) <= 0) break;  // page cache only
    ffile = (char*)mmap(nullptr, flen, PROT_READ, MAP_SHARED, fd2, 0);
    if (ffile == MAP_FAILED) { perror("mmap"); return 1; }
  }
  printf("%10s %10s %12s %12s %12s %12s %s\n", "piece MB", "register s", "probe p50 ms", "probe max ms", "blocked s", "pieces",
         fault_path ? "  faults: GB/s during registration, longest page wait ms" : "");

  for (size_t run = 0; run < pieces.size(); run++) {
    const long pm = pieces[run];
    char* base = file + run * len;
    const size_t piece = pm ? (size_t)pm << 20 : len;
    std::atomic<bool> stop{false};
    std::vector<double> rt;
    std::thread probe([&] {
      CHECK(hipSetDevice(dev));
      while (!stop) {
        auto t = Clock::now();
        tiny<<<1, 64, 0, s>>>(d);
        CHECK(hipMemcpyAsync(d, hp, 20480, hipMemcpyHostToDevice, s));
        CHECK(hipMemcpyAsync(hp, d, 20480, hipMemcpyDeviceToHost, s));
        CHECK(hipEventRecord(ev, s));
        CHECK(hipEventSynchronize(ev));
        rt.push_back(ms_since(t));
      }
    });
    std::atomic<bool> fstop{false};
    double f_gbs = 0, f_max = 0;
    std::thread faulter;
    if (ffile)
      faulter = std::thread([&, run] {
        const char* p0 = ffile + run * fslice;
        size_t done = 0;
        auto t = Clock::now();
        volatile uint64_t acc = 0;
        while (!fstop && done < fslice) {
          auto tp = Clock::now();
          acc += p0[done];
          f_max = std::max(f_max, ms_since(tp));
          done += 4096;
        }
        f_gbs = done / (ms_since(t) / 1000) / 1e9;
        (void)acc;
      });
    double map_max = 0;
    long map_n = 0;
    std::thread mapper([&] {
      while (!fstop) {
        auto tm = Clock::now();
        void* q = mmap(nullptr, 1 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ((volatile char*)q)[0] = 1;
        munmap(q, 1 << 20);
        map_max = std::max(map_max, ms_since(tm));
        map_n++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto t0 = Clock::now();
    std::vector<char*> regs;
    for (size_t off = 0; off < len; off += piece) {
      const size_t n = std::min(piece, len - off);
      CHECK(hipHostRegister(base + off, n, hipHostRegisterReadOnly | hipHostRegisterPortable));
      regs.push_back(base + off);
    }
    const double reg_s = ms_since(t0) / 1000;
    fstop = true;
    if (faulter.joinable()) faulter.join();
    mapper.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    probe.join();
    std::vector<double> sorted = rt;
    std::sort(sorted.begin(), sorted.end());
    double blocked = 0;
    for (double v : rt) if (v > 10) blocked += v / 1000;
    printf("%10ld %10.1f %12.3f %12.1f %12.1f %12zu", pm, reg_s, sorted[sorted.size() / 2], sorted.back(), blocked, regs.size());
    if (ffile) printf("  %8.2f %10.1f", f_gbs, f_max);
    printf("   mmap+munmap: %ld calls, longest %.1f ms", map_n, map_max);
    printf("\n");

    // copies of one expert's size from the registered range (within a piece, across a boundary)
    const size_t E = 18800000;
    char* dst;
    CHECK(hipMalloc(&dst, E));
    auto copy_gbs = [&](const char* src, size_t split) {  // split: bytes in the first copy (0 = one copy)
      auto one = [&]() {
        if (split) {
          CHECK(hipMemcpyAsync(dst, src, split, hipMemcpyHostToDevice, s));
          CHECK(hipMemcpyAsync(dst + split, src + split, E - split, hipMemcpyHostToDevice, s));
        } else {
          CHECK(hipMemcpyAsync(dst, src, E, hipMemcpyHostToDevice, s));
        }
      };
      one();
      CHECK(hipStreamSynchronize(s));
      auto t = Clock::now();
      for (int i = 0; i < 10; i++) one();
      CHECK(hipStreamSynchronize(s));
      return 10 * E / (ms_since(t) / 1000) / 1e9;
    };
    const double within = copy_gbs(base + 4096, 0);
    const double across = piece < len ? copy_gbs(base + piece - E / 2, E / 2) : within;
    printf("%10s copies of 18.8 MB: within a piece %.1f GB/s, across a piece boundary %.1f GB/s\n", "", within, across);
    CHECK(hipFree(dst));
    for (char* r : regs) CHECK(hipHostUnregister(r));
  }
  // unregistered (pageable) for comparison
  {
    const size_t E = 18800000;
    char* dst;
    CHECK(hipMalloc(&dst, E));
    auto t = Clock::now();
    for (int i = 0; i < 10; i++) CHECK(hipMemcpyAsync(dst, file + 4096, E, hipMemcpyHostToDevice, s));
    CHECK(hipStreamSynchronize(s));
    printf("%10s copies of 18.8 MB from unregistered memory: %.1f GB/s\n", "", 10 * E / (ms_since(t) / 1000) / 1e9);
  }
  (void)sink;
  return 0;
}
