// reg-probe: how fast would the expert tensors register in this state of the page cache?
//
//   reg-probe <first shard of the model> [--gb 4]
//
// The engine's start with --cache-wait waits for hipHostRegister of the 289 GB of expert
// tensors, and that call's rate is a property of the page cache's pages: ~1 GB/s when the files
// came in through page faults, ~4 GB/s after one sequential read (README step 199). This
// measures the rate in a few seconds: the loader's ranges (expert_loader.cpp,
// register_expert_tensors: the same tensors, the same 64 MB pieces and flags), of which every
// k-th piece is registered, --gb GiB in all, spread evenly over the ranges. Its lines, which
// scripts/model_cache.py reads:
//
//   shard <GB/s> <GB of expert tensors in it> <the file>        for every shard with expert tensors
//   rate <GB/s> expert_gb <GB of expert tensors> seconds <what the whole registration would take>
// The pieces are mapped first (MADV_POPULATE_READ), which
// reads from disk what is not in the page cache: run it on a cached model (model_cache.py does).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "model.h"

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2) { fprintf(stderr, "usage: reg-probe <first shard of the model> [--gb 4]\n"); return 2; }
  double gb = 4;
  for (int i = 2; i < argc; i++)
    if (std::string(argv[i]) == "--gb" && i + 1 < argc) gb = atof(argv[++i]);
  Model m;
  try {
    m.load(argv[1]);
  } catch (const std::exception& e) {
    fprintf(stderr, "reg-probe: %s\n", e.what());
    return 1;
  }
  // the loader's ranges: every routed expert tensor, page-aligned, merged
  std::vector<std::pair<uintptr_t, uintptr_t>> r, merged;
  for (const Layer& l : m.layers)
    if (l.loaded && l.gate_exps)
      for (const ggml_tensor* t : {l.gate_exps, l.up_exps, l.down_exps})
        r.push_back({(uintptr_t)t->data & ~(uintptr_t)4095, ((uintptr_t)t->data + ggml_nbytes(t) + 4095) & ~(uintptr_t)4095});
  std::sort(r.begin(), r.end());
  for (auto& x : r)
    if (!merged.empty() && x.first <= merged.back().second) merged.back().second = std::max(merged.back().second, x.second);
    else merged.push_back(x);
  constexpr size_t kPiece = 64ull << 20;
  std::vector<std::pair<uintptr_t, size_t>> pieces;
  size_t total = 0;
  for (auto& [a, b] : merged)
    for (uintptr_t p = a; p < b; p += kPiece) pieces.push_back({p, std::min(b, p + kPiece) - p}), total += std::min(b, p + kPiece) - p;
  if (pieces.empty()) { fprintf(stderr, "no expert tensors in %s\n", argv[1]); return 1; }
  const size_t want = std::max<size_t>(1, (size_t)(gb * 1073741824.0 / kPiece));
  const size_t step = std::max<size_t>(1, pieces.size() / want);

  if (hipSetDevice(0) != hipSuccess) { fprintf(stderr, "no GPU 0\n"); return 1; }
  int* d;
  if (hipMalloc(&d, 4) != hipSuccess) { fprintf(stderr, "hipMalloc failed\n"); return 1; }
  auto shard_of = [&](uintptr_t p) {  // the file a mapped address belongs to
    size_t off;
    char link[64], path[4096];
    snprintf(link, sizeof link, "/proc/self/fd/%d", m.file_of((const void*)p, off));
    const ssize_t k = readlink(link, path, sizeof path - 1);
    return std::string(path, k > 0 ? k : 0);
  };
  struct Sum { size_t bytes = 0, expert_bytes = 0; double s = 0; };
  std::map<std::string, Sum> by_shard;
  for (auto& [p, n] : pieces) by_shard[shard_of(p)].expert_bytes += n;
  Sum all;
  bool first = true;
  for (size_t i = step / 2; i < pieces.size(); i += step) {
    auto [p, n] = pieces[i];
    madvise((void*)p, n, MADV_POPULATE_READ);
    if (first) {  // the first call of a process pays for the runtime's setup: not counted
      const auto& w = pieces[i + 1 < pieces.size() ? i + 1 : 0];
      madvise((void*)w.first, w.second, MADV_POPULATE_READ);
      if (hipHostRegister((void*)w.first, w.second, hipHostRegisterReadOnly | hipHostRegisterPortable) == hipSuccess) (void)hipHostUnregister((void*)w.first);
      first = false;
    }
    auto t0 = Clock::now();
    if (hipHostRegister((void*)p, n, hipHostRegisterReadOnly | hipHostRegisterPortable) != hipSuccess) {
      fprintf(stderr, "hipHostRegister failed: %s\n", hipGetErrorString(hipGetLastError()));
      return 1;
    }
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    (void)hipHostUnregister((void*)p);
    Sum& sh = by_shard[shard_of(p)];
    sh.bytes += n, sh.s += s;
    all.bytes += n, all.s += s;
  }
  // a shard without a sample (fewer pieces than the step) counts at the overall rate
  double seconds = 0;
  for (auto& [path, v] : by_shard) {
    const double rate = v.bytes ? v.bytes / 1e9 / v.s : all.bytes / 1e9 / all.s;
    seconds += v.expert_bytes / 1e9 / rate;
    printf("shard %.2f %.1f %s\n", rate, v.expert_bytes / 1e9, path.c_str());
  }
  printf("rate %.2f expert_gb %.1f seconds %.0f\n", total / 1e9 / seconds, total / 1e9, seconds);
  return 0;
}
