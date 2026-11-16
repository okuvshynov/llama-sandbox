// bench-expert-load: the adaptive cache's load path, measured alone (PLAN.md "Adaptive expert
// cache", phase 2; README step 30). A load copies one expert (gate, up, down: 3 x 6.27 MB of
// ggml mxfp4 blocks) from the model's mmap into a slot of a die's expert pool and unpacks it
// (expert_loader.h).
//
//   bench-expert-load -m <model.gguf> [--secs 3]
//
// 1. Throughput: loads/s with 1 die or all 4, 1 or 2 loader threads per die, pinned or direct
//    path, plus the loader threads' CPU time per load.
// 2. CPU interference: the CPU expert kernel (40 layers x 6 random experts, 16 threads, as a
//    token's misses) while the 4 dies load at a paced rate.
// 3. GPU interference: per die, a decode-like step (per layer: copy x in, fp4_moe_chunk of 6
//    tokens on resident experts, copy y out, wait: 40 layers) while the same die loads its share
//    of a 4-die rate on a low-priority stream.
// (Copying in 1 MB pieces instead of whole matrices was tried: no better in 3.)

#include "cpu_moe.h"
#include "expert_loader.h"
#include "expert_pool.h"
#include "model.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
static double thread_cpu_s() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

constexpr int kDies = 4;
constexpr int kTargetSlots = 32;  // load targets per die (after the resident ones)
constexpr int kResident = 64;     // experts resident per die for the decode-like step

struct Bench {
  Model m;
  std::vector<int> layers;  // layers with experts
  size_t mat_bytes;
};

// one thread per pool entry: random experts into the pool's target slots until `stop`, at `rate`
// loads/s each (0 = as fast as possible); loads per thread and their CPU seconds
struct LoadRun {
  std::vector<long> loads;
  std::vector<double> cpu_s;
  long total() const { long n = 0; for (long x : loads) n += x; return n; }
};
static LoadRun run_loaders(Bench& B, std::vector<ExpertPool*>& pools, int priority, bool direct, double rate,
                           std::atomic<bool>& stop, unsigned seed, std::atomic<long>* live = nullptr) {
  LoadRun lr;
  lr.loads.assign(pools.size(), 0);
  lr.cpu_s.assign(pools.size(), 0);
  std::vector<std::thread> th;
  for (size_t i = 0; i < pools.size(); i++)
    th.emplace_back([&, i] {
      ExpertPool& p = *pools[i];
      LoadStage st = load_stage_create(p.dev, B.mat_bytes, priority);
      std::mt19937 rng(seed + (unsigned)i);
      const double c0 = thread_cpu_s();
      auto t0 = Clock::now();
      long n = 0;
      while (!stop.load()) {
        if (rate > 0) {
          const auto due = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(n / rate));
          if (Clock::now() < due) { std::this_thread::sleep_until(std::min(due, Clock::now() + std::chrono::milliseconds(20))); continue; }
        }
        const int L = B.layers[rng() % B.layers.size()], e = (int)(rng() % B.m.hp.n_expert);
        load_expert(st, p, kResident + (int)(rng() % kTargetSlots), B.m.layers[L], e, direct);
        n++;
        if (live) (*live)++;
      }
      CHECK(hipStreamSynchronize(st.s));
      lr.cpu_s[i] = thread_cpu_s() - c0;
      lr.loads[i] = n;
      load_stage_free(st);
    });
  for (auto& t : th) t.join();
  return lr;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string path;
  double secs = 3.0;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m") path = argv[++i];
    else if (a == "--secs") secs = atof(argv[++i]);
  }
  Bench B;
  B.m.load(path);
  const Hparams& hp = B.m.hp;
  const int E = hp.n_embd, F = hp.n_ff_exp, K = hp.n_expert_used;
  for (int L = 0; L < hp.n_layer; L++) if (B.m.layers[L].loaded && B.m.layers[L].gate_exps) B.layers.push_back(L);
  B.mat_bytes = (size_t)F * E / 32 * 17;
  printf("# expert: 3 x %.2f MB; %zu layers with experts\n", B.mat_bytes / 1e6, B.layers.size());

  // pools: kResident experts (one layer's first ones, for the decode-like step) + load targets
  std::vector<ExpertPool> pools(kDies);
  std::vector<ExpertPool*> pp;
  for (int d = 0; d < kDies; d++) {
    pools[d] = pool_create(d, kResident + kTargetSlots, E, F);
    CHECK(hipSetDevice(d));
    LoadStage st = load_stage_create(d, B.mat_bytes, 0);
    for (int k = 0; k < kResident; k++) load_expert(st, pools[d], k, B.m.layers[B.layers[10]], k, false);
    load_stage_free(st);
    pp.push_back(&pools[d]);
  }
  auto t0 = Clock::now();
  const size_t reg = register_expert_tensors(B.m);
  const bool direct_ok = reg > 0;
  if (direct_ok) printf("# direct: registered %.1f GB of expert tensors in %.1f s\n", reg / 1e9, secs_since(t0));
  else printf("# direct: hipHostRegister failed; direct path skipped\n");

  // 1. throughput
  printf("\n## 1. throughput, loading as fast as possible (%.0f s each)\n", secs);
  printf("path    dies  threads/die   loads/s  GB/s   loader CPU ms/load\n");
  for (bool direct : {false, true}) {
    if (direct && !direct_ok) continue;
    for (int nd : {1, 4})
      for (int tpd : {1, 2}) {
        std::vector<ExpertPool*> lp;
        for (int d = 0; d < nd; d++)
          for (int t = 0; t < tpd; t++) lp.push_back(&pools[d]);
        std::atomic<bool> stop{false};
        auto t1 = Clock::now();
        std::thread timer([&] { std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop = true; });
        LoadRun lr = run_loaders(B, lp, 0, direct, 0, stop, 1);
        timer.join();
        const double sec = secs_since(t1);
        double cpu = 0;
        for (double c : lr.cpu_s) cpu += c;
        const long n = lr.total();
        printf("%-7s %4d %12d %9.0f %5.1f %10.2f\n", direct ? "direct" : "pinned", nd, tpd, n / sec,
               n * 3 * B.mat_bytes / 1e9 / sec, cpu / n * 1e3);
      }
  }

  // 2. CPU interference
  printf("\n## 2. CPU expert kernel (40 layers x 6 random experts, 16 threads) with loads on all 4 dies\n");
  printf("path    target loads/s  achieved   CPU kernel ms/token  (vs none)\n");
  {
    CpuMoeKernel cpu(E, F, K, 16);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd;
    std::vector<float> x(E), y(E), w(K, 0.2f);
    for (auto& v : x) v = nd(rng);
    std::vector<int> ids(K);
    auto token = [&] {
      for (int L : B.layers) {
        for (auto& v : ids) v = (int)(rng() % hp.n_expert);
        const Layer& ly = B.m.layers[L];
        cpu.forward(CpuExperts{ly.gate_exps, ly.up_exps, ly.down_exps, hp.swiglu_clamp[L]}, x.data(), ids.data(), w.data(), K,
                    y.data());
      }
    };
    for (int r = 0; r < 5; r++) token();
    double base = 0;
    for (bool direct : {false, true}) {
      if (direct && !direct_ok) continue;
      for (double rate : {0.0, 50.0, 100.0, 200.0, 400.0, -1.0}) {
        if (direct && rate == 0) continue;
        std::atomic<bool> stop{rate == 0};
        std::atomic<long> live{0};
        std::thread lt([&] { if (rate != 0) run_loaders(B, pp, 0, direct, rate > 0 ? rate / kDies : 0, stop, 5, &live); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const long l0 = live;
        auto t1 = Clock::now();
        int n = 0;
        while (secs_since(t1) < secs) { token(); n++; }
        const double sec = secs_since(t1), ms = sec / n * 1e3;
        const long loads = live - l0;
        stop = true;
        lt.join();
        if (rate == 0) base = ms;
        printf("%-7s %14s %9.0f %20.1f  %+5.1f%%\n", direct ? "direct" : "pinned",
               rate < 0 ? "max" : std::to_string((int)rate).c_str(), loads / sec, ms, (ms / base - 1) * 100);
      }
    }
  }

  // 3. GPU interference, one die at a time
  printf("\n## 3. decode-like step on one die (40 x [copy in, fp4_moe_chunk 6 tokens x 6 experts, copy out, wait])\n");
  printf("##    with this die's share of a 4-die load rate, low-priority stream\n");
  printf("die  path    target loads/s  achieved (x4)  step ms  (vs none)\n");
  for (int d = 0; d < kDies; d++) {
    CHECK(hipSetDevice(d));
    hipStream_t s;
    CHECK(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, 0));
    const int M = 6, stride = 1 + 2 * K;
    std::vector<int> sel((size_t)M * stride);
    std::mt19937 rng(11);
    for (int t = 0; t < M; t++) {
      sel[(size_t)t * stride] = K;
      for (int j = 0; j < K; j++) {
        sel[(size_t)t * stride + 1 + j] = (int)(rng() % kResident);
        const float wv = 0.2f;
        memcpy(&sel[(size_t)t * stride + 1 + K + j], &wv, 4);
      }
    }
    int* dsel;
    float *dx, *dy, *hx;
    void* scr;
    CHECK(hipMalloc(&dsel, sel.size() * 4));
    CHECK(hipMalloc(&dx, (size_t)M * E * 4));
    CHECK(hipMalloc(&dy, (size_t)M * E * 4));
    CHECK(hipHostMalloc(&hx, (size_t)M * E * 4, 0));
    CHECK(hipMalloc(&scr, fp4_moe_scratch_bytes(E, F, K, M)));
    CHECK(hipMemcpy(dsel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice));
    for (int i = 0; i < M * E; i++) hx[i] = 0.01f * (i % 97);
    auto step = [&] {
      for (int L = 0; L < 40; L++) {
        CHECK(hipMemcpyAsync(dx, hx, (size_t)M * E * 4, hipMemcpyHostToDevice, s));
        fp4_moe_chunk(pools[d].view, dx, ExpertSel{dsel + 1, (const float*)(dsel + 1 + K), dsel, stride}, K, M, 10.0f, dy, scr, s);
        CHECK(hipMemcpyAsync(hx, dy, (size_t)M * E * 4, hipMemcpyDeviceToHost, s));
        CHECK(hipStreamSynchronize(s));
      }
    };
    for (int r = 0; r < 3; r++) step();
    int lo, hi;
    CHECK(hipDeviceGetStreamPriorityRange(&lo, &hi));  // lo = least priority
    double base = 0;
    std::vector<std::pair<bool, double>> cfgs = {{false, 0}};
    for (double r : {50.0, 100.0, 200.0, -1.0}) {
      cfgs.push_back({false, r});
      if (direct_ok) cfgs.push_back({true, r});
    }
    for (auto [direct, rate] : cfgs) {
      std::vector<ExpertPool*> lp = {&pools[d]};
      std::atomic<bool> stop{rate == 0};
      std::atomic<long> live{0};
      std::thread lt([&, direct = direct, rate = rate] {
        if (rate != 0) run_loaders(B, lp, lo, direct, rate > 0 ? rate / kDies : 0, stop, 9, &live);
      });
      CHECK(hipSetDevice(d));
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      const long l0 = live;
      auto t1 = Clock::now();
      int n = 0;
      while (secs_since(t1) < secs / 2) { step(); n++; }
      const double sec = secs_since(t1), ms = sec / n * 1e3;
      const long loads = live - l0;
      stop = true;
      lt.join();
      if (rate == 0) base = ms;
      printf("%3d  %-7s %14s %13.0f %8.1f  %+5.1f%%\n", d, direct ? "direct" : "pinned",
             rate < 0 ? "max" : std::to_string((int)rate).c_str(), loads / sec * kDies, ms, (ms / base - 1) * 100);
    }
    for (void* p : {(void*)dsel, (void*)dx, (void*)dy, scr}) CHECK(hipFree(p));
    CHECK(hipHostFree(hx));
    CHECK(hipStreamDestroy(s));
  }
  return 0;
}
