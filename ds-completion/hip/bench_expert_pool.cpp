// bench-expert-pool: expert pools on the GPUs from a placement (expert_pool.h).
//
//   bench-expert-pool -m <model.gguf> -p placements/study30.txt [--dies 0,1,2,3] [--reserve0-gb 9.5]
//                     [--reserve-gb 0.75] [--max-slots N]
//
// 1. Fills each die's free memory (die 0 keeps room for the runtime's dense part) with the
//    placement's top pairs, one loader thread per die; reports slots, GB and load time.
// 2. Exactness: on every die, for 0..6 experts of one layer resident there, fp4_moe on the pool
//    (slot ids, device-side count) against the CPU kernel on the same experts.
// 3. Time of fp4_moe on a pool by expert count (fixed cost vs per expert).

#include "cpu_moe.h"
#include "expert_pool.h"
#include "model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string path, placement;
  std::vector<int> dies = {0, 1, 2, 3};
  double reserve0_gb = 9.5, reserve_gb = 0.75;
  int max_slots = -1;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m") path = argv[++i];
    else if (a == "-p") placement = argv[++i];
    else if (a == "--dies") {
      dies.clear();
      for (char* q = argv[++i]; *q;) { dies.push_back((int)strtol(q, &q, 10)); if (*q == ',') q++; }
    } else if (a == "--reserve0-gb") reserve0_gb = atof(argv[++i]);
    else if (a == "--reserve-gb") reserve_gb = atof(argv[++i]);
    else if (a == "--max-slots") max_slots = atoi(argv[++i]);
  }
  Model m;
  m.load(path);
  const Hparams& hp = m.hp;
  const int E = hp.n_embd, F = hp.n_ff_exp, K = hp.n_expert_used, X = hp.n_expert;

  // 1. capacity, placement, pools
  std::vector<int> cap;
  for (int d : dies) {
    int c = pool_slots_that_fit(d, E, F, (size_t)((d == 0 ? reserve0_gb : reserve_gb) * 1e9));
    if (max_slots >= 0) c = std::min(c, max_slots);
    cap.push_back(c);
  }
  std::vector<bool> loaded(hp.n_layer);
  for (int L = 0; L < hp.n_layer; L++) loaded[L] = m.layers[L].loaded;
  const auto ranked = read_placement(placement);
  const Placement pl = place_experts(ranked, hp.n_layer, X, cap, loaded);
  std::vector<ExpertPool> pools(dies.size());
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (size_t i = 0; i < dies.size(); i++)
    th.emplace_back([&, i] {
      const int d = dies[i];
      pools[i] = pool_create(d, pl.used[i], E, F);
      CHECK(hipSetDevice(d));
      hipStream_t s;
      CHECK(hipStreamCreate(&s));
      for (int L = 0; L < hp.n_layer; L++)
        for (int e = 0; e < X; e++) {
          if (pl.at(L, e) != (int)i) continue;
          const Layer& l = m.layers[L];
          auto ex = [&](const ggml_tensor* t) { return (const char*)t->data + (size_t)e * t->nb[2]; };
          pool_load(pools[i], pl.slot[(size_t)L * X + e], ex(l.gate_exps), ex(l.up_exps), ex(l.down_exps), s);
        }
      CHECK(hipStreamSynchronize(s));
      CHECK(hipStreamDestroy(s));
    });
  for (auto& t : th) t.join();
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  size_t total = 0;
  for (size_t i = 0; i < dies.size(); i++) {
    total += (size_t)pl.used[i];
    printf("die %d: %d slots (capacity %d), %.1f GB\n", dies[i], pl.used[i], cap[i], pl.used[i] * pools[i].slot_bytes / 1e9);
  }
  printf("loaded %zu experts (%.1f GB, %.1f%% of all pairs) in %.1f s = %.1f GB/s\n", total,
         total * pools[0].slot_bytes / 1e9, 100.0 * total / (hp.n_layer * X), sec, total * pools[0].slot_bytes / 1e9 / sec);

  // 2. exactness against the CPU kernel, 0..6 experts of one layer, on every die
  CpuMoeKernel cpu(E, F, K, 32);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  for (size_t i = 0; i < dies.size(); i++) {
    if (pl.used[i] == 0) continue;
    CHECK(hipSetDevice(dies[i]));
    hipStream_t s;
    CHECK(hipStreamCreate(&s));
    // a layer with at least 6 experts on this die
    int layer = -1;
    std::vector<int> experts;
    for (int L = 0; L < hp.n_layer && layer < 0; L++) {
      experts.clear();
      for (int e = 0; e < X && (int)experts.size() < K; e++) if (pl.at(L, e) == (int)i) experts.push_back(e);
      if ((int)experts.size() == K) layer = L;
    }
    if (layer < 0) { printf("die %d: no layer with %d experts here\n", dies[i], K); continue; }
    void* scratch;
    float *dx, *dy, *dw;
    int *dids, *dcount;
    CHECK(hipMalloc(&scratch, fp4_moe_scratch_bytes(E, F, K)));
    CHECK(hipMalloc(&dx, E * 4));
    CHECK(hipMalloc(&dy, E * 4));
    CHECK(hipMalloc(&dw, K * 4));
    CHECK(hipMalloc(&dids, K * 4));
    CHECK(hipMalloc(&dcount, 4));
    const Layer& l = m.layers[layer];
    printf("die %d, layer %d:", dies[i], layer);
    for (int n = 0; n <= K; n++) {
      std::vector<float> x(E), yc(E, 0.0f), yg(E), w(K, 0.0f);
      std::vector<int> ids(K, 0), slots(K, 0);
      for (auto& v : x) v = nd(rng);
      for (int j = 0; j < n; j++) {
        ids[j] = experts[j];
        slots[j] = pl.slot[(size_t)layer * X + experts[j]];
        w[j] = 0.1f + (rng() % 1000) / 1000.0f;
      }
      if (n > 0)
        cpu.forward(CpuExperts{l.gate_exps, l.up_exps, l.down_exps, hp.swiglu_clamp[layer]}, x.data(), ids.data(), w.data(),
                    n, yc.data());
      CHECK(hipMemcpy(dx, x.data(), E * 4, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dids, slots.data(), K * 4, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dw, w.data(), K * 4, hipMemcpyHostToDevice));
      CHECK(hipMemcpy(dcount, &n, 4, hipMemcpyHostToDevice));
      fp4_moe(pools[i].view, dx, dids, dw, K, hp.swiglu_clamp[layer], dy, scratch, s, dcount);
      CHECK(hipMemcpyAsync(yg.data(), dy, E * 4, hipMemcpyDeviceToHost, s));
      CHECK(hipStreamSynchronize(s));
      double num = 0, den = 0, mx = 0;
      for (int k = 0; k < E; k++) {
        num += (double)(yg[k] - yc[k]) * (yg[k] - yc[k]);
        den += (double)yc[k] * yc[k];
        mx = std::max(mx, (double)fabsf(yg[k]));
      }
      if (n == 0) printf("  n=0 max|y| %.1e", mx);
      else printf("  n=%d %.1e", n, sqrt(num / den));
    }
    printf("\n");

    // 3. time by count: random resident experts of this die (any layer) each rep
    std::vector<int> die_slots;
    for (int k = 0; k < pl.used[i]; k++) die_slots.push_back(k);
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    const int reps = 200;
    int* dall;
    CHECK(hipMalloc(&dall, (size_t)reps * K * 4));
    std::vector<int> all((size_t)reps * K);
    for (auto& v : all) v = die_slots[rng() % die_slots.size()];
    CHECK(hipMemcpy(dall, all.data(), all.size() * 4, hipMemcpyHostToDevice));
    printf("die %d time:", dies[i]);
    for (int n : {1, 2, 3, 6}) {
      CHECK(hipMemcpy(dcount, &n, 4, hipMemcpyHostToDevice));
      for (int r = 0; r < 5; r++) fp4_moe(pools[i].view, dx, dall + r * K, dw, K, 10.0f, dy, scratch, s, dcount);
      CHECK(hipEventRecord(e0, s));
      for (int r = 0; r < reps; r++) fp4_moe(pools[i].view, dx, dall + r * K, dw, K, 10.0f, dy, scratch, s, dcount);
      CHECK(hipEventRecord(e1, s));
      CHECK(hipEventSynchronize(e1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      printf("  %d experts %.0f us", n, ms * 1000 / reps);
    }
    printf("\n");
    CHECK(hipStreamDestroy(s));
  }
  for (auto& p : pools) pool_free(p);
  return 0;
}
