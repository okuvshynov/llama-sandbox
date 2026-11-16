// bench-fp4-moe: one layer's routed experts resident on a GPU (fp4_moe.h) vs the CPU kernel
// (cpu_moe.h): same inputs, both exact, so they must agree to fp32 summation order; then the
// GPU time per layer with fresh random experts every repetition.
//
//   bench-fp4-moe -m <model.gguf> [--layer 20] [--dev 0] [--reps 200]

#include "cpu_moe.h"
#include "fp4_moe.h"
#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string path;
  int layer = 20, dev = 0, reps = 200;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m") path = argv[++i];
    else if (a == "--layer") layer = atoi(argv[++i]);
    else if (a == "--dev") dev = atoi(argv[++i]);
    else if (a == "--reps") reps = atoi(argv[++i]);
  }
  Model m;
  m.load(path, {layer});
  const Hparams& hp = m.hp;
  const Layer& l = m.layers[layer];
  const int E = hp.n_embd, F = hp.n_ff_exp, K = hp.n_expert_used, X = hp.n_expert;
  CHECK(hipSetDevice(dev));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  printf("# uploading layer %d's %d experts (%.2f GiB) to GPU %d\n", layer, X,
         3.0 * ggml_nbytes(l.gate_exps) / (1 << 30), dev);
  GpuExperts gx = {fp4_upload(l.gate_exps->data, E, F, X, s), fp4_upload(l.up_exps->data, E, F, X, s),
                   fp4_upload(l.down_exps->data, F, E, X, s)};

  std::mt19937 rng(5);
  std::normal_distribution<float> nd;
  auto pick = [&](int* ids, float* w) {
    std::set<int> st;
    while ((int)st.size() < K) st.insert(rng() % X);
    int j = 0;
    float tot = 0;
    for (int e : st) { ids[j] = e; tot += (w[j] = 0.1f + (rng() % 1000) / 1000.0f); j++; }
    for (j = 0; j < K; j++) w[j] = w[j] / tot * hp.expert_weights_scale;
  };
  void* scratch;
  CHECK(hipMalloc(&scratch, fp4_moe_scratch_bytes(E, F, K)));
  void* cscratch;
  CHECK(hipMalloc(&cscratch, fp4_moe_canon_scratch_bytes(E, F, K)));
  float *dx, *dy, *dw;
  int* dids;
  CHECK(hipMalloc(&dx, E * 4));
  CHECK(hipMalloc(&dy, E * 4));
  CHECK(hipMalloc(&dw, (size_t)reps * K * 4));
  CHECK(hipMalloc(&dids, (size_t)reps * K * 4));

  // accuracy: 3 random cases against the CPU kernel
  CpuMoeKernel cpu(E, F, K, 32);
  for (int c = 0; c < 3; c++) {
    std::vector<float> x(E), yc(E), yg(E), w(K);
    std::vector<int> ids(K);
    for (auto& v : x) v = nd(rng);
    pick(ids.data(), w.data());
    cpu.forward(CpuExperts{l.gate_exps, l.up_exps, l.down_exps, hp.swiglu_clamp[layer]}, x.data(), ids.data(), w.data(), K,
                yc.data());
    CHECK(hipMemcpy(dx, x.data(), E * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dids, ids.data(), K * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw, w.data(), K * 4, hipMemcpyHostToDevice));
    fp4_moe(gx, dx, dids, dw, K, hp.swiglu_clamp[layer], dy, scratch, s);
    CHECK(hipMemcpyAsync(yg.data(), dy, E * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    double num = 0, den = 0;
    for (int i = 0; i < E; i++) { num += (double)(yg[i] - yc[i]) * (yg[i] - yc[i]); den += (double)yc[i] * yc[i]; }
    printf("case %d: GPU vs CPU kernel relerr %.2e\n", c, sqrt(num / den));
    // placement-invariant numerics: the GPU must give the CPU's bits exactly
    std::vector<float> ycc(E), ygc(E);
    cpu.forward_canon(CpuExperts{l.gate_exps, l.up_exps, l.down_exps, hp.swiglu_clamp[layer]}, x.data(), ids.data(),
                      w.data(), K, ycc.data());
    fp4_moe_canon(gx, dx, dids, dw, K, hp.swiglu_clamp[layer], dy, cscratch, s);
    CHECK(hipMemcpyAsync(ygc.data(), dy, E * 4, hipMemcpyDeviceToHost, s));
    CHECK(hipStreamSynchronize(s));
    int nd_ = 0;
    for (int i = 0; i < E; i++) nd_ += memcmp(&ycc[i], &ygc[i], 4) != 0;
    double n2 = 0;
    for (int i = 0; i < E; i++) n2 += (double)(ycc[i] - yc[i]) * (ycc[i] - yc[i]);
    printf("        canonical: GPU vs CPU %d of %d outputs differ; canonical vs current CPU relerr %.2e\n", nd_, E,
           sqrt(n2 / den));
  }

  // speed: a fresh expert set per repetition (113 MB of weights each, no cache reuse)
  std::vector<int> all_ids((size_t)reps * K);
  std::vector<float> all_w((size_t)reps * K);
  for (int r = 0; r < reps; r++) pick(&all_ids[(size_t)r * K], &all_w[(size_t)r * K]);
  CHECK(hipMemcpy(dids, all_ids.data(), all_ids.size() * 4, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(dw, all_w.data(), all_w.size() * 4, hipMemcpyHostToDevice));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (int r = 0; r < 5; r++) fp4_moe(gx, dx, dids + r * K, dw + r * K, K, hp.swiglu_clamp[layer], dy, scratch, s);
  CHECK(hipEventRecord(e0, s));
  for (int r = 0; r < reps; r++) fp4_moe(gx, dx, dids + r * K, dw + r * K, K, hp.swiglu_clamp[layer], dy, scratch, s);
  CHECK(hipEventRecord(e1, s));
  CHECK(hipEventSynchronize(e1));
  float ms;
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  const double us = ms * 1000 / reps, bytes = 3.0 * ggml_nbytes(l.gate_exps) / X * K;
  printf("GPU %d: %.1f us per layer (6 experts, %.1f MB) = %.0f GB/s\n", dev, us, bytes / 1e6, bytes / (us * 1e-6) / 1e9);
  CHECK(hipEventRecord(e0, s));
  for (int r = 0; r < reps; r++) fp4_moe_canon(gx, dx, dids + r * K, dw + r * K, K, hp.swiglu_clamp[layer], dy, cscratch, s);
  CHECK(hipEventRecord(e1, s));
  CHECK(hipEventSynchronize(e1));
  CHECK(hipEventElapsedTime(&ms, e0, e1));
  const double usc = ms * 1000 / reps;
  printf("GPU %d canonical: %.1f us per layer = %.0f GB/s (%+.0f%%)\n", dev, usc, bytes / (usc * 1e-6) / 1e9, (usc / us - 1) * 100);
  printf("for scale: the CPU kernel takes ~1650 us per layer (66 ms / 40)\n");
  fp4_free(gx.gate);
  fp4_free(gx.up);
  fp4_free(gx.down);
  return 0;
}
