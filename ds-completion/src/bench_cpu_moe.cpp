// bench-cpu-moe: the CPU routed-expert kernel (cpu_moe.h) against ggml's mul_mat_id path (what
// the runtime used so far) and against a double-precision host computation of the reference math.
//
//   bench-cpu-moe -m <model.gguf> [-t 32] [--tokens 5] [--check-layers 0,20,39] [-k 6]
//
// -k: experts per layer in the speed part (the runtime's CPU usually gets 1-2: its misses)
//
// Accuracy: for a few layers, both against the double reference (same fp8 fake quantization of
// the inputs, exact fp4 weights). Speed: a "token" = all 40 layers in sequence, 6 random experts
// each (fresh experts every token, so the 113 MB per layer stream from DRAM as in decode).

#include "cpu_moe.h"
#include "fakequant.h"
#include "model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>

using Clock = std::chrono::steady_clock;

// the runtime's previous path: one ggml CPU graph per layer
struct GgmlMoe {
  ggml_context *ctx_in, *ctx;
  ggml_backend_buffer_t in_buf;
  ggml_gallocr_t alloc;
  ggml_cgraph* gf;
  ggml_tensor *x, *ids, *w, *y;
  GgmlMoe(ggml_backend_t cpu, const Layer& l, int E, int K, float limit) {
    ggml_init_params pi = {ggml_tensor_overhead() * 4, nullptr, true};
    ctx_in = ggml_init(pi);
    x = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, E, 1);
    ids = ggml_new_tensor_2d(ctx_in, GGML_TYPE_I32, K, 1);
    w = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, 1, K, 1);
    in_buf = ggml_backend_alloc_ctx_tensors(ctx_in, cpu);
    ggml_init_params p = {ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true};
    ctx = ggml_init(p);
    ggml_tensor* xq = ggml_reshape_3d(ctx, fq_fp8(ctx, x), E, 1, 1);
    ggml_tensor* a = ggml_mul(ctx, ggml_swiglu_clamp(ctx, ggml_mul_mat_id(ctx, l.gate_exps, xq, ids),
                                                     ggml_mul_mat_id(ctx, l.up_exps, xq, ids), limit), w);
    ggml_tensor* d = ggml_mul_mat_id(ctx, l.down_exps, fq_fp8(ctx, a), ids);
    y = nullptr;
    for (int j = 0; j < K; j++) {
      ggml_tensor* dj = ggml_view_2d(ctx, d, E, 1, d->nb[2], (size_t)j * d->nb[1]);
      y = y ? ggml_add(ctx, y, dj) : dj;
    }
    ggml_set_output(y);
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu));
    ggml_gallocr_alloc_graph(alloc, gf);
  }
  void run(ggml_backend_t cpu, const float* xh, const int* idh, const float* wh, float* yh) {
    memcpy(x->data, xh, ggml_nbytes(x));
    memcpy(ids->data, idh, ggml_nbytes(ids));
    memcpy(w->data, wh, ggml_nbytes(w));
    ggml_backend_graph_compute(cpu, gf);
    memcpy(yh, y->data, ggml_nbytes(y));
  }
};

// the reference math in double (activations fake-quantized to fp8 exactly as the kernels do)
static std::vector<float> reference(const Layer& l, const float* x, const int* ids, const float* w, int K, int E, int F,
                                    float limit) {
  auto to_float = ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float;
  std::vector<float> xq(E), row(std::max(E, F)), y(E, 0.0f);
  fq_fp8_host(x, xq.data(), E);
  std::vector<double> acc(E, 0.0);
  for (int j = 0; j < K; j++) {
    std::vector<float> a(F);
    for (int r = 0; r < F; r++) {
      double g = 0, u = 0;
      to_float((const char*)l.gate_exps->data + (size_t)ids[j] * l.gate_exps->nb[2] + r * l.gate_exps->nb[1], row.data(), E);
      for (int i = 0; i < E; i++) g += (double)row[i] * xq[i];
      to_float((const char*)l.up_exps->data + (size_t)ids[j] * l.up_exps->nb[2] + r * l.up_exps->nb[1], row.data(), E);
      for (int i = 0; i < E; i++) u += (double)row[i] * xq[i];
      const float gf = std::min((float)g, limit), uf = std::clamp((float)u, -limit, limit);
      a[r] = w[j] * (gf / (1.0f + expf(-gf)) * uf);
    }
    fq_fp8_host(a.data(), a.data(), F);
    for (int r = 0; r < E; r++) {
      to_float((const char*)l.down_exps->data + (size_t)ids[j] * l.down_exps->nb[2] + r * l.down_exps->nb[1], row.data(), F);
      double d = 0;
      for (int i = 0; i < F; i++) d += (double)row[i] * a[i];
      acc[r] += d;
    }
  }
  for (int r = 0; r < E; r++) y[r] = (float)acc[r];
  return y;
}

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
  double n = 0, d = 0;
  for (size_t i = 0; i < a.size(); i++) { n += (double)(a[i] - b[i]) * (a[i] - b[i]); d += (double)b[i] * b[i]; }
  return sqrt(n / d);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string path;
  int threads = 32, tokens = 5, kspeed = 0;
  std::vector<int> check = {0, 20, 39};
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m") path = argv[++i];
    else if (a == "-t") threads = atoi(argv[++i]);
    else if (a == "--tokens") tokens = atoi(argv[++i]);
    else if (a == "-k") kspeed = atoi(argv[++i]);
    else if (a == "--check-layers") { check.clear(); for (char* p = argv[++i]; *p;) { check.push_back((int)strtol(p, &p, 10)); if (*p == ',') p++; } }
  }
  Model m;
  m.load(path);
  const Hparams& hp = m.hp;
  const int E = hp.n_embd, F = hp.n_ff_exp, K = hp.n_expert_used;
  ggml_backend_t cpu = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(cpu, threads);
  CpuMoeKernel kern(E, F, K, threads);
  std::vector<std::unique_ptr<GgmlMoe>> gg;
  for (int L = 0; L < hp.n_layer; L++) gg.emplace_back(new GgmlMoe(cpu, m.layers[L], E, K, hp.swiglu_clamp[L]));

  std::mt19937 rng(11);
  std::normal_distribution<float> nd;
  auto pick = [&](std::vector<int>& ids, std::vector<float>& w, std::vector<float>& x) {
    std::set<int> s;
    while ((int)s.size() < K) s.insert(rng() % hp.n_expert);
    ids.assign(s.begin(), s.end());
    std::shuffle(ids.begin(), ids.end(), rng);
    w.resize(K);
    float tot = 0;
    for (auto& v : w) tot += (v = 0.1f + (rng() % 1000) / 1000.0f);
    for (auto& v : w) v = v / tot * hp.expert_weights_scale;
    x.resize(E);
    for (auto& v : x) v = nd(rng);
  };

  printf("# accuracy vs the double-precision reference math\n");
  for (int L : check) {
    std::vector<int> ids;
    std::vector<float> w, x, y1(E), y2(E);
    pick(ids, w, x);
    const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
    std::vector<float> y3(E);
    kern.forward(ex, x.data(), ids.data(), w.data(), K, y1.data());
    gg[L]->run(cpu, x.data(), ids.data(), w.data(), y2.data());
    kern.forward_canon(ex, x.data(), ids.data(), w.data(), K, y3.data());
    const auto r = reference(m.layers[L], x.data(), ids.data(), w.data(), K, E, F, hp.swiglu_clamp[L]);
    printf("layer %2d: ours relerr %.2e   canonical %.2e   ggml (q8_0 activations) relerr %.2e\n", L, rel(y1, r),
           rel(y3, r), rel(y2, r));
    // the vector canonical dot against its scalar specification, every row of one expert's matrices
    for (int mi = 0; mi < 3; mi++) {
      const ggml_tensor* W = mi == 0 ? m.layers[L].gate_exps : mi == 1 ? m.layers[L].up_exps : m.layers[L].down_exps;
      const int K_in = (int)W->ne[0], N = (int)W->ne[1];
      std::vector<float> xin(K_in), yv(N);
      for (auto& v : xin) v = nd(rng) * (mi == 2 ? 0.05f : 1.0f);
      CanonAct ca;
      canon_act(xin.data(), K_in, ca);
      const uint8_t* base = (const uint8_t*)W->data + (size_t)ids[0] * W->nb[2];
      canon_dot_rows(base, W->nb[1], ca, 0, N, yv.data());
      long diff = 0;
      for (int n = 0; n < N; n++) {
        const float ref = canon_dot_ref(base + (size_t)n * W->nb[1], ca);
        diff += memcmp(&ref, &yv[n], 4) != 0;
      }
      printf("  canonical dot, %s rows of expert %d: %ld of %d differ from the scalar specification\n",
             mi == 0 ? "gate" : mi == 1 ? "up" : "down", ids[0], diff, N);
    }
  }

  const int KS = kspeed > 0 && kspeed <= K ? kspeed : K;
  printf("# speed: one token = %d layers x %d experts (%.1f MB of weights)\n", hp.n_layer, KS,
         hp.n_layer * KS * 3.0 * ggml_nbytes(m.layers[0].gate_exps) / hp.n_expert / 1e6);
  const double bytes = hp.n_layer * KS * 3.0 * ggml_nbytes(m.layers[0].gate_exps) / hp.n_expert;
  for (int t = 0; t < tokens; t++) {
    double ms[3] = {0, 0, 0};
    for (int impl = 0; impl < 3; impl++) {  // fresh experts for each implementation
      std::vector<int> ids;
      std::vector<float> w, x, y(E);
      for (int L = 0; L < hp.n_layer; L++) {
        pick(ids, w, x);
        auto t0 = Clock::now();
        if (impl == 0) {
          const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
          kern.forward(ex, x.data(), ids.data(), w.data(), KS, y.data());
        } else if (impl == 1) {
          const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
          kern.forward_canon(ex, x.data(), ids.data(), w.data(), KS, y.data());
        } else {
          const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
          volatile uint64_t sink = kern.touch(ex, ids.data(), KS);
          (void)sink;
        }
        ms[impl] += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
    }
    printf("token %d: ours %.1f ms (%.0f GB/s)   canonical %.1f ms (%.0f GB/s)   read-only control %.1f ms (%.0f GB/s)\n", t,
           ms[0], bytes / ms[0] / 1e6, ms[1], bytes / ms[1] / 1e6, ms[2], bytes / ms[2] / 1e6);
  }
  return 0;
}
