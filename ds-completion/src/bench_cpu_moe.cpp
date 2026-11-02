// bench-cpu-moe: the CPU routed-expert kernel (cpu_moe.h) against ggml's mul_mat_id path (what
// the runtime used so far) and against a double-precision host computation of the reference math.
//
//   bench-cpu-moe -m <model.gguf> [-t 32] [--tokens 5] [--check-layers 0,20,39] [-k 6] [--batch T]
//
// --batch T: prompt chunks (forward_batch): T tokens all routed to the same k experts per layer,
// batched against T one-token calls, fast and canonical, outputs compared bit for bit.
// --replay LOG [--placement FILE --resident N --half 256]: the runtime's prompt workload: the
// prompt tokens of a --log-routing file in half-chunks, per layer forward_batch on the pairs whose
// expert is not among the placement's top N (the CPU's); random inputs; time per phase.
// -k: experts per layer in the speed part (the runtime's CPU usually gets 1-2: its misses)
// --decode D[,D...] [--layers 4] [--steps 30]: a verify step's CPU work (README step 56): per layer
// one forward_batch (canonical) of D tokens, each with its own fresh expert; the dynamic vs a
// static schedule, on the model's mapping (4 KB pages) vs a copy of the first --layers layers'
// experts in 2 MB pages (madvise(MADV_HUGEPAGE)); the read-only control on both. --gap-us N: idle
// N us between calls (the runtime's host waits ~1-2 ms per layer for the GPU; workers may sleep).
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
#include <sys/mman.h>
#include <thread>
#include <omp.h>

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
  int threads = 32, tokens = 5, kspeed = 0, batch = 0, resident_n = 6578, half = 256, dec_layers = 4, dec_steps = 30;
  std::vector<int> decode;
  CpuMoeKernel::Sched sched = CpuMoeKernel::kDynamic;
  int gap_us = 0;
  std::string replay, placement_path = "placements/study30.txt";
  std::vector<int> check = {0, 20, 39};
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-m") path = argv[++i];
    else if (a == "-t") threads = atoi(argv[++i]);
    else if (a == "--tokens") tokens = atoi(argv[++i]);
    else if (a == "-k") kspeed = atoi(argv[++i]);
    else if (a == "--batch") batch = atoi(argv[++i]);
    else if (a == "--replay") replay = argv[++i];
    else if (a == "--placement") placement_path = argv[++i];
    else if (a == "--resident") resident_n = atoi(argv[++i]);
    else if (a == "--half") half = atoi(argv[++i]);
    else if (a == "--decode") { for (char* q = argv[++i]; *q;) { decode.push_back((int)strtol(q, &q, 10)); if (*q == ',') q++; } }
    else if (a == "--layers") dec_layers = atoi(argv[++i]);
    else if (a == "--steps") dec_steps = atoi(argv[++i]);
    else if (a == "--gap-us") gap_us = atoi(argv[++i]);  // --decode: idle between calls (the GPU's part of a layer)
    else if (a == "--static") sched = CpuMoeKernel::kStatic;  // forward_batch's schedule (--replay, --batch)
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
  kern.batch_sched = sched;
  if (!decode.empty()) {
    const int NL = std::min(dec_layers, hp.n_layer);
    auto anon_huge_kb = [] {
      FILE* f = fopen("/proc/meminfo", "r");
      char line[256];
      long v = -1;
      while (f && fgets(line, sizeof line, f)) if (sscanf(line, "AnonHugePages: %ld kB", &v) == 1) break;
      if (f) fclose(f);
      return v;
    };
    // the 2 MB-page copy of the first NL layers' expert tensors
    constexpr size_t kHuge = 2u << 20;
    std::vector<ggml_tensor> huge(3 * NL);
    size_t total = 0;
    for (int L = 0; L < NL; L++)
      for (const ggml_tensor* t : {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps})
        total += (ggml_nbytes(t) + kHuge - 1) / kHuge * kHuge;
    const long hp0 = anon_huge_kb();
    uint8_t* hb = (uint8_t*)mmap(nullptr, total + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (hb == MAP_FAILED) { perror("mmap"); return 1; }
    hb = (uint8_t*)(((uintptr_t)hb + kHuge - 1) / kHuge * kHuge);
    if (madvise(hb, total, MADV_HUGEPAGE)) perror("madvise");
    size_t off = 0;
    for (int L = 0; L < NL; L++) {
      const ggml_tensor* src[3] = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps};
      for (int i = 0; i < 3; i++) {
        huge[3 * L + i] = *src[i];
        huge[3 * L + i].data = hb + off;
        memcpy(hb + off, src[i]->data, ggml_nbytes(src[i]));
        off += (ggml_nbytes(src[i]) + kHuge - 1) / kHuge * kHuge;
      }
    }
    printf("# decode: %d layers' experts copied (%.1f GB); anonymous huge pages %ld -> %ld MB\n", NL, total / 1e9,
           hp0 / 1024, anon_huge_kb() / 1024);
    const double ex_bytes = 3.0 * ggml_nbytes(m.layers[0].gate_exps) / hp.n_expert;
    printf("# per call: D tokens, each its own fresh expert (%.1f MB); %d steps x %d layers; ms per expert (GB/s)\n",
           ex_bytes / 1e6, dec_steps, NL);
    std::vector<float> x((size_t)8 * E);
    for (auto& v : x) v = nd(rng);
    for (int D : decode) {
      if (D < 1 || D > 8) continue;
      std::vector<float> out((size_t)D * E);
      double res[2][3];  // [pages][dynamic, static, control]
      for (int rep = 0; rep < 2; rep++)  // twice, the same experts, interleaved; the second counts
        for (int pg = 0; pg < 2; pg++)
          for (int v = 0; v < 3; v++) {
            std::mt19937 r2(1000 + D);
            kern.batch_sched = v == 1 ? CpuMoeKernel::kStatic : CpuMoeKernel::kDynamic;
            double ms = 0;
            for (int st = 0; st < dec_steps; st++)
              for (int L = 0; L < NL; L++) {
                std::set<int> s;
                while ((int)s.size() < D) s.insert((int)(r2() % hp.n_expert));
                std::vector<int> ids(s.begin(), s.end());
                std::vector<float> w(D, 0.5f);
                const CpuExperts ex = pg ? CpuExperts{&huge[3 * L], &huge[3 * L + 1], &huge[3 * L + 2], hp.swiglu_clamp[L]}
                                         : CpuExperts{m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
                std::vector<CpuMoeKernel::CpuRoute> r(D);
                for (int t = 0; t < D; t++) r[t] = {ids.data() + t, w.data() + t, 1};
                auto t0 = Clock::now();
                if (v < 2) kern.forward_batch(ex, x.data(), D, r.data(), true, out.data());
                else { volatile uint64_t sink = kern.touch(ex, ids.data(), D); (void)sink; }
                ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                if (gap_us) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
              }
            res[pg][v] = ms / (dec_steps * NL * D);
          }
      kern.batch_sched = sched;
      auto f = [&](double ms) { char b[64]; snprintf(b, sizeof b, "%.3f (%3.0f)", ms, ex_bytes / ms / 1e6); return std::string(b); };
      printf("D %d: 4 KB pages: dynamic %s  static %s  control %s | 2 MB pages: dynamic %s  static %s  control %s\n", D,
             f(res[0][0]).c_str(), f(res[0][1]).c_str(), f(res[0][2]).c_str(), f(res[1][0]).c_str(), f(res[1][1]).c_str(),
             f(res[1][2]).c_str());
    }
    // the static schedule gives the same bits (every row's arithmetic is unchanged)
    {
      std::vector<int> ids = {3, 77, 150, 299};
      std::vector<float> w = {0.3f, 0.2f, 0.4f, 0.1f};
      std::vector<CpuMoeKernel::CpuRoute> r = {{ids.data(), w.data(), 2}, {ids.data() + 1, w.data() + 1, 3}, {ids.data(), w.data(), 4}};
      long diff = 0;
      for (int L = 0; L < NL; L++) {
        const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
        std::vector<float> a(9 * E), b(9 * E);
        for (int canon = 0; canon < 2; canon++) {
          kern.batch_sched = CpuMoeKernel::kDynamic;
          kern.forward_batch(ex, x.data(), 3, r.data(), canon, a.data());
          kern.batch_sched = CpuMoeKernel::kStatic;
          kern.forward_batch(ex, x.data(), 3, r.data(), canon, b.data());
          diff += memcmp(a.data(), b.data(), (canon ? 9 : 3) * E * 4) != 0;
        }
      }
      kern.batch_sched = sched;
      printf("# static vs dynamic, 3 tokens sharing experts, fast and canonical, %d layers: %ld differ\n", NL, diff);
    }
    // where a call's time goes (README step 57): forward_batch's phases per call, the one-token
    // kernel on the same experts, and an empty parallel region (fork + join of the thread team)
    for (int gap : {0, 1500}) {
      printf("# per call, gap %d us: ms\n", gap);
      for (int D : decode) {
        if (D < 1 || D > 8) continue;
        std::vector<float> out((size_t)D * E);
        std::mt19937 r2(2000 + D);
        double wall_b = 0, wall_1 = 0, wall_e = 0;
        const int n = dec_steps * NL;
        kern.bstats = {};
        for (int pass = 0; pass < 3; pass++)
          for (int st = 0; st < dec_steps; st++)
            for (int L = 0; L < NL; L++) {
              std::set<int> sset;
              while ((int)sset.size() < D) sset.insert((int)(r2() % hp.n_expert));
              std::vector<int> ids(sset.begin(), sset.end());
              std::vector<float> w(D, 0.5f);
              const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
              auto t0 = Clock::now();
              if (pass == 0) {
                std::vector<CpuMoeKernel::CpuRoute> r(D);
                for (int t = 0; t < D; t++) r[t] = {ids.data() + t, w.data() + t, 1};
                kern.forward_batch(ex, x.data(), D, r.data(), true, out.data());
              } else if (pass == 1) {
                kern.forward_canon_rows(ex, x.data(), ids.data(), w.data(), D, out.data());  // one token, D experts
              } else {
                volatile int sink = 0;
#pragma omp parallel num_threads(threads)
                { if (omp_get_thread_num() == 0) sink = 1; }
                (void)sink;
              }
              const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
              (pass == 0 ? wall_b : pass == 1 ? wall_1 : wall_e) += ms;
              if (gap) std::this_thread::sleep_for(std::chrono::microseconds(gap));
            }
        const auto& b = kern.bstats;
        printf("D %d: forward_batch %.3f = prep %.3f + inputs %.3f + gate_up %.3f + act %.3f + down %.3f + sum %.3f | "
               "one-token kernel %.3f | empty parallel region %.3f\n", D, wall_b / n, b.prep / n, b.inputs / n,
               b.gate_up / n, b.act / n, b.down / n, b.sum / n, wall_1 / n, wall_e / n);
      }
    }
    return 0;
  }
  if (!replay.empty()) {
    // resident pairs: the placement's top N
    std::vector<char> res((size_t)hp.n_layer * hp.n_expert, 0);
    {
      FILE* pf = fopen(placement_path.c_str(), "r");
      if (!pf) { perror(placement_path.c_str()); return 1; }
      char line[256];
      int n = 0;
      while (n < resident_n && fgets(line, sizeof line, pf)) {
        int L, e;
        if (line[0] == '#' || sscanf(line, "%d %d", &L, &e) != 2) continue;
        if (L < hp.n_layer) res[(size_t)L * hp.n_expert + e] = 1;
        n++;
      }
      fclose(pf);
    }
    // the routing log (version 2): header "DSRT" v n_layer k, records pos token prompt step ids[L][k] w[L][k]
    FILE* lf = fopen(replay.c_str(), "rb");
    if (!lf) { perror(replay.c_str()); return 1; }
    char magic[4];
    int32_t hdr[3];
    if (fread(magic, 1, 4, lf) != 4 || memcmp(magic, "DSRT", 4) || fread(hdr, 4, 3, lf) != 3 || hdr[0] != 2) { fprintf(stderr, "not a v2 routing log\n"); return 1; }
    const int nl = hdr[1], kk = hdr[2];
    std::vector<std::vector<int16_t>> ids;  // per prompt token [nl * kk]
    std::vector<std::vector<float>> ws;
    for (;;) {
      int32_t h4[4];
      std::vector<int16_t> id((size_t)nl * kk);
      std::vector<float> w((size_t)nl * kk);
      if (fread(h4, 4, 4, lf) != 4 || fread(id.data(), 2, id.size(), lf) != id.size() || fread(w.data(), 4, w.size(), lf) != w.size()) break;
      if (h4[2]) ids.push_back(id), ws.push_back(w);
    }
    fclose(lf);
    const int T = (int)ids.size();
    printf("# replay %s: %d prompt tokens, halves of %d, the placement's top %d resident\n", replay.c_str(), T, half, resident_n);
    std::vector<float> x((size_t)half * E);
    for (auto& v : x) v = nd(rng);
    std::vector<float> out((size_t)half * KS * E);
    for (int canon = 0; canon < 2; canon++) {
      kern.bstats = {};
      auto t0 = Clock::now();
      for (int h0 = 0; h0 < T; h0 += half) {
        const int n = std::min(half, T - h0);
        for (int L = 0; L < hp.n_layer && L < nl; L++) {
          std::vector<int> cid((size_t)n * K);
          std::vector<float> cw((size_t)n * K);
          std::vector<CpuMoeKernel::CpuRoute> r(n);
          for (int t = 0; t < n; t++) {
            int c = 0;
            for (int j = 0; j < kk; j++) {
              const int e = ids[h0 + t][(size_t)L * kk + j];
              if (e < 0 || res[(size_t)L * hp.n_expert + e]) continue;
              cid[(size_t)t * K + c] = e;
              cw[(size_t)t * K + c] = ws[h0 + t][(size_t)L * kk + j];
              c++;
            }
            r[t] = {cid.data() + (size_t)t * K, cw.data() + (size_t)t * K, c};
          }
          const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
          kern.forward_batch(ex, x.data(), n, r.data(), canon, out.data());
        }
      }
      const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      const auto& b = kern.bstats;
      printf("%s: %.0f ms, %ld pairs over %ld experts (%.3f ms per pair); ms: prep %.0f inputs %.0f gate_up %.0f act %.0f down %.0f sum %.0f\n",
             canon ? "canonical" : "fast", ms, b.pairs, b.experts, ms / b.pairs, b.prep, b.inputs, b.gate_up, b.act, b.down, b.sum);
    }
    return 0;
  }
  if (batch > 0) {
    const int T = batch;
    printf("# batch: %d tokens x the same %d experts, all %d layers\n", T, KS, hp.n_layer);
    for (int canon = 0; canon < 2; canon++) {
      double ms_one = 0, ms_batch = 0;
      long diff = 0;
      for (int L = 0; L < hp.n_layer; L++) {
        std::vector<int> ids;
        std::vector<float> w, x0, x((size_t)T * E);
        pick(ids, w, x0);
        for (auto& v : x) v = nd(rng);
        const CpuExperts ex = {m.layers[L].gate_exps, m.layers[L].up_exps, m.layers[L].down_exps, hp.swiglu_clamp[L]};
        const size_t rows = canon ? (size_t)T * KS : (size_t)T;
        std::vector<float> y1(rows * E), y2(rows * E);
        auto t0 = Clock::now();
        for (int t = 0; t < T; t++) {
          if (canon) kern.forward_canon_rows(ex, x.data() + (size_t)t * E, ids.data(), w.data(), KS, y1.data() + (size_t)t * KS * E);
          else kern.forward(ex, x.data() + (size_t)t * E, ids.data(), w.data(), KS, y1.data() + (size_t)t * E);
        }
        auto t1 = Clock::now();
        std::vector<CpuMoeKernel::CpuRoute> r(T, {ids.data(), w.data(), KS});
        kern.forward_batch(ex, x.data(), T, r.data(), canon, y2.data());
        auto t2 = Clock::now();
        ms_one += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ms_batch += std::chrono::duration<double, std::milli>(t2 - t1).count();
        diff += memcmp(y1.data(), y2.data(), y1.size() * 4) != 0;
      }
      printf("%s: one token at a time %.1f ms, batched %.1f ms (%.3f vs %.3f ms per expert-token); layers differing: %ld\n",
             canon ? "canonical" : "fast", ms_one, ms_batch, ms_one / (hp.n_layer * T * KS), ms_batch / (hp.n_layer * T * KS), diff);
    }
    return 0;
  }
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
