// mm-split: one MLP matmul split across several dies, same llama.cpp kernel on each slice.
//
//   mm-split <layer.gguf> [--tensor gate|up|down|all] [--mode row,k] [--ns 1,2,..]
//            [--configs 0/0,1/0,2/0,1,2,3] [--min-time S] [--stream R]
//
// Modes (W is K x M: K inputs, M outputs; x is [K, N]):
//   row  die d holds output rows [d*M/D, (d+1)*M/D) and gets the full x.
//        Merge = place the slices side by side. Every output row is computed from
//        the full input, so the merge must be BIT-IDENTICAL to the 1-die result.
//   k    die d holds input columns [d*K/D, (d+1)*K/D) of every row (a cut at quant
//        block boundaries) and gets the matching slice of x. Merge = sum of the D
//        partial [M, N] outputs on the host. Only the summation order changes, so
//        it must match the 1-die result to ~1e-6 and the reference as well as it does.
//
// Per (config, N), wall times (median, after warm-up):
//   slice    one slice's kernel alone on the first die of the config (16 per graph)
//   compute  all dies run their slice concurrently
//   e2e      H2D x (pinned) -> compute -> D2H y (pinned) -> merge on the host
// compute and e2e are measured twice: "ser" issues every die's work from the main
// thread (async calls, then sync all), "thr" gives each die its own spinning worker
// thread. The merge runs on 8 spinning threads and reads each partial once.
// A phases line (serial issue) splits e2e into launch / h2d / d2h / merge.
// The 1-die config runs through the same path, so its e2e is the baseline.
// Cross-card traffic goes through host memory only (ROCm 6.3.4 crashes on
// device-to-device copies between the two Duo cards).

#include "common.h"

#include <atomic>
#include <functional>
#include <immintrin.h>

// Persistent spinning worker threads: run(job) calls job(i) on every worker and
// returns when all are done. Spinning, so a hand-off costs ~1 us, not a futex wake.
struct SpinPool {
  int n = 0;
  std::vector<std::thread> th;
  std::atomic<uint64_t> gen{0};
  std::atomic<int> done{0};
  std::atomic<bool> quit{false};
  std::function<void(int)> job;
  explicit SpinPool(int n_) : n(n_) {
    for (int i = 0; i < n; i++)
      th.emplace_back([this, i] {
        uint64_t seen = 0;
        for (;;) {
          while (gen.load() == seen && !quit.load()) _mm_pause();
          if (quit.load()) return;
          seen = gen.load();
          job(i);
          done.fetch_add(1);
        }
      });
  }
  void run(std::function<void(int)> j) {
    job = std::move(j);
    done.store(0);
    gen.fetch_add(1);
    while (done.load() < n) _mm_pause();
  }
  ~SpinPool() {
    quit.store(true);
    for (auto& t : th) t.join();
  }
};

struct Part {
  int dev = 0;
  ggml_backend_t be = nullptr;
  DevWeight w;
  ggml_backend_buffer_t hbuf = nullptr;  // pinned host staging: x slice, then y slice
  float* hx = nullptr;
  float* hy = nullptr;
  int64_t K = 0, M = 0;  // this slice's shape
};

static std::vector<std::vector<int>> parse_configs(const std::string& s) {
  std::vector<std::vector<int>> out;
  size_t a = 0;
  while (a <= s.size()) {
    size_t b = s.find('/', a);
    if (b == std::string::npos) b = s.size();
    std::vector<int> c;
    for (int64_t v : parse_list(s.substr(a, b - a).c_str())) c.push_back((int)v);
    if (!c.empty()) out.push_back(c);
    a = b + 1;
  }
  return out;
}

static std::string cfg_name(const std::vector<int>& c) {
  std::string s;
  for (size_t i = 0; i < c.size(); i++) s += (i ? "," : "") + std::to_string(c[i]);
  return s;
}

// Weight bytes for slice d of D in the given mode.
static std::vector<uint8_t> slice_bytes(const ggml_tensor* t, const std::vector<uint8_t>& all, bool kmode, int d,
                                        int D, int64_t* K_out, int64_t* M_out) {
  const int64_t K = t->ne[0], M = t->ne[1];
  const size_t rb = ggml_row_size(t->type, K);
  if (!kmode) {
    if (M % D) die("M not divisible by D");
    int64_t Md = M / D;
    *K_out = K;
    *M_out = Md;
    return std::vector<uint8_t>(all.begin() + d * Md * rb, all.begin() + (d + 1) * Md * rb);
  }
  const int64_t Kd = K / D, blck = ggml_blck_size(t->type);
  if (K % D || Kd % blck) die("K slice not on a quant block boundary");
  const size_t off = ggml_row_size(t->type, d * Kd), len = ggml_row_size(t->type, Kd);
  std::vector<uint8_t> out(M * len);
  for (int64_t m = 0; m < M; m++) memcpy(&out[m * len], &all[m * rb + off], len);
  *K_out = Kd;
  *M_out = M;
  return out;
}

static Timing time_loop(double min_time, const std::function<void()>& f) {
  for (int i = 0; i < 5; i++) f();
  std::vector<double> t;
  auto start = Clock::now();
  while (t.size() < 20 || (std::chrono::duration<double>(Clock::now() - start).count() < min_time && t.size() < 20000)) {
    auto t0 = Clock::now();
    f();
    t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return {t[t.size() / 2], t[t.size() / 10], t[t.size() * 9 / 10], (int)t.size()};
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  ggml_log_set(log_filter, nullptr);
  if (argc < 2) {
    fprintf(stderr, "usage: mm-split <layer.gguf> [--tensor gate|up|down|all] [--mode row,k] [--ns 1,2,..]"
                    " [--configs 0/0,1/0,2/0,1,2,3] [--min-time S] [--stream R]\n");
    return 2;
  }
  const char* path = argv[1];
  std::string which = "all", modes = "row,k", cfgs = "0/0,1/0,2/0,1,2,3";
  std::vector<int64_t> ns = {1, 2, 4, 8, 16, 64, 256, 512};
  double min_time = 0.3;
  int R = 16;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--tensor") which = next();
    else if (a == "--mode") modes = next();
    else if (a == "--ns") ns = parse_list(next());
    else if (a == "--configs") cfgs = next();
    else if (a == "--min-time") min_time = atof(next());
    else if (a == "--stream") R = atoi(next());
    else die("unknown option " + a);
  }
  auto configs = parse_configs(cfgs);
  if (configs.empty() || configs[0].size() != 1) die("first config must be a single die (the baseline)");

  print_clocks();
  Layer L = open_layer(path);
  printf("# file %s  (%s, layer %u)\n", path, gguf_get_val_str(L.g, gguf_find_key(L.g, "mlp.source")),
         gguf_get_val_u32(L.g, gguf_find_key(L.g, "mlp.layer")));
  printf("# configs %s   modes %s   min-time %.2fs   slice stream R=%d\n", cfgs.c_str(), modes.c_str(), min_time, R);

  // One backend per die, created once.
  std::vector<ggml_backend_t> bes(8, nullptr);
  for (auto& c : configs)
    for (int d : c)
      if (!bes.at(d)) bes[d] = ggml_backend_dev_init(gpu_dev(d), nullptr);

  std::vector<std::string> parts = which == "all" ? std::vector<std::string>{"ffn_gate", "ffn_up", "ffn_down"}
                                                  : std::vector<std::string>{"ffn_" + which};
  const int64_t NMAX = *std::max_element(ns.begin(), ns.end());
  SpinPool mpool(8);

  for (const auto& part : parts) {
    const ggml_tensor* src = L.find(part);
    const int64_t K = src->ne[0], M = src->ne[1];
    auto bytes = L.bytes(src);
    auto x = make_x(K, NMAX);
    printf("\n== %s  %s  K=%" PRId64 " -> M=%" PRId64 "  %.2f MiB\n", src->name, ggml_type_name(src->type), K, M,
           ggml_nbytes(src) / 1048576.0);
    printf("  reference ... ");
    fflush(stdout);
    auto t0 = Clock::now();
    auto ref = reference(dequant(src, bytes), x, K, M, NMAX);
    printf("%.1fs\n", std::chrono::duration<double>(Clock::now() - t0).count());

    std::vector<std::vector<float>> base_y(ns.size());  // 1-die outputs, per N
    std::vector<double> base_e2e(ns.size(), 0);

    for (const std::string& mode : {std::string("row"), std::string("k")}) {
      if (modes.find(mode) == std::string::npos) continue;
      const bool kmode = mode == "k";
      printf("\n  -- mode %s   (merge: %s)\n", mode.c_str(), kmode ? "sum of partials" : "concatenate rows");
      printf("  %-9s %5s %8s %8s %8s %9s %9s %9s %5s %9s %s\n", "dies", "N", "slice", "comp_ser", "comp_thr",
             "e2e_ser", "e2e_thr", "speedup", "", "relerr", "vs 1-die");
      for (const auto& cfg : configs) {
        const int D = (int)cfg.size();
        if (D == 1 && kmode && !base_y[0].empty()) continue;  // 1 die is the same in both modes
        // Upload slices and pinned staging, once per config.
        std::vector<Part> P(D);
        for (int d = 0; d < D; d++) {
          P[d].dev = cfg[d];
          P[d].be = bes[cfg[d]];
          auto sb = slice_bytes(src, bytes, kmode, d, D, &P[d].K, &P[d].M);
          ggml_tensor tmpl = *src;
          tmpl.ne[0] = P[d].K;
          tmpl.ne[1] = P[d].M;
          P[d].w = upload(P[d].be, &tmpl, sb);
          size_t hb = (P[d].K + P[d].M) * NMAX * sizeof(float) + 4096;
          P[d].hbuf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(gpu_dev(cfg[d])), hb);
          if (!P[d].hbuf) die("pinned host alloc failed");
          P[d].hx = (float*)ggml_backend_buffer_get_base(P[d].hbuf);
          P[d].hy = P[d].hx + P[d].K * NMAX;
        }

        SpinPool dpool(D);
        for (size_t ni = 0; ni < ns.size(); ni++) {
          const int64_t N = ns[ni];
          // Per-die x slice into pinned memory (k mode: rows k0..k1 of every column).
          for (int d = 0; d < D; d++) {
            const int64_t k0 = kmode ? d * P[d].K : 0;
            for (int64_t n = 0; n < N; n++) memcpy(P[d].hx + n * P[d].K, &x[n * K + k0], P[d].K * 4);
          }
          std::vector<MatGraph> g1(D), gR(D);
          for (int d = 0; d < D; d++) g1[d].build(P[d].be, P[d].w.w, N, 1, P[d].hx);
          gR[0].build(P[0].be, P[0].w.w, N, R, P[0].hx);
          std::vector<float> y(M * N);

          // Merge on the host, split over the merge pool. k mode sums in die order
          // 0..D-1 for every element, so the result doesn't depend on thread count.
          auto merge = [&] {
            const int T = mpool.n;
            mpool.run([&](int t) {
              if (!kmode) {
                for (int64_t n = t; n < N; n += T)
                  for (int d = 0; d < D; d++) memcpy(&y[n * M + d * P[d].M], P[d].hy + n * P[d].M, P[d].M * 4);
              } else {
                const size_t cnt = M * N, a = cnt * t / T, b = cnt * (t + 1) / T;
                for (size_t i = a; i < b; i++) {
                  float acc = P[0].hy[i];
                  for (int d = 1; d < D; d++) acc += P[d].hy[i];
                  y[i] = acc;
                }
              }
            });
          };

          auto e2e = [&] {
            for (int d = 0; d < D; d++) {
              ggml_backend_tensor_set_async(P[d].be, g1[d].x, P[d].hx, 0, ggml_nbytes(g1[d].x));
              ggml_backend_graph_compute_async(P[d].be, g1[d].gf);
              ggml_backend_tensor_get_async(P[d].be, g1[d].ys[0], P[d].hy, 0, ggml_nbytes(g1[d].ys[0]));
            }
            for (int d = 0; d < D; d++) ggml_backend_synchronize(P[d].be);
            merge();
          };
          auto compute = [&] {
            for (int d = 0; d < D; d++) ggml_backend_graph_compute_async(P[d].be, g1[d].gf);
            for (int d = 0; d < D; d++) ggml_backend_synchronize(P[d].be);
          };
          // Same work, one worker thread per die.
          auto die_e2e = [&](int d) {
            ggml_backend_tensor_set_async(P[d].be, g1[d].x, P[d].hx, 0, ggml_nbytes(g1[d].x));
            ggml_backend_graph_compute_async(P[d].be, g1[d].gf);
            ggml_backend_tensor_get_async(P[d].be, g1[d].ys[0], P[d].hy, 0, ggml_nbytes(g1[d].ys[0]));
            ggml_backend_synchronize(P[d].be);
          };
          auto compute_thr = [&] {
            dpool.run([&](int d) {
              ggml_backend_graph_compute_async(P[d].be, g1[d].gf);
              ggml_backend_synchronize(P[d].be);
            });
          };
          auto e2e_thr = [&] {
            dpool.run(die_e2e);
            merge();
          };

          // Phases of e2e, each timed on its own (all dies issue, then sync all).
          std::vector<double> launch_t;
          auto compute_launch = [&] {
            auto a = Clock::now();
            for (int d = 0; d < D; d++) ggml_backend_graph_compute_async(P[d].be, g1[d].gf);
            launch_t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - a).count());
            for (int d = 0; d < D; d++) ggml_backend_synchronize(P[d].be);
          };
          auto h2d = [&] {
            for (int d = 0; d < D; d++) ggml_backend_tensor_set_async(P[d].be, g1[d].x, P[d].hx, 0, ggml_nbytes(g1[d].x));
            for (int d = 0; d < D; d++) ggml_backend_synchronize(P[d].be);
          };
          auto d2h = [&] {
            for (int d = 0; d < D; d++) ggml_backend_tensor_get_async(P[d].be, g1[d].ys[0], P[d].hy, 0, ggml_nbytes(g1[d].ys[0]));
            for (int d = 0; d < D; d++) ggml_backend_synchronize(P[d].be);
          };

          Timing ts = time_graph(P[0].be, gR[0].gf, min_time);
          Timing tc = time_loop(min_time, compute);
          Timing te = time_loop(min_time, e2e);
          Timing tct = time_loop(min_time, compute_thr);
          Timing tet = time_loop(min_time, e2e_thr);
          time_loop(min_time / 3, compute_launch);
          std::sort(launch_t.begin(), launch_t.end());
          Timing th = time_loop(min_time / 3, h2d);
          Timing tg = time_loop(min_time / 3, d2h);
          Timing tm = time_loop(min_time / 3, merge);

          // Correctness: a fresh e2e run, then compare.
          // Both issue paths must give the same bits.
          for (int d = 0; d < D; d++) memset(P[d].hy, 0xff, P[d].M * N * 4);  // poison: stale data can't pass
          std::fill(y.begin(), y.end(), NAN);
          e2e();
          std::vector<float> y_ser = y;
          for (int d = 0; d < D; d++) memset(P[d].hy, 0xff, P[d].M * N * 4);
          std::fill(y.begin(), y.end(), NAN);
          e2e_thr();
          const bool ser_thr_same = memcmp(y.data(), y_ser.data(), y.size() * 4) == 0;
          double mx = 0;
          double re = rel_err(y.data(), ref.data(), y.size(), &mx);
          std::string vs;
          if (D == 1) {
            base_y[ni] = y;
            base_e2e[ni] = std::min(te.med, tet.med);
            vs = "baseline";
          } else {
            size_t diff = 0;
            for (size_t i = 0; i < y.size(); i++) diff += memcmp(&y[i], &base_y[ni][i], 4) != 0;
            double rb = rel_err(y.data(), base_y[ni].data(), y.size(), nullptr);
            char s[96];
            if (!kmode) snprintf(s, sizeof s, "%s", diff ? "DIFFERS" : "bit-identical");
            else snprintf(s, sizeof s, "%.1e rel (%zu/%zu differ)", rb, diff, y.size());
            vs = s;
            if (!kmode && diff) vs += " (" + std::to_string(diff) + " values)";
          }
          bool ok = re < 2e-2 && ser_thr_same && (D == 1 || kmode || vs == "bit-identical");
          if (!ser_thr_same) vs += "  SERIAL != THREADED";
          const double best = std::min(te.med, tet.med);
          printf("  %-9s %5" PRId64 " %8.1f %8.1f %8.1f %9.1f %9.1f %8.2fx %5s %9.2e %s\n", cfg_name(cfg).c_str(), N,
                 ts.med / R, tc.med, tct.med, te.med, tet.med, base_e2e[ni] / best, ok ? "ok" : "FAIL", re, vs.c_str());
          printf("  %-9s %5s   phases: launch %.1f  h2d %.1f  d2h %.1f  merge %.1f  (us; bytes h2d %.0fK d2h %.0fK)\n", "",
                 "", launch_t[launch_t.size() / 2], th.med, tg.med, tm.med,
                 D * (double)ggml_nbytes(g1[0].x) / 1024, D * (double)ggml_nbytes(g1[0].ys[0]) / 1024);
          for (int d = 0; d < D; d++) g1[d].free_all();
          gR[0].free_all();
        }
        for (auto& p : P) {
          ggml_backend_buffer_free(p.w.buf);
          ggml_free(p.w.ctx);
          ggml_backend_buffer_free(p.hbuf);
        }
      }
    }
  }
  for (auto* be : bes)
    if (be) ggml_backend_free(be);
  return 0;
}
