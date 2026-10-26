// common.h: shared by mm-bench and mm-split.
// Weights from an mlp-extract file, host reference, device graphs, timing.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static void die(const std::string& m) {
  fprintf(stderr, "error: %s\n", m.c_str());
  exit(1);
}

static std::string slurp_line(const std::string& path) {
  std::ifstream f(path);
  std::string s, all;
  while (std::getline(f, s))
    if (s.find('*') != std::string::npos) all += s;
  return all.empty() ? "?" : all;
}

// Clock state of every amdgpu card, so each result file says what it ran at.
static void print_clocks() {
  for (int c = 0; c < 8; c++) {
    std::string d = "/sys/class/drm/card" + std::to_string(c) + "/device/";
    std::ifstream probe(d + "pp_dpm_sclk");
    if (!probe) continue;
    std::string lvl;
    std::getline(std::ifstream(d + "power_dpm_force_performance_level"), lvl);
    printf("# card%d perf=%s sclk=[%s] mclk=[%s]\n", c, lvl.c_str(), slurp_line(d + "pp_dpm_sclk").c_str(),
           slurp_line(d + "pp_dpm_mclk").c_str());
  }
}

// ---------------------------------------------------------------- weights

struct Layer {
  std::string path;
  gguf_context* g = nullptr;
  ggml_context* meta = nullptr;  // tensor descriptors (no data)
  ggml_tensor* find(const std::string& part) const {
    for (ggml_tensor* t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t))
      if (strstr(t->name, (part + ".weight").c_str())) return t;
    die("tensor not in file: " + part);
    return nullptr;
  }
  std::vector<uint8_t> bytes(const ggml_tensor* t) const {
    int64_t id = gguf_find_tensor(g, t->name);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    std::vector<uint8_t> b(ggml_nbytes(t));
    FILE* f = fopen(path.c_str(), "rb");
    if (!f || fseeko(f, (off_t)off, SEEK_SET) || fread(b.data(), 1, b.size(), f) != b.size())
      die("read failed: " + path);
    fclose(f);
    return b;
  }
};

static Layer open_layer(const char* path) {
  Layer L;
  L.path = path;
  gguf_init_params p = {true, &L.meta};
  L.g = gguf_init_from_file(path, p);
  if (!L.g) die(std::string("cannot open ") + path);
  return L;
}

// Host F32 copy of W, row-major [M][K] (row m = output m), via ggml's to_float.
static std::vector<float> dequant(const ggml_tensor* t, const std::vector<uint8_t>& b) {
  const int64_t K = t->ne[0], M = t->ne[1];
  const auto* tr = ggml_get_type_traits(t->type);
  if (!tr->to_float) die(std::string("no to_float for ") + ggml_type_name(t->type));
  std::vector<float> w(K * M);
  const size_t rb = ggml_row_size(t->type, K);
  for (int64_t m = 0; m < M; m++) tr->to_float(b.data() + m * rb, w.data() + m * K, K);
  return w;
}

// ref[n*M + m] = sum_k W[m][k] * x[n*K + k], double accumulation, all threads.
static std::vector<float> reference(const std::vector<float>& w, const std::vector<float>& x, int64_t K,
                                    int64_t M, int64_t N) {
  std::vector<float> ref(N * M);
  int nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++)
    th.emplace_back([&, t] {
      for (int64_t m = t; m < M; m += nt) {
        const float* wr = &w[m * K];
        for (int64_t n = 0; n < N; n++) {
          const float* xc = &x[n * K];
          double s = 0;
#pragma omp simd reduction(+ : s)
          for (int64_t k = 0; k < K; k++) s += (double)wr[k] * xc[k];
          ref[n * M + m] = (float)s;
        }
      }
    });
  for (auto& t : th) t.join();
  return ref;
}

static double rel_err(const float* y, const float* r, size_t n, double* max_abs) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)y[i] - r[i];
    num += d * d;
    den += (double)r[i] * r[i];
    mx = std::max(mx, std::fabs(d));
  }
  if (max_abs) *max_abs = mx / std::sqrt(den / n);  // max error in units of ref RMS
  return std::sqrt(num / den);
}

// ---------------------------------------------------------------- device side

struct DevWeight {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buf = nullptr;
  ggml_tensor* w = nullptr;
};

static DevWeight upload(ggml_backend_t be, const ggml_tensor* src, const std::vector<uint8_t>& b) {
  DevWeight d;
  ggml_init_params ip = {ggml_tensor_overhead() * 2, nullptr, true};
  d.ctx = ggml_init(ip);
  d.w = ggml_new_tensor_2d(d.ctx, src->type, src->ne[0], src->ne[1]);
  ggml_set_name(d.w, src->name);
  d.buf = ggml_backend_alloc_ctx_tensors(d.ctx, be);
  ggml_backend_tensor_set(d.w, b.data(), 0, b.size());
  return d;
}

// One graph: R independent y_r = W x. x lives in its own static buffer
// (gallocr would otherwise recycle an input's memory mid-graph).
struct MatGraph {
  ggml_context* xctx = nullptr;
  ggml_backend_buffer_t xbuf = nullptr;
  ggml_tensor* x = nullptr;
  ggml_context* ctx = nullptr;
  ggml_cgraph* gf = nullptr;
  ggml_gallocr_t galloc = nullptr;
  std::vector<ggml_tensor*> ys;

  void build(ggml_backend_t be, ggml_tensor* w, int64_t N, int R, const float* xdata) {
    ggml_init_params xp = {ggml_tensor_overhead() * 2, nullptr, true};
    xctx = ggml_init(xp);
    x = ggml_new_tensor_2d(xctx, GGML_TYPE_F32, w->ne[0], N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    xbuf = ggml_backend_alloc_ctx_tensors(xctx, be);
    ggml_backend_tensor_set(x, xdata, 0, ggml_nbytes(x));

    ggml_init_params gp = {ggml_tensor_overhead() * (R + 8) + ggml_graph_overhead(), nullptr, true};
    ctx = ggml_init(gp);
    gf = ggml_new_graph(ctx);
    for (int r = 0; r < R; r++) {
      ggml_tensor* y = ggml_mul_mat(ctx, w, x);
      ggml_set_output(y);
      ggml_build_forward_expand(gf, y);
      ys.push_back(y);
    }
    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) die("graph alloc failed");
  }
  void free_all() {
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(xbuf);
    ggml_free(xctx);
  }
};

struct Timing {
  double med, p10, p90;
  int reps;
};

static Timing time_graph(ggml_backend_t be, ggml_cgraph* gf, double min_time) {
  for (int i = 0; i < 5; i++) ggml_backend_graph_compute(be, gf);  // warm-up (and HIP graph capture)
  std::vector<double> t;
  auto start = Clock::now();
  while (t.size() < 20 || (std::chrono::duration<double>(Clock::now() - start).count() < min_time && t.size() < 20000)) {
    auto t0 = Clock::now();
    ggml_backend_graph_compute(be, gf);
    t.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return {t[t.size() / 2], t[t.size() / 10], t[t.size() * 9 / 10], (int)t.size()};
}

// Keep warnings/errors; drop ggml's per-graph INFO chatter ("CUDA graph warmup ...").
static void log_filter(ggml_log_level level, const char* text, void*) {
  if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static std::vector<int64_t> parse_list(const char* s) {
  std::vector<int64_t> v;
  for (const char* p = s; *p;) {
    v.push_back(strtoll(p, (char**)&p, 10));
    if (*p == ',') p++;
  }
  return v;
}


// The idx-th GPU device (ggml order = HIP order), or die.
static ggml_backend_dev_t gpu_dev(int idx) {
  for (size_t i = 0, k = 0; i < ggml_backend_dev_count(); i++) {
    ggml_backend_dev_t d = ggml_backend_dev_get(i);
    if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
    if ((int)k++ == idx) return d;
  }
  die("no GPU device " + std::to_string(idx));
  return nullptr;
}

// Deterministic uniform[-1,1] activations, [n][k] (column n at x[n*k]).
static std::vector<float> make_x(int64_t k, int64_t n) {
  std::vector<float> x(n * k);
  std::mt19937 rng(1234 + (int)k);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (auto& v : x) v = U(rng);
  return x;
}
