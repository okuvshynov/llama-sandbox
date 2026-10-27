// ds-test: compare ds-completion against the reference's golden runs (ref/ds_ref.py, exported
// with ref/export_golden.py).
//
//   ds-test -m <model> -g data/golden/hello [--layers 0-39] [--no-quant] [-t 16]
//       Teacher-forced, per layer: every block gets the reference's own input for every step
//       (prefill chunk, then each decode token), and its outputs are compared to the
//       reference's. Errors cannot accumulate across layers, so each row tests one layer's
//       arithmetic in isolation, plus the state it builds (KV caches, compressor groups) from
//       reference inputs. Layers run in order up to the highest one asked for, because consumer
//       layers read the caches their source layer wrote.
//   ds-test -m <model> -g data/golden/hello-fp32 --gpu 0 --runtime <dense-fp8.gguf> (bin-hip)
//       As above, but the decode steps run through the own runtime (hip/rt.h); the prefill
//       step still runs through the Engine (GPU, fake quantization kept).
//   ds-test -m <model> -g data/golden/hello --e2e data/ds/hello.f32
//       End to end: the full forward on the reference's token sequence, the logits of every
//       step written as [steps][n_vocab] f32 (compare with ref/compare_logits.py).
//
// Columns: relative L2 error ||ours - ref|| / ||ref|| of each recorded intermediate; "experts"
// is the fraction of tokens whose 6 routed experts are the same set; "topk" the fraction of
// queries whose selected compressed positions are the same set (index-source layers).

#include "forward.h"
#ifdef DS_RUNTIME
#include "rt.h"
#endif

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

struct GTensor {
  bool is_int = false;
  std::vector<int64_t> shape;  // torch order (outermost first)
  const void* data = nullptr;
  int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
  const float* f() const { return (const float*)data; }
  const int32_t* i() const { return (const int32_t*)data; }
};

struct Golden {
  std::map<std::string, GTensor> t;  // "step/layer/key"
  int n_steps = 0;
  void* map = nullptr;

  void open(const std::string& dir) {
    int fd = ::open((dir + "/data.bin").c_str(), O_RDONLY);
    if (fd < 0) { perror("open data.bin"); exit(1); }
    struct stat st;
    fstat(fd, &st);
    map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    std::ifstream idx(dir + "/index.txt");
    std::string line;
    while (std::getline(idx, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream is(line);
      int step, layer, nd;
      std::string key, dt;
      is >> step >> layer >> key >> dt >> nd;
      GTensor g;
      g.is_int = dt == "i32";
      g.shape.resize(nd);
      for (auto& d : g.shape) is >> d;
      size_t off;
      is >> off;
      g.data = (const char*)map + off;
      t[std::to_string(step) + "/" + std::to_string(layer) + "/" + key] = g;
      n_steps = std::max(n_steps, step + 1);
    }
  }
  const GTensor* get(int s, int L, const std::string& key) const {
    auto it = t.find(std::to_string(s) + "/" + std::to_string(L) + "/" + key);
    return it == t.end() ? nullptr : &it->second;
  }
};

// The reference may have kept only the last positions (ds_ref.py --keep-last): positions are the
// outermost dim, so those are the tail of our values.
static double rel_err(const std::vector<float>& a, const GTensor& b) {
  if ((int64_t)a.size() < b.numel()) return NAN;
  const size_t off = a.size() - b.numel();
  double num = 0, den = 0;
  for (size_t i = 0; i < (size_t)b.numel(); i++) {
    const double d = a[off + i] - b.f()[i];
    num += d * d;
    den += (double)b.f()[i] * b.f()[i];
  }
  return sqrt(num / std::max(den, 1e-300));
}

static std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    char* end;
    long a = strtol(p, &end, 10), b = a;
    if (*end == '-') b = strtol(end + 1, &end, 10);
    for (long x = a; x <= b; x++) v.push_back((int)x);
    p = *end == ',' ? end + 1 : end;
  }
  return v;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string model_path, gdir, e2e, rt_path;
  bool determinism = false;
  std::vector<int> layers;
  Options opt;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m") model_path = next();
    else if (a == "-g") gdir = next();
    else if (a == "--layers") layers = parse_list(next());
    else if (a == "--no-quant") opt.exact_quant = false;
    else if (a == "-t") opt.n_threads = atoi(next());
    else if (a == "--gpu") opt.gpu = atoi(next());
    else if (a == "--gpu-exact") opt.gpu_exact_quant = true;
    else if (a == "--e2e") e2e = next();
    else if (a == "--runtime") rt_path = next();
    else if (a == "--determinism") determinism = true;
    else if (a == "--profile") opt.profile = true;  // GPU mode: also syncs at every backend switch
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (model_path.empty() || gdir.empty()) {
    fprintf(stderr, "usage: ds-test -m <model> -g <golden dir> [--layers 0-39] [--no-quant] [-t N] [--gpu D] [--e2e out.f32]\n");
    return 2;
  }
  Golden G;
  G.open(gdir);
  Model m;
  const int L_max = layers.empty() ? 39 : *std::max_element(layers.begin(), layers.end());
  std::vector<int> load;
  if (e2e.empty()) for (int L = 0; L <= L_max; L++) load.push_back(L);
  m.load(model_path, load);
  if (layers.empty()) for (int L = 0; L <= L_max; L++) layers.push_back(L);
  std::set<int> report(layers.begin(), layers.end());
  if (!rt_path.empty()) opt.gpu_exact_quant = true;  // prefill caches with the reference's rounding
  Engine eng(m, opt);
#ifdef DS_RUNTIME
  std::unique_ptr<Runtime> rt;
  if (!rt_path.empty()) rt.reset(new Runtime(eng, rt_path));
#else
  if (!rt_path.empty()) { fprintf(stderr, "--runtime needs the bin-hip build\n"); return 2; }
#endif
  printf("# golden %s: %d steps; exact fake-quant %s\n", gdir.c_str(), G.n_steps, opt.exact_quant ? "on" : "off");

  if (!e2e.empty()) {
    FILE* f = fopen(e2e.c_str(), "wb");
    for (int s = 0; s < G.n_steps; s++) {
      const GTensor* tk = G.get(s, -1, "tokens");
      std::vector<int32_t> toks(tk->i(), tk->i() + tk->numel());
      auto t0 = std::chrono::steady_clock::now();
#ifdef DS_RUNTIME
      std::vector<float> logits = (rt && toks.size() == 1) ? rt->decode(toks[0]) : eng.forward(toks);
#else
      std::vector<float> logits = eng.forward(toks);
#endif
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      fwrite(logits.data(), sizeof(float), logits.size(), f);
      const int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
      const GTensor* rl = G.get(s, -1, "logits");
      const int ref_best = (int)(std::max_element(rl->f(), rl->f() + rl->numel()) - rl->f());
      printf("step %2d: %3zu tokens -> %6d (ref %6d)%s  %.2f s\n", s, toks.size(), best, ref_best,
             best == ref_best ? "" : "  <- differs", dt);
    }
    fclose(f);
    return 0;
  }

  if (determinism) {
    // two engines, the same golden inputs, every tap compared bit for bit (prefill step only)
    Engine eng2(m, opt);
    const GTensor* tk = G.get(0, -1, "tokens");
    std::vector<int32_t> toks(tk->i(), tk->i() + tk->numel());
    eng.begin_chunk(toks);
    eng2.begin_chunk(toks);
    for (int L = 0; L <= L_max; L++) {
      const bool is_engram = m.layers[L].kind.engram >= 0;
      const GTensor* hin = G.get(0, L, is_engram ? "engram_in" : "h_in");
      const GTensor* pin = G.get(0, L, "pre_in");
      std::vector<float> h(hin->f(), hin->f() + hin->numel()), p(pin->f(), pin->f() + pin->numel());
      Taps a = eng.run_layer(L, h, p), b = eng2.run_layer(L, h, p);
      printf("layer %2d:", L);
      for (auto& [name, ta] : a) {
        const Tap& tb = b[name];
        size_t ndiff = 0;
        double mx = 0;
        for (size_t i = 0; i < ta.f.size(); i++)
          if (memcmp(&ta.f[i], &tb.f[i], 4)) { ndiff++; mx = std::max(mx, (double)fabsf(ta.f[i] - tb.f[i])); }
        for (size_t i = 0; i < ta.i.size(); i++) ndiff += ta.i[i] != tb.i[i];
        printf(" %s=%s", name.c_str(), ndiff ? (std::to_string(ndiff) + "(" + std::to_string(mx) + ")").c_str() : "same");
      }
      printf("\n");
    }
    return 0;
  }

  const char* keys[] = {"h_in", "attn_in", "attn_out", "ffn_in", "ffn_out", "h_out", "pre_out", "router_w"};
  printf("%4s %3s", "step", "L");
  for (const char* k : keys) printf(" %9s", k);
  printf(" %8s %8s %7s\n", "experts", "topk", "ms");
  const int hp_topk_limit = m.hp.idx_topk;  // when the reference picked a full top-k, also show the overlap
  for (int s = 0; s < G.n_steps; s++) {
    const GTensor* tk = G.get(s, -1, "tokens");
    std::vector<int32_t> toks(tk->i(), tk->i() + tk->numel());
    const int T = (int)toks.size();
#ifdef DS_RUNTIME
    const bool use_rt = rt && T == 1;  // decode steps through the runtime
#else
    const bool use_rt = false;
#endif
    const int p0_step = eng.st.n_past;
    if (use_rt) {
#ifdef DS_RUNTIME
      rt->begin_step(toks[0]);
#endif
    } else {
      eng.begin_chunk(toks);
    }
    for (int L = 0; L <= L_max; L++) {
      const bool is_engram = m.layers[L].kind.engram >= 0;
      const GTensor* hin = G.get(s, L, is_engram ? "engram_in" : "h_in");
      const GTensor* pin = G.get(s, L, "pre_in");
      if (!hin || !pin) { fprintf(stderr, "golden lacks step %d layer %d\n", s, L); return 1; }
      std::vector<float> h(hin->f(), hin->f() + hin->numel()), p(pin->f(), pin->f() + pin->numel());
      auto t0 = std::chrono::steady_clock::now();
#ifdef DS_RUNTIME
      Taps tp = use_rt ? rt->run_layer(L, h, p) : eng.run_layer(L, h, p);
#else
      Taps tp = eng.run_layer(L, h, p);
#endif
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!report.count(L)) continue;
      printf("%4d %3d", s, L);
      for (const char* k : keys) {
        const GTensor* ref = G.get(s, L, k);
        if (!ref || !tp.count(k) || (!strcmp(k, "h_in") && !is_engram)) { printf(" %9s", "-"); continue; }
        printf(" %9.2e", rel_err(tp[k].f, *ref));
      }
      // routed experts: same set per token
      const GTensor* ri = G.get(s, L, "router_i");
      const int Tr = (int)ri->shape[0], tr0 = T - Tr;  // positions the reference kept
      int same = 0;
      for (int t = 0; t < Tr; t++) {
        std::set<int> a(tp["router_i"].i.begin() + (tr0 + t) * 6, tp["router_i"].i.begin() + (tr0 + t) * 6 + 6);
        std::set<int> b(ri->i() + t * 6, ri->i() + t * 6 + 6);
        same += a == b;
      }
      printf(" %8.3f", (double)same / Tr);
      // indexer: same visible set per query (the reference marks invisible picks -1)
      const GTensor* rk = G.get(s, L, "topk");
      if (rk && tp.count("topk")) {
        const int K = (int)tp["topk"].ne[0], Kr = (int)rk->shape.back();
        const int r = m.layers[L].kind.ratio, p0 = p0_step;
        // the reference stores picks shifted by the window part's length: the prefill chunk's
        // own keys (T) at start_pos 0, the full ring (W) when decoding
        const int off = p0 == 0 ? T : m.hp.window;
        const int Tk = (int)rk->shape[0], tk0 = T - Tk;
        int eq = 0;
        double jac = 0;
        for (int t = 0; t < Tk; t++) {
          const int n_vis = (p0 + tk0 + t + 1) / r;
          std::set<int> a, b, u;
          for (int j = 0; j < K; j++) { int v = tp["topk"].i[(tk0 + t) * K + j]; if (v < n_vis) a.insert(v); }
          for (int j = 0; j < Kr; j++) {
            int v = rk->i()[t * Kr + j];
            if (v >= 0) b.insert(v - off);
          }
          eq += a == b;
          int inter = 0;
          for (int v : a) inter += b.count(v);
          u = a;
          u.insert(b.begin(), b.end());
          jac += u.empty() ? 1.0 : (double)inter / u.size();
        }
        printf(" %8.3f", (double)eq / Tk);
        if (Kr >= hp_topk_limit) printf("(J%.3f)", jac / Tk);
      } else {
        printf(" %8s", "-");
      }
      printf(" %7.0f\n", ms);
    }
    if (use_rt) {
#ifdef DS_RUNTIME
      rt->end_step();
#endif
    } else {
      eng.end_chunk();
    }
  }
  return 0;
}
