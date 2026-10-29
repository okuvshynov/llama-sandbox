// ds-completion: DeepSeek-V4.1-Flash text completion.
//
//   ds-completion -m <model-00001-of-000NN.gguf> -p "prompt" [--chat | --think] [-n 128]
//                 [-t 16] [--no-quant] [--max-ctx 4096]
//                 [--rt <dense-fp8.gguf> [--gpu D] [--placement file [--moe-dies 0,1,2,3] [--load-test R [--load-pinned]] [--cache-loads R [--cache-period N] [--cache-window N] [--cache-wait]]] [--canon] [--profile]]
//
// The prompt is fed as one chunk (prefill), then one token per step, greedy. --chat wraps the
// prompt as one user turn in the model's chat format without thinking, --think with it (the
// strings encoding/encoding.py produces); otherwise the prompt is raw text after BOS.
// Only the tokenizer comes from libllama (vocab-only load).
// Without --rt: everything on the CPU through the ggml Engine (forward.cpp), the prompt as one
// chunk. --rt <dense-fp8.gguf> (bin-hip build): every token through the own runtime (hip/rt.h:
// dense part on GPU D in exact fp8, routed experts on the CPU or, with --placement, partly on
// GPUs), the
// prompt one token at a time.

#include "forward.h"
#include "llama.h"
#ifdef DS_RUNTIME
#include "rt.h"
#endif

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <memory>
#include <string>

using Clock = std::chrono::steady_clock;

static std::vector<int32_t> tokenize(const llama_vocab* v, const std::string& text) {
  std::vector<llama_token> t(text.size() + 16);
  int n = llama_tokenize(v, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(),
                         /*add_special*/ false, /*parse_special*/ true);
  if (n < 0) {
    t.resize(-n);
    n = llama_tokenize(v, text.c_str(), (int32_t)text.size(), t.data(), (int32_t)t.size(), false, true);
  }
  t.resize(n);
  return std::vector<int32_t>(t.begin(), t.end());
}

static std::string piece(const llama_vocab* v, int32_t tok) {
  char buf[256];
  int n = llama_token_to_piece(v, tok, buf, sizeof buf, 0, /*special*/ false);
  return n > 0 ? std::string(buf, n) : std::string();
}

int main(int argc, char** argv) {
  std::string model_path, prompt, mode = "raw", rt_path;
  int n_predict = 128;
  Options opt;
  [[maybe_unused]] int gpu = 0;
  [[maybe_unused]] bool profile = false;
  std::string route_log;
  std::string placement;
  std::vector<int> moe_dies = {0, 1, 2, 3};
  [[maybe_unused]] double load_test = -1;
  [[maybe_unused]] bool load_pinned = false;
  [[maybe_unused]] bool canon = false;
  bool max_ctx_set = false;
  [[maybe_unused]] double cache_loads = 0;
  [[maybe_unused]] int cache_period = 16, cache_window = 256;
  [[maybe_unused]] bool cache_wait = false;
  [[maybe_unused]] std::string dspark_path;
  bool spec = false;
  [[maybe_unused]] int spec_n = 0;
  [[maybe_unused]] std::string spec_log_path;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m") model_path = next();
    else if (a == "-p") prompt = next();
    else if (a == "--chat") mode = "chat";
    else if (a == "--think") mode = "think";
    else if (a == "-n") n_predict = atoi(next());
    else if (a == "-t") opt.n_threads = atoi(next());
    else if (a == "--gpu") gpu = atoi(next());
    else if (a == "--dspark") dspark_path = next();
    else if (a == "--spec") spec = true;
    else if (a == "--spec-log") spec_log_path = next();  // per verify step: n k verify_ms draft_ms conf[0..]
    else if (a == "--spec-n") spec_n = atoi(next());  // fixed drafts verified per step (default: chosen per step)  // drafts verified per step (default: all the draft makes)  // with --rt and --dspark: speculative decoding (greedy, same output)  // runtime: the DSpark draft GGUF (shadow mode: drafts measured, not used)
    else if (a == "--placement") placement = next();  // runtime: routed experts on GPUs (expert_pool.h)
    else if (a == "--load-test") load_test = atof(next());  // runtime: background loads to spare slots (rt.h)
    else if (a == "--load-pinned") load_pinned = true;
    else if (a == "--canon") canon = true;  // runtime: placement-invariant expert numerics (rt.h)
    else if (a == "--cache-loads") cache_loads = atof(next());  // runtime: adaptive expert cache (expert_cache.h)
    else if (a == "--cache-period") cache_period = atoi(next());
    else if (a == "--cache-window") cache_window = atoi(next());
    else if (a == "--cache-wait") cache_wait = true;
    else if (a == "--moe-dies") {
      moe_dies.clear();
      for (char* q = (char*)next(); *q;) { moe_dies.push_back((int)strtol(q, &q, 10)); if (*q == ',') q++; }
    }
    else if (a == "--profile") profile = true;
    else if (a == "--rt") rt_path = next();
    else if (a == "--log-routing") route_log = next();  // runtime: every token's experts per layer
    else if (a == "--no-quant") opt.exact_quant = false;
    else if (a == "--max-ctx") max_ctx_set = true, opt.max_ctx = atoi(next());
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (model_path.empty() || prompt.empty()) {
    fprintf(stderr, "usage: ds-completion -m <model.gguf> -p \"prompt\" [--chat|--think] [-n N] [-t N] [--no-quant] [--max-ctx N] [--rt dense-fp8.gguf [--gpu D] [--placement file [--moe-dies 0,1,2,3] [--load-test R [--load-pinned]] [--cache-loads R [--cache-period N] [--cache-window N] [--cache-wait]]] [--canon] [--profile]]\n");
    return 2;
  }

  // tokenizer only
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* tok_model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!tok_model) { fprintf(stderr, "cannot load the tokenizer from %s\n", model_path.c_str()); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(tok_model);

  const std::string bos = "<｜begin▁of▁sentence｜>";
  std::string text;
  if (mode == "chat") text = bos + "<｜User｜>" + prompt + "<｜Assistant｜></think>";
  else if (mode == "think")
    text = bos + "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>" +
           prompt + "<｜Assistant｜><think>";
  else text = bos + prompt;
  std::vector<int32_t> toks = tokenize(vocab, text);
  // the GPU runtime defaults to the model's original training context (YaRN extends it 16x, to
  // 1M: --max-ctx); ~6.4 KB per position on the home die (compressed caches, index keys). The CPU
  // path keeps 4096 (it builds dense masks per prompt and stops at 16k: no candidate blocks)
  if (!rt_path.empty() && !max_ctx_set) opt.max_ctx = 65536;

  if ((int)toks.size() >= opt.max_ctx) {
    fprintf(stderr, "prompt is %zu tokens; --max-ctx is %d\n", toks.size(), opt.max_ctx);
    return 1;
  }
  if ((int)toks.size() + n_predict > opt.max_ctx) {
    n_predict = opt.max_ctx - (int)toks.size();
    fprintf(stderr, "# generation capped at %d tokens by --max-ctx %d\n", n_predict, opt.max_ctx);
  }
  Model m;
  auto t0 = Clock::now();
  try {
    m.load(model_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "load failed: %s\n", e.what());
    return 1;
  }
  {  // the experts that may run on the CPU: page tables now, not inside decode steps
    auto tm = Clock::now();
    const size_t n = m.map_experts(opt.n_threads);
    fprintf(stderr, "# mapped %.0f GB of CPU experts in %.1f s\n", n / 1e9,
            std::chrono::duration<double>(Clock::now() - tm).count());
  }
  std::unique_ptr<Engine> eng;
#ifdef DS_RUNTIME
  std::unique_ptr<Runtime> rt;
  if (!rt_path.empty()) {
    if (!opt.exact_quant) { fprintf(stderr, "--no-quant is CPU only\n"); return 2; }
    RtOptions ro;
    ro.gpu = gpu;
    ro.n_threads = opt.n_threads;
    ro.max_ctx = opt.max_ctx;
    ro.placement = placement;
    ro.moe_dies = moe_dies;
    ro.load_test = load_test;
    ro.load_pinned = load_pinned;
    ro.canon = canon;
    ro.cache_loads = cache_loads;
    ro.cache_period = cache_period;
    ro.cache_window = cache_window;
    ro.cache_wait = cache_wait;
    ro.dspark = dspark_path;
    if (spec && dspark_path.empty()) { fprintf(stderr, "--spec needs --dspark\n"); return 2; }
    rt.reset(new Runtime(m, ro, rt_path));
    rt->profile = profile;
    if (!route_log.empty()) rt->log_routing(route_log);
  }
#else
  if (!rt_path.empty()) { fprintf(stderr, "--rt needs the bin-hip build\n"); return 2; }
#endif
  if (rt_path.empty()) eng.reset(new Engine(m, opt));
  const double t_load = std::chrono::duration<double>(Clock::now() - t0).count();
  fprintf(stderr, "# loaded in %.1f s; prompt %zu tokens; fake-quant %s; %d threads\n", t_load, toks.size(),
          opt.exact_quant ? "on" : "off", opt.n_threads);

  auto t1 = Clock::now();
#ifdef DS_RUNTIME
  std::vector<float> logits = rt ? rt->prefill(toks) : eng->forward(toks);
#else
  std::vector<float> logits = eng->forward(toks);
#endif
  const double t_pp = std::chrono::duration<double>(Clock::now() - t1).count();
  const int eos = llama_vocab_eos(vocab);
  int n_gen = 0;
  auto t2 = Clock::now();
  std::string out;
  // shadow mode (--dspark): after every generated token, the draft's guesses for the tokens after
  // it; compared with what the main model then generates (nothing is skipped or verified)
  std::vector<int32_t> gen;
  std::vector<std::vector<int32_t>> drafts;
  std::vector<std::vector<float>> confs;
  [[maybe_unused]] double draft_ms = 0;
  [[maybe_unused]] long spec_steps = 0, spec_accepted = 0;
#ifdef DS_RUNTIME
  if (spec) {
    // speculative decoding (greedy): draft from the anchor (the last emitted token, not yet run),
    // verify [anchor, drafts], keep the matching prefix plus the main model's own next token;
    // prints exactly what plain greedy decoding prints
    int32_t anchor = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    auto emit = [&](int32_t t) {  // false: stop
      if (t == eos || n_gen >= n_predict) return false;
      std::string p = piece(vocab, t);
      fwrite(p.data(), 1, p.size(), stdout);
      fflush(stdout);
      gen.push_back(t);
      n_gen++;
      return n_gen < n_predict;
    };
    FILE* spec_log = spec_log_path.empty() ? nullptr : fopen(spec_log_path.c_str(), "w");
    if (spec_log) fprintf(spec_log, "# n_verified accepted verify_ms draft_ms conf[0..block)\n");
    // Draft length per step (unless --spec-n): the confidence head's sigmoid is the chance a draft
    // is accepted given the earlier ones were (calibrated: scripts/spec_conf.py, README step 25);
    // verify the n in 0..5 with the most expected tokens per ms, (1 + p0 + p0 p1 + ...) /
    // (draft + verify(n)); n = 0 is a plain step. Verify cost: a line a + b n, fitted by weighted
    // least squares to this run's steps (weights decay by 0.97 per step, ~30 steps) plus a weak
    // fixed prior, 52 + 28 n ms (sclk 2, study30) at n = 0..5 with a total weight of 3 steps.
    // A line, not one average per n: every step then updates the cost of every n, including
    // those the rule has not tried lately (with a good expert cache the per-token cost is lower).
    const int n_max = rt->draft_block();
    struct Fit { double w = 0, n = 0, t = 0, nn = 0, nt = 0; } obs, prior;
    for (int n = 0; n <= n_max; n++) {
      const double w = 3.0 / (n_max + 1), t = 52.0 + 28.0 * n;
      prior.w += w; prior.n += w * n; prior.t += w * t; prior.nn += w * n * n; prior.nt += w * n * t;
    }
    std::vector<double> verify_cost(n_max + 1);
    auto refit = [&] {
      const Fit f{obs.w + prior.w, obs.n + prior.n, obs.t + prior.t, obs.nn + prior.nn, obs.nt + prior.nt};
      const double b = (f.w * f.nt - f.n * f.t) / (f.w * f.nn - f.n * f.n), a = (f.t - b * f.n) / f.w;
      for (int n = 0; n <= n_max; n++) verify_cost[n] = a + b * n;
    };
    refit();
    double draft_cost = 12.0;
    std::vector<long> n_hist(verify_cost.size(), 0);
    auto choose_n = [&](const std::vector<float>& conf) {
      int best_n = 0;
      double best = -1, run = 1, expect = 1;
      for (int n = 0; n < (int)verify_cost.size(); n++) {
        if (n > 0) {
          run *= 1.0 / (1.0 + std::exp(-(double)conf[n - 1]));
          expect += run;
        }
        const double rate = expect / (draft_cost + verify_cost[n]);
        if (rate > best) { best = rate; best_n = n; }
      }
      return best_n;
    };
    // Backoff: after a step that chose n = 0, the next 1, 2, 4, .. (at most 8) steps are plain,
    // without drafting at all (the 12 ms draft would likely be wasted again); any step that
    // verifies a draft resets it. The draft's window is fed by every committed position either
    // way.
    int backoff = 0, skip = 0;
    long plain_steps = 0;
    if (emit(anchor)) {
      for (;;) {
        DraftResult d;
        double dms = 0;
        int n_verify = 0;
        if (skip > 0) {
          skip--;
          plain_steps++;
          d.ids = {anchor};
        } else {
          auto td = Clock::now();
          d = rt->draft(anchor);
          dms = std::chrono::duration<double, std::milli>(Clock::now() - td).count();
          draft_ms += dms;
          draft_cost += 0.1 * (dms - draft_cost);
          n_verify = spec_n > 0 ? spec_n : choose_n(d.conf);
          if (spec_n == 0) {
            backoff = n_verify == 0 ? std::min(backoff == 0 ? 1 : 2 * backoff, 8) : 0;
            skip = backoff;
          }
        }
        std::vector<int32_t> chunk(d.ids.begin(), d.ids.end());  // [anchor, drafts]
        if ((int)chunk.size() > n_verify + 1) chunk.resize(n_verify + 1);  // verify only the first n drafts
        const int room = opt.max_ctx - rt->n_past();
        if ((int)chunk.size() > room) chunk.resize(std::max(room, 1));
        auto tv = Clock::now();
        const std::vector<int32_t> next = rt->verify(chunk);
        const double vms = std::chrono::duration<double, std::milli>(Clock::now() - tv).count();
        int k = 0;
        while (k + 1 < (int)chunk.size() && chunk[k + 1] == next[k]) k++;
        {
          const double nv = chunk.size() - 1;
          obs = {0.97 * obs.w + 1, 0.97 * obs.n + nv, 0.97 * obs.t + vms, 0.97 * obs.nn + nv * nv, 0.97 * obs.nt + nv * vms};
          refit();
        }
        n_hist[chunk.size() - 1]++;
        if (profile) {
          fprintf(stderr, "\n# verify %d tokens: %.1f ms (cpu experts %.1f; experts gpu %d cpu %d); GPU timeline:", (int)chunk.size(),
                  rt->last.total_ms, rt->last.cpu_moe_ms, rt->last.gpu_experts, rt->last.cpu_experts);
          for (auto& [g, ms] : rt->last_groups) fprintf(stderr, " %s %.1f", g.c_str(), ms);
        }
        if (spec_log && !d.conf.empty()) {
          fprintf(spec_log, "%d %d %.2f %.2f", (int)chunk.size() - 1, k, vms, dms);
          for (float c : d.conf) fprintf(spec_log, " %.4f", c);
          fprintf(spec_log, "\n");
        }
        rt->commit(k + 1);
        spec_steps++;
        spec_accepted += k;
        bool go = true;
        for (int i = 0; i <= k && go; i++) go = emit(next[i]);
        if (!go) break;
        anchor = next[k];
      }
    }
    if (spec_log) fclose(spec_log);
    fprintf(stderr, "\n# drafts verified per step:");
    for (size_t n = 0; n < n_hist.size(); n++) fprintf(stderr, " %zu: %ld", n, n_hist[n]);
    fprintf(stderr, " (of the 0s, %ld plain steps without a draft); verify cost fit %.1f + %.1f n ms", plain_steps,
            verify_cost[0], verify_cost[1] - verify_cost[0]);
  }
#endif
  for (; !spec && n_gen < n_predict; n_gen++) {
    const int32_t next = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    gen.push_back(next);
    if (next == eos) break;
    std::string p = piece(vocab, next);
    fwrite(p.data(), 1, p.size(), stdout);
    fflush(stdout);
    if (n_gen + 1 == n_predict) { n_gen++; break; }
#ifdef DS_RUNTIME
    if (rt && rt->has_draft()) {
      auto td = Clock::now();
      DraftResult d = rt->draft(next);
      draft_ms += std::chrono::duration<double, std::milli>(Clock::now() - td).count();
      drafts.emplace_back(d.ids.begin() + 1, d.ids.end());
      confs.push_back(d.conf);
    }
    if (rt) {
      logits = rt->decode(next);
      if (profile) {
        fprintf(stderr, "\n# step %d: %.1f ms (cpu experts %.1f, waiting for the GPU %.1f; experts gpu %d cpu %d); GPU timeline:",
                n_gen, rt->last.total_ms, rt->last.cpu_moe_ms, rt->last.wait_ms, rt->last.gpu_experts, rt->last.cpu_experts);
        for (auto& [g, ms] : rt->last_groups) fprintf(stderr, " %s %.1f", g.c_str(), ms);
      }
      continue;
    }
#endif
    logits = eng->forward({next});
  }
  const double t_tg = std::chrono::duration<double>(Clock::now() - t2).count();
  if (spec_steps)
    fprintf(stderr, "\n# speculative: %ld verify steps, %.2f drafts accepted per step, %.2f tokens per step; draft %.1f ms per step\n",
            spec_steps, (double)spec_accepted / spec_steps, (double)n_gen / spec_steps, draft_ms / spec_steps);
  printf("\n");
  fflush(stdout);
#ifdef DS_RUNTIME
  if (!drafts.empty()) {
    // draft g (anchor = gen[g]) guesses gen[g + 1 ..]; its accepted length = the matching prefix
    const int B = (int)drafts[0].size();
    std::vector<int> hist(B + 1, 0);
    long n = 0, acc_sum = 0;
    std::vector<long> pos_ok(B, 0), pos_n(B, 0);
    for (size_t g = 0; g < drafts.size(); g++) {
      if (g + B >= gen.size()) break;  // only drafts whose whole block can be checked
      int a = 0;
      while (a < B && drafts[g][a] == gen[g + 1 + a]) a++;
      hist[a]++;
      acc_sum += a;
      n++;
      for (int k = 0; k < B; k++) { pos_n[k]++; pos_ok[k] += drafts[g][k] == gen[g + 1 + k]; }
    }
    if (n > 0) {
      fprintf(stderr, "# dspark shadow: %ld drafts of %d; accepted prefix mean %.2f -> %.2f tokens per main step; %.1f ms per draft\n",
              n, B, (double)acc_sum / n, 1.0 + (double)acc_sum / n, draft_ms / drafts.size());
      fprintf(stderr, "#   accepted 0..%d:", B);
      for (int a = 0; a <= B; a++) fprintf(stderr, " %d", hist[a]);
      fprintf(stderr, "\n#   per position correct:");
      for (int k = 0; k < B; k++) fprintf(stderr, " %.0f%%", 100.0 * pos_ok[k] / pos_n[k]);
      fprintf(stderr, "\n");
    }
  }
  if (rt && rt->total_gpu_experts + rt->total_cpu_experts > 0)
    fprintf(stderr, "# routed experts: %.1f%% on GPUs (%ld of %ld)\n",
            100.0 * rt->total_gpu_experts / (rt->total_gpu_experts + rt->total_cpu_experts), rt->total_gpu_experts,
            rt->total_gpu_experts + rt->total_cpu_experts);
#endif
  fprintf(stderr, "# prompt: %zu tokens in %.2f s = %.2f t/s; generation: %d tokens in %.2f s = %.2f t/s\n",
          toks.size(), t_pp, toks.size() / t_pp, n_gen, t_tg, n_gen > 0 ? n_gen / t_tg : 0.0);
  llama_model_free(tok_model);
  llama_backend_free();
  return 0;
}
