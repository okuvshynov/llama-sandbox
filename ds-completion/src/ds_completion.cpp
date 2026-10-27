// ds-completion: DeepSeek-V4.1-Flash text completion.
//
//   ds-completion -m <model-00001-of-000NN.gguf> -p "prompt" [--chat | --think] [-n 128]
//                 [-t 16] [--no-quant] [--max-ctx 4096]
//                 [--rt <dense-fp8.gguf> [--gpu D] [--gpu-moe dev:layers,...] [--profile]]
//
// The prompt is fed as one chunk (prefill), then one token per step, greedy. --chat wraps the
// prompt as one user turn in the model's chat format without thinking, --think with it (the
// strings encoding/encoding.py produces); otherwise the prompt is raw text after BOS.
// Only the tokenizer comes from libllama (vocab-only load).
// Without --rt: everything on the CPU through the ggml Engine (forward.cpp), the prompt as one
// chunk. --rt <dense-fp8.gguf> (bin-hip build): every token through the own runtime (hip/rt.h:
// dense part on GPU D in exact fp8, routed experts on the CPU or, with --gpu-moe, on GPUs), the
// prompt one token at a time.

#include "forward.h"
#include "llama.h"
#ifdef DS_RUNTIME
#include "rt.h"
#endif

#include <algorithm>
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
  int gpu = 0;
  bool profile = false;
  std::string route_log;
  std::map<int, int> gpu_moe;  // layer -> device holding all its experts (runtime only)
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
    else if (a == "--gpu-moe") {  // dev:layers[,dev:layers...], layers like 21-24
      for (char* q = (char*)next(); *q;) {
        const int dev = (int)strtol(q, &q, 10);
        if (*q++ != ':') { fprintf(stderr, "--gpu-moe: expected dev:layers\n"); return 2; }
        int a0 = (int)strtol(q, &q, 10), a1 = a0;
        if (*q == '-') a1 = (int)strtol(q + 1, &q, 10);
        for (int L = a0; L <= a1; L++) gpu_moe[L] = dev;
        if (*q == ',') q++;
      }
    }
    else if (a == "--profile") profile = true;
    else if (a == "--rt") rt_path = next();
    else if (a == "--log-routing") route_log = next();  // runtime: every token's experts per layer
    else if (a == "--no-quant") opt.exact_quant = false;
    else if (a == "--max-ctx") opt.max_ctx = atoi(next());
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (model_path.empty() || prompt.empty()) {
    fprintf(stderr, "usage: ds-completion -m <model.gguf> -p \"prompt\" [--chat|--think] [-n N] [-t N] [--no-quant] [--max-ctx N] [--rt dense-fp8.gguf [--gpu D] [--gpu-moe dev:layers,...] [--profile]]\n");
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

  Model m;
  auto t0 = Clock::now();
  try {
    m.load(model_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "load failed: %s\n", e.what());
    return 1;
  }
  {  // the experts that will run on the CPU: page tables now, not inside decode steps
    std::set<int> on_gpu;
    for (auto [L, d] : gpu_moe) on_gpu.insert(L);
    auto tm = Clock::now();
    const size_t n = m.map_experts(opt.n_threads, on_gpu);
    fprintf(stderr, "# mapped %.0f GB of CPU experts in %.1f s\n", n / 1e9,
            std::chrono::duration<double>(Clock::now() - tm).count());
  }
  std::unique_ptr<Engine> eng;
#ifdef DS_RUNTIME
  std::unique_ptr<Runtime> rt;
  if (!rt_path.empty()) {
    if (!opt.exact_quant) { fprintf(stderr, "--no-quant is CPU only\n"); return 2; }
    rt.reset(new Runtime(m, RtOptions{gpu, opt.n_threads, opt.max_ctx}, rt_path, gpu_moe));
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
  for (; n_gen < n_predict; n_gen++) {
    const int32_t next = (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (next == eos) break;
    std::string p = piece(vocab, next);
    fwrite(p.data(), 1, p.size(), stdout);
    fflush(stdout);
    if (n_gen + 1 == n_predict) { n_gen++; break; }
#ifdef DS_RUNTIME
    if (rt) {
      logits = rt->decode(next);
      if (profile) {
        fprintf(stderr, "\n# step %d: %.1f ms (cpu experts %.1f, waiting for the GPU %.1f); GPU timeline:", n_gen,
                rt->last.total_ms, rt->last.cpu_moe_ms, rt->last.wait_ms);
        for (auto& [g, ms] : rt->last_groups) fprintf(stderr, " %s %.1f", g.c_str(), ms);
      }
      continue;
    }
#endif
    logits = eng->forward({next});
  }
  const double t_tg = std::chrono::duration<double>(Clock::now() - t2).count();
  printf("\n");
  fflush(stdout);
  fprintf(stderr, "# prompt: %zu tokens in %.2f s = %.2f t/s; generation: %d tokens in %.2f s = %.2f t/s\n",
          toks.size(), t_pp, toks.size() / t_pp, n_gen, t_tg, n_gen > 0 ? n_gen / t_tg : 0.0);
  llama_model_free(tok_model);
  llama_backend_free();
  return 0;
}
