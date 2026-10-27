// llama-logits: the llama.cpp fork's logits for a fixed token sequence (the "second opinion").
//
//   llama-logits -m <model.gguf> --tokens 0,128803,... --prefill P [-t 16] [--out logits.f32]
//
// Mirrors ds_ref.py's steps exactly: one llama_decode over tokens[0:P] (prefill), then one call
// per remaining token, recording the logits of each call's last position. The tokens are given
// (typically ds_ref.py's own sequence), so the two implementations are compared on identical
// inputs even where their greedy choices would differ. --out writes [steps][n_vocab] float32.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string model_path, out_path;
  std::vector<llama_token> toks;
  int prefill = -1, threads = 16;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "-m") model_path = next();
    else if (a == "--tokens") { for (char* p = (char*)next(); *p;) { toks.push_back((llama_token)strtol(p, &p, 10)); if (*p == ',') p++; } }
    else if (a == "--prefill") prefill = atoi(next());
    else if (a == "-t") threads = atoi(next());
    else if (a == "--out") out_path = next();
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  if (model_path.empty() || toks.empty() || prefill <= 0 || prefill > (int)toks.size()) {
    fprintf(stderr, "usage: llama-logits -m <model.gguf> --tokens a,b,c --prefill P [-t N] [--out file]\n");
    return 2;
  }

  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) { fprintf(stderr, "cannot load %s\n", model_path.c_str()); return 1; }
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 1024;
  while ((int)cp.n_ctx < (int)toks.size() + 16) cp.n_ctx *= 2;
  cp.n_batch = cp.n_ubatch = cp.n_ctx;
  cp.n_threads = cp.n_threads_batch = threads;
  llama_context* ctx = llama_init_from_model(model, cp);
  if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  FILE* out = out_path.empty() ? nullptr : fopen(out_path.c_str(), "wb");
  int pos = 0, step = 0;
  while (pos < (int)toks.size()) {
    const int n = step == 0 ? prefill : 1;
    llama_batch b = llama_batch_get_one(toks.data() + pos, n);
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", pos); return 1; }
    const float* lg = llama_get_logits_ith(ctx, -1);
    int best[5] = {0, 0, 0, 0, 0};
    for (int k = 0; k < 5; k++) {
      int arg = -1;
      for (int v = 0; v < n_vocab; v++) {
        bool taken = false;
        for (int j = 0; j < k; j++) taken |= best[j] == v;
        if (!taken && (arg < 0 || lg[v] > lg[arg])) arg = v;
      }
      best[k] = arg;
    }
    printf("step %d: pos %d..%d -> %d  top5 %d %d %d %d %d  (logit %.3f)\n", step, pos, pos + n - 1, best[0], best[0],
           best[1], best[2], best[3], best[4], lg[best[0]]);
    if (out) fwrite(lg, sizeof(float), n_vocab, out);
    pos += n;
    step++;
  }
  if (out) fclose(out);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
