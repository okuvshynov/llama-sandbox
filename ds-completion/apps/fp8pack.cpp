// fp8pack: write the dense linears that were fp8 in the checkpoint back as fp8 (PLAN.md B0).
//
//   fp8pack -m <model-00001-of-000NN.gguf> -o <out.gguf> [--layers 0-39]
//
// The GGUF stores these weights as bf16, an exact expansion of the checkpoint's e4m3 values with
// one power-of-two (e8m0) scale per 32x32 block. This re-blocks them: per 32x32 block,
//   s = 2^ceil(log2(amax / 448)),  code = e4m3(w / s)
// and checks that code * s reproduces every bf16 value exactly. (s may be smaller than the
// checkpoint's own scale; the values are the same either way, see ds_ref.py fp8_blocks.)
//
// Output: a GGUF with, for each packed tensor <name>,
//   <name>        I8  [K, N]        e4m3 codes, row-major like the source ([N rows of K])
//   <name>.scale  I8  [K/32, N/32]  e8m0 exponents (value = 2^(e - 127))
// ~1.001 bytes per weight instead of 2. Our loader maps it like the model shards.

#include "fakequant.h"
#include "gguf.h"
#include "model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <omp.h>

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
  std::string in, out;
  std::vector<int> layers;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) in = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = parse_list(argv[++i]);
    else { fprintf(stderr, "usage: fp8pack -m <model.gguf> -o <out.gguf> [--layers 0-39]\n"); return 2; }
  }
  if (in.empty() || out.empty()) { fprintf(stderr, "usage: fp8pack -m <model.gguf> -o <out.gguf>\n"); return 2; }
  if (!fp8_codec_selftest()) { fprintf(stderr, "e4m3 codec does not match the fake-quant rounding\n"); return 1; }

  Model m;
  m.load(in, layers);
  const Hparams& hp = m.hp;
  // the linears stored as fp8 in the checkpoint (README: "What the model is")
  const char* packed[] = {"attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b", "indexer.attn_q_b",
                          "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "engram_kv"};
  gguf_context* o = gguf_init_empty();
  gguf_set_val_str(o, "general.architecture", "ds-fp8pack");
  gguf_set_val_str(o, "fp8pack.source", in.c_str());
  gguf_set_val_u32(o, "fp8pack.block", 32);
  std::vector<ggml_context*> ctxs;
  size_t n_bytes_in = 0, n_bytes_out = 0, n_tensors = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int L = 0; L < hp.n_layer; L++) {
    if (!m.layers[L].loaded) continue;
    for (const char* p : packed) {
      const std::string name = "blk." + std::to_string(L) + "." + p + ".weight";
      ggml_tensor* t = m.get(name);
      if (!t) continue;  // e.g. indexer.attn_q_b exists only on index-source layers
      if (t->type != GGML_TYPE_BF16 || ggml_n_dims(t) != 2) { fprintf(stderr, "%s: unexpected type/shape\n", name.c_str()); return 1; }
      const int64_t K = t->ne[0], N = t->ne[1];
      if (K % 32 || N % 32) { fprintf(stderr, "%s: not 32x32-blockable\n", name.c_str()); return 1; }
      ggml_init_params ip = {(size_t)(K * N + (K / 32) * (N / 32)) + 2 * ggml_tensor_overhead() + 64, nullptr, false};
      ggml_context* ctx = ggml_init(ip);
      ctxs.push_back(ctx);
      ggml_tensor* q = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, N);
      ggml_tensor* s = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K / 32, N / 32);
      ggml_set_name(q, name.c_str());
      ggml_set_name(s, (name + ".scale").c_str());
      const long bad = (long)fp8_repack_bf16((const uint16_t*)t->data, N, K, (uint8_t*)q->data, (uint8_t*)s->data);
      if (bad) { fprintf(stderr, "%s: %ld values not exact in fp8 x 2^k blocks\n", name.c_str(), bad); return 1; }
      gguf_add_tensor(o, q);
      gguf_add_tensor(o, s);
      n_bytes_in += ggml_nbytes(t);
      n_bytes_out += ggml_nbytes(q) + ggml_nbytes(s);
      n_tensors++;
    }
    printf("layer %2d done: %zu tensors so far, %s -> %s\n", L, n_tensors, fmt_bytes(n_bytes_in).c_str(),
           fmt_bytes(n_bytes_out).c_str());
  }
  printf("writing %s ...\n", out.c_str());
  if (!gguf_write_to_file(o, out.c_str(), false)) { fprintf(stderr, "write failed\n"); return 1; }
  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("wrote %zu tensors, every value exact: %s bf16 -> %s fp8 + scales (%.1f s)\n", n_tensors,
         fmt_bytes(n_bytes_in).c_str(), fmt_bytes(n_bytes_out).c_str(), dt);
  gguf_free(o);
  for (auto c : ctxs) ggml_free(c);
  return 0;
}
