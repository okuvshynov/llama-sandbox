// test-fakequant: our fake-quant ops vs ref/kernel_cpu.py on the inputs of ref/make_fq_test.py.
// Must be bit-identical.
#include "fakequant.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> load(const char* p) {
  FILE* f = fopen(p, "rb");
  if (!f) { perror(p); exit(1); }
  std::vector<float> v;
  float x;
  while (fread(&x, 4, 1, f) == 1) v.push_back(x);
  fclose(f);
  return v;
}

int main() {
  std::vector<float> in = load("data/fq/in.f32");
  const int64_t n = (int64_t)in.size();
  const char* names[] = {"fp8", "fp4", "fp4e4"};
  int bad_total = 0;
  for (int k = 0; k < 3; k++) {
    std::vector<float> want = load((std::string("data/fq/") + names[k] + ".f32").c_str());
    ggml_init_params p = {64 * 1024 * 1024, nullptr, false};
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, n / 64);
    memcpy(x->data, in.data(), n * 4);
    ggml_tensor* y = k == 0 ? fq_fp8(ctx, x) : k == 1 ? fq_fp4(ctx, x) : fq_fp4_e4(ctx, x);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    int bad = 0;
    for (int64_t i = 0; i < n; i++) {
      const float a = ((float*)y->data)[i], b = want[i];
      if (memcmp(&a, &b, 4) != 0 && !(a == 0 && b == 0)) {
        if (bad < 5) printf("  %s[%ld]: in %.9g ours %.9g ref %.9g\n", names[k], (long)i, in[i], a, b);
        bad++;
      }
    }
    printf("%-6s %ld values, %d differ\n", names[k], (long)n, bad);
    bad_total += bad;
    ggml_free(ctx);
  }
  printf(bad_total ? "FAIL\n" : "ALL BIT-EXACT\n");
  return bad_total ? 1 : 0;
}
