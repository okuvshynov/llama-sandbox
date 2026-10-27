// bench-ggml-bf16: ggml's HIP bf16 matmul on the dense shapes, M=1 (the step-4 path), for
// comparison with bench-fp8. 16 matmuls per graph to amortize the launch like a real forward.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <vector>

int main() {
  ggml_backend_dev_t dev = nullptr;
  for (size_t i = 0; i < ggml_backend_dev_count(); i++)
    if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) { dev = ggml_backend_dev_get(i); break; }
  ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
  struct S { const char* name; int K, N; } shapes[] = {
      {"attn_q_a", 5120, 1280}, {"attn_q_b", 1280, 32768}, {"attn_kv", 5120, 512}, {"attn_output_a(x8 groups)", 4096, 1024 * 8},
      {"attn_output_b", 8192, 5120}, {"ffn_gate_shexp", 5120, 2304}, {"ffn_down_shexp", 2304, 5120}, {"engram_kv", 6144, 25600}};
  const int R = 16;
  printf("%-26s %6s %6s %9s %8s\n", "shape", "K", "N", "us/mm", "GB/s");
  for (auto& s : shapes) {
    ggml_init_params p = {ggml_tensor_overhead() * (3 * R + 8) + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, 1);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor*> ws;
    for (int r = 0; r < R; r++) {  // distinct weights so nothing stays in cache
      ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, s.K, s.N);
      ws.push_back(w);
      ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_clear(buf, 0);
    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
    const int reps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++) ggml_backend_graph_compute(be, gf);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps / R;
    printf("%-26s %6d %6d %9.1f %8.1f\n", s.name, s.K, s.N, us, 2.0 * s.K * s.N / (us * 1e-6) / 1e9);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
  }
  return 0;
}
