// test-matmul-prec: precision of the GPU's bf16-weight x f32-activation matmul by batch size,
// against a double-precision host result (which kernel ggml picks depends on the batch).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
  ggml_backend_dev_t dev = nullptr;
  for (size_t i = 0; i < ggml_backend_dev_count(); i++)
    if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) { dev = ggml_backend_dev_get(i); break; }
  ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
  const int K = 5120, M = 1280;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd;
  std::vector<ggml_bf16_t> w(K * M);
  std::vector<float> wf(K * M);
  for (int i = 0; i < K * M; i++) { w[i] = ggml_fp32_to_bf16(nd(rng) * 0.02f); wf[i] = ggml_bf16_to_fp32(w[i]); }
  for (int T : {1, 2, 4, 8, 9, 10, 12, 16, 17, 32, 64, 128, 200}) {
    std::vector<float> x(K * T);
    for (auto& v : x) v = nd(rng);
    ggml_init_params p = {ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(p);
    ggml_tensor* W = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, K, M);
    ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor* Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> y(M * T);
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    double num = 0, den = 0, numb = 0;
    for (int t = 0; t < T; t++)
      for (int m = 0; m < M; m++) {
        double r = 0, rb = 0;
        for (int k = 0; k < K; k++) {
          r += (double)wf[m * K + k] * x[t * K + k];
          rb += (double)wf[m * K + k] * ggml_bf16_to_fp32(ggml_fp32_to_bf16(x[t * K + k]));
        }
        num += (y[t * M + m] - r) * (y[t * M + m] - r);
        numb += (y[t * M + m] - rb) * (y[t * M + m] - rb);
        den += r * r;
      }
    printf("T=%3d  relerr vs exact %.2e   vs bf16-rounded activations %.2e\n", T, sqrt(num / den), sqrt(numb / den));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
  }
  return 0;
}
