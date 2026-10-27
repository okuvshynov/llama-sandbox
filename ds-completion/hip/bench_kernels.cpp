// bench-kernels: each runtime kernel (kernels.h) alone at its decode-time size, back to back on
// one stream (events around N reps), to see which are slow before they meet each other in a
// timeline. Plus the floors: an empty kernel, and a 20 KB device->pinned-host copy.
#include "fp8_gemv.h"
#include "kernels.h"

#include <cstdio>
#include <functional>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(1); } } while (0)

__global__ void empty_kernel() {}

template <class T>
static T* dmalloc(size_t n) {
  void* p;
  CHECK(hipMalloc(&p, n * sizeof(T)));
  CHECK(hipMemset(p, 0, n * sizeof(T)));
  return (T*)p;
}

int main() {
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  auto time = [&](const char* name, int reps, const std::function<void()>& f) {
    for (int i = 0; i < 5; i++) f();
    CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < reps; i++) f();
    CHECK(hipEventRecord(e1, s));
    CHECK(hipEventSynchronize(e1));
    float ms;
    CHECK(hipEventElapsedTime(&ms, e0, e1));
    printf("%-44s %8.1f us\n", name, ms * 1000 / reps);
  };
  const int E = 5120, HD = 512, NH = 64, W = 128, K = 512;
  float* h = dmalloc<float>(4 * E);
  float* h2 = dmalloc<float>(4 * E);
  float* x = dmalloc<float>(32768);
  float* y = dmalloc<float>(32768);
  float* wf = dmalloc<float>((size_t)384 * E);
  float* hcfn = dmalloc<float>((size_t)24 * 4 * E);
  float* part = dmalloc<float>((size_t)kSplitMax * 32768);
  float* small = dmalloc<float>(1024);
  float* ring = dmalloc<float>((size_t)W * HD);
  float* comp = dmalloc<float>((size_t)4096 * HD);
  int* topk = dmalloc<int>(K);
  float* sinks = dmalloc<float>(NH);
  float* scratch = dmalloc<float>((size_t)20 * NH * (HD + 2));
  float* freqs = dmalloc<float>(32);
  uint8_t* q8 = dmalloc<uint8_t>(32768);
  float* s8 = dmalloc<float>(1024);
  int* ids = dmalloc<int>(8);
  float* hostbuf;
  CHECK(hipHostMalloc((void**)&hostbuf, 20 * 1024 + 64, 0));
  std::vector<int> tk(K);
  for (int i = 0; i < K; i++) tk[i] = i * 3;
  CHECK(hipMemcpy(topk, tk.data(), K * 4, hipMemcpyHostToDevice));

  const int R = 200;
  time("empty kernel", R, [&] { empty_kernel<<<1, 64, 0, s>>>(); });
  time("D2H 20 KB to pinned", R, [&] { CHECK(hipMemcpyAsync(hostbuf, x, 20 * 1024 + 64, hipMemcpyDeviceToHost, s)); });
  time("H2D 20 KB from pinned", R, [&] { CHECK(hipMemcpyAsync(x, hostbuf, 20 * 1024, hipMemcpyHostToDevice, s)); });
  time("D2D 2 KB", R, [&] { CHECK(hipMemcpyAsync(y, x, 2048, hipMemcpyDeviceToDevice, s)); });
  time("fp8_quant_act 5120", R, [&] { fp8_quant_act(x, q8, s8, 1, 5120, s); });
  time("fp8_quant_act 32768", R, [&] { fp8_quant_act(x, q8, s8, 1, 32768, s); });
  time("rms_norm 1280", R, [&] { k_rms_norm(x, small, y, 1280, 1, 1e-20f, s); });
  time("rope 64 heads", R, [&] { k_rope(y, 64, 512, 64, freqs, 100, false, s); });
  time("kv_finish 512", R, [&] { k_kv_finish(x, small, 512, 64, freqs, 100, 1e-20f, ring, s); });
  time("gemv_f32 hc_fn 24 x 20480", R, [&] { k_gemv_f32(hcfn, 24, 4 * E, h, small, part, s); });
  time("hc_split", R, [&] { k_hc_split(h, 4 * E, small, small, small, 1e-20f, 1e-6f, 20, small, small + 4, small + 8, s); });
  time("hc_pre_norm 5120", R, [&] { k_hc_pre_norm(h, small, small, E, 1e-20f, x, s); });
  time("hc_post 5120", R, [&] { k_hc_post(x, h, small, small, E, h2, s); });
  time("gemv_f32 router 384 x 5120", R, [&] { k_gemv_f32(wf, 384, E, x, small, part, s); });
  time("router top-6 of 384", R, [&] { k_router(small, small, 384, 6, true, 1.5f, ids, small + 512, s); });
  time("swiglu 2304", R, [&] { k_swiglu(x, y, 10.0f, 2304, h2, s); });
  time("attn_decode 64 heads, 128 + 0 keys", R, [&] { k_attn_decode(y, NH, HD, ring, W, 200, comp, topk, 0, sinks, 0.044f, h2, scratch, s); });
  time("attn_decode 64 heads, 128 + 512 keys", R, [&] { k_attn_decode(y, NH, HD, ring, W, 2000, comp, topk, K, sinks, 0.044f, h2, scratch, s); });
  time("index_scores 32 x 128, 1100 keys", R, [&] { k_index_scores(y, 32, 128, comp, 1100, small, 1.0f, part, s); });
  time("topk 512 of 1100", R, [&] { k_topk(part, 1100, 512, topk, s); });

  // launch overhead: a chain of dependent launches on a stream vs the same chain captured once
  // as a HIP graph and replayed. Per launch, after subtracting nothing: the gap is what matters.
  auto chain_vs_graph = [&](const char* name, int n, const std::function<void()>& one) {
    auto chain = [&] { for (int i = 0; i < n; i++) one(); };
    hipGraph_t g;
    hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    chain();
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    float ms_s, ms_g;
    for (int pass = 0; pass < 2; pass++) {
      for (int i = 0; i < 3; i++) pass ? (void)hipGraphLaunch(ge, s) : chain();
      CHECK(hipEventRecord(e0, s));
      for (int r = 0; r < 20; r++) { if (pass) CHECK(hipGraphLaunch(ge, s)); else chain(); }
      CHECK(hipEventRecord(e1, s));
      CHECK(hipEventSynchronize(e1));
      CHECK(hipEventElapsedTime(pass ? &ms_g : &ms_s, e0, e1));
    }
    printf("%-44s stream %6.2f us/launch, graph %6.2f us/launch (%d launches)\n", name, ms_s * 1000 / 20 / n,
           ms_g * 1000 / 20 / n, n);
    CHECK(hipGraphExecDestroy(ge));
    CHECK(hipGraphDestroy(g));
  };
  chain_vs_graph("chain: empty kernel", 2000, [&] { empty_kernel<<<1, 64, 0, s>>>(); });
  chain_vs_graph("chain: hc_pre_norm 5120", 500, [&] { k_hc_pre_norm(h, small, small, E, 1e-20f, x, s); });
  chain_vs_graph("chain: hc split+pre_norm+post (3 kernels)", 300, [&] {
    k_hc_split(h, 4 * E, small, small, small, 1e-20f, 1e-6f, 20, small, small + 4, small + 8, s);
    k_hc_pre_norm(h, small, small, E, 1e-20f, x, s);
    k_hc_post(x, h, small, small, E, h2, s);
  });
  chain_vs_graph("chain: fp8_quant_act 5120 + rms_norm 1280", 500, [&] {
    fp8_quant_act(x, q8, s8, 1, 5120, s);
    k_rms_norm(x, small, y, 1280, 1, 1e-20f, s);
  });
  return 0;
}
