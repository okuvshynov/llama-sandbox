// mm-bench: one MLP matmul (y = W x) through llama.cpp's HIP backend.
//
//   mm-bench <layer.gguf> [--dev D] [--tensor gate|up|down|all] [--ns 1,2,4,...]
//            [--stream R] [--min-time S] [--no-check]
//
// layer.gguf comes from mlp-extract. W keeps its quant type, x is F32 [K, N].
// For every N it reports:
//   single  wall time of one ggml_backend_graph_compute holding ONE matmul
//           (launch + kernel + sync: what a decode step pays per matmul)
//   stream  wall time per matmul when R matmuls sit in one graph
//           (launch/sync amortized: close to kernel time)
//   GB/s    weight bytes / stream time (N <= 8 should be bandwidth bound)
//   TFLOPS  2*K*M*N / stream time
//   relerr  ||y - ref|| / ||ref|| over ALL N columns, where ref is computed on
//           the host from W dequantized by ggml's own to_float, accumulated in
//           double. The GPU quantizes x to q8_1, so ~1e-3..1e-2 is expected.
// A negative control (gate output scored against up's reference) must fail,
// otherwise the checker is broken.

#include "common.h"

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  ggml_log_set(log_filter, nullptr);
  if (argc < 2) {
    fprintf(stderr, "usage: mm-bench <layer.gguf> [--dev D] [--tensor gate|up|down|all] [--ns 1,2,..]"
                    " [--stream R] [--min-time S] [--no-check]\n");
    return 2;
  }
  const char* path = argv[1];
  int dev_idx = 0, R = 16;
  double min_time = 0.5;
  bool check = true;
  std::string which = "all";
  std::vector<int64_t> ns = {1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512};
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&] { if (i + 1 >= argc) die("missing value for " + a); return argv[++i]; };
    if (a == "--dev") dev_idx = atoi(next());
    else if (a == "--tensor") which = next();
    else if (a == "--ns") ns = parse_list(next());
    else if (a == "--stream") R = atoi(next());
    else if (a == "--min-time") min_time = atof(next());
    else if (a == "--no-check") check = false;
    else die("unknown option " + a);
  }

  print_clocks();
  Layer L = open_layer(path);
  int64_t sid = gguf_find_key(L.g, "mlp.source");
  printf("# file %s  (%s, layer %u)\n", path, sid >= 0 ? gguf_get_val_str(L.g, sid) : "?",
         gguf_get_val_u32(L.g, gguf_find_key(L.g, "mlp.layer")));

  ggml_backend_dev_t dev = gpu_dev(dev_idx);
  ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
  printf("# device %s (%s), stream R=%d, min-time %.2fs per point\n", ggml_backend_dev_name(dev),
         ggml_backend_dev_description(dev), R, min_time);

  std::vector<std::string> parts;
  if (which == "all") parts = {"ffn_gate", "ffn_up", "ffn_down"};
  else parts = {"ffn_" + which};

  const int64_t NMAX = *std::max_element(ns.begin(), ns.end());
  std::vector<float> ref_up_for_control;  // negative control

  for (const auto& part : parts) {
    const ggml_tensor* src = L.find(part);
    const int64_t K = src->ne[0], M = src->ne[1];
    const double wbytes = (double)ggml_nbytes(src);
    printf("\n== %s  %s  K=%" PRId64 " -> M=%" PRId64 "  %.2f MiB\n", src->name, ggml_type_name(src->type), K, M,
           wbytes / 1048576.0);

    auto bytes = L.bytes(src);
    DevWeight dw = upload(be, src, bytes);

    std::vector<float> x = make_x(K, NMAX);

    std::vector<float> ref;
    if (check) {
      auto t0 = Clock::now();
      printf("  reference: dequant + host matmul for N=%" PRId64 " ... ", NMAX);
      fflush(stdout);
      auto w = dequant(src, bytes);
      ref = reference(w, x, K, M, NMAX);
      printf("%.1fs\n", std::chrono::duration<double>(Clock::now() - t0).count());
    }

    printf("  %5s %10s %18s %10s %8s %7s %10s %9s %5s\n", "N", "single_us", "(p10-p90)", "stream_us", "GB/s",
           "TFLOPS", "relerr", "maxerr", "ok");
    for (int64_t N : ns) {
      MatGraph g1, gR;
      g1.build(be, dw.w, N, 1, x.data());
      gR.build(be, dw.w, N, R, x.data());
      Timing ts = time_graph(be, g1.gf, min_time);
      Timing tr = time_graph(be, gR.gf, min_time);
      double per = tr.med / R;
      double gbs = wbytes / (per * 1e-6) / 1e9;
      double tflops = 2.0 * K * M * N / (per * 1e-6) / 1e12;

      std::string ok = "-";
      double re = NAN, mx = NAN;
      if (check) {
        std::vector<float> y(N * M), yR(N * M);
        ggml_backend_tensor_get(g1.ys[0], y.data(), 0, y.size() * 4);
        ggml_backend_tensor_get(gR.ys[R - 1], yR.data(), 0, yR.size() * 4);
        re = rel_err(y.data(), ref.data(), y.size(), &mx);  // ref's first N columns
        bool same = memcmp(y.data(), yR.data(), y.size() * 4) == 0;
        ok = (re < 2e-2 ? std::string("yes") : std::string("NO")) + (same ? "" : " (single!=stream)");
        if (part == "ffn_gate" && N == ns[0]) ref_up_for_control = y;  // y of gate, scored later
      }
      printf("  %5" PRId64 " %10.1f %8.1f-%-9.1f %10.1f %8.1f %7.2f %10.2e %9.2e %5s\n", N, ts.med, ts.p10, ts.p90,
             per, gbs, tflops, re, mx, ok.c_str());
      g1.free_all();
      gR.free_all();
    }
    if (check && part == "ffn_up" && !ref_up_for_control.empty()) {
      double c = rel_err(ref_up_for_control.data(), ref.data(), M * ns[0], nullptr);
      printf("  negative control: gate output vs up reference relerr %.2e -> %s\n", c,
             c > 0.5 ? "fails as it must" : "PASSES - checker is broken");
    }
    ggml_backend_buffer_free(dw.buf);
    ggml_free(dw.ctx);
  }
  ggml_backend_free(be);
  return 0;
}
