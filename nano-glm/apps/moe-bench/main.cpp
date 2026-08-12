// moe-bench — the backend's expert kernel, without the model.
//
// `moe-server` needs 86 s to load 150 GiB and another 77 s to fill four dies
// before it can be timed at all, which makes kernel tuning a ~3 minute edit
// cycle. Nothing about the expert dispatch depends on the *values* in the
// weights, only on their shapes, their quantization and how many (token,
// expert) pairs a device is handed. So this allocates weights of exactly the
// right shape and type, fills them with synthetic bytes, and runs the same
// graph `run_device_compact` runs — in seconds, from nothing.
//
// The graph is copied deliberately rather than shared, and that is the one
// thing to watch: if `run_device_compact` changes, this must change with it or
// it silently measures something else. It is five ops, reproduced here in the
// same order:
//
//     up   = mul_mat_id(w_up,   x, ids)
//     gate = mul_mat_id(w_gate, x, ids)
//     act  = swiglu_split(gate, up)
//     res  = mul_mat_id(w_down, act, ids)
//     res  = mul(res, weights)
//
// **One device.** The server dispatches to its dies from one thread each and
// waits for all of them, so a layer's wall time is the slowest die, not the
// sum. Timing one device with the pair count that device actually receives is
// therefore the right model, and `--pairs` is that count: at decode, 6 slots
// over 4 dies is 1 or 2; at a 111-token prefill, 666 pairs over 4 dies is ~166.
//
// What it cannot tell you: anything about routing, compaction, the scatter-add
// combine, or the socket. Those are host-side and measured where they happen —
// `--moe-log` reports the server's own route/compute/serialize split per call.
//
// models/deepseek4/graph.h first, for the winsock2-before-windows.h ordering
// that gguf_store.h needs.
#include "models/deepseek4/graph.h"

#include "cpu_topology.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

struct bench_params {
    int64_t n_embd    = 4096;   // DeepSeek-V4-Flash
    int64_t n_ff      = 2048;
    int64_t n_expert  = 60;     // resident per die at --gpu-experts 240 over 4
    int64_t n_pairs   = 2;      // (token, expert) pairs this device receives
    int     device    = 0;
    int     iters     = 200;
    int     n_layer   = 40;     // for the per-token extrapolation only
    bool    list      = false;
};

static void usage() {
    fprintf(stderr,
        "moe-bench — time the backend's expert graph with synthetic weights\n\n"
        "  --list             enumerate devices and exit\n"
        "  --device N         which GPU (default 0)\n"
        "  --experts N        resident experts on this device (default 60)\n"
        "  --pairs N          (token,expert) pairs handed to it (default 2)\n"
        "  --embd N           model n_embd (default 4096)\n"
        "  --ffn N            expert n_ff (default 2048)\n"
        "  --layers N         routed layers, for the per-token figure (default 40)\n"
        "  --iters N          timed iterations (default 200)\n\n"
        "decode is --pairs 1 or 2; a 111-token prefill over four dies is ~166.\n");
}

// MXFP4 is a byte of E8M0 exponent then sixteen packed nibbles per 32 values.
// The exponent is fixed at 127 (2^0) rather than randomized: a random E8M0 byte
// can be 0 or 255, which are 2^-127 and NaN, and denormal or NaN inputs change
// what the hardware does. The nibbles are random because they do not.
static void fill_mxfp4(std::vector<uint8_t> & buf, size_t bytes, std::mt19937 & rng) {
    buf.resize(bytes);
    for (size_t i = 0; i < bytes; i += 17) {
        buf[i] = 127;
        const size_t n = std::min<size_t>(16, bytes - i - 1);
        for (size_t j = 0; j < n; j++) buf[i + 1 + j] = (uint8_t) (rng() & 0xFF);
    }
}

static double p50(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

int main(int argc, char ** argv) {
    bench_params p;
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "--list"))                      p.list = true;
        else if (!strcmp(a, "--device")  && i + 1 < argc)   p.device   = atoi(argv[++i]);
        else if (!strcmp(a, "--experts") && i + 1 < argc)   p.n_expert = atoll(argv[++i]);
        else if (!strcmp(a, "--pairs")   && i + 1 < argc)   p.n_pairs  = atoll(argv[++i]);
        else if (!strcmp(a, "--embd")    && i + 1 < argc)   p.n_embd   = atoll(argv[++i]);
        else if (!strcmp(a, "--ffn")     && i + 1 < argc)   p.n_ff     = atoll(argv[++i]);
        else if (!strcmp(a, "--layers")  && i + 1 < argc)   p.n_layer  = atoi(argv[++i]);
        else if (!strcmp(a, "--iters")   && i + 1 < argc)   p.iters    = atoi(argv[++i]);
        else { usage(); return 1; }
    }

    ggml_backend_load_all();

    std::vector<ggml_backend_dev_t> gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) gpus.push_back(d);
    }
    if (p.list || gpus.empty()) {
        fprintf(stderr, "moe-bench: %zu GPU device(s)\n", gpus.size());
        for (size_t i = 0; i < gpus.size(); i++) {
            size_t free_b = 0, total_b = 0;
            ggml_backend_dev_memory(gpus[i], &free_b, &total_b);
            fprintf(stderr, "  %zu: %s (%.2f GiB free / %.2f GiB)\n", i,
                    ggml_backend_dev_description(gpus[i]),
                    free_b / 1073741824.0, total_b / 1073741824.0);
        }
        if (gpus.empty()) fprintf(stderr, "moe-bench: build with build.ps1 -Vk\n");
        return gpus.empty() ? 1 : 0;
    }
    if (p.device < 0 || p.device >= (int) gpus.size()) {
        fprintf(stderr, "moe-bench: no device %d (have %zu)\n", p.device, gpus.size());
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(gpus[p.device], nullptr);
    if (!backend) { fprintf(stderr, "moe-bench: cannot init device\n"); return 1; }
    fprintf(stderr, "moe-bench: %s\n", ggml_backend_dev_description(gpus[p.device]));

    // Weights, shaped exactly as `device_load_experts` shapes a device's slice.
    ggml_init_params wip = { ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/ true };
    ggml_context * wctx = ggml_init(wip);
    ggml_tensor * w_up   = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, p.n_embd, p.n_ff,   p.n_expert);
    ggml_tensor * w_gate = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, p.n_embd, p.n_ff,   p.n_expert);
    ggml_tensor * w_down = ggml_new_tensor_3d(wctx, GGML_TYPE_MXFP4, p.n_ff,   p.n_embd, p.n_expert);

    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) {
        fprintf(stderr, "moe-bench: cannot allocate %.2f GiB of experts — lower --experts\n",
                3.0 * ggml_nbytes(w_up) / 1073741824.0);
        return 1;
    }
    const double w_gib = ggml_backend_buffer_get_size(wbuf) / 1073741824.0;

    {
        std::mt19937 rng(1234);
        std::vector<uint8_t> staging;
        for (ggml_tensor * t : { w_up, w_gate, w_down }) {
            fill_mxfp4(staging, ggml_nbytes(t), rng);
            ggml_backend_tensor_set(t, staging.data(), 0, staging.size());
        }
    }

    // The graph, op for op as run_device_compact builds it.
    std::vector<uint8_t> meta(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    std::vector<float>   host_x((size_t) p.n_embd * p.n_pairs);
    std::vector<int32_t> host_ids((size_t) p.n_pairs);
    std::vector<float>   host_w((size_t) p.n_pairs);
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto & v : host_x) v = u(rng);
        for (auto & v : host_w) v = 0.125f;
        // Distinct experts per pair where possible: two pairs landing on the
        // same expert would read one set of weights twice and read warm.
        for (int64_t j = 0; j < p.n_pairs; j++) host_ids[j] = (int32_t) (j % p.n_expert);
    }

    std::vector<double> t_total;
    for (int it = 0; it < p.iters + 5; it++) {          // five discarded
        const auto t0 = clk::now();

        ggml_init_params ip = { meta.size(), meta.data(), /*no_alloc=*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, p.n_embd, 1, p.n_pairs);
        ggml_set_input(x);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, p.n_pairs);
        ggml_set_input(ids);
        ggml_tensor * wts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, p.n_pairs);
        ggml_set_input(wts);

        ggml_tensor * up   = ggml_mul_mat_id(ctx, w_up,   x,   ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, x,   ids);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * res  = ggml_mul_mat_id(ctx, w_down, act, ids);
        res = ggml_mul(ctx, res, wts);
        ggml_set_output(res);
        ggml_build_forward_expand(gf, res);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "moe-bench: alloc failed\n"); return 1; }
        ggml_backend_tensor_set(x,   host_x.data(),   0, host_x.size()   * sizeof(float));
        ggml_backend_tensor_set(ids, host_ids.data(), 0, host_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(wts, host_w.data(),   0, host_w.size()   * sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "moe-bench: compute failed\n");
            return 1;
        }
        std::vector<float> out((size_t) p.n_embd * p.n_pairs);
        ggml_backend_tensor_get(res, out.data(), 0, out.size() * sizeof(float));

        ggml_free(ctx);
        const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
        if (it >= 5) t_total.push_back(us);

        if (it == 5) {   // once: a value check, so a kernel that returns zeros is visible
            size_t bad = 0;
            for (float v : out) if (!(v == v) || v > 3.0e38f || v < -3.0e38f) bad++;
            double sum = 0.0;
            for (float v : out) sum += v < 0 ? -v : v;
            fprintf(stderr, "moe-bench: output mean|.| %.4e, %zu non-finite\n",
                    sum / (double) out.size(), bad);
        }
    }

    // Bytes a pair reads: one expert's three matrices at 17 bytes per 32 values.
    const double bytes_pair = (double) (p.n_embd * p.n_ff * 2 + p.n_ff * p.n_embd) * 17.0 / 32.0;
    const double med = p50(t_total);

    fprintf(stderr,
        "\nmoe-bench: %lld experts resident (%.2f GiB), %lld pairs, n_embd %lld, n_ff %lld\n",
        (long long) p.n_expert, w_gib, (long long) p.n_pairs,
        (long long) p.n_embd, (long long) p.n_ff);
    fprintf(stderr, "moe-bench: per dispatch  p50 %8.1f us   min %8.1f us   max %8.1f us\n",
            med, *std::min_element(t_total.begin(), t_total.end()),
            *std::max_element(t_total.begin(), t_total.end()));
    fprintf(stderr, "moe-bench: reads %.2f MB -> %.1f GB/s\n",
            bytes_pair * (double) p.n_pairs / 1e6,
            bytes_pair * (double) p.n_pairs / (med * 1e-6) / 1e9);
    fprintf(stderr, "moe-bench: x %d layers = %.2f ms per token\n",
            p.n_layer, med * p.n_layer / 1000.0);

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
