// moe-ggml-bench — one MoE layer of DeepSeek-V4-Pro through llama.cpp's own
// HIP kernels, single die, batch 1..8.
//
// The graph is a faithful transcription of llm_graph_context::build_moe_ffn
// (llama-graph.cpp) on its deepseek4 path, at V4-Pro-0813 shapes:
//   hidden 7168, expert FFN 3072, 384 routed experts (top-6, sqrt-softplus
//   gating with selection bias, weights normalized then scaled 2.5), one
//   shared expert, SwiGLU clamped at 10 (up symmetric, gate one-sided).
// Weights are synthetic MXFP4 (random nibbles, e8m0 scales 117..123 — the
// probe's magnitude regime); the HIP kernels dequantize on the fly, which is
// the thing being measured. ~13.5 GB of routed experts: fits one die.
//
// What varies: batch size n = 1..8 — the speculative-verification regime.
// Reported per n: graph time, per-token time, and the *exact* bytes of
// expert weights the routed ids touch (read back from the selection tensor,
// deduplicated) so the effective bandwidth is honest about expert-union
// sublinearity.
//
// Sanity (not an exactness gate): the same graph runs once on the CPU
// backend at n=1 and outputs are compared loosely. Both sides quantize
// activations (MMVQ vs CPU repack), so ~1e-3-relative disagreement is two
// correct kernels; the gate is 2e-2 relative. Exactness rests on
// test-backend-ops (12,926/12,926 on this build).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

// --- V4-Pro-0813 MoE shape ---------------------------------------------------
static const int64_t N_EMBD   = 7168;
static const int64_t N_FF     = 3072;
static const int64_t N_EXPERT = 384;
static const int64_t N_USED   = 6;
static const float   CLAMP    = 10.0f;
static const float   W_SCALE  = 2.5f;   // routed_scaling_factor
static const int     N_BATCH_MAX = 8;

struct weights_t {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx2 = nullptr;               // mxfp4 tensors when a
    ggml_backend_buffer_t buf2 = nullptr;        // buft override is in play
    ggml_tensor * gate_inp, * exp_probs_b;
    ggml_tensor * gate_exps, * up_exps, * down_exps;
    ggml_tensor * sh_gate, * sh_up, * sh_down;
};

// Deterministic per-tensor fill: same name -> same bytes on every backend.
static void fill_tensor(ggml_tensor * t, const char * name) {
    std::seed_seq seq(name, name + strlen(name));
    std::mt19937 rng(seq);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> data(nbytes);
    if (t->type == GGML_TYPE_MXFP4) {
        // 17-byte blocks: e8m0 scale then 16 nibble bytes.
        std::uniform_int_distribution<int> byte(0, 255);
        std::uniform_int_distribution<int> escale(117, 123);
        for (size_t off = 0; off < nbytes; off += 17) {
            data[off] = (uint8_t) escale(rng);
            for (int j = 1; j < 17; j++) data[off + j] = (uint8_t) byte(rng);
        }
    } else {
        std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
        float * f = (float *) data.data();
        for (size_t i = 0; i < nbytes / 4; i++) f[i] = dist(rng);
    }
    ggml_backend_tensor_set(t, data.data(), 0, nbytes);
}

static weights_t make_weights(ggml_backend_t backend, ggml_backend_buffer_type_t mx_buft = nullptr) {
    weights_t w;
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    w.ctx = ggml_init(ip);

    w.gate_inp    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32,   N_EMBD, N_EXPERT);
    w.exp_probs_b = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32,   N_EXPERT);

    // mxfp4 tensors go into mx_buft when given (e.g. CPU_REPACK — the layout
    // llama.cpp serving actually runs); f32 router tensors stay in the
    // backend's default buffer.
    ggml_context * mctx = w.ctx;
    if (mx_buft) {
        w.ctx2 = ggml_init(ip);
        mctx = w.ctx2;
    }
    w.gate_exps   = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
    w.up_exps     = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
    w.down_exps   = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_FF, N_EMBD, N_EXPERT);
    w.sh_gate     = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF);
    w.sh_up       = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF);
    w.sh_down     = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_FF, N_EMBD);

    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    if (!w.buf) { fprintf(stderr, "weight buffer allocation failed\n"); exit(2); }
    if (mx_buft) {
        w.buf2 = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx2, mx_buft);
        if (!w.buf2) { fprintf(stderr, "mxfp4 buffer allocation failed (%s)\n",
                               ggml_backend_buft_name(mx_buft)); exit(2); }
        printf("mxfp4 weights in buffer: %s\n", ggml_backend_buffer_name(w.buf2));
    }

    fill_tensor(w.gate_inp,    "gate_inp");
    fill_tensor(w.exp_probs_b, "exp_probs_b");
    fill_tensor(w.gate_exps,   "gate_exps");
    fill_tensor(w.up_exps,     "up_exps");
    fill_tensor(w.down_exps,   "down_exps");
    fill_tensor(w.sh_gate,     "sh_gate");
    fill_tensor(w.sh_up,       "sh_up");
    fill_tensor(w.sh_down,     "sh_down");
    return w;
}

struct graph_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x, * selected, * selected_out, * out;
    ggml_gallocr_t galloc = nullptr;
};

// build_moe_ffn's deepseek4 path, plus the shared expert. With ids_as_input
// the router still computes probabilities but the selection comes from an
// input tensor — build_moe_ffn's own selected_experts_in escape hatch, used
// by the sanity check to hold routing fixed across backends (near-tied
// scores legitimately pick different top-6 on different arithmetic).
static graph_t build_graph(const weights_t & w, ggml_backend_t backend, int64_t n_tokens,
                           bool ids_as_input = false) {
    graph_t g;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    g.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
    ggml_set_input(g.x);

    // router: sqrt(softplus(logits)) at F32 prec, selection bias, top-6
    ggml_tensor * logits = ggml_mul_mat(ctx, w.gate_inp, g.x);          // [n_expert, n]
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    if (ids_as_input) {
        g.selected = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n_tokens);
        ggml_set_input(g.selected);
    } else {
        ggml_tensor * selection = ggml_add(ctx, probs, w.exp_probs_b);
        g.selected = ggml_argsort_top_k(ctx, selection, N_USED);        // [n_used, n]
    }
    // argsort_top_k returns a VIEW; an output flag on a view protects nothing
    // once the allocator reuses its parent (repo lesson). Snapshot via cont.
    g.selected_out = ggml_cont(ctx, g.selected);
    ggml_set_output(g.selected_out);
    ggml_build_forward_expand(g.gf, g.selected_out);

    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, g.selected);     // [1, n_used, n]
    weights = ggml_reshape_2d(ctx, weights, N_USED, n_tokens);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n_tokens);
    weights = ggml_scale(ctx, weights, W_SCALE);
    ggml_build_forward_expand(g.gf, weights);

    // routed experts
    ggml_tensor * xr   = ggml_reshape_3d(ctx, g.x, N_EMBD, 1, n_tokens);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, w.up_exps,   xr, g.selected); // [n_ff, n_used, n]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w.gate_exps, xr, g.selected);
    up   = ggml_clamp(ctx, up,   -CLAMP, CLAMP);
    gate = ggml_clamp(ctx, gate, -INFINITY, CLAMP);
    ggml_tensor * h = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, w.down_exps, h, g.selected); // [n_embd, n_used, n]
    experts = ggml_mul(ctx, experts, weights);
    ggml_build_forward_expand(g.gf, experts);

    ggml_tensor * moe_out = nullptr;
    for (int64_t i = 0; i < N_USED; i++) {
        ggml_tensor * v = ggml_view_2d(ctx, experts, N_EMBD, n_tokens,
                                       experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(g.gf, v);
        moe_out = i == 0 ? v : ggml_add(ctx, moe_out, v);
        ggml_build_forward_expand(g.gf, moe_out);
    }

    // shared expert, same clamps
    ggml_tensor * sg = ggml_mul_mat(ctx, w.sh_gate, g.x);
    ggml_tensor * su = ggml_mul_mat(ctx, w.sh_up, g.x);
    su = ggml_clamp(ctx, su, -CLAMP, CLAMP);
    sg = ggml_clamp(ctx, sg, -INFINITY, CLAMP);
    ggml_tensor * sh = ggml_swiglu_split(ctx, sg, su);
    ggml_tensor * sd = ggml_mul_mat(ctx, w.sh_down, sh);

    g.out = ggml_add(ctx, moe_out, sd);
    ggml_set_output(g.out);
    ggml_build_forward_expand(g.gf, g.out);

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) {
        fprintf(stderr, "graph allocation failed (n=%" PRId64 ")\n", n_tokens);
        exit(2);
    }
    return g;
}

static void free_graph(graph_t & g) {
    ggml_gallocr_free(g.galloc);
    ggml_free(g.ctx);
}

static void set_input(graph_t & g, int64_t n_tokens) {
    // fixed seed: same tokens for every batch size prefix and both backends
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) (N_EMBD * n_tokens));
    for (auto & v : x) v = dist(rng);
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * 4);
}

int main(int argc, char ** argv) {
    bool check = true;
    bool cpu_mode = false;   // run the sweep on the CPU backend instead —
                             // the CPU-offloaded-experts number for hybrid
                             // placements of models too big for VRAM
    bool repack = false;     // CPU mode with mxfp4 weights in the CPU_REPACK
                             // buffer type — what llama.cpp serving runs;
                             // the sanity gate then compares repack vs plain
    int reps = 100, threads = 16, only_n = 0;
    std::vector<int> tsweep;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-check")) check = false;
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) { cpu_mode = true; }
        else if (!strcmp(argv[i], "--repack")) { repack = true; cpu_mode = true; }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tsweep") && i + 1 < argc) {
            for (char * tok = strtok(argv[++i], ","); tok; tok = strtok(nullptr, ","))
                tsweep.push_back(atoi(tok));
        }
        else { fprintf(stderr, "usage: %s [--no-check] [--cpu] [--repack] [--threads N] [--reps N]\n", argv[0]); return 2; }
    }
    if (cpu_mode && !repack) check = false;   // plain-CPU vs CPU is a self-comparison

    const char * dir = getenv("GGML_BACKEND_DIR");
    ggml_backend_load_all_from_path(dir ? dir : "/home/oleksandr/projects/llama.cpp/build-hip/bin");

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(
        cpu_mode ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no device (set GGML_BACKEND_DIR)\n"); return 2; }
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    if (cpu_mode) ggml_backend_cpu_set_n_threads(gpu, threads);
    printf("device: %s (%s)%s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
           cpu_mode ? " [CPU mode]" : "");

    const double bytes_per_mat_gu = (double) ggml_row_size(GGML_TYPE_MXFP4, N_EMBD) * N_FF;
    const double bytes_per_expert = 2 * bytes_per_mat_gu
                                  + (double) ggml_row_size(GGML_TYPE_MXFP4, N_FF) * N_EMBD;
    const double bytes_shared = bytes_per_expert;
    printf("V4-Pro MoE layer: %" PRId64 "x(%" PRId64 "->%" PRId64 "), top-%" PRId64 "+1 shared, "
           "routed experts %.2f GiB\n\n",
           N_EXPERT, N_EMBD, N_FF, N_USED, N_EXPERT * bytes_per_expert / (1u << 30));

    ggml_backend_buffer_type_t mx_buft = nullptr;
    if (repack) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra) {
            for (ggml_backend_buffer_type_t * b = get_extra(dev); b && *b; b++) {
                printf("extra buft: %s\n", ggml_backend_buft_name(*b));
                if (strstr(ggml_backend_buft_name(*b), "REPACK")) mx_buft = *b;
            }
        }
        if (!mx_buft) { fprintf(stderr, "no REPACK buffer type on this build/ISA\n"); return 2; }
    }

    printf("loading weights to device...\n");
    weights_t w = make_weights(gpu, mx_buft);

    // --- sanity: n=1 on the CPU backend vs the GPU ---------------------------
    if (check) {
        printf("sanity: n=1 vs CPU backend... ");
        fflush(stdout);
        ggml_backend_dev_t cdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_t cpu = ggml_backend_dev_init(cdev, nullptr);
        weights_t wc = make_weights(cpu);
        graph_t gg = build_graph(w, gpu, 1);
        graph_t gc = build_graph(wc, cpu, 1, /*ids_as_input*/ true);
        set_input(gg, 1); set_input(gc, 1);
        ggml_backend_graph_compute(gpu, gg.gf);
        std::vector<int32_t> gids(N_USED);
        ggml_backend_tensor_get(gg.selected_out, gids.data(), 0, N_USED * 4);
        ggml_backend_tensor_set(gc.selected, gids.data(), 0, N_USED * 4);
        ggml_backend_graph_compute(cpu, gc.gf);
        std::vector<float> a(N_EMBD), b(N_EMBD);
        ggml_backend_tensor_get(gg.out, a.data(), 0, N_EMBD * 4);
        ggml_backend_tensor_get(gc.out, b.data(), 0, N_EMBD * 4);
        // Outputs are sums of +-hundreds-scale partials and both sides
        // quantize activations, so absolute dust ~0.4% of the *partial*
        // magnitude lands on every element, including near-zero ones — a
        // plain relative gate manufactures failures there (repo lesson,
        // twice now). Scale-aware gate: normalize by the reference RMS.
        double rms_ref = 0, rms_diff = 0, max_abs = 0;
        for (int64_t i = 0; i < N_EMBD; i++) {
            rms_ref  += (double) b[i] * b[i];
            rms_diff += (double) (a[i] - b[i]) * (a[i] - b[i]);
            max_abs   = fmax(max_abs, fabs(a[i] - b[i]));
        }
        rms_ref  = sqrt(rms_ref  / N_EMBD);
        rms_diff = sqrt(rms_diff / N_EMBD);
        const bool ok = max_abs / rms_ref <= 5e-2 && rms_diff / rms_ref <= 1e-2;
        printf("rms(ref) %.3e  rms(diff)/rms(ref) %.3e  max|diff|/rms(ref) %.3e  %s\n",
               rms_ref, rms_diff / rms_ref, max_abs / rms_ref, ok ? "ok" : "SUSPECT");
        free_graph(gg); free_graph(gc);
        ggml_backend_buffer_free(wc.buf); ggml_free(wc.ctx);
        if (wc.buf2) { ggml_backend_buffer_free(wc.buf2); ggml_free(wc.ctx2); }
        ggml_backend_free(cpu);
        if (!ok) return 1;
    }

    // --- the sweep -----------------------------------------------------------
    if (tsweep.empty()) tsweep.push_back(threads);
    printf("\n%-3s %-3s %12s %12s %10s %14s %10s\n",
           "t", "n", "us/graph", "us/token", "uniq exp", "bytes/graph", "GB/s");
    for (int t : tsweep) {
    if (cpu_mode) ggml_backend_cpu_set_n_threads(gpu, t);
    for (int64_t n = 1; n <= N_BATCH_MAX; n++) {
        if (only_n && n != only_n) continue;
        graph_t g = build_graph(w, gpu, n);
        set_input(g, n);

        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(gpu, g.gf);   // warmup

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) ggml_backend_graph_compute(gpu, g.gf);
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() / reps;

        std::vector<int32_t> ids((size_t) (N_USED * n));
        ggml_backend_tensor_get(g.selected_out, ids.data(), 0, ids.size() * 4);
        std::set<int32_t> uniq(ids.begin(), ids.end());

        const double bytes = uniq.size() * bytes_per_expert + bytes_shared;
        printf("%-3d %-3" PRId64 " %12.1f %12.1f %10zu %14.0f %10.1f\n",
               t, n, us, us / n, uniq.size(), bytes, bytes / us / 1e3);
        free_graph(g);
    }
    }

    if (w.buf2) { ggml_backend_buffer_free(w.buf2); ggml_free(w.ctx2); }
    ggml_backend_buffer_free(w.buf);
    ggml_free(w.ctx);
    ggml_backend_free(gpu);
    return 0;
}
