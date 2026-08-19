// moe-contention-bench — do the GPU's and the CPU's memory bandwidths sum on
// unified memory, or does the fabric cap them?
//
// The go/no-go for any hybrid expert placement on Apple Silicon: each side
// runs its OWN full V4-Pro MoE layer (independent 13.5 GB synthetic MXFP4
// weight sets — Metal default buffer / CPU_REPACK), first solo, then
// concurrently from two host threads. Solo vs concurrent per-graph time is
// the contention, with nothing else varying. Representative traffic on both
// sides (mxfp4 expert streaming), no routing coupling between them.
//
// Time-based phases (the two sides run at very different speeds): each phase
// runs for --secs and counts completed graphs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <pthread.h>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

// --- V4-Pro-0813 MoE shape (same as moe_ggml_bench.cpp) ----------------------
static const int64_t N_EMBD   = 7168;
static const int64_t N_FF     = 3072;
static const int64_t N_EXPERT = 384;
static const int64_t N_USED   = 6;
static const float   CLAMP    = 10.0f;
static const float   W_SCALE  = 2.5f;

struct weights_t {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx2 = nullptr;
    ggml_backend_buffer_t buf2 = nullptr;
    ggml_tensor * gate_inp, * exp_probs_b;
    ggml_tensor * gate_exps, * up_exps, * down_exps;
    ggml_tensor * sh_gate, * sh_up, * sh_down;
};

static void fill_tensor(ggml_tensor * t, const char * name) {
    std::seed_seq seq(name, name + strlen(name));
    std::mt19937 rng(seq);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> data(nbytes);
    if (t->type == GGML_TYPE_MXFP4) {
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
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
    w.ctx = ggml_init(ip);

    w.gate_inp    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_EXPERT);
    w.exp_probs_b = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EXPERT);

    ggml_context * mctx = w.ctx;
    if (mx_buft) {
        w.ctx2 = ggml_init(ip);
        mctx = w.ctx2;
    }
    w.gate_exps = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
    w.up_exps   = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
    w.down_exps = ggml_new_tensor_3d(mctx, GGML_TYPE_MXFP4, N_FF, N_EMBD, N_EXPERT);
    w.sh_gate   = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF);
    w.sh_up     = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_EMBD, N_FF);
    w.sh_down   = ggml_new_tensor_2d(mctx, GGML_TYPE_MXFP4, N_FF, N_EMBD);

    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    if (!w.buf) { fprintf(stderr, "weight buffer allocation failed\n"); exit(2); }
    if (mx_buft) {
        w.buf2 = ggml_backend_alloc_ctx_tensors_from_buft(w.ctx2, mx_buft);
        if (!w.buf2) { fprintf(stderr, "mxfp4 buffer allocation failed\n"); exit(2); }
        printf("mxfp4 weights in buffer: %s\n", ggml_backend_buffer_name(w.buf2));
    }

    // sequential fill, same as moe_ggml_bench.cpp
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

static graph_t build_graph(const weights_t & w, ggml_backend_t backend, int64_t n_tokens) {
    graph_t g;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    g.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
    ggml_set_input(g.x);

    ggml_tensor * logits = ggml_mul_mat(ctx, w.gate_inp, g.x);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * selection = ggml_add(ctx, probs, w.exp_probs_b);
    g.selected = ggml_argsort_top_k(ctx, selection, N_USED);
    g.selected_out = ggml_cont(ctx, g.selected);
    ggml_set_output(g.selected_out);
    ggml_build_forward_expand(g.gf, g.selected_out);

    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, g.selected);
    weights = ggml_reshape_2d(ctx, weights, N_USED, n_tokens);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n_tokens);
    weights = ggml_scale(ctx, weights, W_SCALE);
    ggml_build_forward_expand(g.gf, weights);

    ggml_tensor * xr   = ggml_reshape_3d(ctx, g.x, N_EMBD, 1, n_tokens);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, w.up_exps,   xr, g.selected);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w.gate_exps, xr, g.selected);
    up   = ggml_clamp(ctx, up,   -CLAMP, CLAMP);
    gate = ggml_clamp(ctx, gate, -INFINITY, CLAMP);
    ggml_tensor * h = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * experts = ggml_mul_mat_id(ctx, w.down_exps, h, g.selected);
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
        fprintf(stderr, "graph allocation failed\n");
        exit(2);
    }
    return g;
}

static void set_input(graph_t & g, int64_t n_tokens, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) (N_EMBD * n_tokens));
    for (auto & v : x) v = dist(rng);
    ggml_backend_tensor_set(g.x, x.data(), 0, x.size() * 4);
}

// run graphs for `secs`, return mean us/graph
static double run_phase(ggml_backend_t be, ggml_cgraph * gf, double secs, long * out_reps = nullptr) {
    const auto t0 = std::chrono::steady_clock::now();
    long reps = 0;
    for (;;) {
        ggml_backend_graph_compute(be, gf);
        reps++;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= secs) {
            if (out_reps) *out_reps = reps;
            return el * 1e6 / reps;
        }
    }
}

int main(int argc, char ** argv) {
    int n_gpu = 4, n_cpu = 4, threads = 16;
    int prio = 2, poll = 100;   // the standalone bench's low-variance config;
                                // under concurrency the right policy is an
                                // open question this tool exists to answer
    double secs = 3.0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n-gpu")   && i + 1 < argc) n_gpu   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-cpu")   && i + 1 < argc) n_cpu   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prio")    && i + 1 < argc) prio    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--poll")    && i + 1 < argc) poll    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--secs")    && i + 1 < argc) secs    = atof(argv[++i]);
        else { fprintf(stderr, "usage: %s [--n-gpu N] [--n-cpu N] [--threads N] [--prio P] [--poll P] [--secs S]\n", argv[0]); return 2; }
    }

    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!gdev || !cdev) { fprintf(stderr, "missing device\n"); return 2; }
    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    ggml_backend_t cpu = ggml_backend_dev_init(cdev, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, threads);
    printf("gpu: %s (%s)   cpu: %s, %d threads\n",
           ggml_backend_dev_name(gdev), ggml_backend_dev_description(gdev),
           ggml_backend_dev_name(cdev), threads);

    // CPU side gets the repack layout — what serving actually runs
    ggml_backend_buffer_type_t mx_buft = nullptr;
    {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cdev);
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
        if (get_extra)
            for (ggml_backend_buffer_type_t * b = get_extra(cdev); b && *b; b++)
                if (strstr(ggml_backend_buft_name(*b), "REPACK")) mx_buft = *b;
        if (!mx_buft) { fprintf(stderr, "no REPACK buffer type\n"); return 2; }
    }

    const double bytes_per_mat_gu = (double) ggml_row_size(GGML_TYPE_MXFP4, N_EMBD) * N_FF;
    const double bytes_per_expert = 2 * bytes_per_mat_gu
                                  + (double) ggml_row_size(GGML_TYPE_MXFP4, N_FF) * N_EMBD;

    // --n-gpu 0: CPU-only diagnostic — no Metal weights, no Metal graph, so a
    // slow CPU number here indicts this bench's own setup rather than the
    // GPU side's presence
    const bool gpu_on = n_gpu > 0;

    printf("loading weight sets (13.5 GiB each)...\n");
    weights_t wg; graph_t gg;
    if (gpu_on) {
        wg = make_weights(gpu);
        gg = build_graph(wg, gpu, n_gpu);
        set_input(gg, n_gpu, 42);
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(gpu, gg.gf);
    }
    weights_t wc = make_weights(cpu, mx_buft);
    graph_t gc = build_graph(wc, cpu, n_cpu);
    // different seeds -> different routing on the two sides; both fixed per rep
    set_input(gc, n_cpu, 43);

    // prio HIGH + aggressive poll: the config that collapsed the CPU bench's
    // load-to-load spread from 20-55% to <3% (default-QoS workers wander onto
    // E cores). MUST be created here, after the weights load, not at startup:
    // poll=100 workers busy-spin while idle, and 16 SCHED_FIFO-80 threads
    // spinning through the multi-minute fill left ALL later compute at a
    // sticky 3.4x penalty (70 GB/s vs 222 measured in this very bench) —
    // macOS demotes FIFO threads that hog their quantum and the demotion
    // outlives the fill.
    ggml_threadpool_t tp = nullptr;
    if (prio > 0 || poll >= 0) {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
        tpp.prio = (ggml_sched_priority) prio;
        if (poll >= 0) tpp.poll = poll;
        tp = ggml_threadpool_new(&tpp);
        ggml_backend_cpu_set_threadpool(cpu, tp);
        printf("cpu threadpool: prio %d, poll %d\n", prio, poll);
    }

    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(cpu, gc.gf);

    auto graph_bytes = [&](graph_t & g, int n) {
        std::vector<int32_t> ids((size_t) (N_USED * n));
        ggml_backend_tensor_get(g.selected_out, ids.data(), 0, ids.size() * 4);
        std::set<int32_t> uniq(ids.begin(), ids.end());
        return (uniq.size() + 1) * bytes_per_expert;   // + shared
    };
    const double bg = gpu_on ? graph_bytes(gg, n_gpu) : 0;
    const double bc = graph_bytes(gc, n_cpu);

    printf("solo phases (%.1f s each)...\n", secs);
    const double us_c_solo = run_phase(cpu, gc.gf, secs);
    const double us_g_solo = gpu_on ? run_phase(gpu, gg.gf, secs) : 0;

    // With FIFO workers, the driver threads must be boosted too: the CPU
    // chief thread (= the graph's worker 0) at default priority gets exiled
    // to an E core by its own P-core-pinning workers, and the whole graph
    // runs at chief speed (measured: +663% CPU tax; theory tested by this
    // very boost).
    auto boost = [&](int fifo_prio) {
        if (prio <= 0) return;
        sched_param p; p.sched_priority = fifo_prio;
        pthread_setschedparam(pthread_self(), SCHED_FIFO, &p);
    };

    double us_g_conc = 0, us_c_conc = 0;
    if (gpu_on) {
        printf("concurrent phase (%.1f s)...\n", secs);
        std::atomic<int> ready{0};
        std::thread tg([&] {
            boost(60);
            ready++; while (ready.load() < 2) {}
            us_g_conc = run_phase(gpu, gg.gf, secs);
        });
        std::thread tc([&] {
            boost(80);
            ready++; while (ready.load() < 2) {}
            us_c_conc = run_phase(cpu, gc.gf, secs);
        });
        tg.join(); tc.join();
    }

    printf("\n%-14s %-3s %12s %12s %8s %10s %10s\n",
           "side", "n", "solo us/g", "conc us/g", "tax", "solo GB/s", "conc GB/s");
    if (gpu_on)
        printf("%-14s %-3d %12.1f %12.1f %7.1f%% %10.1f %10.1f\n",
               "Metal", n_gpu, us_g_solo, us_g_conc,
               (us_g_conc / us_g_solo - 1) * 100, bg / us_g_solo / 1e3, bg / us_g_conc / 1e3);
    printf("%-14s %-3d %12.1f %12.1f %7.1f%% %10.1f %10.1f\n",
           "CPU(repack)", n_cpu, us_c_solo, us_c_conc,
           gpu_on ? (us_c_conc / us_c_solo - 1) * 100 : 0.0,
           bc / us_c_solo / 1e3, gpu_on ? bc / us_c_conc / 1e3 : 0.0);
    if (gpu_on)
        printf("combined streaming: solo-sum %.1f GB/s, concurrent %.1f GB/s\n",
               bg / us_g_solo / 1e3 + bc / us_c_solo / 1e3,
               bg / us_g_conc / 1e3 + bc / us_c_conc / 1e3);

    return 0;
}
