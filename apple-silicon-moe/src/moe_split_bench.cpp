// moe-split-bench — minimize the LATENCY of one n=4 MoE request by splitting
// its routed experts between Metal and the CPU, both running concurrently.
//
// Scenario: a request with n=4 token vectors arrives in main memory. The
// router runs on the CPU (no 154 us Metal round trip for an 11 MB matmul),
// the ~22-24 (token, slot) pairs are partitioned by unique expert — k
// experts to the GPU, the rest to the CPU — and both sides run a compact
// per-pair mul_mat_id graph (the hip-moe EP layout: xs [7168, 1, P],
// ids [1, P]) with the pair->token fold done on-graph via a [P, n] weight
// matrix, so each side returns a fixed [7168, n] partial and the host
// combine is three adds. The shared expert rides in one side's graph
// (--shared gpu|cpu), one command buffer per side per request.
//
// The sweep over k maps the latency valley between the two pure baselines
// (Metal-alone standard graph ~1.96 ms, CPU-alone ~3.8 ms at n=4).
// Prediction from solo yardsticks + the measured ~15% concurrency tax:
// optimum near k ~= 15 of 22, ~1.4-1.5 ms.
//
// Scheduler lessons applied (see moe_contention_bench.cpp): the CPU
// threadpool (FIFO HIGH, poll 100) is created AFTER the weights load, and
// both driver threads are FIFO-boosted so the pool's own workers cannot
// exile them to E cores.
//
// Sanity: the combined output is gated against the CPU standard graph with
// routing held fixed (scale-aware RMS gate, both sides quantize activations
// so ~1e-3 relative is two correct kernels).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
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

// GOTCHA that cost a debugging round here: ggml_gallocr only protects
// OUTPUT-flagged tensors from in-graph memory reuse (ggml-alloc.c,
// ggml_gallocr_free_node) — an INPUT's region is recycled once its last
// consumer runs. A graph whose inputs are set once and computed many times
// therefore reads garbage from rep 2 on if any later node reuses the slot
// (the CPU repack mul_mat_id asserted on out-of-range expert ids; Metal
// read the same garbage silently). llama.cpp survives because it rewrites
// inputs before every compute. Fix: allocate inputs in their own static
// buffer (ictx/ibuf), like llama.cpp's input buffer — gallocr then treats
// them as pre-allocated and never touches their memory.
struct input_buf_t {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    void alloc(ggml_backend_t backend) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "input buffer allocation failed\n"); exit(2); }
    }
    void free() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

// --- router graph (CPU): x -> ids [6, n] + scaled weights [6, n] --------------
struct router_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x, * ids_out, * w_out;
    input_buf_t in;
    ggml_gallocr_t galloc = nullptr;
};

static router_t build_router(const weights_t & w, ggml_backend_t backend, int64_t n) {
    router_t r;
    ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    r.ctx = ggml_init(ip);
    ggml_context * ctx = r.ctx;
    r.gf = ggml_new_graph(ctx);

    ggml_init_params iip = { 4 * ggml_tensor_overhead(), nullptr, true };
    r.in.ctx = ggml_init(iip);
    r.x = ggml_new_tensor_2d(r.in.ctx, GGML_TYPE_F32, N_EMBD, n);
    ggml_set_input(r.x);
    r.in.alloc(backend);

    ggml_tensor * logits = ggml_mul_mat(ctx, w.gate_inp, r.x);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * selection = ggml_add(ctx, probs, w.exp_probs_b);
    ggml_tensor * selected = ggml_argsort_top_k(ctx, selection, N_USED);   // view!
    r.ids_out = ggml_cont(ctx, selected);
    ggml_set_output(r.ids_out);
    ggml_build_forward_expand(r.gf, r.ids_out);

    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);          // [1, 6, n]
    weights = ggml_reshape_2d(ctx, weights, N_USED, n);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_scale(ctx, weights, W_SCALE);
    r.w_out = ggml_cont(ctx, weights);                                     // [6, n]
    ggml_set_output(r.w_out);
    ggml_build_forward_expand(r.gf, r.w_out);

    r.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(r.galloc, r.gf)) { fprintf(stderr, "router alloc failed\n"); exit(2); }
    return r;
}

// --- per-side compact expert graph + on-graph pair->token fold ----------------
// xs [7168, 1, P] and ids [1, P] are inputs (host-filled; routing is fixed per
// request), wmat [P, n] carries the router weights so partial [7168, n] is the
// side's finished contribution. Optionally the shared expert rides along.
struct side_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * xs = nullptr, * ids = nullptr, * wmat = nullptr;
    ggml_tensor * x_sh = nullptr, * sh_out = nullptr;
    ggml_tensor * partial = nullptr;
    input_buf_t in;
    ggml_gallocr_t galloc = nullptr;
    int64_t P = 0;
};

static side_t build_side(const weights_t & w, ggml_backend_t backend, int64_t P, int64_t n,
                         int64_t chunk, bool with_shared) {
    side_t g;
    g.P = P;
    if (P == 0 && !with_shared) return g;
    const int64_t n_chunks = P ? (P + chunk - 1) / chunk : 0;
    ggml_init_params ip = { (32 + 10 * (size_t) n_chunks) * ggml_tensor_overhead() + ggml_graph_overhead(),
                            nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    ggml_init_params iip = { 8 * ggml_tensor_overhead(), nullptr, true };
    g.in.ctx = ggml_init(iip);
    if (P > 0) {
        g.xs   = ggml_new_tensor_3d(g.in.ctx, GGML_TYPE_F32, N_EMBD, 1, P);
        g.ids  = ggml_new_tensor_2d(g.in.ctx, GGML_TYPE_I32, 1, P);
        g.wmat = ggml_new_tensor_2d(g.in.ctx, GGML_TYPE_F32, P, n);
        for (ggml_tensor * t : { g.xs, g.ids, g.wmat }) ggml_set_input(t);
    }
    if (with_shared) {
        g.x_sh = ggml_new_tensor_2d(g.in.ctx, GGML_TYPE_F32, N_EMBD, n);
        ggml_set_input(g.x_sh);
    }
    g.in.alloc(backend);

    if (P > 0) {
        ggml_tensor * pairs2d = nullptr;                                   // [7168, P]
        for (int64_t off = 0; off < P; off += chunk) {
            const int64_t c = std::min(chunk, P - off);
            ggml_tensor * xs_v  = ggml_view_3d(ctx, g.xs, N_EMBD, 1, c,
                                               g.xs->nb[1], g.xs->nb[2], off * g.xs->nb[2]);
            ggml_tensor * ids_v = ggml_view_2d(ctx, g.ids, 1, c, g.ids->nb[1], off * g.ids->nb[1]);
            ggml_tensor * up   = ggml_mul_mat_id(ctx, w.up_exps,   xs_v, ids_v);
            ggml_tensor * gate = ggml_mul_mat_id(ctx, w.gate_exps, xs_v, ids_v);
            up   = ggml_clamp(ctx, up,   -CLAMP, CLAMP);
            gate = ggml_clamp(ctx, gate, -INFINITY, CLAMP);
            ggml_tensor * h   = ggml_swiglu_split(ctx, gate, up);
            ggml_tensor * out = ggml_mul_mat_id(ctx, w.down_exps, h, ids_v); // [7168, 1, c]
            out = ggml_reshape_2d(ctx, out, N_EMBD, c);
            pairs2d = pairs2d ? ggml_concat(ctx, pairs2d, out, 1) : out;
        }
        ggml_tensor * pairsT = ggml_cont(ctx, ggml_transpose(ctx, pairs2d)); // [P, 7168]
        g.partial = ggml_mul_mat(ctx, pairsT, g.wmat);                       // [7168, n]
        ggml_set_output(g.partial);
        ggml_build_forward_expand(g.gf, g.partial);
    }

    if (with_shared) {
        ggml_tensor * sg = ggml_mul_mat(ctx, w.sh_gate, g.x_sh);
        ggml_tensor * su = ggml_mul_mat(ctx, w.sh_up, g.x_sh);
        su = ggml_clamp(ctx, su, -CLAMP, CLAMP);
        sg = ggml_clamp(ctx, sg, -INFINITY, CLAMP);
        ggml_tensor * h = ggml_swiglu_split(ctx, sg, su);
        g.sh_out = ggml_mul_mat(ctx, w.sh_down, h);
        ggml_set_output(g.sh_out);
        ggml_build_forward_expand(g.gf, g.sh_out);
    }

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { fprintf(stderr, "side graph alloc failed\n"); exit(2); }
    return g;
}

static void free_side(side_t & g) {
    if (g.galloc) ggml_gallocr_free(g.galloc);
    if (g.ctx) ggml_free(g.ctx);
    g.in.free();
    g = side_t{};
}

// --- full standard graph (baselines + sanity reference), routing as input ----
struct full_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * x, * selected, * out;
    input_buf_t in;
    ggml_gallocr_t galloc = nullptr;
};

static full_t build_full(const weights_t & w, ggml_backend_t backend, int64_t n, bool ids_as_input) {
    full_t g;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    ggml_init_params iip = { 4 * ggml_tensor_overhead(), nullptr, true };
    g.in.ctx = ggml_init(iip);
    g.x = ggml_new_tensor_2d(g.in.ctx, GGML_TYPE_F32, N_EMBD, n);
    ggml_set_input(g.x);
    if (ids_as_input) {
        g.selected = ggml_new_tensor_2d(g.in.ctx, GGML_TYPE_I32, N_USED, n);
        ggml_set_input(g.selected);
    }
    g.in.alloc(backend);

    ggml_tensor * logits = ggml_mul_mat(ctx, w.gate_inp, g.x);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    if (ids_as_input) {
        // g.selected already allocated in the input buffer
    } else {
        ggml_tensor * selection = ggml_add(ctx, probs, w.exp_probs_b);
        g.selected = ggml_argsort_top_k(ctx, selection, N_USED);
    }

    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, g.selected);
    weights = ggml_reshape_2d(ctx, weights, N_USED, n);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, N_USED, n);
    weights = ggml_scale(ctx, weights, W_SCALE);
    ggml_build_forward_expand(g.gf, weights);

    ggml_tensor * xr   = ggml_reshape_3d(ctx, g.x, N_EMBD, 1, n);
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
        ggml_tensor * v = ggml_view_2d(ctx, experts, N_EMBD, n,
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
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { fprintf(stderr, "full graph alloc failed\n"); exit(2); }
    return g;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// persistent worker, FIFO-boosted while COMPUTING but asleep while idle: a
// worker that busy-spins at FIFO priority between epochs steals a P core
// from whichever side is still working (measured in this bench's first
// version: CPU wall mean 2x its floor at k=0, the chief-exile trap
// self-inflicted). Condition variables cost ~10-20 us of wake latency per
// rep — noise at 1.5-4 ms scale — and idle threads that sleep hog nothing.
#include <condition_variable>
#include <mutex>
struct worker_t {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    uint64_t go = 0, done = 0;
    bool quit = false;
    ggml_backend_t be = nullptr;
    ggml_cgraph * gf = nullptr;
    double busy_us = 0;

    void start(int fifo_prio) {
        th = std::thread([this, fifo_prio] {
            sched_param p; p.sched_priority = fifo_prio;
            pthread_setschedparam(pthread_self(), SCHED_FIFO, &p);
            uint64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return quit || go != seen; });
                    if (quit) return;
                    seen = go;
                }
                const double t0 = now_us();
                if (gf) ggml_backend_graph_compute(be, gf);
                busy_us = now_us() - t0;
                {
                    std::lock_guard<std::mutex> lk(m);
                    done = seen;
                }
                cv.notify_all();
            }
        });
    }
    void kick(uint64_t epoch) {
        { std::lock_guard<std::mutex> lk(m); go = epoch; }
        cv.notify_all();
    }
    void wait(uint64_t epoch) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return done == epoch; });
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(m); quit = true; }
        cv.notify_all();
        th.join();
    }
};

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int n = 4, threads = 16, reps = 100;
    int poll = 0;                  // pool workers SLEEP between graphs by
                                   // default here — a spinning idle pool
                                   // starves the Metal driver thread during
                                   // the GPU-only tail of each request
    int64_t chunk = 64;            // one dispatch by default; --chunk to probe cliffs
    std::string shared_on = "gpu";
    bool check = true;
    std::vector<int> ks;           // GPU expert counts to sweep; default 0..uniq
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--reps")    && i + 1 < argc) reps    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--poll")    && i + 1 < argc) poll    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk")   && i + 1 < argc) chunk   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shared")  && i + 1 < argc) shared_on = argv[++i];
        else if (!strcmp(argv[i], "--no-check")) check = false;
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) {
            for (char * tok = strtok(argv[++i], ","); tok; tok = strtok(nullptr, ","))
                ks.push_back(atoi(tok));
        }
        else { fprintf(stderr, "usage: %s [--reps N] [--threads N] [--chunk N] [--shared gpu|cpu] [--k a,b,c] [--no-check]\n", argv[0]); return 2; }
    }

    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_dev_t cdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!gdev || !cdev) { fprintf(stderr, "missing device\n"); return 2; }
    ggml_backend_t gpu = ggml_backend_dev_init(gdev, nullptr);
    ggml_backend_t cpu = ggml_backend_dev_init(cdev, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, threads);

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

    printf("loading two weight sets (13.5 GiB each)...\n");
    weights_t wg = make_weights(gpu);
    weights_t wc = make_weights(cpu, mx_buft);

    // threadpool AFTER the load (repo lesson: an idle poll=100 FIFO pool
    // spinning through the load leaves later compute at a sticky 3.4x penalty)
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    tpp.prio = GGML_SCHED_PRIO_HIGH;
    tpp.poll = poll;
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(cpu, tp);

    // the main thread is the router's chief and the combiner — it must not be
    // outranked by the FIFO workers it coordinates
    { sched_param p; p.sched_priority = 70; pthread_setschedparam(pthread_self(), SCHED_FIFO, &p); }

    // --- the fixed request ---------------------------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x((size_t) (N_EMBD * n));
    for (auto & v : x) v = dist(rng);

    router_t router = build_router(wc, cpu, n);
    ggml_backend_tensor_set(router.x, x.data(), 0, x.size() * 4);
    ggml_backend_graph_compute(cpu, router.gf);

    std::vector<int32_t> ids((size_t) (N_USED * n));
    std::vector<float>   rw((size_t) (N_USED * n));
    ggml_backend_tensor_get(router.ids_out, ids.data(), 0, ids.size() * 4);
    ggml_backend_tensor_get(router.w_out,   rw.data(),  0, rw.size() * 4);

    // pairs grouped by unique expert (a pair is (token, slot) -> expert)
    struct pair_t { int tok, slot; int32_t exp; float w; };
    std::vector<pair_t> pairs;
    for (int t = 0; t < n; t++)
        for (int s = 0; s < (int) N_USED; s++)
            pairs.push_back({t, s, ids[(size_t) (t * N_USED + s)], rw[(size_t) (t * N_USED + s)]});
    std::set<int32_t> uniq_set;
    for (auto & p : pairs) uniq_set.insert(p.exp);
    std::vector<int32_t> uniq(uniq_set.begin(), uniq_set.end());
    const int U = (int) uniq.size();
    printf("request: n=%d, %zu pairs, %d unique experts, shared on %s, chunk %lld\n",
           n, pairs.size(), U, shared_on.c_str(), (long long) chunk);

    // --- baselines: standard full graph on each backend alone ----------------
    auto time_full = [&](ggml_backend_t be, const weights_t & w, const char * tag) {
        full_t f = build_full(w, be, n, false);
        ggml_backend_tensor_set(f.x, x.data(), 0, x.size() * 4);
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, f.gf);
        double best = 1e30;
        const double t0 = now_us();
        for (int i = 0; i < reps; i++) {
            const double r0 = now_us();
            ggml_backend_graph_compute(be, f.gf);
            best = std::min(best, now_us() - r0);
        }
        const double mean = (now_us() - t0) / reps;
        printf("baseline %-12s %8.1f us mean, %8.1f us min\n", tag, mean, best);
        ggml_gallocr_free(f.galloc); ggml_free(f.ctx); f.in.free();
        return mean;
    };
    const double base_gpu = time_full(gpu, wg, "Metal");
    const double base_cpu = time_full(cpu, wc, "CPU(repack)");

    // DVFS diagnostic: the shared-expert-only GPU graph, back to back. If
    // this reads ~250 us here but ~1200 us inside the split loop (where the
    // GPU works a fraction of each rep), the gap is clock ramp, not compute.
    {
        side_t s = build_side(wg, gpu, 0, n, chunk, true);
        ggml_backend_tensor_set(s.x_sh, x.data(), 0, x.size() * 4);
        for (int i = 0; i < 3; i++) ggml_backend_graph_compute(gpu, s.gf);
        double best = 1e30;
        const double t0 = now_us();
        for (int i = 0; i < 100; i++) {
            const double r0 = now_us();
            ggml_backend_graph_compute(gpu, s.gf);
            best = std::min(best, now_us() - r0);
        }
        printf("shared-only GPU graph, tight loop: %8.1f us mean, %8.1f us min\n",
               (now_us() - t0) / 100, best);
        free_side(s);
    }

    // --- sanity reference: CPU full graph with routing held fixed -------------
    std::vector<float> yref((size_t) (N_EMBD * n));
    {
        full_t f = build_full(wc, cpu, n, true);
        ggml_backend_tensor_set(f.x, x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(f.selected, ids.data(), 0, ids.size() * 4);
        ggml_backend_graph_compute(cpu, f.gf);
        ggml_backend_tensor_get(f.out, yref.data(), 0, yref.size() * 4);
        ggml_gallocr_free(f.galloc); ggml_free(f.ctx); f.in.free();
    }

    // --- the sweep over k = experts on GPU ------------------------------------
    if (ks.empty()) for (int k = 0; k <= U; k += 1) ks.push_back(k);

    worker_t wk_g, wk_c;
    wk_g.be = gpu; wk_c.be = cpu;
    wk_g.start(60);
    wk_c.start(80);

    printf("\n%-4s %-6s %-6s %10s %10s %10s %10s %10s %8s\n",
           "k", "P_gpu", "P_cpu", "wall mean", "wall min", "gpu busy", "cpu busy", "router", "vs base");
    double best_wall = 1e30; int best_k = -1;

    for (int k : ks) {
        // experts [0, k) of the unique list -> GPU, rest -> CPU
        std::set<int32_t> gpu_exp(uniq.begin(), uniq.begin() + std::min(k, U));
        std::vector<pair_t> pg, pc;
        for (auto & p : pairs) (gpu_exp.count(p.exp) ? pg : pc).push_back(p);

        const bool sh_gpu = shared_on == "gpu";
        side_t sg = build_side(wg, gpu, (int64_t) pg.size(), n, chunk, sh_gpu);
        side_t sc = build_side(wc, cpu, (int64_t) pc.size(), n, chunk, !sh_gpu);

        auto fill_side = [&](side_t & s, std::vector<pair_t> & ps) {
            if (s.P > 0) {
                std::vector<float> xs((size_t) (N_EMBD * s.P));
                std::vector<int32_t> sids((size_t) s.P);
                std::vector<float> wmat((size_t) (s.P * n), 0.0f);
                for (size_t i = 0; i < ps.size(); i++) {
                    memcpy(&xs[i * N_EMBD], &x[(size_t) ps[i].tok * N_EMBD], N_EMBD * 4);
                    sids[i] = ps[i].exp;
                    wmat[(size_t) ps[i].tok * s.P + i] = ps[i].w;
                }
                ggml_backend_tensor_set(s.xs,   xs.data(),   0, xs.size() * 4);
                ggml_backend_tensor_set(s.ids,  sids.data(), 0, sids.size() * 4);
                ggml_backend_tensor_set(s.wmat, wmat.data(), 0, wmat.size() * 4);
            }
            if (s.x_sh) ggml_backend_tensor_set(s.x_sh, x.data(), 0, x.size() * 4);
        };
        fill_side(sg, pg);
        fill_side(sc, pc);

        wk_g.gf = sg.gf; wk_c.gf = sc.gf;

        static uint64_t epoch = 0;
        auto one_rep = [&](double & router_us) {
            const double t0 = now_us();
            ggml_backend_graph_compute(cpu, router.gf);       // request latency includes routing
            router_us = now_us() - t0;
            const uint64_t e = ++epoch;
            wk_g.kick(e); wk_c.kick(e);
            wk_g.wait(e); wk_c.wait(e);
            // host combine: partial_gpu + partial_cpu + shared
            static std::vector<float> a, b, s;
            a.assign((size_t) (N_EMBD * n), 0.0f); b = a; s = a;
            if (sg.partial) ggml_backend_tensor_get(sg.partial, a.data(), 0, a.size() * 4);
            if (sc.partial) ggml_backend_tensor_get(sc.partial, b.data(), 0, b.size() * 4);
            ggml_tensor * sh = sg.sh_out ? sg.sh_out : sc.sh_out;
            ggml_backend_tensor_get(sh, s.data(), 0, s.size() * 4);
            for (size_t i = 0; i < a.size(); i++) a[i] += b[i] + s[i];
            return a;
        };

        double router_us = 0;
        auto y = one_rep(router_us);   // warmup + sanity output
        for (int i = 0; i < 2; i++) one_rep(router_us);

        if (check) {
            double rms_ref = 0, rms_diff = 0, max_abs = 0;
            for (size_t i = 0; i < y.size(); i++) {
                rms_ref  += (double) yref[i] * yref[i];
                rms_diff += (double) (y[i] - yref[i]) * (y[i] - yref[i]);
                max_abs   = fmax(max_abs, fabs(y[i] - yref[i]));
            }
            rms_ref = sqrt(rms_ref / y.size()); rms_diff = sqrt(rms_diff / y.size());
            if (max_abs / rms_ref > 5e-2 || rms_diff / rms_ref > 1e-2) {
                printf("k=%d SANITY FAILED: rms(diff)/rms(ref) %.3e max %.3e\n",
                       k, rms_diff / rms_ref, max_abs / rms_ref);
                return 1;
            }
        }

        double wall_sum = 0, wall_min = 1e30, gb = 0, cb = 0, ru = 0;
        for (int i = 0; i < reps; i++) {
            const double t0 = now_us();
            one_rep(router_us);
            const double us = now_us() - t0;
            wall_sum += us; wall_min = std::min(wall_min, us);
            gb += wk_g.busy_us; cb += wk_c.busy_us; ru += router_us;
        }
        const double wall = wall_sum / reps;
        printf("%-4d %-6zu %-6zu %10.1f %10.1f %10.1f %10.1f %10.1f %7.2fx\n",
               k, pg.size(), pc.size(), wall, wall_min, gb / reps, cb / reps, ru / reps,
               base_gpu / wall);
        if (wall < best_wall) { best_wall = wall; best_k = k; }

        free_side(sg); free_side(sc);
    }

    printf("\nbest: k=%d, wall %.1f us vs Metal-alone %.1f (%.1f%% faster), CPU-alone %.1f\n",
           best_k, best_wall, base_gpu, (base_gpu / best_wall - 1) * 100, base_cpu);

    wk_g.stop(); wk_c.stop();
    return 0;
}
