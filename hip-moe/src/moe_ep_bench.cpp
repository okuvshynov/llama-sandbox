// moe-ep-bench — expert parallelism for the V4-Pro MoE layer across 2 or 4
// dies, same llama.cpp HIP kernels, batch n in {1,2,4,6,8}.
//
// Die d owns the contiguous expert range [d*384/D, (d+1)*384/D): its slice of
// gate/up/down_exps (6.3 GiB at D=2, 3.1 at D=4). The router, the selection,
// and the shared expert live on die 0. Each step:
//
//   router (die 0)  -> ids + weights read back to host
//   partition       -> (token, slot) pairs grouped by owning die; per-die
//                      compact batch: xs [7168, 1, P_d], local ids [1, P_d]
//   submit          -> per-die graph (mul_mat_id up/gate -> clamp -> swiglu
//                      -> mul_mat_id down), compute_async on every die
//   wait + reduce   -> per-pair outputs read back, host does the weighted
//                      sum into [7168, n] (exact: the split is by expert)
//
// This is moe-serv's run_device_compact scheme rebuilt on stock kernels. The
// phase timers separate what the user asked to see: routing round trip,
// partition+upload, submission, and wait+readback — i.e. the price of
// communication and multiple submissions — while per-die pair counts and
// solo compute times show the imbalance. A Monte-Carlo pass over random
// inputs (through the real router weights) reports the *expected* max load,
// since one x sample is one draw from the multinomial.
//
// Sanity: the full graph on the CPU backend with routing held fixed, gate
// normalized by reference RMS (both sides quantize activations).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t N_EMBD   = 7168;
static const int64_t N_FF     = 3072;
static const int64_t N_EXPERT = 384;
static const int64_t N_USED   = 6;
static const float   CLAMP    = 10.0f;
static const float   W_SCALE  = 2.5f;

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// Same deterministic per-name streams as moe_ggml_bench: EP slices and the
// CPU reference see identical bytes. Each stream is generated once and
// cached — the expert tensors are 4.2 GB each and get sliced per die and
// reused by the CPU check, so regeneration would dominate the runtime.
#include <map>
static const std::vector<uint8_t> & gen_bytes(const char * name, size_t nbytes, bool mxfp4) {
    static std::map<std::string, std::vector<uint8_t>> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;

    std::seed_seq seq(name, name + strlen(name));
    std::mt19937 rng(seq);
    std::vector<uint8_t> data(nbytes);
    if (mxfp4) {
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
    return cache.emplace(name, std::move(data)).first->second;
}

static void fill_full(ggml_tensor * t, const char * name) {
    const auto & data = gen_bytes(name, ggml_nbytes(t), t->type == GGML_TYPE_MXFP4);
    ggml_backend_tensor_set(t, data.data(), 0, data.size());
}

// Upload one die's expert slice out of the full deterministic stream.
static void fill_slice(ggml_tensor * t, const char * name, size_t full_bytes,
                       int64_t e0, int64_t ne) {
    const auto & data = gen_bytes(name, full_bytes, true);
    const size_t per_expert = full_bytes / N_EXPERT;
    ggml_backend_tensor_set(t, data.data() + e0 * per_expert, 0, ne * per_expert);
}

// --- router + shared expert graph on die 0 ----------------------------------
struct router_t {
    ggml_context * ctx;
    ggml_cgraph * gf;
    ggml_tensor * x, * ids_out, * w_out, * sh_out;
    ggml_gallocr_t galloc;
};

static router_t build_router(ggml_context * wctx, ggml_tensor * gate_inp, ggml_tensor * exp_probs_b,
                             ggml_tensor * sh_gate, ggml_tensor * sh_up, ggml_tensor * sh_down,
                             ggml_backend_t backend, int64_t n) {
    (void) wctx;
    router_t r;
    ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    r.ctx = ggml_init(ip);
    ggml_context * ctx = r.ctx;
    r.gf = ggml_new_graph(ctx);

    r.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n);
    ggml_set_input(r.x);

    ggml_tensor * logits = ggml_mul_mat(ctx, gate_inp, r.x);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sqrt(ctx, ggml_softplus(ctx, logits));
    ggml_tensor * selection = ggml_add(ctx, probs, exp_probs_b);
    ggml_tensor * selected = ggml_argsort_top_k(ctx, selection, N_USED);   // view!
    r.ids_out = ggml_cont(ctx, selected);
    ggml_set_output(r.ids_out);
    ggml_build_forward_expand(r.gf, r.ids_out);

    ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, N_EXPERT, n);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);
    weights = ggml_reshape_2d(ctx, weights, N_USED, n);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_scale(ctx, weights, W_SCALE);       // [n_used, n]
    r.w_out = ggml_cont(ctx, weights);
    ggml_set_output(r.w_out);
    ggml_build_forward_expand(r.gf, r.w_out);

    ggml_tensor * sg = ggml_mul_mat(ctx, sh_gate, r.x);
    ggml_tensor * su = ggml_mul_mat(ctx, sh_up, r.x);
    su = ggml_clamp(ctx, su, -CLAMP, CLAMP);
    sg = ggml_clamp(ctx, sg, -INFINITY, CLAMP);
    ggml_tensor * sh = ggml_swiglu_split(ctx, sg, su);
    r.sh_out = ggml_mul_mat(ctx, sh_down, sh);         // [n_embd, n]
    ggml_set_output(r.sh_out);
    ggml_build_forward_expand(r.gf, r.sh_out);

    r.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(r.galloc, r.gf)) { fprintf(stderr, "router alloc failed\n"); exit(2); }
    return r;
}

// --- per-die compact expert graph -------------------------------------------
struct die_graph_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * xs = nullptr, * ids = nullptr;
    std::vector<ggml_tensor *> outs;   // one per <=CHUNK-pair dispatch
    // --ondie variant: full-x input + on-die gather and pair->token fold
    ggml_tensor * x_full = nullptr, * tok = nullptr, * wmat = nullptr, * partial = nullptr;
    ggml_gallocr_t galloc = nullptr;
    int64_t P = 0;
};

// ggml-cuda's mul_mat_id takes the MMVQ fast path only while dst->ne[2] stays
// under a per-type cap (get_mmvq_mmid_max_batch; measured boundary 8/9 for
// MXFP4 on gfx906 — 8 pairs run at ~85 us/pair, 9 pairs at ~240). The compact
// EP layout puts P in exactly that dimension, so the block is issued in
// <=8-pair chunks — the moe-serv 8-token chunking lesson, HIP edition.
static const int64_t CHUNK = 8;

static die_graph_t build_die_graph(ggml_tensor * gate_e, ggml_tensor * up_e, ggml_tensor * down_e,
                                   ggml_backend_t backend, int64_t P) {
    die_graph_t g;
    g.P = P;
    if (P == 0) return g;
    const int64_t n_chunks = (P + CHUNK - 1) / CHUNK;
    ggml_init_params ip = { (16 + 8 * (size_t) n_chunks) * ggml_tensor_overhead() + ggml_graph_overhead(),
                            nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    g.xs  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, P);
    g.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, P);
    ggml_set_input(g.xs);
    ggml_set_input(g.ids);

    for (int64_t off = 0; off < P; off += CHUNK) {
        const int64_t c = std::min(CHUNK, P - off);
        ggml_tensor * xs_v  = ggml_view_3d(ctx, g.xs, N_EMBD, 1, c,
                                           g.xs->nb[1], g.xs->nb[2], off * g.xs->nb[2]);
        ggml_tensor * ids_v = ggml_view_2d(ctx, g.ids, 1, c, g.ids->nb[1], off * g.ids->nb[1]);

        ggml_tensor * up   = ggml_mul_mat_id(ctx, up_e,   xs_v, ids_v);   // [n_ff, 1, c]
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_e, xs_v, ids_v);
        up   = ggml_clamp(ctx, up,   -CLAMP, CLAMP);
        gate = ggml_clamp(ctx, gate, -INFINITY, CLAMP);
        ggml_tensor * h = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * out = ggml_mul_mat_id(ctx, down_e, h, ids_v);       // [n_embd, 1, c]
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
        g.outs.push_back(out);
    }

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { fprintf(stderr, "die graph alloc failed\n"); exit(2); }
    return g;
}

// --ondie: the die receives the FULL x (uploaded before routing) and, after
// routing, only tiny per-pair metadata: token index, local expert id, and a
// [P, n] weight matrix. It gathers its pair inputs with get_rows, runs the
// chunked pipeline, and folds pairs into per-token partials with one small
// matmul — so the readback is a fixed [n_embd, n] and the host reduce is a
// plain sum (exact: the fold matrix carries the router weights).
static die_graph_t build_die_graph_ondie(ggml_tensor * gate_e, ggml_tensor * up_e, ggml_tensor * down_e,
                                         ggml_backend_t backend, int64_t P, int64_t n) {
    die_graph_t g;
    g.P = P;
    if (P == 0) return g;
    const int64_t n_chunks = (P + CHUNK - 1) / CHUNK;
    ggml_init_params ip = { (24 + 10 * (size_t) n_chunks) * ggml_tensor_overhead() + ggml_graph_overhead(),
                            nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph(ctx);

    g.x_full = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n);
    g.tok    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    g.ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, P);
    g.wmat   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, P, n);
    for (ggml_tensor * t : { g.x_full, g.tok, g.ids, g.wmat }) ggml_set_input(t);

    ggml_tensor * xg = ggml_get_rows(ctx, g.x_full, g.tok);              // [n_embd, P]
    xg = ggml_reshape_3d(ctx, xg, N_EMBD, 1, P);

    ggml_tensor * pairs2d = nullptr;                                     // [n_embd, P]
    for (int64_t off = 0; off < P; off += CHUNK) {
        const int64_t c = std::min(CHUNK, P - off);
        ggml_tensor * xs_v  = ggml_view_3d(ctx, xg, N_EMBD, 1, c,
                                           xg->nb[1], xg->nb[2], off * xg->nb[2]);
        ggml_tensor * ids_v = ggml_view_2d(ctx, g.ids, 1, c, g.ids->nb[1], off * g.ids->nb[1]);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, up_e,   xs_v, ids_v);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_e, xs_v, ids_v);
        up   = ggml_clamp(ctx, up,   -CLAMP, CLAMP);
        gate = ggml_clamp(ctx, gate, -INFINITY, CLAMP);
        ggml_tensor * h   = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * out = ggml_mul_mat_id(ctx, down_e, h, ids_v);      // [n_embd, 1, c]
        out = ggml_reshape_2d(ctx, out, N_EMBD, c);
        pairs2d = pairs2d ? ggml_concat(ctx, pairs2d, out, 1) : out;
    }
    // fold pairs -> tokens: partial[i, t] = sum_p pairs[i, p] * wmat[p, t]
    ggml_tensor * pairsT = ggml_cont(ctx, ggml_transpose(ctx, pairs2d)); // [P, n_embd]
    g.partial = ggml_mul_mat(ctx, pairsT, g.wmat);                       // [n_embd, n]
    ggml_set_output(g.partial);
    ggml_build_forward_expand(g.gf, g.partial);

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(g.galloc, g.gf)) { fprintf(stderr, "ondie graph alloc failed\n"); exit(2); }
    return g;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int n_dies = 4, reps = 50;
    bool check = true;
    // Fixes named by the rocprofv3 timeline (2026-08-18): the sequential
    // per-die upload loop staggers die starts (~60% realized concurrency at
    // n=4), and pageable staging makes hipMemcpyAsync block (~87 us/call).
    bool reorder = false;   // fuse upload->launch per die; submit all
                            // readbacks before the first sync
    bool pinned  = false;   // stage xs/ids/out in pinned host memory
    bool ondie   = false;   // full-x upload before routing, on-die gather +
                            // pair->token fold, tiny post-routing metadata,
                            // shared-expert readback off the critical path
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dies") && i + 1 < argc) n_dies = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-check")) check = false;
        else if (!strcmp(argv[i], "--reorder")) reorder = true;
        else if (!strcmp(argv[i], "--pinned")) pinned = true;
        else if (!strcmp(argv[i], "--ondie")) { ondie = true; pinned = true; }
        else { fprintf(stderr, "usage: %s [--dies 2|4] [--reps N] [--no-check] [--reorder] [--pinned] [--ondie]\n", argv[0]); return 2; }
    }
    const int64_t E_PER = N_EXPERT / n_dies;

    const char * bdir = getenv("GGML_BACKEND_DIR");
    ggml_backend_load_all_from_path(bdir ? bdir : "/home/oleksandr/projects/llama.cpp/build-hip/bin");

    std::vector<ggml_backend_t> dies(n_dies);
    for (int d = 0; d < n_dies; d++) {
        char name[16];
        snprintf(name, sizeof(name), "ROCm%d", d);
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
        if (!dev) { fprintf(stderr, "no device %s\n", name); return 2; }
        dies[d] = ggml_backend_dev_init(dev, nullptr);
    }
    printf("EP over %d dies, %" PRId64 " experts each (%.2f GiB/die)\n", n_dies, E_PER,
           E_PER * 3.0 * ggml_row_size(GGML_TYPE_MXFP4, N_EMBD) * N_FF / (1u << 30));

    // --- weights ------------------------------------------------------------
    const size_t full_gu = (size_t) ggml_row_size(GGML_TYPE_MXFP4, N_EMBD) * N_FF * N_EXPERT;
    const size_t full_dn = (size_t) ggml_row_size(GGML_TYPE_MXFP4, N_FF) * N_EMBD * N_EXPERT;

    std::vector<ggml_context *> wctx(n_dies);
    std::vector<ggml_backend_buffer_t> wbuf(n_dies);
    std::vector<ggml_tensor *> gate_e(n_dies), up_e(n_dies), down_e(n_dies);
    ggml_tensor * gate_inp, * exp_probs_b, * sh_gate, * sh_up, * sh_down;

    for (int d = 0; d < n_dies; d++) {
        ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
        wctx[d] = ggml_init(ip);
        gate_e[d] = ggml_new_tensor_3d(wctx[d], GGML_TYPE_MXFP4, N_EMBD, N_FF, E_PER);
        up_e[d]   = ggml_new_tensor_3d(wctx[d], GGML_TYPE_MXFP4, N_EMBD, N_FF, E_PER);
        down_e[d] = ggml_new_tensor_3d(wctx[d], GGML_TYPE_MXFP4, N_FF, N_EMBD, E_PER);
        if (d == 0) {
            gate_inp    = ggml_new_tensor_2d(wctx[d], GGML_TYPE_F32, N_EMBD, N_EXPERT);
            exp_probs_b = ggml_new_tensor_1d(wctx[d], GGML_TYPE_F32, N_EXPERT);
            sh_gate     = ggml_new_tensor_2d(wctx[d], GGML_TYPE_MXFP4, N_EMBD, N_FF);
            sh_up       = ggml_new_tensor_2d(wctx[d], GGML_TYPE_MXFP4, N_EMBD, N_FF);
            sh_down     = ggml_new_tensor_2d(wctx[d], GGML_TYPE_MXFP4, N_FF, N_EMBD);
        }
        wbuf[d] = ggml_backend_alloc_ctx_tensors(wctx[d], dies[d]);
        if (!wbuf[d]) { fprintf(stderr, "weights alloc failed on die %d\n", d); return 2; }
    }
    printf("loading weight slices...\n");
    for (int d = 0; d < n_dies; d++) {
        fill_slice(gate_e[d], "gate_exps", full_gu, d * E_PER, E_PER);
        fill_slice(up_e[d],   "up_exps",   full_gu, d * E_PER, E_PER);
        fill_slice(down_e[d], "down_exps", full_dn, d * E_PER, E_PER);
    }
    fill_full(gate_inp, "gate_inp");
    fill_full(exp_probs_b, "exp_probs_b");
    fill_full(sh_gate, "sh_gate");
    fill_full(sh_up, "sh_up");
    fill_full(sh_down, "sh_down");

    // --- Monte-Carlo expected imbalance through the real router -------------
    {
        auto gi = gen_bytes("gate_inp", (size_t) N_EMBD * N_EXPERT * 4, false);
        auto eb = gen_bytes("exp_probs_b", (size_t) N_EXPERT * 4, false);
        const float * W = (const float *) gi.data();
        const float * B = (const float *) eb.data();
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        const int S = 200;
        printf("\nexpected imbalance, %d random tokens through the real router "
               "(ideal per-die share = 6n/%d):\n", S, n_dies);
        printf("%-3s %14s %14s\n", "n", "E[max pairs]", "E[max]/ideal");
        for (int64_t n : {1, 2, 4, 6, 8}) {
            double sum_max = 0;
            std::vector<float> x(N_EMBD);
            for (int s = 0; s < S; s++) {
                std::vector<int> load(n_dies, 0);
                for (int64_t t = 0; t < n; t++) {
                    for (auto & v : x) v = dist(rng);
                    std::vector<std::pair<float,int>> sc(N_EXPERT);
                    for (int64_t e = 0; e < N_EXPERT; e++) {
                        double acc = 0;
                        for (int64_t k = 0; k < N_EMBD; k++) acc += (double) W[e * N_EMBD + k] * x[k];
                        const double sp = acc > 20 ? acc : log1p(exp(acc));
                        sc[e] = { (float) (sqrt(sp) + B[e]), (int) e };
                    }
                    std::partial_sort(sc.begin(), sc.begin() + N_USED, sc.end(),
                                      [](auto & a, auto & b) { return a.first > b.first; });
                    for (int64_t j = 0; j < N_USED; j++) load[sc[j].second / E_PER]++;
                }
                sum_max += *std::max_element(load.begin(), load.end());
            }
            const double ideal = (double) N_USED * n / n_dies;
            printf("%-3" PRId64 " %14.2f %14.2f\n", n, sum_max / S, sum_max / S / ideal);
        }
    }

    // --- the sweep -----------------------------------------------------------
    printf("\n%-3s %-14s %10s | %8s %8s %8s %8s | %10s %10s\n",
           "n", "pairs/die", "solo us", "router", "prep", "submit", "wait+rd", "us/graph", "us/token");
    for (int64_t n : {1, 2, 4, 6, 8}) {
        router_t r = build_router(wctx[0], gate_inp, exp_probs_b, sh_gate, sh_up, sh_down, dies[0], n);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> x((size_t) (N_EMBD * n));
        for (auto & v : x) v = dist(rng);

        // one routing pass to learn the partition (x fixed -> ids fixed)
        ggml_backend_tensor_set(r.x, x.data(), 0, x.size() * 4);
        ggml_backend_graph_compute(dies[0], r.gf);
        std::vector<int32_t> ids((size_t) (N_USED * n));
        std::vector<float> wts((size_t) (N_USED * n));
        ggml_backend_tensor_get(r.ids_out, ids.data(), 0, ids.size() * 4);
        ggml_backend_tensor_get(r.w_out, wts.data(), 0, wts.size() * 4);

        std::vector<std::vector<std::pair<int,int>>> pairs(n_dies);  // (token, slot)
        for (int64_t t = 0; t < n; t++)
            for (int64_t s = 0; s < N_USED; s++)
                pairs[ids[t * N_USED + s] / E_PER].push_back({ (int) t, (int) (t * N_USED + s) });

        std::vector<die_graph_t> dg(n_dies);
        for (int d = 0; d < n_dies; d++)
            dg[d] = ondie
                ? build_die_graph_ondie(gate_e[d], up_e[d], down_e[d], dies[d], (int64_t) pairs[d].size(), n)
                : build_die_graph(gate_e[d], up_e[d], down_e[d], dies[d], (int64_t) pairs[d].size());

        std::vector<std::vector<float>> xs_h(n_dies), out_h(n_dies);
        std::vector<std::vector<int32_t>> lid_h(n_dies);
        for (int d = 0; d < n_dies; d++) {
            xs_h[d].resize(pairs[d].size() * N_EMBD);
            out_h[d].resize(pairs[d].size() * N_EMBD);
            lid_h[d].resize(pairs[d].size());
        }
        // staging pointers: the vectors above, or carved out of a pinned
        // host buffer so the async copies are actually asynchronous
        std::vector<ggml_backend_buffer_t> pin(n_dies, nullptr);
        std::vector<float *> xs_p(n_dies), out_p(n_dies);
        std::vector<int32_t *> id_p(n_dies);
        for (int d = 0; d < n_dies; d++) {
            const size_t P = pairs[d].size();
            xs_p[d] = xs_h[d].data(); out_p[d] = out_h[d].data(); id_p[d] = lid_h[d].data();
            if (pinned && P) {
                const size_t nb_xs = P * N_EMBD * 4, nb_out = nb_xs;
                const size_t nb_id = (P * 4 + 63) / 64 * 64;
                ggml_backend_buffer_type_t hbuft =
                    ggml_backend_dev_host_buffer_type(ggml_backend_get_device(dies[d]));
                pin[d] = ggml_backend_buft_alloc_buffer(hbuft, nb_xs + nb_out + nb_id);
                char * base = (char *) ggml_backend_buffer_get_base(pin[d]);
                xs_p[d]  = (float *) base;
                out_p[d] = (float *) (base + nb_xs);
                id_p[d]  = (int32_t *) (base + nb_xs + nb_out);
            }
        }
        std::vector<float> y((size_t) (N_EMBD * n));

        // ondie staging: one pinned x (portable across devices), pinned sh +
        // per-die partial, and tiny per-die tok/lids/wmat regions
        ggml_backend_buffer_t opin = nullptr;
        float * x_pin = nullptr, * sh_pin = nullptr;
        std::vector<float *> part_p(n_dies), wm_p(n_dies);
        std::vector<int32_t *> tok_p(n_dies), lid2_p(n_dies);
        if (ondie) {
            const size_t nb_x = (size_t) N_EMBD * n * 4;
            size_t total = 2 * nb_x;                       // x + sh
            std::vector<size_t> off(n_dies);
            for (int d = 0; d < n_dies; d++) {
                off[d] = total;
                const size_t P = pairs[d].size();
                total += nb_x + ((P * 4 + 63) / 64 * 64) * 2 + (size_t) P * n * 4 + 64;
            }
            ggml_backend_buffer_type_t hbuft =
                ggml_backend_dev_host_buffer_type(ggml_backend_get_device(dies[0]));
            opin = ggml_backend_buft_alloc_buffer(hbuft, total);
            char * base = (char *) ggml_backend_buffer_get_base(opin);
            x_pin  = (float *) base;
            sh_pin = (float *) (base + nb_x);
            for (int d = 0; d < n_dies; d++) {
                const size_t P = pairs[d].size(), pal = (P * 4 + 63) / 64 * 64;
                char * b = base + off[d];
                part_p[d] = (float *) b;
                tok_p[d]  = (int32_t *) (b + nb_x);
                lid2_p[d] = (int32_t *) (b + nb_x + pal);
                wm_p[d]   = (float *) (b + nb_x + 2 * pal);
            }
            memcpy(x_pin, x.data(), nb_x);
        }

        auto one_rep_ondie = [&](double * ph) {
            const size_t nb_x = (size_t) N_EMBD * n * 4;
            double t0 = now_us();
            // x to every die + router compute, all queued before any wait
            for (int d = 0; d < n_dies; d++)
                if (dg[d].P) ggml_backend_tensor_set_async(dies[d], dg[d].x_full, x_pin, 0, nb_x);
            ggml_backend_tensor_set_async(dies[0], r.x, x_pin, 0, nb_x);
            ggml_backend_graph_compute_async(dies[0], r.gf);
            ggml_backend_synchronize(dies[0]);
            ggml_backend_tensor_get(r.ids_out, ids.data(), 0, ids.size() * 4);
            ggml_backend_tensor_get(r.w_out, wts.data(), 0, wts.size() * 4);
            double t1 = now_us();

            std::vector<int> fill(n_dies, 0);
            for (int d = 0; d < n_dies; d++)
                if (dg[d].P) memset(wm_p[d], 0, pairs[d].size() * n * 4);
            for (int64_t t = 0; t < n; t++)
                for (int64_t s = 0; s < N_USED; s++) {
                    const int e = ids[t * N_USED + s], d = e / (int) E_PER;
                    const int p = fill[d]++;
                    lid2_p[d][p] = e % (int) E_PER;
                    tok_p[d][p] = (int) t;
                    wm_p[d][t * (int64_t) pairs[d].size() + p] = wts[t * N_USED + s];
                }
            for (int d = 0; d < n_dies; d++) {
                if (!dg[d].P) continue;
                const size_t P = pairs[d].size();
                ggml_backend_tensor_set_async(dies[d], dg[d].tok,  tok_p[d],  0, P * 4);
                ggml_backend_tensor_set_async(dies[d], dg[d].ids,  lid2_p[d], 0, P * 4);
                ggml_backend_tensor_set_async(dies[d], dg[d].wmat, wm_p[d],   0, P * n * 4);
                ggml_backend_graph_compute_async(dies[d], dg[d].gf);
            }
            double t2 = now_us();
            double t3 = t2;   // submit fused into prep above

            for (int d = 0; d < n_dies; d++)
                if (dg[d].P) ggml_backend_tensor_get_async(dies[d], dg[d].partial, part_p[d], 0, nb_x);
            ggml_backend_tensor_get_async(dies[0], r.sh_out, sh_pin, 0, nb_x);
            for (int d = 0; d < n_dies; d++)
                if (dg[d].P) ggml_backend_synchronize(dies[d]);
            ggml_backend_synchronize(dies[0]);

            memcpy(y.data(), sh_pin, nb_x);
            for (int d = 0; d < n_dies; d++) {
                if (!dg[d].P) continue;
                for (size_t i = 0; i < y.size(); i++) y[i] += part_p[d][i];
            }
            double t4 = now_us();
            ph[0] = t1 - t0; ph[1] = t2 - t1; ph[2] = t3 - t2; ph[3] = t4 - t3; ph[4] = t4 - t0;
        };

        auto one_rep_classic = [&](double * ph) {
            double t0 = now_us();
            ggml_backend_tensor_set_async(dies[0], r.x, x.data(), 0, x.size() * 4);
            ggml_backend_graph_compute_async(dies[0], r.gf);
            ggml_backend_synchronize(dies[0]);
            ggml_backend_tensor_get(r.ids_out, ids.data(), 0, ids.size() * 4);
            ggml_backend_tensor_get(r.w_out, wts.data(), 0, wts.size() * 4);
            ggml_backend_tensor_get(r.sh_out, y.data(), 0, y.size() * 4);   // y := shared
            double t1 = now_us();

            std::vector<int> fill(n_dies, 0);
            for (int64_t t = 0; t < n; t++)
                for (int64_t s = 0; s < N_USED; s++) {
                    const int e = ids[t * N_USED + s], d = e / (int) E_PER;
                    const int p = fill[d]++;
                    id_p[d][p] = e % (int) E_PER;
                    memcpy(xs_p[d] + (size_t) p * N_EMBD, x.data() + (size_t) t * N_EMBD, N_EMBD * 4);
                }
            if (reorder) {
                // fused per-die pipeline: die d computes while die d+1 uploads
                for (int d = 0; d < n_dies; d++) {
                    if (!dg[d].P) continue;
                    ggml_backend_tensor_set_async(dies[d], dg[d].xs, xs_p[d], 0, pairs[d].size() * N_EMBD * 4);
                    ggml_backend_tensor_set_async(dies[d], dg[d].ids, id_p[d], 0, pairs[d].size() * 4);
                    ggml_backend_graph_compute_async(dies[d], dg[d].gf);
                }
            } else {
                for (int d = 0; d < n_dies; d++) {
                    if (!dg[d].P) continue;
                    ggml_backend_tensor_set_async(dies[d], dg[d].xs, xs_p[d], 0, pairs[d].size() * N_EMBD * 4);
                    ggml_backend_tensor_set_async(dies[d], dg[d].ids, id_p[d], 0, pairs[d].size() * 4);
                }
            }
            double t2 = now_us();

            if (!reorder) {
                for (int d = 0; d < n_dies; d++)
                    if (dg[d].P) ggml_backend_graph_compute_async(dies[d], dg[d].gf);
            }
            double t3 = now_us();

            if (reorder) {
                // submit every readback, then wait — no sync between submits
                for (int d = 0; d < n_dies; d++) {
                    if (!dg[d].P) continue;
                    size_t off = 0;
                    for (ggml_tensor * o : dg[d].outs) {
                        ggml_backend_tensor_get_async(dies[d], o, out_p[d] + off, 0, ggml_nbytes(o));
                        off += (size_t) ggml_nelements(o);
                    }
                }
                for (int d = 0; d < n_dies; d++)
                    if (dg[d].P) ggml_backend_synchronize(dies[d]);
            } else {
                for (int d = 0; d < n_dies; d++) {
                    if (!dg[d].P) continue;
                    size_t off = 0;
                    for (ggml_tensor * o : dg[d].outs) {
                        ggml_backend_tensor_get_async(dies[d], o, out_p[d] + off, 0, ggml_nbytes(o));
                        off += (size_t) ggml_nelements(o);
                    }
                    ggml_backend_synchronize(dies[d]);
                }
            }
            for (int d = 0; d < n_dies; d++)
                for (size_t p = 0; p < pairs[d].size(); p++) {
                    const int t = pairs[d][p].first;
                    const float wt = wts[pairs[d][p].second];
                    const float * src = out_p[d] + p * N_EMBD;
                    float * dst = y.data() + (size_t) t * N_EMBD;
                    for (int64_t i = 0; i < N_EMBD; i++) dst[i] += wt * src[i];
                }
            double t4 = now_us();
            ph[0] = t1 - t0; ph[1] = t2 - t1; ph[2] = t3 - t2; ph[3] = t4 - t3; ph[4] = t4 - t0;
        };
        auto one_rep = [&](double * ph) { ondie ? one_rep_ondie(ph) : one_rep_classic(ph); };

        double ph[5];
        for (int i = 0; i < 3; i++) one_rep(ph);   // warmup

        // solo compute per die: the imbalance in time
        std::vector<double> solo(n_dies, 0);
        for (int d = 0; d < n_dies; d++) {
            if (!dg[d].P) continue;
            const double t0 = now_us();
            for (int i = 0; i < 10; i++) {
                ggml_backend_graph_compute_async(dies[d], dg[d].gf);
                ggml_backend_synchronize(dies[d]);
            }
            solo[d] = (now_us() - t0) / 10;
        }

        std::vector<std::vector<double>> phases(5);
        for (int i = 0; i < reps; i++) {
            one_rep(ph);
            for (int k = 0; k < 5; k++) phases[k].push_back(ph[k]);
        }

        // sanity vs the CPU backend, routing held fixed
        if (check) {
            ggml_backend_dev_t cdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_t cpu = ggml_backend_dev_init(cdev, nullptr);
            ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * cctx = ggml_init(ip);
            ggml_tensor * c_gate = ggml_new_tensor_3d(cctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
            ggml_tensor * c_up   = ggml_new_tensor_3d(cctx, GGML_TYPE_MXFP4, N_EMBD, N_FF, N_EXPERT);
            ggml_tensor * c_down = ggml_new_tensor_3d(cctx, GGML_TYPE_MXFP4, N_FF, N_EMBD, N_EXPERT);
            ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, cpu);
            fill_full(c_gate, "gate_exps");
            fill_full(c_up, "up_exps");
            fill_full(c_down, "down_exps");

            die_graph_t cg = build_die_graph(c_gate, c_up, c_down, cpu, N_USED * n);
            std::vector<float> cxs((size_t) (N_USED * n * N_EMBD));
            std::vector<int32_t> cid((size_t) (N_USED * n));
            for (int64_t t = 0; t < n; t++)
                for (int64_t s = 0; s < N_USED; s++) {
                    cid[t * N_USED + s] = ids[t * N_USED + s];
                    memcpy(cxs.data() + (size_t) (t * N_USED + s) * N_EMBD,
                           x.data() + (size_t) t * N_EMBD, N_EMBD * 4);
                }
            ggml_backend_tensor_set(cg.xs, cxs.data(), 0, cxs.size() * 4);
            ggml_backend_tensor_set(cg.ids, cid.data(), 0, cid.size() * 4);
            ggml_backend_graph_compute(cpu, cg.gf);
            std::vector<float> cout((size_t) (N_USED * n * N_EMBD));
            size_t coff = 0;
            for (ggml_tensor * o : cg.outs) {
                ggml_backend_tensor_get(o, cout.data() + coff, 0, ggml_nbytes(o));
                coff += (size_t) ggml_nelements(o);
            }

            std::vector<float> yref((size_t) (N_EMBD * n), 0.0f);
            // shared expert from the GPU run stays in y; reference only the routed part
            for (int64_t t = 0; t < n; t++)
                for (int64_t s = 0; s < N_USED; s++) {
                    const float wt = wts[t * N_USED + s];
                    const float * src = cout.data() + (size_t) (t * N_USED + s) * N_EMBD;
                    for (int64_t i = 0; i < N_EMBD; i++) yref[(size_t) t * N_EMBD + i] += wt * src[i];
                }
            // y currently holds shared + routed from the last rep; subtract shared
            std::vector<float> sh((size_t) (N_EMBD * n));
            ggml_backend_tensor_get(r.sh_out, sh.data(), 0, sh.size() * 4);
            double rms_ref = 0, rms_diff = 0;
            for (size_t i = 0; i < yref.size(); i++) {
                const double a = y[i] - sh[i], b = yref[i];
                rms_ref += b * b; rms_diff += (a - b) * (a - b);
            }
            rms_ref = sqrt(rms_ref / yref.size());
            rms_diff = sqrt(rms_diff / yref.size());
            if (rms_diff / rms_ref > 1e-2) {
                printf("n=%" PRId64 ": SANITY SUSPECT rms(diff)/rms(ref) = %.3e\n", n, rms_diff / rms_ref);
                return 1;
            }
            ggml_gallocr_free(cg.galloc); ggml_free(cg.ctx);
            ggml_backend_buffer_free(cbuf); ggml_free(cctx);
            ggml_backend_free(cpu);
        }

        char pstr[64] = "", sstr[64] = "";
        for (int d = 0; d < n_dies; d++) {
            snprintf(pstr + strlen(pstr), sizeof(pstr) - strlen(pstr), "%s%zu", d ? "/" : "", pairs[d].size());
            snprintf(sstr + strlen(sstr), sizeof(sstr) - strlen(sstr), "%s%.0f", d ? "/" : "", solo[d]);
        }
        const double tot = median(phases[4]);
        printf("%-3" PRId64 " %-14s %10s | %8.1f %8.1f %8.1f %8.1f | %10.1f %10.1f\n",
               n, pstr, sstr,
               median(phases[0]), median(phases[1]), median(phases[2]), median(phases[3]),
               tot, tot / n);

        for (int d = 0; d < n_dies; d++) {
            if (dg[d].P) { ggml_gallocr_free(dg[d].galloc); ggml_free(dg[d].ctx); }
            if (pin[d]) ggml_backend_buffer_free(pin[d]);
        }
        if (opin) ggml_backend_buffer_free(opin);
        ggml_gallocr_free(r.galloc); ggml_free(r.ctx);
    }

    for (int d = 0; d < n_dies; d++) {
        ggml_backend_buffer_free(wbuf[d]);
        ggml_free(wctx[d]);
        ggml_backend_free(dies[d]);
    }
    return 0;
}
