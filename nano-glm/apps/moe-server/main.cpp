// moe-server — the MoE backend for nano-glm (PLAN.md step 1).
//
// Holds the router and every routed expert; the trunk holds everything else.
// One request is one MoE layer for one batch: activation in, combined row out.
// Stateless — no KV, nothing carried between calls — so a restart costs only
// the model load.
//
// Correctness bar is the same as everywhere else here: the graphs it builds
// come from models/glm_dsa/moe_block.h, the identical source the in-process client uses, so a
// KL == 0 gate against llama.cpp still applies once the trunk talks to this
// over a socket.
//
// A layer is evaluated in two graphs rather than one — `build_moe_router` on
// the host, then `build_moe_experts` on each device — because expert work
// cannot be partitioned across devices until the routing decision is in host
// memory, and in a single graph it is not. With today's one device that
// composes back to exactly `build_moe_block`, and the gate holds it to that:
// `gate.py rpc` is byte-identical to the local path. PLAN.md step 3.
//
// moe_proto.h must come first: winsock2.h has to precede windows.h, which
// models/glm_dsa/model.h pulls in.
//
// Deferred optimisations for this backend — graph caching, zero-copy transfer,
// fusing up+gate, f16 on the wire, and why request pipelining is not worth
// doing — live in OPTIMIZATION.md under "Backend micro-optimisations", each
// with the condition that would make it matter.

#include "moe_proto.h"

#include "build_info.h"
#include "models/glm_dsa/model.h"
#include "models/glm_dsa/moe_block.h"
#include "cpu_topology.h"

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
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static uint32_t us_since(clk::time_point t0) {
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
    return (uint32_t) (d < 0 ? 0 : d);
}

struct server_params {
    std::string model_path;
    std::string host       = "127.0.0.1";
    uint16_t    port       = 5711;
    int32_t     n_threads  = physical_core_count();
    bool        verbose    = false;
    uint32_t    gpu_experts = 0;   // experts per MoE layer to place on a GPU
    uint32_t    cpu_experts = 0;   // ... or on a second CPU device, for testing
    uint32_t    gpu_devices = 0;   // how many GPU dies to spread over; 0 = all
    bool        compare    = false; // evaluate both paths, report per-layer error
    std::string force_split;       // e.g. "2,2,2,2" — timing only, wrong output
};

static bool parse_args(int argc, char ** argv, server_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m")       && i + 1 < argc) p.model_path = argv[++i];
        else if (!strcmp(a, "--host")   && i + 1 < argc) p.host       = argv[++i];
        else if (!strcmp(a, "--port")   && i + 1 < argc) p.port       = (uint16_t) atoi(argv[++i]);
        else if (!strcmp(a, "-t")       && i + 1 < argc) p.n_threads  = atoi(argv[++i]);
        else if (!strcmp(a, "--gpu-experts") && i + 1 < argc) p.gpu_experts = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(a, "--cpu-experts") && i + 1 < argc) p.cpu_experts = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(a, "--gpu-devices") && i + 1 < argc) p.gpu_devices = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(a, "--force-split") && i + 1 < argc) p.force_split = argv[++i];
        else if (!strcmp(a, "--compare"))                     p.compare     = true;
        else if (!strcmp(a, "-v"))                      p.verbose    = true;
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty()) {
        fprintf(stderr,
            "Usage: moe-server -m <first-shard.gguf> [options]\n"
            "  --host <ip>   bind address (default 127.0.0.1)\n"
            "  --port <n>    port (default 5711)\n"
            "  -t <int>      threads (default: physical cores, ignoring SMT siblings)\n"
            "  --gpu-experts <k>  place experts 0..k-1 of every MoE layer on the GPUs,\n"
            "                     dealt round-robin (needs build.ps1 -Vk; 0 = CPU only)\n"
            "  --gpu-devices <n>  spread over n GPU dies (default: every one found).\n"
            "                     Each device gets its own host thread\n"
            "  --cpu-experts <k>  same, but onto a second CPU device. Same compaction\n"
            "                     path, same arithmetic — so it separates a compaction\n"
            "                     bug from a GPU numerics difference\n"
            "  --force-split <a,b,..>  ignore routing and give device 1 a slots of\n"
            "                every token, device 2 b, ...; the rest stay on the CPU.\n"
            "                e.g. with 8 used experts: 2,2,2,2 = all on four dies,\n"
            "                0 on CPU; 2,2,2,1 = 7 on dies, 1 on CPU; 7,0,0,0 = one\n"
            "                die does 7. *** PRODUCES WRONG OUTPUT ON PURPOSE ***\n"
            "                — the work has the right shape and cost but the wrong\n"
            "                weights, so it times distributions residency cannot\n"
            "                currently reach. Never use it to produce logits.\n"
            "  --compare     evaluate every layer on BOTH the full CPU path and the\n"
            "                split path, return the CPU one, and report the per-layer\n"
            "                error at disconnect. Roughly 2x slower; it is the only\n"
            "                measurement that sees the expert path without 75 layers\n"
            "                of amplification on top of it\n"
            "  -v            log every request\n"
            "  --devices     list the ggml devices this build can see, and exit\n"
            "  --version     print the build fingerprint, and exit\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// evaluation

// One MoE layer's worth of experts held on a device that does not hold them
// all. The three tensors mirror the model's `ffn_*_exps` but with only the
// resident experts, so the third dimension is k rather than n_expert.
struct device_layer {
    ggml_tensor * up   = nullptr;   // [n_embd, n_ff_exp, k]
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    std::vector<int32_t> local_of;  // n_expert entries; -1 where not resident
};

// A device is *a backend, the experts it holds, and — from increment 3 — a
// thread to drive it*. Deliberately not GPU-shaped: a node of the 4-socket
// NUMA machine is the same object, a backend with thread affinity and expert
// weights in its own memory, and the partition-then-combine machinery below is
// identical for both. PLAN.md step 3.
struct moe_device {
    ggml_backend_t        backend = nullptr;
    ggml_gallocr_t        galloc  = nullptr;
    std::vector<uint8_t>  meta;      // graph scratch, reused across requests
    std::vector<float>    partial;   // full-path contribution [n_embd, n_tokens]

    // Residency. `holds_all` is the DRAM device: it uses the model's own
    // tensors and is the fallthrough for every expert no other device has.
    bool                    holds_all = true;
    uint32_t                n_resident = 0;   // experts per layer held here
    ggml_context *          wctx = nullptr;   // tensor structs for the slices
    ggml_backend_buffer_t   wbuf = nullptr;   // the device memory holding them
    std::vector<device_layer> dl;             // indexed by model layer

    // This request's share, as (token, slot) pairs. Rebuilt per request; the
    // capacity survives, so the hot path does not allocate.
    std::vector<int32_t>  pair_token;
    std::vector<int32_t>  pair_expert;  // index into this device's weights
    std::vector<float>    pair_weight;
    std::vector<float>    gathered_x;   // [n_embd, m], x rows repeated per pair
    std::vector<float>    rows_out;     // [n_embd, m], what the device returned
    bool                  took_all = false;  // owns every pair this request
    uint64_t              n_pairs_total = 0; // across the connection, for the log
};

// Per-layer error of the split path against the full CPU path, accumulated
// over a connection. This is the measurement end-to-end logit KL cannot make:
// by layer, before 75 layers of amplification turn everything into the same
// saturated number.
struct layer_error {
    double   max_abs  = 0.0;   // largest |split - reference| in this layer
    double   sum_sq   = 0.0;   // for an RMS relative to the reference's own RMS
    double   ref_sq   = 0.0;
    uint64_t n        = 0;
    uint64_t n_calls  = 0;
};

struct moe_backend {
    nano_model              M;
    int                     n_threads = 1;
    std::vector<moe_device> devices;

    // Forced placement (--force-split). Slot counts for devices 1..n; whatever
    // is left of n_expert_used stays on device 0. Produces WRONG OUTPUT by
    // construction — it exists to time work distributions that residency
    // cannot currently produce.
    std::vector<uint32_t>    force_split;

    // Compare mode (--compare). Runs both paths, hands the trunk the *CPU*
    // one, and records how far the split path was from it.
    bool                     compare = false;
    std::vector<float>       ref_out;    // the full path's answer for this layer
    std::vector<layer_error> err;        // indexed by model layer

    // The router's own graph and its result, read back to the host between the
    // two halves. Runs on devices[0]'s backend: the DRAM device *is* the host
    // in this deployment, and sharing the backend avoids a second CPU thread
    // pool that would only ever idle.
    ggml_gallocr_t        router_galloc = nullptr;
    std::vector<uint8_t>  router_meta;
    std::vector<int32_t>  ids_buf;      // [n_expert_used, n_tokens]
    std::vector<float>    weights_buf;  // [1, n_expert_used, n_tokens]

    // reused across requests so the hot path does no allocation
    std::vector<float>    in_buf;
    std::vector<float>    out_buf;
};

// ---- the router half ------------------------------------------------------
//
// Runs the router and reads its decision back to the host. This read-back is
// the reason the layer is two graphs instead of one: which experts a token
// wants is a *node* in the graph (`argsort_top_k`), so in a single graph it is
// not known until the graph has already run — too late to have partitioned the
// expert work across devices.
static bool run_router(moe_backend & B, uint32_t layer, int32_t n_tokens, const float * x) {
    const nano_hparams & h = B.M.h;

    ggml_init_params ip = {
        /*.mem_size   =*/ B.router_meta.size(),
        /*.mem_buffer =*/ B.router_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n_tokens);
    ggml_set_name(inp, "x");
    ggml_set_input(inp);

    const moe_routing r = build_moe_router(ctx, gf, h, layer, B.M.layers[layer], inp, n_tokens);

    // Both are read back, so both are made contiguous first. `ids` needs it:
    // top-k is a *view* of the argsort output with the unselected rows still
    // underneath, so its bytes are not a flat range. `weights` needs it for a
    // subtler reason — it can end as a reshape view, and marking a view as a
    // graph output does not keep its parent's buffer alive.
    //
    // Neither copy changes a value, and `mul_mat_id` only ever reads the ids,
    // so byte-identity with the single-graph version is preserved.
    ggml_tensor * ids = ggml_cont(ctx, r.ids);
    ggml_tensor * wts = ggml_cont(ctx, r.weights);
    ggml_set_output(ids);
    ggml_set_output(wts);
    ggml_build_forward_expand(gf, ids);
    ggml_build_forward_expand(gf, wts);

    bool ok = ggml_gallocr_alloc_graph(B.router_galloc, gf);
    if (ok) {
        ggml_backend_tensor_set(inp, x, 0, (size_t) h.n_embd * n_tokens * sizeof(float));
        ok = ggml_backend_graph_compute(B.devices[0].backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        const size_t n_sel = (size_t) h.n_expert_used * n_tokens;
        ggml_backend_tensor_get(ids, B.ids_buf.data(),     0, n_sel * sizeof(int32_t));
        ggml_backend_tensor_get(wts, B.weights_buf.data(), 0, n_sel * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

// ---- residency ------------------------------------------------------------
//
// Copy the given experts of every MoE layer onto a device. *Which* experts is
// the caller's business and deliberately the dumbest possible choice — a
// prefix, split round-robin across devices — because placement is a separate
// question that lives in OPTIMIZATION.md, whose current answer is that no
// static choice transfers between prompts.
//
// The copy is cheap to express because experts are contiguous in the model's
// `ffn_*_exps`: expert e is exactly the byte range [e*nb[2], (e+1)*nb[2]), and
// the model is mmap'd into host memory, so each expert is one tensor_set.
static bool device_load_experts(moe_backend & B, moe_device & D,
                                const std::vector<int32_t> & experts) {
    const nano_hparams & h = B.M.h;
    const uint32_t n_moe = h.n_layer - h.n_dense_lead;
    const uint32_t k = (uint32_t) experts.size();

    D.holds_all  = false;
    D.n_resident = k;
    D.dl.resize(h.n_layer);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 3 * n_moe + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    D.wctx = ggml_init(ip);
    if (!D.wctx) { fprintf(stderr, "moe-server: no context for resident experts\n"); return false; }

    for (uint32_t il = h.n_dense_lead; il < h.n_layer; il++) {
        const nano_layer & L = B.M.layers[il];
        device_layer & dl = D.dl[il];
        dl.up   = ggml_new_tensor_3d(D.wctx, L.ffn_up_exps->type,
                                     L.ffn_up_exps->ne[0],   L.ffn_up_exps->ne[1],   k);
        dl.gate = ggml_new_tensor_3d(D.wctx, L.ffn_gate_exps->type,
                                     L.ffn_gate_exps->ne[0], L.ffn_gate_exps->ne[1], k);
        dl.down = ggml_new_tensor_3d(D.wctx, L.ffn_down_exps->type,
                                     L.ffn_down_exps->ne[0], L.ffn_down_exps->ne[1], k);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(D.backend);
    const size_t need = ggml_backend_alloc_ctx_tensors_from_buft_size(D.wctx, buft);
    size_t dev_free = 0, dev_total = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(D.backend), &dev_free, &dev_total);
    fprintf(stderr, "moe-server: resident experts need %.2f GiB, device has %.2f GiB free\n",
            need / 1073741824.0, dev_free / 1073741824.0);
    if (need > dev_free) {
        fprintf(stderr, "moe-server: not enough device memory — lower --gpu-experts\n");
        return false;
    }

    D.wbuf = ggml_backend_alloc_ctx_tensors(D.wctx, D.backend);
    if (!D.wbuf) { fprintf(stderr, "moe-server: failed to allocate resident experts\n"); return false; }

    const auto t0 = clk::now();
    for (uint32_t il = h.n_dense_lead; il < h.n_layer; il++) {
        const nano_layer & L = B.M.layers[il];
        device_layer & dl = D.dl[il];
        dl.local_of.assign(h.n_expert, -1);

        ggml_tensor * src[3] = { L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps };
        ggml_tensor * dst[3] = { dl.up,         dl.gate,         dl.down         };
        for (uint32_t j = 0; j < k; j++) {
            const uint32_t e = (uint32_t) experts[j];
            dl.local_of[e] = (int32_t) j;
            for (int w = 0; w < 3; w++) {
                const size_t stride = src[w]->nb[2];
                ggml_backend_tensor_set(dst[w], (const char *) src[w]->data + (size_t) e * stride,
                                        (size_t) j * stride, stride);
            }
        }
    }
    fprintf(stderr, "moe-server: uploaded %u experts/layer x %u layers in %.1fs\n",
            k, n_moe, std::chrono::duration<double>(clk::now() - t0).count());
    return true;
}

// ---- assigning this request's work to devices -----------------------------
//
// Residency is per *expert*; slots are per *token*. So a device's share is a
// set of (token, slot) pairs and it varies across the batch — there is no
// per-device slot subset to speak of, which is why the work is compacted
// rather than masked.
//
// devices[0] holds everything and is the fallthrough, so every pair lands
// somewhere and no pair lands twice.
static void assign_pairs(moe_backend & B, uint32_t layer, int32_t n_tokens) {
    const nano_hparams & h = B.M.h;
    const size_t n_pairs = (size_t) n_tokens * h.n_expert_used;

    for (moe_device & D : B.devices) {
        D.pair_token.clear();
        D.pair_expert.clear();
        D.pair_weight.clear();
    }

    // Slot-major within a token, and devices scanned in a fixed order, so the
    // assignment — and therefore the combine — is deterministic run to run.
    for (int32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < h.n_expert_used; s++) {
            const size_t  idx = (size_t) t * h.n_expert_used + s;
            const int32_t e   = B.ids_buf[idx];

            size_t  dev   = 0;   // the fallthrough device holds everything
            int32_t local = e;   // and indexes its weights by global expert id

            if (!B.force_split.empty()) {
                // Forced placement: slot s goes wherever the pattern says,
                // regardless of which expert the router picked. The expert is
                // then remapped into that device's resident range, because it
                // almost certainly does not hold the one that was chosen.
                //
                // **This computes the wrong answer on purpose.** The shape and
                // cost of the work are exactly right — same matmul dimensions,
                // same transfers, same number of pairs per device — and only
                // the weights are wrong, so it measures the timing of a work
                // distribution that residency cannot currently produce.
                uint32_t cursor = 0;
                for (size_t d = 1; d < B.devices.size(); d++) {
                    const uint32_t take = d - 1 < B.force_split.size() ? B.force_split[d - 1] : 0;
                    if (s >= cursor && s < cursor + take) {
                        const uint32_t k = B.devices[d].n_resident;
                        dev   = d;
                        local = k ? (int32_t) ((uint32_t) e % k) : 0;
                        break;
                    }
                    cursor += take;
                }
            } else {
                for (size_t d = 1; d < B.devices.size(); d++) {
                    const std::vector<int32_t> & lo = B.devices[d].dl[layer].local_of;
                    if (!lo.empty() && lo[e] >= 0) {
                        dev   = d;
                        local = lo[e];
                        break;
                    }
                }
            }

            B.devices[dev].pair_token.push_back(t);
            B.devices[dev].pair_expert.push_back(local);
            B.devices[dev].pair_weight.push_back(B.weights_buf[idx]);
        }
    }

    for (moe_device & D : B.devices) {
        D.took_all = D.pair_token.size() == n_pairs;
        D.n_pairs_total += D.pair_token.size();
    }
}

// ---- one device's share of the expert half --------------------------------
//
// `ids` and `weights` come in as ordinary input tensors rather than as the
// router's own nodes. `ggml_mul_mat_id` cannot tell the difference, which is
// what makes the split free.
//
// The full path: this device owns every pair and holds every expert, so it
// runs exactly the op sequence the single graph ran. That is what keeps the
// gate byte-identical in the CPU-only configuration, and it is why the
// compacted path below is used *only* when some other device took work.
static bool run_device_full(moe_backend & B, moe_device & D, uint32_t layer,
                            int32_t n_tokens, const float * x) {
    const nano_hparams & h = B.M.h;

    ggml_init_params ip = {
        /*.mem_size   =*/ D.meta.size(),
        /*.mem_buffer =*/ D.meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h.n_embd, n_tokens);
    ggml_set_name(inp, "x");
    ggml_set_input(inp);

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, h.n_expert_used, n_tokens);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor * wts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, h.n_expert_used, n_tokens);
    ggml_set_name(wts, "weights");
    ggml_set_input(wts);

    ggml_tensor * moe_out = build_moe_experts(ctx, gf, h, B.M.layers[layer], inp,
                                              moe_routing{ ids, wts }, n_tokens);
    ggml_set_output(moe_out);
    ggml_build_forward_expand(gf, moe_out);

    bool ok = ggml_gallocr_alloc_graph(D.galloc, gf);
    if (ok) {
        const size_t n_act = (size_t) h.n_embd        * n_tokens;
        const size_t n_sel = (size_t) h.n_expert_used * n_tokens;
        ggml_backend_tensor_set(inp, x,                    0, n_act * sizeof(float));
        ggml_backend_tensor_set(ids, B.ids_buf.data(),     0, n_sel * sizeof(int32_t));
        ggml_backend_tensor_set(wts, B.weights_buf.data(), 0, n_sel * sizeof(float));
        ok = ggml_backend_graph_compute(D.backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        ggml_backend_tensor_get(moe_out, D.partial.data(), 0,
                                (size_t) h.n_embd * n_tokens * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

// The compacted path: this device owns only some of the (token, slot) pairs.
//
// Every pair becomes one column of a dense batch, so the shape presented to
// `mul_mat_id` is [n_embd, 1, m] with a [1, m] id tensor — one expert per
// column — rather than [n_embd, 1, n_tokens] with n_expert_used per column. A
// token that has three of its eight experts here appears three times in `x`.
// Duplicating the activation is the price of not making the device evaluate
// experts it does not hold, and at 24 KiB a row it is a good trade.
//
// The op sequence is the same as `build_moe_experts` up to the weighted
// multiply; the per-token sum that `build_moe_experts` does with pairwise adds
// happens on the host instead, in the scatter-add, because which rows belong
// to a token is host knowledge.
static bool run_device_compact(moe_backend & B, moe_device & D, uint32_t layer,
                               const float * x) {
    const nano_hparams & h = B.M.h;
    const int32_t m = (int32_t) D.pair_token.size();
    if (m == 0) return true;

    ggml_tensor * w_up;
    ggml_tensor * w_gate;
    ggml_tensor * w_down;
    if (D.holds_all) {
        const nano_layer & L = B.M.layers[layer];
        w_up = L.ffn_up_exps; w_gate = L.ffn_gate_exps; w_down = L.ffn_down_exps;
    } else {
        const device_layer & dl = D.dl[layer];
        w_up = dl.up; w_gate = dl.gate; w_down = dl.down;
    }

    ggml_init_params ip = {
        /*.mem_size   =*/ D.meta.size(),
        /*.mem_buffer =*/ D.meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, h.n_embd, 1, m);
    ggml_set_name(inp, "x");
    ggml_set_input(inp);

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, m);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor * wts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, m);
    ggml_set_name(wts, "weights");
    ggml_set_input(wts);

    ggml_tensor * up   = ggml_mul_mat_id(ctx, w_up,   inp, ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, inp, ids);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * res  = ggml_mul_mat_id(ctx, w_down, act, ids);
    res = ggml_mul(ctx, res, wts);
    ggml_set_output(res);
    ggml_build_forward_expand(gf, res);

    bool ok = ggml_gallocr_alloc_graph(D.galloc, gf);
    if (ok) {
        D.gathered_x.resize((size_t) h.n_embd * m);
        for (int32_t j = 0; j < m; j++) {
            memcpy(&D.gathered_x[(size_t) j * h.n_embd],
                   x + (size_t) D.pair_token[j] * h.n_embd,
                   (size_t) h.n_embd * sizeof(float));
        }
        ggml_backend_tensor_set(inp, D.gathered_x.data(),  0, (size_t) h.n_embd * m * sizeof(float));
        ggml_backend_tensor_set(ids, D.pair_expert.data(), 0, (size_t) m * sizeof(int32_t));
        ggml_backend_tensor_set(wts, D.pair_weight.data(), 0, (size_t) m * sizeof(float));
        ok = ggml_backend_graph_compute(D.backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        D.rows_out.resize((size_t) h.n_embd * m);
        ggml_backend_tensor_get(res, D.rows_out.data(), 0,
                                (size_t) h.n_embd * m * sizeof(float));
    }
    ggml_free(ctx);
    return ok;
}

// Builds and runs one MoE layer: route on the host, evaluate experts on each
// device, combine. The graphs are a few dozen nodes and are rebuilt per
// request; caching them is a live question rather than a settled one now that
// there are N+1 graphs per layer and the GPU has shortened the layer they are
// measured against (OPTIMIZATION.md, "Backend micro-optimisations").
static bool eval_layer(moe_backend & B, uint32_t layer, int32_t n_tokens,
                       const float * x, float * out,
                       uint32_t & t_route_us, uint32_t & t_compute_us) {
    const nano_hparams & h = B.M.h;

    const size_t n_act = (size_t) h.n_embd        * n_tokens;
    const size_t n_sel = (size_t) h.n_expert_used * n_tokens;

    if (B.ids_buf.size()     < n_sel) B.ids_buf.resize(n_sel);
    if (B.weights_buf.size() < n_sel) B.weights_buf.resize(n_sel);
    // Hoisted out of the compute loop below: once that loop is one thread per
    // device, a device must not be reallocating a buffer another thread sizes.
    for (moe_device & D : B.devices) {
        if (D.partial.size() < n_act) D.partial.resize(n_act);
    }

    const auto t_route0 = clk::now();
    if (!run_router(B, layer, n_tokens, x)) return false;
    t_route_us = us_since(t_route0);   // the router itself, not graph construction

    const auto t_compute0 = clk::now();

    // Compare mode: evaluate the reference *first*, on the full CPU path with
    // every pair, and keep it. The split path is measured against it below and
    // then discarded, so the trunk always receives CPU numerics.
    //
    // Discarding the split result is the entire point. If the split answer
    // were fed forward, layer i+1 would start from perturbed activations and
    // its "error" would include everything layers 0..i did — which is exactly
    // the compounding that makes the end-to-end KL saturate and stop
    // distinguishing a wrong kernel from a right one. Handing every layer the
    // same input makes each measurement independent and local.
    if (B.compare) {
        for (moe_device & D : B.devices) { D.pair_token.clear(); }
        B.devices[0].took_all = true;
        if (!run_device_full(B, B.devices[0], layer, n_tokens, x)) return false;
        B.ref_out.assign(B.devices[0].partial.begin(),
                         B.devices[0].partial.begin() + n_act);
    }

    assign_pairs(B, layer, n_tokens);

    // One host thread per device, because ggml graphs are sequential: a single
    // `ggml_backend_graph_compute` is one backend, in order, and
    // `..._async` is only that call minus the synchronize — whether it truly
    // returns early is backend-specific. Several graphs driven by several
    // threads is the formulation that actually overlaps, and the one that
    // generalizes to NUMA nodes rather than only to GPUs.
    //
    // devices[0] runs on *this* thread rather than a spawned one: it is the
    // DRAM device with the largest share, so giving it the caller's thread
    // saves a spawn and a join on the critical path.
    //
    // Everything a device touches during compute is its own — backend, galloc,
    // meta scratch, pair vectors, gathered_x, rows_out — and the two things
    // that are shared, `B.ids_buf`/`B.weights_buf` and the model, are read
    // only. The `partial` resize was hoisted above for exactly this reason.
    {
        const size_t n_dev = B.devices.size();
        auto run_one = [&](size_t d) -> bool {
            moe_device & D = B.devices[d];
            return (D.holds_all && D.took_all)
                 ? run_device_full(B, D, layer, n_tokens, x)
                 : run_device_compact(B, D, layer, x);
        };

        if (n_dev == 1) {
            if (!run_one(0)) return false;
        } else {
            // TODO: create thread pool here
            std::vector<std::thread> workers;
            std::vector<char> ok(n_dev, 1);
            workers.reserve(n_dev - 1);
            for (size_t d = 1; d < n_dev; d++) {
                workers.emplace_back([&, d] { ok[d] = run_one(d) ? 1 : 0; });
            }
            ok[0] = run_one(0) ? 1 : 0;
            for (std::thread & t : workers) t.join();   // join before any early return
            for (size_t d = 0; d < n_dev; d++) if (!ok[d]) return false;
        }
    }

    // Combine in device order — fixed, so the sum is deterministic run to run.
    // That is the property that matters once devices compute in different
    // arithmetic and bit-identity is no longer available.
    //
    // The CPU-only configuration takes the memcpy branch with an empty loop
    // after it, which is bit-for-bit what increment 1 did. Note this is not
    // merely an optimisation of `memset` + accumulate: 0.0f + -0.0f is +0.0f,
    // so zero-then-add would flip the sign of any negative zero and break the
    // byte gate.
    size_t d0 = 0;
    if (B.devices[0].holds_all && B.devices[0].took_all) {
        memcpy(out, B.devices[0].partial.data(), n_act * sizeof(float));
        d0 = 1;
    } else {
        memset(out, 0, n_act * sizeof(float));
    }
    for (size_t d = d0; d < B.devices.size(); d++) {
        moe_device & D = B.devices[d];
        if (D.holds_all && D.took_all) {
            const float * p = D.partial.data();
            for (size_t i = 0; i < n_act; i++) out[i] += p[i];
            continue;
        }
        // Scatter-add: row j of this device's output belongs to token
        // pair_token[j]. Rows for one token arrive in slot order, so a token
        // whose experts all live on one device accumulates in exactly the
        // order build_moe_experts would have used.
        const size_t m = D.pair_token.size();
        for (size_t j = 0; j < m; j++) {
            const float * src = &D.rows_out[j * h.n_embd];
            float * dst = out + (size_t) D.pair_token[j] * h.n_embd;
            for (uint32_t i = 0; i < h.n_embd; i++) dst[i] += src[i];
        }
    }

    if (B.compare) {
        layer_error & e = B.err[layer];
        for (size_t i = 0; i < n_act; i++) {
            const double r = B.ref_out[i];
            const double d = (double) out[i] - r;
            const double a = d < 0 ? -d : d;
            if (a > e.max_abs) e.max_abs = a;
            e.sum_sq += d * d;
            e.ref_sq += r * r;
        }
        e.n += n_act;
        e.n_calls++;
        // Hand back the reference, not the split result — see above.
        memcpy(out, B.ref_out.data(), n_act * sizeof(float));
    }

    t_compute_us = us_since(t_compute0);
    return true;
}

// ---------------------------------------------------------------------------
// request handling

static void send_error(moe_socket c, uint32_t status, const char * msg) {
    moe_response_header rh = {};
    rh.magic         = MOE_MAGIC;
    rh.version       = MOE_VERSION;
    rh.msg_type      = MOE_MSG_RESPONSE;
    rh.status        = status;
    rh.payload_bytes = strlen(msg);
    moe_send_all(c, &rh, sizeof(rh));
    moe_send_all(c, msg, (size_t) rh.payload_bytes);
}

// The v2 handshake, exchanged once before any request. The server answers and
// serves whatever connects: it cannot know whether a client built differently
// is a mistake or the operator deliberately pairing a Q4_K expert store with a
// Q6_K trunk. Enforcement lives on the client, which owns the correctness
// claim; the server's job is to make the pairing visible in its log.
static bool serve_hello(moe_backend & B, moe_socket c, const std::string & model_path) {
    moe_hello_request rq;
    if (!moe_recv_all(c, &rq, sizeof(rq))) return false;

    if (rq.magic != MOE_MAGIC || rq.msg_type != MOE_MSG_HELLO) {
        fprintf(stderr, "moe-server: first message was not a hello (magic %08x, type %u) — "
                        "a v1 client cannot talk to this server\n", rq.magic, rq.msg_type);
        send_error(c, MOE_ERR_VERSION, "expected MOE_MSG_HELLO as the first message");
        return false;
    }
    if (rq.version != MOE_VERSION) {
        fprintf(stderr, "moe-server: client speaks protocol v%u, this is v%u\n",
                rq.version, MOE_VERSION);
        send_error(c, MOE_ERR_VERSION, "protocol version mismatch");
        return false;
    }

    std::string peer(rq.payload_bytes, '\0');
    if (rq.payload_bytes && !moe_recv_all(c, &peer[0], (size_t) rq.payload_bytes)) return false;
    for (const std::string & line : { peer }) {
        size_t pos = 0;
        while (pos < line.size()) {
            size_t nl = line.find('\n', pos);
            if (nl == std::string::npos) nl = line.size();
            if (nl > pos) fprintf(stderr, "moe-server:   client %s\n", line.substr(pos, nl - pos).c_str());
            pos = nl + 1;
        }
    }

    const nano_hparams & h = B.M.h;
    const std::string me = nano_build_info() + nano_run_info(B.n_threads)
                         + nano_model_info(model_path, B.M.bytes_mapped, B.M.n_shards);

    moe_hello_response rh = {};
    rh.magic         = MOE_MAGIC;
    rh.version       = MOE_VERSION;
    rh.msg_type      = MOE_MSG_HELLO;
    rh.status        = MOE_OK;
    snprintf(rh.arch, sizeof(rh.arch), "%s", h.arch.c_str());
    rh.n_embd        = h.n_embd;
    rh.n_layer       = h.n_layer;
    rh.n_dense_lead  = h.n_dense_lead;
    rh.n_expert      = h.n_expert;
    rh.n_expert_used = h.n_expert_used;
    rh.n_ff_exp      = h.n_ff_exp;
    rh.expert_scale  = h.expert_scale;
    rh.expert_norm   = h.expert_norm ? 1u : 0u;
    rh.payload_bytes = me.size();

    return moe_send_all(c, &rh, sizeof(rh)) && moe_send_all(c, me.data(), me.size());
}

// Returns false when the peer disconnects or the stream desyncs.
static bool serve_one(moe_backend & B, moe_socket c, bool verbose) {
    // Deliberately NOT timed: this blocks until the client sends, so it is the
    // client's think time, not the server's cost. Including it makes
    // server_total exceed the client's RTT and drives `RTT - server_total`
    // negative, which is the subtraction PLAN.md uses to isolate network and
    // queueing. The server clock starts once a request header is in hand.
    moe_request_header rq;
    if (!moe_recv_all(c, &rq, sizeof(rq))) return false;

    const auto t_parse0 = clk::now();

    if (rq.magic != MOE_MAGIC || rq.msg_type != MOE_MSG_REQUEST) {
        send_error(c, MOE_ERR_VERSION, "bad magic or message type");
        return false;   // stream position is no longer trustworthy
    }
    if (rq.version != MOE_VERSION) {
        send_error(c, MOE_ERR_VERSION, "protocol version mismatch");
        return false;
    }

    const nano_hparams & h = B.M.h;
    if (rq.n_embd != h.n_embd || rq.n_tokens == 0) {
        send_error(c, MOE_ERR_DIMS, "n_embd mismatch or n_tokens == 0");
        return false;
    }
    if (rq.layer >= h.n_layer || rq.layer < h.n_dense_lead) {
        send_error(c, MOE_ERR_LAYER, "layer out of range or not a MoE layer");
        return false;
    }
    if (rq.return_mode != MOE_RET_COMBINED) {
        send_error(c, MOE_ERR_MODE, "only MOE_RET_COMBINED is implemented");
        return false;
    }
    const size_t want = (size_t) rq.n_embd * rq.n_tokens * sizeof(float);
    if (rq.payload_bytes != want) {
        send_error(c, MOE_ERR_DIMS, "payload_bytes does not match dims");
        return false;
    }

    const size_t n_elem = (size_t) rq.n_embd * rq.n_tokens;
    if (B.in_buf.size()  < n_elem) B.in_buf.resize(n_elem);
    if (B.out_buf.size() < n_elem) B.out_buf.resize(n_elem);

    // Validation only. The payload transfer below is wire time and belongs on
    // the network side of `RTT - server_total`, not in the server's cost.
    const uint32_t t_parse_us = us_since(t_parse0);

    if (!moe_recv_all(c, B.in_buf.data(), want)) return false;

    uint32_t t_route_us = 0, t_compute_us = 0;
    if (!eval_layer(B, rq.layer, (int32_t) rq.n_tokens, B.in_buf.data(), B.out_buf.data(),
                    t_route_us, t_compute_us)) {
        send_error(c, MOE_ERR_INTERNAL, "graph evaluation failed");
        return false;
    }

    const auto t_ser0 = clk::now();
    moe_response_header rh = {};
    rh.magic          = MOE_MAGIC;
    rh.version        = MOE_VERSION;
    rh.msg_type       = MOE_MSG_RESPONSE;
    rh.status         = MOE_OK;
    rh.n_embd         = rq.n_embd;
    rh.n_tokens       = rq.n_tokens;
    rh.n_slots        = 0;
    rh.t_parse_us     = t_parse_us;
    rh.t_route_us     = t_route_us;
    rh.t_compute_us   = t_compute_us;
    rh.t_serialize_us = 0;          // filled just below; measured up to the send
    rh.payload_bytes  = want;
    rh.t_serialize_us = us_since(t_ser0);

    if (!moe_send_all(c, &rh, sizeof(rh)))                 return false;
    if (!moe_send_all(c, B.out_buf.data(), want))          return false;

    if (verbose) {
        fprintf(stderr, "moe-server: layer %3u  n_tokens %4u  route %5u us  compute %7u us\n",
                rq.layer, rq.n_tokens, t_route_us, t_compute_us);
    }
    return true;
}

// What ggml can actually see. Needs no model, which is the point: a 583 GiB
// load is a poor way to find out whether the Vulkan backend registered, and
// choosing which experts a device can hold needs its VRAM figure first.
static void list_devices() {
    ggml_backend_load_all();
    const size_t n = ggml_backend_dev_count();
    printf("%zu ggml device(s)\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t dev_free = 0, dev_total = 0;
        ggml_backend_dev_memory(dev, &dev_free, &dev_total);

        const char * kind = "?";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   kind = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   kind = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  kind = "IGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: kind = "ACCEL"; break;
        }
        printf("  %zu  %-5s %-16s %6.1f / %6.1f GiB free  %s\n",
               i, kind, ggml_backend_dev_name(dev),
               dev_free / 1073741824.0, dev_total / 1073741824.0,
               ggml_backend_dev_description(dev));
    }
}

int main(int argc, char ** argv) {
    // Both work without a model: the gate reads --version, and it is the
    // quickest way to answer "which build is that server?" without touching
    // the socket.
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version")) {
            fputs(nano_build_info().c_str(), stdout);
            fputs(nano_run_info((int) physical_core_count()).c_str(), stdout);
            return 0;
        }
        if (!strcmp(argv[i], "--devices")) {
            list_devices();
            return 0;
        }
    }

    server_params p;
    if (!parse_args(argc, argv, p)) return 1;

    if (!moe_net_init()) {
        fprintf(stderr, "moe-server: socket init failed\n");
        return 1;
    }

    moe_backend B;
    const auto t_load0 = clk::now();
    load_model(B.M, p.model_path);
    B.n_threads = p.n_threads;

    // devices[0] is always the DRAM device: it holds every expert and is the
    // fallthrough for anything no other device has. Optional GPU devices follow
    // it; increment 3 adds the remaining dies, and later a NUMA node is just
    // another entry here.
    if (p.gpu_experts > 0 && p.cpu_experts > 0) {
        fprintf(stderr, "moe-server: --gpu-experts and --cpu-experts are alternatives\n");
        return 1;
    }
    const uint32_t split_experts = p.gpu_experts > 0 ? p.gpu_experts : p.cpu_experts;

    // How many devices the split spreads over. GPUs: as many dies as were
    // asked for and exist. The CPU control stays a single extra device — it is
    // there to isolate the compaction, and more of them would only add thread
    // oversubscription to what it measures.
    std::vector<ggml_backend_dev_t> gpus;
    if (p.gpu_experts > 0) {
        ggml_backend_load_all();
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpus.push_back(dev);
            }
        }
        if (gpus.empty()) {
            fprintf(stderr, "moe-server: --gpu-experts needs a GPU device; this build has "
                            "none (rebuild with build.ps1 -Vk). See --devices.\n");
            return 1;
        }
        if (p.gpu_devices > 0 && p.gpu_devices < gpus.size()) gpus.resize(p.gpu_devices);
    }

    const size_t n_split = split_experts == 0 ? 0 : (p.gpu_experts > 0 ? gpus.size() : 1);
    B.devices.resize(1 + n_split);
    moe_device & D0 = B.devices[0];

    D0.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!D0.backend) { fprintf(stderr, "moe-server: no CPU backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(D0.backend, B.n_threads);
    D0.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(D0.backend));
    D0.holds_all = true;
    D0.dl.resize(B.M.h.n_layer);   // empty local_of everywhere: indexes globally

    // room for the tensor structs each graph needs, plus the graph itself
    D0.meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    B.router_meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    B.router_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(D0.backend));

    if (split_experts > 0) {
        if (split_experts > B.M.h.n_expert) {
            fprintf(stderr, "moe-server: %u experts exceeds n_expert %u\n",
                    split_experts, B.M.h.n_expert);
            return 1;
        }

        // Experts 0..split_experts-1 dealt round-robin over the split devices:
        // device d gets {e < split_experts : e % n_split == d}. Trivial on
        // purpose — the point of this increment is that N devices work at all
        // and overlap, not which experts they should hold.
        for (size_t s = 0; s < n_split; s++) {
            moe_device & D = B.devices[1 + s];

            if (p.gpu_experts > 0) {
                ggml_backend_dev_t gpu = gpus[s];
                D.backend = ggml_backend_dev_init(gpu, nullptr);
                if (!D.backend) { fprintf(stderr, "moe-server: failed to init %s\n",
                                          ggml_backend_dev_name(gpu)); return 1; }
                fprintf(stderr, "moe-server: split device %zu: %s (%s)\n",
                        s, ggml_backend_dev_name(gpu), ggml_backend_dev_description(gpu));
            } else {
                // A second CPU device. Same compaction path, same kernels, same
                // arithmetic — so a difference against the CPU-only run is the
                // compaction's doing and nothing else. That is the control the
                // GPU comparison needs, and the only way to tell a routing or
                // scatter bug from an honest numerics difference.
                D.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
                if (!D.backend) { fprintf(stderr, "moe-server: no second CPU backend\n"); return 1; }
                ggml_backend_cpu_set_n_threads(D.backend, B.n_threads);
                fprintf(stderr, "moe-server: split device CPU (compaction control)\n");
            }

            std::vector<int32_t> mine;
            for (uint32_t e = 0; e < split_experts; e++) {
                if (e % n_split == s) mine.push_back((int32_t) e);
            }

            D.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(D.backend));
            D.meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
            if (!device_load_experts(B, D, mine)) return 1;
        }
    }

    if (!p.force_split.empty()) {
        if (B.devices.size() < 2) {
            fprintf(stderr, "moe-server: --force-split needs split devices "
                            "(--gpu-experts or --cpu-experts)\n");
            return 1;
        }
        uint32_t total = 0;
        for (size_t pos = 0; pos <= p.force_split.size(); ) {
            const size_t c = p.force_split.find(',', pos);
            const std::string tok = p.force_split.substr(pos, c == std::string::npos
                                                              ? std::string::npos : c - pos);
            if (!tok.empty()) { B.force_split.push_back((uint32_t) atoi(tok.c_str())); }
            if (c == std::string::npos) break;
            pos = c + 1;
        }
        for (uint32_t v : B.force_split) total += v;
        if (B.force_split.size() > B.devices.size() - 1) {
            fprintf(stderr, "moe-server: --force-split lists %zu devices, only %zu split "
                            "devices exist\n", B.force_split.size(), B.devices.size() - 1);
            return 1;
        }
        if (total > B.M.h.n_expert_used) {
            fprintf(stderr, "moe-server: --force-split sums to %u, only %u experts are used "
                            "per token\n", total, B.M.h.n_expert_used);
            return 1;
        }
        fprintf(stderr, "moe-server: *** --force-split active: OUTPUT WILL BE WRONG ***\n");
        fprintf(stderr, "moe-server: per token, %u of %u slots forced onto split devices "
                        "(CPU keeps %u); timing only\n",
                total, B.M.h.n_expert_used, B.M.h.n_expert_used - total);
    }

    if (p.compare) {
        if (!p.force_split.empty()) {
            fprintf(stderr, "moe-server: --compare and --force-split are incompatible — "
                            "forced placement deliberately computes the wrong answer, so "
                            "there is nothing meaningful to compare against\n");
            return 1;
        }
        if (B.devices.size() < 2) {
            fprintf(stderr, "moe-server: --compare needs a split device "
                            "(--gpu-experts or --cpu-experts)\n");
            return 1;
        }
        B.compare = true;
        B.err.resize(B.M.h.n_layer);
        fprintf(stderr, "moe-server: compare mode — both paths per layer, "
                        "CPU result returned, ~2x slower\n");
    }

    const nano_hparams & h = B.M.h;
    fprintf(stderr, "moe-server: %s\n", nano_build_line().c_str());
    fprintf(stderr, "moe-server: %s | n_layer=%u (dense lead %u) n_embd=%u n_expert=%u used=%u\n",
            B.M.desc.c_str(), h.n_layer, h.n_dense_lead, h.n_embd, h.n_expert, h.n_expert_used);
    fprintf(stderr, "moe-server: load+init %.1fs, threads %d (%d physical / %d logical)\n",
            std::chrono::duration<double>(clk::now() - t_load0).count(),
            B.n_threads, physical_core_count(), (int) std::thread::hardware_concurrency());

    moe_socket srv = moe_listen(p.host, p.port);
    if (srv == MOE_INVALID_SOCKET) {
        fprintf(stderr, "moe-server: listen on %s:%u failed (%s)\n",
                p.host.c_str(), p.port, moe_net_error().c_str());
        return 1;
    }
    fprintf(stderr, "moe-server: listening on %s:%u\n", p.host.c_str(), p.port);

    // One client at a time: the trunk is strictly sequential (layer i+1 cannot
    // start until layer i returns), so concurrency here would buy nothing.
    for (;;) {
        moe_socket c = accept(srv, nullptr, nullptr);
        if (c == MOE_INVALID_SOCKET) continue;
        moe_set_nodelay(c);
        fprintf(stderr, "moe-server: client connected\n");

        uint64_t n_req = 0;
        const auto t_conn0 = clk::now();
        if (serve_hello(B, c, p.model_path)) {
            while (serve_one(B, c, p.verbose)) n_req++;
        }

        const double secs = std::chrono::duration<double>(clk::now() - t_conn0).count();
        fprintf(stderr, "moe-server: client gone after %" PRIu64 " requests (%.1fs, %.0f req/s)\n",
                n_req, secs, secs > 0 ? n_req / secs : 0.0);

        // How the (token, slot) pairs actually divided. Worth printing rather
        // than deriving from residency: the share depends on how often the
        // router picks a resident expert, which is a property of the prompt,
        // not of k.
        if (B.devices.size() > 1) {
            uint64_t total = 0;
            for (const moe_device & D : B.devices) total += D.n_pairs_total;
            for (size_t d = 0; d < B.devices.size(); d++) {
                const moe_device & D = B.devices[d];
                fprintf(stderr, "moe-server:   device %zu (%s) %" PRIu64 " pairs (%.2f%%)\n",
                        d, D.holds_all ? "DRAM, fallthrough" : "split",
                        D.n_pairs_total, total ? 100.0 * D.n_pairs_total / total : 0.0);
            }
        }
        for (moe_device & D : B.devices) D.n_pairs_total = 0;

        // Per-layer error of the split path against the full CPU path. Layers
        // are independent here — each was handed the same input — so a layer
        // that stands out is a real signal about that layer, not an artifact
        // of everything before it.
        if (B.compare && n_req > 0) {
            double worst = 0.0; uint32_t worst_l = 0;
            double tot_sq = 0.0, tot_ref = 0.0;
            for (uint32_t il = 0; il < B.err.size(); il++) {
                const layer_error & e = B.err[il];
                if (e.n == 0) continue;
                if (e.max_abs > worst) { worst = e.max_abs; worst_l = il; }
                tot_sq += e.sum_sq; tot_ref += e.ref_sq;
            }
            fprintf(stderr, "moe-server: compare — worst layer %u, max |split-cpu| %.3e; "
                            "overall rel RMS %.3e\n",
                    worst_l, worst, tot_ref > 0 ? sqrt(tot_sq / tot_ref) : 0.0);
            fprintf(stderr, "moe-server:   layer  max_abs      rel_rms    calls\n");
            for (uint32_t il = 0; il < B.err.size(); il++) {
                const layer_error & e = B.err[il];
                if (e.n == 0) continue;
                fprintf(stderr, "moe-server:   %5u  %.3e  %.3e  %" PRIu64 "\n",
                        il, e.max_abs,
                        e.ref_sq > 0 ? sqrt(e.sum_sq / e.ref_sq) : 0.0, e.n_calls);
            }
            for (layer_error & e : B.err) e = layer_error();
        }
        moe_close(c);
    }
}
