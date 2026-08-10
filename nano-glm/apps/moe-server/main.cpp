// moe-server — the MoE backend for nano-glm (PLAN.md step 1).
//
// Holds the router and every routed expert; the trunk holds everything else.
// One request is one MoE layer for one batch: activation in, combined row out.
// Stateless — no KV, nothing carried between calls — so a restart costs only
// the model load.
//
// Correctness bar is the same as everywhere else here: the graphs it builds
// come from moe_block.h, the identical source the in-process client uses, so a
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
// nano_model.h pulls in.
//
// ---------------------------------------------------------------------------
// Deferred optimisations
//
// None of these are worth doing now — measured against a warm 3072 us/layer
// compute, they are all noise. Each is listed with the condition that would
// make it matter, because that condition is the useful part: the backend gets
// faster in later phases (GPU-resident experts, lower-bit quant) and anything
// currently at 1% becomes 10% once compute drops by 10x.
//
//  - Graph cache keyed on (layer, n_tokens). Building the graphs cost
//    16-43 us/request when it was one ~15-node graph, i.e. 0.5-1.4% today. At
//    ~300 us/layer it would be 5-14%. Cost of doing it: 75 layers x every
//    batch shape of compute buffers held resident, which is why it was not
//    done up front. Re-measure before acting on it: there are two graphs now,
//    and `t_route_us` no longer reports construction time.
//  - Zero-copy request/response. Two 24 KiB memcpys per request (recv into
//    in_buf then tensor_set; tensor_get into out_buf then send). Could recv
//    straight into the allocated input tensor and send straight from the
//    output tensor. Sub-microsecond today; matters only if per-request cost
//    approaches the transfer cost.
//  - Fusing up+gate. They are independent (both read only x) but sit in
//    separate graph nodes with a barrier between them, so their weight
//    streams do not overlap. Fusing would double memory-level parallelism and
//    drop a barrier. Speculative: needs a custom op, and the win is unknown.
//    Worth trying only while we are still at ~57% of theoretical bandwidth.
//  - Large pages for the expert store. Neither Windows nor macOS offers huge
//    pages for file-backed mmap, so this only exists in the non-mmap load
//    path the plan already requires. 4 KiB pages break the L2 streamer about
//    once per output row and cost ~7900 TLB entries per expert.
//  - f16 on the wire. Halves transfer, costs exactness — a phase-3 trade to
//    be measured with compare.py, not assumed.
//
// Not worth doing at all, recorded so it is not re-derived: request
// pipelining. The trunk is strictly sequential (layer i+1 needs layer i's
// output), so there is never more than one request in flight from one
// sequence. Concurrency here only pays if the backend serves several
// independent sequences at once.

#include "moe_proto.h"

#include "build_info.h"
#include "moe_block.h"
#include "nano_model.h"
#include "cpu_topology.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cinttypes>
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
    std::string host      = "127.0.0.1";
    uint16_t    port      = 5711;
    int32_t     n_threads = physical_core_count();
    bool        verbose   = false;
};

static bool parse_args(int argc, char ** argv, server_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m")       && i + 1 < argc) p.model_path = argv[++i];
        else if (!strcmp(a, "--host")   && i + 1 < argc) p.host       = argv[++i];
        else if (!strcmp(a, "--port")   && i + 1 < argc) p.port       = (uint16_t) atoi(argv[++i]);
        else if (!strcmp(a, "-t")       && i + 1 < argc) p.n_threads  = atoi(argv[++i]);
        else if (!strcmp(a, "-v"))                      p.verbose    = true;
        else { p.model_path.clear(); break; }
    }
    if (p.model_path.empty()) {
        fprintf(stderr,
            "Usage: moe-server -m <first-shard.gguf> [options]\n"
            "  --host <ip>   bind address (default 127.0.0.1)\n"
            "  --port <n>    port (default 5711)\n"
            "  -t <int>      threads (default: physical cores, ignoring SMT siblings)\n"
            "  -v            log every request\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// evaluation

// A device is *a backend, the experts it holds, and — from increment 3 — a
// thread to drive it*. Deliberately not GPU-shaped: a node of the 4-socket
// NUMA machine is the same object, a backend with thread affinity and expert
// weights in its own memory, and the partition-then-combine machinery below is
// identical for both. PLAN.md step 3.
struct moe_device {
    ggml_backend_t        backend = nullptr;
    ggml_gallocr_t        galloc  = nullptr;
    std::vector<uint8_t>  meta;      // graph scratch, reused across requests
    std::vector<float>    partial;   // this device's contribution [n_embd, n_tokens]

    // Residency arrives in increment 2. Today device 0 holds every expert.
};

struct moe_backend {
    nano_model              M;
    int                     n_threads = 1;
    std::vector<moe_device> devices;

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

// ---- one device's share of the expert half --------------------------------
//
// `ids` and `weights` come in as ordinary input tensors rather than as the
// router's own nodes. `ggml_mul_mat_id` cannot tell the difference, which is
// what makes the split free.
//
// Increment 1: devices[0] holds every expert, so its share is the whole
// routing decision and this runs exactly the op sequence the single graph ran.
// That is what keeps the gate byte-identical. Increment 2 narrows ids/weights
// to the (token, slot) pairs whose expert is resident here and lets the rest
// fall through to the DRAM device — which is where the interesting problem
// lives, because residency is per *expert* while slots are per *token*.
static bool run_device(moe_backend & B, moe_device & D, uint32_t layer,
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

// Builds and runs one MoE layer: route on the host, evaluate experts on each
// device, combine. Both graphs are a few dozen nodes, so rebuilding them per
// request costs microseconds against a multi-millisecond expert evaluation —
// not worth caching 75 layers x every batch shape of compute buffers.
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

    // Increment 3 replaces this loop with one thread per device and a join.
    // Sequential today because there is exactly one device, and a thread per
    // device is only worth its synchronisation once there are several.
    for (moe_device & D : B.devices) {
        if (!run_device(B, D, layer, n_tokens, x)) return false;
    }

    // Combine in device order. Fixed order, so the sum is deterministic run to
    // run — which is the property that matters once devices compute in
    // different arithmetic and bit-identity is no longer available.
    memcpy(out, B.devices[0].partial.data(), n_act * sizeof(float));
    for (size_t d = 1; d < B.devices.size(); d++) {
        const float * p = B.devices[d].partial.data();
        for (size_t i = 0; i < n_act; i++) out[i] += p[i];
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

int main(int argc, char ** argv) {
    // Works without a model: the gate reads it, and it is the quickest way to
    // answer "which build is that server?" without touching the socket.
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--version")) {
            fputs(nano_build_info().c_str(), stdout);
            fputs(nano_run_info((int) physical_core_count()).c_str(), stdout);
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

    // One device today: the CPU, holding every expert. Increment 2 adds a
    // Vulkan device with a subset of them; increment 3 adds the rest of the
    // dies, and later a NUMA node is just another entry here.
    B.devices.resize(1);
    moe_device & D0 = B.devices[0];

    D0.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!D0.backend) { fprintf(stderr, "moe-server: no CPU backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(D0.backend, B.n_threads);
    D0.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(D0.backend));

    // room for the tensor structs each graph needs, plus the graph itself
    D0.meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    B.router_meta.resize(ggml_tensor_overhead() * 64 + ggml_graph_overhead());
    B.router_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(D0.backend));

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
        moe_close(c);
    }
}
