#pragma once

// The trunk's client for a remote MoE backend: connection, handshake, per-RPC
// statistics, and the ggml custom-op callback that stands in for the routed
// block. Lives in lib/ because it is not specific to one app — any trunk, chat
// or benchmark, wants the same seam.
//
// moe_proto.h must come first here as everywhere: winsock2.h has to precede
// windows.h, which nano_model.h pulls in.
//
// State note: g_moe and g_rpc_ctxs are file-static, so this header is safe
// only while each app is a single translation unit. See lib/README.md before
// adding a second .cpp to any app.

#include "moe_proto.h"

#include "build_info.h"
#include "nano_model.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// routed experts over RPC (PLAN.md step 1)
//
// With --moe-addr, the routed-expert block leaves the process: the callback
// ships the post-norm activation to moe-server, which routes, evaluates the
// selected experts and combines, and ships one row per token back. The shared
// expert stays here on the trunk, so the graph below the callback is
// unchanged. Without the flag, build_moe_block() runs locally exactly as
// before — keeping both paths in one binary is what makes local-vs-remote a
// direct A/B rather than a comparison across builds.
//
// One socket, one thread: the callback returns immediately on every worker but
// ith == 0 (see moe_rpc_cb). The rest then wait out the round trip at the next
// barrier, which is the overlap opportunity OPTIMIZATION.md step 9 picks up
// (moe_send/moe_recv around the shared expert).

struct moe_rpc_record {
    uint32_t layer, n_tokens;
    uint32_t bytes_out, bytes_in;
    uint32_t rtt_us;
    uint32_t srv_parse_us, srv_route_us, srv_compute_us, srv_ser_us;
};

// Always-on counters. A few adds per RPC against milliseconds of expert
// compute, so they cost nothing and are never conditional — a number you only
// collect when you remember to ask for it is a number you do not have when
// something looks wrong.
struct moe_stats {
    uint64_t n_calls   = 0;
    uint64_t bytes_out = 0;   // header + activation, per request
    uint64_t bytes_in  = 0;   // header + combined rows, per response
    uint64_t rtt_us    = 0;   // summed client-side round trip
    uint64_t srv_us    = 0;   // summed server-reported total
    // rtt - srv is network + queueing, by the subtraction PLAN.md relies on
};

struct moe_client {
    moe_socket sock = MOE_INVALID_SOCKET;
    moe_stats  st;

    // Kept always: 4 bytes per call, and percentiles need the distribution,
    // not just the sum. ~76 KB for a 256-token run.
    // TODO: needs to be bounded. Ok to keep as is for now.
    std::vector<uint32_t> rtt_us;

    // Kept only for --moe-log: 36 bytes per call, and a long run makes tens of
    // thousands of them.
    // TODO: eventually limit, rotate and/or flush somewhere.
    bool want_log = false;
    std::vector<moe_rpc_record> log;

    bool active() const { return sock != MOE_INVALID_SOCKET; }
};

static moe_client g_moe;

// One per RPC node in the graph. Stable addresses (deque) because the graph is
// built once and reused; cleared whenever it is rebuilt.
struct moe_rpc_ctx { uint32_t layer; };
static std::deque<moe_rpc_ctx> g_rpc_ctxs;

// TODO: shall this be uint64_t? uint32_t is just a few hours in microseconds.
// The v2 handshake, client side. Three tiers, and the split is the whole point
// (see moe_proto.h): structural disagreement is always fatal because the graph
// assumes those values; reproducibility disagreement is fatal only under
// --strict, because a Q4_K expert store behind a Q6_K trunk is a planned
// configuration and not an error; everything else is printed.
//
// Printed either way, strict or not. The durable half of this is
// observability: a run whose log records what it was actually talking to can
// be diagnosed later, whereas a refusal only helps someone who is watching.
static void moe_hello(const nano_model & M, const std::string & addr, bool strict,
                      const std::string & model_path, int n_threads) {
    const nano_hparams & h = M.h;
    const std::string me = nano_build_info() + nano_run_info(n_threads)
                         + nano_model_info(model_path, M.bytes_mapped, M.n_shards);

    moe_hello_request rq = {};
    rq.magic         = MOE_MAGIC;
    rq.version       = MOE_VERSION;
    rq.msg_type      = MOE_MSG_HELLO;
    rq.payload_bytes = me.size();
    if (!moe_send_all(g_moe.sock, &rq, sizeof(rq)) || !moe_send_all(g_moe.sock, me.data(), me.size())) {
        NANO_ABORT("moe hello: send failed (%s)", moe_net_error().c_str());
    }

    moe_hello_response rh;
    if (!moe_recv_all(g_moe.sock, &rh, sizeof(rh))) {
        NANO_ABORT("moe hello: no reply from %s — a v1 moe-server cannot answer this "
                   "(%s)", addr.c_str(), moe_net_error().c_str());
    }
    if (rh.magic != MOE_MAGIC || rh.msg_type != MOE_MSG_HELLO) {
        NANO_ABORT("moe hello: malformed reply from %s", addr.c_str());
    }
    if (rh.version != MOE_VERSION) {
        NANO_ABORT("moe hello: server speaks protocol v%u, this client is v%u",
                   rh.version, MOE_VERSION);
    }
    std::string peer(rh.payload_bytes, '\0');
    if (rh.payload_bytes && !moe_recv_all(g_moe.sock, &peer[0], (size_t) rh.payload_bytes)) {
        NANO_ABORT("moe hello: short fingerprint from %s", addr.c_str());
    }

    // ---- structural: the client's graph is built around these ----
    std::string bad;
    auto want = [&](const char * name, uint64_t mine, uint64_t theirs) {
        if (mine != theirs) {
            bad += "\n  " + std::string(name) + ": client " + std::to_string(mine)
                 + ", server " + std::to_string(theirs);
        }
    };
    rh.arch[sizeof(rh.arch) - 1] = '\0';
    if (h.arch != rh.arch) bad += "\n  arch: client " + h.arch + ", server " + rh.arch;
    want("n_embd",        h.n_embd,        rh.n_embd);
    want("n_layer",       h.n_layer,       rh.n_layer);
    want("n_dense_lead",  h.n_dense_lead,  rh.n_dense_lead);
    want("n_expert",      h.n_expert,      rh.n_expert);
    want("n_expert_used", h.n_expert_used, rh.n_expert_used);
    want("n_ff_exp",      h.n_ff_exp,      rh.n_ff_exp);
    want("expert_norm",   h.expert_norm ? 1u : 0u, rh.expert_norm);
    if (h.expert_scale != rh.expert_scale) {
        bad += "\n  expert_scale: client " + std::to_string(h.expert_scale)
             + ", server " + std::to_string(rh.expert_scale);
    }
    if (!bad.empty()) {
        // No flag permits this: the two halves are of different models, and
        // the output would be fluent and wrong rather than obviously broken.
        NANO_ABORT("moe hello: %s holds a structurally different model:%s", addr.c_str(), bad.c_str());
    }

    // ---- reproducibility: bit-exactness only ----
    const auto theirs = nano_kv_parse(peer);
    const auto mine   = nano_kv_parse(me);
    std::vector<std::string> drift;
    for (const char * k : NANO_REPRO_KEYS) {
        auto a = mine.find(k), b = theirs.find(k);
        const std::string va = a == mine.end()   ? "?" : a->second;
        const std::string vb = b == theirs.end() ? "?" : b->second;
        if (va != vb) drift.push_back(std::string(k) + ": client " + va + ", server " + vb);
    }

    fprintf(stderr, "nano-glm: moe-server %s | %s\n", addr.c_str(),
            theirs.count("git_rev") ? theirs.at("git_rev").c_str() : "?");
    for (const char * k : { "compiler", "ggml_commit", "n_threads", "model_first", "model_bytes" }) {
        if (theirs.count(k)) fprintf(stderr, "nano-glm:   server %s=%s\n", k, theirs.at(k).c_str());
    }
    for (const std::string & d : drift) {
        fprintf(stderr, "nano-glm:   DIFFERS %s\n", d.c_str());
    }
    if (!drift.empty() && strict) {
        NANO_ABORT("moe hello: --strict and the backend differs in %zu reproducibility "
                   "field(s) above; bit-exactness against a reference is void", drift.size());
    }
    if (!drift.empty()) {
        fprintf(stderr, "nano-glm:   (run with --strict to make this fatal)\n");
    }
}

static uint32_t elapsed_us(std::chrono::steady_clock::time_point t0) {
    const auto d = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count();
    return (uint32_t) (d < 0 ? 0 : d);
}

// ggml's custom-op callback signature (ggml_custom_op_t). What each argument
// is *here*:
//
//   dst      the node built by ggml_custom_4d above — f32 [n_embd, n_tokens],
//            the combined routed-expert output. Writing dst->data is the whole
//            job; whatever this leaves there is what the shared expert gets
//            added to. Its one input is dst->src[0]: the post-ffn_norm
//            activation `cur`, same shape, which is what goes on the wire.
//            ggml has already allocated both; the callback owns neither.
//   ith/nth  this worker's index and the size of the thread pool. NOT the
//            n_tasks=1 passed to ggml_custom_4d — that number only reaches
//            ggml_graph_plan's work-buffer sizing. ggml_compute_forward_custom
//            calls this function from *every* thread with nth = the pool size
//            (16 here), so the early return below is load-bearing, not a
//            formality: without it sixteen threads would race on one socket.
//   userdata the moe_rpc_ctx pushed when this node was built — just the layer
//            index. It has to be a stable address (hence the deque), because
//            ggml stores the pointer in op_params at build time and hands it
//            back at compute time, one graph rebuild later.
//
// Called mid-graph with ggml's barriers on either side: every node before this
// one has finished, none after has started. That ordering is what lets a
// blocking send/recv sit here at all, and it is also why the other fifteen
// threads simply wait out the round trip — the overlap OPTIMIZATION.md step 9 reclaims.
static void moe_rpc_cb(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) return;   // one socket, one owner — see nth above

    const moe_rpc_ctx * c = (const moe_rpc_ctx *) userdata;
    const ggml_tensor * x = dst->src[0];

    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) {
        NANO_ABORT("moe rpc: layer %u tensors must be contiguous", c->layer);
    }

    const uint32_t n_embd   = (uint32_t) dst->ne[0];
    const uint32_t n_tokens = (uint32_t) dst->ne[1];
    const size_t   bytes    = (size_t) n_embd * n_tokens * sizeof(float);

    moe_request_header rq = {};
    rq.magic         = MOE_MAGIC;
    rq.version       = MOE_VERSION;
    rq.msg_type      = MOE_MSG_REQUEST;
    rq.layer         = c->layer;
    rq.n_embd        = n_embd;
    rq.n_tokens      = n_tokens;
    rq.return_mode   = MOE_RET_COMBINED;
    rq.payload_bytes = bytes;

    const auto t0 = std::chrono::steady_clock::now();

    if (!moe_send_all(g_moe.sock, &rq, sizeof(rq)) ||
        !moe_send_all(g_moe.sock, x->data, bytes)) {
        NANO_ABORT("moe rpc: send failed on layer %u (%s)", c->layer, moe_net_error().c_str());
    }

    moe_response_header rh;
    if (!moe_recv_all(g_moe.sock, &rh, sizeof(rh))) {
        NANO_ABORT("moe rpc: no response for layer %u (%s)", c->layer, moe_net_error().c_str());
    }
    if (rh.magic != MOE_MAGIC || rh.msg_type != MOE_MSG_RESPONSE) {
        NANO_ABORT("moe rpc: malformed response header on layer %u", c->layer);
    }
    if (rh.status != MOE_OK) {
        std::string msg(rh.payload_bytes, '\0');
        if (rh.payload_bytes) moe_recv_all(g_moe.sock, &msg[0], (size_t) rh.payload_bytes);
        NANO_ABORT("moe rpc: server error %u on layer %u: %s", rh.status, c->layer, msg.c_str());
    }
    if (rh.n_embd != n_embd || rh.n_tokens != n_tokens || rh.payload_bytes != bytes) {
        NANO_ABORT("moe rpc: response dims mismatch on layer %u", c->layer);
    }
    if (!moe_recv_all(g_moe.sock, dst->data, bytes)) {
        NANO_ABORT("moe rpc: short response payload on layer %u", c->layer);
    }

    const uint32_t rtt = elapsed_us(t0);
    const uint32_t srv = rh.t_parse_us + rh.t_route_us + rh.t_compute_us + rh.t_serialize_us;

    g_moe.st.n_calls   += 1;
    g_moe.st.bytes_out += sizeof(rq) + bytes;
    g_moe.st.bytes_in  += sizeof(rh) + bytes;
    g_moe.st.rtt_us    += rtt;
    g_moe.st.srv_us    += srv;
    g_moe.rtt_us.push_back(rtt);

    if (g_moe.want_log) {
        g_moe.log.push_back({ c->layer, n_tokens,
                              (uint32_t) (sizeof(rq) + bytes), (uint32_t) (sizeof(rh) + bytes),
                              rtt, rh.t_parse_us, rh.t_route_us, rh.t_compute_us, rh.t_serialize_us });
    }
}
