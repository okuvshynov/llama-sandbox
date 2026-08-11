#pragma once

// Routing trace: for every position and every MoE layer, the ids of the
// experts the router selected. This is the raw material for the questions
// PLAN.md steps 3-4 turn on — how skewed is expert usage, does a static
// resident subset pay, is there enough token-to-token locality for prefetch —
// none of which can be answered from timings.
//
// **Compile-time gated.** Without -DNANO_EXPERT_TRACE the calls below expand
// to nothing, so the shipped binary carries no branch, no allocation and no
// changed graph. That matters more than the cost would suggest: the trace has
// to keep an intermediate tensor alive (see expert_trace_node), which is a
// change to how the graph is allocated, and the bit-exactness gate is the one
// property this project cannot spend. Build the traced variant separately:
//
//     .\build.ps1 -Trace        # -> build-trace\bin, -DNANO_EXPERT_TRACE=ON
//     nano-glm ... --expert-log run.trace
//
// Even in a traced build nothing is written unless --expert-log is passed.
//
// File format (v1, text, position-major):
//
//     # comment lines, key=value metadata
//     p <pos> <token_id>
//     l <layer> <id,id,...>      x one per MoE layer, ascending
//
// Read it with expert_stats.py. Text rather than binary because it is a
// study artefact, not a hot path: ~2.4 KB per position, and being able to
// grep it is worth more than the bytes.

#include "gguf_store.h"
#include "moe_shape.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(NANO_EXPERT_TRACE)

struct expert_trace_node {
    uint32_t      layer;
    ggml_tensor * argsort;   // [n_expert, n_tokens] i32, full descending order
};

struct expert_trace_state {
    bool     enabled  = false;
    FILE *   f        = nullptr;
    uint32_t n_expert = 0;
    uint32_t n_used   = 0;
    uint64_t n_pos    = 0;

    std::vector<expert_trace_node> nodes;   // one per MoE layer of the current graph
    std::vector<int32_t> scratch;           // one node's read-back
    std::vector<int32_t> compact;           // [node][token][n_used]
};

static expert_trace_state g_expert_trace;

static void expert_trace_open(const std::string & path, const moe_shape & h,
                              const std::string & model_desc, int32_t n_prompt) {
    FILE * f = fopen(path.c_str(), "w");
    if (!f) NANO_ABORT("cannot write expert trace '%s'", path.c_str());

    fprintf(f, "# nano-glm expert-trace v1\n");
    fprintf(f, "# n_layer=%u n_dense_lead=%u n_expert=%u n_expert_used=%u n_prompt=%d\n",
            h.n_layer, h.n_dense_lead, h.n_expert, h.n_expert_used, n_prompt);
    fprintf(f, "# model=%s\n", model_desc.c_str());
    fprintf(f, "# p <pos> <token_id>\n");
    fprintf(f, "# l <layer> <expert ids, router rank order: highest biased score first>\n");

    g_expert_trace.enabled  = true;
    g_expert_trace.f        = f;
    g_expert_trace.n_expert = h.n_expert;
    g_expert_trace.n_used   = h.n_expert_used;
}

// Called when a graph is discarded: its tensors go with it.
static void expert_trace_reset() {
    g_expert_trace.nodes.clear();
}

// Called from build_moe_block with the top-k selection tensor.
//
// That tensor is a *view* of the full argsort output (ggml_argsort_top_k is
// argsort-descending plus a view of the first k columns), so the trace keeps
// the argsort itself: it is contiguous, and its first n_used entries per row
// are exactly the view. Marking it an output is what makes it readable after
// ggml_backend_graph_compute — ggml_gallocr recycles an intermediate's memory
// as soon as its last consumer has run, and here that memory would be handed
// to a later layer. Outputs are never freed (ggml-alloc.c, ggml_gallocr_free_node).
//
// The flag changes allocation only. It adds no node, and ggml_backend_sched
// does not look at it, so backend assignment — and therefore the numerics —
// is unchanged. Verified by byte-comparing a traced against an untraced run.
static void expert_trace_node_add(uint32_t layer, ggml_tensor * selected) {
    if (!g_expert_trace.enabled) return;

    ggml_tensor * src = selected->view_src ? selected->view_src : selected;
    if ((uint32_t) src->ne[0] != g_expert_trace.n_expert || src->type != GGML_TYPE_I32) {
        NANO_ABORT("expert trace: layer %u selection tensor is %" PRId64 " x %s, expected %u x i32",
                   layer, src->ne[0], ggml_type_name(src->type), g_expert_trace.n_expert);
    }
    ggml_set_output(src);
    g_expert_trace.nodes.push_back({ layer, src });
}

// Called after the graph has been computed, with the batch it was computed for.
static void expert_trace_flush(int32_t pos_base, int32_t n_tokens, const int32_t * tokens) {
    expert_trace_state & T = g_expert_trace;
    if (!T.enabled || T.nodes.empty()) return;

    const size_t ne = T.n_expert;
    const size_t k  = T.n_used;

    T.scratch.resize(ne * (size_t) n_tokens);
    T.compact.resize(T.nodes.size() * (size_t) n_tokens * k);

    for (size_t n = 0; n < T.nodes.size(); n++) {
        ggml_tensor * t = T.nodes[n].argsort;
        if (t->ne[1] != n_tokens) {
            NANO_ABORT("expert trace: layer %u has %" PRId64 " rows, batch is %d",
                       T.nodes[n].layer, t->ne[1], n_tokens);
        }
        ggml_backend_tensor_get(t, T.scratch.data(), 0, T.scratch.size() * sizeof(int32_t));
        for (int32_t i = 0; i < n_tokens; i++) {
            memcpy(&T.compact[(n * (size_t) n_tokens + i) * k], &T.scratch[i * ne], k * sizeof(int32_t));
        }
    }

    for (int32_t i = 0; i < n_tokens; i++) {
        fprintf(T.f, "p %d %d\n", pos_base + i, tokens[i]);
        for (size_t n = 0; n < T.nodes.size(); n++) {
            fprintf(T.f, "l %u", T.nodes[n].layer);
            const int32_t * ids = &T.compact[(n * (size_t) n_tokens + i) * k];
            for (size_t j = 0; j < k; j++) fprintf(T.f, "%c%d", j ? ',' : ' ', ids[j]);
            fputc('\n', T.f);
        }
    }
    T.n_pos += (uint64_t) n_tokens;
}

static void expert_trace_close() {
    if (!g_expert_trace.enabled) return;
    fclose(g_expert_trace.f);
    g_expert_trace.f       = nullptr;
    g_expert_trace.enabled = false;
}

static uint64_t expert_trace_n_pos() { return g_expert_trace.n_pos; }
static bool     expert_trace_built() { return true; }

#else  // !NANO_EXPERT_TRACE — every hook compiles away

static inline void expert_trace_open(const std::string &, const moe_shape &,
                                     const std::string &, int32_t) {}
static inline void expert_trace_reset() {}
static inline void expert_trace_node_add(uint32_t, ggml_tensor *) {}
static inline void expert_trace_flush(int32_t, int32_t, const int32_t *) {}
static inline void expert_trace_close() {}
static inline uint64_t expert_trace_n_pos() { return 0; }
static inline bool     expert_trace_built() { return false; }

#endif
