#pragma once

// DeepSeek-V4-Flash's routed-expert block: router, top-k, expert FFNs,
// weighted combine. Split into a router half and an expert half exactly as
// glm-dsa's is, because moe-server's device machinery (host-side routing,
// per-device compaction, combine) is written against that seam and does not
// care which architecture supplies the halves.
//
// Written out rather than shared with glm-dsa. lib/README.md used to claim
// this file would be a reusable "DeepSeek lineage" tier; it is not. Two
// differences, and both are the kind that produce fluent wrong output:
//
//   gating       sqrt(softplus(x)) here, sigmoid in glm-dsa.
//   clamped FFN  `up` is clamped symmetrically to [-limit, +limit] and `gate`
//                is clamped one-sided to (-inf, +limit] *before* the SiLU.
//                Note the asymmetry: the generic llama.cpp path clamps the
//                activation *after* silu, and deepseek4 takes a different
//                branch (llama-graph.cpp, LLM_FFN_SILU). Clamping in the wrong
//                order is a silent few-percent error, not a crash.
//
// Everything else — the selection bias, the 2^-14 sum clamp, the weight
// normalisation, the expert scale, the pairwise combine — is the same
// sequence, and is written the same way here so the two can be diffed.
//
// Hash-routed layers (0..n_hash_layer-1) pick experts from a token-id lookup
// rather than from the router's ranking. The lookup needs the token ids, which
// the wire protocol does not carry, so those layers stay on the trunk
// (`moe_shape.n_dense_lead`, PLAN.md) — but the block itself handles them: the
// caller passes the already-selected expert ids in and everything downstream,
// including the router matmul that produces the *weights*, is unchanged.

#include "expert_trace.h"
#include "model.h"

#include "ggml.h"

#include <cmath>
#include <vector>

// Same shape as glm-dsa's `moe_routing`, deliberately: the backend reads these
// back and re-uploads them, and one struct layout keeps that code common.
struct ds4_routing {
    ggml_tensor * ids;      // i32 [n_expert_used, n_tokens], router rank order
    ggml_tensor * weights;  // f32 [1, n_expert_used, n_tokens], normalised
};

// Optional naming hook, the same shape as llama.cpp's `llm_graph_context::cb`
// and for the same purpose: label intermediates so a dump can compare them
// (logit-kld's `dump`, nano-glm/dump_inspect.py).
//
// Optional rather than always-on because naming here means `ggml_set_output`,
// and an output is a tensor `ggml_gallocr` may not reuse. moe-server's graph is
// the one the bit-exactness gate measures and it passes nothing, so its
// allocation is exactly what it was. The trunk passes `ds4_name` while the port
// is being checked.
using ds4_cb = void (*)(ggml_tensor *, const char *, int);

static inline void ds4_cb_call(ds4_cb cb, ggml_tensor * t, const char * name, int il) {
    if (cb) {
        cb(t, name, il);
    }
}

// Name a tensor that a later **in-place** op will write through.
//
// `ggml_clamp` returns a view of its input (ggml.c: `ggml_view_tensor(ctx, a)`)
// and writes into it, so the pre-clamp tensor's memory holds post-clamp values
// once the graph has run. llama.cpp never notices: its dump reads through an
// eval callback the moment each node is computed, before the clamp runs. This
// harness reads after the whole graph, so it needs a copy of its own.
//
// It cost four layers to find, because a clamp only changes values past its
// limit: `ffn_moe_gate` agreed exactly through layer 4 and differed in two
// elements of layer 5, both of which were above 10.0.
static inline void ds4_cb_call_pre_inplace(ds4_cb cb, ggml_context * ctx0, ggml_cgraph * gf,
                                           ggml_tensor * t, const char * name, int il) {
    if (!cb) {
        return;
    }
    ggml_tensor * snap = ggml_cont(ctx0, t);
    cb(snap, name, il);
    ggml_build_forward_expand(gf, snap);
}

// x: [n_embd, n_tokens] post-ffn_norm activation.
//
// `selected_experts_in` supplies the expert ids instead of ranking them here.
// Required for a hash-routed layer and unused otherwise. The router matmul runs
// either way: on a hash layer it no longer decides *which* experts, but it
// still produces the probabilities the weights are gathered from.
static ds4_routing build_ds4_moe_router(ggml_context * ctx0, ggml_cgraph * gf,
                                        const ds4_hparams & h, uint32_t il, const ds4_layer & L,
                                        ggml_tensor * x, int32_t n_tokens,
                                        ggml_tensor * selected_experts_in = nullptr,
                                        ds4_cb cb = nullptr) {
    if (h.is_hash_routed(il) && !selected_experts_in) {
        NANO_ABORT("layer %u is hash-routed; its experts come from a token-id "
                   "lookup and cannot be routed from activations alone", il);
    }

    ggml_tensor * logits = ggml_mul_mat(ctx0, L.ffn_gate_inp, x);  // [n_expert, n_tokens]
    // Load-bearing, and not an optimisation: llama.cpp forces F32 accumulation
    // for this matmul under sqrt-softplus gating. The router output feeds an
    // argsort, so a lower-precision sum can flip an expert choice outright.
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    ds4_cb_call(cb, logits, "ffn_moe_logits", (int) il);

    ggml_tensor * probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits));
    ds4_cb_call(cb, probs, "ffn_moe_probs", (int) il);

    ggml_tensor * selected_experts = selected_experts_in;
    if (!selected_experts) {
        // The bias steers *selection* only; the weights come from the unbiased
        // probs below. Same rule as glm-dsa, same reason (DeepSeek-V3's design).
        // A hash-routed layer has no bias tensor at all — the loader only reads
        // one for the layers that rank.
        ggml_tensor * selection_probs = probs;
        if (L.ffn_exp_probs_b) {
            selection_probs = ggml_add(ctx0, probs, L.ffn_exp_probs_b);
            ds4_cb_call(cb, selection_probs, "ffn_moe_probs_biased", (int) il);
        }
        selected_experts = ggml_argsort_top_k(ctx0, selection_probs, h.n_expert_used);
        // The top-k is a *view* of the argsort, so naming only the view would
        // leave its parent freeable (see ds4_snapshot in graph.h). llama.cpp
        // names both, which is also what keeps its own read-back valid.
        ds4_cb_call(cb, selected_experts->src[0], "ffn_moe_argsort", (int) il);
    }
    ds4_cb_call(cb, selected_experts, "ffn_moe_topk", (int) il);

    // No-op unless built with -DNANO_EXPERT_TRACE (expert_trace.h)
    expert_trace_node_add(il, selected_experts);

    probs = ggml_reshape_3d(ctx0, probs, 1, h.n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    ds4_cb_call(cb, weights, "ffn_moe_weights", (int) il);

    if (h.expert_norm) {
        weights = ggml_reshape_2d(ctx0, weights, h.n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        ds4_cb_call_pre_inplace(cb, ctx0, gf, weights_sum, "ffn_moe_weights_sum", (int) il);
        // 2^-14, the smallest positive *normal* fp16, floors the top-k sum so a
        // token whose experts all score ~0 cannot divide to inf. Copied from
        // llama.cpp verbatim and load-bearing for bit-exactness.
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        ds4_cb_call(cb, weights_sum, "ffn_moe_weights_sum_clamped", (int) il);
        weights = ggml_div(ctx0, weights, weights_sum);
        ds4_cb_call(cb, weights, "ffn_moe_weights_norm", (int) il);
        weights = ggml_reshape_3d(ctx0, weights, 1, h.n_expert_used, n_tokens);
    }
    if (h.expert_scale != 0.0f && h.expert_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, h.expert_scale);
        ds4_cb_call(cb, weights, "ffn_moe_weights_scaled", (int) il);
    }

    ggml_build_forward_expand(gf, weights);

    return { selected_experts, weights };
}

// The expert half: given the router's decision, evaluate and combine.
//
// `limit` is the per-layer SwiGLU clamp (`swiglu_clamp_exp[il]`, 10.0
// throughout this checkpoint). Passed rather than looked up so the backend can
// call this with a layer's weights it holds and a limit it was told.
//
// returns [n_embd, n_tokens] — routed experts only, weighted and summed. The
// shared expert is NOT included: it stays on the trunk host.
static ggml_tensor * build_ds4_moe_experts(ggml_context * ctx0, ggml_cgraph * gf,
                                           const ds4_hparams & h, const ds4_layer & L,
                                           ggml_tensor * x, const ds4_routing & r,
                                           float limit, int32_t n_tokens,
                                           ds4_cb cb = nullptr, int il = -1) {
    ggml_tensor * selected_experts = r.ids;
    ggml_tensor * weights          = r.weights;

    ggml_tensor * moe_x = ggml_reshape_3d(ctx0, x, h.n_embd, 1, n_tokens);

    ggml_tensor * up   = ggml_mul_mat_id(ctx0, L.ffn_up_exps,   moe_x, selected_experts);
    ds4_cb_call_pre_inplace(cb, ctx0, gf, up, "ffn_moe_up", il);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, L.ffn_gate_exps, moe_x, selected_experts);
    ds4_cb_call_pre_inplace(cb, ctx0, gf, gate, "ffn_moe_gate", il);

    // The clamp, in deepseek4's order: `up` symmetric, `gate` one-sided and
    // *before* SiLU (ggml_swiglu_split applies silu to its first argument).
    if (limit > 1e-6f) {
        up   = ggml_clamp(ctx0, up,   -limit,    limit);
        ds4_cb_call(cb, up, "ffn_moe_up_clamped", il);
        gate = ggml_clamp(ctx0, gate, -INFINITY, limit);
        ds4_cb_call(cb, gate, "ffn_moe_gate_clamped", il);
    }

    ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
    ds4_cb_call(cb, act, limit > 1e-6f ? "ffn_moe_swiglu_limited" : "ffn_moe_swiglu", il);

    ggml_tensor * experts = ggml_mul_mat_id(ctx0, L.ffn_down_exps, act, selected_experts);
    ds4_cb_call(cb, experts, "ffn_moe_down", il);
    experts = ggml_mul(ctx0, experts, weights);
    ds4_cb_call(cb, experts, "ffn_moe_weighted", il);

    ggml_build_forward_expand(gf, experts);

    // aggregate: one 2D view per used expert, summed pairwise (llama.cpp order)
    std::vector<ggml_tensor *> cur_experts(h.n_expert_used);
    for (uint32_t i = 0; i < h.n_expert_used; i++) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, h.n_embd, n_tokens,
                                      experts->nb[2], i * experts->nb[1]);
        ggml_build_forward_expand(gf, cur_experts[i]);
    }
    ggml_tensor * moe_out = cur_experts[0];
    for (uint32_t i = 1; i < h.n_expert_used; i++) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
        ggml_build_forward_expand(gf, moe_out);
    }
    return moe_out;
}

// Route and evaluate in one graph, for a caller that wants the whole block.
// Kept as a composition of the halves rather than a third implementation, so
// there is exactly one definition of the op sequence.
static ggml_tensor * build_ds4_moe_block(ggml_context * ctx0, ggml_cgraph * gf,
                                         const ds4_hparams & h, uint32_t il, const ds4_layer & L,
                                         ggml_tensor * x, int32_t n_tokens,
                                         ggml_tensor * selected_experts_in = nullptr,
                                         ds4_cb cb = nullptr) {
    const ds4_routing r = build_ds4_moe_router(ctx0, gf, h, il, L, x, n_tokens,
                                               selected_experts_in, cb);
    return build_ds4_moe_experts(ctx0, gf, h, L, x, r, h.swiglu_clamp_exp[il], n_tokens,
                                 cb, (int) il);
}
