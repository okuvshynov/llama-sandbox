#pragma once

// The routed-expert block: router, top-k selection, expert FFNs, weighted
// combine. This is the unit that moves across the wire — the backend evaluates
// it, the trunk host does not. Shared so client and server build the *same*
// graph from the same source; a divergence here is a silently wrong answer,
// not a failure.
//
// Deliberately plain ggml ops. Step 0 (commit 29ea296) proved a host-side
// dispatch callback can reproduce `ggml_mul_mat_id` byte for byte, which is
// what established that the boundary can sit here at all — but with one
// backend holding every expert there is nothing for the client to dispatch,
// and `mul_mat_id` *is* the bit-exact baseline. The dispatch code returns in
// step 3, inside the backend, to choose between GPU-resident and DRAM experts.
//
// Op order is load-bearing: it mirrors llama.cpp's build_moe_ffn, including
// the seemingly redundant reshapes and the pairwise combine. Do not tidy it.

#include "nano_model.h"

#include "ggml.h"

#include <cmath>
#include <vector>

// x: [n_embd, n_tokens] post-ffn_norm activation.
// returns moe_out: [n_embd, n_tokens] — routed experts only, weighted and
// summed. The shared expert is NOT included: it stays on the trunk host.
static ggml_tensor * build_moe_block(ggml_context * ctx0, ggml_cgraph * gf,
                                     const nano_hparams & h, const nano_layer & L,
                                     ggml_tensor * x, int32_t n_tokens) {
    // ---- build_moe_ffn: sigmoid gating + selection bias, no groups ----
    ggml_tensor * logits = ggml_mul_mat(ctx0, L.ffn_gate_inp, x); // [n_expert, n_tokens]
    ggml_tensor * probs  = ggml_sigmoid(ctx0, logits);

    ggml_tensor * selection_probs = probs;
    if (L.ffn_exp_probs_b) {
        selection_probs = ggml_add(ctx0, probs, L.ffn_exp_probs_b);
    }

    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, h.n_expert_used);

    probs = ggml_reshape_3d(ctx0, probs, 1, h.n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]

    if (h.expert_norm) {
        weights = ggml_reshape_2d(ctx0, weights, h.n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        weights = ggml_div(ctx0, weights, weights_sum);
        weights = ggml_reshape_3d(ctx0, weights, 1, h.n_expert_used, n_tokens);
    }
    if (h.expert_scale != 0.0f && h.expert_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, h.expert_scale);
    }

    ggml_build_forward_expand(gf, weights);

    ggml_tensor * moe_x = ggml_reshape_3d(ctx0, x, h.n_embd, 1, n_tokens);

    ggml_tensor * up   = ggml_mul_mat_id(ctx0, L.ffn_up_exps,   moe_x, selected_experts);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, L.ffn_gate_exps, moe_x, selected_experts);
    ggml_tensor * act  = ggml_swiglu_split(ctx0, gate, up);

    ggml_tensor * experts = ggml_mul_mat_id(ctx0, L.ffn_down_exps, act, selected_experts);
    experts = ggml_mul(ctx0, experts, weights);

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
