#pragma once

// The op sequence a device runs when it holds only *some* of the experts.
//
// One definition, for a reason that was paid for rather than assumed. This
// sequence existed twice — once in `moe-server`'s `run_device_compact` and once,
// with the clamp, in `build_ds4_moe_experts` — and the two silently disagreed:
// deepseek4 clamps its SwiGLU and the compacted path did not, so every expert a
// split device evaluated came back wrong by 5.6% RMS for four commits. Then
// `moe-bench` copied the sequence a third time to measure it. Three hand-synced
// copies of five ops is how that bug happens again, so there is now one.
//
// **This is not a claim that MoE architectures share arithmetic.** `lib/README.md`
// records that the "family tier" was wishful — deepseek4 gates with
// sqrt(softplus(x)) where glm-dsa uses sigmoid, and each writes its own expert
// graph. What is shared is narrower and real: *after* routing has happened and a
// device has been handed a flat list of (token, expert) pairs, both models do
// the same five ops on them. The clamp is the one place they differ, and it is a
// parameter here rather than an assumption, which is exactly the shape the bug
// argued for.
//
// The compaction itself — which pairs go where, gathering the activation rows,
// scattering the results back — stays in the backend. That is host bookkeeping
// over positions and device ids, and it has no ggml in it.

#include "ggml.h"

#include <cmath>

// The tensors a caller needs to fill and read. Inputs are created here because
// their shapes are part of the sequence: `[n_embd, 1, m]` with a `[1, m]` id
// tensor is what makes `mul_mat_id` see one expert per column, and a caller that
// built them differently would be building a different graph.
struct moe_compact_io {
    ggml_tensor * x;    // f32 [n_embd, 1, m] — one gathered activation per pair
    ggml_tensor * ids;  // i32 [1, m]         — the expert each column wants
    ggml_tensor * wts;  // f32 [1, 1, m]      — the router weight for that pair
    ggml_tensor * out;  // f32 [n_embd, 1, m] — weighted rows, to be scattered
};

// `swiglu_limit` is the architecture's SwiGLU clamp, or 0 for one that does not
// clamp. deepseek4 passes `swiglu_clamp_exp[il]`, glm-dsa passes 0. The
// `> 1e-6f` test is shared with `build_ds4_moe_experts` so the two agree by
// construction rather than by coincidence.
//
// Note the per-token sum is *not* here. `build_ds4_moe_experts` finishes with
// pairwise adds because it has all of a token's experts; a device may hold only
// some, so the sum happens host-side in the scatter-add. That is a real
// difference between the two paths and the reason a split run is not bit-exact
// with a local one even when correct — summation order differs. See
// `TESTING.md` on reading `--compare`.
static moe_compact_io build_moe_compact(ggml_context * ctx, ggml_cgraph * gf,
                                        ggml_tensor * w_up, ggml_tensor * w_gate,
                                        ggml_tensor * w_down,
                                        int64_t n_embd, int64_t m, float swiglu_limit) {
    moe_compact_io io = {};

    io.x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, m);
    ggml_set_name(io.x, "x");
    ggml_set_input(io.x);

    io.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, m);
    ggml_set_name(io.ids, "ids");
    ggml_set_input(io.ids);

    io.wts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, m);
    ggml_set_name(io.wts, "weights");
    ggml_set_input(io.wts);

    ggml_tensor * up   = ggml_mul_mat_id(ctx, w_up,   io.x, io.ids);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, w_gate, io.x, io.ids);

    // deepseek4's order: `up` symmetric, `gate` one-sided and *before* SiLU,
    // which `ggml_swiglu_split` applies to its first argument. `ggml_clamp`
    // writes through a view of its input rather than allocating (repo
    // CLAUDE.md), which is safe here because nothing else reads `up` or `gate`.
    if (swiglu_limit > 1e-6f) {
        up   = ggml_clamp(ctx, up,   -swiglu_limit, swiglu_limit);
        gate = ggml_clamp(ctx, gate, -INFINITY,     swiglu_limit);
    }

    ggml_tensor * act = ggml_swiglu_split(ctx, gate, up);
    io.out = ggml_mul_mat_id(ctx, w_down, act, io.ids);
    io.out = ggml_mul(ctx, io.out, io.wts);
    ggml_set_output(io.out);
    ggml_build_forward_expand(gf, io.out);

    return io;
}
