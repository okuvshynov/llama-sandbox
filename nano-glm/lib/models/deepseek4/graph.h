#pragma once

// DeepSeek-V4-Flash trunk graph. **Under construction** — this builds as far
// as the port has been verified against llama.cpp, and aborts past that. The
// point of stopping rather than stubbing is that a stub produces numbers, and
// numbers that mean nothing are worse than an error.
//
// Ported one tensor at a time against `logit-kld/src/dump.cpp`'s capture of
// llama.cpp's own intermediates (see nano-glm/dump_inspect.py). The reason for
// that discipline rather than "write it all and compare the logits" is in
// OPTIMIZATION.md: end-to-end logit KL saturates on a deep model, so it cannot
// distinguish a subtly wrong kernel from a correct one.
//
// Progress, in the order llama.cpp emits the tensors for a layer:
//
//   hc_mixes       bit-identical
//   hc_pre         bit-identical
//   hc_post/comb   bit-identical
//   hc_attn_pre    bit-identical
//   norm           bit-identical
//   attn_norm      bit-identical
//   qr, qr_norm    bit-identical
//   q_norm, q_pe, q          bit-identical
//   kv_norm, kv_pe, kv       bit-identical
//   attn_raw, attn_derope, attn_wo_a, attn_out   bit-identical
//   hc_attn_post   bit-identical
//   hc_ffn_pre, ffn_norm                         bit-identical
//   ffn_moe_* (router, hash routing, experts)    bit-identical
//   ffn_gate/up/swiglu, ffn_shexp, ffn_out       bit-identical
//   l_last         bit-identical
//
// Layer 0 is therefore complete: 53/53 tensors at 0.0000e+00. Layer 1 shares
// this shape (compress_ratio 0, hash-routed); the compressor and indexer layers
// do not.
//
// The attention core cost one wrong assumption, worth recording because the
// shape of the mistake generalises. It was off by 4.0e-03 — ~F16 epsilon
// relative — with every tensor feeding it exact, which reads like a precision
// difference and is not: the port applied a Hadamard rotation to q and k that
// llama.cpp does not apply to this cache (see ds4_build_attention_raw). The
// rotation is orthonormal and self-inverse, so it changes no mathematics and
// only moves rounding; a wrong graph that is mathematically right hides in
// exactly the band you would write off as precision. Two precision hypotheses
// were tried and rejected before that, both correctly.
//
// What found it: dumping the reference's **unnamed** graph nodes. llama.cpp
// names some of what it builds and the difference sat between two named
// tensors, where a name-filtered dump cannot look. `dump --max-records 150`
// (logit-kld) writes every node in graph order, named or not, and the node
// feeding flash attention turned out to be "q-0 (view) (permuted)" — a view of
// `q` with no matmul in between.
//
// The FFN half cost one more, and it was not in the graph at all: llama.cpp
// **repacks** MXFP4 expert weights into `mxfp4_8x8` at load and runs a
// different GEMM against them, which put ~1e-6 on every routed-expert matmul
// while the router, the shared expert and every plain matmul stayed exact.
// nano-glm mmaps weights as they sit in the file and cannot follow, so the
// reference is now built with `GGML_CPU_REPACK OFF` (logit-kld/CMakeLists.txt,
// which explains why this never surfaced with GLM-5.2). The tell was the
// *pattern*: exactly the `mul_mat_id` tensors wrong and nothing else.
//
// Shapes for this checkpoint, since they make the code readable:
//   n_embd 4096, hc 4, hc_dim 16384, hc_mix_dim (2+hc)*hc = 24
//   n_head 64, d_key 512, rope_dim 64  => nope 448
//   o_groups 8, o_lora_rank 1024

#include "model.h"
#include "moe_block.h"

#include "ggml.h"

#include <cmath>
#include <vector>

// llama.cpp names its intermediates through a callback; we name the tensors
// themselves, so a dump can match by name. Same strings, deliberately — the
// comparison is by name and a rename is a silent mismatch.
// Matches llama_context::graph_get_cb exactly, including that a negative layer
// index gets *no* suffix. A rename here is a silent mismatch in the comparison,
// so this convention is copied rather than invented.
static void ds4_name(ggml_tensor * t, const char * base, int il) {
    if (il >= 0) {
        char buf[GGML_MAX_NAME];
        snprintf(buf, sizeof(buf), "%s-%d", base, il);
        ggml_set_name(t, buf);
    } else {
        ggml_set_name(t, base);
    }
    ggml_set_output(t);
}

// Name a tensor for the dump *safely*, copying it first if it is a view.
//
// A view marked GGML_TENSOR_FLAG_OUTPUT is not readable after the graph runs.
// `ggml_gallocr_alloc_graph_impl` frees a view's **parent** as soon as the view
// has no children left, and the OUTPUT flag it tests belongs to the node being
// freed — the view's flag protects nothing (ggml-alloc.c, and the same trap
// `run_router` hit in moe-server). A later op is then handed those bytes and
// the dump reads whatever the op left behind. Copying gives the dump storage of
// its own; the graph downstream keeps using the original view, so nothing about
// the arithmetic changes.
static void ds4_snapshot(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * t,
                         const char * base, int il) {
    ggml_tensor * s = t->view_src ? ggml_cont(ctx, t) : t;
    ds4_name(s, base, il);
    ggml_build_forward_expand(gf, s);
}

// ---------------------------------------------------------------------------
// hyper-connections
//
// The residual stream is `hc` parallel copies rather than one. Each layer
// reads a learned mixture of them (`pre`), writes its output back scaled by
// `post`, and additionally mixes the streams among themselves by a
// Sinkhorn-normalised `comb` matrix. So the network's spine differs from a
// conventional transformer, not just an op inside it.
//
// A useful invariant while porting: `comb` is doubly stochastic, so a
// [hc, hc, n_tokens] tensor sums to hc*n_tokens. llama.cpp's layer 0 gives
// 19.99998 for hc=4, nt=5.

// Weighted sum of the hc streams: sum_ih x[:, ih, :] * w[ih, :].
static ggml_tensor * ds4_hc_mix(ggml_context * ctx, const ds4_hparams & h,
                                ggml_tensor * x, ggml_tensor * w) {
    const int64_t nt = x->ne[2];
    ggml_tensor * acc = nullptr;
    for (uint32_t ih = 0; ih < h.hc_mult; ih++) {
        ggml_tensor * xh = ggml_view_2d(ctx, x, h.n_embd, nt, x->nb[2], ih * x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx, w, 1, nt, w->nb[1], ih * w->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx, xh, wh);
        acc = acc ? ggml_add(ctx, acc, cur) : cur;
    }
    return acc;
}

// `x * scale + base`, where scale is a single value and base is per-row.
// llama.cpp's `dsv4_hc_affine`.
static ggml_tensor * ds4_hc_affine(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * scale, ggml_tensor * base) {
    // scale is [1]; ggml_mul broadcasts it over the whole tensor.
    ggml_tensor * out = ggml_mul(ctx, x, scale);
    return ggml_add(ctx, out, base);
}

static ggml_tensor * ds4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t n, int64_t off) {
    return ggml_view_1d(ctx, t, n, off * ggml_element_size(t));
}

// A [n, nt] window into a [rows, nt] tensor, starting at row `off`.
static ggml_tensor * ds4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t n, int64_t nt, int64_t off) {
    return ggml_view_2d(ctx, t, n, nt, t->nb[1], off * ggml_element_size(t));
}

// Sinkhorn: row softmax, then one column normalisation, then alternating
// row/column normalisations. Follows llama.cpp exactly, including that the
// first pass is columns-only and the loop starts at 1.
static ggml_tensor * ds4_hc_sinkhorn(ggml_context * ctx, const ds4_hparams & h, ggml_tensor * comb) {
    comb = ggml_soft_max(ctx, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, h.hc_eps);

    comb = ggml_add(ctx, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx, t);
        col_sum = ggml_add(ctx, col_sum, eps);
        col_sum = ggml_permute(ctx, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx, comb, col_sum);
    };
    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx, comb);
        row_sum = ggml_add(ctx, row_sum, eps);
        comb = ggml_div(ctx, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < h.hc_sinkhorn_iters; i++) {
        norm_rows();
        norm_cols();
    }
    return comb;
}

// The three mixtures a layer needs, from one matmul against `hc_fn`.
// `mixes` is [24, nt] for hc=4: rows 0..3 pre, 4..7 post, 8..23 comb.
static ggml_tensor * ds4_hc_pre(ggml_context * ctx, ggml_cgraph * gf, const ds4_hparams & h,
                                ggml_tensor * x, ggml_tensor * hc_fn, ggml_tensor * hc_scale,
                                ggml_tensor * hc_base, ggml_tensor ** post, ggml_tensor ** comb,
                                int il) {
    const int64_t hc  = h.hc_mult;
    const int64_t nt  = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc * h.n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, h.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);
    ds4_name(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = ds4_view_1d(ctx, hc_scale, 1, 0);
    ggml_tensor * scale_post = ds4_view_1d(ctx, hc_scale, 1, 1);
    ggml_tensor * scale_comb = ds4_view_1d(ctx, hc_scale, 1, 2);

    ggml_tensor * base_pre  = ds4_view_1d(ctx, hc_base, hc,      0);
    ggml_tensor * base_post = ds4_view_1d(ctx, hc_base, hc,      hc);
    ggml_tensor * base_comb = ds4_view_1d(ctx, hc_base, hc * hc, 2 * hc);

    ggml_tensor * pre = ds4_view_2d(ctx, mixes, hc, nt, 0);
    pre = ds4_hc_affine(ctx, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_scale_bias(ctx, pre, 1.0f, h.hc_eps);
    ds4_name(pre, "hc_pre", il);

    *post = ds4_view_2d(ctx, mixes, hc, nt, hc);
    *post = ds4_hc_affine(ctx, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx, *post);
    *post = ggml_scale(ctx, *post, 2.0f);
    ds4_name(*post, "hc_post", il);

    // The fused op, not the explicit sequence above it — and the choice is
    // measured, not stylistic. llama.cpp resolves `fused DeepSeek V4 HC comb`
    // at startup and there is no public way to turn that off, so the reference
    // is always fused. Our unfused Sinkhorn (`ds4_hc_sinkhorn`, kept for the
    // math it documents) agrees to 1.19e-07 absolute — one ULP-ish, accumulated
    // over 20 normalisation passes — and every other tensor in this layer is
    // bit-identical. Using the same op removes the only difference.
    //
    // `ggml_dsv4_hc_comb` takes the raw mixes plus the whole scale/base
    // tensors and does the affine, the softmax and the Sinkhorn itself.
    *comb = ggml_dsv4_hc_comb(ctx, mixes, hc_scale, hc_base, h.hc_eps,
                              (int32_t) h.hc_sinkhorn_iters);
    ds4_name(*comb, "hc_comb", il);
    (void) scale_comb;
    (void) base_comb;

    ggml_build_forward_expand(gf, *post);
    ggml_build_forward_expand(gf, *comb);

    return ds4_hc_mix(ctx, h, x, pre);
}

// Write a layer's output back into the hc streams: each destination stream is
// the layer output scaled by its `post` weight, plus a comb-weighted mixture of
// the incoming streams.
// The name is the caller's because llama.cpp's differs by half: the attention
// half's result is "hc_attn_post", the FFN half's is "l_last".
static ggml_tensor * ds4_hc_post(ggml_context * ctx, const ds4_hparams & h,
                                 ggml_tensor * x, ggml_tensor * residual,
                                 ggml_tensor * post, ggml_tensor * comb,
                                 const char * name, int il) {
    const int64_t hc = h.hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; dst++) {
        ggml_tensor * post_dst = ggml_view_2d(ctx, post, 1, nt, post->nb[1], dst * post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx, x, post_dst);

        for (int64_t src = 0; src < hc; src++) {
            ggml_tensor * res_src = ggml_view_2d(ctx, residual, h.n_embd, nt,
                                                 residual->nb[2], src * residual->nb[1]);
            ggml_tensor * c = ggml_view_2d(ctx, comb, 1, nt, comb->nb[2],
                                           dst * comb->nb[0] + src * comb->nb[1]);
            cur = ggml_add(ctx, cur, ggml_mul(ctx, res_src, c));
        }
        cur = ggml_reshape_3d(ctx, cur, h.n_embd, 1, nt);
        out = out ? ggml_concat(ctx, out, cur, 1) : cur;
    }
    ds4_name(out, name, il);
    return out;
}

// Fold the hc streams back to one before the output head. Same shape as
// hc_pre but with a single scale and no post/comb.
static ggml_tensor * ds4_hc_head(ggml_context * ctx, const ds4_hparams & h, ggml_tensor * x,
                                 ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base) {
    const int64_t nt = x->ne[2];
    ggml_tensor * flat = ggml_reshape_2d(ctx, x, h.hc_mult * h.n_embd, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, h.f_norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);
    ds4_name(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = ds4_hc_affine(ctx, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_scale_bias(ctx, pre, 1.0f, h.hc_eps);
    ds4_name(pre, "hc_head_pre", -1);

    return ds4_hc_mix(ctx, h, x, pre);
}

// ---------------------------------------------------------------------------
// RMS norm with a weight, named the way llama.cpp names it: the unweighted
// result is "norm", the weighted one takes the caller's name.
//
// `name == nullptr` leaves the weighted result unnamed, which is not fussiness:
// llama.cpp labels the kv path's weighted norm nowhere, and a name we invent
// shows up in the comparison as a tensor missing from the reference. The dump
// is only a check if both sides emit the same set.
static ggml_tensor * ds4_norm(ggml_context * ctx, const ds4_hparams & h, ggml_tensor * x,
                              ggml_tensor * w, const char * name, int il) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, h.f_norm_rms_eps);
    ds4_name(n, "norm", il);
    ggml_tensor * out = ggml_mul(ctx, n, w);
    if (name) {
        ds4_name(out, name, il);
    }
    return out;
}

// ---------------------------------------------------------------------------
// The Hadamard rotation. **Not used by this model's main attention** — see the
// note in `ds4_build_attention_raw`. It is kept because the lightning indexer
// does rotate, at order `indexer_head_size` (128) rather than `d_key`, and that
// is the next thing here that will need it.
//
// Copied from models/glm_dsa/graph.h rather than shared: it is fifteen lines of
// fixed mathematics, and lib/README.md argues for copying over abstracting at
// two models. That copy is bit-exact against llama.cpp's `ggml_gen_hadamard`
// (which is not in ggml's public header), proven by glm-dsa's KL == 0 gate.
static void ds4_fill_hadamard(std::vector<float> & data, int n) {
    if (n <= 0 || (n & (n - 1)) != 0) {
        NANO_ABORT("hadamard order %d is not a power of two", n);
    }
    data.assign((size_t) n * n, 0.0f);
    data[0] = 1.0f / sqrtf((float) n);
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[(size_t) i * n + j];
                data[(size_t) (i + s) * n + j]       =  val;
                data[(size_t) i * n + (j + s)]       =  val;
                data[(size_t) (i + s) * n + (j + s)] = -val;
            }
        }
    }
}

// Reshape to [n, rest], rotate, reshape back. llama_mul_mat_hadamard.
static ggml_tensor * ds4_hadamard(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];
    ggml_tensor * res = ggml_is_contiguous(cur)
                      ? ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur) / n)
                      : ggml_cont_2d(ctx, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(ctx, rot, res);
    // The hint is copied along with everything else: it steers kernel choice,
    // and a different kernel is a different summation order.
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
}

// ---------------------------------------------------------------------------
// Attention, layer-0 shape only: `attn_raw`, i.e. no KV compressor and no
// lightning indexer. That is the path a layer with `compress_ratio == 0` takes,
// which is layers 0 and 1.
//
// `kv_all` is the full key sequence. In a real eval that comes from the KV
// cache; the porting harness passes the current chunk, which is the same thing
// for a single prefill from position 0 — and it keeps a cache out of the first
// comparison.
static ggml_tensor * ds4_build_attention_raw(ggml_context * ctx, ggml_cgraph * gf,
                                             const ds4_hparams & h, const ds4_layer & L,
                                             ggml_tensor * cur, ggml_tensor * inp_pos,
                                             ggml_tensor * kq_mask,
                                             int32_t n_tokens, uint32_t il) {
    const int64_t d_head = h.d_key;                    // 512
    const int64_t d_rope = h.rope_dim;                 // 64
    const int64_t d_nope = d_head - d_rope;            // 448
    const int64_t n_head = h.n_head;                   // 64
    const int64_t n_grp  = h.out_group_count;          // 8
    const int64_t grp_dim = (n_head / n_grp) * d_head; // 8*512 = 4096

    // Layer 0 has no compressor, so rope runs unscaled: llama.cpp's
    // `use_compress_rope` is false and every yarn parameter collapses.
    const float   freq_base   = h.rope_freq_base;
    const float   freq_scale  = 1.0f;
    const float   ext_factor  = 0.0f;
    const float   attn_factor = 1.0f;   // dsv4_rope_attn_factor(_, 0) == 1
    const float   beta_fast   = 0.0f;
    const float   beta_slow   = 0.0f;
    const int32_t n_ctx_orig  = 0;
    const int     rope_type   = 0;      // LLAMA_ROPE_TYPE_NORM, as glm-dsa

    // ---- q: a LoRA, a weighted norm, an expansion, then a *plain* norm ----
    ggml_tensor * qr = ggml_mul_mat(ctx, L.attn_q_a, cur);
    ds4_name(qr, "qr", il);
    qr = ds4_norm(ctx, h, qr, L.attn_q_a_norm, "qr_norm", il);

    ggml_tensor * q = ggml_mul_mat(ctx, L.attn_q_b, qr);
    q = ggml_reshape_3d(ctx, q, d_head, n_head, n_tokens);
    // Note: no weight. The other norms in this layer have one; this does not.
    q = ggml_rms_norm(ctx, q, h.f_norm_rms_eps);
    ds4_name(q, "q_norm", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx, q, d_nope, n_head, n_tokens,
                                        ggml_row_size(q->type, d_head),
                                        ggml_row_size(q->type, d_head) * n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx, q, d_rope, n_head, n_tokens,
                                      ggml_row_size(q->type, d_head),
                                      ggml_row_size(q->type, d_head) * n_head,
                                      ggml_row_size(q->type, d_nope));
    q_pe = ggml_rope_ext(ctx, q_pe, inp_pos, nullptr, d_rope, rope_type, n_ctx_orig,
                         freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ds4_name(q_pe, "q_pe", il);
    q = ggml_concat(ctx, q_nope, q_pe, 0);
    ds4_name(q, "q", il);
    // Fix the q path's place in the node order before starting kv. Both layers
    // emit a tensor called "norm" and the comparison pairs repeated names by
    // position, so emission order is part of the contract, not a detail.
    ggml_build_forward_expand(gf, q);

    // ---- kv: one latent row per token, serving as both K and V ----
    ggml_tensor * kv = ggml_mul_mat(ctx, L.attn_kv, cur);
    kv = ds4_norm(ctx, h, kv, L.attn_kv_a_norm, nullptr, il);
    kv = ggml_reshape_3d(ctx, kv, d_head, 1, n_tokens);
    // A view, and its parent is no longer an output now that the weighted norm
    // is unnamed — so it needs a copy of its own to be readable. See
    // ds4_snapshot.
    ds4_snapshot(ctx, gf, kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx, kv, d_nope, 1, n_tokens,
                                         ggml_row_size(kv->type, d_head),
                                         ggml_row_size(kv->type, d_head), 0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx, kv, d_rope, 1, n_tokens,
                                       ggml_row_size(kv->type, d_head),
                                       ggml_row_size(kv->type, d_head),
                                       ggml_row_size(kv->type, d_nope));
    kv_pe = ggml_rope_ext(ctx, kv_pe, inp_pos, nullptr, d_rope, rope_type, n_ctx_orig,
                          freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    ds4_name(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx, kv_nope, kv_pe, 0);
    ds4_name(kv, "kv", il);

    // ---- MHA with sinks ----
    //
    // No Hadamard rotation here, and that was the whole of the 4e-03 gap this
    // file used to record. llama.cpp rotates q and k only when the rotation
    // tensor exists, and `llama_kv_cache::build_input_k_rot` only builds one
    // when `attn_rot_k` is set — which needs a **quantized** KV cache
    // (`ggml_is_quantized(type_k)`, llama-kv-cache.cpp:319). The default cache
    // is F16, so for this model the reference log says `attn_rot_k = 0` for
    // every 512-wide cache and the rotation is simply absent.
    //
    // The rotation is orthonormal and its own inverse, so applying it to q and
    // k and undoing it on the output leaves the mathematics unchanged — which
    // is exactly why this was invisible: every tensor around it stayed correct
    // and only rounding moved, by ~F16 epsilon. What found it was dumping the
    // reference's *unnamed* graph nodes (`dump --max-records`): the node
    // feeding flash-attention is named "q-0 (view) (permuted)", a view of `q`
    // itself with no matmul in between.
    //
    // The one cache that does rotate is the lightning indexer's, at order 128
    // (`attn_rot_k = 1, n_embd_head_k_all = 128` in the same log), so
    // `ds4_hadamard` comes back when the indexer layers do.
    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    // The reference stores kv into the KV cache before attending, and that
    // cache is F16 by default — so the scores are computed against an F16 key,
    // not the F32 tensor just built. Round-tripping through F16 here is not a
    // space optimisation, it is what makes this agree: without it every tensor
    // through `kv` matched exactly and `attn_raw` onward did not.
    //
    // glm-dsa's KV cache is F16 for the same reason (models/glm_dsa/graph.h),
    // which is how it reaches KL == 0.
    ggml_tensor * kv16 = ggml_cast(ctx, kv, GGML_TYPE_F16);

    ggml_tensor * qp = ggml_permute(ctx, q,    0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, kv16, 0, 2, 1, 3);
    ggml_tensor * vp = kp;   // the latent is both K and V

    // Flash attention, not the explicit kq/softmax/kqv path. llama.cpp resolves
    // `Flash Attention enabled` at startup for this model, which is why its
    // dump goes straight from `kv` to `attn_raw` with no `kq`/`kq_soft_max`
    // between them — those are only named on the explicit path.
    //
    // The mask must be F16 for this op, where the explicit path wants F32.
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, vp, kq_mask,
                                            1.0f / sqrtf((float) d_head), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, L.attn_sinks);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    out = ggml_reshape_2d(ctx, out, out->ne[0] * out->ne[1], out->ne[2] * out->ne[3]);
    ds4_snapshot(ctx, gf, out, "attn_raw", il);

    // ---- undo the positional part, then the grouped-LoRA output ----
    out = ggml_reshape_3d(ctx, out, d_head, n_head, n_tokens);
    ggml_tensor * out_nope = ggml_view_3d(ctx, out, d_nope, n_head, n_tokens,
                                          ggml_row_size(out->type, d_head),
                                          ggml_row_size(out->type, d_head) * n_head, 0);
    ggml_tensor * out_pe = ggml_view_3d(ctx, out, d_rope, n_head, n_tokens,
                                        ggml_row_size(out->type, d_head),
                                        ggml_row_size(out->type, d_head) * n_head,
                                        ggml_row_size(out->type, d_nope));
    out_pe = ggml_rope_ext_back(ctx, out_pe, inp_pos, nullptr, d_rope, rope_type, n_ctx_orig,
                                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    out = ggml_concat(ctx, out_nope, out_pe, 0);
    ds4_name(out, "attn_derope", il);

    // wo_a is [4096, 8192] in the file but is used as
    // [n_head*d_head/n_grp, o_lora_rank, n_grp] — llama.cpp reshapes it at load
    // and only a comment says so. Doing it here keeps the loader honest about
    // what is in the file.
    ggml_tensor * wo_a = ggml_reshape_3d(ctx, L.attn_output_a, grp_dim, h.out_lora_rank, n_grp);

    out = ggml_reshape_3d(ctx, out, grp_dim, n_grp, n_tokens);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx, wo_a, out);
    ds4_name(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx, oa, h.out_lora_rank * n_grp, n_tokens);

    out = ggml_mul_mat(ctx, L.attn_output_b, oa);
    ds4_name(out, "attn_out", il);
    return out;
}

// ---------------------------------------------------------------------------
// The shared expert: an ordinary SwiGLU FFN over the same post-`ffn_norm`
// activation the routed experts see, summed with them. Every layer has one.
//
// It stays on the trunk rather than going to the backend with the routed
// experts — it is one dense pair of [4096, 2048] matmuls, not a lookup, so
// there is nothing to distribute.
//
// Its clamp limit is `swiglu_clamp_shexp`, a *different* array from the routed
// experts' `swiglu_clamp_exp` (they happen to be equal in this checkpoint, and
// llama.cpp falls back to the routed one when the key is absent, so an
// accidental swap would go unnoticed here and not in the next checkpoint).
// The clamp order is deepseek4's, as in the routed block: `up` symmetric,
// `gate` one-sided and *before* the SiLU.
static ggml_tensor * ds4_build_shared_expert(ggml_context * ctx, const ds4_layer & L,
                                             ggml_tensor * cur, float limit, uint32_t il) {
    ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
    ds4_name(up, "ffn_up", il);

    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ds4_name(gate, "ffn_gate", il);

    if (limit > 1e-6f) {
        up = ggml_clamp(ctx, up, -limit, limit);
        ds4_name(up, "ffn_up_clamped", il);
        gate = ggml_clamp(ctx, gate, -INFINITY, limit);
        ds4_name(gate, "ffn_gate_clamped", il);
    }

    ggml_tensor * act = ggml_swiglu_split(ctx, gate, up);
    ds4_name(act, limit > 1e-6f ? "ffn_swiglu_limited" : "ffn_swiglu", il);

    // llama.cpp leaves the down projection unnamed and the caller names the
    // result "ffn_shexp".
    return ggml_mul_mat(ctx, L.ffn_down_shexp, act);
}

// ---------------------------------------------------------------------------
// The FFN half of a layer: its own hyper-connection mixture, a norm, the routed
// experts plus the shared one, and the write back into the streams.
static ggml_tensor * ds4_build_ffn_half(ggml_context * ctx, ggml_cgraph * gf,
                                        const ds4_hparams & h, const ds4_layer & L,
                                        ggml_tensor * inpL, ggml_tensor * inp_tokens,
                                        int32_t n_tokens, uint32_t il) {
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * residual = inpL;
    ggml_tensor * cur = ds4_hc_pre(ctx, gf, h, inpL, L.hc_ffn_fn, L.hc_ffn_scale,
                                   L.hc_ffn_base, &post, &comb, il);
    ds4_name(cur, "hc_ffn_pre", il);

    ggml_build_forward_expand(gf, residual);
    ggml_build_forward_expand(gf, post);
    ggml_build_forward_expand(gf, comb);

    cur = ds4_norm(ctx, h, cur, L.ffn_norm, "ffn_norm", il);

    // Hash routing: the first `n_hash_layer` layers do not rank experts, they
    // look them up by token id. `ffn_gate_tid2eid` is [n_expert_used, n_vocab]
    // of i32, so one `get_rows` with the prompt's ids *is* the whole routing
    // decision — and it is why these layers cannot go to the backend, which is
    // sent activations and never sees a token id.
    ggml_tensor * selected = nullptr;
    if (h.is_hash_routed(il)) {
        selected = ggml_get_rows(ctx, L.ffn_tid2eid, inp_tokens);
    }

    ggml_tensor * moe_out = build_ds4_moe_block(ctx, gf, h, il, L, cur, n_tokens,
                                                selected, ds4_name);
    // `snapshot` rather than `name`: the block's aggregate is a chain of adds
    // for n_expert_used > 1, but the bare view of the single expert when it is
    // 1 — and naming a view is not enough to keep it readable (ds4_snapshot).
    ds4_snapshot(ctx, gf, moe_out, "ffn_moe_out", il);

    ggml_tensor * shexp = ds4_build_shared_expert(ctx, L, cur, h.swiglu_clamp_shexp[il], il);
    ds4_name(shexp, "ffn_shexp", il);

    cur = ggml_add(ctx, moe_out, shexp);
    ds4_name(cur, "ffn_out", il);

    return ds4_hc_post(ctx, h, cur, residual, post, comb, "l_last", il);
}

// ---------------------------------------------------------------------------
// How far the port goes. Everything up to this point has been checked against
// llama.cpp tensor by tensor; past it, nothing has.
enum ds4_stage {
    DS4_STAGE_HC_PRE = 0,   // through hc_attn_pre / attn_norm for layer 0
    DS4_STAGE_ATTN   = 1,   // ... through attn_out / hc_attn_post for layer 0
    DS4_STAGE_LAYER  = 2,   // ... through ffn_out / l_last: one whole layer
};

// Build the trunk as far as `stage`. `tokens` is the prompt's ids.
//
// Returns the last tensor built. Every named intermediate is marked as a graph
// output so it survives to be read back and compared.
static ggml_tensor * ds4_build_graph(ggml_context * ctx, ggml_cgraph * gf, const ds4_model & M,
                                     ggml_tensor * inp_tokens, ggml_tensor * inp_pos,
                                     ggml_tensor * kq_mask,
                                     int32_t n_tokens, ds4_stage stage) {
    const ds4_hparams & h = M.h;

    ggml_tensor * inp = ggml_get_rows(ctx, M.tok_embd, inp_tokens);
    ds4_name(inp, "inp_embd", -1);

    // The hc streams start as `hc` identical copies of the embedding.
    ggml_tensor * inpL = ggml_reshape_3d(ctx, inp, h.n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx, inpL, h.n_embd, h.hc_mult, n_tokens, 1);
    ds4_name(inpL, "hc_init", -1);

    const uint32_t il = 0;
    const ds4_layer & L = M.layers[il];

    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;
    ggml_tensor * cur = ds4_hc_pre(ctx, gf, h, inpL, L.hc_attn_fn, L.hc_attn_scale,
                                   L.hc_attn_base, &post, &comb, il);
    ds4_name(cur, "hc_attn_pre", il);

    cur = ds4_norm(ctx, h, cur, L.attn_norm, "attn_norm", il);

    if (stage == DS4_STAGE_HC_PRE) {
        ggml_build_forward_expand(gf, cur);
        return cur;
    }

    cur = ds4_build_attention_raw(ctx, gf, h, L, cur, inp_pos, kq_mask, n_tokens, il);

    inpL = ds4_hc_post(ctx, h, cur, inpL, post, comb, "hc_attn_post", il);

    if (stage == DS4_STAGE_ATTN) {
        ggml_build_forward_expand(gf, inpL);
        return inpL;
    }

    inpL = ds4_build_ffn_half(ctx, gf, h, L, inpL, inp_tokens, n_tokens, il);

    if (stage == DS4_STAGE_LAYER) {
        ggml_build_forward_expand(gf, inpL);
        return inpL;
    }

    NANO_ABORT("deepseek4 trunk: stage %d is not ported yet", (int) stage);
}
