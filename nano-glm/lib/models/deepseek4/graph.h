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
//   attn_raw ...   **NOT MATCHING** — 4.0e-03 absolute, ~7e-4 relative to the
//                  tensor's scale. Everything feeding it is exact, so the gap
//                  is inside the attention core and nowhere else.
//
//                  Two hypotheses tried and rejected: the KV cache dtype (an
//                  F16 round-trip on the key moved it from 4.4e-03 to 4.0e-03
//                  but did not close it) and fused-vs-explicit attention
//                  (llama.cpp resolves "Flash Attention enabled" for this
//                  model, so we use ggml_flash_attn_ext too — same result).
//
//                  ~7e-4 is F16 epsilon, so the live hypothesis is a remaining
//                  precision difference: the sinks, the flash-attn internal
//                  accumulation, or the Hadamard kernel selected by
//                  GGML_HINT_SRC0_IS_HADAMARD. The next diagnostic is to run
//                  without sinks and see whether the difference grows, which
//                  says whether they are being applied at all.
//   ffn half       NOT YET
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
static ggml_tensor * ds4_hc_post(ggml_context * ctx, const ds4_hparams & h,
                                 ggml_tensor * x, ggml_tensor * residual,
                                 ggml_tensor * post, ggml_tensor * comb, int il) {
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
    ds4_name(out, il < 0 ? "hc_post_out" : "hc_attn_post", il);
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
static ggml_tensor * ds4_norm(ggml_context * ctx, const ds4_hparams & h, ggml_tensor * x,
                              ggml_tensor * w, const char * name, int il) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, h.f_norm_rms_eps);
    ds4_name(n, "norm", il);
    ggml_tensor * out = ggml_mul(ctx, n, w);
    ds4_name(out, name, il);
    return out;
}

// ---------------------------------------------------------------------------
// The Hadamard rotation applied to q and kv before attention, and undone after.
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
                                             ggml_tensor * kq_mask, ggml_tensor * rot,
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

    // ---- kv: one latent row per token, serving as both K and V ----
    ggml_tensor * kv = ggml_mul_mat(ctx, L.attn_kv, cur);
    kv = ds4_norm(ctx, h, kv, L.attn_kv_a_norm, "kv_a_norm_out", il);
    kv = ggml_reshape_3d(ctx, kv, d_head, 1, n_tokens);
    ds4_name(kv, "kv_norm", il);

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

    // ---- the rotation, then MHA with sinks ----
    q  = ds4_hadamard(ctx, q,  rot);
    kv = ds4_hadamard(ctx, kv, rot);
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

    // Flash attention, not the explicit kq/softmax/kqv path — and again the
    // choice is measured. llama.cpp resolves `Flash Attention enabled` at
    // startup for this model (its log says so), which is why its dump goes
    // straight from `kv` to `attn_raw` with no intermediates in between. The
    // explicit path gave everything through `kv` bit-identical and `attn_raw`
    // off by 4.4e-03: the same answer computed in a different order.
    //
    // The mask must be F16 for this op, where the explicit path wants F32.
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, vp, kq_mask,
                                            1.0f / sqrtf((float) d_head), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, L.attn_sinks);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    out = ggml_reshape_2d(ctx, out, out->ne[0] * out->ne[1], out->ne[2] * out->ne[3]);
    out = ds4_hadamard(ctx, out, rot);
    ds4_name(out, "attn_raw", il);

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
// How far the port goes. Everything up to this point has been checked against
// llama.cpp tensor by tensor; past it, nothing has.
enum ds4_stage {
    DS4_STAGE_HC_PRE = 0,   // through hc_attn_pre / attn_norm for layer 0
    DS4_STAGE_ATTN   = 1,   // ... through attn_out / hc_attn_post for layer 0
};

// Build the trunk as far as `stage`. `tokens` is the prompt's ids.
//
// Returns the last tensor built. Every named intermediate is marked as a graph
// output so it survives to be read back and compared.
static ggml_tensor * ds4_build_graph(ggml_context * ctx, ggml_cgraph * gf, const ds4_model & M,
                                     ggml_tensor * inp_tokens, ggml_tensor * inp_pos,
                                     ggml_tensor * kq_mask, ggml_tensor * rot,
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

    cur = ds4_build_attention_raw(ctx, gf, h, L, cur, inp_pos, kq_mask, rot, n_tokens, il);

    inpL = ds4_hc_post(ctx, h, cur, inpL, post, comb, il);

    if (stage == DS4_STAGE_ATTN) {
        ggml_build_forward_expand(gf, inpL);
        return inpL;
    }

    NANO_ABORT("deepseek4 trunk: stage %d is not ported yet", (int) stage);
}
