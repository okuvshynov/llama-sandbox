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
// **Every layer of the trunk is ported**: `compress_ratios` holds only 0, 4 and
// 128, and all three are handled. Verified at 384 tokens, layer by layer,
// against llama.cpp — 394 tensors at 0.0000e+00 over layers 0-5, which covers
// every combination the model contains:
//
//   layer 0, 1   ratio 0    raw attention,        hash-routed FFN
//   layer 2      ratio 4    compressor + indexer, hash-routed FFN
//   layer 3      ratio 128  compressor,           routed FFN (bias + argsort)
//   layer 4      ratio 4    compressor + indexer, routed FFN
//   layer 5      ratio 128  compressor,           routed FFN
//
// Layer 2 was checked again at 2560 tokens (47/47), where its 640 compressed
// blocks exceed `indexer_top_k` and the indexer's selection genuinely drops
// 128 of them; below that length the top-k takes everything and cannot fail.
// Layer 3 was checked again at 12 tokens, where no complete 128-token block
// exists — llama.cpp then leaves the compressed half out entirely and falls
// back to plain raw attention, and so does this.
//
// What remains before the model runs end to end is the head (`hc_head`,
// `result_norm`, `result_output`) and a real KV cache to replace the harness's
// fresh-cache, single-prefill assumptions.
//
// **Eleven tensors cannot be compared at all**, and that is not a gap in the
// port: the `csa_*_k` / `hca_*_k` key views and the mask concatenations are
// shaped by llama.cpp's KV-cache *padding* (384 tokens become 512 rows, 96
// blocks become 256) and the padded tail holds whatever was in the buffer. A
// view sized to the data can never equal one sized to the cache. It costs
// nothing, because everything downstream of them — `attn_csa_lid` / `attn_hca`
// onward — is compared and exact, and a wrong selection or a wrong mask would
// land there. `dump_inspect.py --exclude` names them rather than passing them
// silently.
//
// The sequence length is part of what is being checked, and has twice been
// what a mistake hid behind. Five tokens make a single compressed block, and
// one block cannot distinguish a block *index* from the block's *first token
// position* — both are 0, and rope at 0 is the identity — which is exactly the
// parameter that was wrong here. Below 128 tokens the sliding window never
// bites, so a plain causal mask passes. Neither is visible in the comparison's
// output; both need a longer prompt to become visible at all.
//
// Layer 1 needed no new arithmetic — turning the single layer into a loop was
// the whole change — but it is not a free result: it is the first layer whose
// input is a computed layer output rather than the repeated embedding, so it
// checks `l_last` and the whole layer-0 chain feeding it in a way that layer 0
// alone could not.
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
// Always copy, for a tensor a later in-place op will write through — see
// ds4_cb_call_pre_inplace in moe_block.h.
static void ds4_snapshot_copy(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * t,
                              const char * base, int il) {
    ggml_tensor * s = ggml_cont(ctx, t);
    ds4_name(s, base, il);
    ggml_build_forward_expand(gf, s);
}

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
// Rope, which this model runs in two regimes.
//
// A layer *with* a compressor ropes at the compressor's own base (160000) with
// the checkpoint's yarn parameters. A layer without one collapses every yarn
// knob and ropes at the training base (10000, scale 1). llama.cpp calls the
// distinction `use_compress_rope` and drives it from `compress_ratios[il]`, so
// it is a property of the layer, not of the tensor being roped — the compressed
// block, q_pe and kv_pe of a compressor layer all use the same set.
struct ds4_rope {
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
    int     type;
};

// llama.cpp's `dsv4_rope_attn_factor`. Note it does *not* use the context's
// `yarn_attn_factor`, which llama.cpp separately resolves to 1.0 here; this
// model computes its own and it is 1/(1 + 0.1*ln 16) = 0.78293.
static float ds4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }
    return 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
}

static ds4_rope ds4_rope_of(const ds4_hparams & h, uint32_t il) {
    ds4_rope r = {};
    r.type = 0;   // LLAMA_ROPE_TYPE_NORM, as glm-dsa
    if (h.has_compressor(il)) {
        r.freq_base  = h.rope_compress_freq_base;
        r.freq_scale = 1.0f / h.rope_yarn_factor;
        // llama.cpp leaves cparams.yarn_ext_factor unset, which it then
        // resolves to 1.0 for a yarn checkpoint (llama-context.cpp:190).
        r.ext_factor = 1.0f;
        r.beta_fast  = h.rope_yarn_beta_fast;
        r.beta_slow  = h.rope_yarn_beta_slow;
        r.n_ctx_orig = (int32_t) h.rope_yarn_orig_ctx;
    } else {
        r.freq_base  = h.rope_freq_base;
        r.freq_scale = 1.0f;
        r.ext_factor = 0.0f;
        r.beta_fast  = 0.0f;
        r.beta_slow  = 0.0f;
        r.n_ctx_orig = 0;
    }
    r.attn_factor = ds4_rope_attn_factor(r.freq_scale, r.ext_factor);
    return r;
}

static ggml_tensor * ds4_rope_ext(ggml_context * ctx, ggml_tensor * t, ggml_tensor * pos,
                                  int64_t d_rope, const ds4_rope & r) {
    return ggml_rope_ext(ctx, t, pos, nullptr, (int) d_rope, r.type, r.n_ctx_orig,
                         r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor,
                         r.beta_fast, r.beta_slow);
}

// ---------------------------------------------------------------------------
// The KV compressor, shared by the attention path and the lightning indexer.
//
// Every `ratio` tokens are folded into one key. The fold is a softmax-weighted
// average, and the weights come from a second projection of the same
// activation (`*_compressor_gate`) plus a learned per-slot bias
// (`*_compressor_ape`) — so the model chooses which of the tokens in a block
// the compressed key should look like.
//
// "Overlap" is the part worth reading twice: each block attends over `2*ratio`
// slots, the previous block's tokens *and* its own. The projection is
// `2*n_embd_head` wide and the two halves are used for the two roles — the
// first half for a token acting as somebody else's history, the second for a
// token in its own block. That is why `read_idxs` gathers `2*ratio*n_blocks`
// rows and the two views take opposite halves.

// llama.cpp's `dsv4_append_zero_row`: one extra row so an index can mean
// "nothing here". Zero for values, -inf for scores, which the softmax then
// drops.
static ggml_tensor * ds4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);
    return ggml_concat(ctx, t, row, 1);
}

// What the harness has to supply for one compressor. In llama.cpp these come
// from `llm_graph_input_dsv4`, computed on the host from the ubatch and the
// cache; here they are graph inputs so the cache can be written later without
// touching the graph.
struct ds4_comp_inputs {
    ggml_tensor * base_kv    = nullptr;  // f32 [coff*n_embd_head, n_base] carried state
    ggml_tensor * base_score = nullptr;  // f32 [coff*n_embd_head, n_base]
    ggml_tensor * read_idxs  = nullptr;  // i32 [coff*ratio*n_blocks]
    ggml_tensor * comp_pos   = nullptr;  // i32 [n_blocks] rope position per block
    ggml_tensor * ape_pos    = nullptr;  // i32 [n_tokens] slot within the block
    // Which compressed blocks a token may see: block j is visible to the token
    // at position p when `j < (p + 1)/ratio`, i.e. every block whose tokens all
    // lie at or before it. f16 — the indexer op requires it and so does flash
    // attention.
    ggml_tensor * mask       = nullptr;  // f16 [n_blocks, n_tokens]
};

static ggml_tensor * ds4_build_compressed_kv(ggml_context * ctx, ggml_cgraph * gf,
                                             const ds4_hparams & h, const ds4_comp_inputs & in,
                                             ggml_tensor * cur_kv, ggml_tensor * cur_score,
                                             ggml_tensor * norm_w, int64_t n_embd_head,
                                             const char * name, uint32_t il) {
    const int64_t ratio    = 4;   // DSV4_CSA_RATIO; the 128-ratio layers take another path
    const int64_t d_rope   = h.rope_dim;
    const int64_t d_nope   = n_embd_head - d_rope;
    const int64_t n_blocks = in.comp_pos->ne[0];
    const int64_t n_read   = ratio * n_blocks;

    ggml_tensor * kv_state    = ggml_concat(ctx, in.base_kv,    cur_kv,    1);
    ggml_tensor * score_state = ggml_concat(ctx, in.base_score, cur_score, 1);

    kv_state    = ds4_append_zero_row(ctx, kv_state,    false);
    score_state = ds4_append_zero_row(ctx, score_state, true);

    ggml_tensor * kv_rows    = ggml_get_rows(ctx, kv_state,    in.read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx, score_state, in.read_idxs);

    // First half of the first n_read gathered rows...
    ggml_tensor * kv_prev = ggml_cont(ctx,
        ggml_view_2d(ctx, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx, kv_prev, n_embd_head, ratio, n_blocks);
    ds4_snapshot(ctx, gf, kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx,
        ggml_view_2d(ctx, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx, score_prev, n_embd_head, ratio, n_blocks);
    ds4_snapshot(ctx, gf, score_prev, name, il);

    // ...second half of the last n_read. The extra row_size offset is what
    // selects the other half of the projection, not a different token.
    ggml_tensor * kv_cur = ggml_cont(ctx,
        ggml_view_2d(ctx, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                     n_read * kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx,
        ggml_view_2d(ctx, score_rows, n_embd_head, n_read, score_rows->nb[1],
                     n_read * score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx, kv_prev,    kv_cur,    1);
    ggml_tensor * scores = ggml_concat(ctx, score_prev, score_cur, 1);

    values = ggml_cont(ctx, ggml_permute(ctx, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx, scores);
    ggml_tensor * comp = ggml_mul(ctx, values, weights);
    comp = ggml_sum_rows(ctx, comp);
    comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));
    ds4_name(comp, name, il);

    comp = ds4_norm(ctx, h, comp, norm_w, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, d_nope, 1, n_blocks,
                                           ggml_row_size(comp->type, n_embd_head),
                                           ggml_row_size(comp->type, n_embd_head), 0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, d_rope, 1, n_blocks,
                                         ggml_row_size(comp->type, n_embd_head),
                                         ggml_row_size(comp->type, n_embd_head),
                                         ggml_row_size(comp->type, d_nope));
    const ds4_rope rope = ds4_rope_of(h, il);
    comp_pe = ds4_rope_ext(ctx, comp_pe, in.comp_pos, d_rope, rope);
    ds4_name(comp_pe, name, il);

    comp = ggml_concat(ctx, comp_nope, comp_pe, 0);
    ds4_name(comp, name, il);

    return comp;
}

// Everything a compressor layer needs from outside the graph. In llama.cpp
// these are `llm_graph_input_dsv4`'s fields, filled on the host from the ubatch
// and the cache; keeping them as inputs here means a real cache can be written
// later without the graph changing.
struct ds4_layer_inputs {
    // One per cache, because they do not share a ratio: `csa` and `lid` are the
    // ratio-4 layers' attention and indexer compressors, `hca` the ratio-128
    // layers' single compressor. `csa` and `lid` carry equal values, `hca`
    // does not — a different modulus for the slot, a different block width for
    // the mask, and a state ring of `ratio` rows rather than `2*ratio`.
    ds4_comp_inputs csa;
    ds4_comp_inputs lid;
    ds4_comp_inputs hca;
    ggml_tensor *   rot = nullptr;       // f32 [idx_key_len, idx_key_len] Hadamard
};

struct ds4_compressed {
    ggml_tensor * csa_k = nullptr;   // [d_key,       1, n_blocks]
    ggml_tensor * lid_k = nullptr;   // [idx_key_len, 1, n_blocks], rotated
};

// The two compressors a ratio-4 layer runs: one producing the attention's
// compressed keys, one the indexer's. Identical shape, different widths, and
// only the indexer's output is rotated.
static ds4_compressed ds4_build_compressors(ggml_context * ctx, ggml_cgraph * gf,
                                            const ds4_hparams & h, const ds4_layer & L,
                                            const ds4_layer_inputs & in,
                                            ggml_tensor * cur, uint32_t il) {
    ds4_compressed out;

    // ---- the attention's compressor ----
    ggml_tensor * csa_kv = ggml_mul_mat(ctx, L.cmp_kv, cur);
    ds4_name(csa_kv, "csa_state_kv", il);

    ggml_tensor * csa_score = ggml_mul_mat(ctx, L.cmp_gate, cur);
    ds4_name(csa_score, "csa_state_score", il);

    // The learned per-slot bias: which position within its block a token is
    // sitting at. `cmp_ape` is [2*d_key, ratio], so this is a gather, not a
    // matmul.
    csa_score = ggml_add(ctx, csa_score, ggml_get_rows(ctx, L.cmp_ape, in.csa.ape_pos));
    ds4_name(csa_score, "csa_state_score_ape", il);

    out.csa_k = ds4_build_compressed_kv(ctx, gf, h, in.csa, csa_kv, csa_score,
                                        L.cmp_norm, h.d_key, "csa_state_compress", il);
    // Finish this compressor before starting the next. llama.cpp expands its
    // cache write here for the same reason, and without it the indexer's chain
    // lands in front of this one's tail — both layers emit a tensor called
    // "norm", and the comparison pairs repeated names by position.
    ggml_build_forward_expand(gf, out.csa_k);

    // ---- the indexer's compressor: same shape, 128 wide, and rotated ----
    ggml_tensor * lid_kv = ggml_mul_mat(ctx, L.idx_cmp_kv, cur);
    ds4_name(lid_kv, "lid_state_kv", il);

    ggml_tensor * lid_score = ggml_mul_mat(ctx, L.idx_cmp_gate, cur);
    ds4_name(lid_score, "lid_state_score", il);

    lid_score = ggml_add(ctx, lid_score, ggml_get_rows(ctx, L.idx_cmp_ape, in.lid.ape_pos));
    ds4_name(lid_score, "lid_state_score_ape", il);

    ggml_tensor * lid_k = ds4_build_compressed_kv(ctx, gf, h, in.lid, lid_kv, lid_score,
                                                  L.idx_cmp_norm, h.idx_key_len,
                                                  "lid_state_compress", il);

    // The one rotation this model really applies. `attn_rot_k` needs a
    // quantized cache, and the indexer's is the only one that qualifies — the
    // reference log reads `attn_rot_k = 1, n_embd_head_k_all = 128` for it and
    // 0 for all three 512-wide caches. Getting this backwards cost a day; see
    // the header.
    lid_k = ds4_hadamard(ctx, lid_k, in.rot);
    ds4_snapshot(ctx, gf, lid_k, "lid_state_compress_rot", il);
    out.lid_k = lid_k;

    return out;
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
// q, kv and the LoRA rank they come from. Shared by every layer kind: only
// what happens *between* kv and the output differs.
struct ds4_qkv {
    ggml_tensor * qr = nullptr;   // [q_lora_rank, n_tokens], the indexer needs it too
    ggml_tensor * q  = nullptr;   // [d_key, n_head, n_tokens]
    ggml_tensor * kv = nullptr;   // [d_key, 1, n_tokens]
};

static ds4_qkv ds4_build_qkv(ggml_context * ctx, ggml_cgraph * gf,
                             const ds4_hparams & h, const ds4_layer & L,
                             ggml_tensor * cur, ggml_tensor * inp_pos,
                             int32_t n_tokens, uint32_t il) {
    const int64_t d_head = h.d_key;                    // 512
    const int64_t d_rope = h.rope_dim;                 // 64
    const int64_t d_nope = d_head - d_rope;            // 448
    const int64_t n_head = h.n_head;                   // 64

    const ds4_rope rope = ds4_rope_of(h, il);

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
    q_pe = ds4_rope_ext(ctx, q_pe, inp_pos, d_rope, rope);
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
    kv_pe = ds4_rope_ext(ctx, kv_pe, inp_pos, d_rope, rope);
    ds4_name(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx, kv_nope, kv_pe, 0);
    ds4_name(kv, "kv", il);

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);
    return { qr, q, kv };
}

// ---------------------------------------------------------------------------
// The ratio-128 compressor. Same idea as the ratio-4 one and a *different*
// function in llama.cpp too (`build_hca_compressed_kv_from_state`), because
// these blocks do not overlap: a block folds its own `ratio` tokens and
// nothing else. So the projection is `n_embd_head` wide rather than twice it,
// the gather reads `ratio` rows per block rather than `2*ratio`, there is no
// prev/cur split, and no pad row is needed — every index points at a real
// token.
static ggml_tensor * ds4_build_compressed_kv_plain(ggml_context * ctx, ggml_cgraph * gf,
                                                   const ds4_hparams & h,
                                                   const ds4_comp_inputs & in,
                                                   ggml_tensor * cur_kv, ggml_tensor * cur_score,
                                                   ggml_tensor * norm_w, int64_t ratio,
                                                   int64_t n_embd_head, const char * name,
                                                   uint32_t il) {
    const int64_t d_rope   = h.rope_dim;
    const int64_t d_nope   = n_embd_head - d_rope;
    const int64_t n_blocks = in.comp_pos->ne[0];

    ggml_tensor * kv_state    = ggml_concat(ctx, in.base_kv,    cur_kv,    1);
    ggml_tensor * score_state = ggml_concat(ctx, in.base_score, cur_score, 1);

    ggml_tensor * kv = ggml_get_rows(ctx, kv_state, in.read_idxs);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, ratio, n_blocks);
    ds4_snapshot(ctx, gf, kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx, score_state, in.read_idxs);
    score = ggml_reshape_3d(ctx, score, n_embd_head, ratio, n_blocks);
    ds4_snapshot(ctx, gf, score, name, il);

    ggml_tensor * values = ggml_cont(ctx, ggml_permute(ctx, kv,    1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx, scores);
    ggml_tensor * comp = ggml_mul(ctx, values, weights);
    comp = ggml_sum_rows(ctx, comp);
    comp = ggml_cont(ctx, ggml_permute(ctx, comp, 1, 0, 2, 3));
    ds4_name(comp, name, il);

    comp = ds4_norm(ctx, h, comp, norm_w, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx, comp, d_nope, 1, n_blocks,
                                           ggml_row_size(comp->type, n_embd_head),
                                           ggml_row_size(comp->type, n_embd_head), 0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx, comp, d_rope, 1, n_blocks,
                                         ggml_row_size(comp->type, n_embd_head),
                                         ggml_row_size(comp->type, n_embd_head),
                                         ggml_row_size(comp->type, d_nope));
    const ds4_rope rope = ds4_rope_of(h, il);
    comp_pe = ds4_rope_ext(ctx, comp_pe, in.comp_pos, d_rope, rope);
    ds4_name(comp_pe, name, il);

    comp = ggml_concat(ctx, comp_nope, comp_pe, 0);
    ds4_name(comp, name, il);

    return comp;
}

// The ratio-128 layer's compressor, named as llama.cpp names it.
static ggml_tensor * ds4_build_hca_compressor(ggml_context * ctx, ggml_cgraph * gf,
                                              const ds4_hparams & h, const ds4_layer & L,
                                              const ds4_layer_inputs & in,
                                              ggml_tensor * cur, uint32_t il) {
    ggml_tensor * kv = ggml_mul_mat(ctx, L.cmp_kv, cur);
    ds4_name(kv, "hca_state_kv", il);

    ggml_tensor * score = ggml_mul_mat(ctx, L.cmp_gate, cur);
    ds4_name(score, "hca_state_score", il);

    score = ggml_add(ctx, score, ggml_get_rows(ctx, L.cmp_ape, in.hca.ape_pos));
    ds4_name(score, "hca_state_score_ape", il);

    return ds4_build_compressed_kv_plain(ctx, gf, h, in.hca, kv, score, L.cmp_norm,
                                         (int64_t) h.compress_ratio[il], h.d_key,
                                         "hca_state_compress", il);
}

// ---------------------------------------------------------------------------
// Attention, the shape layers 0 and 1 take: `attn_raw`, i.e. no KV compressor
// and no lightning indexer, so the key sequence is just this chunk.
static ggml_tensor * ds4_build_attention_raw(ggml_context * ctx, ggml_cgraph * gf,
                                             const ds4_hparams & h, const ds4_layer & L,
                                             const ds4_qkv & x, ggml_tensor * kq_mask,
                                             int32_t n_tokens, uint32_t il) {
    const int64_t d_head = h.d_key;
    ggml_tensor * q  = x.q;
    ggml_tensor * kv = x.kv;
    (void) n_tokens;

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
    return out;
}

// ---------------------------------------------------------------------------
// The DSA lightning indexer, and the attention that uses it.
//
// A ratio-4 layer attends over two key sequences at once: the raw keys of the
// last `sliding_window` tokens, and the compressed keys of every complete block
// before that. The compressed half can be long, so a small side network — the
// indexer — scores the blocks and only the best `indexer_top_k` are unmasked.
//
// The indexer is an attention in miniature: its own 128-wide q from the same
// LoRA rank, its own compressed keys, and a per-head weight so heads can be
// weighted rather than averaged. Its output is one score per (block, token).

// The score for every compressed block, already masked.
static ggml_tensor * ds4_build_indexer(ggml_context * ctx, const ds4_hparams & h,
                                       const ds4_layer & L, const ds4_layer_inputs & in,
                                       ggml_tensor * qr, ggml_tensor * cur,
                                       ggml_tensor * lid_k, ggml_tensor * inp_pos,
                                       int32_t n_tokens, uint32_t il) {
    const int64_t d_idx  = h.idx_key_len;              // 128
    const int64_t d_rope = h.rope_dim;                 // 64
    const int64_t d_nope = d_idx - d_rope;             // 64
    const int64_t n_idx  = h.idx_n_head;               // 64

    const ds4_rope rope = ds4_rope_of(h, il);

    ggml_tensor * q = ggml_mul_mat(ctx, L.idx_attn_q_b, qr);
    q = ggml_reshape_3d(ctx, q, d_idx, n_idx, n_tokens);
    ds4_name(q, "lid_q", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx, q, d_nope, n_idx, n_tokens,
                                        ggml_row_size(q->type, d_idx),
                                        ggml_row_size(q->type, d_idx) * n_idx, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx, q, d_rope, n_idx, n_tokens,
                                      ggml_row_size(q->type, d_idx),
                                      ggml_row_size(q->type, d_idx) * n_idx,
                                      ggml_row_size(q->type, d_nope));
    q_pe = ds4_rope_ext(ctx, q_pe, inp_pos, d_rope, rope);
    ds4_name(q_pe, "lid_q_pe", il);

    q = ggml_concat(ctx, q_nope, q_pe, 0);
    q = ds4_hadamard(ctx, q, in.rot);
    ds4_name(q, "lid_q_rot", il);

    // Prescaled, as the op's contract requires: it does not scale internally.
    ggml_tensor * weights = ggml_mul_mat(ctx, L.idx_proj, cur);
    weights = ggml_scale(ctx, weights, 1.0f / sqrtf((float) (d_idx * n_idx)));
    ds4_name(weights, "lid_weights", il);

    ggml_tensor * score = ggml_lightning_indexer(ctx, q, lid_k, weights, in.lid.mask);
    ds4_name(score, "lid_score_masked", il);
    return score;
}

// -inf everywhere except the top-k blocks, then the visibility mask added back
// on top — so a block that was selected but is not yet visible stays masked.
// llama.cpp's `build_top_k_mask`.
static ggml_tensor * ds4_build_top_k_mask(ggml_context * ctx, ggml_tensor * kq_mask,
                                          ggml_tensor * top_k, uint32_t il) {
    ggml_tensor * all = ggml_fill(ctx, kq_mask, -INFINITY);
    all = ggml_view_4d(ctx, all, 1, all->ne[0], all->ne[1], all->ne[3],
                       all->nb[0], all->nb[1], all->nb[2], 0);

    ggml_tensor * idx = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
                                     top_k->nb[1], top_k->nb[2], top_k->ne[3] * top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1,
                                             idx->ne[0], idx->ne[1], idx->ne[2]);
    zeros = ggml_fill(ctx, zeros, 0.0f);

    ggml_tensor * out = ggml_set_rows(ctx, all, zeros, idx);
    out = ggml_view_4d(ctx, out, out->ne[1], out->ne[2], 1, out->ne[3],
                       out->nb[2], out->nb[3], out->nb[3], 0);
    out = ggml_add(ctx, out, kq_mask);
    ds4_name(out, "csa_top_k_mask", il);
    return out;
}

static ggml_tensor * ds4_build_attention_csa_lid(ggml_context * ctx, ggml_cgraph * gf,
                                                 const ds4_hparams & h, const ds4_layer & L,
                                                 const ds4_layer_inputs & in,
                                                 const ds4_qkv & x, const ds4_compressed & c,
                                                 ggml_tensor * cur, ggml_tensor * inp_pos,
                                                 ggml_tensor * kq_mask, int32_t n_tokens,
                                                 uint32_t il) {
    const int64_t d_head = h.d_key;

    // The indexer's keys are the rotated compressed ones, read back from an F16
    // cache in the reference — so they are rounded before it scores them.
    ggml_tensor * lid_k = ggml_cast(ctx, c.lid_k, GGML_TYPE_F16);
    ggml_tensor * top_k_src = ds4_build_indexer(ctx, h, L, in, x.qr, cur, lid_k,
                                                inp_pos, n_tokens, il);

    // `min` because the model may hold fewer blocks than it would select.
    const int64_t n_csa   = in.csa.mask->ne[0];
    const int64_t n_top_k = n_csa < (int64_t) h.idx_top_k ? n_csa : (int64_t) h.idx_top_k;

    ggml_tensor * top_k = ggml_cont(ctx, ggml_top_k(ctx, top_k_src, (int) n_top_k));
    ds4_name(top_k, "lid_top_k", il);

    ggml_tensor * raw_k  = ggml_cast(ctx, x.kv,  GGML_TYPE_F16);
    ds4_name(raw_k, "csa_raw_k", il);
    ggml_tensor * comp_k = ggml_cast(ctx, c.csa_k, GGML_TYPE_F16);
    ds4_name(comp_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx, raw_k, comp_k, 2);
    ds4_name(k_all, "csa_k_all", il);

    ggml_tensor * csa_mask = ds4_build_top_k_mask(ctx, in.csa.mask, top_k, il);
    ggml_tensor * mask_all = ggml_concat(ctx, kq_mask, csa_mask, 0);
    ds4_name(mask_all, "csa_lid_kq_mask", il);

    ggml_tensor * qp = ggml_permute(ctx, x.q,   0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, kp, mask_all,
                                            1.0f / sqrtf((float) d_head), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, L.attn_sinks);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    out = ggml_reshape_2d(ctx, out, out->ne[0] * out->ne[1], out->ne[2] * out->ne[3]);
    ds4_snapshot(ctx, gf, out, "attn_csa_lid", il);
    return out;
}

// The ratio-128 attention: the same two-sequence shape as the ratio-4 one with
// the indexer removed. Every compressed block its mask allows is attended,
// because 128-token blocks are 32x coarser and there are correspondingly few.
static ggml_tensor * ds4_build_attention_hca(ggml_context * ctx, ggml_cgraph * gf,
                                             const ds4_hparams & h, const ds4_layer & L,
                                             const ds4_layer_inputs & in, const ds4_qkv & x,
                                             ggml_tensor * comp_k, ggml_tensor * kq_mask,
                                             uint32_t il) {
    const int64_t d_head = h.d_key;

    ggml_tensor * raw_k = ggml_cast(ctx, x.kv, GGML_TYPE_F16);
    ds4_name(raw_k, "hca_raw_k", il);
    ggml_tensor * hca_k = ggml_cast(ctx, comp_k, GGML_TYPE_F16);
    ds4_name(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx, raw_k, hca_k, 2);
    ds4_name(k_all, "hca_k_all", il);

    ggml_tensor * mask_all = ggml_concat(ctx, kq_mask, in.hca.mask, 0);
    ds4_name(mask_all, "hca_kq_mask", il);

    ggml_tensor * qp = ggml_permute(ctx, x.q,   0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k_all, 0, 2, 1, 3);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kp, kp, mask_all,
                                            1.0f / sqrtf((float) d_head), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, L.attn_sinks);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    out = ggml_reshape_2d(ctx, out, out->ne[0] * out->ne[1], out->ne[2] * out->ne[3]);
    ds4_snapshot(ctx, gf, out, "attn_hca", il);
    return out;
}

// ---------------------------------------------------------------------------
// Undo the positional part of the attention output, then the grouped-LoRA
// output projection. Identical for every layer kind, which is why the three
// attention cores return the same [n_head*d_key, n_tokens] shape.
static ggml_tensor * ds4_build_attn_out(ggml_context * ctx, const ds4_hparams & h,
                                        const ds4_layer & L, ggml_tensor * out,
                                        ggml_tensor * inp_pos, int32_t n_tokens, uint32_t il) {
    const int64_t d_head  = h.d_key;
    const int64_t d_rope  = h.rope_dim;
    const int64_t d_nope  = d_head - d_rope;
    const int64_t n_head  = h.n_head;
    const int64_t n_grp   = h.out_group_count;
    const int64_t grp_dim = (n_head / n_grp) * d_head;

    const ds4_rope rope = ds4_rope_of(h, il);

    out = ggml_reshape_3d(ctx, out, d_head, n_head, n_tokens);
    ggml_tensor * out_nope = ggml_view_3d(ctx, out, d_nope, n_head, n_tokens,
                                          ggml_row_size(out->type, d_head),
                                          ggml_row_size(out->type, d_head) * n_head, 0);
    ggml_tensor * out_pe = ggml_view_3d(ctx, out, d_rope, n_head, n_tokens,
                                        ggml_row_size(out->type, d_head),
                                        ggml_row_size(out->type, d_head) * n_head,
                                        ggml_row_size(out->type, d_nope));
    out_pe = ggml_rope_ext_back(ctx, out_pe, inp_pos, nullptr, (int) d_rope, rope.type,
                                rope.n_ctx_orig, rope.freq_base, rope.freq_scale,
                                rope.ext_factor, rope.attn_factor, rope.beta_fast, rope.beta_slow);
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
static ggml_tensor * ds4_build_shared_expert(ggml_context * ctx, ggml_cgraph * gf,
                                             const ds4_layer & L, ggml_tensor * cur,
                                             float limit, uint32_t il) {
    // Both named through a copy: `ggml_clamp` below writes through a view of
    // its input, so naming these directly would dump post-clamp values. See
    // ds4_cb_call_pre_inplace in moe_block.h for the full story.
    ggml_tensor * up = ggml_mul_mat(ctx, L.ffn_up_shexp, cur);
    ds4_snapshot_copy(ctx, gf, up, "ffn_up", il);

    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);
    ds4_snapshot_copy(ctx, gf, gate, "ffn_gate", il);

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

    ggml_tensor * shexp = ds4_build_shared_expert(ctx, gf, L, cur, h.swiglu_clamp_shexp[il], il);
    ds4_name(shexp, "ffn_shexp", il);

    cur = ggml_add(ctx, moe_out, shexp);
    ds4_name(cur, "ffn_out", il);

    return ds4_hc_post(ctx, h, cur, residual, post, comb, "l_last", il);
}

// ---------------------------------------------------------------------------
// How far the port goes. Everything up to this point has been checked against
// llama.cpp tensor by tensor; past it, nothing has.
// `stage` says how far into the **last** layer built to go; everything before
// it is built whole. That is what lets a comparison bisect: when layer N's
// attention is suspect, build N+1 layers and stop at DS4_STAGE_ATTN.
enum ds4_stage {
    DS4_STAGE_HC_PRE   = 0,   // through hc_attn_pre / attn_norm
    DS4_STAGE_COMPRESS = 1,   // ... through the two compressors (ratio-4 layers)
    DS4_STAGE_ATTN     = 2,   // ... through attn_out / hc_attn_post
    DS4_STAGE_LAYER    = 3,   // ... through ffn_out / l_last: the whole layer
};

static const char * ds4_stage_name(ds4_stage s) {
    switch (s) {
        case DS4_STAGE_HC_PRE:   return "hc_pre";
        case DS4_STAGE_COMPRESS: return "compress";
        case DS4_STAGE_ATTN:     return "attn";
        case DS4_STAGE_LAYER:    return "layer";
    }
    return "?";
}

// How many leading layers this file can build, read off the checkpoint rather
// than hard-coded. All three ratios this checkpoint uses are handled, so for
// `[0, 0, 4, 128, 4, 128, ...]` the answer is every layer — but the loop stays
// because a checkpoint with a ratio nobody has ported should stop here rather
// than at an assert deep in the graph.
static uint32_t ds4_ported_layers(const ds4_hparams & h) {
    uint32_t n = 0;
    while (n < h.n_layer) {
        const uint32_t r = h.compress_ratio[n];
        if (r != 0 && r != 4 && r != 128) {
            break;
        }
        n++;
    }
    return n;
}

// Build `n_layers` leading layers, stopping at `stage` inside the last one.
//
// Returns the last tensor built. Every named intermediate is marked as a graph
// output so it survives to be read back and compared.
static ggml_tensor * ds4_build_graph(ggml_context * ctx, ggml_cgraph * gf, const ds4_model & M,
                                     ggml_tensor * inp_tokens, ggml_tensor * inp_pos,
                                     ggml_tensor * kq_mask, const ds4_layer_inputs & in,
                                     int32_t n_tokens, ds4_stage stage, uint32_t n_layers) {
    const ds4_hparams & h = M.h;

    if (n_layers == 0 || n_layers > h.n_layer) {
        NANO_ABORT("deepseek4 trunk: n_layers %u out of range (1..%u)", n_layers, h.n_layer);
    }

    ggml_tensor * inp = ggml_get_rows(ctx, M.tok_embd, inp_tokens);
    ds4_name(inp, "inp_embd", -1);

    // The hc streams start as `hc` identical copies of the embedding.
    ggml_tensor * inpL = ggml_reshape_3d(ctx, inp, h.n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx, inpL, h.n_embd, h.hc_mult, n_tokens, 1);
    ds4_name(inpL, "hc_init", -1);

    for (uint32_t il = 0; il < n_layers; il++) {
        const bool        last = (il + 1 == n_layers);
        const ds4_layer & L    = M.layers[il];

        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * cur = ds4_hc_pre(ctx, gf, h, inpL, L.hc_attn_fn, L.hc_attn_scale,
                                       L.hc_attn_base, &post, &comb, il);
        ds4_name(cur, "hc_attn_pre", il);

        cur = ds4_norm(ctx, h, cur, L.attn_norm, "attn_norm", il);

        if (last && stage == DS4_STAGE_HC_PRE) {
            ggml_build_forward_expand(gf, cur);
            return cur;
        }

        if (h.has_compressor(il) && !h.has_indexer(il)) {
            // A ratio-128 layer. With fewer than `ratio` tokens behind it there
            // is no complete block, and llama.cpp then leaves `inp_hca.kq_mask`
            // null and takes the plain raw-attention path — so a short prompt
            // makes this layer look exactly like layer 0. Following that is not
            // an optimisation: the compressed half genuinely does not exist.
            const bool has_blocks = in.hca.comp_pos != nullptr;

            ggml_tensor * comp_k = nullptr;
            if (has_blocks) {
                comp_k = ds4_build_hca_compressor(ctx, gf, h, L, in, cur, il);
                ggml_build_forward_expand(gf, comp_k);
            }

            if (last && stage == DS4_STAGE_COMPRESS) {
                if (!has_blocks) {
                    NANO_ABORT("deepseek4 trunk: layer %u has no complete "
                               "%u-token block, so stage 'compress' has nothing "
                               "to build", il, h.compress_ratio[il]);
                }
                return comp_k;
            }

            const ds4_qkv x = ds4_build_qkv(ctx, gf, h, L, cur, inp_pos, n_tokens, il);
            cur = has_blocks
                ? ds4_build_attention_hca(ctx, gf, h, L, in, x, comp_k, kq_mask, il)
                : ds4_build_attention_raw(ctx, gf, h, L, x, kq_mask, n_tokens, il);
        } else if (h.has_indexer(il)) {
            // The compressors run before q/kv, which is the order llama.cpp's
            // graph ends up executing them in even though it builds q first.
            const ds4_compressed c = ds4_build_compressors(ctx, gf, h, L, in, cur, il);
            ggml_build_forward_expand(gf, c.lid_k);

            if (last && stage == DS4_STAGE_COMPRESS) {
                return c.lid_k;
            }

            const ds4_qkv x = ds4_build_qkv(ctx, gf, h, L, cur, inp_pos, n_tokens, il);
            cur = ds4_build_attention_csa_lid(ctx, gf, h, L, in, x, c, cur, inp_pos,
                                              kq_mask, n_tokens, il);
        } else {
            if (last && stage == DS4_STAGE_COMPRESS) {
                NANO_ABORT("deepseek4 trunk: layer %u has no compressor, so "
                           "stage 'compress' has nothing to build", il);
            }
            const ds4_qkv x = ds4_build_qkv(ctx, gf, h, L, cur, inp_pos, n_tokens, il);
            cur = ds4_build_attention_raw(ctx, gf, h, L, x, kq_mask, n_tokens, il);
        }

        cur = ds4_build_attn_out(ctx, h, L, cur, inp_pos, n_tokens, il);

        inpL = ds4_hc_post(ctx, h, cur, inpL, post, comb, "hc_attn_post", il);

        if (last && stage == DS4_STAGE_ATTN) {
            ggml_build_forward_expand(gf, inpL);
            return inpL;
        }

        inpL = ds4_build_ffn_half(ctx, gf, h, L, inpL, inp_tokens, n_tokens, il);
    }

    ggml_build_forward_expand(gf, inpL);
    return inpL;
}
