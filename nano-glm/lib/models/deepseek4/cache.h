#pragma once

// DeepSeek-V4-Flash's KV state: the tensors that live between chunks, and the
// index arrays the host computes for each one.
//
// This model carries four kinds of state, which is three more than a
// conventional transformer:
//
//   raw keys        one per token, per layer. A 128-wide sliding window, so
//                   only the last 128 positions are ever read — but they are
//                   stored contiguously by position, as llama.cpp does.
//   compressed keys one per *block* of `ratio` tokens. Ratio-4 layers keep two
//                   (the attention's and the lightning indexer's), ratio-128
//                   layers keep one.
//   compressor ring the raw `*_state_kv` / `*_state_score` projections of the
//                   last few tokens, because a block can only be folded once
//                   all its tokens have arrived, and the last one may arrive in
//                   a later chunk. `2*ratio` rows where blocks overlap, `ratio`
//                   where they do not.
//
// Splitting this out from graph.h is a seam worth keeping: everything here is
// plain bookkeeping over positions and cell indices, testable by reading, with
// no ggml ops in sight. graph.h consumes the result as input tensors — which is
// the shape it already had when `ds4-port` computed these analytically for a
// single prefill from an empty cache.
//
// llama.cpp's equivalents live in llama-kv-cache-dsv4.cpp; the correspondences
// are noted per field.

#include "model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// One compressor's carried state and the plan for a chunk.

struct ds4_comp_plan {
    uint32_t ratio    = 0;
    bool     overlap  = false;
    int32_t  n_base   = 0;   // ring rows: 2*ratio when overlapping, else ratio
    int32_t  n_blocks = 0;   // blocks *closed* by this chunk
    int32_t  n_kv     = 0;   // compressed cells visible to this chunk

    // Into the concatenated [ring | this chunk] tensor the graph builds, plus
    // one pad row at the end for an overlapping block 0 with no history.
    std::vector<int32_t> read_idxs;    // (overlap ? 2 : 1) * ratio * n_blocks
    std::vector<int64_t> write_idxs;   // n_blocks, cells in the compressed cache
    std::vector<int32_t> write_pos;    // n_blocks, rope position per block
    std::vector<int32_t> ape_pos;      // n_tokens, slot within the block

    // Which of this chunk's rows to keep in the ring, and where.
    std::vector<int32_t> persist_src;  // indices into this chunk
    std::vector<int64_t> persist_dst;  // ring rows

    std::vector<ggml_fp16_t> mask;     // [n_kv, n_tokens]
};

// llama.cpp's `state_source_idx`: where position `q` sits in [ring | chunk],
// with the pad row last. Positions inside the chunk are addressed directly;
// earlier ones come from the ring, which is indexed `pos % n_base` and is
// exactly large enough for what an overlapping block can reach back to.
static inline int32_t ds4_state_source_idx(const ds4_comp_plan & p, int32_t q,
                                           int32_t n_past, int32_t n_tokens) {
    if (q < 0) {
        return p.n_base + n_tokens;      // the pad row
    }
    if (q >= n_past) {
        return p.n_base + (q - n_past);
    }
    return q % p.n_base;
}

// The plan for one chunk of `n_tokens` tokens starting at position `n_past`.
//
// `n_kv_cells` is the width the graph will give the compressed cache view, and
// bounds `n_kv`; the mask hides everything past what this chunk can see.
static ds4_comp_plan ds4_plan_comp(uint32_t ratio, bool overlap, int32_t n_past,
                                   int32_t n_tokens, int32_t n_kv_cells) {
    ds4_comp_plan p;
    p.ratio   = ratio;
    p.overlap = overlap;
    p.n_base  = (int32_t) ((overlap ? 2 : 1) * ratio);

    const int32_t r = (int32_t) ratio;

    // A block closes at the token whose position satisfies (pos + 1) % ratio.
    // llama-kv-cache-dsv4.cpp:519.
    for (int32_t i = 0; i < n_tokens; i++) {
        const int32_t pos = n_past + i;
        if ((pos + 1) % r != 0) {
            continue;
        }
        const int32_t source_start = pos + 1 - r;
        p.write_idxs.push_back(pos / r);
        p.write_pos .push_back(source_start);
        p.n_blocks++;
    }

    // Reads: for an overlapping compressor all the previous blocks' rows first,
    // then all the current ones, because the graph takes the first half of one
    // group and the second half of the other. llama-kv-cache-dsv4.cpp:531-541.
    if (overlap) {
        for (int32_t b = 0; b < p.n_blocks; b++) {
            const int32_t source_start = p.write_pos[b];
            for (int32_t j = 0; j < r; j++) {
                p.read_idxs.push_back(
                    ds4_state_source_idx(p, source_start - r + j, n_past, n_tokens));
            }
        }
        for (int32_t b = 0; b < p.n_blocks; b++) {
            const int32_t source_start = p.write_pos[b];
            for (int32_t j = 0; j < r; j++) {
                p.read_idxs.push_back(
                    ds4_state_source_idx(p, source_start + j, n_past, n_tokens));
            }
        }
    } else {
        for (int32_t b = 0; b < p.n_blocks; b++) {
            const int32_t source_start = p.write_pos[b];
            for (int32_t j = 0; j < r; j++) {
                p.read_idxs.push_back(
                    ds4_state_source_idx(p, source_start + j, n_past, n_tokens));
            }
        }
    }

    p.ape_pos.resize((size_t) n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) {
        p.ape_pos[i] = (n_past + i) % r;
    }

    // Persist this chunk into the ring, latest position wins when two of them
    // land on the same slot (llama-kv-cache-dsv4.cpp:508-517).
    {
        std::vector<int32_t> src(p.n_base, -1);
        for (int32_t i = 0; i < n_tokens; i++) {
            src[(n_past + i) % p.n_base] = i;   // later i overwrites earlier
        }
        for (int32_t slot = 0; slot < p.n_base; slot++) {
            if (src[slot] >= 0) {
                p.persist_src.push_back(src[slot]);
                p.persist_dst.push_back(slot);
            }
        }
    }

    // Visibility: block j once all of its tokens are at or before the query.
    // llama-kv-cache-dsv4.cpp:499.
    p.n_kv = n_kv_cells;
    p.mask.resize((size_t) p.n_kv * n_tokens);
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (int32_t i = 0; i < n_tokens; i++) {
        const int32_t n_visible = (n_past + i + 1) / r;
        for (int32_t j = 0; j < p.n_kv; j++) {
            p.mask[(size_t) i * p.n_kv + j] = j < n_visible ? keep : drop;
        }
    }
    return p;
}

// The raw sliding-window mask: causal, and banded by `n_swa`. Cell j holds
// position j, so the two rules are both in terms of positions.
static inline void ds4_plan_raw_mask(std::vector<ggml_fp16_t> & out, int32_t n_kv,
                                     int32_t n_past, int32_t n_tokens, int32_t n_swa) {
    out.resize((size_t) n_kv * n_tokens);
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (int32_t i = 0; i < n_tokens; i++) {
        const int32_t p = n_past + i;
        for (int32_t j = 0; j < n_kv; j++) {
            out[(size_t) i * n_kv + j] = (j <= p && p - j < n_swa) ? keep : drop;
        }
    }
}

// ---------------------------------------------------------------------------
// The tensors themselves.

struct ds4_cache {
    uint32_t kv_size      = 0;   // raw cells, i.e. maximum context
    uint32_t csa_size     = 0;   // ratio-4 compressed cells
    uint32_t hca_size     = 0;   // ratio-128 compressed cells

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    std::vector<ggml_tensor *> k_raw;        // [d_key, kv_size] f16, every layer
    std::vector<ggml_tensor *> k_csa, k_lid; // ratio-4 layers
    std::vector<ggml_tensor *> k_hca;        // ratio-128 layers

    // Carried compressor projections, f32. Indexed by layer; null where the
    // layer has no compressor of that kind.
    std::vector<ggml_tensor *> s_csa_kv, s_csa_score;
    std::vector<ggml_tensor *> s_lid_kv, s_lid_score;
    std::vector<ggml_tensor *> s_hca_kv, s_hca_score;
};

static void ds4_cache_init(ds4_cache & C, const ds4_hparams & h, uint32_t kv_size) {
    C.kv_size  = kv_size;
    // Rounded up, because a chunk can close a block whose index is
    // (kv_size - 1)/ratio — and floored at 256, because `get_n_kv` never asks
    // for fewer than that and caps its padding at the cache size. A ratio-128
    // cache sized only by the context would be 32 cells for a 4096-token one,
    // and the graph would then see a 32-wide window where llama.cpp sees 256.
    C.csa_size = std::max(256u, (kv_size + 4   - 1) / 4);
    C.hca_size = std::max(256u, (kv_size + 128 - 1) / 128);

    const uint32_t n_layer = h.n_layer;
    C.k_raw.assign(n_layer, nullptr);
    C.k_csa.assign(n_layer, nullptr);
    C.k_lid.assign(n_layer, nullptr);
    C.k_hca.assign(n_layer, nullptr);
    C.s_csa_kv.assign(n_layer, nullptr);
    C.s_csa_score.assign(n_layer, nullptr);
    C.s_lid_kv.assign(n_layer, nullptr);
    C.s_lid_score.assign(n_layer, nullptr);
    C.s_hca_kv.assign(n_layer, nullptr);
    C.s_hca_score.assign(n_layer, nullptr);

    // Two tensors per cache plus two per ring, generously: 12 per layer.
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) n_layer * 16, nullptr, true };
    C.ctx = ggml_init(ip);
    if (!C.ctx) NANO_ABORT("deepseek4 cache: no context");

    const int64_t d_key = h.d_key;
    const int64_t d_idx = h.idx_key_len;

    for (uint32_t il = 0; il < n_layer; il++) {
        C.k_raw[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F16, d_key, kv_size);
        ggml_format_name(C.k_raw[il], "cache_k_raw_l%u", il);

        if (h.has_indexer(il)) {
            C.k_csa[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F16, d_key, C.csa_size);
            C.k_lid[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F16, d_idx, C.csa_size);
            ggml_format_name(C.k_csa[il], "cache_k_csa_l%u", il);
            ggml_format_name(C.k_lid[il], "cache_k_lid_l%u", il);

            C.s_csa_kv[il]    = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, 2 * d_key, 8);
            C.s_csa_score[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, 2 * d_key, 8);
            C.s_lid_kv[il]    = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, 2 * d_idx, 8);
            C.s_lid_score[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, 2 * d_idx, 8);
            ggml_format_name(C.s_csa_kv[il],    "state_csa_kv_l%u", il);
            ggml_format_name(C.s_csa_score[il], "state_csa_score_l%u", il);
            ggml_format_name(C.s_lid_kv[il],    "state_lid_kv_l%u", il);
            ggml_format_name(C.s_lid_score[il], "state_lid_score_l%u", il);
        } else if (h.has_compressor(il)) {
            C.k_hca[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F16, d_key, C.hca_size);
            ggml_format_name(C.k_hca[il], "cache_k_hca_l%u", il);

            C.s_hca_kv[il]    = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, d_key, 128);
            C.s_hca_score[il] = ggml_new_tensor_2d(C.ctx, GGML_TYPE_F32, d_key, 128);
            ggml_format_name(C.s_hca_kv[il],    "state_hca_kv_l%u", il);
            ggml_format_name(C.s_hca_score[il], "state_hca_score_l%u", il);
        }
    }

    C.buf = ggml_backend_alloc_ctx_tensors_from_buft(C.ctx, ggml_backend_cpu_buffer_type());
    if (!C.buf) NANO_ABORT("deepseek4 cache: allocation failed");

    // Zeroed, and load-bearing rather than tidy: an unfolded block reads ring
    // rows for tokens that have not arrived, and the fold is a softmax over
    // scores — a zero score is a real weight, so the *values* must be zero for
    // the arithmetic to come out where llama.cpp's does. It also makes the
    // first chunk of a fresh cache identical to the analytic case `ds4-port`
    // used before this file existed.
    ggml_backend_buffer_clear(C.buf, 0);

    fprintf(stderr, "ds4: kv cache %.1f MiB (%u raw cells, %u csa, %u hca)\n",
            ggml_backend_buffer_get_size(C.buf) / (1024.0 * 1024.0),
            kv_size, C.csa_size, C.hca_size);
}

static void ds4_cache_free(ds4_cache & C) {
    if (C.buf) ggml_backend_buffer_free(C.buf);
    if (C.ctx) ggml_free(C.ctx);
    C.buf = nullptr;
    C.ctx = nullptr;
}
