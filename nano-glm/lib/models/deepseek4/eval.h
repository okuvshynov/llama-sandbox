#pragma once

// DeepSeek-V4-Flash at runtime: backend, cache, and one chunk of evaluation.
//
// The same shape as models/glm_dsa/graph.h's `nano_state` / `eval_chunk`, and
// deliberately so — an app driving either model wants exactly four things from
// it (vocabulary size, an end token, a description, and "evaluate these tokens
// at this position and give me logits"), and everything else is policy.
//
// Separate from graph.h because that file is the *graph* and this is the loop
// around it. graph.h is already the largest file in the module and none of what
// is here needs reading to understand the model.

#include "cache.h"
#include "graph.h"
#include "model.h"
#include "phase_timer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <vector>

struct ds4_state {
    ggml_backend_t backend = nullptr;
    // Kept across chunks, not built per chunk. The graph still is — see
    // `ds4_eval_chunk` — but the allocator holds the compute buffer, and
    // releasing that buffer every chunk cost 37 ms against 9 ms to allocate it.
    // `ggml_gallocr_reserve` only reallocates when the new graph needs *more*
    // than the current buffer, so after the first (largest) chunk a decode step
    // reuses it and pays neither.
    ggml_gallocr_t galloc  = nullptr;
    ds4_cache      cache;
    int32_t        n_ubatch = 512;   // the reserve the compressor plans pad to
    int32_t        n_threads = 1;

    std::vector<float>       logits;   // [n_vocab * n_outputs] of the last chunk
    std::vector<uint8_t>     graph_buf;
    std::vector<ggml_fp16_t> mask_buf;

    nano_phase_stats prof;   // always on; see lib/phase_timer.h
};

static void ds4_init_state(ds4_state & S, const ds4_model & M, uint32_t kv_size,
                           int32_t n_ubatch, int32_t n_threads) {
    S.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!S.backend) NANO_ABORT("no CPU backend");
    ggml_backend_cpu_set_n_threads(S.backend, n_threads);
    S.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(S.backend));
    if (!S.galloc) NANO_ABORT("deepseek4: no graph allocator");

    S.n_ubatch  = n_ubatch;
    S.n_threads = n_threads;
    ds4_cache_init(S.cache, M.h, kv_size);
}

static void ds4_free_state(ds4_state & S) {
    ds4_cache_free(S.cache);
    if (S.galloc) ggml_gallocr_free(S.galloc);
    S.galloc = nullptr;
    if (S.backend) ggml_backend_free(S.backend);
    S.backend = nullptr;
}

// llama.cpp's `get_n_kv`: at least 256 cells, rounded up to 256, capped by the
// cache. Matching it is not cosmetic — the padded window is what the reference
// attends over, and a narrower one is a different computation.
static inline int32_t ds4_pad_n_kv(int32_t used, uint32_t size) {
    return (int32_t) std::min<int64_t>(size, std::max<int64_t>(256, GGML_PAD(used, 256)));
}

// Evaluate `n_tok` tokens starting at position `n_past`, leaving logits in
// `S.logits`: every position when `all_logits`, otherwise the last one only.
//
// Both, because both are real llama.cpp behaviours and they build different
// graphs. `collect` wants a top-k for every prompt position and so asks for all;
// a generation step asks for one. llama.cpp passes no `inp_out_ids` in the first
// case rather than a full index list, and so does this — a gather over every row
// is not the same graph as no gather at all.
//
// The graph is rebuilt per chunk. glm-dsa's `eval_chunk` caches it by
// (n_tokens, n_kv) and this deliberately does not, and `S.prof` is the
// measurement that says it need not: **building the ~6000-node graph is 0.4% of
// a run** (5.7 ms against a 1452 ms chunk). Caching it here would also be harder
// than in glm-dsa, because the shape depends on the compressor plans as well as
// on (n_tokens, n_kv) — a ratio-4 block closes on one decode step in four, and
// the index inputs change size when it does.
//
// The *allocator* was a different story and is why `S.galloc` now outlives the
// chunk. Constructing and destroying it per chunk cost 9 ms to allocate and
// **37 ms to free** — four times the allocation it undid, and the largest
// host-side cost in the profile.
//
// It also cost about as much again *inside* `ggml_backend_graph_compute`, which
// is the part no phase timer could have attributed. A/B of the two binaries,
// alternating, three runs each on `01_prose` (111 prompt + 32 generated):
//
//     per chunk        alloc+free      compute        decode
//     per-chunk galloc   41.1 ms      1401.7 ms    1.947 tok/s
//     kept in state       3.0 ms      1356.6 ms    2.217 tok/s   +13.9%
//
// `compute` is 45 ms/chunk lower with the allocator reused, with no overlap
// between the two sets of three — releasing the buffer every chunk means the
// next chunk soft-faults it back in on first touch, and those faults land in
// the forward pass rather than in the free. So the direct cost and the
// second-order one are roughly equal, and only the A/B could see the second.
//
// The general point, since the first version of this comment guessed wrong in
// both directions: the expensive host-side thing was not the one that looked
// expensive. 6000 nodes of graph construction sounds costly and is not; freeing
// a compute buffer sounds free and is not.
static void ds4_eval_chunk(const ds4_model & M, ds4_state & S,
                           const int32_t * tokens, int32_t n_tok, int32_t n_past,
                           bool all_logits) {
    const ds4_hparams & h = M.h;
    ds4_cache & C = S.cache;

    nano_phase_timer T;
    S.prof.n_chunks += 1;
    S.prof.n_tokens += (uint64_t) n_tok;

    const size_t n_nodes_max = 32768;
    const size_t buf_size = ggml_tensor_overhead() * n_nodes_max +
                            ggml_graph_overhead_custom(n_nodes_max, false);
    if (S.graph_buf.size() < buf_size) S.graph_buf.resize(buf_size);

    ggml_init_params ip = { S.graph_buf.size(), S.graph_buf.data(), /*no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) NANO_ABORT("deepseek4: no graph context");
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes_max, false);

    auto new_input = [&](ggml_type type, int64_t ne0, int64_t ne1, const char * name) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };

    const int32_t n_used   = n_past + n_tok;
    const int32_t n_kv_raw = ds4_pad_n_kv(n_used, C.kv_size);
    // Zero until a block has actually closed: llama.cpp leaves the compressed
    // mask null while none exists and the layer runs plain raw attention, so
    // padding before asking would take the compressed path from token 0.
    const int32_t n_kv_csa = n_used / 4   > 0 ? ds4_pad_n_kv(n_used / 4,   C.csa_size) : 0;
    const int32_t n_kv_hca = n_used / 128 > 0 ? ds4_pad_n_kv(n_used / 128, C.hca_size) : 0;

    // The dummy-block reserve is set by the *configured* ubatch, not by this
    // chunk, because that is what llama.cpp reserves the graph for.
    const int32_t reserve = (S.n_ubatch + 3) / 4;

    const ds4_comp_plan p_csa = ds4_plan_comp(4,   true,  n_past, n_tok, n_kv_csa,
                                              (int32_t) C.csa_size, reserve);
    const ds4_comp_plan p_hca = ds4_plan_comp(128, false, n_past, n_tok, n_kv_hca,
                                              (int32_t) C.hca_size, 0);

    ds4_plan_raw_mask(S.mask_buf, n_kv_raw, n_past, n_tok, (int32_t) h.sliding_window);

    std::vector<int64_t> k_idxs((size_t) n_tok);
    for (int32_t i = 0; i < n_tok; i++) k_idxs[i] = n_past + i;

    ggml_tensor * inp_tokens = new_input(GGML_TYPE_I32, n_tok, 1, "inp_tokens");
    ggml_tensor * inp_pos    = new_input(GGML_TYPE_I32, n_tok, 1, "inp_pos");

    auto declare = [&](ds4_comp_inputs & d, const ds4_comp_plan & pl) {
        auto i32 = [&](const std::vector<int32_t> & v) {
            return v.empty() ? (ggml_tensor *) nullptr
                             : new_input(GGML_TYPE_I32, (int64_t) v.size(), 1, "idx");
        };
        auto i64 = [&](const std::vector<int64_t> & v) {
            return v.empty() ? (ggml_tensor *) nullptr
                             : new_input(GGML_TYPE_I64, (int64_t) v.size(), 1, "idx");
        };
        d.read_idxs   = i32(pl.read_idxs);
        d.comp_pos    = i32(pl.write_pos);
        d.ape_pos     = i32(pl.ape_pos);
        d.write_idxs  = i64(pl.write_idxs);
        d.persist_src = i32(pl.persist_src);
        d.persist_dst = i64(pl.persist_dst);
        d.mask        = pl.n_kv > 0
                      ? new_input(GGML_TYPE_F16, pl.n_kv, n_tok, "comp_mask") : nullptr;
    };

    ds4_layer_inputs lin;
    declare(lin.csa, p_csa);
    declare(lin.lid, p_csa);
    declare(lin.hca, p_hca);
    lin.rot    = new_input(GGML_TYPE_F32, h.idx_key_len, h.idx_key_len, "k_rot");
    lin.k_idxs = new_input(GGML_TYPE_I64, n_tok, 1, "k_idxs");

    ggml_tensor * kq_mask = new_input(GGML_TYPE_F16, n_kv_raw, n_tok, "kq_mask");
    ggml_tensor * out_ids = all_logits ? nullptr : new_input(GGML_TYPE_I32, 1, 1, "inp_out_ids");

    ggml_tensor * logits = ds4_build_graph(ctx, gf, M, C, inp_tokens, inp_pos, kq_mask, lin,
                                           out_ids, n_tok, DS4_STAGE_HEAD, h.n_layer);
    ggml_build_forward_expand(gf, logits);
    S.prof.build_us += T.lap();

    if (!ggml_gallocr_alloc_graph(S.galloc, gf)) NANO_ABORT("deepseek4: graph alloc failed");
    S.prof.alloc_us += T.lap();

    // An input the graph never referenced has no buffer, because gallocr only
    // allocates what the graph reaches — a chunk that closes no block does not
    // rotate anything, so `k_rot` is genuinely unused. An input the graph *does*
    // reference is always allocated, so nothing needed is skipped here.
    auto set = [](ggml_tensor * t, const void * d, size_t n) {
        if (t && t->buffer) ggml_backend_tensor_set(t, d, 0, n);
    };
    auto upload = [&](const ds4_comp_inputs & d, const ds4_comp_plan & pl) {
        set(d.read_idxs,   pl.read_idxs.data(),   pl.read_idxs.size()   * sizeof(int32_t));
        set(d.comp_pos,    pl.write_pos.data(),   pl.write_pos.size()   * sizeof(int32_t));
        set(d.ape_pos,     pl.ape_pos.data(),     pl.ape_pos.size()     * sizeof(int32_t));
        set(d.write_idxs,  pl.write_idxs.data(),  pl.write_idxs.size()  * sizeof(int64_t));
        set(d.persist_src, pl.persist_src.data(), pl.persist_src.size() * sizeof(int32_t));
        set(d.persist_dst, pl.persist_dst.data(), pl.persist_dst.size() * sizeof(int64_t));
        set(d.mask,        pl.mask.data(),        pl.mask.size()        * sizeof(ggml_fp16_t));
    };

    set(inp_tokens, tokens, (size_t) n_tok * sizeof(int32_t));
    {
        std::vector<int32_t> pos((size_t) n_tok);
        for (int32_t i = 0; i < n_tok; i++) pos[i] = n_past + i;
        set(inp_pos, pos.data(), pos.size() * sizeof(int32_t));
    }
    set(kq_mask, S.mask_buf.data(), S.mask_buf.size() * sizeof(ggml_fp16_t));
    set(lin.k_idxs, k_idxs.data(), k_idxs.size() * sizeof(int64_t));
    upload(lin.csa, p_csa);
    upload(lin.lid, p_csa);
    upload(lin.hca, p_hca);
    {
        std::vector<float> had;
        ds4_fill_hadamard(had, (int) h.idx_key_len);
        set(lin.rot, had.data(), had.size() * sizeof(float));
    }
    if (out_ids) {
        const int32_t last = n_tok - 1;
        set(out_ids, &last, sizeof(int32_t));
    }
    S.prof.input_us += T.lap();

    if (ggml_backend_graph_compute(S.backend, gf) != GGML_STATUS_SUCCESS) {
        NANO_ABORT("deepseek4: graph compute failed");
    }
    S.prof.compute_us += T.lap();

    S.logits.resize((size_t) h.n_vocab * (all_logits ? n_tok : 1));
    ggml_backend_tensor_get(logits, S.logits.data(), 0, S.logits.size() * sizeof(float));
    S.prof.read_us += T.lap();

    ggml_free(ctx);
    S.prof.free_us += T.lap();
}
