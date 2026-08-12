#pragma once

// The trunk engine: backend setup, KV cache, the forward graph, and one eval
// of a token chunk. This is the op-for-op port of llama.cpp's glm-dsa trunk
// graph, and its faithfulness is the acceptance test — see ../README.md.
//
// No I/O policy here: what to feed it and what to do with the logits belongs
// to the app. The routed-expert block is either built locally (moe_block.h) or
// dispatched to a backend (moe_client.h), decided per graph build by whether
// the client is connected.
//
// moe_client.h first, for the winsock2-before-windows.h ordering.

#include "moe_client.h"

#include "expert_trace.h"
#include "moe_block.h"
#include "model.h"
#include "phase_timer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "cpu_topology.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// runtime state: backends, KV cache, hadamard

struct nano_state {
    std::vector<ggml_backend_t> backends;   // [BLAS?, CPU] — CPU last, like llama.cpp
    ggml_backend_sched_t sched;

    uint32_t kv_size;
    ggml_context * ctx_kv;
    ggml_backend_buffer_t buf_kv;
    std::vector<ggml_tensor *> k_mla;       // [576, kv_size, 1] f16, per trunk layer
    std::vector<ggml_tensor *> k_lid;       // [128, kv_size, 1] f16, full-indexer layers only
    ggml_tensor * hadamard;                 // [128, 128] f32, orthonormal Walsh-Hadamard

    int n_threads;
};

// same construction as llama.cpp's ggml_gen_hadamard (llama-kv-cache.cpp):
// Sylvester's recursion H_2m = [[H_m, H_m], [H_m, -H_m]] done in place, with
// the 1/sqrt(n) normalization seeded into the one starting cell instead of
// applied as a final pass. Every other entry is then a bitwise copy or sign
// flip of that cell, so the whole matrix has exactly one rounding in it — and
// a 1-ulp difference there would move every indexer score in every
// full-indexer layer. Do not "simplify" this into a ±1 fill plus a scale.
static void fill_hadamard(std::vector<float> & data, int n) {
    // Power of two or the recursion never reaches n: the doubling loop would
    // leave part of the matrix at its zero fill and the scores would be
    // quietly wrong rather than loudly absent. llama.cpp asserts the same.
    if (n <= 0 || (n & (n - 1)) != 0) {
        NANO_ABORT("hadamard order %d is not a power of two (indexer.key_length)", n);
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

static void init_state(nano_state & S, const nano_model & M, uint32_t kv_size, int n_threads) {
    const nano_hparams & h = M.h;
    S.kv_size   = kv_size;
    S.n_threads = n_threads;

    // backends: ACCEL devices (BLAS) first, CPU last — same priority order as
    // llama.cpp, so ggml_backend_sched offloads big-batch matmuls identically
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            // TODO: this is for easy correctness testing. Eventually trunk might run on GPU, and we'll update it
            NANO_ABORT("GPU backend '%s' present — this build must be CPU-only", ggml_backend_dev_name(dev));
        }
        if (type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (b) S.backends.push_back(b);
    }
    {
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu) NANO_ABORT("failed to init CPU backend");
        S.backends.push_back(cpu);
    }
    for (ggml_backend_t b : S.backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto set_nt = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (set_nt) set_nt(b, n_threads);
        }
        fprintf(stderr, "nano-glm: backend %s\n", ggml_backend_name(b));
    }

    // KV cache + hadamard, all in one plain CPU buffer, zero-initialized
    // (zeros in the padded cells keep flash attention NaN-free, same as llama.cpp)
    uint32_t n_full = 0;
    for (uint32_t il = 0; il < h.n_layer; il++) n_full += h.idx_full[il];

    ggml_init_params ip = {
        /*mem_size  =*/ (h.n_layer + n_full + 1 + 2) * ggml_tensor_overhead(),
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    S.ctx_kv = ggml_init(ip);

    const int64_t n_embd_k_mla = h.kv_lora_rank + h.n_rot; // 576
    S.k_mla.resize(h.n_layer);
    S.k_lid.assign(h.n_layer, nullptr);
    for (uint32_t il = 0; il < h.n_layer; il++) {
        S.k_mla[il] = ggml_new_tensor_3d(S.ctx_kv, GGML_TYPE_F16, n_embd_k_mla, kv_size, 1);
        ggml_format_name(S.k_mla[il], "cache_k_mla_l%u", il);
        if (h.idx_full[il]) {
            S.k_lid[il] = ggml_new_tensor_3d(S.ctx_kv, GGML_TYPE_F16, h.idx_head_size, kv_size, 1);
            ggml_format_name(S.k_lid[il], "cache_k_lid_l%u", il);
        }
    }
    S.hadamard = ggml_new_tensor_2d(S.ctx_kv, GGML_TYPE_F32, h.idx_head_size, h.idx_head_size);
    ggml_set_name(S.hadamard, "hadamard");

    S.buf_kv = ggml_backend_alloc_ctx_tensors(S.ctx_kv, S.backends.back());
    if (!S.buf_kv) NANO_ABORT("failed to allocate KV cache");
    ggml_backend_buffer_clear(S.buf_kv, 0);

    std::vector<float> had;
    fill_hadamard(had, h.idx_head_size);
    ggml_backend_tensor_set(S.hadamard, had.data(), 0, had.size() * sizeof(float));

    const size_t graph_size = 32768;
    S.sched = ggml_backend_sched_new(S.backends.data(), nullptr, (int) S.backends.size(),
                                     graph_size, /*parallel=*/ false, /*op_offload=*/ true);

    fprintf(stderr, "nano-glm: kv cache %.1f MiB (%u cells, %u mla + %u lid layers)\n",
            ggml_backend_buffer_get_size(S.buf_kv) / (1024.0 * 1024.0), kv_size, h.n_layer, n_full);
}

// ---------------------------------------------------------------------------
// forward graph — op-for-op port of llama.cpp src/models/glm-dsa.cpp (trunk)

struct graph_io {
    ggml_tensor * tokens;      // i32 [n_tokens]
    ggml_tensor * pos;         // i32 [n_tokens]
    ggml_tensor * out_ids;     // i32 [n_tokens]
    ggml_tensor * k_idxs_mla;  // i64 [n_tokens]
    ggml_tensor * k_idxs_lid;  // i64 [n_tokens]
    ggml_tensor * mask_mla;    // f16 [n_kv, n_tokens]
    ggml_tensor * mask_lid;    // f16 [n_kv, n_tokens]
    ggml_tensor * logits;      // f32 [n_vocab, n_tokens] (output)
};

static graph_io build_graph(ggml_context * ctx0, ggml_cgraph * gf,
                            const nano_model & M, const nano_state & S,
                            int32_t n_tokens, int32_t n_kv) {
    const nano_hparams & h = M.h;
    graph_io io = {};

    // the previous graph is discarded on rebuild, and its RPC nodes with it
    g_rpc_ctxs.clear();
    expert_trace_reset();

    const int64_t n_embd_head_k       = h.n_embd_head_k_mla;                // 256
    const int64_t n_embd_head_qk_rope = h.n_rot;                            // 64
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;// 192
    const int64_t n_head              = h.n_head;
    const int64_t kv_lora_rank        = h.kv_lora_rank;                     // 512

    const int64_t n_indexer_head           = h.idx_n_head;
    const int64_t n_embd_indexer_head      = h.idx_head_size;               // 128
    const int64_t n_embd_indexer_head_rope = h.n_rot;                       // 64
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;

    // degenerate YaRN case asserted at load: freq_scale == 1 → mscale == 1
    const float freq_scale  = 1.0f;
    const float ext_factor  = 0.0f;
    const float attn_factor = 1.0f;
    const float beta_fast   = 32.0f;
    const float beta_slow   = 1.0f;
    const float kq_scale    = 1.0f / sqrtf((float) n_embd_head_k);

    io.tokens     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens); ggml_set_input(io.tokens);
    io.pos        = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens); ggml_set_input(io.pos);
    io.out_ids    = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens); ggml_set_input(io.out_ids);
    io.k_idxs_mla = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens); ggml_set_input(io.k_idxs_mla);
    io.k_idxs_lid = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_tokens); ggml_set_input(io.k_idxs_lid);
    io.mask_mla   = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv, n_tokens, 1, 1); ggml_set_input(io.mask_mla);
    io.mask_lid   = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_kv, n_tokens, 1, 1); ggml_set_input(io.mask_lid);

    auto norm_rms = [&](ggml_tensor * cur, ggml_tensor * w) {
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, h.rms_eps), w);
    };
    // LLM_NORM with llama's default f_norm_eps (0.0f — glm-dsa never sets it)
    auto norm_std = [&](ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b) {
        return ggml_add(ctx0, ggml_mul(ctx0, ggml_norm(ctx0, cur, 0.0f), w), b);
    };
    auto rope = [&](ggml_tensor * t) { // rope type NORM (mode 0) for both MLA and indexer
        return ggml_rope_ext(ctx0, t, io.pos, nullptr, h.n_rot, 0, h.n_ctx_orig,
                             h.freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    };

    ggml_tensor * cur;
    ggml_tensor * inpL = ggml_get_rows(ctx0, M.tok_embd, io.tokens); // build_inp_embd

    ggml_tensor * prev_top_k = nullptr;

    for (uint32_t il = 0; il < h.n_layer; il++) {
        const nano_layer & L = M.layers[il];
        ggml_tensor * inpSA = inpL;

        cur = norm_rms(inpL, L.attn_norm);

        // self-attention
        {
            ggml_tensor * qr = ggml_mul_mat(ctx0, L.wq_a, cur);
            qr = norm_rms(qr, L.attn_q_a_norm);

            ggml_tensor * top_k = nullptr;

            // lightning indexer
            if (h.idx_full[il]) {
                ggml_tensor * indexer_q = ggml_mul_mat(ctx0, L.indexer_attn_q_b, qr);

                ggml_tensor * indexer_q_pe =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);
                ggml_tensor * indexer_q_nope =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

                indexer_q_pe = rope(indexer_q_pe);
                indexer_q = ggml_concat(ctx0, indexer_q_pe, indexer_q_nope, 0);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, L.indexer_attn_k, cur);
                indexer_k = norm_std(indexer_k, L.indexer_k_norm, L.indexer_k_norm_b);

                ggml_tensor * indexer_k_pe =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1, 0);
                ggml_tensor * indexer_k_nope =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head_nope));

                indexer_k_pe = rope(indexer_k_pe);
                indexer_k = ggml_concat(ctx0, indexer_k_pe, indexer_k_nope, 0);

                // Hadamard transform on indexer q and k
                indexer_q = ggml_mul_mat(ctx0, S.hadamard, indexer_q);
                indexer_k = ggml_mul_mat(ctx0, S.hadamard, indexer_k);

                // store indexer keys to the lid cache
                {
                    ggml_tensor * kc = ggml_view_2d(ctx0, indexer_k, n_embd_indexer_head, n_tokens,
                                                    indexer_k->nb[2], 0);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, S.k_lid[il], kc, io.k_idxs_lid));
                }

                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, L.indexer_proj, cur);

                // cached indexer keys, [128, 1, n_kv, 1]
                indexer_k = ggml_view_4d(ctx0, S.k_lid[il],
                        n_embd_indexer_head, 1, n_kv, 1,
                        ggml_row_size(GGML_TYPE_F16, n_embd_indexer_head),
                        ggml_row_size(GGML_TYPE_F16, n_embd_indexer_head),
                        ggml_row_size(GGML_TYPE_F16, (int64_t) n_embd_indexer_head * S.kv_size), 0);

                // single stream: these views mirror llama.cpp's stream split (no-ops here)
                indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2], 1,
                                         indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3], 0);
                indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1],
                                               indexer_weights->ne[2], 1,
                                               indexer_weights->nb[1], indexer_weights->nb[2], indexer_weights->nb[3], 0);

                indexer_weights = ggml_scale(ctx0, indexer_weights,
                                             1.0f / sqrtf((float) (n_embd_indexer_head * n_indexer_head)));

                // fused path — the config the baselines ran under
                ggml_tensor * indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k,
                                                                     indexer_weights, io.mask_lid);

                uint32_t n_top_k = indexer_score->ne[0] < h.idx_top_k ? (uint32_t) indexer_score->ne[0] : h.idx_top_k;
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                prev_top_k = top_k;
            } else {
                if (!prev_top_k) NANO_ABORT("shared indexer layer %u has no preceding full layer", il);
                top_k = prev_top_k;
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, L.wq_b, qr);

            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, L.wkv_a_mqa, cur);

            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

            q_pe = rope(q_pe);
            k_pe = rope(k_pe);

            kv_cmpr = norm_rms(kv_cmpr, L.attn_kv_a_norm);

            // MLA with the absorption optimization (MQA over the compressed cache)
            {
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, L.wk_b, q_nope);
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

                // note: rope must go first for in-place context shifting in llama.cpp;
                // kept in the same order here for graph identity
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                ggml_tensor * Vcur = kv_cmpr;

                // ---- build_attn (llm_graph_input_attn_k_dsa variant) ----
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, Kcur);

                // store K to the mla cache
                {
                    ggml_tensor * kc = ggml_view_2d(ctx0, Kcur, kv_lora_rank + n_embd_head_qk_rope, n_tokens,
                                                    Kcur->nb[2], 0);
                    ggml_build_forward_expand(gf, ggml_set_rows(ctx0, S.k_mla[il], kc, io.k_idxs_mla));
                }

                // unmask the top-k positions on top of a fresh all--inf mask,
                // then combine with the causal mask
                ggml_tensor * kq_mask_all = ggml_fill(ctx0, io.mask_mla, -INFINITY);
                kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
                                           kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

                ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
                                                      top_k->nb[1], top_k->nb[2], top_k->ne[3] * top_k->nb[3], 0);

                ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
                zeros = ggml_fill(ctx0, zeros, 0.0f);

                ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
                kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
                                             kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);
                kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, io.mask_mla);

                // cached K, [576, 1, n_kv, 1]; V is a view of K, [512, 1, n_kv, 1]
                ggml_tensor * k = ggml_view_4d(ctx0, S.k_mla[il],
                        kv_lora_rank + n_embd_head_qk_rope, 1, n_kv, 1,
                        ggml_row_size(GGML_TYPE_F16, kv_lora_rank + n_embd_head_qk_rope),
                        ggml_row_size(GGML_TYPE_F16, kv_lora_rank + n_embd_head_qk_rope),
                        ggml_row_size(GGML_TYPE_F16, (int64_t) (kv_lora_rank + n_embd_head_qk_rope) * S.kv_size), 0);
                ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3],
                                               k->nb[1], k->nb[2], k->nb[3], 0);

                // ---- build_attn_mha, flash-attention path ----
                ggml_tensor * q_att = ggml_view_4d(ctx0, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1,
                                                   Qcur->nb[1], Qcur->nb[2], Qcur->nb[3], 0);
                q_att = ggml_permute(ctx0, q_att, 0, 2, 1, 3);
                k     = ggml_permute(ctx0, k,     0, 2, 1, 3);
                v     = ggml_permute(ctx0, v,     0, 2, 1, 3);

                cur = ggml_flash_attn_ext(ctx0, q_att, k, v, kq_mask_top_k, kq_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

                // v_mla "decompression" back to per-head values
                cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
                cur = ggml_mul_mat(ctx0, L.wv_b, cur);
                cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
                cur = ggml_cont(ctx0, cur);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);

                ggml_build_forward_expand(gf, cur);

                cur = ggml_mul_mat(ctx0, L.wo, cur);
            }
        }

        if (il == h.n_layer - 1) {
            cur   = ggml_get_rows(ctx0, cur, io.out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, io.out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);

        cur = norm_rms(ffn_inp, L.ffn_norm);

        if (il < h.n_dense_lead) {
            // build_ffn: SILU, PAR gate
            ggml_tensor * up   = ggml_mul_mat(ctx0, L.ffn_up, cur);
            ggml_tensor * gate = ggml_mul_mat(ctx0, L.ffn_gate, cur);
            cur = ggml_swiglu_split(ctx0, gate, up);
            cur = ggml_mul_mat(ctx0, L.ffn_down, cur);
        } else {
            ggml_tensor * moe_out;
            if (g_moe.active()) {
                g_rpc_ctxs.push_back({ (uint32_t) il });
                ggml_tensor * args[1] = { cur };
                moe_out = ggml_custom_4d(ctx0, GGML_TYPE_F32, h.n_embd, n_tokens, 1, 1,
                                         args, 1, moe_rpc_cb, 1, &g_rpc_ctxs.back());
                ggml_build_forward_expand(gf, moe_out);
            } else {
                moe_out = build_moe_block(ctx0, gf, h, il, L, cur, n_tokens);
            }

            // shared expert (build_ffn: SILU, PAR)
            {
                ggml_tensor * s_up   = ggml_mul_mat(ctx0, L.ffn_up_shexp, cur);
                ggml_tensor * s_gate = ggml_mul_mat(ctx0, L.ffn_gate_shexp, cur);
                ggml_tensor * ffn_shexp = ggml_swiglu_split(ctx0, s_gate, s_up);
                ffn_shexp = ggml_mul_mat(ctx0, L.ffn_down_shexp, ffn_shexp);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            }
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        inpL = cur;
    }

    cur = inpL;
    cur = norm_rms(cur, M.output_norm);

    // lm_head
    cur = ggml_mul_mat(ctx0, M.output, cur);
    ggml_set_output(cur);
    io.logits = cur;

    ggml_build_forward_expand(gf, cur);

    return io;
}

// ---------------------------------------------------------------------------
// eval: one chunk of tokens through the graph

struct eval_ctx {
    std::vector<uint8_t> graph_buf;
    std::vector<float>   logits;       // [n_vocab * n_tokens] of the last eval
    std::vector<ggml_fp16_t> mask_buf;

    // graph reuse (same trick as llama.cpp): topology depends only on
    // (n_tokens, n_kv), so between rebuilds only the input data changes
    ggml_context * ctx0 = nullptr;
    ggml_cgraph *  gf   = nullptr;
    graph_io       io   = {};
    int32_t        cur_n_tokens = -1;
    int32_t        cur_n_kv     = -1;

    // Always on; see lib/phase_timer.h. The point of collecting it here as well
    // as in deepseek4 is that this model *does* cache its graph, so its `build`
    // and `alloc` shares are what the other one's would fall to.
    nano_phase_split prof;
};

// n_kv padding rule from llama_kv_cache::get_n_kv (n_pad=1 for this arch, min 256)
static int32_t pad_n_kv(int32_t n_used, uint32_t kv_size) {
    int32_t n = std::max(256, (int32_t) GGML_PAD(n_used, 256));
    return std::min((int32_t) kv_size, n);
}

static void eval_chunk(const nano_model & M, nano_state & S, eval_ctx & E,
                       const int32_t * tokens, int32_t n_tokens, int32_t n_past) {
    const nano_hparams & h = M.h;
    const int32_t n_kv = pad_n_kv(n_past + n_tokens, S.kv_size);

    nano_phase_timer   T;
    nano_phase_stats & P = E.prof.bucket(n_tokens);
    P.n_chunks += 1;
    P.n_tokens += (uint64_t) n_tokens;

    if (n_tokens != E.cur_n_tokens || n_kv != E.cur_n_kv) {
        if (E.ctx0) ggml_free(E.ctx0);

        const size_t graph_size = 32768;
        const size_t buf_size = graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
        if (E.graph_buf.size() < buf_size) E.graph_buf.resize(buf_size);

        ggml_init_params ip = { E.graph_buf.size(), E.graph_buf.data(), /*no_alloc=*/ true };
        E.ctx0 = ggml_init(ip);
        E.gf   = ggml_new_graph_custom(E.ctx0, graph_size, false);

        E.io = build_graph(E.ctx0, E.gf, M, S, n_tokens, n_kv);
        P.build_us += T.lap();

        ggml_backend_sched_reset(S.sched);
        if (!ggml_backend_sched_alloc_graph(S.sched, E.gf)) NANO_ABORT("graph alloc failed");
        P.alloc_us += T.lap();

        E.cur_n_tokens = n_tokens;
        E.cur_n_kv     = n_kv;
    }
    // On a cache hit neither lap ran, so the (negligible) time to here falls
    // into `input` below. That is the intended reading: a reused graph costs
    // nothing to build, and the phase table should show it as nothing.
    const graph_io & io = E.io;
    ggml_cgraph *    gf = E.gf;

    // inputs
    ggml_backend_tensor_set(io.tokens, tokens, 0, n_tokens * sizeof(int32_t));
    {
        std::vector<int32_t> v(n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) v[i] = n_past + i;
        ggml_backend_tensor_set(io.pos, v.data(), 0, n_tokens * sizeof(int32_t));
        for (int32_t i = 0; i < n_tokens; i++) v[i] = i;
        ggml_backend_tensor_set(io.out_ids, v.data(), 0, n_tokens * sizeof(int32_t));
    }
    {
        std::vector<int64_t> v(n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) v[i] = n_past + i;
        ggml_backend_tensor_set(io.k_idxs_mla, v.data(), 0, n_tokens * sizeof(int64_t));
        ggml_backend_tensor_set(io.k_idxs_lid, v.data(), 0, n_tokens * sizeof(int64_t));
    }
    {
        // causal mask over contiguous cells: cell j holds pos j
        E.mask_buf.resize((size_t) n_kv * n_tokens);
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
        for (int32_t i = 0; i < n_tokens; i++) {
            const int32_t p = n_past + i;
            for (int32_t j = 0; j < n_kv; j++) {
                E.mask_buf[(size_t) i * n_kv + j] = j <= p ? keep : drop;
            }
        }
        ggml_backend_tensor_set(io.mask_mla, E.mask_buf.data(), 0, E.mask_buf.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(io.mask_lid, E.mask_buf.data(), 0, E.mask_buf.size() * sizeof(ggml_fp16_t));
    }
    P.input_us += T.lap();

    if (ggml_backend_sched_graph_compute(S.sched, gf) != GGML_STATUS_SUCCESS) {
        NANO_ABORT("graph compute failed");
    }
    P.compute_us += T.lap();

    E.logits.resize((size_t) h.n_vocab * n_tokens);
    ggml_backend_tensor_get(io.logits, E.logits.data(), 0, E.logits.size() * sizeof(float));
    P.read_us += T.lap();

    // routing trace, if this is a -DNANO_EXPERT_TRACE build with --expert-log
    expert_trace_flush(n_past, n_tokens, tokens);
}
