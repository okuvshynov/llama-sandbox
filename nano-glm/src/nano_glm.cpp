// nano-glm: minimal CPU-only inference for GLM-5.2 (arch "glm-dsa") on bare ggml.
//
// This is deliberately NOT a framework: one model family, one backend setup,
// one code path. It links only ggml (kernels, GGUF reader, backend scheduler)
// and reimplements the thin slice of llama.cpp that this one model needs:
// shard loader, single-sequence KV cache, forward graph, greedy loop.
//
// The forward graph is a faithful op-for-op port of llama.cpp's glm-dsa trunk
// graph (src/models/glm-dsa.cpp + the llm_graph_context helpers it calls),
// with the configuration the logit-kld baselines ran under baked in:
// flash attention ON, fused lightning indexer ON, F16 K caches, BLAS offload
// for batches >= 32. Faithfulness is load-bearing: the acceptance test is
// bit-identical logits vs the llama.cpp collect baseline (see README).
//
// Input is raw token ids (no tokenizer here — the ids are the interface,
// same policy as logit-kld). Output is an lkldtopk v1 file for compare.py.
#include "cpu_topology.h"
#include "logits_file.h"
#include "moe_block.h"
#include "nano_model.h"
#include "topk_utils.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// params

struct nano_params {
    std::string model_path;      // first shard
    std::string input_bin;       // -i: lkldtopk file, prompt tokens taken from it
    std::string tokens_str;      // -T: comma-separated token ids
    std::string output_path = "nano.bin";
    int32_t     n_predict   = 256;
    int32_t     top_k       = 128;
    int32_t     n_ctx       = 4096;
    int32_t     n_batch     = 512;
    int32_t     n_threads   = (int32_t) physical_core_count();
};

static bool parse_args(int argc, char ** argv, nano_params & p) {
    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if      (!strcmp(a, "-m") && i + 1 < argc) { p.model_path  = argv[++i]; }
        else if (!strcmp(a, "-i") && i + 1 < argc) { p.input_bin   = argv[++i]; }
        else if (!strcmp(a, "-T") && i + 1 < argc) { p.tokens_str  = argv[++i]; }
        else if (!strcmp(a, "-o") && i + 1 < argc) { p.output_path = argv[++i]; }
        else if (!strcmp(a, "-n") && i + 1 < argc) { p.n_predict   = atoi(argv[++i]); }
        else if (!strcmp(a, "-k") && i + 1 < argc) { p.top_k       = atoi(argv[++i]); }
        else if (!strcmp(a, "-c") && i + 1 < argc) { p.n_ctx       = atoi(argv[++i]); }
        else if (!strcmp(a, "-b") && i + 1 < argc) { p.n_batch     = atoi(argv[++i]); }
        else if (!strcmp(a, "-t") && i + 1 < argc) { p.n_threads   = atoi(argv[++i]); }
        else {
            fprintf(stderr, "nano-glm: unknown argument '%s'\n", a);
            p.model_path.clear();
            break;
        }
    }
    if (p.model_path.empty() || (p.input_bin.empty() == p.tokens_str.empty())) {
        fprintf(stderr,
            "Usage: nano-glm -m <first-shard.gguf> (-i <lkldtopk.bin> | -T <id,id,...>) [options]\n"
            "  -i <path>   take prompt token ids from an lkldtopk file (its n_prompt tokens)\n"
            "  -T <ids>    comma-separated prompt token ids\n"
            "  -o <path>   output lkldtopk file (default: nano.bin)\n"
            "  -n <int>    tokens to generate, greedy (default: 256)\n"
            "  -k <int>    top-K logits stored per position (default: 128)\n"
            "  -c <int>    context size, auto-raised to fit (default: 4096)\n"
            "  -b <int>    prompt chunk size (default: 512)\n"
            "  -t <int>    threads (default: physical cores, ignoring SMT siblings)\n");
        return false;
    }
    return true;
}

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

// same construction as llama.cpp's ggml_gen_hadamard (llama-kv-cache.cpp)
static void fill_hadamard(std::vector<float> & data, int n) {
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
            ggml_tensor * moe_out = build_moe_block(ctx0, gf, h, L, cur, n_tokens);

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

    if (n_tokens != E.cur_n_tokens || n_kv != E.cur_n_kv) {
        if (E.ctx0) ggml_free(E.ctx0);

        const size_t graph_size = 32768;
        const size_t buf_size = graph_size * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
        if (E.graph_buf.size() < buf_size) E.graph_buf.resize(buf_size);

        ggml_init_params ip = { E.graph_buf.size(), E.graph_buf.data(), /*no_alloc=*/ true };
        E.ctx0 = ggml_init(ip);
        E.gf   = ggml_new_graph_custom(E.ctx0, graph_size, false);

        E.io = build_graph(E.ctx0, E.gf, M, S, n_tokens, n_kv);

        ggml_backend_sched_reset(S.sched);
        if (!ggml_backend_sched_alloc_graph(S.sched, E.gf)) NANO_ABORT("graph alloc failed");

        E.cur_n_tokens = n_tokens;
        E.cur_n_kv     = n_kv;
    }
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

    if (ggml_backend_sched_graph_compute(S.sched, gf) != GGML_STATUS_SUCCESS) {
        NANO_ABORT("graph compute failed");
    }

    E.logits.resize((size_t) h.n_vocab * n_tokens);
    ggml_backend_tensor_get(io.logits, E.logits.data(), 0, E.logits.size() * sizeof(float));
}

// ---------------------------------------------------------------------------
// main

static std::vector<int32_t> load_prompt_tokens(const nano_params & p, std::string & label) {
    std::vector<int32_t> toks;
    if (!p.input_bin.empty()) {
        lkld_file f;
        if (!lkld_read(p.input_bin, f)) NANO_ABORT("cannot read '%s'", p.input_bin.c_str());
        if (f.seqs.empty()) NANO_ABORT("'%s' has no sequences", p.input_bin.c_str());
        const lkld_seq & s = f.seqs[0];
        toks.assign(s.tokens.begin(), s.tokens.begin() + s.n_prompt);
        label = s.label;
        fprintf(stderr, "nano-glm: prompt = %d tokens from %s (seq label '%s')\n",
                s.n_prompt, p.input_bin.c_str(), s.label.c_str());
    } else {
        const char * s = p.tokens_str.c_str();
        while (*s) {
            char * end = nullptr;
            long v = strtol(s, &end, 10);
            if (end == s) NANO_ABORT("bad token list near '%s'", s);
            toks.push_back((int32_t) v);
            s = *end == ',' ? end + 1 : end;
        }
        label = "tokens";
    }
    if (toks.empty()) NANO_ABORT("empty prompt");
    return toks;
}

int main(int argc, char ** argv) {
    nano_params params;
    if (!parse_args(argc, argv, params)) return 1;

    std::string label;
    std::vector<int32_t> prompt = load_prompt_tokens(params, label);
    const int32_t n_prompt = (int32_t) prompt.size();

    nano_model M;
    const auto t_load_start = std::chrono::steady_clock::now();
    load_model(M, params.model_path);
    const nano_hparams & h = M.h;

    for (int32_t t : prompt) {
        if (t < 0 || (uint32_t) t >= h.n_vocab) NANO_ABORT("prompt token %d out of vocab range [0, %u)", t, h.n_vocab);
    }

    const uint32_t kv_size = std::max((uint32_t) params.n_ctx, (uint32_t) (n_prompt + params.n_predict));
    nano_state S;
    init_state(S, M, kv_size, params.n_threads);
    const auto t_load_end = std::chrono::steady_clock::now();

    fprintf(stderr, "nano-glm: %s | n_vocab=%u n_layer=%u n_embd=%u | n_prompt=%d n_predict=%d top_k=%d kv=%u threads=%d (%d physical / %d logical cores)\n",
            M.desc.c_str(), h.n_vocab, h.n_layer, h.n_embd,
            n_prompt, params.n_predict, params.top_k, kv_size, params.n_threads,
            physical_core_count(), (int) std::thread::hardware_concurrency());
    fprintf(stderr, "nano-glm: load+init %.1fs (mmap is lazy; first eval pages weights in)\n",
            std::chrono::duration<double>(t_load_end - t_load_start).count());

    eval_ctx E;
    std::vector<int32_t>       tokens = prompt;
    std::vector<lkld_position> positions;
    positions.reserve(n_prompt + params.n_predict);
    std::vector<int32_t> idx_buf;

    // prompt, chunked like collect (-b), all positions recorded
    const auto t_prompt_start = std::chrono::steady_clock::now();
    for (int32_t start = 0; start < n_prompt; ) {
        const int32_t end = std::min(n_prompt, start + params.n_batch);
        eval_chunk(M, S, E, prompt.data() + start, end - start, start);
        for (int32_t i = 0; i < end - start; i++) {
            positions.push_back(extract_topk_lse(E.logits.data() + (size_t) i * h.n_vocab,
                                                 h.n_vocab, params.top_k, idx_buf));
        }
        start = end;
    }
    const auto t_prompt_end = std::chrono::steady_clock::now();

    // greedy generation: next token is the stored top-1 of the previous position
    const char * stop_reason = params.n_predict > 0 ? "length" : "none";
    int32_t next = positions.back().ids[0];
    for (int32_t step = 0; step < params.n_predict; step++) {
        tokens.push_back(next);
        fprintf(stdout, "%d ", next);
        fflush(stdout);

        eval_chunk(M, S, E, &next, 1, n_prompt + step);
        positions.push_back(extract_topk_lse(E.logits.data(), h.n_vocab, params.top_k, idx_buf));

        if (next == h.eos_id) {
            stop_reason = "eos";
            break;
        }
        next = positions.back().ids[0];
    }
    const auto t_gen_end = std::chrono::steady_clock::now();
    fprintf(stdout, "\n");

    const int32_t n_total = (int32_t) tokens.size();
    const int32_t n_gen   = n_total - n_prompt;

    lkld_file out;
    out.n_vocab    = (int32_t) h.n_vocab;
    out.top_k      = std::min(params.top_k, (int32_t) h.n_vocab);
    out.model_desc = params.model_path + " | " + M.desc;
    out.seqs.push_back({label, n_prompt, n_total, std::move(tokens), std::move(positions)});

    if (!lkld_write(params.output_path, out)) return 1;

    double tail_mean = 0.0, tail_max = 0.0;
    for (const lkld_position & p : out.seqs[0].positions) {
        double t = tail_mass(p);
        tail_mean += t;
        tail_max = std::max(tail_max, t);
    }
    tail_mean /= out.seqs[0].positions.size();

    const double prompt_s = std::chrono::duration<double>(t_prompt_end - t_prompt_start).count();
    const double gen_s    = std::chrono::duration<double>(t_gen_end - t_prompt_end).count();
    fprintf(stderr, "nano-glm: n_prompt=%d (%.1f tok/s), n_gen=%d (%.2f tok/s), stop=%s\n",
            n_prompt, n_prompt / prompt_s, n_gen, n_gen > 0 ? n_gen / gen_s : 0.0, stop_reason);
    fprintf(stderr, "nano-glm: top-%d tail mass: mean=%.3e max=%.3e\n", out.top_k, tail_mean, tail_max);
    fprintf(stderr, "nano-glm: wrote %s (%d positions)\n", params.output_path.c_str(),
            (int32_t) out.seqs[0].positions.size());

    return 0;
}
