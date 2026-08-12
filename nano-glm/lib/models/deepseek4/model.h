#pragma once

// DeepSeek-V4-Flash ("deepseek4"): hparams, tensor names, and the loader.
//
// Tier three of lib/README.md's split, written out rather than shared with
// glm-dsa: the two architectures agree on the *shape* of a MoE layer and on
// almost nothing else. Copy, do not abstract — see that file's argument.
//
// What is genuinely new here, relative to glm-dsa:
//
//   hyper-connections   The residual stream is `hc_mult` parallel streams
//                       mixed by a learned, Sinkhorn-normalised matrix, so
//                       the *spine* of the network differs rather than one
//                       op inside it. hc_{attn,ffn}_{base,fn,scale} per layer.
//   hash-routed layers  Layers 0..2 pick their experts from a token-id lookup
//                       (`ffn_gate_tid2eid`, n_expert_used x n_vocab of i32)
//                       instead of a router, and correspondingly have no
//                       `exp_probs_b` bias.
//   sqrt-softplus gate  Gating function 4, where glm-dsa uses sigmoid.
//   clamped SwiGLU      Per-layer clamp (10.0 throughout this checkpoint) on
//                       both the routed and the shared expert.
//   KV compressors      Layers 2..42 compress the KV stream; the per-layer
//                       ratio comes from `attention.compress_ratios`.
//   attention sinks     One learned logit per head.
//
// The per-layer taxonomy is *derived from* `compress_ratios`, which is the
// only place the file states it:
//
//   ratio 0    layers 0-1    no compressor, no indexer
//   ratio 4    even 2..42    compressor + DSA lightning indexer
//   ratio 128  odd  3..41    compressor, no indexer
//
// Verified against the tensor inventory rather than assumed: the layers
// holding `indexer.proj.weight` are exactly those with ratio 4, and the ones
// holding `ffn_gate_tid2eid` are exactly those without `exp_probs_b`.

#include "gguf_store.h"
#include "moe_shape.h"

#include <cmath>

// Gating function, as llama.cpp numbers them (llama-hparams.h). Only the one
// this model uses is named; anything else is a hard abort in load_hparams,
// because a wrong gate is fluent-and-wrong rather than broken.
enum ds4_gating_func {
    DS4_GATING_SQRT_SOFTPLUS = 4,
};

struct ds4_hparams {
    std::string arch;

    uint32_t n_vocab      = 0;
    int32_t  eos_id       = -1;   // greedy generation stops here
    uint32_t n_layer      = 0;
    uint32_t n_embd       = 0;
    uint32_t n_ctx_train  = 0;

    // attention: MLA with a single latent KV, plus a grouped-LoRA output
    uint32_t n_head           = 0;   // 64
    uint32_t n_head_kv        = 0;   // 1
    uint32_t d_key            = 0;   // 512
    uint32_t d_value          = 0;   // 512
    uint32_t q_lora_rank      = 0;   // 1024
    uint32_t out_lora_rank    = 0;   // 1024
    uint32_t out_group_count  = 0;   // 8
    uint32_t sliding_window   = 0;   // 128
    float    f_norm_rms_eps   = 0.0f;

    // DSA lightning indexer, on the even layers from 2
    uint32_t idx_n_head    = 0;      // 64
    uint32_t idx_key_len   = 0;      // 128
    uint32_t idx_top_k     = 0;      // 512

    // rope: yarn
    uint32_t rope_dim               = 0;      // 64
    float    rope_freq_base         = 0.0f;   // 10000
    float    rope_compress_freq_base= 0.0f;   // 160000
    float    rope_yarn_factor       = 0.0f;   // 16
    uint32_t rope_yarn_orig_ctx     = 0;      // 65536
    float    rope_yarn_beta_fast    = 0.0f;
    float    rope_yarn_beta_slow    = 0.0f;

    // MoE
    uint32_t n_expert          = 0;  // 256
    uint32_t n_expert_used     = 0;  // 6
    uint32_t n_expert_shared   = 0;  // 1
    uint32_t n_ff_exp          = 0;  // 2048
    uint32_t expert_gating     = 0;  // 4 = sqrt(softplus(x))
    bool     expert_norm       = false;
    float    expert_scale      = 1.0f;
    uint32_t n_hash_layer      = 0;  // 3: layers 0..2 route by token id

    // hyper-connections
    uint32_t hc_mult            = 0; // 4 parallel residual streams
    uint32_t hc_sinkhorn_iters  = 0; // 20
    float    hc_eps             = 0.0f;

    // per layer, all sized n_layer
    std::vector<uint32_t> compress_ratio;   // 0 = no compressor
    std::vector<float>    swiglu_clamp_exp;
    std::vector<float>    swiglu_clamp_shexp;

    bool has_compressor(uint32_t il) const { return compress_ratio[il] != 0; }
    bool has_indexer   (uint32_t il) const { return compress_ratio[il] == 4; }
    bool is_hash_routed(uint32_t il) const { return il < n_hash_layer; }
};

// Read an array KV of n_layer entries. The file may carry more than n_layer
// (compress_ratios has 46 for 43 layers, zero-padded); the extra entries are
// not ours to interpret, so only the first n_layer are read.
static std::vector<uint32_t> ds4_arr_u32(const gguf_context * g, const char * key, uint32_t n) {
    const int64_t id = kv_id(g, key, true);
    const size_t have = gguf_get_arr_n(g, id);
    if (have < n) NANO_ABORT("%s has %zu entries, need %u", key, have, n);
    const auto t = gguf_get_arr_type(g, id);
    if (t != GGUF_TYPE_UINT32 && t != GGUF_TYPE_INT32) NANO_ABORT("%s is not an int array", key);
    const uint32_t * a = (const uint32_t *) gguf_get_arr_data(g, id);
    return std::vector<uint32_t>(a, a + n);
}

static std::vector<float> ds4_arr_f32(const gguf_context * g, const char * key, uint32_t n) {
    const int64_t id = kv_id(g, key, true);
    const size_t have = gguf_get_arr_n(g, id);
    if (have < n) NANO_ABORT("%s has %zu entries, need %u", key, have, n);
    if (gguf_get_arr_type(g, id) != GGUF_TYPE_FLOAT32) NANO_ABORT("%s is not a f32 array", key);
    const float * a = (const float *) gguf_get_arr_data(g, id);
    return std::vector<float>(a, a + n);
}

static ds4_hparams ds4_load_hparams(const gguf_context * g) {
    ds4_hparams h;
    h.arch = kv_str_opt(g, "general.architecture", "?");
    if (h.arch != "deepseek4") {
        NANO_ABORT("architecture is '%s', this loader is deepseek4 only", h.arch.c_str());
    }

    h.n_layer     = kv_u32(g, "deepseek4.block_count");
    h.n_embd      = kv_u32(g, "deepseek4.embedding_length");
    h.n_ctx_train = kv_u32(g, "deepseek4.context_length");

    h.n_head          = kv_u32(g, "deepseek4.attention.head_count");
    h.n_head_kv       = kv_u32(g, "deepseek4.attention.head_count_kv");
    h.d_key           = kv_u32(g, "deepseek4.attention.key_length");
    h.d_value         = kv_u32(g, "deepseek4.attention.value_length");
    h.q_lora_rank     = kv_u32(g, "deepseek4.attention.q_lora_rank");
    h.out_lora_rank   = kv_u32(g, "deepseek4.attention.output_lora_rank");
    h.out_group_count = kv_u32(g, "deepseek4.attention.output_group_count");
    h.sliding_window  = kv_u32_opt(g, "deepseek4.attention.sliding_window", 0);
    h.f_norm_rms_eps  = kv_f32(g, "deepseek4.attention.layer_norm_rms_epsilon");

    h.idx_n_head  = kv_u32(g, "deepseek4.attention.indexer.head_count");
    h.idx_key_len = kv_u32(g, "deepseek4.attention.indexer.key_length");
    h.idx_top_k   = kv_u32(g, "deepseek4.attention.indexer.top_k");

    h.rope_dim                = kv_u32(g, "deepseek4.rope.dimension_count");
    h.rope_freq_base          = kv_f32(g, "deepseek4.rope.freq_base");
    h.rope_compress_freq_base = kv_f32_opt(g, "deepseek4.attention.compress_rope_freq_base", 0.0f);
    h.rope_yarn_factor        = kv_f32_opt(g, "deepseek4.rope.scaling.factor", 1.0f);
    h.rope_yarn_orig_ctx      = kv_u32_opt(g, "deepseek4.rope.scaling.original_context_length", 0);
    h.rope_yarn_beta_fast     = kv_f32_opt(g, "deepseek4.rope.scaling.yarn_beta_fast", 32.0f);
    h.rope_yarn_beta_slow     = kv_f32_opt(g, "deepseek4.rope.scaling.yarn_beta_slow", 1.0f);
    {
        const std::string t = kv_str_opt(g, "deepseek4.rope.scaling.type", "none");
        if (t != "yarn") NANO_ABORT("rope scaling '%s' is not ported (expected yarn)", t.c_str());
    }

    h.n_expert        = kv_u32(g, "deepseek4.expert_count");
    h.n_expert_used   = kv_u32(g, "deepseek4.expert_used_count");
    h.n_expert_shared = kv_u32_opt(g, "deepseek4.expert_shared_count", 0);
    h.n_ff_exp        = kv_u32(g, "deepseek4.expert_feed_forward_length");
    h.expert_gating   = kv_u32(g, "deepseek4.expert_gating_func");
    h.expert_norm     = kv_bool_opt(g, "deepseek4.expert_weights_norm", false);
    h.expert_scale    = kv_f32_opt(g, "deepseek4.expert_weights_scale", 1.0f);
    h.n_hash_layer    = kv_u32_opt(g, "deepseek4.hash_layer_count", 0);
    if (h.expert_gating != DS4_GATING_SQRT_SOFTPLUS) {
        NANO_ABORT("expert_gating_func is %u, only 4 (sqrt-softplus) is ported", h.expert_gating);
    }

    h.hc_mult           = kv_u32_opt(g, "deepseek4.hyper_connection.count", 0);
    h.hc_sinkhorn_iters = kv_u32_opt(g, "deepseek4.hyper_connection.sinkhorn_iterations", 0);
    h.hc_eps            = kv_f32_opt(g, "deepseek4.hyper_connection.epsilon", 0.0f);

    h.compress_ratio     = ds4_arr_u32(g, "deepseek4.attention.compress_ratios", h.n_layer);
    h.swiglu_clamp_exp   = ds4_arr_f32(g, "deepseek4.swiglu_clamp_exp",   h.n_layer);
    h.swiglu_clamp_shexp = ds4_arr_f32(g, "deepseek4.swiglu_clamp_shexp", h.n_layer);

    {
        const int64_t id = kv_id(g, "tokenizer.ggml.tokens", true);
        h.n_vocab = (uint32_t) gguf_get_arr_n(g, id);
    }
    {
        const int64_t id = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        if (id >= 0) h.eos_id = (int32_t) kv_u32(g, "tokenizer.ggml.eos_token_id");
    }

    // Cross-checks that would otherwise surface as a fluent wrong answer.
    if (h.n_head == 0 || h.n_embd % h.n_head != 0) {
        NANO_ABORT("n_embd %u not divisible by n_head %u", h.n_embd, h.n_head);
    }
    if (h.hc_mult == 0) {
        NANO_ABORT("hyper_connection.count is 0; this checkpoint is expected to use them");
    }
    for (uint32_t il = 0; il < h.n_layer; il++) {
        const uint32_t r = h.compress_ratio[il];
        if (r != 0 && r != 4 && r != 128) {
            NANO_ABORT("layer %u has compress_ratio %u; only 0, 4 and 128 are understood", il, r);
        }
    }
    return h;
}

struct ds4_layer {
    // norms
    ggml_tensor * attn_norm      = nullptr;
    ggml_tensor * ffn_norm       = nullptr;

    // attention: q through a LoRA, a single latent kv, grouped-LoRA output
    ggml_tensor * attn_q_a       = nullptr;
    ggml_tensor * attn_q_a_norm  = nullptr;
    ggml_tensor * attn_q_b       = nullptr;
    ggml_tensor * attn_kv        = nullptr;
    ggml_tensor * attn_kv_a_norm = nullptr;
    ggml_tensor * attn_output_a  = nullptr;
    ggml_tensor * attn_output_b  = nullptr;
    ggml_tensor * attn_sinks     = nullptr;

    // KV compressor (layers 2..n_layer-1)
    ggml_tensor * cmp_kv    = nullptr;
    ggml_tensor * cmp_gate  = nullptr;
    ggml_tensor * cmp_norm  = nullptr;
    ggml_tensor * cmp_ape   = nullptr;

    // DSA lightning indexer (even layers from 2)
    ggml_tensor * idx_proj      = nullptr;
    ggml_tensor * idx_attn_q_b  = nullptr;
    ggml_tensor * idx_cmp_kv    = nullptr;
    ggml_tensor * idx_cmp_gate  = nullptr;
    ggml_tensor * idx_cmp_norm  = nullptr;
    ggml_tensor * idx_cmp_ape   = nullptr;

    // hyper-connections, for the attention and the FFN half of the layer
    ggml_tensor * hc_attn_base  = nullptr;
    ggml_tensor * hc_attn_fn    = nullptr;
    ggml_tensor * hc_attn_scale = nullptr;
    ggml_tensor * hc_ffn_base   = nullptr;
    ggml_tensor * hc_ffn_fn     = nullptr;
    ggml_tensor * hc_ffn_scale  = nullptr;

    // MoE: every layer is routed. Layers < n_hash_layer use `tid2eid` and have
    // no `exp_probs_b`; the rest use `ffn_gate_inp` plus the bias.
    ggml_tensor * ffn_gate_inp    = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_tid2eid     = nullptr;
    ggml_tensor * ffn_gate_exps   = nullptr;
    ggml_tensor * ffn_up_exps     = nullptr;
    ggml_tensor * ffn_down_exps   = nullptr;
    ggml_tensor * ffn_gate_shexp  = nullptr;
    ggml_tensor * ffn_up_shexp    = nullptr;
    ggml_tensor * ffn_down_shexp  = nullptr;
};

struct ds4_model : gguf_store {
    ds4_hparams h;

    ggml_tensor * tok_embd    = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output      = nullptr;

    // hyper-connections have an output-side triple too, folding the streams
    // back into one before the head.
    ggml_tensor * out_hc_base  = nullptr;
    ggml_tensor * out_hc_fn    = nullptr;
    ggml_tensor * out_hc_scale = nullptr;

    std::vector<ds4_layer> layers;
};

static moe_shape ds4_moe_shape_of(const ds4_hparams & h) {
    moe_shape s;
    s.arch          = h.arch;
    s.n_embd        = h.n_embd;
    s.n_layer       = h.n_layer;
    // Every layer is routed, but layers 0..n_hash_layer-1 route by token id
    // and stay on the client, which is what this field tells it.
    s.n_dense_lead  = h.n_hash_layer;
    s.n_expert      = h.n_expert;
    s.n_expert_used = h.n_expert_used;
    s.n_ff_exp      = h.n_ff_exp;
    s.expert_scale  = h.expert_scale;
    s.expert_norm   = h.expert_norm;
    return s;
}

static void ds4_load_model(ds4_model & M, const std::string & first_shard) {
    load_shards(M, first_shard, [&](const gguf_context * meta) {
        M.h = ds4_load_hparams(meta);
    });

    const ds4_hparams & h = M.h;

    M.tok_embd    = get_tensor(M, "token_embd.weight");
    M.output_norm = get_tensor(M, "output_norm.weight");
    M.output      = get_tensor(M, "output.weight", false);
    if (!M.output) M.output = M.tok_embd;   // tied embeddings

    M.out_hc_base  = get_tensor(M, "output_hc_base.weight");
    M.out_hc_fn    = get_tensor(M, "output_hc_fn.weight");
    M.out_hc_scale = get_tensor(M, "output_hc_scale.weight");

    if ((uint32_t) M.tok_embd->ne[1] != h.n_vocab) {
        NANO_ABORT("token_embd n_vocab mismatch: %" PRId64 " vs %u", M.tok_embd->ne[1], h.n_vocab);
    }

    M.layers.resize(h.n_layer);
    for (uint32_t i = 0; i < h.n_layer; i++) {
        ds4_layer & L = M.layers[i];

        L.attn_norm      = get_tensor(M, blk(i, "attn_norm.weight"));
        L.ffn_norm       = get_tensor(M, blk(i, "ffn_norm.weight"));

        L.attn_q_a       = get_tensor(M, blk(i, "attn_q_a.weight"));
        L.attn_q_a_norm  = get_tensor(M, blk(i, "attn_q_a_norm.weight"));
        L.attn_q_b       = get_tensor(M, blk(i, "attn_q_b.weight"));
        L.attn_kv        = get_tensor(M, blk(i, "attn_kv.weight"));
        L.attn_kv_a_norm = get_tensor(M, blk(i, "attn_kv_a_norm.weight"));
        L.attn_output_a  = get_tensor(M, blk(i, "attn_output_a.weight"));
        L.attn_output_b  = get_tensor(M, blk(i, "attn_output_b.weight"));
        L.attn_sinks     = get_tensor(M, blk(i, "attn_sinks.weight"));

        L.hc_attn_base  = get_tensor(M, blk(i, "hc_attn_base.weight"));
        L.hc_attn_fn    = get_tensor(M, blk(i, "hc_attn_fn.weight"));
        L.hc_attn_scale = get_tensor(M, blk(i, "hc_attn_scale.weight"));
        L.hc_ffn_base   = get_tensor(M, blk(i, "hc_ffn_base.weight"));
        L.hc_ffn_fn     = get_tensor(M, blk(i, "hc_ffn_fn.weight"));
        L.hc_ffn_scale  = get_tensor(M, blk(i, "hc_ffn_scale.weight"));

        if (h.has_compressor(i)) {
            L.cmp_kv   = get_tensor(M, blk(i, "attn_compressor_kv.weight"));
            L.cmp_gate = get_tensor(M, blk(i, "attn_compressor_gate.weight"));
            L.cmp_norm = get_tensor(M, blk(i, "attn_compressor_norm.weight"));
            L.cmp_ape  = get_tensor(M, blk(i, "attn_compressor_ape.weight"));
        }

        if (h.has_indexer(i)) {
            L.idx_proj     = get_tensor(M, blk(i, "indexer.proj.weight"));
            L.idx_attn_q_b = get_tensor(M, blk(i, "indexer.attn_q_b.weight"));
            L.idx_cmp_kv   = get_tensor(M, blk(i, "indexer_compressor_kv.weight"));
            L.idx_cmp_gate = get_tensor(M, blk(i, "indexer_compressor_gate.weight"));
            L.idx_cmp_norm = get_tensor(M, blk(i, "indexer_compressor_norm.weight"));
            L.idx_cmp_ape  = get_tensor(M, blk(i, "indexer_compressor_ape.weight"));
        }

        if (h.is_hash_routed(i)) {
            L.ffn_tid2eid = get_tensor(M, blk(i, "ffn_gate_tid2eid.weight"));
            if ((uint32_t) L.ffn_tid2eid->ne[0] != h.n_expert_used) {
                NANO_ABORT("layer %u tid2eid has %" PRId64 " rows, n_expert_used is %u",
                           i, L.ffn_tid2eid->ne[0], h.n_expert_used);
            }
        } else {
            L.ffn_exp_probs_b = get_tensor(M, blk(i, "exp_probs_b.bias"));
        }
        // Present on every layer, including the hash-routed ones, where it is
        // simply unused — so bind it and let the graph decide.
        L.ffn_gate_inp   = get_tensor(M, blk(i, "ffn_gate_inp.weight"));

        L.ffn_gate_exps  = get_tensor(M, blk(i, "ffn_gate_exps.weight"));
        L.ffn_up_exps    = get_tensor(M, blk(i, "ffn_up_exps.weight"));
        L.ffn_down_exps  = get_tensor(M, blk(i, "ffn_down_exps.weight"));
        L.ffn_gate_shexp = get_tensor(M, blk(i, "ffn_gate_shexp.weight"));
        L.ffn_up_shexp   = get_tensor(M, blk(i, "ffn_up_shexp.weight"));
        L.ffn_down_shexp = get_tensor(M, blk(i, "ffn_down_shexp.weight"));
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "deepseek4 %uL nano-glm", h.n_layer);
    M.desc = buf;
}
