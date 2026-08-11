#pragma once

// GLM-5.2 ("glm-dsa"): hparams, tensor names, and the loader that binds them.
//
// Tier three of lib/README.md's split — one model, and deliberately so. The
// generic half (metadata helpers, mmap, shard enumeration, the tensor map) is
// gguf_store.h; everything here knows it is looking at glm-dsa.
//
// Shared by the nano-glm client and the moe-server backend: both mmap the same
// shards and touch only the tensors their role needs, so the page cache is
// shared and neither pays for the other's weights. The hparams asserts and
// tensor names must stay in one place or the two binaries can disagree about
// the model while both appearing to work.

#include "gguf_store.h"
#include "moe_shape.h"

// ---------------------------------------------------------------------------
// hparams (GLM-5.2 / glm-dsa; values asserted where the graph hardcodes structure)

struct nano_hparams {
    std::string arch;             // asserted "glm-dsa"; kept so the handshake
                                  // reports what was loaded, not a literal
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_layer;             // trunk layers (block_count - nextn)
    uint32_t n_layer_all;
    uint32_t n_head;
    uint32_t n_ff_dense;
    uint32_t n_dense_lead;
    float    rms_eps;

    // MLA
    uint32_t q_lora_rank;
    uint32_t kv_lora_rank;
    uint32_t n_embd_head_k_mla;   // 256 = 192 nope + 64 rope
    uint32_t n_embd_head_v_mla;   // 256
    uint32_t n_rot;               // 64

    // MoE
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    float    expert_scale;
    bool     expert_norm;

    // DSA indexer
    uint32_t idx_n_head;
    uint32_t idx_head_size;       // 128
    uint32_t idx_top_k;
    std::vector<uint8_t> idx_full; // per trunk layer: 1 = full indexer, 0 = shared

    // rope
    float    freq_base;
    uint32_t n_ctx_train;
    uint32_t n_ctx_orig;

    int32_t  eos_id = -1;
};

// https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json#L26 — the BC
// default when the GGUF carries no indexer_types metadata (same as llama.cpp).
static std::vector<uint8_t> glm_5_2_default_indexer_types(uint32_t n_layer) {
    std::vector<uint8_t> v(n_layer, 0);
    for (uint32_t i = 0; i < n_layer; i++) {
        v[i] = (i < 2) || ((i - 2) % 4 == 0);
    }
    return v;
}

static nano_hparams load_hparams(const gguf_context * g) {
    const std::string arch = kv_str_opt(g, "general.architecture", "?");
    if (arch != "glm-dsa") NANO_ABORT("expected arch glm-dsa, got '%s'", arch.c_str());

    nano_hparams h = {};
    h.arch         = arch;
    h.n_embd       = kv_u32(g, "glm-dsa.embedding_length");
    h.n_layer_all  = kv_u32(g, "glm-dsa.block_count");
    const uint32_t nextn = kv_u32_opt(g, "glm-dsa.nextn_predict_layers", 0);
    h.n_layer      = h.n_layer_all - nextn;
    h.n_head       = kv_u32(g, "glm-dsa.attention.head_count");
    h.n_ff_dense   = kv_u32(g, "glm-dsa.feed_forward_length");
    h.n_dense_lead = kv_u32_opt(g, "glm-dsa.leading_dense_block_count", 0);
    h.rms_eps      = kv_f32(g, "glm-dsa.attention.layer_norm_rms_epsilon");

    h.q_lora_rank       = kv_u32(g, "glm-dsa.attention.q_lora_rank");
    h.kv_lora_rank      = kv_u32(g, "glm-dsa.attention.kv_lora_rank");
    h.n_embd_head_k_mla = kv_u32(g, "glm-dsa.attention.key_length_mla");
    h.n_embd_head_v_mla = kv_u32(g, "glm-dsa.attention.value_length_mla");
    h.n_rot             = kv_u32(g, "glm-dsa.rope.dimension_count");

    h.n_expert        = kv_u32(g, "glm-dsa.expert_count");
    h.n_expert_used   = kv_u32(g, "glm-dsa.expert_used_count");
    h.n_expert_shared = kv_u32(g, "glm-dsa.expert_shared_count");
    h.n_ff_exp        = kv_u32(g, "glm-dsa.expert_feed_forward_length");
    h.expert_scale    = kv_f32_opt(g, "glm-dsa.expert_weights_scale", 1.0f);
    h.expert_norm     = kv_bool_opt(g, "glm-dsa.expert_weights_norm", false);
    // llama.cpp maps "no gating func in metadata" to sigmoid for this arch
    const uint32_t gating = kv_u32_opt(g, "glm-dsa.expert_gating_func", 2);
    if (gating != 2) NANO_ABORT("expert_gating_func=%u, only sigmoid (2) is ported", gating);

    h.idx_n_head    = kv_u32(g, "glm-dsa.attention.indexer.head_count");
    h.idx_head_size = kv_u32(g, "glm-dsa.attention.indexer.key_length");
    h.idx_top_k     = kv_u32(g, "glm-dsa.attention.indexer.top_k");

    h.freq_base   = kv_f32(g, "glm-dsa.rope.freq_base");
    h.n_ctx_train = kv_u32(g, "glm-dsa.context_length");
    h.n_ctx_orig  = kv_u32_opt(g, "glm-dsa.rope.scaling.original_context_length", h.n_ctx_train);

    // The graph bakes in the degenerate YaRN case (freq_scale == 1, so every
    // mscale/attn_factor term collapses to 1 and kq_scale = 1/sqrt(head_k)).
    // Assert the GGUF actually is in that case rather than silently mis-scaling.
    const float freq_scale_train = 1.0f / kv_f32_opt(g, "glm-dsa.rope.scaling.factor", 1.0f);
    const std::string scaling = kv_str_opt(g, "glm-dsa.rope.scaling.type", "none");
    if (freq_scale_train != 1.0f || scaling == "yarn") {
        NANO_ABORT("rope scaling '%s' with freq_scale %f is not ported (expected the degenerate freq_scale==1 case)",
                   scaling.c_str(), freq_scale_train);
    }

    // indexer types: metadata if present, else the GLM-5.2 default pattern
    // (llama.cpp treats pre-5.2 models — n_ctx_train < 1M — as all-full)
    h.idx_full = h.n_ctx_train < 1048576 ? std::vector<uint8_t>(h.n_layer, 1)
                                         : glm_5_2_default_indexer_types(h.n_layer);
    {
        int64_t id = gguf_find_key(g, "glm-dsa.attention.indexer.types");
        if (id >= 0) {
            const size_t n = gguf_get_arr_n(g, id);
            if (n != h.n_layer) NANO_ABORT("indexer.types has %zu entries, expected %u", n, h.n_layer);
            if (gguf_get_arr_type(g, id) != GGUF_TYPE_UINT32 && gguf_get_arr_type(g, id) != GGUF_TYPE_INT32) {
                NANO_ABORT("indexer.types is not an i32/u32 array");
            }
            const uint32_t * a = (const uint32_t *) gguf_get_arr_data(g, id);
            for (size_t i = 0; i < n; i++) h.idx_full[i] = a[i] != 0;
        }
    }

    {
        int64_t id = gguf_find_key(g, "tokenizer.ggml.eos_token_id");
        if (id >= 0) h.eos_id = (int32_t) kv_u32(g, "tokenizer.ggml.eos_token_id");
    }

    h.n_vocab = kv_u32_opt(g, "glm-dsa.vocab_size", 0);
    if (h.n_vocab == 0) {
        int64_t id = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (id < 0) NANO_ABORT("cannot determine n_vocab (no vocab_size key, no tokenizer.ggml.tokens)");
        h.n_vocab = (uint32_t) gguf_get_arr_n(g, id);
    }

    return h;
}

// ---------------------------------------------------------------------------
// model: mmap'd shards + tensor map

struct nano_layer {
    ggml_tensor * attn_norm;
    ggml_tensor * attn_q_a_norm;
    ggml_tensor * attn_kv_a_norm;
    ggml_tensor * wq_a;
    ggml_tensor * wq_b;
    ggml_tensor * wkv_a_mqa;
    ggml_tensor * wk_b;
    ggml_tensor * wv_b;
    ggml_tensor * wo;
    ggml_tensor * ffn_norm;
    // dense (layers < n_dense_lead)
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_down;
    // MoE
    ggml_tensor * ffn_gate_inp;
    ggml_tensor * ffn_exp_probs_b;   // optional
    ggml_tensor * ffn_gate_exps;
    ggml_tensor * ffn_up_exps;
    ggml_tensor * ffn_down_exps;
    ggml_tensor * ffn_gate_shexp;
    ggml_tensor * ffn_up_shexp;
    ggml_tensor * ffn_down_shexp;
    // DSA indexer (full-indexer layers only)
    ggml_tensor * indexer_k_norm;
    ggml_tensor * indexer_k_norm_b;
    ggml_tensor * indexer_proj;
    ggml_tensor * indexer_attn_k;
    ggml_tensor * indexer_attn_q_b;
};

// What the RPC handshake and the routing trace need, and nothing more.
static moe_shape moe_shape_of(const nano_hparams & h) {
    moe_shape s;
    s.arch          = h.arch;
    s.n_embd        = h.n_embd;
    s.n_layer       = h.n_layer;
    s.n_dense_lead  = h.n_dense_lead;
    s.n_expert      = h.n_expert;
    s.n_expert_used = h.n_expert_used;
    s.n_ff_exp      = h.n_ff_exp;
    s.expert_scale  = h.expert_scale;
    s.expert_norm   = h.expert_norm;
    return s;
}

// Derives from gguf_store, so `M.tensors`, `M.bytes_mapped` and `M.desc` read
// exactly as they did when this was one struct.
struct nano_model : gguf_store {
    nano_hparams h;

    ggml_tensor * tok_embd;
    ggml_tensor * output_norm;
    ggml_tensor * output;
    std::vector<nano_layer> layers;
};


static void load_model(nano_model & M, const std::string & first_shard) {
    load_shards(M, first_shard, [&](const gguf_context * meta) {
        M.h = load_hparams(meta);
    });

    const nano_hparams & h = M.h;
    M.tok_embd    = get_tensor(M, "token_embd.weight");
    M.output_norm = get_tensor(M, "output_norm.weight");
    M.output      = get_tensor(M, "output.weight", false);
    if (!M.output) M.output = M.tok_embd; // tied embeddings

    if ((uint32_t) M.tok_embd->ne[1] != h.n_vocab) {
        NANO_ABORT("token_embd n_vocab mismatch: %" PRId64 " vs %u", M.tok_embd->ne[1], h.n_vocab);
    }

    M.layers.resize(h.n_layer);
    for (uint32_t i = 0; i < h.n_layer; i++) {
        nano_layer & L = M.layers[i];
        L.attn_norm      = get_tensor(M, blk(i, "attn_norm.weight"));
        L.attn_q_a_norm  = get_tensor(M, blk(i, "attn_q_a_norm.weight"));
        L.attn_kv_a_norm = get_tensor(M, blk(i, "attn_kv_a_norm.weight"));
        L.wq_a           = get_tensor(M, blk(i, "attn_q_a.weight"));
        L.wq_b           = get_tensor(M, blk(i, "attn_q_b.weight"));
        L.wkv_a_mqa      = get_tensor(M, blk(i, "attn_kv_a_mqa.weight"));
        L.wk_b           = get_tensor(M, blk(i, "attn_k_b.weight"));
        L.wv_b           = get_tensor(M, blk(i, "attn_v_b.weight"));
        L.wo             = get_tensor(M, blk(i, "attn_output.weight"));
        L.ffn_norm       = get_tensor(M, blk(i, "ffn_norm.weight"));

        if (i < h.n_dense_lead) {
            L.ffn_gate = get_tensor(M, blk(i, "ffn_gate.weight"));
            L.ffn_up   = get_tensor(M, blk(i, "ffn_up.weight"));
            L.ffn_down = get_tensor(M, blk(i, "ffn_down.weight"));
        } else {
            L.ffn_gate_inp    = get_tensor(M, blk(i, "ffn_gate_inp.weight"));
            L.ffn_exp_probs_b = get_tensor(M, blk(i, "exp_probs_b.bias"), false);
            L.ffn_gate_exps   = get_tensor(M, blk(i, "ffn_gate_exps.weight"));
            L.ffn_up_exps     = get_tensor(M, blk(i, "ffn_up_exps.weight"));
            L.ffn_down_exps   = get_tensor(M, blk(i, "ffn_down_exps.weight"));
            L.ffn_gate_shexp  = get_tensor(M, blk(i, "ffn_gate_shexp.weight"));
            L.ffn_up_shexp    = get_tensor(M, blk(i, "ffn_up_shexp.weight"));
            L.ffn_down_shexp  = get_tensor(M, blk(i, "ffn_down_shexp.weight"));
        }

        if (h.idx_full[i]) {
            L.indexer_k_norm   = get_tensor(M, blk(i, "indexer.k_norm.weight"));
            L.indexer_k_norm_b = get_tensor(M, blk(i, "indexer.k_norm.bias"));
            L.indexer_proj     = get_tensor(M, blk(i, "indexer.proj.weight"));
            L.indexer_attn_k   = get_tensor(M, blk(i, "indexer.attn_k.weight"));
            L.indexer_attn_q_b = get_tensor(M, blk(i, "indexer.attn_q_b.weight"));
        }
    }

    char buf[64];
    snprintf(buf, sizeof(buf), "glm-dsa %uL nano-glm", h.n_layer);
    M.desc = buf;
}
