#pragma once

// Model definition and GGUF loader for GLM-5.2 ("glm-dsa"), shared by the
// nano-glm client and the moe-server backend. Both mmap the same shards and
// touch only the tensors their role needs, so the page cache is shared and
// neither pays for the other's weights.
//
// Extracted from nano_glm.cpp when the MoE block moved behind a network
// service; the hparams asserts and tensor names must stay in one place or the
// two binaries can disagree about the model while both appearing to work.

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"


// File mapping for the weight shards (map_file_ro below). Guarded because
// cpu_topology.h may have pulled windows.h in already; NOMINMAX has to be set
// before the first include of it either way, or the min/max macros eat the
// std::min / std::max calls further down.
#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <fcntl.h>
#   include <sys/mman.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define NANO_ABORT(...) do { fprintf(stderr, "nano-glm: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

// ---------------------------------------------------------------------------
// GGUF metadata helpers (hard error on missing key unless *_opt)

static int64_t kv_id(const gguf_context * g, const char * key, bool required) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0 && required) NANO_ABORT("missing GGUF key '%s'", key);
    return id;
}

static uint32_t kv_u32(const gguf_context * g, const char * key) {
    int64_t id = kv_id(g, key, true);
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        case GGUF_TYPE_INT16:  return (uint32_t) gguf_get_val_i16(g, id);
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(g, id);
        default: NANO_ABORT("GGUF key '%s' is not an integer", key);
    }
}

static uint32_t kv_u32_opt(const gguf_context * g, const char * key, uint32_t dflt) {
    return gguf_find_key(g, key) < 0 ? dflt : kv_u32(g, key);
}

static float kv_f32(const gguf_context * g, const char * key) {
    return gguf_get_val_f32(g, kv_id(g, key, true));
}

static float kv_f32_opt(const gguf_context * g, const char * key, float dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_f32(g, id);
}

static bool kv_bool_opt(const gguf_context * g, const char * key, bool dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_bool(g, id);
}

static std::string kv_str_opt(const gguf_context * g, const char * key, const std::string & dflt) {
    int64_t id = gguf_find_key(g, key);
    return id < 0 ? dflt : gguf_get_val_str(g, id);
}

// ---------------------------------------------------------------------------
// hparams (GLM-5.2 / glm-dsa; values asserted where the graph hardcodes structure)

struct nano_hparams {
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

struct nano_model {
    nano_hparams h;
    std::string  desc;

    std::vector<ggml_context *>        meta_ctxs;   // own the tensor structs
    std::vector<ggml_backend_buffer_t> map_bufs;    // wrap the mmap'd data regions
    std::map<std::string, ggml_tensor *> tensors;

    ggml_tensor * tok_embd;
    ggml_tensor * output_norm;
    ggml_tensor * output;
    std::vector<nano_layer> layers;
};

// Read-only whole-file mapping. Weights are used straight from the mapping
// (ggml_backend_cpu_buffer_from_ptr), so it must outlive the model; nothing
// unmaps — the process exits and the OS reclaims.
static void * map_file_ro(const std::string & path, size_t * size_out) {
#if defined(_WIN32)
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) NANO_ABORT("cannot open '%s' (err %lu)", path.c_str(), GetLastError());
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) NANO_ABORT("cannot size '%s' (err %lu)", path.c_str(), GetLastError());
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mh) NANO_ABORT("CreateFileMapping failed for '%s' (err %lu)", path.c_str(), GetLastError());
    void * addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    // the view keeps the file and section alive on its own
    CloseHandle(mh);
    CloseHandle(fh);
    if (!addr) NANO_ABORT("MapViewOfFile failed for '%s' (err %lu)", path.c_str(), GetLastError());
    *size_out = (size_t) sz.QuadPart;
    return addr;
#else
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) NANO_ABORT("cannot open '%s'", path.c_str());
    struct stat st;
    fstat(fd, &st);
    void * addr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) NANO_ABORT("mmap failed for '%s'", path.c_str());
    *size_out = (size_t) st.st_size;
    return addr;
#endif
}

static void load_shard(nano_model & M, const std::string & path, const gguf_context ** meta_out) {
    ggml_context * ctx_meta = nullptr;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &ctx_meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) NANO_ABORT("failed to read GGUF '%s'", path.c_str());
    M.meta_ctxs.push_back(ctx_meta);

    size_t file_size = 0;
    void * addr = map_file_ro(path, &file_size);

    const size_t data_off = gguf_get_data_offset(g);
    ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr((char *) addr + data_off, file_size - data_off);
    M.map_bufs.push_back(buf);

    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_tensor * t = ggml_get_tensor(ctx_meta, name);
        if (!t) NANO_ABORT("tensor '%s' missing from meta context", name);
        ggml_backend_tensor_alloc(buf, t, (char *) addr + data_off + gguf_get_tensor_offset(g, i));
        M.tensors[name] = t;
    }

    if (meta_out) {
        *meta_out = g; // caller reads hparams from shard 1 and frees it
    } else {
        gguf_free(g);
    }
}

static ggml_tensor * get_tensor(nano_model & M, const std::string & name, bool required = true) {
    auto it = M.tensors.find(name);
    if (it == M.tensors.end()) {
        if (required) NANO_ABORT("missing tensor '%s'", name.c_str());
        return nullptr;
    }
    return it->second;
}

static std::string blk(int il, const char * suffix) {
    return "blk." + std::to_string(il) + "." + suffix;
}

static void load_model(nano_model & M, const std::string & first_shard) {
    // shard file names: ...-00001-of-000NN.gguf
    const gguf_context * meta = nullptr;
    load_shard(M, first_shard, &meta);
    M.h = load_hparams(meta);

    const uint32_t n_split = kv_u32_opt(meta, "split.count", 1);
    gguf_free((gguf_context *) meta);

    if (n_split > 1) {
        const std::string pat = "-00001-of-";
        size_t at = first_shard.find(pat);
        if (at == std::string::npos) NANO_ABORT("split.count=%u but path has no '-00001-of-' pattern", n_split);
        for (uint32_t s = 2; s <= n_split; s++) {
            char idx[16];
            snprintf(idx, sizeof(idx), "-%05u-of-", s);
            std::string path = first_shard;
            path.replace(at, pat.size(), idx);
            load_shard(M, path, nullptr);
        }
    }

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
