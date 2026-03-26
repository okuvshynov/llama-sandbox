// moe-eval-size: estimate bytes read per forward pass for (MoE) GGUF models
//
// Uses only the GGUF metadata API — reads tensor headers, not tensor data.
// Works instantly even on 400B+ model files.
// Supports split GGUF files (e.g. model-00001-of-00004.gguf).

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tensor classification
// ---------------------------------------------------------------------------

enum tensor_category {
    CAT_EMBEDDING,     // token_embd
    CAT_OUTPUT,        // output (lm_head)
    CAT_NORM,          // *_norm (rms_norm, layer_norm, etc.)
    CAT_ATTN,          // attn_* (q, k, v, o projections)
    CAT_FFN_DENSE,     // ffn_* that is NOT expert and NOT shared expert (dense FFN layers)
    CAT_FFN_EXPERT,    // ffn_*_exps (MoE expert weights)
    CAT_FFN_SHARED,    // ffn_*_shexp (shared expert weights)
    CAT_MOE_GATE,      // ffn_gate_inp (router)
    CAT_SSM,           // ssm_* (Mamba/state-space model components)
    CAT_OTHER,
};

static const char * cat_name(tensor_category cat) {
    switch (cat) {
        case CAT_EMBEDDING:  return "embedding";
        case CAT_OUTPUT:     return "output_head";
        case CAT_NORM:       return "norm";
        case CAT_ATTN:       return "attention";
        case CAT_FFN_DENSE:  return "ffn_dense";
        case CAT_FFN_EXPERT: return "ffn_expert";
        case CAT_FFN_SHARED: return "ffn_shared_expert";
        case CAT_MOE_GATE:   return "moe_gate";
        case CAT_SSM:        return "ssm";
        case CAT_OTHER:      return "other";
    }
    return "unknown";
}

// Classify a tensor by its name.
// Tensor names in llama.cpp GGUF files follow the pattern:
//   blk.{layer}.{component}.weight   (per-layer)
//   token_embd.weight                 (embedding)
//   output.weight / output_norm.weight
static tensor_category classify(const char * name) {
    // Embedding
    if (strstr(name, "token_embd"))  return CAT_EMBEDDING;

    // Output head
    if (strcmp(name, "output.weight") == 0 || strcmp(name, "output.bias") == 0)
        return CAT_OUTPUT;

    // Norms (must check before attn/ffn since some norms contain "attn" or "ffn")
    if (strstr(name, "_norm"))       return CAT_NORM;

    // MoE router
    if (strstr(name, "ffn_gate_inp")) return CAT_MOE_GATE;

    // MoE expert tensors (merged experts: *_exps)
    if (strstr(name, "_exps"))       return CAT_FFN_EXPERT;

    // Shared expert tensors
    if (strstr(name, "_shexp"))      return CAT_FFN_SHARED;

    // Attention (includes attn_gate for gated attention)
    if (strstr(name, "attn_"))       return CAT_ATTN;

    // SSM / Mamba components
    if (strstr(name, "ssm_"))        return CAT_SSM;

    // Dense FFN (non-MoE feed-forward)
    if (strstr(name, "ffn_"))        return CAT_FFN_DENSE;

    return CAT_OTHER;
}

// ---------------------------------------------------------------------------
// Split GGUF detection
// ---------------------------------------------------------------------------

// Detect split GGUF pattern: *-NNNNN-of-NNNNN.gguf
// Returns true if this is a split file, fills split_no (0-based) and split_count.
static bool detect_split(const std::string & path, int & split_no, int & split_count, std::string & prefix) {
    // Match pattern like: /path/to/Model-Name-00001-of-00004.gguf
    std::regex re(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
    std::smatch m;
    if (std::regex_match(path, m, re)) {
        prefix = m[1].str();
        split_no = std::stoi(m[2].str()) - 1;  // convert to 0-based
        split_count = std::stoi(m[3].str());
        return true;
    }
    return false;
}

static std::string make_split_path(const std::string & prefix, int idx, int count) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%05d-of-%05d", idx + 1, count);
    return prefix + "-" + buf + ".gguf";
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string fmt_size(size_t bytes) {
    char buf[64];
    if (bytes >= (size_t)1 << 30) {
        snprintf(buf, sizeof(buf), "%.2f GiB", (double)bytes / (1 << 30));
    } else if (bytes >= (size_t)1 << 20) {
        snprintf(buf, sizeof(buf), "%.2f MiB", (double)bytes / (1 << 20));
    } else if (bytes >= (size_t)1 << 10) {
        snprintf(buf, sizeof(buf), "%.2f KiB", (double)bytes / (1 << 10));
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

static uint32_t get_u32_kv(const struct gguf_context * ctx, const char * arch, const char * suffix) {
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    int64_t id = gguf_find_key(ctx, key);
    if (id >= 0) return gguf_get_val_u32(ctx, id);
    return 0;
}

static uint16_t get_u16_kv(const struct gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id >= 0) return (uint16_t)gguf_get_val_u16(ctx, id);
    return 0;
}

static std::string get_str_kv(const struct gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id >= 0) return gguf_get_val_str(ctx, id);
    return "";
}

// ---------------------------------------------------------------------------
// Accumulate tensor stats from a single GGUF context
// ---------------------------------------------------------------------------

struct type_stats {
    int count = 0;
    size_t bytes = 0;
};

struct model_stats {
    std::map<tensor_category, size_t> cat_bytes;
    std::map<tensor_category, int>    cat_count;
    std::map<enum ggml_type, type_stats> type_totals;
    std::map<tensor_category, std::map<enum ggml_type, type_stats>> cat_type_breakdown;
    size_t total_bytes = 0;
    int64_t n_tensors = 0;
};

static void accumulate_tensors(struct gguf_context * ctx, model_stats & stats, bool verbose) {
    int64_t n = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n; i++) {
        const char * name    = gguf_get_tensor_name(ctx, i);
        enum ggml_type type  = gguf_get_tensor_type(ctx, i);
        size_t size          = gguf_get_tensor_size(ctx, i);

        tensor_category cat = classify(name);

        stats.cat_bytes[cat] += size;
        stats.cat_count[cat]++;
        stats.type_totals[type].count++;
        stats.type_totals[type].bytes += size;
        stats.cat_type_breakdown[cat][type].count++;
        stats.cat_type_breakdown[cat][type].bytes += size;
        stats.total_bytes += size;
        stats.n_tensors++;

        if (verbose) {
            printf("  %-60s  %-8s  %s\n", name, ggml_type_name(type), fmt_size(size).c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [--verbose]\n", argv[0]);
        fprintf(stderr, "\nEstimate bytes read per forward pass for GGUF models (especially MoE).\n");
        fprintf(stderr, "Only reads metadata — no tensor data is loaded.\n");
        fprintf(stderr, "Supports split GGUF files (pass any shard).\n");
        return 1;
    }

    const char * fname = argv[1];
    bool verbose = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    // Open the first/main GGUF — metadata only
    struct gguf_init_params params = { /* no_alloc */ true, /* ctx */ nullptr };
    struct gguf_context * ctx = gguf_init_from_file(fname, params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to open '%s'\n", fname);
        return 1;
    }

    // Read architecture and model info from the primary shard
    std::string arch = get_str_kv(ctx, "general.architecture");
    std::string model_name = get_str_kv(ctx, "general.name");

    uint32_t n_expert       = get_u32_kv(ctx, arch.c_str(), "expert_count");
    uint32_t n_expert_used  = get_u32_kv(ctx, arch.c_str(), "expert_used_count");
    uint32_t n_expert_shared = get_u32_kv(ctx, arch.c_str(), "expert_shared_count");
    uint32_t n_layer        = get_u32_kv(ctx, arch.c_str(), "block_count");

    bool is_moe = n_expert > 1;

    // Detect split files
    int split_no = 0, split_count = 1;
    std::string split_prefix;
    bool is_split = detect_split(fname, split_no, split_count, split_prefix);

    // Also check the GGUF KV for split info (more authoritative)
    uint16_t kv_split_count = get_u16_kv(ctx, "split.count");
    if (kv_split_count > 1) {
        is_split = true;
        split_count = kv_split_count;
        // Derive prefix from filename if not already detected
        if (split_prefix.empty()) {
            detect_split(fname, split_no, split_count, split_prefix);
        }
    }

    printf("=== Model info ===\n");
    printf("File:          %s\n", fname);
    if (!model_name.empty()) {
        printf("Name:          %s\n", model_name.c_str());
    }
    printf("Architecture:  %s\n", arch.c_str());
    printf("Layers:        %u\n", n_layer);
    if (is_moe) {
        printf("Experts:       %u total, %u active per token", n_expert, n_expert_used);
        if (n_expert_shared > 0) {
            printf(", %u shared (always active)", n_expert_shared);
        }
        printf("\n");
    }
    if (is_split) {
        printf("Split:         %d shards\n", split_count);
    }

    // Accumulate tensor stats across all shards
    model_stats stats;

    // Process the already-opened context (shard 0 or the only file)
    accumulate_tensors(ctx, stats, verbose);
    gguf_free(ctx);

    // If split, open remaining shards
    if (is_split) {
        for (int idx = 0; idx < split_count; idx++) {
            if (idx == split_no) continue;  // already processed

            std::string shard_path = make_split_path(split_prefix, idx, split_count);
            struct gguf_context * shard = gguf_init_from_file(shard_path.c_str(), params);
            if (!shard) {
                fprintf(stderr, "Warning: failed to open shard '%s', skipping\n", shard_path.c_str());
                continue;
            }
            accumulate_tensors(shard, stats, verbose);
            gguf_free(shard);
        }
    }

    // Print type summary
    printf("\n=== Quantization type summary ===\n");
    printf("%-12s  %6s  %12s  %6s\n", "Type", "Count", "Size", "% of total");
    printf("%-12s  %6s  %12s  %6s\n", "----", "-----", "----", "----------");

    std::vector<std::pair<enum ggml_type, type_stats>> type_vec(stats.type_totals.begin(), stats.type_totals.end());
    std::sort(type_vec.begin(), type_vec.end(),
        [](const auto & a, const auto & b) { return a.second.bytes > b.second.bytes; });

    for (const auto & [type, ts] : type_vec) {
        printf("%-12s  %6d  %12s  %5.1f%%\n",
            ggml_type_name(type), ts.count,
            fmt_size(ts.bytes).c_str(),
            100.0 * ts.bytes / stats.total_bytes);
    }
    printf("%-12s  %6lld  %12s\n", "TOTAL", (long long)stats.n_tensors, fmt_size(stats.total_bytes).c_str());

    // Print per-category breakdown
    printf("\n=== Category breakdown ===\n");

    tensor_category cat_order[] = {
        CAT_EMBEDDING, CAT_ATTN, CAT_SSM, CAT_FFN_DENSE, CAT_FFN_EXPERT,
        CAT_FFN_SHARED, CAT_MOE_GATE, CAT_NORM, CAT_OUTPUT, CAT_OTHER
    };

    printf("%-20s  %6s  %12s  %6s\n", "Category", "Count", "Size", "% of total");
    printf("%-20s  %6s  %12s  %6s\n", "--------", "-----", "----", "----------");
    for (auto cat : cat_order) {
        if (stats.cat_count.find(cat) == stats.cat_count.end()) continue;
        printf("%-20s  %6d  %12s  %5.1f%%\n",
            cat_name(cat), stats.cat_count[cat],
            fmt_size(stats.cat_bytes[cat]).c_str(),
            100.0 * stats.cat_bytes[cat] / stats.total_bytes);

        auto & breakdown = stats.cat_type_breakdown[cat];
        if (breakdown.size() > 1) {
            std::vector<std::pair<enum ggml_type, type_stats>> bvec(breakdown.begin(), breakdown.end());
            std::sort(bvec.begin(), bvec.end(),
                [](const auto & a, const auto & b) { return a.second.bytes > b.second.bytes; });
            for (const auto & [type, ts] : bvec) {
                printf("  %-18s  %6d  %12s\n",
                    ggml_type_name(type), ts.count, fmt_size(ts.bytes).c_str());
            }
        }
    }

    // Compute forward pass estimate
    printf("\n=== Forward pass estimate (single token) ===\n");

    size_t always_read = 0;
    size_t expert_total = stats.cat_bytes.count(CAT_FFN_EXPERT) ? stats.cat_bytes[CAT_FFN_EXPERT] : 0;

    for (auto & [cat, bytes] : stats.cat_bytes) {
        if (cat != CAT_FFN_EXPERT) {
            always_read += bytes;
        }
    }

    if (is_moe && n_expert > 0 && expert_total > 0) {
        size_t expert_per_token = expert_total * n_expert_used / n_expert;
        size_t forward_total = always_read + expert_per_token;

        printf("Always read (attn + norms + embed + output + shared experts + gates):\n");
        printf("  %s\n", fmt_size(always_read).c_str());
        printf("Expert data (total for all %u experts across all layers):\n", n_expert);
        printf("  %s\n", fmt_size(expert_total).c_str());
        printf("Expert data per token (%u of %u experts active):\n", n_expert_used, n_expert);
        printf("  %s\n", fmt_size(expert_per_token).c_str());

        printf("\n>>> Estimated bytes read per forward pass: %s <<<\n", fmt_size(forward_total).c_str());
        printf("    (%.1f%% of total model size %s)\n",
            100.0 * forward_total / stats.total_bytes, fmt_size(stats.total_bytes).c_str());

        printf("\n  Breakdown of always-read portion:\n");
        for (auto cat : cat_order) {
            if (cat == CAT_FFN_EXPERT) continue;
            if (stats.cat_bytes.find(cat) == stats.cat_bytes.end()) continue;
            printf("    %-20s  %s\n", cat_name(cat), fmt_size(stats.cat_bytes[cat]).c_str());
        }
    } else {
        printf("Dense model — all weights read every forward pass.\n");
        printf(">>> Total bytes read per forward pass: %s <<<\n", fmt_size(stats.total_bytes).c_str());
    }

    return 0;
}
