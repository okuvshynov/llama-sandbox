// nano-probe — load a model with the real loader and report what it is.
//
// Bringing up an architecture, the first question is not "does it generate
// text" but "does the loader agree with the file": every tensor found at the
// name and shape the hparams imply, every assert passed. That is a minute of
// mmap and a page of output, and it fails loudly and specifically where a
// half-built graph would fail vaguely and late.
//
// It also prints the expert footprint and what fraction of it fits in the
// GPUs present, because that number decides whether offload is worth
// measuring on a given model at all (OPTIMIZATION.md).
//
// moe_proto.h first: winsock2.h must precede the windows.h that gguf_store.h
// pulls in, even though nothing here speaks the protocol.

#include "moe_proto.h"

#include "build_info.h"
#include "cpu_topology.h"
#include "models/deepseek4/model.h"
#include "models/glm_dsa/model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static const char * type_name(ggml_type t) { return ggml_type_name(t); }

static void print_common(const gguf_store & S, const char * arch, uint32_t n_layer) {
    printf("  arch          %s\n", arch);
    printf("  shards        %u, %.2f GiB mapped\n", S.n_shards, S.bytes_mapped / 1073741824.0);
    printf("  layers        %u\n", n_layer);
    printf("  tensors       %zu\n", S.tensors.size());
}

// Bytes of routed-expert weight per layer, and how much of the whole model
// that is. This is the number that decides how much a GPU can hold.
static void print_experts(const char * label, ggml_tensor * up, ggml_tensor * gate,
                          ggml_tensor * down, uint32_t n_layer, uint64_t total_bytes) {
    const size_t per_layer = ggml_nbytes(up) + ggml_nbytes(gate) + ggml_nbytes(down);
    const size_t all       = per_layer * n_layer;
    const int64_t n_expert = up->ne[2];
    printf("\n  %s\n", label);
    printf("    type          %s (up/gate/down)\n", type_name(up->type));
    printf("    per expert    %.2f MiB\n", per_layer / 1048576.0 / (double) n_expert);
    printf("    per layer     %.2f GiB  (%" PRId64 " experts)\n", per_layer / 1073741824.0, n_expert);
    printf("    all layers    %.2f GiB  = %.1f%% of the model\n",
           all / 1073741824.0, 100.0 * all / (double) total_bytes);

    // What the GPUs present could hold. Reported per die and in total, since
    // residency is per-device: an expert has to fit somewhere, not on average.
    ggml_backend_load_all();
    size_t vram = 0;
    int n_gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto ty = ggml_backend_dev_type(dev);
        if (ty != GGML_BACKEND_DEVICE_TYPE_GPU && ty != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        vram += free_mem;
        n_gpu++;
    }
    if (n_gpu == 0) {
        printf("    residency     no GPU in this build (build.ps1 -Vk to see one)\n");
        return;
    }
    const double frac = (double) vram / (double) all;
    printf("    residency     %.1f GiB across %d GPU(s) holds %.1f%% of the experts\n",
           vram / 1073741824.0, n_gpu, 100.0 * (frac > 1.0 ? 1.0 : frac));
    printf("                  = %d of %" PRId64 " experts per layer\n",
           (int) ((frac > 1.0 ? 1.0 : frac) * n_expert), n_expert);
}

static void probe_glm(const std::string & path) {
    nano_model M;
    load_model(M, path);
    const nano_hparams & h = M.h;
    printf("\n%s\n", M.desc.c_str());
    print_common(M, h.arch.c_str(), h.n_layer);
    printf("  n_embd        %u\n", h.n_embd);
    printf("  n_vocab       %u\n", h.n_vocab);
    printf("  experts       %u used of %u, n_ff_exp %u, dense lead %u\n",
           h.n_expert_used, h.n_expert, h.n_ff_exp, h.n_dense_lead);
    printf("  expert gate   sigmoid, norm=%d scale=%.3f\n", (int) h.expert_norm, h.expert_scale);
    const nano_layer & L = M.layers[h.n_dense_lead];
    print_experts("routed experts", L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps,
                  h.n_layer - h.n_dense_lead, M.bytes_mapped);
}

static void probe_ds4(const std::string & path) {
    ds4_model M;
    ds4_load_model(M, path);
    const ds4_hparams & h = M.h;
    printf("\n%s\n", M.desc.c_str());
    print_common(M, h.arch.c_str(), h.n_layer);
    printf("  n_embd        %u\n", h.n_embd);
    printf("  n_vocab       %u\n", h.n_vocab);
    printf("  n_ctx_train   %u\n", h.n_ctx_train);
    printf("  attention     %u heads / %u kv, d_key %u, q_lora %u, out_lora %u x %u groups\n",
           h.n_head, h.n_head_kv, h.d_key, h.q_lora_rank, h.out_lora_rank, h.out_group_count);
    printf("  indexer       %u heads, key %u, top_k %u\n", h.idx_n_head, h.idx_key_len, h.idx_top_k);
    printf("  rope          yarn, dim %u, base %.0f, factor %.1f, orig ctx %u\n",
           h.rope_dim, h.rope_freq_base, h.rope_yarn_factor, h.rope_yarn_orig_ctx);
    printf("  hyper-conn    %u streams, %u sinkhorn iters, eps %.1e\n",
           h.hc_mult, h.hc_sinkhorn_iters, h.hc_eps);
    printf("  experts       %u used of %u, %u shared, n_ff_exp %u\n",
           h.n_expert_used, h.n_expert, h.n_expert_shared, h.n_ff_exp);
    printf("  expert gate   sqrt-softplus (func %u), norm=%d scale=%.3f\n",
           h.expert_gating, (int) h.expert_norm, h.expert_scale);
    printf("  swiglu clamp  %.1f routed / %.1f shared\n",
           h.swiglu_clamp_exp[0], h.swiglu_clamp_shexp[0]);

    // The layer taxonomy, derived rather than assumed — this is the thing most
    // likely to be wrong in a new port, so print it and eyeball it once.
    uint32_t n_cmp = 0, n_idx = 0, n_hash = 0;
    for (uint32_t i = 0; i < h.n_layer; i++) {
        n_cmp  += h.has_compressor(i);
        n_idx  += h.has_indexer(i);
        n_hash += h.is_hash_routed(i);
    }
    printf("  layer kinds   %u with a KV compressor, %u with a DSA indexer, "
           "%u hash-routed\n", n_cmp, n_idx, n_hash);
    printf("  compress      ");
    for (uint32_t i = 0; i < h.n_layer && i < 24; i++) printf("%u ", h.compress_ratio[i]);
    printf("%s\n", h.n_layer > 24 ? "..." : "");

    const ds4_layer & L = M.layers[0];
    print_experts("routed experts", L.ffn_up_exps, L.ffn_gate_exps, L.ffn_down_exps,
                  h.n_layer, M.bytes_mapped);
}

int main(int argc, char ** argv) {
    std::string path;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--version")) { fputs(nano_build_info().c_str(), stdout); return 0; }
    }
    if (path.empty()) {
        fprintf(stderr,
            "Usage: nano-probe -m <first-shard.gguf>\n"
            "  Loads the model with the real loader and prints its structure,\n"
            "  expert footprint, and how much of it would fit in the GPUs here.\n");
        return 1;
    }

    // Peek at the architecture before choosing a loader: each one hard-aborts
    // on the wrong arch, which is the right behaviour but a poor error here.
    ggml_context * ctx = nullptr;
    gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) { fprintf(stderr, "nano-probe: cannot read GGUF '%s'\n", path.c_str()); return 1; }
    const std::string arch = kv_str_opt(g, "general.architecture", "?");
    gguf_free(g);
    ggml_free(ctx);

    const auto t0 = clk::now();
    if      (arch == "glm-dsa")   probe_glm(path);
    else if (arch == "deepseek4") probe_ds4(path);
    else {
        fprintf(stderr, "nano-probe: architecture '%s' has no loader here "
                        "(have: glm-dsa, deepseek4)\n", arch.c_str());
        return 1;
    }
    printf("\n  loaded in     %.1fs (mmap is lazy; nothing was read)\n",
           std::chrono::duration<double>(clk::now() - t0).count());
    return 0;
}
