#include "target.h"
#include "trace_io.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct target_params {
    std::string model_path;
    std::string input_path;
    std::string output_path = "target.bin";
    int32_t     n_gpu_layers = 99;
    int32_t     n_ctx        = 0;
    int32_t     n_threads    = 0;
};

static bool parse_args(int argc, char ** argv, target_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-i") == 0 && i + 1 < argc) {
            params.input_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "target: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.model_path.empty() || params.input_path.empty()) {
        fprintf(stderr, "Usage: kv-transfer target -m <model> -i <ref.bin> [options]\n"
                        "  -o <path>    output file (default: target.bin)\n"
                        "  -ngl <int>   GPU layers (default: 99)\n"
                        "  -c <int>     context size (default: auto)\n"
                        "  -t <int>     threads (default: llama.cpp default)\n");
        return false;
    }
    return true;
}

int cmd_target(int argc, char ** argv) {
    target_params params;
    if (!parse_args(argc, argv, params)) return 1;

    // read tokens from reference (skip logits)
    trace_file ref;
    if (!trace_read_tokens(params.input_path, ref)) {
        fprintf(stderr, "target: failed to read '%s'\n", params.input_path.c_str());
        return 1;
    }

    if (ref.n_prompts != 1) {
        fprintf(stderr, "target: expected single-prompt file, got %d\n", ref.n_prompts);
        return 1;
    }

    const auto & rp = ref.prompts[0];
    const int32_t n_tokens = rp.n_tokens;

    fprintf(stderr, "target: read %d tokens from '%s'\n", n_tokens, params.input_path.c_str());

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "target: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (n_vocab != ref.n_vocab) {
        fprintf(stderr, "target: vocab mismatch: model=%d, ref=%d\n", n_vocab, ref.n_vocab);
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : n_tokens;
    ctx_params.n_batch = n_tokens;
    if (params.n_threads > 0) {
        ctx_params.n_threads       = params.n_threads;
        ctx_params.n_threads_batch = params.n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "target: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int32_t n_batch = llama_n_batch(ctx);

    std::vector<float> all_logits;
    all_logits.reserve((size_t)(n_tokens - 1) * n_vocab);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int32_t n_processed = 0;
    while (n_processed < n_tokens) {
        common_batch_clear(batch);
        int32_t batch_end = std::min(n_tokens, n_processed + n_batch);
        for (int32_t i = n_processed; i < batch_end; i++) {
            common_batch_add(batch, rp.tokens[i], i, {0}, i > 0);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "target: decode failed at position %d\n", n_processed);
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        for (int32_t i = n_processed; i < batch_end; i++) {
            if (i == 0) continue;
            const float * logits = llama_get_logits_ith(ctx, i - n_processed);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        }
        n_processed = batch_end;
        fprintf(stderr, "target: processed %d / %d tokens\r", n_processed, n_tokens);
    }
    fprintf(stderr, "\n");

    // write output
    trace_file out;
    out.n_vocab   = n_vocab;
    out.n_prompts = 1;

    trace_entry & p = out.prompts.emplace_back();
    p.path     = rp.path;
    p.n_tokens = n_tokens;
    p.n_prompt = rp.n_prompt;
    p.tokens.assign(rp.tokens.begin(), rp.tokens.end());
    p.logits   = std::move(all_logits);

    if (!trace_write(params.output_path, out)) {
        fprintf(stderr, "target: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "target: wrote %s (%d tokens)\n", params.output_path.c_str(), n_tokens);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
