#include "target.h"
#include "trace_io.h"
#include "stats_io.h"
#include "kl_utils.h"

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

    // read reference trace with full logits (needed for inline KL computation)
    trace_file ref;
    if (!trace_read(params.input_path, ref)) {
        fprintf(stderr, "target: failed to read '%s'\n", params.input_path.c_str());
        return 1;
    }

    if (ref.n_prompts != 1) {
        fprintf(stderr, "target: expected single-prompt file, got %d\n", ref.n_prompts);
        return 1;
    }

    const auto & rp = ref.prompts[0];
    const int32_t n_tokens = rp.n_tokens;
    const int32_t n_prompt = rp.n_prompt;
    const int32_t n_gen    = n_tokens - n_prompt;
    const int32_t n_vocab  = ref.n_vocab;
    const double  temp     = (ref.temp > 0.0f) ? (double)ref.temp : 1.0;

    fprintf(stderr, "target: read %d tokens (%d prompt + %d gen) from '%s'\n",
            n_tokens, n_prompt, n_gen, params.input_path.c_str());

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "target: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t model_n_vocab = llama_vocab_n_tokens(vocab);

    if (model_n_vocab != n_vocab) {
        fprintf(stderr, "target: vocab mismatch: model=%d, ref=%d\n", model_n_vocab, n_vocab);
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

    // per-token stats
    std::vector<float>   kl_per_token(n_gen);
    std::vector<uint8_t> top1_per_token(n_gen);

    std::vector<double> log_p_ref, log_p_tgt;

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // decode all tokens, compute KL for generation positions inline
    int32_t n_processed = 0;
    while (n_processed < n_tokens) {
        common_batch_clear(batch);
        int32_t batch_end = std::min(n_tokens, n_processed + n_batch);
        for (int32_t i = n_processed; i < batch_end; i++) {
            bool need_logits = (i >= n_prompt);
            common_batch_add(batch, rp.tokens[i], i, {0}, need_logits);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "target: decode failed at position %d\n", n_processed);
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        // compute KL for generation positions in this batch
        for (int32_t i = n_processed; i < batch_end; i++) {
            if (i < n_prompt) continue;
            int32_t gen_idx = i - n_prompt;

            const float * tgt_logits = llama_get_logits_ith(ctx, i - n_processed);
            const float * ref_logits = rp.logits.data() + (size_t)gen_idx * n_vocab;

            log_softmax_temp(ref_logits, n_vocab, temp, log_p_ref);
            log_softmax_temp(tgt_logits, n_vocab, temp, log_p_tgt);

            kl_per_token[gen_idx] = (float)kl_divergence(log_p_ref, log_p_tgt, n_vocab);
            top1_per_token[gen_idx] = (argmax(ref_logits, n_vocab) == argmax(tgt_logits, n_vocab)) ? 1 : 0;
        }
        n_processed = batch_end;
        fprintf(stderr, "target: processed %d / %d tokens\r", n_processed, n_tokens);
    }
    fprintf(stderr, "\n");

    // write stats
    stats_file out;
    out.n_gen    = n_gen;
    out.n_prompt = n_prompt;
    out.temp     = (float)temp;
    out.kl         = std::move(kl_per_token);
    out.top1_match = std::move(top1_per_token);

    if (!stats_write(params.output_path, out)) {
        fprintf(stderr, "target: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "target: wrote %s (%d gen tokens, per-token stats)\n",
            params.output_path.c_str(), n_gen);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
