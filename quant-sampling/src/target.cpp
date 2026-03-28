#include "target.h"
#include "logits_io.h"

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct target_params {
    std::string model_path;
    std::string input_path;
    std::string output_path = "target.bin";
    int         n_gpu_layers = 99;
    int         n_ctx        = 0; // 0 = auto from input
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
        } else {
            fprintf(stderr, "target: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.model_path.empty() || params.input_path.empty()) {
        fprintf(stderr, "Usage: quant-sampling target -m <model> -i <ref.bin> [options]\n"
                        "  -o <path>    output file (default: target.bin)\n"
                        "  -ngl <int>   GPU layers (default: 99)\n"
                        "  -c <int>     context size (default: auto)\n");
        return false;
    }
    return true;
}

static bool process_one_prompt(
    llama_model * model,
    const qmlog_prompt & ref_prompt,
    const target_params & params,
    int n_vocab,
    qmlog_prompt & out
) {
    const int n_tokens = ref_prompt.n_tokens;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : n_tokens;
    ctx_params.n_batch = n_tokens;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "target: failed to create context\n");
        return false;
    }

    const int n_batch = llama_n_batch(ctx);

    std::vector<float> all_logits;
    all_logits.reserve((size_t)(n_tokens - 1) * n_vocab);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_processed = 0;
    while (n_processed < n_tokens) {
        common_batch_clear(batch);
        int batch_end = std::min(n_tokens, n_processed + n_batch);
        for (int i = n_processed; i < batch_end; i++) {
            common_batch_add(batch, ref_prompt.tokens[i], i, {0}, i > 0);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "target: decode failed at position %d\n", n_processed);
            llama_batch_free(batch); llama_free(ctx);
            return false;
        }
        for (int i = n_processed; i < batch_end; i++) {
            if (i == 0) continue;
            const float * logits = llama_get_logits_ith(ctx, i - n_processed);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        }
        n_processed = batch_end;
        fprintf(stderr, "target: processed %d / %d tokens\r", n_processed, n_tokens);
    }
    fprintf(stderr, "\n");

    out.n_tokens = n_tokens;
    out.n_prompt = ref_prompt.n_prompt;
    out.tokens.assign(ref_prompt.tokens.begin(), ref_prompt.tokens.end());
    out.logits = std::move(all_logits);

    llama_batch_free(batch);
    llama_free(ctx);
    return true;
}

int cmd_target(int argc, char ** argv) {
    target_params params;
    if (!parse_args(argc, argv, params)) return 1;

    // read tokens from reference file (skips logits)
    qmlog_file ref;
    if (!qmlog_read_tokens(params.input_path, ref)) {
        fprintf(stderr, "target: failed to read '%s'\n", params.input_path.c_str());
        return 1;
    }

    fprintf(stderr, "target: read %d prompt(s) from '%s'\n", ref.n_prompts, params.input_path.c_str());

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "target: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (n_vocab != ref.n_vocab) {
        fprintf(stderr, "target: vocab mismatch: model has %d, reference has %d\n", n_vocab, ref.n_vocab);
        llama_model_free(model);
        return 1;
    }

    qmlog_file out;
    out.n_vocab   = n_vocab;
    out.n_prompts = ref.n_prompts;
    out.temp      = 0.0f;  // no sampling
    out.top_p     = 0.0f;
    out.top_k     = 0;
    out.seed      = 0;
    out.prompts.resize(ref.n_prompts);

    for (int pi = 0; pi < ref.n_prompts; pi++) {
        fprintf(stderr, "\n=== Prompt %d / %d (%d tokens) ===\n",
                pi + 1, ref.n_prompts, ref.prompts[pi].n_tokens);
        if (!process_one_prompt(model, ref.prompts[pi], params, n_vocab, out.prompts[pi])) {
            llama_model_free(model);
            return 1;
        }
    }

    if (!qmlog_write(params.output_path, out)) {
        fprintf(stderr, "target: failed to write '%s'\n", params.output_path.c_str());
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "\ntarget: wrote %s (%d prompt(s))\n", params.output_path.c_str(), out.n_prompts);

    llama_model_free(model);
    return 0;
}
