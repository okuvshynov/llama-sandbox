#include "collect.h"
#include "logits_io.h"

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct collect_params {
    std::string model_path;
    std::string input_path;
    std::string output_path = "quant.qmlog";
    int         n_gpu_layers = 99;
    int         n_ctx        = 0; // 0 = auto from input
};

static bool parse_args(int argc, char ** argv, collect_params & params) {
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
            fprintf(stderr, "collect: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.model_path.empty() || params.input_path.empty()) {
        fprintf(stderr, "Usage: quant-sampling collect -m <model> -i <ref.qmlog> [options]\n"
                        "  -o <path>    output file (default: quant.qmlog)\n"
                        "  -ngl <int>   GPU layers (default: 99)\n"
                        "  -c <int>     context size (default: auto)\n");
        return false;
    }
    return true;
}

int cmd_collect(int argc, char ** argv) {
    collect_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // read tokens from reference file
    qmlog_file ref;
    if (!qmlog_read_tokens(params.input_path, ref)) {
        fprintf(stderr, "collect: failed to read '%s'\n", params.input_path.c_str());
        return 1;
    }

    const int n_tokens = ref.header.n_tokens;
    fprintf(stderr, "collect: read %d tokens from '%s'\n", n_tokens, params.input_path.c_str());

    // init backends
    ggml_backend_load_all();

    // load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "collect: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // verify vocab matches
    if (n_vocab != ref.header.n_vocab) {
        fprintf(stderr, "collect: vocab mismatch: model has %d, reference has %d\n",
                n_vocab, ref.header.n_vocab);
        llama_model_free(model);
        return 1;
    }

    // create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = params.n_ctx > 0 ? params.n_ctx : n_tokens;
    ctx_params.n_batch = n_tokens; // try to fit everything in one batch

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "collect: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int n_batch = llama_n_batch(ctx);

    fprintf(stderr, "collect: n_vocab=%d, n_tokens=%d, n_batch=%d\n", n_vocab, n_tokens, n_batch);

    // allocate logit storage: (n_tokens - 1) logit vectors
    std::vector<float> all_logits;
    all_logits.reserve((size_t)(n_tokens - 1) * n_vocab);

    // process tokens in batches, continuing KV cache
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_processed = 0;
    while (n_processed < n_tokens) {
        common_batch_clear(batch);

        int batch_end = std::min(n_tokens, n_processed + n_batch);
        for (int i = n_processed; i < batch_end; i++) {
            // collect logits at every position except position 0
            common_batch_add(batch, ref.tokens[i], i, {0}, i > 0);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "collect: decode failed at position %d\n", n_processed);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // collect logits
        for (int i = n_processed; i < batch_end; i++) {
            if (i == 0) continue;
            const float * logits = llama_get_logits_ith(ctx, i - n_processed);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        }

        n_processed = batch_end;

        fprintf(stderr, "collect: processed %d / %d tokens\r", n_processed, n_tokens);
    }
    fprintf(stderr, "\n");

    // write output
    qmlog_file out;
    out.header.version  = QMLOG_VERSION;
    out.header.n_vocab  = n_vocab;
    out.header.n_tokens = n_tokens;
    out.header.n_prompt = ref.header.n_prompt;
    out.header.temp     = 0.0f; // no sampling
    out.header.top_p    = 0.0f;
    out.header.top_k    = 0;
    out.header.seed     = 0;
    out.tokens          = std::move(ref.tokens);
    out.logits          = std::move(all_logits);

    if (!qmlog_write(params.output_path, out)) {
        fprintf(stderr, "collect: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "collect: wrote %s (%.1f MB)\n",
            params.output_path.c_str(),
            (double)(QMLOG_HEADER_SZ + n_tokens * 4 +
                     (size_t)(n_tokens - 1) * n_vocab * 4) / (1024 * 1024));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
