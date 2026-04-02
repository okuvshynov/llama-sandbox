#include "handoff.h"
#include "trace_io.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct handoff_params {
    std::string ref_model_path;
    std::string tgt_model_path;
    std::string input_path;       // ref.bin with token sequence
    std::string output_path = "handoff.bin";
    std::string state_path  = "";  // temp file for KV state (auto if empty)
    int32_t     n_gpu_layers = 99;
    int32_t     n_ctx        = 0;
    int32_t     n_threads    = 0;
};

static bool parse_args(int argc, char ** argv, handoff_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m-ref") == 0 && i + 1 < argc) {
            params.ref_model_path = argv[++i];
        } else if (strcmp(arg, "-m-tgt") == 0 && i + 1 < argc) {
            params.tgt_model_path = argv[++i];
        } else if (strcmp(arg, "-i") == 0 && i + 1 < argc) {
            params.input_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (strcmp(arg, "--state") == 0 && i + 1 < argc) {
            params.state_path = argv[++i];
        } else if (strcmp(arg, "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "handoff: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.ref_model_path.empty() || params.tgt_model_path.empty() || params.input_path.empty()) {
        fprintf(stderr, "Usage: kv-transfer handoff -m-ref <ref_model> -m-tgt <tgt_model> -i <ref.bin> [options]\n"
                        "  -o <path>      output file (default: handoff.bin)\n"
                        "  --state <path> KV state file (default: auto temp file)\n"
                        "  -ngl <int>     GPU layers (default: 99)\n"
                        "  -c <int>       context size (default: auto)\n"
                        "  -t <int>       threads (default: llama.cpp default)\n");
        return false;
    }
    if (params.state_path.empty()) {
        params.state_path = params.output_path + ".state.tmp";
    }
    return true;
}

int cmd_handoff(int argc, char ** argv) {
    handoff_params params;
    if (!parse_args(argc, argv, params)) return 1;

    // read token sequence from reference
    trace_file ref;
    if (!trace_read_tokens(params.input_path, ref)) {
        fprintf(stderr, "handoff: failed to read '%s'\n", params.input_path.c_str());
        return 1;
    }

    if (ref.n_prompts != 1) {
        fprintf(stderr, "handoff: expected single-prompt file, got %d\n", ref.n_prompts);
        return 1;
    }

    const auto & rp = ref.prompts[0];
    const int32_t n_tokens = rp.n_tokens;
    const int32_t n_prompt = rp.n_prompt;
    const int32_t n_ctx = params.n_ctx > 0 ? params.n_ctx : n_tokens;

    fprintf(stderr, "handoff: %d tokens (%d prompt + %d generated)\n",
            n_tokens, n_prompt, n_tokens - n_prompt);

    ggml_backend_load_all();

    // =========================================================================
    // Phase 1: Process prompt with reference model, save KV state
    // =========================================================================

    fprintf(stderr, "\n=== Phase 1: prompt processing with ref model ===\n");

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * ref_model = llama_model_load_from_file(params.ref_model_path.c_str(), model_params);
    if (!ref_model) {
        fprintf(stderr, "handoff: failed to load ref model '%s'\n", params.ref_model_path.c_str());
        return 1;
    }

    const llama_vocab * ref_vocab = llama_model_get_vocab(ref_model);
    const int32_t n_vocab = llama_vocab_n_tokens(ref_vocab);

    if (n_vocab != ref.n_vocab) {
        fprintf(stderr, "handoff: vocab mismatch: ref model=%d, ref.bin=%d\n", n_vocab, ref.n_vocab);
        llama_model_free(ref_model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = n_ctx;
    ctx_params.n_batch = n_ctx;
    if (params.n_threads > 0) {
        ctx_params.n_threads       = params.n_threads;
        ctx_params.n_threads_batch = params.n_threads;
    }

    llama_context * ref_ctx = llama_init_from_model(ref_model, ctx_params);
    if (!ref_ctx) {
        fprintf(stderr, "handoff: failed to create ref context\n");
        llama_model_free(ref_model);
        return 1;
    }

    const int32_t n_batch = llama_n_batch(ref_ctx);

    // decode prompt with ref model (logits not stored — only need KV cache)
    const int32_t n_gen = n_tokens - n_prompt;
    std::vector<float> all_logits;
    all_logits.reserve((size_t)n_gen * n_vocab);

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int32_t n_decoded = 0;
    while (n_decoded < n_prompt) {
        common_batch_clear(batch);
        int32_t batch_end = std::min(n_prompt, n_decoded + n_batch);
        for (int32_t i = n_decoded; i < batch_end; i++) {
            common_batch_add(batch, rp.tokens[i], i, {0}, false);
        }
        if (llama_decode(ref_ctx, batch) != 0) {
            fprintf(stderr, "handoff: ref decode failed at position %d\n", n_decoded);
            llama_batch_free(batch); llama_free(ref_ctx); llama_model_free(ref_model);
            return 1;
        }
        n_decoded = batch_end;
    }

    fprintf(stderr, "handoff: prompt decoded (%d tokens), saving KV state...\n", n_prompt);

    // save KV state — includes prompt tokens
    std::vector<llama_token> prompt_tok(rp.tokens.begin(), rp.tokens.begin() + n_prompt);
    size_t state_size = llama_state_save_file(ref_ctx, params.state_path.c_str(),
                                               prompt_tok.data(), n_prompt);
    if (state_size == 0) {
        fprintf(stderr, "handoff: failed to save KV state to '%s'\n", params.state_path.c_str());
        llama_batch_free(batch); llama_free(ref_ctx); llama_model_free(ref_model);
        return 1;
    }

    fprintf(stderr, "handoff: saved KV state (%.1f MB) to '%s'\n",
            (double)state_size / (1024 * 1024), params.state_path.c_str());

    // free ref model
    llama_batch_free(batch);
    llama_free(ref_ctx);
    llama_model_free(ref_model);

    // =========================================================================
    // Phase 2: Load target model, restore KV state, replay generation tokens
    // =========================================================================

    fprintf(stderr, "\n=== Phase 2: generation with target model (using ref KV) ===\n");

    llama_model * tgt_model = llama_model_load_from_file(params.tgt_model_path.c_str(), model_params);
    if (!tgt_model) {
        fprintf(stderr, "handoff: failed to load target model '%s'\n", params.tgt_model_path.c_str());
        return 1;
    }

    const llama_vocab * tgt_vocab = llama_model_get_vocab(tgt_model);
    const int32_t tgt_n_vocab = llama_vocab_n_tokens(tgt_vocab);

    if (tgt_n_vocab != n_vocab) {
        fprintf(stderr, "handoff: vocab mismatch: target=%d, ref=%d\n", tgt_n_vocab, n_vocab);
        llama_model_free(tgt_model);
        return 1;
    }

    llama_context * tgt_ctx = llama_init_from_model(tgt_model, ctx_params);
    if (!tgt_ctx) {
        fprintf(stderr, "handoff: failed to create target context\n");
        llama_model_free(tgt_model);
        return 1;
    }

    // restore KV state
    std::vector<llama_token> loaded_tokens(n_prompt);
    size_t n_loaded = 0;
    size_t loaded = llama_state_load_file(tgt_ctx, params.state_path.c_str(),
                                           loaded_tokens.data(), n_prompt, &n_loaded);
    if (loaded == 0) {
        fprintf(stderr, "handoff: failed to load KV state from '%s'\n", params.state_path.c_str());
        llama_free(tgt_ctx); llama_model_free(tgt_model);
        return 1;
    }

    fprintf(stderr, "handoff: restored KV state (%zu tokens)\n", n_loaded);

    // now replay the generation tokens through target model, collecting logits
    // the KV cache already has the prompt, so we feed generation tokens one by one
    batch = llama_batch_init(1, 0, 1);

    for (int32_t i = 0; i < n_gen; i++) {
        int32_t pos = n_prompt + i;
        llama_token tok = rp.tokens[pos];

        common_batch_clear(batch);
        common_batch_add(batch, tok, pos, {0}, true);

        if (llama_decode(tgt_ctx, batch) != 0) {
            fprintf(stderr, "handoff: target decode failed at generation step %d\n", i);
            break;
        }

        const float * logits = llama_get_logits_ith(tgt_ctx, 0);
        all_logits.insert(all_logits.end(), logits, logits + n_vocab);

        if ((i + 1) % 100 == 0 || i == n_gen - 1) {
            fprintf(stderr, "handoff: generation %d / %d\r", i + 1, n_gen);
            fflush(stderr);
        }
    }
    fprintf(stderr, "\n");

    // clean up temp state file
    remove(params.state_path.c_str());

    // write output
    trace_file out;
    out.n_vocab   = n_vocab;
    out.n_prompts = 1;
    out.temp      = ref.temp;
    out.top_p     = ref.top_p;
    out.top_k     = ref.top_k;
    out.seed      = ref.seed;

    trace_entry & p = out.prompts.emplace_back();
    p.path     = rp.path;
    p.n_tokens = n_tokens;
    p.n_prompt = n_prompt;
    p.tokens.assign(rp.tokens.begin(), rp.tokens.end());
    p.logits   = std::move(all_logits);

    if (!trace_write(params.output_path, out)) {
        fprintf(stderr, "handoff: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch); llama_free(tgt_ctx); llama_model_free(tgt_model);
        return 1;
    }

    fprintf(stderr, "handoff: wrote %s (%d tokens, %d generation logits)\n",
            params.output_path.c_str(), n_tokens, n_gen);

    llama_batch_free(batch);
    llama_free(tgt_ctx);
    llama_model_free(tgt_model);
    return 0;
}
