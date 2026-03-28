#include "ref.h"
#include "logits_io.h"

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ref_params {
    std::string model_path;
    std::string prompt;
    std::string output_path = "ref.qmlog";
    int         n_predict   = 256;
    float       temp        = 1.0f;
    float       top_p       = 0.95f;
    int         top_k       = 40;
    uint32_t    seed        = 42;
    int         n_gpu_layers = 99;
    int         n_ctx        = 2048;
    bool        no_chat      = false;
};

static bool parse_args(int argc, char ** argv, ref_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-p") == 0 && i + 1 < argc) {
            params.prompt = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-n") == 0 && i + 1 < argc) {
            params.n_predict = atoi(argv[++i]);
        } else if (strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            params.temp = atof(argv[++i]);
        } else if (strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            params.top_p = atof(argv[++i]);
        } else if (strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            params.seed = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(arg, "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "--no-chat") == 0) {
            params.no_chat = true;
        } else {
            fprintf(stderr, "ref: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.model_path.empty() || params.prompt.empty()) {
        fprintf(stderr, "Usage: quant-sampling ref -m <model> -p <prompt> [options]\n"
                        "  -o <path>     output file (default: ref.qmlog)\n"
                        "  -n <int>      tokens to generate (default: 256)\n"
                        "  --temp <f>    temperature (default: 1.0)\n"
                        "  --top-p <f>   top-p (default: 0.95)\n"
                        "  --top-k <int> top-k (default: 40)\n"
                        "  --seed <int>  RNG seed (default: 42)\n"
                        "  -ngl <int>    GPU layers (default: 99)\n"
                        "  -c <int>      context size (default: 2048)\n"
                        "  --no-chat     skip chat template formatting\n");
        return false;
    }
    return true;
}

int cmd_ref(int argc, char ** argv) {
    ref_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // init backends
    ggml_backend_load_all();

    // load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "ref: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // apply chat template if available
    std::string formatted_prompt = params.prompt;
    if (!params.no_chat) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl) {
            llama_chat_message msg = {"user", params.prompt.c_str()};
            // first call to get required size
            int32_t len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
            if (len > 0) {
                std::vector<char> buf(len + 1);
                llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
                formatted_prompt.assign(buf.data(), len);
                fprintf(stderr, "ref: applied chat template (%d chars)\n", len);
            } else {
                fprintf(stderr, "ref: warning: chat template failed, using raw prompt\n");
            }
        } else {
            fprintf(stderr, "ref: no chat template in model, using raw prompt\n");
        }
    }

    // tokenize prompt
    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, formatted_prompt, true, true);
    const int n_prompt = (int)prompt_tokens.size();

    fprintf(stderr, "ref: n_vocab=%d, n_prompt=%d, n_predict=%d\n",
            n_vocab, n_prompt, params.n_predict);

    const int total_tokens = n_prompt + params.n_predict;

    // create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = std::max(params.n_ctx, total_tokens);
    ctx_params.n_batch = std::max(params.n_ctx, total_tokens);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ref: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int n_batch = llama_n_batch(ctx);

    // prepare output
    std::vector<int32_t> all_tokens(prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.reserve(total_tokens);

    // we'll collect (total_tokens - 1) logit vectors
    std::vector<float> all_logits;
    all_logits.reserve((size_t)(total_tokens - 1) * n_vocab);

    // build sampler chain: top_k -> top_p -> temp -> dist
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    // --- decode prompt in batches, collecting logits at every position ---
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_decoded = 0;
    while (n_decoded < n_prompt) {
        common_batch_clear(batch);

        int batch_end = std::min(n_prompt, n_decoded + n_batch);
        for (int i = n_decoded; i < batch_end; i++) {
            // collect logits at every position (except the very first token, position 0,
            // which has no "previous" token to predict)
            common_batch_add(batch, prompt_tokens[i], i, {0}, i > 0);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "ref: decode failed at prompt position %d\n", n_decoded);
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        // collect logits for positions that had logits=true
        for (int i = n_decoded; i < batch_end; i++) {
            if (i == 0) continue; // no logits for position 0
            const float * logits = llama_get_logits_ith(ctx, i - n_decoded);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        }

        n_decoded = batch_end;
    }

    fprintf(stderr, "ref: prompt decoded (%d tokens), starting generation...\n", n_prompt);

    // --- autoregressive generation ---
    for (int i = 0; i < params.n_predict; i++) {
        // sample from the last logits
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            fprintf(stderr, "\nref: EOS at step %d\n", i);
            break;
        }

        all_tokens.push_back(new_token);

        // print token
        std::string piece = common_token_to_piece(vocab, new_token);
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);

        // decode single token
        common_batch_clear(batch);
        common_batch_add(batch, new_token, n_prompt + i, {0}, true);

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nref: decode failed at generation step %d\n", i);
            break;
        }

        // collect logits
        const float * logits = llama_get_logits_ith(ctx, 0);
        all_logits.insert(all_logits.end(), logits, logits + n_vocab);
    }

    fprintf(stdout, "\n");

    const int final_n_tokens = (int)all_tokens.size();

    fprintf(stderr, "ref: total tokens = %d, logit vectors = %d\n",
            final_n_tokens, (int)(all_logits.size() / n_vocab));

    // --- write .qmlog ---
    qmlog_file out;
    out.header.version  = QMLOG_VERSION;
    out.header.n_vocab  = n_vocab;
    out.header.n_tokens = final_n_tokens;
    out.header.n_prompt = n_prompt;
    out.header.temp     = params.temp;
    out.header.top_p    = params.top_p;
    out.header.top_k    = params.top_k;
    out.header.seed     = params.seed;
    out.tokens          = std::move(all_tokens);
    out.logits          = std::move(all_logits);

    if (!qmlog_write(params.output_path, out)) {
        fprintf(stderr, "ref: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "ref: wrote %s (%.1f MB)\n",
            params.output_path.c_str(),
            (double)(QMLOG_HEADER_SZ + final_n_tokens * 4 +
                     (size_t)(final_n_tokens - 1) * n_vocab * 4) / (1024 * 1024));

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
