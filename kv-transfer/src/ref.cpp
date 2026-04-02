#include "ref.h"
#include "trace_io.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ref_params {
    std::string model_path;
    std::string prompt;
    std::string output_path = "ref.bin";
    int32_t     n_predict    = 256;
    float       temp         = 1.0f;
    float       top_p        = 0.95f;
    int32_t     top_k        = 40;
    uint32_t    seed         = 42;
    int32_t     n_gpu_layers = 99;
    int32_t     n_ctx        = 2048;
    int32_t     n_threads    = 0;   // 0 = llama.cpp default
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
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else if (strcmp(arg, "--no-chat") == 0) {
            params.no_chat = true;
        } else {
            fprintf(stderr, "ref: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.model_path.empty() || params.prompt.empty()) {
        fprintf(stderr, "Usage: kv-transfer ref -m <model> -p <prompt> [options]\n"
                        "  -o <path>     output file (default: ref.bin)\n"
                        "  -n <int>      tokens to generate (default: 256)\n"
                        "  --temp <f>    temperature (default: 1.0)\n"
                        "  --top-p <f>   top-p (default: 0.95)\n"
                        "  --top-k <int> top-k (default: 40)\n"
                        "  --seed <int>  RNG seed (default: 42)\n"
                        "  -ngl <int>    GPU layers (default: 99)\n"
                        "  -c <int>      context size (default: 2048)\n"
                        "  -t <int>      threads (default: llama.cpp default)\n"
                        "  --no-chat     skip chat template formatting\n");
        return false;
    }
    return true;
}

int cmd_ref(int argc, char ** argv) {
    ref_params params;
    if (!parse_args(argc, argv, params)) return 1;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "ref: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // apply chat template
    std::string formatted = params.prompt;
    if (!params.no_chat) {
        const char * tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl) {
            llama_chat_message msg = {"user", params.prompt.c_str()};
            int32_t len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
            if (len > 0) {
                std::vector<char> buf(len + 1);
                llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
                formatted.assign(buf.data(), len);
                fprintf(stderr, "ref: applied chat template (%d chars)\n", len);
            }
        }
    }

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, formatted, true, true);
    const int32_t n_prompt = (int32_t)prompt_tokens.size();
    const int32_t total_tokens = n_prompt + params.n_predict;

    fprintf(stderr, "ref: n_vocab=%d, n_prompt=%d, n_predict=%d\n", n_vocab, n_prompt, params.n_predict);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = std::max(params.n_ctx, total_tokens);
    ctx_params.n_batch = std::max(params.n_ctx, total_tokens);
    if (params.n_threads > 0) {
        ctx_params.n_threads       = params.n_threads;
        ctx_params.n_threads_batch = params.n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ref: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int32_t n_batch = llama_n_batch(ctx);

    std::vector<int32_t> all_tokens(prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.reserve(total_tokens);

    std::vector<float> all_logits;
    all_logits.reserve((size_t)params.n_predict * n_vocab);

    // sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    // decode prompt
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int32_t n_decoded = 0;
    while (n_decoded < n_prompt) {
        common_batch_clear(batch);
        int32_t batch_end = std::min(n_prompt, n_decoded + n_batch);
        for (int32_t i = n_decoded; i < batch_end; i++) {
            common_batch_add(batch, prompt_tokens[i], i, {0}, i > 0);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "ref: decode failed at position %d\n", n_decoded);
            llama_batch_free(batch); llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        // prompt logits not stored — only need decode for KV cache + sampling
        n_decoded = batch_end;
    }

    fprintf(stderr, "ref: prompt decoded, generating...\n");

    // generate
    std::string generated_text;
    for (int32_t i = 0; i < params.n_predict; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            fprintf(stderr, "\nref: EOS at step %d\n", i);
            break;
        }
        all_tokens.push_back(new_token);

        std::string piece = common_token_to_piece(vocab, new_token);
        generated_text += piece;
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);

        common_batch_clear(batch);
        common_batch_add(batch, new_token, n_prompt + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\nref: decode failed at generation step %d\n", i);
            break;
        }
        const float * logits = llama_get_logits_ith(ctx, 0);
        all_logits.insert(all_logits.end(), logits, logits + n_vocab);
    }
    fprintf(stdout, "\n");

    // write output (single-prompt v3 format)
    trace_file out;
    out.n_vocab   = n_vocab;
    out.n_prompts = 1;
    out.temp      = params.temp;
    out.top_p     = params.top_p;
    out.top_k     = params.top_k;
    out.seed      = params.seed;

    trace_entry & p = out.prompts.emplace_back();
    p.path     = "inline";
    p.n_tokens = (int32_t)all_tokens.size();
    p.n_prompt = n_prompt;
    p.tokens   = std::move(all_tokens);
    p.logits   = std::move(all_logits);

    if (!trace_write(params.output_path, out)) {
        fprintf(stderr, "ref: failed to write '%s'\n", params.output_path.c_str());
        llama_batch_free(batch); llama_sampler_free(smpl); llama_free(ctx); llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "ref: wrote %s (%d tokens)\n", params.output_path.c_str(), p.n_tokens);

    // write generated text sidecar
    std::string txt_path = params.output_path;
    if (txt_path.size() > 4 && txt_path.substr(txt_path.size() - 4) == ".bin") {
        txt_path.replace(txt_path.size() - 4, 4, ".txt");
    } else {
        txt_path += ".txt";
    }
    FILE * txt_fp = fopen(txt_path.c_str(), "w");
    if (txt_fp) {
        fprintf(txt_fp, "%s\n", generated_text.c_str());
        fclose(txt_fp);
        fprintf(stderr, "ref: wrote %s\n", txt_path.c_str());
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
