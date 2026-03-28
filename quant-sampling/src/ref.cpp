#include "ref.h"
#include "logits_io.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

struct ref_params {
    std::string model_path;
    std::string prompt;       // single prompt via -p
    std::string prompt_dir;   // directory of .txt files via -P
    std::string output_path = "ref.bin";
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
        } else if (strcmp(arg, "-P") == 0 && i + 1 < argc) {
            params.prompt_dir = argv[++i];
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
    if (params.model_path.empty() || (params.prompt.empty() && params.prompt_dir.empty())) {
        fprintf(stderr, "Usage: quant-sampling ref -m <model> (-p <prompt> | -P <dir>) [options]\n"
                        "  -o <path>     output file (default: ref.bin)\n"
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
    if (!params.prompt.empty() && !params.prompt_dir.empty()) {
        fprintf(stderr, "ref: specify either -p or -P, not both\n");
        return false;
    }
    return true;
}

static std::vector<std::string> load_prompts(const ref_params & params) {
    std::vector<std::string> prompts;

    if (!params.prompt.empty()) {
        prompts.push_back(params.prompt);
        return prompts;
    }

    namespace fs = std::filesystem;
    std::vector<fs::path> files;
    for (const auto & entry : fs::recursive_directory_iterator(params.prompt_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto & path : files) {
        FILE * fp = fopen(path.c_str(), "r");
        if (!fp) {
            fprintf(stderr, "ref: warning: cannot open '%s', skipping\n", path.c_str());
            continue;
        }
        fseek(fp, 0, SEEK_END);
        long sz = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        std::string content(sz, '\0');
        size_t n_read = fread(&content[0], 1, sz, fp);
        fclose(fp);
        content.resize(n_read);

        // trim trailing whitespace
        while (!content.empty() && (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
            content.pop_back();
        }
        if (!content.empty()) {
            prompts.push_back(std::move(content));
        }
    }

    return prompts;
}

static std::string apply_chat_template(llama_model * model, const std::string & prompt, bool no_chat) {
    if (no_chat) return prompt;

    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        fprintf(stderr, "ref: no chat template in model, using raw prompt\n");
        return prompt;
    }

    llama_chat_message msg = {"user", prompt.c_str()};
    int32_t len = llama_chat_apply_template(tmpl, &msg, 1, true, nullptr, 0);
    if (len <= 0) {
        fprintf(stderr, "ref: warning: chat template failed, using raw prompt\n");
        return prompt;
    }

    std::vector<char> buf(len + 1);
    llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
    return std::string(buf.data(), len);
}

static bool process_one_prompt(
    llama_model * model,
    const llama_vocab * vocab,
    const std::string & raw_prompt,
    const ref_params & params,
    int n_vocab,
    qmlog_prompt & out
) {
    std::string formatted = apply_chat_template(model, raw_prompt, params.no_chat);

    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, formatted, true, true);
    const int n_prompt = (int)prompt_tokens.size();
    const int total_tokens = n_prompt + params.n_predict;

    fprintf(stderr, "ref: n_prompt=%d, n_predict=%d\n", n_prompt, params.n_predict);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx   = std::max(params.n_ctx, total_tokens);
    ctx_params.n_batch = std::max(params.n_ctx, total_tokens);

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "ref: failed to create context\n");
        return false;
    }

    const int n_batch = llama_n_batch(ctx);

    std::vector<int32_t> all_tokens(prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.reserve(total_tokens);

    std::vector<float> all_logits;
    all_logits.reserve((size_t)(total_tokens - 1) * n_vocab);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(params.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    int n_decoded = 0;
    while (n_decoded < n_prompt) {
        common_batch_clear(batch);
        int batch_end = std::min(n_prompt, n_decoded + n_batch);
        for (int i = n_decoded; i < batch_end; i++) {
            common_batch_add(batch, prompt_tokens[i], i, {0}, i > 0);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "ref: decode failed at prompt position %d\n", n_decoded);
            llama_batch_free(batch); llama_sampler_free(smpl); llama_free(ctx);
            return false;
        }
        for (int i = n_decoded; i < batch_end; i++) {
            if (i == 0) continue;
            const float * logits = llama_get_logits_ith(ctx, i - n_decoded);
            all_logits.insert(all_logits.end(), logits, logits + n_vocab);
        }
        n_decoded = batch_end;
    }

    fprintf(stderr, "ref: prompt decoded, generating...\n");

    for (int i = 0; i < params.n_predict; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
            fprintf(stderr, "\nref: EOS at step %d\n", i);
            break;
        }
        all_tokens.push_back(new_token);

        std::string piece = common_token_to_piece(vocab, new_token);
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

    out.n_tokens = (int)all_tokens.size();
    out.n_prompt = n_prompt;
    out.tokens   = std::move(all_tokens);
    out.logits   = std::move(all_logits);

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    return true;
}

int cmd_ref(int argc, char ** argv) {
    ref_params params;
    if (!parse_args(argc, argv, params)) return 1;

    std::vector<std::string> prompts = load_prompts(params);
    if (prompts.empty()) {
        fprintf(stderr, "ref: no prompts found\n");
        return 1;
    }

    fprintf(stderr, "ref: %zu prompt(s)\n", prompts.size());

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "ref: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    qmlog_file out;
    out.n_vocab   = n_vocab;
    out.n_prompts = (int)prompts.size();
    out.temp      = params.temp;
    out.top_p     = params.top_p;
    out.top_k     = params.top_k;
    out.seed      = params.seed;
    out.prompts.resize(prompts.size());

    for (size_t pi = 0; pi < prompts.size(); pi++) {
        fprintf(stderr, "\n=== Prompt %zu / %zu ===\n", pi + 1, prompts.size());
        if (!process_one_prompt(model, vocab, prompts[pi], params, n_vocab, out.prompts[pi])) {
            llama_model_free(model);
            return 1;
        }
    }

    if (!qmlog_write(params.output_path, out)) {
        fprintf(stderr, "ref: failed to write '%s'\n", params.output_path.c_str());
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "\nref: wrote %s (%d prompt(s))\n", params.output_path.c_str(), out.n_prompts);

    llama_model_free(model);
    return 0;
}
