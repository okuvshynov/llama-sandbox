#include "logits_file.h"
#include "topk_utils.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

struct collect_params {
    std::string model_path;
    std::string prompt;        // -p inline text
    std::string prompt_file;   // -f file
    std::string output_path = "logits.bin";
    int32_t     n_predict   = 256;
    int32_t     top_k       = 128;
    int32_t     n_ctx       = 4096;
    int32_t     n_batch     = 512;
    int32_t     n_threads   = (int32_t)std::thread::hardware_concurrency();
};

static bool parse_args(int argc, char ** argv, collect_params & params) {
    for (int32_t i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-p") == 0 && i + 1 < argc) {
            params.prompt = argv[++i];
        } else if (strcmp(arg, "-f") == 0 && i + 1 < argc) {
            params.prompt_file = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-n") == 0 && i + 1 < argc) {
            params.n_predict = atoi(argv[++i]);
        } else if (strcmp(arg, "-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "-b") == 0 && i + 1 < argc) {
            params.n_batch = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else {
            fprintf(stderr, "collect: unknown argument '%s'\n", arg);
            params.model_path.clear();
            break;
        }
    }
    if (params.model_path.empty() || (params.prompt.empty() == params.prompt_file.empty())) {
        fprintf(stderr, "Usage: collect -m <model.gguf> (-p <text> | -f <file>) [options]\n"
                        "  -o <path>   output file (default: logits.bin)\n"
                        "  -n <int>    tokens to generate, greedy (default: 256)\n"
                        "  -k <int>    top-K logits stored per position (default: 128)\n"
                        "  -c <int>    context size, auto-raised to fit (default: 4096)\n"
                        "  -b <int>    decode chunk size (default: 512)\n"
                        "  -t <int>    threads (default: all cores)\n"
                        "The prompt is tokenized raw — no chat template is ever applied.\n");
        return false;
    }
    return true;
}

static bool load_prompt_text(const collect_params & params, std::string & text, std::string & label) {
    if (!params.prompt.empty()) {
        text  = params.prompt;
        label = "inline";
        return true;
    }
    FILE * fp = fopen(params.prompt_file.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "collect: cannot open prompt file '%s'\n", params.prompt_file.c_str());
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    text.resize(sz);
    size_t n_read = fread(&text[0], 1, sz, fp);
    fclose(fp);
    text.resize(n_read);
    label = params.prompt_file;
    return true;
}

int main(int argc, char ** argv) {
    collect_params params;
    if (!parse_args(argc, argv, params)) return 1;

    std::string prompt_text, prompt_label;
    if (!load_prompt_text(params, prompt_text, prompt_label)) return 1;

    ggml_backend_load_all();

    // CPU-only build (GGML_METAL forced OFF in CMakeLists) — no offload params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "collect: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));
    std::string model_desc = params.model_path + " | " + desc;

    // raw tokenization: BOS per model metadata, literal special tokens honored,
    // no chat template — the resulting ids are the interface to downstream rescoring
    std::vector<llama_token> prompt_tokens = common_tokenize(vocab, prompt_text, true, true);
    const int32_t n_prompt = (int32_t)prompt_tokens.size();
    if (n_prompt == 0) {
        fprintf(stderr, "collect: prompt tokenized to 0 tokens\n");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = std::max(params.n_ctx, n_prompt + params.n_predict);
    ctx_params.n_batch         = params.n_batch;
    ctx_params.n_ubatch        = std::min(params.n_batch, 512);
    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "collect: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "collect: n_vocab=%d n_prompt=%d n_predict=%d top_k=%d n_ctx=%u threads=%d\n",
            n_vocab, n_prompt, params.n_predict, params.top_k, ctx_params.n_ctx, params.n_threads);

    std::vector<int32_t>       tokens(prompt_tokens.begin(), prompt_tokens.end());
    std::vector<lkld_position> positions;
    positions.reserve(n_prompt + params.n_predict);
    std::vector<int32_t> idx_buf;

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // prompt eval, chunked; ALL positions flagged so llama_get_logits_ith's
    // flagged-output index equals the batch-local index, and extraction must
    // happen before the next llama_decode invalidates the logit pointers
    const auto t_prompt_start = std::chrono::steady_clock::now();
    for (int32_t start = 0; start < n_prompt; ) {
        const int32_t end = std::min(n_prompt, start + params.n_batch);
        common_batch_clear(batch);
        for (int32_t i = start; i < end; i++) {
            common_batch_add(batch, prompt_tokens[i], i, {0}, true);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "collect: decode failed at prompt position %d\n", start);
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        for (int32_t i = start; i < end; i++) {
            const float * row = llama_get_logits_ith(ctx, i - start);
            positions.push_back(extract_topk_lse(row, n_vocab, params.top_k, idx_buf));
        }
        start = end;
    }
    const auto t_prompt_end = std::chrono::steady_clock::now();

    // greedy generation: next token is the stored top-1 of the previous position
    const char * stop_reason = params.n_predict > 0 ? "length" : "none";
    llama_token next = positions.back().ids[0];
    for (int32_t step = 0; step < params.n_predict; step++) {
        tokens.push_back(next);
        std::string piece = common_token_to_piece(vocab, next);
        fprintf(stdout, "%s", piece.c_str());
        fflush(stdout);

        common_batch_clear(batch);
        common_batch_add(batch, next, n_prompt + step, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "\ncollect: decode failed at generation step %d\n", step);
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        const float * row = llama_get_logits_ith(ctx, 0);
        positions.push_back(extract_topk_lse(row, n_vocab, params.top_k, idx_buf));

        if (llama_vocab_is_eog(vocab, next)) {
            stop_reason = "eog";
            break;
        }
        next = positions.back().ids[0];
    }
    const auto t_gen_end = std::chrono::steady_clock::now();
    fprintf(stdout, "\n");

    const int32_t n_total = (int32_t)tokens.size();
    const int32_t n_gen   = n_total - n_prompt;

    lkld_file out;
    out.n_vocab    = n_vocab;
    out.top_k      = std::min(params.top_k, n_vocab);
    out.model_desc = model_desc;
    out.seqs.push_back({prompt_label, n_prompt, n_total, std::move(tokens), std::move(positions)});

    if (!lkld_write(params.output_path, out)) {
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        return 1;
    }

    double tail_mean = 0.0, tail_max = 0.0;
    for (const lkld_position & p : out.seqs[0].positions) {
        double t = tail_mass(p);
        tail_mean += t;
        tail_max = std::max(tail_max, t);
    }
    tail_mean /= out.seqs[0].positions.size();

    const double prompt_s = std::chrono::duration<double>(t_prompt_end - t_prompt_start).count();
    const double gen_s    = std::chrono::duration<double>(t_gen_end - t_prompt_end).count();

    fprintf(stderr, "collect: model: %s\n", model_desc.c_str());
    fprintf(stderr, "collect: n_prompt=%d (%.1f tok/s), n_gen=%d (%.2f tok/s), stop=%s\n",
            n_prompt, n_prompt / prompt_s, n_gen, n_gen > 0 ? n_gen / gen_s : 0.0, stop_reason);
    fprintf(stderr, "collect: top-%d tail mass: mean=%.3e max=%.3e\n",
            out.top_k, tail_mean, tail_max);
    fprintf(stderr, "collect: wrote %s (%.2f MB, %d positions)\n",
            params.output_path.c_str(),
            std::filesystem::file_size(params.output_path) / 1e6,
            (int32_t)out.seqs[0].positions.size());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
