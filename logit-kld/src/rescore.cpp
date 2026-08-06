#include "cpu_topology.h"
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

struct rescore_params {
    std::string model_path;
    std::string input_path;
    std::string output_path = "rescore.bin";
    int32_t     top_k       = 0;    // 0 = same as input file
    int32_t     n_ctx       = 4096;
    int32_t     n_batch     = 512;
    int32_t     n_threads   = (int32_t)physical_core_count();
    bool        sim_gen     = false;
};

static bool parse_args(int argc, char ** argv, rescore_params & params) {
    for (int32_t i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "-i") == 0 && i + 1 < argc) {
            params.input_path = argv[++i];
        } else if (strcmp(arg, "-o") == 0 && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (strcmp(arg, "-k") == 0 && i + 1 < argc) {
            params.top_k = atoi(argv[++i]);
        } else if (strcmp(arg, "-c") == 0 && i + 1 < argc) {
            params.n_ctx = atoi(argv[++i]);
        } else if (strcmp(arg, "-b") == 0 && i + 1 < argc) {
            params.n_batch = atoi(argv[++i]);
        } else if (strcmp(arg, "-t") == 0 && i + 1 < argc) {
            params.n_threads = atoi(argv[++i]);
        } else if (strcmp(arg, "--sim-gen") == 0) {
            params.sim_gen = true;
        } else {
            fprintf(stderr, "rescore: unknown argument '%s'\n", arg);
            params.model_path.clear();
            break;
        }
    }
    if (params.model_path.empty() || params.input_path.empty()) {
        fprintf(stderr, "Usage: rescore -m <model.gguf> -i <collect.bin> [options]\n"
                        "  -o <path>   output file (default: rescore.bin)\n"
                        "  -k <int>    top-K logits stored (default: same as input)\n"
                        "  -c <int>    context size, auto-raised to fit (default: 4096)\n"
                        "  -b <int>    decode chunk size (default: 512)\n"
                        "  -t <int>    threads (default: physical cores, ignoring SMT siblings)\n"
                        "  --sim-gen   simulate generation batching: prefill prompt\n"
                        "              positions in -b chunks, then decode completion\n"
                        "              positions one token at a time (tests the decode\n"
                        "              path; mirrors collect's execution shape exactly)\n"
                        "Scores the input file's raw token ids under the given model —\n"
                        "no tokenization, no chat template, no generation.\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    rescore_params params;
    if (!parse_args(argc, argv, params)) return 1;

    lkld_file in;
    if (!lkld_read(params.input_path, in)) return 1;
    if (params.top_k == 0) params.top_k = in.top_k;

    int32_t n_total_max = 0;
    for (const lkld_seq & s : in.seqs) {
        n_total_max = std::max(n_total_max, s.n_total);
    }

    ggml_backend_load_all();

    // CPU-only build (GGML_METAL forced OFF in CMakeLists) — no offload params
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "rescore: failed to load model '%s'\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    if (n_vocab != in.n_vocab) {
        // token ids are the interface — a different vocab size means the files
        // are not directly comparable position-by-position; warn, don't refuse
        fprintf(stderr, "rescore: WARNING: model n_vocab=%d != input n_vocab=%d\n",
                n_vocab, in.n_vocab);
    }
    for (const lkld_seq & s : in.seqs) {
        for (int32_t tok : s.tokens) {
            if (tok < 0 || tok >= n_vocab) {
                fprintf(stderr, "rescore: token id %d out of range for this model (n_vocab=%d)\n",
                        tok, n_vocab);
                llama_model_free(model);
                return 1;
            }
        }
    }

    char desc[256];
    llama_model_desc(model, desc, sizeof(desc));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = std::max(params.n_ctx, n_total_max);
    ctx_params.n_batch         = params.n_batch;
    ctx_params.n_ubatch        = std::min(params.n_batch, 512);
    ctx_params.n_threads       = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "rescore: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    fprintf(stderr, "rescore: n_vocab=%d top_k=%d n_seq=%zu n_ctx=%u threads=%d\n",
            n_vocab, params.top_k, in.seqs.size(), ctx_params.n_ctx, params.n_threads);

    lkld_file out;
    out.n_vocab    = n_vocab;
    out.top_k      = std::min(params.top_k, n_vocab);
    out.model_desc = params.model_path + " | " + desc;

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    std::vector<int32_t> idx_buf;

    const auto t_start = std::chrono::steady_clock::now();
    int32_t n_scored_all = 0;
    for (const lkld_seq & s : in.seqs) {
        llama_memory_clear(llama_get_memory(ctx), true);

        lkld_seq r;
        r.label    = s.label;
        r.n_prompt = s.n_prompt;
        r.n_total  = s.n_total;
        r.tokens   = s.tokens;
        r.positions.reserve(s.n_total);

        // chunked all-positions-flagged decode of [from, to), chunk_size per batch
        auto decode_range = [&](int32_t from, int32_t to, int32_t chunk_size) -> bool {
            for (int32_t start = from; start < to; ) {
                const int32_t end = std::min(to, start + chunk_size);
                common_batch_clear(batch);
                for (int32_t i = start; i < end; i++) {
                    common_batch_add(batch, s.tokens[i], i, {0}, true);
                }
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "rescore: decode failed at position %d of [%s]\n",
                            start, s.label.c_str());
                    return false;
                }
                for (int32_t i = start; i < end; i++) {
                    const float * row = llama_get_logits_ith(ctx, i - start);
                    r.positions.push_back(extract_topk_lse(row, n_vocab, params.top_k, idx_buf));
                }
                start = end;
            }
            return true;
        };

        // default: whole sequence in -b chunks (prefill path only).
        // --sim-gen: prompt prefilled in -b chunks, completion decoded one token
        // per batch — the same execution shape a real generation (and collect) has
        const bool ok = params.sim_gen
            ? decode_range(0, s.n_prompt, params.n_batch) && decode_range(s.n_prompt, s.n_total, 1)
            : decode_range(0, s.n_total, params.n_batch);
        if (!ok) {
            llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
            return 1;
        }
        n_scored_all += (int32_t)r.positions.size();
        out.seqs.push_back(std::move(r));
    }
    const auto t_end = std::chrono::steady_clock::now();

    if (!lkld_write(params.output_path, out)) {
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
        return 1;
    }

    double tail_mean = 0.0, tail_max = 0.0;
    for (const lkld_seq & s : out.seqs) {
        for (const lkld_position & p : s.positions) {
            double t = tail_mass(p);
            tail_mean += t;
            tail_max = std::max(tail_max, t);
        }
    }
    tail_mean /= n_scored_all;

    const double secs = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "rescore: model: %s\n", out.model_desc.c_str());
    fprintf(stderr, "rescore: scored %d positions in %.1fs (%.1f tok/s)\n",
            n_scored_all, secs, n_scored_all / secs);
    fprintf(stderr, "rescore: top-%d tail mass: mean=%.3e max=%.3e\n",
            out.top_k, tail_mean, tail_max);
    fprintf(stderr, "rescore: wrote %s (%.2f MB)\n",
            params.output_path.c_str(),
            std::filesystem::file_size(params.output_path) / 1e6);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
