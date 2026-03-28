#include "diff.h"
#include "kl_utils.h"
#include "logits_io.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

struct diff_params {
    std::string path_a;
    std::string path_b;
    std::string model_path;
    int32_t     top_n        = 10;
    int32_t     n_gpu_layers = 0;
};

static bool parse_args(int argc, char ** argv, diff_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-a") == 0 && i + 1 < argc) {
            params.path_a = argv[++i];
        } else if (strcmp(arg, "-b") == 0 && i + 1 < argc) {
            params.path_b = argv[++i];
        } else if (strcmp(arg, "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(arg, "--top-n") == 0 && i + 1 < argc) {
            params.top_n = atoi(argv[++i]);
        } else if (strcmp(arg, "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else {
            fprintf(stderr, "diff: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_a.empty() || params.path_b.empty()) {
        fprintf(stderr, "Usage: quant-sampling diff -a <ref.bin> -b <target.bin> [-m <model>] [--top-n 10]\n"
                        "  -m <model>    model file for token decoding (optional)\n"
                        "  --top-n <int> show top N tokens per side (default: 10)\n"
                        "  -ngl <int>    GPU layers for model loading (default: 0)\n");
        return false;
    }
    return true;
}

static std::string escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '\n')      out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

// Process disagreements for a single prompt pair. Returns number of disagreements.
static int32_t diff_one_prompt(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int32_t n_vocab, int32_t top_n,
    const llama_vocab * vocab
) {
    const int32_t n_logits = pa.n_tokens - 1;
    int32_t n_disagree = 0;

    auto tok_str = [&](int32_t tok) -> std::string {
        if (vocab) return escape(common_token_to_piece(vocab, tok));
        return std::to_string(tok);
    };

    std::vector<double> log_p_ref, log_p_tgt;
    std::vector<int32_t> idx_ref(n_vocab), idx_tgt(n_vocab);

    for (int32_t i = 0; i < n_logits; i++) {
        const float * la = pa.logits.data() + (size_t)i * n_vocab;
        const float * lb = pb.logits.data() + (size_t)i * n_vocab;

        int32_t top1_a = argmax(la, n_vocab);
        int32_t top1_b = argmax(lb, n_vocab);

        if (top1_a == top1_b) continue;
        n_disagree++;

        log_softmax(la, n_vocab, log_p_ref);
        log_softmax(lb, n_vocab, log_p_tgt);

        double kl = kl_divergence(log_p_ref, log_p_tgt, n_vocab);

        std::iota(idx_ref.begin(), idx_ref.end(), 0);
        std::iota(idx_tgt.begin(), idx_tgt.end(), 0);
        std::partial_sort(idx_ref.begin(), idx_ref.begin() + top_n, idx_ref.end(),
                          [&](int32_t a, int32_t b) { return log_p_ref[a] > log_p_ref[b]; });
        std::partial_sort(idx_tgt.begin(), idx_tgt.begin() + top_n, idx_tgt.end(),
                          [&](int32_t a, int32_t b) { return log_p_tgt[a] > log_p_tgt[b]; });

        printf("--- pos %d  |  context: '%s' -> actual next: '%s'  |  KL=%.4f ---\n",
               i, tok_str(pa.tokens[i]).c_str(), tok_str(pa.tokens[i + 1]).c_str(), kl);

        printf("  ref top-1: '%s' (%.2f%%)    target top-1: '%s' (%.2f%%)\n",
               tok_str(top1_a).c_str(), 100.0 * exp(log_p_ref[top1_a]),
               tok_str(top1_b).c_str(), 100.0 * exp(log_p_tgt[top1_b]));

        printf("  %-4s  %-20s  %8s  %8s   |   %-20s  %8s  %8s\n",
               "rank", "ref token", "prob", "logit",
               "target token", "prob", "logit");
        printf("  %-4s  %-20s  %8s  %8s   |   %-20s  %8s  %8s\n",
               "----", "--------------------", "--------", "--------",
               "--------------------", "--------", "--------");

        for (int32_t k = 0; k < top_n; k++) {
            int32_t ti_a = idx_ref[k];
            int32_t ti_b = idx_tgt[k];

            std::string tok_a = tok_str(ti_a);
            std::string tok_b = tok_str(ti_b);
            if (tok_a.size() > 20) tok_a = tok_a.substr(0, 17) + "...";
            if (tok_b.size() > 20) tok_b = tok_b.substr(0, 17) + "...";

            printf("  %-4d  %-20s  %7.2f%%  %8.2f   |   %-20s  %7.2f%%  %8.2f\n",
                   k + 1,
                   tok_a.c_str(), 100.0 * exp(log_p_ref[ti_a]), la[ti_a],
                   tok_b.c_str(), 100.0 * exp(log_p_tgt[ti_b]), lb[ti_b]);
        }
        printf("\n");
    }

    return n_disagree;
}

int cmd_diff(int argc, char ** argv) {
    diff_params params;
    if (!parse_args(argc, argv, params)) return 1;

    fprintf(stderr, "diff: loading '%s'...\n", params.path_a.c_str());
    qmlog_file fa;
    if (!qmlog_read(params.path_a, fa)) return 1;

    fprintf(stderr, "diff: loading '%s'...\n", params.path_b.c_str());
    qmlog_file fb;
    if (!qmlog_read(params.path_b, fb)) return 1;

    if (fa.n_vocab != fb.n_vocab) {
        fprintf(stderr, "diff: vocab size mismatch (%d vs %d)\n", fa.n_vocab, fb.n_vocab);
        return 1;
    }
    if (fa.n_prompts != fb.n_prompts) {
        fprintf(stderr, "diff: prompt count mismatch (%d vs %d)\n", fa.n_prompts, fb.n_prompts);
        return 1;
    }

    const int32_t n_vocab  = fa.n_vocab;
    const int32_t top_n    = std::min(params.top_n, n_vocab);

    // optionally load model for token text
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;

    if (!params.model_path.empty()) {
        ggml_backend_load_all();
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = params.n_gpu_layers;
        model = llama_model_load_from_file(params.model_path.c_str(), mparams);
        if (model) {
            vocab = llama_model_get_vocab(model);
            fprintf(stderr, "diff: loaded model for token decoding\n");
        } else {
            fprintf(stderr, "diff: warning: failed to load model, showing token IDs only\n");
        }
    }

    int32_t total_disagree = 0;
    int32_t total_logits   = 0;

    for (int32_t p = 0; p < fa.n_prompts; p++) {
        if (fa.prompts[p].n_tokens != fb.prompts[p].n_tokens) {
            fprintf(stderr, "diff: token count mismatch in prompt %d (%d vs %d)\n",
                    p + 1, fa.prompts[p].n_tokens, fb.prompts[p].n_tokens);
            if (model) llama_model_free(model);
            return 1;
        }

        const int32_t n_logits = fa.prompts[p].n_tokens - 1;

        if (fa.n_prompts > 1) {
            printf("========== Prompt %d / %d (%d tokens) ==========\n\n",
                   p + 1, fa.n_prompts, fa.prompts[p].n_tokens);
        }

        int32_t nd = diff_one_prompt(fa.prompts[p], fb.prompts[p], n_vocab, top_n, vocab);
        total_disagree += nd;
        total_logits   += n_logits;

        if (fa.n_prompts > 1) {
            printf("  Prompt %d: %d / %d disagreements (%.1f%%)\n\n",
                   p + 1, nd, n_logits, 100.0 * nd / n_logits);
        }
    }

    printf("=== Summary: %d / %d positions with top-1 disagreement (%.1f%%) ===\n",
           total_disagree, total_logits, 100.0 * total_disagree / total_logits);

    if (model) llama_model_free(model);
    return 0;
}
