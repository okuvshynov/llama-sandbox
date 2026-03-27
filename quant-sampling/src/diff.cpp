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
    int         top_n        = 10;
    int         n_gpu_layers = 0; // CPU-only by default, we only need vocab
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
        fprintf(stderr, "Usage: quant-sampling diff -a <ref.qmlog> -b <quant.qmlog> [-m <model>] [--top-n 10]\n"
                        "  -m <model>    model file for token decoding (optional)\n"
                        "  --top-n <int> show top N tokens per side (default: 10)\n"
                        "  -ngl <int>    GPU layers for model loading (default: 0)\n");
        return false;
    }
    return true;
}

// Escape non-printable characters for display
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

int cmd_diff(int argc, char ** argv) {
    diff_params params;
    if (!parse_args(argc, argv, params)) return 1;

    // load files
    fprintf(stderr, "diff: loading '%s'...\n", params.path_a.c_str());
    qmlog_file fa;
    if (!qmlog_read(params.path_a, fa)) return 1;

    fprintf(stderr, "diff: loading '%s'...\n", params.path_b.c_str());
    qmlog_file fb;
    if (!qmlog_read(params.path_b, fb)) return 1;

    if (fa.header.n_vocab != fb.header.n_vocab) {
        fprintf(stderr, "diff: vocab size mismatch (%d vs %d)\n",
                fa.header.n_vocab, fb.header.n_vocab);
        return 1;
    }
    if (fa.header.n_tokens != fb.header.n_tokens) {
        fprintf(stderr, "diff: token count mismatch (%d vs %d)\n",
                fa.header.n_tokens, fb.header.n_tokens);
        return 1;
    }

    const int n_vocab  = fa.header.n_vocab;
    const int n_tokens = fa.header.n_tokens;
    const int n_logits = n_tokens - 1;
    const int top_n    = std::min(params.top_n, n_vocab);

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

    auto tok_str = [&](int32_t tok) -> std::string {
        if (vocab) {
            return escape(common_token_to_piece(vocab, tok));
        }
        return std::to_string(tok);
    };

    // find disagreements
    int n_disagree = 0;
    std::vector<double> log_p_ref, log_p_quant;
    std::vector<int> idx_ref(n_vocab), idx_quant(n_vocab);

    for (int i = 0; i < n_logits; i++) {
        const float * logits_a = fa.logits.data() + (size_t)i * n_vocab;
        const float * logits_b = fb.logits.data() + (size_t)i * n_vocab;

        int top1_a = argmax(logits_a, n_vocab);
        int top1_b = argmax(logits_b, n_vocab);

        if (top1_a == top1_b) continue;
        n_disagree++;

        // compute log-probs
        log_softmax(logits_a, n_vocab, log_p_ref);
        log_softmax(logits_b, n_vocab, log_p_quant);

        double kl = kl_divergence(log_p_ref, log_p_quant, n_vocab);

        // sort by probability (descending)
        std::iota(idx_ref.begin(), idx_ref.end(), 0);
        std::iota(idx_quant.begin(), idx_quant.end(), 0);
        std::partial_sort(idx_ref.begin(), idx_ref.begin() + top_n, idx_ref.end(),
                          [&](int a, int b) { return log_p_ref[a] > log_p_ref[b]; });
        std::partial_sort(idx_quant.begin(), idx_quant.begin() + top_n, idx_quant.end(),
                          [&](int a, int b) { return log_p_quant[a] > log_p_quant[b]; });

        // context: what token was at this position and what came next
        printf("--- pos %d  |  context: '%s' -> actual next: '%s'  |  KL=%.4f ---\n",
               i, tok_str(fa.tokens[i]).c_str(), tok_str(fa.tokens[i + 1]).c_str(), kl);

        printf("  ref top-1: '%s' (%.2f%%)    quant top-1: '%s' (%.2f%%)\n",
               tok_str(top1_a).c_str(), 100.0 * exp(log_p_ref[top1_a]),
               tok_str(top1_b).c_str(), 100.0 * exp(log_p_quant[top1_b]));

        printf("  %-4s  %-20s  %8s  %8s   |   %-20s  %8s  %8s\n",
               "rank", "ref token", "prob", "logit",
               "quant token", "prob", "logit");
        printf("  %-4s  %-20s  %8s  %8s   |   %-20s  %8s  %8s\n",
               "----", "--------------------", "--------", "--------",
               "--------------------", "--------", "--------");

        for (int k = 0; k < top_n; k++) {
            int ti_a = idx_ref[k];
            int ti_b = idx_quant[k];

            std::string tok_a = tok_str(ti_a);
            std::string tok_b = tok_str(ti_b);
            if (tok_a.size() > 20) tok_a = tok_a.substr(0, 17) + "...";
            if (tok_b.size() > 20) tok_b = tok_b.substr(0, 17) + "...";

            printf("  %-4d  %-20s  %7.2f%%  %8.2f   |   %-20s  %7.2f%%  %8.2f\n",
                   k + 1,
                   tok_a.c_str(), 100.0 * exp(log_p_ref[ti_a]), logits_a[ti_a],
                   tok_b.c_str(), 100.0 * exp(log_p_quant[ti_b]), logits_b[ti_b]);
        }
        printf("\n");
    }

    printf("=== Summary: %d / %d positions with top-1 disagreement (%.1f%%) ===\n",
           n_disagree, n_logits, 100.0 * n_disagree / n_logits);

    if (model) llama_model_free(model);
    return 0;
}
