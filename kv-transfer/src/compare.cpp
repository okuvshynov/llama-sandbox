#include "compare.h"
#include "trace_io.h"
#include "kl_utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct compare_params {
    std::string path_a;
    std::string path_b;
};

static bool parse_args(int argc, char ** argv, compare_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-a") == 0 && i + 1 < argc) {
            params.path_a = argv[++i];
        } else if (strcmp(arg, "-b") == 0 && i + 1 < argc) {
            params.path_b = argv[++i];
        } else {
            fprintf(stderr, "compare: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_a.empty() || params.path_b.empty()) {
        fprintf(stderr, "Usage: kv-transfer compare -a <ref.bin> -b <other.bin>\n");
        return false;
    }
    return true;
}

int cmd_compare(int argc, char ** argv) {
    compare_params params;
    if (!parse_args(argc, argv, params)) return 1;

    fprintf(stderr, "compare: loading '%s'...\n", params.path_a.c_str());
    trace_file fa;
    if (!trace_read(params.path_a, fa)) return 1;

    fprintf(stderr, "compare: loading '%s'...\n", params.path_b.c_str());
    trace_file fb;
    if (!trace_read(params.path_b, fb)) return 1;

    if (fa.n_vocab != fb.n_vocab) {
        fprintf(stderr, "compare: vocab mismatch (%d vs %d)\n", fa.n_vocab, fb.n_vocab);
        return 1;
    }
    if (fa.n_prompts != 1 || fb.n_prompts != 1) {
        fprintf(stderr, "compare: expected single-prompt files\n");
        return 1;
    }

    const auto & pa = fa.prompts[0];
    const auto & pb = fb.prompts[0];

    if (pa.n_tokens != pb.n_tokens) {
        fprintf(stderr, "compare: token count mismatch (%d vs %d)\n", pa.n_tokens, pb.n_tokens);
        return 1;
    }

    const int32_t n_vocab  = fa.n_vocab;
    const int32_t n_gen    = pa.n_tokens - pa.n_prompt;
    const int32_t n_logits = n_gen;  // generation logits only

    if (n_logits <= 0) {
        fprintf(stderr, "compare: no generation logits (n_tokens=%d, n_prompt=%d)\n",
                pa.n_tokens, pa.n_prompt);
        return 1;
    }

    const int32_t n_threads = std::max(1, (int32_t)std::thread::hardware_concurrency());
    const double ref_temp = (fa.temp > 0.0f) ? (double)fa.temp : 1.0;

    std::vector<double> kl_per_pos(n_logits);
    std::atomic<int32_t> top1_agree{0};
    std::atomic<int32_t> counter{0};

    auto worker = [&]() {
        std::vector<double> log_p_ref, log_p_tgt;
        int32_t local_agree = 0;
        while (true) {
            int32_t i = counter.fetch_add(1);
            if (i >= n_logits) break;

            const float * la = pa.logits.data() + (size_t)i * n_vocab;
            const float * lb = pb.logits.data() + (size_t)i * n_vocab;

            log_softmax_temp(la, n_vocab, ref_temp, log_p_ref);
            log_softmax_temp(lb, n_vocab, ref_temp, log_p_tgt);

            kl_per_pos[i] = kl_divergence(log_p_ref, log_p_tgt, n_vocab);

            if (argmax(la, n_vocab) == argmax(lb, n_vocab)) local_agree++;
        }
        top1_agree.fetch_add(local_agree);
    };

    std::vector<std::thread> threads;
    for (int32_t t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for (auto & t : threads) t.join();

    double kl_sum = 0.0;
    for (int32_t i = 0; i < n_logits; i++) kl_sum += kl_per_pos[i];
    double kl_mean = kl_sum / n_logits;
    double top1_pct = 100.0 * top1_agree.load() / n_logits;

    // percentiles
    std::vector<double> sorted_kl(kl_per_pos.begin(), kl_per_pos.end());
    std::sort(sorted_kl.begin(), sorted_kl.end());
    auto percentile = [&](double p) -> double {
        double k = (sorted_kl.size() - 1) * p;
        int32_t lo = (int32_t)k;
        int32_t hi = lo + 1;
        if (hi >= (int32_t)sorted_kl.size()) return sorted_kl.back();
        double frac = k - lo;
        return sorted_kl[lo] * (1.0 - frac) + sorted_kl[hi] * frac;
    };

    printf("  KL divergence: %.6f\n", kl_mean);
    printf("  KL p95:        %.6f\n", percentile(0.95));
    printf("  KL p99:        %.6f\n", percentile(0.99));
    printf("  Top-1 agree:   %.1f%%\n", top1_pct);

    return 0;
}
