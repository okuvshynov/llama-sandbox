#include "compare.h"
#include "kl_utils.h"
#include "logits_io.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

struct compare_params {
    std::string path_a;
    std::string path_b;
    bool        optimize  = false;
    double      temp_min  = 0.0;
    double      temp_max  = 2.0;
    double      temp_step = 0.05;
};

static bool parse_args(int argc, char ** argv, compare_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-a") == 0 && i + 1 < argc) {
            params.path_a = argv[++i];
        } else if (strcmp(arg, "-b") == 0 && i + 1 < argc) {
            params.path_b = argv[++i];
        } else if (strcmp(arg, "--optimize") == 0) {
            params.optimize = true;
        } else if (strcmp(arg, "--temp-min") == 0 && i + 1 < argc) {
            params.temp_min = atof(argv[++i]);
        } else if (strcmp(arg, "--temp-max") == 0 && i + 1 < argc) {
            params.temp_max = atof(argv[++i]);
        } else if (strcmp(arg, "--temp-step") == 0 && i + 1 < argc) {
            params.temp_step = atof(argv[++i]);
        } else {
            fprintf(stderr, "compare: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_a.empty() || params.path_b.empty()) {
        fprintf(stderr, "Usage: quant-sampling compare -a <ref.bin> -b <target.bin> [--optimize]\n"
                        "  --temp-min <val>   temperature scan start (default 0.0)\n"
                        "  --temp-max <val>   temperature scan end   (default 2.0)\n"
                        "  --temp-step <val>  temperature scan step  (default 0.05)\n");
        return false;
    }
    return true;
}

struct prompt_stats {
    int    n_logits;
    double kl_mean;
    double top1_pct;
};

// Compute KL stats for a single prompt pair.
static prompt_stats compute_prompt_stats(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int n_vocab, double ref_temp, int n_threads
) {
    const int n_logits = pa.n_tokens - 1;

    std::vector<double> kl_per_pos(n_logits);
    std::atomic<int> top1_agree{0};
    std::atomic<int> counter{0};

    auto worker = [&]() {
        std::vector<double> log_p_ref, log_p_tgt;
        int local_agree = 0;
        while (true) {
            int i = counter.fetch_add(1);
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
    for (int t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for (auto & t : threads) t.join();

    double kl_sum = 0.0;
    for (int i = 0; i < n_logits; i++) kl_sum += kl_per_pos[i];

    return { n_logits, kl_sum / n_logits, 100.0 * top1_agree.load() / n_logits };
}

// Compute mean KL for a prompt pair at a given target temperature.
static double compute_kl_at_temp(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int n_vocab, double target_temp, double ref_temp
) {
    const int n_logits = pa.n_tokens - 1;
    std::vector<double> log_p_ref, log_p_tgt;
    double total_kl = 0.0;
    for (int i = 0; i < n_logits; i++) {
        const float * la = pa.logits.data() + (size_t)i * n_vocab;
        const float * lb = pb.logits.data() + (size_t)i * n_vocab;
        log_softmax_temp(la, n_vocab, ref_temp, log_p_ref);
        log_softmax_temp(lb, n_vocab, target_temp, log_p_tgt);
        total_kl += kl_divergence(log_p_ref, log_p_tgt, n_vocab);
    }
    return total_kl / n_logits;
}

// Detailed single-prompt output (top-5 KL positions, prob diff).
static void print_single_detail(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int n_vocab, double ref_temp, int n_threads
) {
    const int n_logits = pa.n_tokens - 1;

    std::vector<double> kl_per_pos(n_logits);
    std::atomic<int>    top1_agree{0};
    std::vector<double> prob_diff(n_logits);

    std::atomic<int> counter{0};
    auto worker_fn = [&]() {
        std::vector<double> log_p_ref, log_p_tgt;
        int local_agree = 0;
        while (true) {
            int i = counter.fetch_add(1);
            if (i >= n_logits) break;

            const float * la = pa.logits.data() + (size_t)i * n_vocab;
            const float * lb = pb.logits.data() + (size_t)i * n_vocab;

            log_softmax_temp(la, n_vocab, ref_temp, log_p_ref);
            log_softmax_temp(lb, n_vocab, ref_temp, log_p_tgt);

            kl_per_pos[i] = kl_divergence(log_p_ref, log_p_tgt, n_vocab);

            if (argmax(la, n_vocab) == argmax(lb, n_vocab)) local_agree++;

            int next_tok = pa.tokens[i + 1];
            prob_diff[i] = exp(log_p_ref[next_tok]) - exp(log_p_tgt[next_tok]);
        }
        top1_agree.fetch_add(local_agree);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) threads.emplace_back(worker_fn);
    for (auto & t : threads) t.join();

    double kl_sum = 0.0, kl_sq_sum = 0.0;
    for (int i = 0; i < n_logits; i++) {
        kl_sum    += kl_per_pos[i];
        kl_sq_sum += kl_per_pos[i] * kl_per_pos[i];
    }
    double kl_mean = kl_sum / n_logits;
    double kl_std  = sqrt(kl_sq_sum / n_logits - kl_mean * kl_mean);

    double pd_sum = 0.0, pd_sq_sum = 0.0;
    for (int i = 0; i < n_logits; i++) {
        pd_sum    += prob_diff[i];
        pd_sq_sum += prob_diff[i] * prob_diff[i];
    }

    printf("\n=== Basic Statistics ===\n");
    printf("KL divergence (mean):     %.6f nats\n", kl_mean);
    printf("KL divergence (std):      %.6f nats\n", kl_std);
    printf("Top-1 agreement:          %.1f%% (%d / %d)\n",
           100.0 * top1_agree.load() / n_logits, top1_agree.load(), n_logits);
    printf("Prob diff for next token:\n");
    printf("  Mean:                   %+.6f\n", pd_sum / n_logits);
    printf("  RMS:                    %.6f\n", sqrt(pd_sq_sum / n_logits));

    std::vector<int> pos_idx(n_logits);
    std::iota(pos_idx.begin(), pos_idx.end(), 0);
    std::partial_sort(pos_idx.begin(), pos_idx.begin() + std::min(5, n_logits),
                      pos_idx.end(), [&](int a, int b) {
                          return kl_per_pos[a] > kl_per_pos[b];
                      });

    printf("\nTop-5 highest KL positions:\n");
    for (int k = 0; k < std::min(5, n_logits); k++) {
        int idx = pos_idx[k];
        printf("  pos %4d: KL = %.4f  (token %d -> %d)\n",
               idx, kl_per_pos[idx], pa.tokens[idx], pa.tokens[idx + 1]);
    }
}

int cmd_compare(int argc, char ** argv) {
    compare_params params;
    if (!parse_args(argc, argv, params)) return 1;

    fprintf(stderr, "compare: loading '%s'...\n", params.path_a.c_str());
    qmlog_file fa;
    if (!qmlog_read(params.path_a, fa)) return 1;

    fprintf(stderr, "compare: loading '%s'...\n", params.path_b.c_str());
    qmlog_file fb;
    if (!qmlog_read(params.path_b, fb)) return 1;

    // validate
    if (fa.n_vocab != fb.n_vocab) {
        fprintf(stderr, "compare: vocab size mismatch (%d vs %d)\n", fa.n_vocab, fb.n_vocab);
        return 1;
    }
    if (fa.n_prompts != fb.n_prompts) {
        fprintf(stderr, "compare: prompt count mismatch (%d vs %d)\n", fa.n_prompts, fb.n_prompts);
        return 1;
    }
    for (int p = 0; p < fa.n_prompts; p++) {
        if (fa.prompts[p].n_tokens != fb.prompts[p].n_tokens) {
            fprintf(stderr, "compare: token count mismatch in prompt %d (%d vs %d)\n",
                    p + 1, fa.prompts[p].n_tokens, fb.prompts[p].n_tokens);
            return 1;
        }
    }

    const int n_vocab   = fa.n_vocab;
    const int n_prompts = fa.n_prompts;
    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());

    const double ref_temp = (fa.temp > 0.0f) ? (double)fa.temp : 1.0;
    fprintf(stderr, "compare: n_vocab=%d, n_prompts=%d, ref_temp=%.2f\n",
            n_vocab, n_prompts, ref_temp);

    // === Single-prompt: detailed output ===
    if (n_prompts == 1) {
        print_single_detail(fa.prompts[0], fb.prompts[0], n_vocab, ref_temp, n_threads);

        if (!params.optimize) return 0;

        // temperature scan for single prompt
        printf("\n=== Temperature Optimization ===\n");

        struct temp_result { double temp; double kl_mean; };
        std::vector<temp_result> temp_results;
        for (double T = params.temp_min + params.temp_step; T <= params.temp_max + 1e-9; T += params.temp_step) {
            temp_results.push_back({T, compute_kl_at_temp(fa.prompts[0], fb.prompts[0], n_vocab, T, ref_temp)});
        }

        auto best = std::min_element(temp_results.begin(), temp_results.end(),
            [](const temp_result & a, const temp_result & b) { return a.kl_mean < b.kl_mean; });

        // baseline KL at ref_temp
        double baseline_kl = compute_kl_at_temp(fa.prompts[0], fb.prompts[0], n_vocab, ref_temp, ref_temp);

        printf("Scanned %zu temperatures:\n", temp_results.size());
        printf("  Baseline (T=%.2f):  KL = %.6f\n", ref_temp, baseline_kl);
        printf("  Best temperature:   T = %.2f, KL = %.6f\n", best->temp, best->kl_mean);
        printf("  KL reduction:       %.1f%%\n", 100.0 * (1.0 - best->kl_mean / baseline_kl));

        printf("\nTemperature scan (selected):\n");
        for (const auto & tr : temp_results) {
            int idx = (int)(&tr - &temp_results[0]);
            bool near_opt = fabs(tr.temp - best->temp) < 0.16;
            if (idx % 5 == 0 || near_opt) {
                printf("  T=%.2f  KL=%.6f%s\n", tr.temp, tr.kl_mean,
                       (fabs(tr.temp - best->temp) < 0.001) ? "  <-- best" : "");
            }
        }

        printf("\nRecommended temperature for target model:  --temp %.2f\n", best->temp);
        return 0;
    }

    // === Multi-prompt: per-prompt summary + aggregate ===
    std::vector<prompt_stats> stats(n_prompts);
    for (int p = 0; p < n_prompts; p++) {
        stats[p] = compute_prompt_stats(fa.prompts[p], fb.prompts[p], n_vocab, ref_temp, n_threads);
    }

    printf("\n=== Per-Prompt Statistics ===\n");
    printf("  %3s  %6s  %9s  %7s\n", "#", "Tokens", "KL mean", "Top-1%");
    for (int p = 0; p < n_prompts; p++) {
        printf("  %3d  %6d  %9.6f  %6.1f%%\n",
               p + 1, stats[p].n_logits, stats[p].kl_mean, stats[p].top1_pct);
    }

    // aggregate
    double kl_sum = 0.0, kl_sq = 0.0;
    double t1_sum = 0.0, t1_sq = 0.0;
    for (int p = 0; p < n_prompts; p++) {
        kl_sum += stats[p].kl_mean;
        kl_sq  += stats[p].kl_mean * stats[p].kl_mean;
        t1_sum += stats[p].top1_pct;
        t1_sq  += stats[p].top1_pct * stats[p].top1_pct;
    }
    double kl_agg_mean = kl_sum / n_prompts;
    double kl_agg_std  = (n_prompts > 1) ? sqrt(kl_sq / n_prompts - kl_agg_mean * kl_agg_mean) : 0.0;
    double t1_agg_mean = t1_sum / n_prompts;
    double t1_agg_std  = (n_prompts > 1) ? sqrt(t1_sq / n_prompts - t1_agg_mean * t1_agg_mean) : 0.0;

    printf("\n=== Aggregate ===\n");
    printf("  KL divergence: %.6f +/- %.6f\n", kl_agg_mean, kl_agg_std);
    printf("  Top-1 agree:   %.1f%% +/- %.1f%%\n", t1_agg_mean, t1_agg_std);

    if (!params.optimize) return 0;

    // === Temperature optimization across all prompts ===
    printf("\n=== Temperature Optimization ===\n");

    std::vector<double> temp_vals;
    for (double T = params.temp_min + params.temp_step; T <= params.temp_max + 1e-9; T += params.temp_step) {
        temp_vals.push_back(T);
    }
    const int n_temps = (int)temp_vals.size();

    // per_prompt_kl[p][ti]
    std::vector<std::vector<double>> per_prompt_kl(n_prompts, std::vector<double>(n_temps, 0.0));

    // parallelize across temperature indices
    std::atomic<int> temp_ctr{0};
    auto temp_worker = [&]() {
        while (true) {
            int ti = temp_ctr.fetch_add(1);
            if (ti >= n_temps) break;
            for (int p = 0; p < n_prompts; p++) {
                per_prompt_kl[p][ti] = compute_kl_at_temp(
                    fa.prompts[p], fb.prompts[p], n_vocab, temp_vals[ti], ref_temp);
            }
            if (ti % 10 == 0) fprintf(stderr, "  temp scan: %d / %d\r", ti, n_temps);
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) threads.emplace_back(temp_worker);
    for (auto & t : threads) t.join();
    fprintf(stderr, "\n");

    // per-prompt best temperature
    std::vector<double> best_T(n_prompts);
    printf("  %6s  %6s  %10s\n", "Prompt", "Best T", "KL at best");
    for (int p = 0; p < n_prompts; p++) {
        int best_ti = 0;
        for (int ti = 1; ti < n_temps; ti++) {
            if (per_prompt_kl[p][ti] < per_prompt_kl[p][best_ti]) best_ti = ti;
        }
        best_T[p] = temp_vals[best_ti];
        printf("  %6d  %6.2f  %10.6f\n", p + 1, best_T[p], per_prompt_kl[p][best_ti]);
    }

    double T_sum = 0.0, T_sq = 0.0;
    for (int p = 0; p < n_prompts; p++) {
        T_sum += best_T[p];
        T_sq  += best_T[p] * best_T[p];
    }
    double T_mean = T_sum / n_prompts;
    double T_std  = (n_prompts > 1) ? sqrt(T_sq / n_prompts - T_mean * T_mean) : 0.0;

    printf("\n  Mean optimal T: %.2f +/- %.2f\n", T_mean, T_std);
    printf("\nRecommended temperature for target model:  --temp %.2f\n", T_mean);

    return 0;
}
