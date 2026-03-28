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

static constexpr double UNSET = -1.0;

struct compare_params {
    std::string path_a;
    std::string path_b;
    std::string csv_path;
    std::string rank_csv_path;
    bool        optimize  = false;
    double      temp_min  = UNSET;
    double      temp_max  = UNSET;
    double      temp_step = UNSET;
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
        } else if (strcmp(arg, "--csv") == 0 && i + 1 < argc) {
            params.csv_path = argv[++i];
        } else if (strcmp(arg, "--rank-csv") == 0 && i + 1 < argc) {
            params.rank_csv_path = argv[++i];
        } else {
            fprintf(stderr, "compare: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_a.empty() || params.path_b.empty()) {
        fprintf(stderr, "Usage: quant-sampling compare -a <ref.bin> -b <target.bin> [--optimize]\n"
                        "  --temp-min <val>   temperature scan start (default: ref_temp * 0.5)\n"
                        "  --temp-max <val>   temperature scan end   (default: ref_temp * 1.5)\n"
                        "  --temp-step <val>  temperature scan step  (default: 0.01)\n"
                        "  --csv <path>       export full prompt x temperature KL data to CSV\n"
                        "  --rank-csv <path>  export KL contribution by logit rank bucket\n");
        return false;
    }
    return true;
}

struct prompt_stats {
    int32_t n_logits;
    double kl_mean;
    double top1_pct;
};

// Compute KL stats for a single prompt pair.
static prompt_stats compute_prompt_stats(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int32_t n_vocab, double ref_temp, int32_t n_threads
) {
    const int32_t n_logits = pa.n_tokens - 1;

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

    return { n_logits, kl_sum / n_logits, 100.0 * top1_agree.load() / n_logits };
}

// Compute mean KL for a prompt pair at a given target temperature.
static double compute_kl_at_temp(
    const qmlog_prompt & pa, const qmlog_prompt & pb,
    int32_t n_vocab, double target_temp, double ref_temp
) {
    const int32_t n_logits = pa.n_tokens - 1;
    std::vector<double> log_p_ref, log_p_tgt;
    double total_kl = 0.0;
    for (int32_t i = 0; i < n_logits; i++) {
        const float * la = pa.logits.data() + (size_t)i * n_vocab;
        const float * lb = pb.logits.data() + (size_t)i * n_vocab;
        log_softmax_temp(la, n_vocab, ref_temp, log_p_ref);
        log_softmax_temp(lb, n_vocab, target_temp, log_p_tgt);
        total_kl += kl_divergence(log_p_ref, log_p_tgt, n_vocab);
    }
    return total_kl / n_logits;
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
    for (int32_t p = 0; p < fa.n_prompts; p++) {
        if (fa.prompts[p].n_tokens != fb.prompts[p].n_tokens) {
            fprintf(stderr, "compare: token count mismatch in prompt %d (%d vs %d)\n",
                    p + 1, fa.prompts[p].n_tokens, fb.prompts[p].n_tokens);
            return 1;
        }
    }

    const int32_t n_vocab   = fa.n_vocab;
    const int32_t n_prompts = fa.n_prompts;
    const int32_t n_threads = std::max(1, (int32_t)std::thread::hardware_concurrency());

    const double ref_temp = (fa.temp > 0.0f) ? (double)fa.temp : 1.0;
    fprintf(stderr, "compare: n_vocab=%d, n_prompts=%d, ref_temp=%.2f\n",
            n_vocab, n_prompts, ref_temp);

    // === Per-prompt statistics ===
    std::vector<prompt_stats> stats(n_prompts);
    for (int32_t p = 0; p < n_prompts; p++) {
        stats[p] = compute_prompt_stats(fa.prompts[p], fb.prompts[p], n_vocab, ref_temp, n_threads);
    }

    printf("\n=== Per-Prompt Statistics ===\n");
    printf("  %3s  %-20s  %6s  %9s  %7s\n", "#", "Path", "Tokens", "KL mean", "Top-1%");
    for (int32_t p = 0; p < n_prompts; p++) {
        const char * path = fa.prompts[p].path.empty() ? "" : fa.prompts[p].path.c_str();
        printf("  %3d  %-20s  %6d  %9.6f  %6.1f%%\n",
               p + 1, path, stats[p].n_logits, stats[p].kl_mean, stats[p].top1_pct);
    }

    // aggregate
    double kl_sum = 0.0, kl_sq = 0.0;
    double t1_sum = 0.0, t1_sq = 0.0;
    for (int32_t p = 0; p < n_prompts; p++) {
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

    // === Rank analysis ===
    if (!params.rank_csv_path.empty()) {
        fprintf(stderr, "compare: computing rank-stratified KL...\n");

        // bucket boundaries: top-1, top-2–10, top-11–100, tail (101+)
        static const int32_t BUCKET_BOUNDS[] = {1, 10, 100};
        static const int32_t N_BUCKETS = 4;
        static const char * BUCKET_NAMES[] = {"top1", "top2_10", "top11_100", "tail"};

        struct rank_stats {
            double kl_contribution[4] = {};  // sum of per-token KL contributions
            double prob_ref[4]        = {};  // sum of ref probability mass
            double prob_tgt[4]        = {};  // sum of target probability mass
            int32_t count             = 0;   // number of positions averaged over
        };

        std::vector<rank_stats> per_prompt_rank(n_prompts);

        for (int32_t p = 0; p < n_prompts; p++) {
            const auto & pa = fa.prompts[p];
            const auto & pb = fb.prompts[p];
            const int32_t n_logits = pa.n_tokens - 1;

            rank_stats & rs = per_prompt_rank[p];
            rs.count = n_logits;

            std::vector<double> log_p_ref, log_p_tgt;
            std::vector<int32_t> rank_order(n_vocab);

            for (int32_t i = 0; i < n_logits; i++) {
                const float * la = pa.logits.data() + (size_t)i * n_vocab;
                const float * lb = pb.logits.data() + (size_t)i * n_vocab;

                log_softmax_temp(la, n_vocab, ref_temp, log_p_ref);
                log_softmax_temp(lb, n_vocab, ref_temp, log_p_tgt);

                // sort by ref probability (descending)
                std::iota(rank_order.begin(), rank_order.end(), 0);
                std::partial_sort(rank_order.begin(), rank_order.begin() + std::min((int32_t)100, n_vocab),
                                  rank_order.end(),
                                  [&](int32_t a, int32_t b) { return log_p_ref[a] > log_p_ref[b]; });

                // accumulate into buckets
                int32_t bucket = 0;
                for (int32_t rank = 0; rank < n_vocab; rank++) {
                    // advance bucket if needed
                    while (bucket < N_BUCKETS - 1 && rank >= BUCKET_BOUNDS[bucket]) bucket++;

                    int32_t tok = rank_order[rank];
                    double p_r = exp(log_p_ref[tok]);
                    double p_t = exp(log_p_tgt[tok]);
                    double kl_tok = (p_r > 1e-30) ? p_r * (log_p_ref[tok] - log_p_tgt[tok]) : 0.0;

                    rs.kl_contribution[bucket] += kl_tok;
                    rs.prob_ref[bucket]        += p_r;
                    rs.prob_tgt[bucket]        += p_t;
                }
            }

            // average over positions
            for (int32_t b = 0; b < N_BUCKETS; b++) {
                rs.kl_contribution[b] /= n_logits;
                rs.prob_ref[b]        /= n_logits;
                rs.prob_tgt[b]        /= n_logits;
            }

            fprintf(stderr, "  rank analysis: prompt %d / %d\r", p + 1, n_prompts);
            fflush(stderr);
        }
        fprintf(stderr, "\n");

        // print summary
        printf("\n=== Rank Analysis (averaged across positions) ===\n");
        printf("  %3s  %-20s  %-10s  %10s  %8s  %8s  %8s\n",
               "#", "Path", "Bucket", "KL contrib", "P(ref)", "P(tgt)", "P shift");
        for (int32_t p = 0; p < n_prompts; p++) {
            const char * path = fa.prompts[p].path.empty() ? "" : fa.prompts[p].path.c_str();
            const rank_stats & rs = per_prompt_rank[p];
            for (int32_t b = 0; b < N_BUCKETS; b++) {
                double shift = rs.prob_tgt[b] - rs.prob_ref[b];
                printf("  %3d  %-20s  %-10s  %10.6f  %8.4f  %8.4f  %+8.4f\n",
                       p + 1, path, BUCKET_NAMES[b],
                       rs.kl_contribution[b], rs.prob_ref[b], rs.prob_tgt[b], shift);
            }
        }

        // write CSV
        FILE * csv = fopen(params.rank_csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "compare: cannot open rank CSV '%s'\n", params.rank_csv_path.c_str());
            return 1;
        }
        fprintf(csv, "prompt,path,bucket,kl_contribution,prob_ref,prob_target,prob_shift\n");
        for (int32_t p = 0; p < n_prompts; p++) {
            const char * path = fa.prompts[p].path.empty() ? "" : fa.prompts[p].path.c_str();
            const rank_stats & rs = per_prompt_rank[p];
            for (int32_t b = 0; b < N_BUCKETS; b++) {
                double shift = rs.prob_tgt[b] - rs.prob_ref[b];
                fprintf(csv, "%d,%s,%s,%.6f,%.4f,%.4f,%+.4f\n",
                        p + 1, path, BUCKET_NAMES[b],
                        rs.kl_contribution[b], rs.prob_ref[b], rs.prob_tgt[b], shift);
            }
        }
        fclose(csv);
        fprintf(stderr, "compare: wrote %s (%d prompts x %d buckets)\n",
                params.rank_csv_path.c_str(), n_prompts, N_BUCKETS);
    }

    if (!params.optimize) return 0;

    // === Temperature optimization across all prompts ===
    printf("\n=== Temperature Optimization ===\n");

    // derive defaults from ref_temp, let user overrides take precedence
    double t_min  = (params.temp_min  != UNSET) ? params.temp_min  : ref_temp * 0.5;
    double t_max  = (params.temp_max  != UNSET) ? params.temp_max  : ref_temp * 1.5;
    double t_step = (params.temp_step != UNSET) ? params.temp_step : 0.01;

    printf("  ref_temp=%.2f  scan: min=%.2f%s  max=%.2f%s  step=%.2f%s\n",
           ref_temp,
           t_min,  (params.temp_min  != UNSET) ? " (override)" : "",
           t_max,  (params.temp_max  != UNSET) ? " (override)" : "",
           t_step, (params.temp_step != UNSET) ? " (override)" : "");

    std::vector<double> temp_vals;
    for (double T = t_min + t_step; T <= t_max + 1e-9; T += t_step) {
        temp_vals.push_back(T);
    }
    const int32_t n_temps = (int32_t)temp_vals.size();

    if (n_temps == 0) {
        printf("  (empty scan range, nothing to do)\n");
        return 0;
    }

    // per_prompt_kl[p][ti]
    std::vector<std::vector<double>> per_prompt_kl(n_prompts, std::vector<double>(n_temps, 0.0));

    // total work = n_temps * sum(n_logits per prompt)
    int64_t total_logits = 0;
    for (int32_t p = 0; p < n_prompts; p++) {
        total_logits += fa.prompts[p].n_tokens - 1;
    }
    int64_t total_work = (int64_t)n_temps * total_logits;
    std::atomic<int64_t> work_done{0};
    std::atomic<int32_t> last_pct{-1};

    // parallelize across temperature indices
    std::atomic<int32_t> temp_ctr{0};
    auto temp_worker = [&]() {
        while (true) {
            int32_t ti = temp_ctr.fetch_add(1);
            if (ti >= n_temps) break;
            for (int32_t p = 0; p < n_prompts; p++) {
                int32_t n_logits_p = fa.prompts[p].n_tokens - 1;
                per_prompt_kl[p][ti] = compute_kl_at_temp(
                    fa.prompts[p], fb.prompts[p], n_vocab, temp_vals[ti], ref_temp);
                int64_t done = work_done.fetch_add(n_logits_p) + n_logits_p;
                int32_t pct = (int32_t)(100 * done / total_work);
                int32_t prev = last_pct.load();
                if (pct > prev && last_pct.compare_exchange_strong(prev, pct)) {
                    fprintf(stderr, "  %3d%%\r", pct);
                    fflush(stderr);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (int32_t t = 0; t < n_threads; t++) threads.emplace_back(temp_worker);
    for (auto & t : threads) t.join();
    fprintf(stderr, "  100%%\n");

    // per-prompt best temperature
    printf("  %6s  %-20s  %6s  %10s\n", "Prompt", "Path", "Best T", "KL at best");
    for (int32_t p = 0; p < n_prompts; p++) {
        int32_t best_ti = 0;
        for (int32_t ti = 1; ti < n_temps; ti++) {
            if (per_prompt_kl[p][ti] < per_prompt_kl[p][best_ti]) best_ti = ti;
        }
        const char * path = fa.prompts[p].path.empty() ? "" : fa.prompts[p].path.c_str();
        printf("  %6d  %-20s  %6.2f  %10.6f\n", p + 1, path, temp_vals[best_ti], per_prompt_kl[p][best_ti]);
    }

    // CSV export
    if (!params.csv_path.empty()) {
        FILE * csv = fopen(params.csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "compare: cannot open CSV file '%s'\n", params.csv_path.c_str());
            return 1;
        }
        fprintf(csv, "prompt,path,temp,kl\n");
        for (int32_t p = 0; p < n_prompts; p++) {
            const char * path = fa.prompts[p].path.empty() ? "" : fa.prompts[p].path.c_str();
            for (int32_t ti = 0; ti < n_temps; ti++) {
                fprintf(csv, "%d,%s,%.4f,%.6f\n", p + 1, path, temp_vals[ti], per_prompt_kl[p][ti]);
            }
        }
        fclose(csv);
        fprintf(stderr, "compare: wrote %s (%d prompts x %d temps)\n",
                params.csv_path.c_str(), n_prompts, n_temps);
    }

    return 0;
}
