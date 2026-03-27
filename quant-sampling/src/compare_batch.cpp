#include "compare_batch.h"
#include "kl_utils.h"
#include "logits_io.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct compare_batch_params {
    std::string manifest_path;
    bool        optimize  = false;
    double      temp_min  = 0.0;
    double      temp_max  = 2.0;
    double      temp_step = 0.05;
};

struct prompt_pair {
    std::string path_ref;
    std::string path_quant;
};

struct prompt_stats {
    int    n_tokens;
    double kl_mean;
    double top1_pct;
};

static bool parse_args(int argc, char ** argv, compare_batch_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "--manifest") == 0 && i + 1 < argc) {
            params.manifest_path = argv[++i];
        } else if (strcmp(arg, "--optimize") == 0) {
            params.optimize = true;
        } else if (strcmp(arg, "--temp-min") == 0 && i + 1 < argc) {
            params.temp_min = atof(argv[++i]);
        } else if (strcmp(arg, "--temp-max") == 0 && i + 1 < argc) {
            params.temp_max = atof(argv[++i]);
        } else if (strcmp(arg, "--temp-step") == 0 && i + 1 < argc) {
            params.temp_step = atof(argv[++i]);
        } else {
            fprintf(stderr, "compare-batch: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.manifest_path.empty()) {
        fprintf(stderr, "Usage: quant-sampling compare-batch --manifest <manifest.txt> [--optimize]\n"
                        "  --temp-min <val>   temperature scan start (default 0.0)\n"
                        "  --temp-max <val>   temperature scan end   (default 2.0)\n"
                        "  --temp-step <val>  temperature scan step  (default 0.05)\n");
        return false;
    }
    return true;
}

static bool parse_manifest(const std::string & path, std::vector<prompt_pair> & pairs) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "compare-batch: cannot open manifest '%s'\n", path.c_str());
        return false;
    }

    std::string line;
    int lineno = 0;
    while (std::getline(ifs, line)) {
        lineno++;
        // skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        prompt_pair pp;
        if (!(iss >> pp.path_ref >> pp.path_quant)) {
            fprintf(stderr, "compare-batch: malformed manifest line %d: '%s'\n",
                    lineno, line.c_str());
            return false;
        }
        pairs.push_back(std::move(pp));
    }

    if (pairs.empty()) {
        fprintf(stderr, "compare-batch: manifest is empty\n");
        return false;
    }
    return true;
}

// Compute basic stats (KL mean, top-1 agreement) for one prompt pair.
static prompt_stats compute_prompt_stats(const qmlog_file & fa, const qmlog_file & fb,
                                         int n_vocab, int n_logits, int n_threads) {
    std::vector<double> kl_per_pos(n_logits);
    std::atomic<int> top1_agree{0};
    std::atomic<int> counter{0};

    auto worker = [&]() {
        std::vector<double> log_p_ref, log_p_quant;
        int local_agree = 0;

        while (true) {
            int i = counter.fetch_add(1);
            if (i >= n_logits) break;

            const float * logits_a = fa.logits.data() + (size_t)i * n_vocab;
            const float * logits_b = fb.logits.data() + (size_t)i * n_vocab;

            log_softmax(logits_a, n_vocab, log_p_ref);
            log_softmax(logits_b, n_vocab, log_p_quant);

            kl_per_pos[i] = kl_divergence(log_p_ref, log_p_quant, n_vocab);

            if (argmax(logits_a, n_vocab) == argmax(logits_b, n_vocab)) {
                local_agree++;
            }
        }

        top1_agree.fetch_add(local_agree);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for (auto & t : threads) t.join();

    double kl_sum = 0.0;
    for (int i = 0; i < n_logits; i++) kl_sum += kl_per_pos[i];

    prompt_stats ps;
    ps.n_tokens = n_logits;
    ps.kl_mean  = kl_sum / n_logits;
    ps.top1_pct = 100.0 * top1_agree.load() / n_logits;
    return ps;
}

// Compute mean KL for a single prompt pair at a given temperature.
static double compute_kl_at_temp(const qmlog_file & fa, const qmlog_file & fb,
                                 int n_vocab, int n_logits, double temp) {
    std::vector<double> log_p_ref, log_p_quant;
    double total_kl = 0.0;
    for (int i = 0; i < n_logits; i++) {
        const float * logits_a = fa.logits.data() + (size_t)i * n_vocab;
        const float * logits_b = fb.logits.data() + (size_t)i * n_vocab;

        log_softmax(logits_a, n_vocab, log_p_ref);
        log_softmax_temp(logits_b, n_vocab, temp, log_p_quant);

        total_kl += kl_divergence(log_p_ref, log_p_quant, n_vocab);
    }
    return total_kl / n_logits;
}


int cmd_compare_batch(int argc, char ** argv) {
    compare_batch_params params;
    if (!parse_args(argc, argv, params)) return 1;

    // parse manifest
    std::vector<prompt_pair> pairs;
    if (!parse_manifest(params.manifest_path, pairs)) return 1;

    const int n_prompts = (int)pairs.size();
    fprintf(stderr, "compare-batch: %d prompt pairs in manifest\n", n_prompts);

    // load all files
    std::vector<qmlog_file> refs(n_prompts), quants(n_prompts);
    int n_vocab = 0;

    for (int p = 0; p < n_prompts; p++) {
        fprintf(stderr, "  loading pair %d: %s + %s\n", p + 1,
                pairs[p].path_ref.c_str(), pairs[p].path_quant.c_str());

        if (!qmlog_read(pairs[p].path_ref, refs[p])) return 1;
        if (!qmlog_read(pairs[p].path_quant, quants[p])) return 1;

        if (refs[p].header.n_vocab != quants[p].header.n_vocab) {
            fprintf(stderr, "compare-batch: vocab mismatch in pair %d (%d vs %d)\n",
                    p + 1, refs[p].header.n_vocab, quants[p].header.n_vocab);
            return 1;
        }
        if (refs[p].header.n_tokens != quants[p].header.n_tokens) {
            fprintf(stderr, "compare-batch: token count mismatch in pair %d (%d vs %d)\n",
                    p + 1, refs[p].header.n_tokens, quants[p].header.n_tokens);
            return 1;
        }

        if (p == 0) {
            n_vocab = refs[p].header.n_vocab;
        } else if (refs[p].header.n_vocab != n_vocab) {
            fprintf(stderr, "compare-batch: vocab size differs across pairs (%d vs %d)\n",
                    refs[p].header.n_vocab, n_vocab);
            return 1;
        }
    }

    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());

    // --- Per-Prompt Statistics ---
    std::vector<prompt_stats> stats(n_prompts);
    for (int p = 0; p < n_prompts; p++) {
        int n_logits = refs[p].header.n_tokens - 1;
        stats[p] = compute_prompt_stats(refs[p], quants[p], n_vocab, n_logits, n_threads);
    }

    printf("\n=== Per-Prompt Statistics ===\n");
    printf("  %3s  %6s  %9s  %7s  %s\n", "#", "Tokens", "KL mean", "Top-1%", "File");
    for (int p = 0; p < n_prompts; p++) {
        printf("  %3d  %6d  %9.6f  %6.1f%%  %s\n",
               p + 1, stats[p].n_tokens, stats[p].kl_mean, stats[p].top1_pct,
               pairs[p].path_ref.c_str());
    }

    // --- Aggregate ---
    double kl_sum = 0.0, kl_sq = 0.0;
    double t1_sum = 0.0, t1_sq = 0.0;
    for (int p = 0; p < n_prompts; p++) {
        kl_sum += stats[p].kl_mean;
        kl_sq  += stats[p].kl_mean * stats[p].kl_mean;
        t1_sum += stats[p].top1_pct;
        t1_sq  += stats[p].top1_pct * stats[p].top1_pct;
    }
    double kl_agg_mean = kl_sum / n_prompts;
    double kl_agg_std  = (n_prompts > 1)
        ? sqrt((kl_sq / n_prompts) - kl_agg_mean * kl_agg_mean)
        : 0.0;
    double t1_agg_mean = t1_sum / n_prompts;
    double t1_agg_std  = (n_prompts > 1)
        ? sqrt((t1_sq / n_prompts) - t1_agg_mean * t1_agg_mean)
        : 0.0;

    printf("\n=== Aggregate ===\n");
    printf("  KL divergence: %.6f +/- %.6f\n", kl_agg_mean, kl_agg_std);
    printf("  Top-1 agree:   %.1f%% +/- %.1f%%\n", t1_agg_mean, t1_agg_std);

    if (!params.optimize) return 0;

    // --- Temperature Optimization (per-prompt) ---
    printf("\n=== Temperature Optimization (per-prompt) ===\n");

    // temperature scan from (temp_min + temp_step) to temp_max
    // temp_min=0 naturally starts from the first nonzero step
    std::vector<double> temp_vals;
    for (double T = params.temp_min + params.temp_step; T <= params.temp_max + 1e-9; T += params.temp_step) {
        temp_vals.push_back(T);
    }
    const int n_temps = (int)temp_vals.size();

    // per_prompt_kl[p][ti] = mean KL for prompt p at temperature ti
    std::vector<std::vector<double>> per_prompt_kl(n_prompts, std::vector<double>(n_temps, 0.0));

    // parallelise across temperature indices
    std::atomic<int> temp_ctr{0};
    auto temp_worker = [&]() {
        while (true) {
            int ti = temp_ctr.fetch_add(1);
            if (ti >= n_temps) break;

            double T = temp_vals[ti];
            for (int p = 0; p < n_prompts; p++) {
                int n_logits = refs[p].header.n_tokens - 1;
                per_prompt_kl[p][ti] = compute_kl_at_temp(refs[p], quants[p],
                                                          n_vocab, n_logits, T);
            }

            if (ti % 10 == 0) {
                fprintf(stderr, "  temp scan: %d / %d\r", ti, n_temps);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) threads.emplace_back(temp_worker);
    for (auto & t : threads) t.join();
    fprintf(stderr, "\n");

    // find per-prompt best temperature
    std::vector<double> best_T(n_prompts);
    std::vector<double> best_kl(n_prompts);

    printf("  %6s  %6s  %10s\n", "Prompt", "Best T", "KL at best");
    for (int p = 0; p < n_prompts; p++) {
        int best_ti = 0;
        for (int ti = 1; ti < n_temps; ti++) {
            if (per_prompt_kl[p][ti] < per_prompt_kl[p][best_ti]) best_ti = ti;
        }
        best_T[p]  = temp_vals[best_ti];
        best_kl[p] = per_prompt_kl[p][best_ti];
        printf("  %6d  %6.2f  %10.6f\n", p + 1, best_T[p], best_kl[p]);
    }

    double T_sum = 0.0, T_sq = 0.0;
    for (int p = 0; p < n_prompts; p++) {
        T_sum += best_T[p];
        T_sq  += best_T[p] * best_T[p];
    }
    double T_mean = T_sum / n_prompts;
    double T_std  = (n_prompts > 1)
        ? sqrt((T_sq / n_prompts) - T_mean * T_mean)
        : 0.0;

    printf("\n  Mean optimal T: %.2f +/- %.2f\n", T_mean, T_std);
    printf("\nRecommended temperature for quantized model:  --temp %.2f\n", T_mean);

    return 0;
}
