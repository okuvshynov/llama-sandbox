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
        fprintf(stderr, "Usage: quant-sampling compare -a <ref.qmlog> -b <quant.qmlog> [--optimize]\n"
                        "  --temp-min <val>   temperature scan start (default 0.0)\n"
                        "  --temp-max <val>   temperature scan end   (default 2.0)\n"
                        "  --temp-step <val>  temperature scan step  (default 0.05)\n");
        return false;
    }
    return true;
}

int cmd_compare(int argc, char ** argv) {
    compare_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    // load both files
    fprintf(stderr, "compare: loading '%s'...\n", params.path_a.c_str());
    qmlog_file fa;
    if (!qmlog_read(params.path_a, fa)) return 1;

    fprintf(stderr, "compare: loading '%s'...\n", params.path_b.c_str());
    qmlog_file fb;
    if (!qmlog_read(params.path_b, fb)) return 1;

    // validate
    if (fa.header.n_vocab != fb.header.n_vocab) {
        fprintf(stderr, "compare: vocab size mismatch (%d vs %d)\n",
                fa.header.n_vocab, fb.header.n_vocab);
        return 1;
    }
    if (fa.header.n_tokens != fb.header.n_tokens) {
        fprintf(stderr, "compare: token count mismatch (%d vs %d)\n",
                fa.header.n_tokens, fb.header.n_tokens);
        return 1;
    }

    const int n_vocab   = fa.header.n_vocab;
    const int n_tokens  = fa.header.n_tokens;
    const int n_logits  = n_tokens - 1; // number of logit vectors

    fprintf(stderr, "compare: n_vocab=%d, n_tokens=%d, n_logits=%d\n",
            n_vocab, n_tokens, n_logits);

    // --- basic statistics ---
    const int n_threads = std::max(1, (int)std::thread::hardware_concurrency());

    // use reference temperature from the generate step (e.g. 0.6)
    // so we compare distributions at the actual sampling temperature
    const double ref_temp = (fa.header.temp > 0.0f) ? (double)fa.header.temp : 1.0;

    fprintf(stderr, "compare: using reference temperature = %.2f\n", ref_temp);

    std::vector<double> kl_per_pos(n_logits);
    std::atomic<int>    top1_agree{0};
    std::vector<double> prob_diff(n_logits);

    // work-stealing parallel loop
    std::atomic<int> counter{0};
    auto worker_fn = [&]() {
        std::vector<double> log_p_ref, log_p_quant;
        int local_agree = 0;

        while (true) {
            int i = counter.fetch_add(1);
            if (i >= n_logits) break;

            const float * logits_a = fa.logits.data() + (size_t)i * n_vocab;
            const float * logits_b = fb.logits.data() + (size_t)i * n_vocab;

            log_softmax_temp(logits_a, n_vocab, ref_temp, log_p_ref);
            log_softmax_temp(logits_b, n_vocab, ref_temp, log_p_quant);

            kl_per_pos[i] = kl_divergence(log_p_ref, log_p_quant, n_vocab);

            // top-1 agreement (at reference temperature)
            // argmax is temperature-invariant, so raw logits are fine
            if (argmax(logits_a, n_vocab) == argmax(logits_b, n_vocab)) {
                local_agree++;
            }

            // probability difference for "correct" next token
            int next_tok = fa.tokens[i + 1];
            double p_ref   = exp(log_p_ref[next_tok]);
            double p_quant = exp(log_p_quant[next_tok]);
            prob_diff[i] = p_ref - p_quant;
        }

        top1_agree.fetch_add(local_agree);
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back(worker_fn);
    }
    for (auto & t : threads) t.join();

    // compute stats
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
    double pd_mean = pd_sum / n_logits;
    double pd_rms  = sqrt(pd_sq_sum / n_logits);

    printf("\n=== Basic Statistics ===\n");
    printf("KL divergence (mean):     %.6f nats\n", kl_mean);
    printf("KL divergence (std):      %.6f nats\n", kl_std);
    printf("Top-1 agreement:          %.1f%% (%d / %d)\n",
           100.0 * top1_agree.load() / n_logits, top1_agree.load(), n_logits);
    printf("Prob diff for next token:\n");
    printf("  Mean:                   %+.6f\n", pd_mean);
    printf("  RMS:                    %.6f\n", pd_rms);

    // top-5 highest KL positions
    std::vector<int> pos_idx(n_logits);
    std::iota(pos_idx.begin(), pos_idx.end(), 0);
    std::partial_sort(pos_idx.begin(), pos_idx.begin() + std::min(5, n_logits),
                      pos_idx.end(), [&](int a, int b) {
                          return kl_per_pos[a] > kl_per_pos[b];
                      });

    printf("\nTop-5 highest KL positions:\n");
    for (int k = 0; k < std::min(5, n_logits); k++) {
        int idx = pos_idx[k];
        printf("  pos %4d: KL = %.4f  (token %d → %d)\n",
               idx, kl_per_pos[idx], fa.tokens[idx], fa.tokens[idx + 1]);
    }

    if (!params.optimize) {
        return 0;
    }

    // --- V1: temperature-only 1D search ---
    printf("\n=== Temperature Optimization (1D) ===\n");

    struct temp_result {
        double temp;
        double kl_mean;
    };

    // scan T from (temp_min + temp_step) to temp_max in steps of temp_step
    // temp_min=0 naturally starts from the first nonzero step
    std::vector<temp_result> temp_results;
    for (double T = params.temp_min + params.temp_step; T <= params.temp_max + 1e-9; T += params.temp_step) {
        temp_results.push_back({T, 0.0});
    }

    // parallel evaluation of temperatures
    for (auto & tr : temp_results) {
        std::atomic<int> ctr{0};

        // each thread accumulates locally
        std::vector<double> local_kl_sums(n_threads, 0.0);

        auto temp_worker = [&](int tid) {
            std::vector<double> log_p_ref, log_p_quant;
            double local_sum = 0.0;

            while (true) {
                int i = ctr.fetch_add(1);
                if (i >= n_logits) break;

                const float * logits_a = fa.logits.data() + (size_t)i * n_vocab;
                const float * logits_b = fb.logits.data() + (size_t)i * n_vocab;

                log_softmax_temp(logits_a, n_vocab, ref_temp, log_p_ref);
                log_softmax_temp(logits_b, n_vocab, tr.temp, log_p_quant);

                local_sum += kl_divergence(log_p_ref, log_p_quant, n_vocab);
            }

            local_kl_sums[tid] = local_sum;
        };

        std::vector<std::thread> thr;
        for (int t = 0; t < n_threads; t++) {
            thr.emplace_back(temp_worker, t);
        }
        for (auto & t : thr) t.join();

        double total = 0.0;
        for (auto v : local_kl_sums) total += v;
        tr.kl_mean = total / n_logits;
    }

    // find best temperature
    auto best_temp = std::min_element(temp_results.begin(), temp_results.end(),
                                      [](const temp_result & a, const temp_result & b) {
                                          return a.kl_mean < b.kl_mean;
                                      });

    printf("Scanned %zu temperatures:\n", temp_results.size());
    printf("  Baseline (T=%.2f):  KL = %.6f\n", ref_temp, kl_mean);
    printf("  Best temperature:   T = %.2f, KL = %.6f\n", best_temp->temp, best_temp->kl_mean);
    printf("  KL reduction:       %.1f%%\n",
           100.0 * (1.0 - best_temp->kl_mean / kl_mean));

    // print a few samples around the optimum
    printf("\nTemperature scan (selected):\n");
    for (const auto & tr : temp_results) {
        // print every 5th value, plus anything within 0.15 of optimum
        int idx = (int)(&tr - &temp_results[0]);
        bool near_opt = fabs(tr.temp - best_temp->temp) < 0.16;
        if (idx % 5 == 0 || near_opt) {
            printf("  T=%.2f  KL=%.6f%s\n", tr.temp, tr.kl_mean,
                   (fabs(tr.temp - best_temp->temp) < 0.001) ? "  <-- best" : "");
        }
    }

    printf("\nRecommended temperature for quantized model:  --temp %.2f\n", best_temp->temp);

    return 0;
}
