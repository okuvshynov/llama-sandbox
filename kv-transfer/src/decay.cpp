#include "decay.h"
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

struct decay_params {
    std::string path_ref;
    std::string path_target;
    std::string path_handoff;
    std::string csv_path;
    int32_t     window = 64;
    double      temp   = 0.6;
};

static bool parse_args(int argc, char ** argv, decay_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "--ref") == 0 && i + 1 < argc) {
            params.path_ref = argv[++i];
        } else if (strcmp(arg, "--target") == 0 && i + 1 < argc) {
            params.path_target = argv[++i];
        } else if (strcmp(arg, "--handoff") == 0 && i + 1 < argc) {
            params.path_handoff = argv[++i];
        } else if (strcmp(arg, "--csv") == 0 && i + 1 < argc) {
            params.csv_path = argv[++i];
        } else if (strcmp(arg, "--window") == 0 && i + 1 < argc) {
            params.window = atoi(argv[++i]);
        } else if (strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            params.temp = atof(argv[++i]);
        } else {
            fprintf(stderr, "decay: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_ref.empty() || params.path_target.empty() || params.path_handoff.empty()) {
        fprintf(stderr, "Usage: kv-transfer decay --ref <ref.bin> --target <target.bin> --handoff <handoff.bin> [options]\n"
                        "  --csv <path>     output CSV\n"
                        "  --window <int>   window size (default: 64)\n"
                        "  --temp <float>   temperature (default: 0.6)\n");
        return false;
    }
    return true;
}

struct window_stats {
    int32_t start;
    int32_t end;
    double  kl_target;
    double  top1_target;
    double  kl_handoff;
    double  top1_handoff;
};

static void compute_window_kl(
    const trace_entry & ref, const trace_entry & other,
    int32_t n_vocab, double temp, int32_t gen_start, int32_t win_start, int32_t win_end,
    double & kl_out, double & top1_out, int32_t n_threads
) {
    const int32_t n = win_end - win_start;
    std::vector<double> kl_per_pos(n);
    std::atomic<int32_t> agree{0};
    std::atomic<int32_t> counter{0};

    auto worker = [&]() {
        std::vector<double> lp_ref, lp_other;
        int32_t local_agree = 0;
        while (true) {
            int32_t idx = counter.fetch_add(1);
            if (idx >= n) break;
            int32_t li = gen_start + win_start + idx;

            const float * la = ref.logits.data() + (size_t)li * n_vocab;
            const float * lb = other.logits.data() + (size_t)li * n_vocab;

            log_softmax_temp(la, n_vocab, temp, lp_ref);
            log_softmax_temp(lb, n_vocab, temp, lp_other);

            kl_per_pos[idx] = kl_divergence(lp_ref, lp_other, n_vocab);
            if (argmax(la, n_vocab) == argmax(lb, n_vocab)) local_agree++;
        }
        agree.fetch_add(local_agree);
    };

    std::vector<std::thread> threads;
    for (int32_t t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for (auto & t : threads) t.join();

    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) sum += kl_per_pos[i];
    kl_out = sum / n;
    top1_out = 100.0 * agree.load() / n;
}

int cmd_decay(int argc, char ** argv) {
    decay_params params;
    if (!parse_args(argc, argv, params)) return 1;

    fprintf(stderr, "decay: loading ref...\n");
    trace_file fa;
    if (!trace_read(params.path_ref, fa)) return 1;

    fprintf(stderr, "decay: loading target...\n");
    trace_file fb;
    if (!trace_read(params.path_target, fb)) return 1;

    fprintf(stderr, "decay: loading handoff...\n");
    trace_file fc;
    if (!trace_read(params.path_handoff, fc)) return 1;

    if (fa.n_prompts != 1 || fb.n_prompts != 1 || fc.n_prompts != 1) {
        fprintf(stderr, "decay: expected single-prompt files\n");
        return 1;
    }

    const auto & ref = fa.prompts[0];
    const auto & tgt = fb.prompts[0];
    const auto & hoff = fc.prompts[0];
    const int32_t n_vocab = fa.n_vocab;
    const int32_t n_logits = ref.n_tokens - 1;
    const int32_t gen_start = ref.n_prompt - 1;
    const int32_t gen_count = n_logits - gen_start;
    const int32_t n_threads = std::max(1, (int32_t)std::thread::hardware_concurrency());

    fprintf(stderr, "decay: n_vocab=%d, n_prompt=%d, gen=%d, window=%d, threads=%d\n",
            n_vocab, ref.n_prompt, gen_count, params.window, n_threads);

    std::vector<window_stats> windows;
    int32_t pos = 0;
    while (pos < gen_count) {
        int32_t end = std::min(pos + params.window, gen_count);

        window_stats ws;
        ws.start = pos;
        ws.end = end;

        compute_window_kl(ref, tgt, n_vocab, params.temp, gen_start, pos, end,
                          ws.kl_target, ws.top1_target, n_threads);
        compute_window_kl(ref, hoff, n_vocab, params.temp, gen_start, pos, end,
                          ws.kl_handoff, ws.top1_handoff, n_threads);

        windows.push_back(ws);
        pos = end;

        fprintf(stderr, "  window %d-%d done\r", ws.start, ws.end);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    // print table
    printf("%10s  %10s  %10s  %10s  %11s  %10s\n",
           "Window", "KL(tgt)", "top1%(tgt)", "KL(hoff)", "top1%(hoff)", "KL ratio");
    for (const auto & ws : windows) {
        double ratio = ws.kl_target > 0 ? ws.kl_handoff / ws.kl_target : 0;
        printf("%4d-%-4d   %10.4f  %9.1f%%  %10.4f  %10.1f%%  %10.3f\n",
               ws.start, ws.end, ws.kl_target, ws.top1_target,
               ws.kl_handoff, ws.top1_handoff, ratio);
    }

    // CSV
    if (!params.csv_path.empty()) {
        FILE * csv = fopen(params.csv_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "decay: cannot open '%s'\n", params.csv_path.c_str());
            return 1;
        }
        fprintf(csv, "window_start,window_end,kl_target,top1_target,kl_handoff,top1_handoff\n");
        for (const auto & ws : windows) {
            fprintf(csv, "%d,%d,%.6f,%.1f,%.6f,%.1f\n",
                    ws.start, ws.end, ws.kl_target, ws.top1_target,
                    ws.kl_handoff, ws.top1_handoff);
        }
        fclose(csv);
        fprintf(stderr, "decay: wrote %s\n", params.csv_path.c_str());
    }

    return 0;
}
