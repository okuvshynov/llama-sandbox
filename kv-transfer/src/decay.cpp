#include "decay.h"
#include "stats_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct decay_params {
    std::string path_target;
    std::string path_handoff;
    std::string csv_path;
    int32_t     window = 64;
};

static bool parse_args(int argc, char ** argv, decay_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "--target") == 0 && i + 1 < argc) {
            params.path_target = argv[++i];
        } else if (strcmp(arg, "--handoff") == 0 && i + 1 < argc) {
            params.path_handoff = argv[++i];
        } else if (strcmp(arg, "--csv") == 0 && i + 1 < argc) {
            params.csv_path = argv[++i];
        } else if (strcmp(arg, "--window") == 0 && i + 1 < argc) {
            params.window = atoi(argv[++i]);
        } else {
            fprintf(stderr, "decay: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path_target.empty() || params.path_handoff.empty()) {
        fprintf(stderr, "Usage: kv-transfer decay --target <target.bin> --handoff <handoff.bin> [options]\n"
                        "  --csv <path>     output CSV\n"
                        "  --window <int>   window size (default: 64)\n");
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

static void compute_window(const stats_file & sf, int32_t win_start, int32_t win_end,
                           double & kl_out, double & top1_out) {
    const int32_t n = win_end - win_start;
    double kl_sum = 0.0;
    int32_t agree = 0;
    for (int32_t i = win_start; i < win_end; i++) {
        kl_sum += sf.kl[i];
        agree  += sf.top1_match[i];
    }
    kl_out   = kl_sum / n;
    top1_out = 100.0 * agree / n;
}

int cmd_decay(int argc, char ** argv) {
    decay_params params;
    if (!parse_args(argc, argv, params)) return 1;

    stats_file tgt, hoff;
    if (!stats_read(params.path_target, tgt)) return 1;
    if (!stats_read(params.path_handoff, hoff)) return 1;

    if (tgt.n_gen != hoff.n_gen) {
        fprintf(stderr, "decay: n_gen mismatch (target=%d, handoff=%d)\n", tgt.n_gen, hoff.n_gen);
        return 1;
    }

    const int32_t gen_count = tgt.n_gen;

    fprintf(stderr, "decay: n_prompt=%d, gen=%d, window=%d\n",
            tgt.n_prompt, gen_count, params.window);

    std::vector<window_stats> windows;
    int32_t pos = 0;
    while (pos < gen_count) {
        int32_t end = std::min(pos + params.window, gen_count);

        window_stats ws;
        ws.start = pos;
        ws.end = end;

        compute_window(tgt, pos, end, ws.kl_target, ws.top1_target);
        compute_window(hoff, pos, end, ws.kl_handoff, ws.top1_handoff);

        windows.push_back(ws);
        pos = end;
    }

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
