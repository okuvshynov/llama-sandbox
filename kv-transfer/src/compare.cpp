#include "compare.h"
#include "stats_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct compare_params {
    std::string path;
};

static bool parse_args(int argc, char ** argv, compare_params & params) {
    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (strcmp(arg, "-f") == 0 && i + 1 < argc) {
            params.path = argv[++i];
        } else {
            fprintf(stderr, "compare: unknown argument '%s'\n", arg);
            return false;
        }
    }
    if (params.path.empty()) {
        fprintf(stderr, "Usage: kv-transfer compare -f <stats.bin>\n");
        return false;
    }
    return true;
}

int cmd_compare(int argc, char ** argv) {
    compare_params params;
    if (!parse_args(argc, argv, params)) return 1;

    stats_file sf;
    if (!stats_read(params.path, sf)) return 1;

    const int32_t n = sf.n_gen;
    if (n <= 0) {
        fprintf(stderr, "compare: no generation tokens\n");
        return 1;
    }

    // mean KL
    double kl_sum = 0.0;
    for (int32_t i = 0; i < n; i++) kl_sum += sf.kl[i];
    double kl_mean = kl_sum / n;

    // top-1 agreement
    int32_t agree = 0;
    for (int32_t i = 0; i < n; i++) agree += sf.top1_match[i];
    double top1_pct = 100.0 * agree / n;

    // percentiles
    std::vector<float> sorted_kl(sf.kl.begin(), sf.kl.end());
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
