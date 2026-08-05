#pragma once

#include "logits_file.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// Extract the top-K entries and the full-vocab log-sum-exp from one logit row.
// idx_buf is caller-provided scratch to avoid re-allocating n_vocab ints per row.
inline lkld_position extract_topk_lse(const float * row, int32_t n_vocab, int32_t k,
                                      std::vector<int32_t> & idx_buf) {
    lkld_position out;

    float max_logit = row[0];
    for (int32_t j = 1; j < n_vocab; j++) {
        if (row[j] > max_logit) max_logit = row[j];
    }

    // double accumulation: fp32 would lose precision summing ~150k terms
    double sum_exp = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        sum_exp += exp((double)row[j] - max_logit);
    }

    out.max_logit = max_logit;
    out.lse_rest  = (float)log(sum_exp);

    k = std::min(k, n_vocab);
    idx_buf.resize(n_vocab);
    std::iota(idx_buf.begin(), idx_buf.end(), 0);
    auto cmp = [row](int32_t a, int32_t b) {
        if (row[a] != row[b]) return row[a] > row[b];
        return a < b;
    };
    std::nth_element(idx_buf.begin(), idx_buf.begin() + k, idx_buf.end(), cmp);
    std::sort(idx_buf.begin(), idx_buf.begin() + k, cmp);

    out.ids.resize(k);
    out.logits.resize(k);
    for (int32_t j = 0; j < k; j++) {
        out.ids[j]    = idx_buf[j];
        out.logits[j] = row[idx_buf[j]];
    }
    return out;
}

// Probability mass NOT covered by the stored top-K entries.
inline double tail_mass(const lkld_position & p) {
    double lse = (double)p.max_logit + (double)p.lse_rest;
    double covered = 0.0;
    for (float l : p.logits) {
        covered += exp((double)l - lse);
    }
    return 1.0 - covered;
}
