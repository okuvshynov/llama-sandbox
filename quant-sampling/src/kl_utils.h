#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

// Compute log-softmax for a logit vector, returning log-probs in out[].
// Uses numerically stable max-subtraction.
inline void log_softmax(const float * logits, int32_t n_vocab, std::vector<double> & out) {
    out.resize(n_vocab);

    double max_val = logits[0];
    for (int32_t j = 1; j < n_vocab; j++) {
        if (logits[j] > max_val) max_val = logits[j];
    }

    double sum_exp = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        sum_exp += exp((double)logits[j] - max_val);
    }
    double log_sum = log(sum_exp);

    for (int32_t j = 0; j < n_vocab; j++) {
        out[j] = (double)logits[j] - max_val - log_sum;
    }
}

// Compute log-softmax with temperature scaling applied to logits.
inline void log_softmax_temp(const float * logits, int32_t n_vocab, double temp,
                             std::vector<double> & out) {
    out.resize(n_vocab);

    double inv_t = 1.0 / temp;
    double max_val = (double)logits[0] * inv_t;
    for (int32_t j = 1; j < n_vocab; j++) {
        double v = (double)logits[j] * inv_t;
        if (v > max_val) max_val = v;
    }

    double sum_exp = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        sum_exp += exp((double)logits[j] * inv_t - max_val);
    }
    double log_sum = log(sum_exp);

    for (int32_t j = 0; j < n_vocab; j++) {
        out[j] = (double)logits[j] * inv_t - max_val - log_sum;
    }
}

// Compute the truncation mask from a reference distribution: apply top-k, then top-p.
// Returns a bitmask of which tokens to keep. Uses idx_buf as scratch space.
inline void compute_truncation_mask(const float * logits_ref, int32_t n_vocab, double ref_temp,
                                    int32_t top_k, double top_p,
                                    std::vector<bool> & mask,
                                    std::vector<int32_t> & idx_buf) {
    // compute reference log-probs
    std::vector<double> lp(n_vocab);
    double inv_t = 1.0 / ref_temp;
    double max_val = (double)logits_ref[0] * inv_t;
    for (int32_t j = 1; j < n_vocab; j++) {
        double v = (double)logits_ref[j] * inv_t;
        if (v > max_val) max_val = v;
    }
    double sum_exp = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        lp[j] = (double)logits_ref[j] * inv_t - max_val;
        sum_exp += exp(lp[j]);
    }
    double log_sum = log(sum_exp);
    for (int32_t j = 0; j < n_vocab; j++) lp[j] -= log_sum;

    // sort by ref probability
    int32_t k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    idx_buf.resize(n_vocab);
    for (int32_t j = 0; j < n_vocab; j++) idx_buf[j] = j;
    std::partial_sort(idx_buf.begin(), idx_buf.begin() + k, idx_buf.end(),
                      [&](int32_t a, int32_t b) { return lp[a] > lp[b]; });

    // top-k + top-p
    double cumulative = 0.0;
    int32_t n_keep = 0;
    for (int32_t i = 0; i < k; i++) {
        cumulative += exp(lp[idx_buf[i]]);
        n_keep++;
        if (top_p > 0.0 && top_p < 1.0 && cumulative >= top_p) break;
    }

    mask.assign(n_vocab, false);
    for (int32_t i = 0; i < n_keep; i++) mask[idx_buf[i]] = true;
}

// Apply a precomputed truncation mask to log-softmax output: zero out
// non-kept tokens and renormalize the kept ones.
inline void apply_mask_and_renorm(std::vector<double> & log_probs, int32_t n_vocab,
                                  const std::vector<bool> & mask) {
    double kept_sum = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        if (mask[j]) kept_sum += exp(log_probs[j]);
    }
    double log_kept = log(kept_sum);
    for (int32_t j = 0; j < n_vocab; j++) {
        if (mask[j]) {
            log_probs[j] -= log_kept;
        } else {
            log_probs[j] = -1e30;
        }
    }
}

// KL(P || Q) = sum_j P[j] * (log P[j] - log Q[j])
inline double kl_divergence(const std::vector<double> & log_p,
                            const std::vector<double> & log_q, int32_t n_vocab) {
    double kl = 0.0;
    for (int32_t j = 0; j < n_vocab; j++) {
        double p_j = exp(log_p[j]);
        if (p_j > 1e-30) {
            kl += p_j * (log_p[j] - log_q[j]);
        }
    }
    return kl;
}

inline int32_t argmax(const float * v, int32_t n) {
    int32_t best = 0;
    for (int32_t j = 1; j < n; j++) {
        if (v[j] > v[best]) best = j;
    }
    return best;
}
