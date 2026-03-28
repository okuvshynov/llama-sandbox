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
