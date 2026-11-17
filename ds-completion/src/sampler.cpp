#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

static uint64_t mix64(uint64_t z) {  // splitmix64's finalizer
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double sampler_uniform(uint64_t seed, uint64_t position) {
  const uint64_t h = mix64(mix64(seed + 0x9E3779B97F4A7C15ull) ^ (position * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull));
  return (double)(h >> 11) * 0x1.0p-53;
}

int32_t sample_token(const float* logits, int n, const SamplerOptions& opt, uint64_t position) {
  const int32_t best = (int32_t)(std::max_element(logits, logits + n) - logits);
  if (opt.temp <= 0) return best;
  const double inv_t = 1.0 / opt.temp, top = logits[best];
  std::vector<double> p(n);
  double total = 0;
  for (int i = 0; i < n; i++) {
    p[i] = std::exp(((double)logits[i] - top) * inv_t);  // <= 1; the maximum is exactly 1
    total += p[i];
  }
  const double u = sampler_uniform(opt.seed, position);
  if (opt.top_p >= 1) {
    const double target = u * total;
    double run = 0;
    int32_t last = best;
    for (int i = 0; i < n; i++) {
      if (p[i] == 0) continue;
      run += p[i];
      last = i;
      if (run > target) return i;
    }
    return last;  // (rounding: the running sum can end a little under the total)
  }
  // top-p. Sorting the whole vocabulary per token would be the cost here, and is not needed: the
  // tokens with p < floor together hold less than n * floor = half of what top-p leaves out,
  // so the tokens above the floor hold more than top_p of the total and the kept set, a prefix
  // of the sorted order, lies inside them.
  const double floor = 0.5 * (1.0 - opt.top_p) * total / n;
  std::vector<int32_t> cand;
  for (int i = 0; i < n; i++)
    if (p[i] >= floor) cand.push_back(i);
  // ... and of those only as many of the first as the sum needs are sorted: the first 256 by the
  // order (falling p, then rising id: a total order, so the result is the full sort's prefix),
  // then 8 times more while their sum is short
  const auto before = [&](int32_t a, int32_t b) { return p[a] != p[b] ? p[a] > p[b] : a < b; };
  const double want = (double)opt.top_p * total;
  size_t keep = 0, sorted = 0;
  double kept = 0;
  for (size_t k = 256; keep == 0 || kept < want; k *= 8) {
    const size_t end = std::min(k, cand.size());
    if (end < cand.size()) std::nth_element(cand.begin() + sorted, cand.begin() + end - 1, cand.end(), before);
    std::sort(cand.begin() + sorted, cand.begin() + end, before);
    sorted = end;
    while (keep < sorted) {
      kept += p[cand[keep++]];
      if (kept >= want) break;
    }
    if (sorted == cand.size()) break;
  }
  const double target = u * kept;
  double run = 0;
  for (size_t i = 0; i < keep; i++) {
    run += p[cand[i]];
    if (run > target) return cand[i];
  }
  return cand[keep - 1];
}
