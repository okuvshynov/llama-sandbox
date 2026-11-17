#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__AVX512F__) && defined(__GLIBC__)
#include <immintrin.h>
extern "C" __m512d _ZGVeN8v_exp(__m512d);  // glibc's libmvec: exp of 8 doubles
#define SAMPLER_VECTOR_EXP 1
#endif

static uint64_t mix64(uint64_t z) {  // splitmix64's finalizer
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

double sampler_uniform(uint64_t seed, uint64_t position) {
  const uint64_t h = mix64(mix64(seed + 0x9E3779B97F4A7C15ull) ^ (position * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull));
  return (double)(h >> 11) * 0x1.0p-53;
}

// the first maximum, as std::max_element (which took 0.2 ms of a draw: it is not vectorized)
static int32_t first_max(const float* l, int n) {
#ifdef SAMPLER_VECTOR_EXP
  if (n >= 16) {
    __m512 vm = _mm512_loadu_ps(l);
    int i = 16;
    for (; i + 16 <= n; i += 16) vm = _mm512_max_ps(vm, _mm512_loadu_ps(l + i));
    float m = _mm512_reduce_max_ps(vm);
    for (; i < n; i++) m = std::max(m, l[i]);
    vm = _mm512_set1_ps(m);
    for (i = 0; i + 16 <= n; i += 16) {
      const __mmask16 eq = _mm512_cmpeq_ps_mask(_mm512_loadu_ps(l + i), vm);
      if (eq) return i + __builtin_ctz(eq);
    }
    for (; i < n; i++)
      if (l[i] == m) return i;
  }
#endif
  return (int32_t)(std::max_element(l, l + n) - l);
}

int32_t sample_token(const float* logits, int n, const SamplerOptions& opt, uint64_t position, SamplerScratch* scratch) {
  const int32_t best = first_max(logits, n);
  if (opt.temp <= 0) return best;
  const double inv_t = 1.0 / opt.temp, top = logits[best];
  SamplerScratch own;
  if (!scratch) scratch = &own;
  std::vector<double>& p = scratch->p;
  std::vector<int32_t>& cand = scratch->cand;
  p.resize(n);
  int i0 = 0;
#ifdef SAMPLER_VECTOR_EXP
  // 129,280 scalar exp calls were ~0.9 ms of a draw's 1.0 ms (README step 151)
  const __m512d vtop = _mm512_set1_pd(top), vinv = _mm512_set1_pd(inv_t);
  for (; i0 + 8 <= n; i0 += 8) {
    const __m512d x = _mm512_mul_pd(_mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(logits + i0)), vtop), vinv);
    _mm512_storeu_pd(p.data() + i0, _ZGVeN8v_exp(x));
  }
  p[best] = 1;  // exactly, whatever the vector exp gives for 0
#endif
  for (int i = i0; i < n; i++) p[i] = std::exp(((double)logits[i] - top) * inv_t);  // <= 1; the maximum is exactly 1
  double total = 0;
  for (int i = 0; i < n; i++) total += p[i];  // in the order of the ids
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
  cand.clear();
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
