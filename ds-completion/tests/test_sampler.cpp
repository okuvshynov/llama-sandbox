// test-sampler: src/sampler.h on given logits (README step 148). No model, no GPU.
//
//   test-sampler            exit code 1 if a check fails
//
// The draws are a function of (seed, position), so every number printed is the same on every
// run: a chi-square over its limit is a failure of the code, not bad luck of this run.
#include "sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int failed = 0;
static void check(bool ok, const char* what) {
  printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
  fflush(stdout);
  if (!ok) failed++;
}

// chi-square of counts against probabilities (which sum to 1) over `draws`
static double chi2(const std::vector<long>& count, const std::vector<double>& prob, long draws) {
  double x = 0;
  for (size_t i = 0; i < count.size(); i++) {
    const double e = prob[i] * draws;
    if (e > 0) x += (count[i] - e) * (count[i] - e) / e;
  }
  return x;
}

static std::vector<double> softmax(const std::vector<float>& l, double temp) {
  std::vector<double> p(l.size());
  double s = 0;
  for (size_t i = 0; i < l.size(); i++) s += p[i] = std::exp(l[i] / temp);
  for (double& v : p) v /= s;
  return p;
}

// top-p the plain way: the whole vocabulary sorted. The order depends on the logits and the
// temperature only, so it is made once for them (Sorted) and used for every top-p and position:
// sorting 129,280 values for each of 4,800 draws was 79 of this test's 80 seconds (README step 186).
struct Sorted {
  std::vector<double> p;
  std::vector<int32_t> id;
  double total = 0;
};
static Sorted sorted(const std::vector<float>& l, double temp) {
  const int n = l.size();
  const double top = *std::max_element(l.begin(), l.end());
  Sorted s;
  s.p.resize(n);
  for (int i = 0; i < n; i++) s.total += s.p[i] = std::exp(((double)l[i] - top) / temp);
  s.id.resize(n);
  for (int i = 0; i < n; i++) s.id[i] = i;
  std::sort(s.id.begin(), s.id.end(), [&](int32_t a, int32_t b) { return s.p[a] != s.p[b] ? s.p[a] > s.p[b] : a < b; });
  return s;
}
static int32_t reference(const Sorted& s, const SamplerOptions& o, uint64_t pos) {
  const int n = s.p.size();
  const std::vector<double>& p = s.p;
  const std::vector<int32_t>& id = s.id;
  const double total = s.total;
  int keep = 0;
  double kept = 0;
  while (keep < n) {
    kept += p[id[keep++]];
    if (kept >= (double)o.top_p * total) break;
  }
  const double target = sampler_uniform(o.seed, pos) * kept;
  double run = 0;
  for (int i = 0; i < keep; i++) {
    run += p[id[i]];
    if (run > target) return id[i];
  }
  return id[keep - 1];
}

int main() {
  char what[256];
  // chi-square limits at p = 0.001: 7 degrees of freedom 24.32, 15: 37.70, 2: 13.82
  const double lim7 = 24.32, lim15 = 37.70, lim2 = 13.82;

  {  // greedy
    const std::vector<float> l = {0.5f, 3.0f, -1.0f, 3.0f, 2.0f};
    SamplerOptions o;
    check(sample_token(l.data(), l.size(), o, 0) == 1 && sample_token(l.data(), l.size(), o, 12345) == 1,
          "temperature 0: the first of the equal maxima, at any position");
  }
  {  // the uniform numbers
    const long N = 1000000;
    for (int pass = 0; pass < 2; pass++) {  // over positions for one seed; over seeds for one position
      std::vector<long> c(16);
      double sum = 0;
      bool in = true;
      for (long i = 0; i < N; i++) {
        const double u = pass == 0 ? sampler_uniform(7, i) : sampler_uniform(i, 7);
        in = in && u >= 0 && u < 1;
        sum += u;
        c[(int)(u * 16)]++;
      }
      const double x = chi2(c, std::vector<double>(16, 1.0 / 16), N);
      snprintf(what, sizeof what, "uniform over %s: in [0, 1), mean %.5f, chi-square of 16 bins %.2f (limit %.2f)",
               pass == 0 ? "positions" : "seeds", sum / N, x, lim15);
      check(in && std::fabs(sum / N - 0.5) < 0.002 && x < lim15, what);
    }
    check(sampler_uniform(1, 2) == sampler_uniform(1, 2) && sampler_uniform(1, 2) != sampler_uniform(2, 1) &&
              sampler_uniform(0, 0) != sampler_uniform(0, 1) && sampler_uniform(0, 0) != sampler_uniform(1, 0),
          "uniform: a function of (seed, position), not symmetric in them");
  }
  const std::vector<float> l8 = {2.0f, 1.5f, 1.0f, 0.0f, -0.5f, 0.7f, -2.0f, 1.9f};
  {  // the distribution at three temperatures and three seeds
    const long N = 400000;
    for (float temp : {0.5f, 1.0f, 2.0f})
      for (uint64_t seed : {0ull, 1ull, 123456789ull}) {
        SamplerOptions o;
        o.temp = temp;
        o.seed = seed;
        std::vector<long> c(8);
        for (long i = 0; i < N; i++) c[sample_token(l8.data(), 8, o, i)]++;
        const double x = chi2(c, softmax(l8, temp), N);
        snprintf(what, sizeof what, "8 logits, temperature %.1f, seed %llu: chi-square against softmax %.2f (limit %.2f)", temp,
                 (unsigned long long)seed, x, lim7);
        check(x < lim7, what);
      }
  }
  {  // top-p on known probabilities
    const std::vector<double> p = {0.04, 0.5, 0.01, 0.15, 0.3};  // by falling p: ids 1, 4, 3, 0, 2
    std::vector<float> l;
    for (double v : p) l.push_back((float)std::log(v));
    const long N = 400000;
    SamplerOptions o;
    o.temp = 1;
    o.top_p = 0.9f;  // 0.5 + 0.3 = 0.8 < 0.9, + 0.15 = 0.95: three kept
    o.seed = 5;
    std::vector<long> c(5);
    for (long i = 0; i < N; i++) c[sample_token(l.data(), 5, o, i)]++;
    const double x = chi2({c[1], c[4], c[3]}, {0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95}, N);
    snprintf(what, sizeof what, "top-p 0.9 of {0.5, 0.3, 0.15, 0.04, 0.01}: the two smallest never (%ld, %ld), the rest renormalized, chi-square %.2f (limit %.2f)",
             c[0], c[2], x, lim2);
    check(c[0] == 0 && c[2] == 0 && x < lim2, what);
    o.top_p = 0.3f;  // the first alone reaches it
    bool one = true;
    for (long i = 0; i < 10000; i++) one = one && sample_token(l.data(), 5, o, i) == 1;
    check(one, "top-p 0.3 of the same: always the most probable token");
  }
  {  // one seed, one answer; another seed, another sequence
    SamplerOptions a, b;
    a.temp = b.temp = 1;
    a.seed = 11;
    b.seed = 12;
    int same = 0, differ = 0;
    for (long i = 0; i < 1000; i++) {
      same += sample_token(l8.data(), 8, a, i) == sample_token(l8.data(), 8, a, i);
      differ += sample_token(l8.data(), 8, a, i) != sample_token(l8.data(), 8, b, i);
    }
    snprintf(what, sizeof what, "1000 positions: the same token for the same seed (%d), another for another seed at %d of them", same, differ);
    check(same == 1000 && differ > 500, what);
  }
  // a vocabulary of the model's size, two kinds of logits. "flat": the wide tail holds most of
  // the probability, so top-p keeps tens of thousands of tokens (the sort's worst case, not what
  // a model gives). "peaked": the same with the likely tokens 10 higher, so a few tokens hold it.
  const int V = 129280;
  std::vector<std::vector<float>> flat, peaked;
  {
    std::mt19937 gen(1);
    std::normal_distribution<float> tail(0.0f, 2.0f);
    for (int k = 0; k < 8; k++) {
      std::vector<float> l(V);
      for (float& v : l) v = tail(gen);
      std::vector<float> lp = l;
      for (int j = 0; j < 4 + 3 * k; j++) {
        const int id = gen() % V;
        l[id] = 8.0f + 6.0f * (float)j / (4 + 3 * k);
        lp[id] = l[id] + 10.0f;
      }
      if (k == 7)
        for (int j = 0; j < 1000; j++) l[gen() % V] = 9.0f;  // many equal values: ties in the order
      flat.push_back(l);
      peaked.push_back(lp);
    }
  }
  std::vector<std::vector<float>> big = flat;
  big.insert(big.end(), peaked.begin(), peaked.end());
  {  // top-p's shortcut (only the tokens above a floor are sorted) against the whole vocabulary sorted
    long n = 0, same = 0;
    for (const auto& l : big)
      for (float temp : {0.6f, 1.0f, 1.5f}) {
        const Sorted srt = sorted(l, temp);
        for (float top_p : {0.5f, 0.9f, 0.95f, 0.999f}) {
          SamplerOptions o;
          o.temp = temp;
          o.top_p = top_p;
          o.seed = 3;
          for (uint64_t pos = 0; pos < 25; pos++, n++) same += sample_token(l.data(), V, o, pos) == reference(srt, o, pos);
        }
      }
    snprintf(what, sizeof what, "%d logits, top-p: the same token as with the whole vocabulary sorted in %ld of %ld draws", V, same, n);
    check(same == n, what);
  }
  {  // what a draw costs
    for (int kind = 0; kind < 2; kind++)
      for (float top_p : {1.0f, 0.95f}) {
        const auto& set = kind == 0 ? peaked : flat;
        SamplerOptions o;
        o.temp = 1;
        o.top_p = top_p;
        long sink = 0;
        const int reps = 200;
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) sink += sample_token(set[r % set.size()].data(), V, o, r);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("time  %d %s logits, temperature 1, top-p %.2f: %.3f ms per token (one thread; checksum %ld)\n", V,
               kind == 0 ? "peaked" : "flat", top_p, ms, sink);
      }
  }
  printf("%s\n", failed ? "FAILED" : "all checks passed");
  return failed ? 1 : 0;
}
