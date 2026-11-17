// Choosing the next token from the logits: greedy, or sampled with a temperature and top-p
// (README step 148). The model card's settings are temperature 1.0 with top-p 0.95 or 1.0.
//
// The random number of a token is a function of the seed and the token's position in the
// context, not the next value of a running generator. So the choice at a position does not
// depend on how the positions before it were computed: one seed gives one output with and
// without speculative decoding, whatever the drafts' lengths.
#pragma once

#include <cstdint>
#include <vector>

struct SamplerOptions {
  float temp = 0;     // 0: greedy (the first of equal maxima, as std::max_element)
  float top_p = 1;    // keep the most probable tokens up to this total probability; 1: all
  uint64_t seed = 0;
};

// uniform in [0, 1): 53 bits from a hash of (seed, position)
double sampler_uniform(uint64_t seed, uint64_t position);

// The token for `position` from logits[0..n). With temp > 0: p = softmax(logits / temp); with
// top_p < 1 the tokens are taken by falling p (equal p: by rising id) until their sum reaches
// top_p of the total, and the choice is among those, in that order; with top_p = 1 among all,
// by rising id. Everything is computed in double, in a fixed order.
//
// On AVX-512 with glibc the exp is libmvec's 8-wide one (README step 151), whose values can
// differ from the scalar exp's in the last bits. A build without it uses the scalar exp and
// could take another token where the random number falls that close to a boundary; no such
// case was seen (tests/test_sampler.cpp, the stored outputs of steps 149 and 150).
// `scratch`: working memory to reuse between calls (one per thread); the result does not depend
// on it. Without it every call allocates and frees 8 * n bytes (README step 151).
struct SamplerScratch {
  std::vector<double> p;
  std::vector<int32_t> cand;
};
int32_t sample_token(const float* logits, int n, const SamplerOptions& opt, uint64_t position, SamplerScratch* scratch = nullptr);
