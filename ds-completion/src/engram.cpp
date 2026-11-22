#include "engram.h"

std::vector<int32_t> engram_rows(const Hparams& hp, const std::vector<int32_t>& hist, int e, int q) {
  // engram.py NgramHashState.forward: hash the 2..n-grams ending at position q
  const int n_gram = hp.engram_ngram, n_head = hp.engram_heads, n_cols = hp.engram_cols();
  const uint64_t pad = (uint64_t)hp.engram_token_map[hp.engram_pad];
  const uint64_t* mult = hp.engram_mult.data() + (size_t)e * n_gram;
  const uint64_t* primes = hp.engram_primes.data() + (size_t)e * n_cols;
  const uint64_t* offs = hp.engram_offsets.data() + (size_t)e * n_cols;
  uint64_t tok[8];
  for (int s = 0; s < n_gram; s++)  // once a look-back hits the start, every longer one is padded
    tok[s] = q - s < 0 ? pad : (uint64_t)hp.engram_token_map[hist[q - s]];
  std::vector<int32_t> ids(n_cols);
  uint64_t rolling = tok[0] * mult[0];
  for (int o = 1; o < n_gram; o++) {
    rolling ^= tok[o] * mult[o];
    for (int h = 0; h < n_head; h++) {
      const int col = (o - 1) * n_head + h;
      ids[col] = (int32_t)(rolling % primes[col] + offs[col]);
    }
  }
  return ids;
}
