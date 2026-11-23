// Which rows of the Engram table a position reads (README step 171; the hash was in forward.cpp
// until then). The GPU runtime (hip/rt.cpp) uses it; so did the CPU reference path until README
// step 194. The rows' bytes are decoded by mxfp8_to_float (model.h).
#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

// The engram table rows (absolute, n_cols of them) for the token at position q of `hist`
// (all tokens so far, including q), for engram slot e.
std::vector<int32_t> engram_rows(const Hparams& hp, const std::vector<int32_t>& hist, int e, int q);
