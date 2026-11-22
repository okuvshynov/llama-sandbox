// A layer's named intermediates, copied out for a test (README step 171).
//
// Both implementations of the model can run one block on given inputs and return what the
// reference records under the same names (h_in, attn_out, ffn_out, router_w, ...):
// Engine::run_layer (src/forward.h, the CPU reference path) and Runtime::run_layer (hip/rt.h, the
// GPU runtime). tests/ds_test.cpp compares either with the golden data. When a program serves,
// nothing asks for taps: the runtime's layer code gets a null Taps* and copies nothing.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct Tap {
  std::vector<int64_t> ne;  // ggml order
  std::vector<float> f;     // f32 values, or
  std::vector<int32_t> i;   // i32 values
};
using Taps = std::map<std::string, Tap>;
