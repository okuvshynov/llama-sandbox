// The options the program takes for the model itself (README step 173). apps/ds_completion.cpp and
// tests/ds_test.cpp fill one; the GPU runtime copies what it uses into RtOptions. (Until README
// step 194 the CPU reference path's Engine took it too, with a switch for the rounding.)
#pragma once

struct Options {
  int max_ctx = 4096;  // capacity of the compressed caches (both programs set 65536 unless --max-ctx is given)
  int n_threads = 16;
};
