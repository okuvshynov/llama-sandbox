// The options both implementations of the model take from the command line (README step 173;
// in forward.h until then). apps/ds_completion.cpp fills one for ds-completion (the GPU runtime
// copies what it uses into RtOptions) and for ds-ref (the CPU reference path's Engine takes it).
#pragma once

struct Options {
  bool exact_quant = true;  // apply the reference's fp8/fp4 fake quantization (fakequant.h); off: ds-ref only
  int max_ctx = 4096;       // capacity of the compressed caches
  int n_threads = 16;
};
