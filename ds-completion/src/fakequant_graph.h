// The reference's fp8 / fp4 "fake quantization" (quantize, then dequantize back to float), as
// ggml custom ops on the CPU backend: for the CPU reference path (src/forward.cpp) and its tests.
// The scalar rules and the host functions, which the GPU runtime uses, are in fakequant.h.
//
// The model was trained with these roundings in place, so the reference (inference/kernel.py)
// applies them at fixed points; matching them is what makes our per-layer outputs comparable
// to the reference bit for bit-ish instead of to ~1e-2:
//
//   fq_fp8     act_quant(x, 32, "ue8m0"): e4m3 values, one power-of-two scale per 32 along ne0,
//              scale = 2^ceil(log2(max(amax, 1e-4) / 448)). Used on the input of every fp8 GEMM,
//              on the window KV, and on the attention output before wo_a.
//   fq_fp4     fp4_act_quant(x, 32): e2m1 values (round half to even), power-of-two scale per 32,
//              scale = 2^ceil(log2(max(amax, 6*2^-126) / 6)). Indexer queries and keys.
//   fq_fp4_e4  fp4_act_quant(x, 16, e4m3 scale): e2m1 values, scale = e4m3(max(amax, 6*2^-9) / 6)
//              per 16. The compressed KV.
//
// All three take an f32 tensor whose ne0 is a multiple of the block and is contiguous in ne0.
#pragma once

#include "ggml.h"

ggml_tensor* fq_fp8(ggml_context* ctx, ggml_tensor* x);
ggml_tensor* fq_fp4(ggml_context* ctx, ggml_tensor* x);
ggml_tensor* fq_fp4_e4(ggml_context* ctx, ggml_tensor* x);
