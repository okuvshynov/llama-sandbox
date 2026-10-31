// fp8 x fp8 GEMV/skinny-GEMM for gfx906 (PLAN.md B1): y[n, m] = W[n, :] . x[:, m] for M <= 8.
//
// Weights: e4m3 codes, row-major [N][K], one e8m0 scale per 32x32 block ([N/32][K/32]),
// exactly as written by fp8pack. Activations: fake-quantized to fp8 like the reference's
// act_quant (e4m3 values with one power-of-two scale per 32 along K), stored as e4m3 codes
// plus one f32 scale per 32.
//
// Exactness: gfx906 has no fp8, but an e4m3 code shifted left by 7 bits *is* an fp16 bit
// pattern, equal to the e4m3 value times 2^-8 (normals and subnormals alike). Both operands
// are decoded that way; v_dot2_f32_f16 multiplies fp16 pairs exactly into fp32, so every
// product of a 32-block is exact, and the block partial is scaled by (weight scale x
// activation scale x 2^16) in fp32.
//
// Work split: a wave holds R rows (all inside one 32-row scale block) and LPR lanes per row
// segment; every activation load is shared by the R rows, every weight load by the M columns.
// That is the reference's fp8_gemm up to the order of the fp32 additions.
//
// Grouped mode (attn_output_a): rows [g * rows_per_group, (g + 1) * rows_per_group) read
// activation slice g (x + g * K). rows_per_group = N for a plain GEMV.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

struct Fp8Act {      // one activation matrix in the kernel's input format
  const uint8_t* q;  // [M][K_total] e4m3 codes
  const float* s;    // [M][K_total / 32] per-block scales (powers of two)
};

// x: [M][K_total] f32 (K_total = K * groups) -> fp8 codes + scales (the reference's act_quant)
void fp8_quant_act(const float* x, uint8_t* q, float* s, int M, int K_total, hipStream_t stream);

// y: [M][N] f32. lpr (lanes per row: 8/16/32/64) and r (rows per wave: 1/2/4/8) pick the work
// split explicitly; 0 = the built-in choice.
void fp8_gemv_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int lpr, int r);
void fp8_gemv(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
              float* y, hipStream_t stream);
// the lanes-per-row split fp8_gemv uses for a shape (it fixes every output's order of additions)
int fp8_lpr(int N, int K);
// Any number of inputs (prompt chunks): tiles of mt inputs, r rows per lane group; every result
// bit-identical to fp8_gemv on that input alone (same split, same per-lane order, same tree).
// r / mt = 0: the built-in choice; minb: blocks per CU the registers must allow (0 = 1).
void fp8_gemm_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int r, int mt, int minb = 0);
void fp8_gemm(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
              float* y, hipStream_t stream);
