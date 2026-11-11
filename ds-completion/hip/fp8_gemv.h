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

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

// Row stride of a weight matrix's device copy (README step 101). Rows a multiple of 4096 bytes
// apart start at the same offset in every page and collide in the GPU's caches (the attention
// output matrices: K = 4096 and 8192, read at 580 and 490 GB/s; 64 more bytes per row: 740 and
// 730). The kernels take the stride (w_stride; 0 = K); the bytes between rows are never read.
// DS_FP8_PAD=0 turns it off (rows K apart, as in the file).
int fp8_row_stride(int K);

// A launch of 2017-2048 waves (the GPU's 2048 wave slots filled exactly) takes ~48 us whatever it
// computes: an empty kernel too, against 3-5 us at 2016 or 2052 waves (README step 102). A light
// kernel with such a grid goes out as two launches instead: blocks [0, b1) and [b1, nx) of its x.
// Returns b1 (nx: one launch). waves_per_block = threads / 64; DS_STALL_SPLIT=0 turns it off.
inline int stall_split(int nx, int ny, int waves_per_block) {
  static const bool on = !getenv("DS_STALL_SPLIT") || atoi(getenv("DS_STALL_SPLIT"));
  const int waves = nx * ny * waves_per_block;
  return on && waves > 2016 && waves <= 2048 && nx >= 2 ? nx * 3 / 8 : nx;
}

struct Fp8Act {      // one activation matrix in the kernel's input format
  const uint8_t* q;  // [M][K_total] e4m3 codes
  const float* s;    // [M][K_total / 32] per-block scales (powers of two)
};

// x: [M][K_total] f32 (K_total = K * groups) -> fp8 codes + scales (the reference's act_quant)
void fp8_quant_act(const float* x, uint8_t* q, float* s, int M, int K_total, hipStream_t stream);

// y: [M][N] f32. lpr (lanes per row: 8/16/32/64) and r (rows per wave: 1/2/4/8) pick the work
// split explicitly; 0 = the built-in choice.
void fp8_gemv_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int lpr, int r, int w_stride = 0);
void fp8_gemv(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
              float* y, hipStream_t stream, int w_stride = 0);
// the lanes-per-row split fp8_gemv uses for a shape (it fixes every output's order of additions)
int fp8_lpr(int N, int K);
// Any number of inputs (prompt chunks): tiles of mt inputs, r rows per lane group; every result
// bit-identical to fp8_gemv on that input alone (same split, same per-lane order, same tree).
// r / mt = 0: the built-in choice; minb: blocks per CU the registers must allow (0 = 1).
void fp8_gemm_cfg(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
                  float* y, hipStream_t stream, int r, int mt, int minb = 0, int w_stride = 0);
void fp8_gemm(const uint8_t* w, const uint8_t* w_scale, int N, int K, int rows_per_group, Fp8Act x, int M,
              float* y, hipStream_t stream, int w_stride = 0);
