// Placement-invariant expert numerics (PLAN.md "Adaptive expert cache", phase 0, option A): one
// specification of an mxfp4 weight row times fp8 activations that the CPU and the GPU kernels
// must both reproduce bit for bit, so that an expert gives the same bits wherever it runs.
//
// Row of K = 32 nb inputs; block b: weight e2m1 values w_i with scale 2^(e_b - 127), activation
// e4m3 values x_i with scale s_b (act_quant: a power of two).
//
//   S_b   = sum_i (2 w_i) (2^9 x_i)                exact, in integers: 2 w in -12..12,
//                                                  2^9 x in -229376..229376, |S_b| < 2^27
//   v_b   = float(S_b) * (2^(e_b - 127) * c_b)     c_b = s_b * 2^-10; float() rounds to nearest
//                                                  even; the scale product is a power of two, so
//                                                  the multiplications are exact
//   acc_l = ((0 + v_l) + v_{l+16}) + v_{l+32} ...  l = 0..15, blocks in increasing order
//   for o = 8, 4, 2, 1: acc_l <- acc_l + acc_{l^o} (all l at once); the result is acc_0
//
// The block sum is exact (order-free), so each side may use its fastest integer instructions;
// only the conversion, the 16 accumulators and the tree are fixed. That is the GPU's lane
// pattern with 16 lanes per row (lane l takes blocks l, l + 16, ...; then xor shuffles).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// The activations of one GEMV, fake-quantized (act_quant(x, 32, ue8m0)) and split for the
// integer block sums.
struct CanonAct {
  int nb = 0;
  // 2^9 x = d0 + d1 2^7 + d2 2^14, every digit with the sign of x and |d| <= 127; [nb * 32] each,
  // natural element order within a block
  std::vector<int8_t> d0, d1, d2;
  std::vector<int32_t> corr;  // per block: 12 * sum of 2^9 x (the weights enter as 2 w + 12 >= 0)
  std::vector<float> c;       // per block: s_b * 2^-10
  std::vector<int32_t> x9;    // [nb * 32]: 2^9 x (for the scalar specification)
};

// x [n] (n a multiple of 32, not yet quantized) -> a
void canon_act(const float* x, int n, CanonAct& a);
// the same in parts (for threads): size a for n values, then fill blocks [b0, b1) from x (the
// whole vector's pointer)
void canon_act_size(int n, CanonAct& a);
void canon_act_blocks(const float* x, int b0, int b1, CanonAct& a);

// One row in ggml block_mxfp4 layout (nb blocks of 17 bytes: e8m0 scale, 16 code bytes; code
// byte j holds element j in its low nibble, j + 16 in its high one). nb even.
float canon_dot_ref(const uint8_t* row, const CanonAct& a);  // scalar: the specification
// rows r0 .. r1 - 1 (row stride row_bytes) -> y[r0 .. r1 - 1]; AVX-512 VNNI
void canon_dot_rows(const uint8_t* rows, size_t row_bytes, const CanonAct& a, int r0, int r1, float* y);
