// Routed experts on the GPU (PLAN.md A1 groundwork): one layer's experts resident in HBM, the
// 6 routed ones computed for one token in the reference's numerics (fp4 weights x fp8
// activations, exact products), with the expert ids read on the device (no host round trip).
//
// Layout on the device (GpuExperts): every expert's matrix as e2m1 codes [N][K/2] (16 bytes per
// 32-block: byte j = element j | element j+16 << 4, ggml's order) and e8m0 scales [N][K/32].
// Activations are fake-quantized to fp8 in "paired" order, (x_j, x_{j+16}) per 32-block, so
// each weight byte meets its two activations in one fp16 pair.
//
// Exactness: an e2m1 code c becomes the fp16 bit pattern ((c & 7) << 9) | ((c & 8) << 12),
// whose value is the e2m1 value x 2^-14 (subnormal 0.5 included); fp8 codes become fp16 as in
// fp8_gemv (x 2^-8). v_dot2_f32_f16 multiplies exactly; the block partial is scaled by
// 2^(e_w - 127) x activation scale x 2^22.
#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

struct GpuMatrix {          // one expert tensor of a layer, all experts
  uint8_t* codes;           // [n_expert][N][K / 2]
  uint8_t* scales;          // [n_expert][N][K / 32]
  int N, K;
};

struct GpuExperts { GpuMatrix gate, up, down; };

// host -> device: repack a layer's ggml mxfp4 tensor ([K, N, n_expert]) into a GpuMatrix
// (allocates). `src` is the host (mmap) data.
GpuMatrix fp4_upload(const void* src, int K, int N, int n_expert, hipStream_t s);
void fp4_free(GpuMatrix& m);

// y = sum_j down_j(fq8(w_j * swiglu_clamp(gate_j(fq8 x), up_j(fq8 x)))), k routed experts with
// ids/w on the device. scratch: k * (3 * F + 2 * F / 32 + E) floats + (E + k * F) bytes, from
// fp4_moe_scratch_bytes.
size_t fp4_moe_scratch_bytes(int E, int F, int k);
void fp4_moe(const GpuExperts& ex, const float* x, const int* ids, const float* w, int k, float limit, float* y,
             void* scratch, hipStream_t s);
