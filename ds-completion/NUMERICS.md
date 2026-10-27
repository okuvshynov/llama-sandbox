# Numerics: operand types, how they map to the GPUs, and accumulation

Short version: fp8 and fp4 operands are dequantized **exactly** into fp16 (bf16 into fp32),
products are exact, and **every accumulation is fp32**. Results differ from "fp32 after
dequantization" only by summation order.

## What the hardware has (Vega II = gfx906)

- No fp8 or fp4 hardware, and no bf16 math.
- fp32 FMA: ~14 TFLOPS per die.
- Packed fp16: 2x the fp32 rate.
- `v_dot2_f32_f16`: two fp16 x fp16 products summed into an fp32 accumulator, one instruction.
- `v_dot4_i32_i8`: int8 dot products (not used).

## Why fp8/fp4 can be exact on fp16 hardware

Every e4m3 and e2m1 value is exactly representable in fp16 after a power-of-two shift, so
decoding is a few bit operations:

- An e4m3 code `<< 7`, read as fp16, is the value x 2^-8 (subnormals included).
- An e2m1 code `c` as fp16 bits `((c & 7) << 9) | ((c & 8) << 12)` is the value x 2^-14
  (the subnormal 0.5 included).

An fp16 x fp16 product has at most 22 significant bits, so it is exact in fp32. Block scales are
powers of two (e8m0 / ue8m0), applied in fp32, also exactly.

## Per operation

| operation | weights | activations | multiply | accumulate |
|---|---|---|---|---|
| dense linears (attention q / kv / out, shared expert, engram, indexer q) | fp8 e4m3, ue8m0 scale per 32x32 block | fake-quantized to e4m3, ue8m0 scale per 32 | fp16 pairs via `v_dot2_f32_f16`, exact | fp32 per 32-block, then x both scales; fp32 across blocks and lanes |
| routed experts, GPU (`hip/fp4_moe.hip`) | fp4 e2m1, e8m0 scale per 32 | e4m3, ue8m0 scale per 32 | fp16 pairs via `v_dot2_f32_f16`, exact | fp32, as above |
| routed experts, CPU (`src/cpu_moe.cpp`, AVX-512) | e2m1 decoded to fp32 by a 16-entry lookup | e4m3 values as fp32 | fp32 FMA (exact: tiny mantissas) | fp32 |
| hyper-connection mixes, router | fp32 | fp32, not quantized | fp32 FMA | fp32; split-K partials summed in a fixed order |
| compressor, indexer keys | bf16 in the file, converted to fp32 at load | fp32 | fp32 FMA | fp32 |
| indexer weights projection, output head | bf16, widened to fp32 in the kernel | fp32 | fp32 FMA | fp32 |
| attention scores, softmax, P x V | window KV: fp8-representable values; compressed KV and indexer q/k: fp4-representable (stored as fp32) | fp32 | fp32 | fp32 |
| norms, RoPE, Sinkhorn, swiglu, router scoring | - | fp32 | fp32 | fp32 |

Fake quantization (the reference's rounding points): activations before every fp8/fp4 GEMM
(e4m3, ue8m0 per 32); window KV (fp8); compressed KV (fp4 with an e4m3 scale per 16); indexer
q/k (fp4, e8m0 per 32).

## Why fp16 and not fp32 after dequantization

- **The results are the same:** the products are exact either way.
- **Fewer instructions:** `v_dot2` does two multiply-adds per instruction, and fp16 needs fewer
  bit operations to decode.
- **Speed is about memory:** the GEMVs are memory-bound, so what matters is reading 1 byte (fp8)
  or 0.53 bytes (fp4) per weight instead of 2 (bf16) or 4 (fp32), with a cheap decode.

## Against the official model

The reference prescribes the same fake-quantization points and fp8 matmuls with fp32
accumulation (fp8 tensor cores on newer GPUs). Our reference runs are the CPU in fp32. The only
remaining difference is summation order: per-layer errors are ~1e-7, apart from occasional fp8
rounding flips at near ties.

## Not used yet

int8 dot products (`v_dot4_i32_i8`, 4 MACs per instruction). fp4 values could map to small
integers, but e4m3 activations do not fit int8 exactly. Not needed while the GEMVs are
memory-bound.
