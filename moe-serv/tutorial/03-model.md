# 3. The model, and its number format

## DeepSeek-V4-Flash, the shapes that matter

The checkpoint is DeepSeek-V4-Flash in a mixed-precision GGUF file
(`UD-Q8_K_XL`), 150.75 GiB total. What the expert block sees:

| quantity | value |
|---|---|
| hidden size (the vector a layer passes along) | 4096 |
| MoE layers | 43 |
| experts per layer | 256 |
| experts active per token | 6 (+ 1 shared) |
| expert FFN intermediate size | 2048 |
| expert matrices | gate 4096→2048, up 4096→2048, down 2048→4096 |
| storage format of expert weights | MXFP4 |
| routed experts, total | ~137 GiB |

Per expert that is three ~4.5 MB matrices (~13.4 MB); per layer, 256 experts
≈ 3.4 GB; times 43 layers ≈ 137 GiB. One decode token touches 6 experts per
layer ≈ 80 MB/layer ≈ 3.4 GB per token across the model.

One architectural detail the code must respect: after the gate and up
matmuls, this model **clamps** both results to ±10 before the SwiGLU
nonlinearity. A port that skips the clamp is wrong by up to 5.6% RMS on the
block output — this was once shipped for four commits because the test
configuration never exercised the code path (a war story told in
[06](06-optimization.md#test-what-you-think-youre-testing)).

## MXFP4: 4.25 bits per weight

The experts are stored in **MXFP4**, a block format:

- Weights are grouped in blocks of 32 (along the reduction dimension).
- Each block stores one **shared scale**: an 8-bit power-of-two exponent
  (format "e8m0" — exponent only, no mantissa, no sign).
- Each weight is a **4-bit float** ("e2m1"): 1 sign bit and 3 bits encoding
  one of the magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6} — stored doubled in
  ggml's tables as {0,1,2,3,4,6,8,12} with the scale halved to compensate.
- A block is therefore 17 bytes for 32 weights: **4.25 bits/weight**, an
  8x compression over float32.

Decoding one weight is: look up the 4-bit code in a 16-entry table, multiply
by the block's scale. The GPU shader does exactly this — the 16-entry table
lives in LDS where a whole wave can read one entry in a single broadcast
access
([`shaders/mxfp4_pass1.comp:46`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/shaders/mxfp4_pass1.comp#L46)).

Why a weird 4-bit *float* rather than 4-bit integer? Weights are roughly
bell-curve distributed: most are small, a few are large. A float grid spends
its codes densely near zero where the weights actually are.

The practical consequence of quantization for this project: computing on
MXFP4 directly means every weight read comes with unpacking arithmetic, and
that arithmetic — not bandwidth — is what made the stock GPU kernels slow
([02](02-hardware.md#what-instruction-bound-means-and-why-it-mattered)).

## The stub: cutting the model down to an instrument

Loading 150 GiB takes minutes and, worse, decode timings on the full model
vary several percent load-to-load, which drowns small effects. The project's
answer is
[`make_stub.py`](https://github.com/okuvshynov/llama-sandbox/blob/9ef58c9d825c58987d36aa70285a6224d9ed8c8b/moe-serv/make_stub.py):
keep only the **first 4 layers** of the GGUF. A prefix of layers is a valid
model — every piece of per-layer metadata stays correctly indexed — so no
surgery is needed beyond editing the layer count. The result is 15.78 GiB,
loads in ~14 s, and measures decode to ±0.3%.

The stub's text output is gibberish (4 layers of a 40+ layer model), which
does not matter: correctness is judged on *logits* (the raw next-token
scores) against a reference, not on whether the text reads well. Why logits
and not generated text is its own lesson
([06](06-optimization.md#compare-logits-not-text)).
