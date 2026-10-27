"""Pure-torch CPU stand-ins for the reference's tilelang kernels (inference/kernel.py).

Same function names, signatures and semantics, so model.py runs unmodified on top of them
(ds_ref.py installs this module as `kernel` before importing model.py). Each function follows
its tilelang original line by line; the known deviations are listed where they occur and are
all below bf16 resolution:
  - GEMMs dequantize to fp32 and use one fp32 matmul, instead of fp8 MMA per 32-wide K block
    with fp32 accumulation. Exact products either way (fp8 x fp8 fits fp32); only the order
    of the fp32 additions differs.
  - sparse_attn keeps the probabilities in fp32 for P @ V (the kernel casts P to bf16 first).
"""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 magnitudes, ordered even-mantissa codes first so that argmin's first-hit rule
# implements round-half-to-even (the hardware cast's rounding)
_FP4_GRID = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0])


def _pow2_ceil_log2(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(v)) for positive normal fp32 v (fast_round_scale's bit trick)."""
    m, e = torch.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(v), e - (m == 0.5).to(e.dtype))


def _e8m0_decode(s: torch.Tensor) -> torch.Tensor:
    if s.dtype == torch.float32:
        return s
    return torch.exp2(s.view(torch.uint8).float() - 127)


def _e8m0_encode(s: torch.Tensor) -> torch.Tensor:
    """fp32 power of two -> e8m0 byte (as torch.float8_e8m0fnu)."""
    _, e = torch.frexp(s)
    return (e - 1 + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value (ties to even), input already clamped to [-6, 6]."""
    a = v.abs()
    idx = (a.unsqueeze(-1) - _FP4_GRID).abs().argmin(dim=-1)
    return torch.copysign(_FP4_GRID[idx], v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise fp8 (e4m3) quantization along the last dim. inplace: quantize-dequantize into x."""
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    s = amax * (1 / FP8_MAX)
    if scale_fmt is not None:
        s = _pow2_ceil_log2(s)
    q = (xf / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s).flatten(-2).to(x.dtype))
        return x
    s = s.squeeze(-1)
    return q.flatten(-2), (_e8m0_encode(s) if scale_dtype == torch.float8_e8m0fnu else s)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise fp4 (e2m1) quantization, e8m0 (indexer) or e4m3 (compressed KV) scales.
    Only the inplace (quantize-dequantize) form is used by model.py."""
    assert inplace, "only fake-quant is used by model.py"
    N = x.size(-1)
    assert N % block_size == 0
    xf = x.float().unflatten(-1, (N // block_size, block_size))
    amax = xf.abs().amax(dim=-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2**-9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2**-126)
        s = _pow2_ceil_log2(amax * (1 / FP4_MAX))
    q = _round_fp4((xf / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def _dequant_weight_fp8(b: torch.Tensor, b_s: torch.Tensor, block: int) -> torch.Tensor:
    # cached on the Parameter: decode steps reuse every dense weight
    w = getattr(b, "_f32", None)
    if w is None:
        n, k = b.shape
        s = _e8m0_decode(b_s).repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
        w = b.float() * s
        b._f32 = w
    return w


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M, N] = A[M, K] @ B[N, K]^T; A per-row per-K-block scales, B per (N, K) block scales."""
    K = a.size(-1)
    af = a.float().unflatten(-1, (K // block_size, block_size)) * _e8m0_decode(a_s).unsqueeze(-1)
    w = _dequant_weight_fp8(b, b_s, block_size)
    return (af.flatten(-2) @ w.t()).to(torch.get_default_dtype())


_FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T; B packed two codes per byte along K (element 2i low),
    one e8m0 scale per 32. Routed experts: dequantized per call, not cached (too many)."""
    K = a.size(-1)
    af = a.float().unflatten(-1, (K // act_block_size, act_block_size)) * _e8m0_decode(a_s).unsqueeze(-1)
    u = b.view(torch.uint8)
    w = torch.stack([_FP4_TABLE[(u & 0xF).long()], _FP4_TABLE[(u >> 4).long()]], dim=-1).flatten(-2)
    w = (w.unflatten(-1, (-1, 32)) * _e8m0_decode(b_s).unsqueeze(-1)).flatten(-2)
    return (af.flatten(-2) @ w.t()).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """o[b, m, h] = softmax over {kv[b, idx] : idx in topk_idxs[b, m], idx != -1} plus a per-head
    sink logit that only adds to the denominator. A row with no valid index gives zeros."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    g = kv.float()[torch.arange(b).view(b, 1, 1), idx.clamp_min(0)]  # [b, m, k, d]
    s = torch.einsum("bmhd,bmkd->bmhk", q.float(), g) * softmax_scale
    s = s.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, m, h, 1)
    p = torch.cat([s, sink], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bmhk,bmkd->bmhd", p, g).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split the hyper-connection mixes into pre [hc], post [hc] and a Sinkhorn-normalized comb [hc, hc]."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
