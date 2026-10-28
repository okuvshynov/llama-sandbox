#!/usr/bin/env python3
"""The official DeepSeek-V4.1 reference (inference/model.py, unmodified) on the CPU, weights from the GGUF.

    ~/venvs/ds-ref/bin/python ref/ds_ref.py --prompt "Hello, how are you?" --chat --n-decode 8 \
        --out data/ref/hello.pt

Runs exactly the reference's generate() loop: one prefill forward over the prompt, then one
forward per decoded token (greedy). Every forward records, per layer:

    h_in, pre_in        hyper-connection stream [s, 4, 5120] and pre-mix [s, 4] entering the block
                        (after the engram, for engram layers)
    engram_in           stream before the engram (engram layers only)
    attn_in, attn_out   attention sublayer input (after hc_pre + norm) and output (after wo_b)
    ffn_in, ffn_out     MoE input / output (routed + shared)
    router_w, router_i  routing weights and expert ids [s, 6]
    topk                indexer result (index-source layers; compressed positions, -1 = none)
    h_out, pre_out      block output
plus the step's tokens, start_pos and final logits [vocab] (last position).

--dspark DIR: also the DSpark draft head (model.py's forward_spec, unmodified) after every forward,
with its weights from the DSpark GGUF in DIR. The prefill forward seeds its caches; every decode
step records the draft: "draft_ids" [6] (the anchor = this step's token, then 5 drafts),
"draft_logits" [5, vocab] (Markov bias included) and "draft_conf" [5].

What is swapped out under model.py, and why:
  - kernel.py (tilelang, CUDA only) -> kernel_cpu.py, same semantics in plain torch.
  - Weights come from the GGUF (bit-exact with the 1017 checkpoint, see check_weights.py).
    The fp8 linears are re-blocked from the GGUF's exact bf16 into the reference's own format
    (e4m3 + 32x32 e8m0 scales), the experts are repacked from mxfp4 into its fp4 format. Both are
    exact (asserted). A different (but exact) fp8 factorization does not change any GEMM result:
    the scales are powers of two.
  - Routed experts are materialized from the GGUF only when routed to, and dropped after the
    layer (all 15360 would be 269 GB). The engram table lookup reads rows straight from the GGUF.
  - image_processor (needs PIL) is replaced by its five constants: text only.
"""
import argparse
import functools
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

REF = Path(__file__).resolve().parent
sys.path.insert(0, str(REF))
import kernel_cpu  # noqa: E402
from weights import GGUF, HF_DIR, LAYER_MAP  # noqa: E402

print = functools.partial(print, flush=True)

# ------------------------------------------------------------------ import the reference
sys.modules["kernel"] = kernel_cpu
_img = types.ModuleType("image_processor")
_img.TEXT = -1
_img.IMAGE_START, _img.IMAGE, _img.IMAGE_NEW_LINE, _img.IMAGE_END = range(4)
sys.modules["image_processor"] = _img
sys.path.append(str(HF_DIR / "inference"))
sys.path.append(str(HF_DIR / "encoding"))
import model as ref  # noqa: E402  the reference model.py itself


# ------------------------------------------------------------------ patches (construction only)
_linear_init = ref.Linear.__init__


def _linear_init_meta_fp4(self, in_features, out_features, bias=False, dtype=None):
    """Routed experts are created on the meta device: real tensors appear only when routed to."""
    if (dtype or ref.default_dtype) == torch.float4_e2m1fn_x2:
        with torch.device("meta"):
            return _linear_init(self, in_features, out_features, bias, dtype)
    return _linear_init(self, in_features, out_features, bias, dtype)


ref.Linear.__init__ = _linear_init_meta_fp4


class GGUFEngramEmbedding(nn.Module):
    """ParallelEngramEmbedding with rows read from the GGUF table (mxfp8 = the checkpoint's fp8 rows).
    Same output as the reference: e4m3 * 2^(e8m0 - 127) in fp32, then bf16."""

    def __init__(self, G: GGUF, row_base: int, n_rows: int, dim: int):
        super().__init__()
        self.base, self.n_rows, self.dim = row_base, n_rows, dim
        raw = G.raw("per_layer_token_embd.weight")
        self.table = raw.reshape(-1, dim // 32 * 33)  # [rows, 8 blocks x (e + 32 codes)]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.flatten().numpy()
        assert idx.min() >= 0 and idx.max() < self.n_rows
        blk = torch.from_numpy(np.array(self.table[self.base + idx])).view(-1, self.dim // 32, 33)
        e = blk[..., 0].float()
        v = blk[..., 1:].contiguous().view(torch.float8_e4m3fn).float()
        out = (v * torch.exp2(e - 127).unsqueeze(-1)).flatten(-2).to(torch.get_default_dtype())
        return out.view(*indices.shape, self.dim)


# ------------------------------------------------------------------ weight conversion

def fp8_blocks(w: torch.Tensor, block: int = 32):
    """Exact bf16/fp32 weight -> (e4m3 [n, k], e8m0 scales [n/32, k/32]); asserts the round trip."""
    n, k = w.shape
    wb = w.view(n // block, block, k // block, block)
    amax = wb.abs().amax(dim=(1, 3), keepdim=True)
    s = kernel_cpu._pow2_ceil_log2((amax / 448).clamp_min(2.0**-126))
    q = (wb / s).to(torch.float8_e4m3fn)
    assert torch.equal(q.float() * s, wb), "weight is not exactly fp8 x 2^k in 32x32 blocks"
    return q.view(n, k), kernel_cpu._e8m0_encode(s.view(n // block, k // block))


def mxfp4_to_ref(G: GGUF, name: str, expert: int):
    """One expert of a GGUF mxfp4 [n_exp, rows, cols] tensor -> reference (int8 packed [rows, cols/2], e8m0 [rows, cols/32]).
    ggml block: e, qs[16] with qs[j] = element j | element j+16 << 4. Reference: byte i = element 2i | 2i+1 << 4."""
    cols, rows = G.ne(name)[0], G.ne(name)[1]
    nb = cols // 32
    raw = G.raw(name)
    per = rows * nb * 17
    blk = torch.from_numpy(np.array(raw[expert * per:(expert + 1) * per])).view(rows, nb, 17)
    scale = blk[..., 0].contiguous()
    qs = blk[..., 1:]
    codes = torch.cat([qs & 0xF, qs >> 4], dim=-1)  # [rows, nb, 32] in element order
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(rows, cols // 2)
    return packed.view(torch.float4_e2m1fn_x2), scale.view(torch.float8_e8m0fnu)


GLOBAL = {"embed.weight": "token_embd.weight", "head.weight": "output.weight", "norm.weight": "output_norm.weight"}
REV = {v: k for k, v in LAYER_MAP.items()}  # HF suffix (or linear prefix) -> GGUF suffix


# DSpark head (mtp.*) tensors outside the per-layer map, in the DSpark GGUF
DSPARK_GLOBAL = {"main_proj": "fc.weight", "main_norm.weight": "enc.output_norm.weight", "norm.weight": "output_norm.weight",
                 "markov_head.embed.weight": "markov_w1.weight", "markov_head.head.weight": "markov_w2.weight",
                 "confidence_head.proj.weight": "conf_proj.weight"}


def load_dense(model: ref.Transformer, G: GGUF, Gd: GGUF | None = None):
    """Fill every non-expert, non-engram-table parameter from the GGUF, exactly (mtp.* = the DSpark
    head, from Gd)."""
    t0 = time.time()
    n = 0
    params = dict(model.named_parameters())
    for name, p in params.items():
        if p.is_meta or ".engram.embed." in name or name.endswith(".scale"):
            continue
        if name in GLOBAL:
            src = G.tensor(GLOBAL[name])
        elif name.startswith("mtp."):
            assert Gd is not None, f"{name}: no DSpark GGUF"
            _, L, suffix = name.split(".", 2)
            prefix = suffix.removesuffix(".weight")
            if suffix in DSPARK_GLOBAL or prefix in DSPARK_GLOBAL:
                src = Gd.tensor(DSPARK_GLOBAL.get(suffix) or DSPARK_GLOBAL[prefix])
            else:
                g = REV.get(prefix) or REV.get(suffix)
                assert g is not None, f"no DSpark GGUF tensor for {name}"
                src = Gd.tensor(f"blk.{L}.{g}")
        else:
            _, L, suffix = name.split(".", 2)
            if suffix == "engram.q_weight":  # the GGUF stores q_weight * k_weight
                p.data = G.tensor(f"blk.{L}.engram_gate.weight")
                params[name.replace("q_weight", "k_weight")].data = torch.ones_like(p.data)
                n += 1
                continue
            if suffix == "engram.k_weight":
                continue
            prefix = suffix.removesuffix(".weight")
            g = REV.get(prefix) or REV.get(suffix)
            assert g is not None, f"no GGUF tensor for {name}"
            src = G.tensor(f"blk.{L}.{g}")
        assert tuple(src.shape) == tuple(p.shape), (name, tuple(src.shape), tuple(p.shape))
        if p.dtype == torch.float8_e4m3fn:
            q, s = fp8_blocks(src)
            p.data = q
            scale = params[name.removesuffix(".weight") + ".scale"]
            assert tuple(scale.shape) == tuple(s.shape), name
            scale.data = s
        else:
            v = src.to(p.dtype)
            assert torch.equal(v.float(), src), f"{name}: GGUF values not exact in {p.dtype}"
            p.data = v
        n += 1
    print(f"# loaded {n} dense tensors from the GGUF in {time.time() - t0:.0f}s")


def install_lazy_experts(model: ref.Transformer, G: GGUF, Gd: GGUF | None = None, n_layers: int = 40):
    """MoE.forward: materialize the routed experts it is about to use, run, drop them again
    (layers from n_layers on are the DSpark head's, in Gd)."""
    def forward(self, x, image_mask=None):
        _, indices = self.gate(x.view(-1, self.dim), None)
        used = sorted(set(indices.flatten().tolist()))
        L = self.layer_id
        src, Ls = (G, L) if L < n_layers else (Gd, L - n_layers)
        for e in used:
            ex = self.experts[e]
            for w, g in (("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")):
                packed, scale = mxfp4_to_ref(src, f"blk.{Ls}.{g}.weight", e)
                lin = getattr(ex, w)
                lin.weight = nn.Parameter(packed, requires_grad=False)
                lin.scale = nn.Parameter(scale, requires_grad=False)
                lin.weight.scale = lin.scale
        try:
            return orig_forward(self, x, image_mask)
        finally:
            for e in used:  # back to meta: the next forward materializes again
                ex = self.experts[e]
                for w in ("w1", "w2", "w3"):
                    lin = getattr(ex, w)
                    lin.weight = nn.Parameter(torch.empty_like(lin.weight, device="meta"), requires_grad=False)
                    lin.scale = nn.Parameter(torch.empty_like(lin.scale, device="meta"), requires_grad=False)
                    lin.weight.scale = lin.scale

    orig_forward = ref.MoE.forward
    ref.MoE.forward = forward


# ------------------------------------------------------------------ recording

class Recorder:
    """keep_last > 0: for activations other than the block inputs (h_in, pre_in, engram_in), keep
    only the last `keep_last` positions (a 1k-token prefill would otherwise be ~14 GB). The block
    inputs are always kept whole: teacher-forced layer tests rebuild the KV caches from them."""
    FULL = {"h_in", "pre_in", "engram_in"}

    def __init__(self, model: ref.Transformer, keep_last: int = 0):
        self.layers = {}
        self.keep_last = keep_last
        for L, blk in enumerate(model.layers):
            self._hook_block(L, blk)

    def _rec(self, L, key, t):
        t = t.detach()
        if self.keep_last and key not in self.FULL:
            if key in ("router_w", "router_i"):  # [tokens, k], no batch dim
                t = t[-self.keep_last:]
            elif t.dim() >= 2:  # [batch, positions, ...]
                t = t[:, -self.keep_last:]
        self.layers.setdefault(L, {})[key] = t.clone()  # batch dim kept (always 1)

    def _hook_block(self, L, blk):
        def pre(mod, args):
            x, start_pos, pre_mix = args[0], args[1], args[2]
            self._rec(L, "h_in", x)
            self._rec(L, "pre_in", pre_mix)

        def post(mod, args, out):
            self._rec(L, "h_out", out[0])
            self._rec(L, "pre_out", out[1])
        blk.register_forward_pre_hook(pre)
        blk.register_forward_hook(post)
        blk.attn.register_forward_pre_hook(lambda m, a: self._rec(L, "attn_in", a[0]))
        blk.attn.register_forward_hook(lambda m, a, o: self._rec(L, "attn_out", o))
        blk.ffn.register_forward_pre_hook(lambda m, a: self._rec(L, "ffn_in", a[0]))
        blk.ffn.register_forward_hook(lambda m, a, o: self._rec(L, "ffn_out", o))

        def gate_out(m, a, o):  # called twice per MoE forward (lazy-expert pre-pass): keep the last
            self._rec(L, "router_w", o[0])
            self._rec(L, "router_i", o[1])
        blk.ffn.gate.register_forward_hook(gate_out)
        if blk.attn.indexer is not None:
            blk.attn.indexer.register_forward_hook(lambda m, a, o: self._rec(L, "topk", o))
        if blk.engram is not None:
            blk.engram.register_forward_pre_hook(lambda m, a: self._rec(L, "engram_in", a[0]))

    def take(self):
        out, self.layers = self.layers, {}
        return out


# ------------------------------------------------------------------ main

class TokenizerAdapter:
    """The two members build_compressed_token_map uses (the reference passes a HF AutoTokenizer)."""

    def __init__(self, tok):
        self.backend_tokenizer = tok

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="text prompt")
    g.add_argument("--tokens", help="comma-separated token ids")
    g.add_argument("--file", help="text file as a raw prompt (BOS + text), truncated to --prompt-tokens")
    ap.add_argument("--chat", action="store_true", help="wrap --prompt as one user turn (chat mode, no thinking)")
    ap.add_argument("--n-decode", type=int, default=4, help="greedy decode steps after the prefill")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=0)
    ap.add_argument("--keep-last", type=int, default=0, help="see Recorder")
    ap.add_argument("--fp32", action="store_true",
                    help="variant: activations in fp32 instead of the reference's bf16 (bf16 weights widened)")
    ap.add_argument("--no-quant", action="store_true",
                    help="variant: no fp8/fp4 fake quantization of activations, KV and indexer")
    ap.add_argument("--dspark", help="directory of the DSpark draft GGUF: also run and record the draft head")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HF_DIR / "tokenizer.json"))
    if a.tokens:
        prompt_ids = [int(x) for x in a.tokens.split(",")]
    elif a.file:
        prompt_ids = tok.encode("<｜begin▁of▁sentence｜>" + open(a.file).read()).ids
        if a.prompt_tokens:
            prompt_ids = prompt_ids[:a.prompt_tokens]
    else:
        text = a.prompt
        if a.chat:
            from encoding import encode_messages
            text = encode_messages([{"role": "user", "content": a.prompt}], thinking_mode="chat")
        prompt_ids = tok.encode(text).ids
    print(f"# prompt: {len(prompt_ids)} tokens {prompt_ids}")

    # the real reference is bf16 + fake quant; the variants exist to test our implementation
    # with less rounding noise in the comparison (see README "Step 3")
    torch.set_default_dtype(torch.float32 if a.fp32 else torch.bfloat16)
    if a.no_quant:
        def act_quant_off(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
            if inplace:
                return x
            return x, torch.ones(*x.shape[:-1], x.size(-1) // block_size, dtype=torch.float32)
        ref.act_quant = act_quant_off
        ref.fp4_act_quant = lambda x, block_size=32, inplace=False, scale_dtype=None: x
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cfg = json.load(open(HF_DIR / "inference/config.json"))
    args = ref.ModelArgs(**cfg)
    args.max_batch_size, args.max_seq_len = 1, a.max_seq_len
    args.temperature = 0.0  # greedy
    args.vision_n_layers = 0  # text only
    if not a.dspark:
        args.dspark_block_size = 0  # no draft head
    assert len(prompt_ids) + a.n_decode <= args.max_seq_len

    G = GGUF()
    t0 = time.time()
    # the engram table is the one tensor too big to allocate: build its modules on meta, then swap
    _emb_init = ref.ParallelEngramEmbedding.__init__

    def emb_init(self, num_embeddings, dim):
        with torch.device("meta"):
            _emb_init(self, num_embeddings, dim)
    ref.ParallelEngramEmbedding.__init__ = emb_init
    model = ref.Transformer(args, TokenizerAdapter(tok))
    ref.ParallelEngramEmbedding.__init__ = _emb_init
    base = 0
    for L, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
        model.layers[L].engram.embed = GGUFEngramEmbedding(G, base, rows, args.engram_head_dim)
        base += rows
    print(f"# built the reference Transformer in {time.time() - t0:.0f}s")
    Gd = GGUF(Path(a.dspark)) if a.dspark else None
    load_dense(model, G, Gd)
    if a.fp32:  # the few bf16 parameters (exact in fp32) would otherwise meet fp32 activations
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.float()
    install_lazy_experts(model, G, Gd, args.n_layers)
    rec = Recorder(model, a.keep_last)

    # the generate() loop of the reference, batch 1, greedy, recording every forward
    steps = []
    tokens = list(prompt_ids)
    prev = 0
    for step in range(a.n_decode + 1):
        cur = len(tokens)
        inp = torch.tensor([tokens[prev:cur]], dtype=torch.long)
        t1 = time.time()
        with torch.inference_mode():
            out_ids, logits, main_hidden = model.forward(inp, prev)
            spec = model.forward_spec(out_ids, main_hidden, prev) if a.dspark else None
        nxt = int(out_ids[0])
        steps.append({"start_pos": prev, "tokens": tokens[prev:cur], "logits": logits[0].float().clone(),
                      "next": nxt, "layers": rec.take()})
        if spec is not None:
            draft_ids, draft_logits, conf = spec
            steps[-1].update({"draft_ids": draft_ids[0].tolist(), "draft_logits": draft_logits[0].float().clone(),
                              "draft_conf": conf[0].float().tolist()})
            print(f"  draft: {draft_ids[0].tolist()}  conf {[round(c, 3) for c in conf[0].float().tolist()]}")
        top = logits[0].float().topk(5)
        print(f"step {step}: pos {prev}..{cur - 1} ({cur - prev} tok) -> {nxt} {tok.decode([nxt])!r}  "
              f"top5 {top.indices.tolist()}  {time.time() - t1:.0f}s")
        tokens.append(nxt)
        prev = cur
    print(f"# completion: {tok.decode(tokens[len(prompt_ids):])!r}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_ids": prompt_ids, "tokens": tokens, "steps": steps,
                "variant": {"fp32": a.fp32, "no_quant": a.no_quant},
                "args": {k: v for k, v in vars(args).items() if not k.startswith("_")}}, a.out)
    print(f"# wrote {a.out}")


if __name__ == "__main__":
    main()
