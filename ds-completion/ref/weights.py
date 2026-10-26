"""Read the same weights from both sources, decoded to float32 torch tensors.

  HF    ~/llms/ds-flash               original safetensors (fp8 e4m3 + 32x32 e8m0 scales,
                                      fp4 e2m1 packed in int8 + per-32 e8m0 scales, bf16, f32)
  GGUF  ~/llms/UD-Q8_K_XL/*.gguf      what ds-completion loads (bf16, f32, mxfp4, mxfp8)

The two decoders are written independently on purpose: the HF side follows the reference
(inference/convert.py FP4_TABLE, model.py e8m0 = 2^(e-127)), the GGUF side follows ggml's own
block layouts (ggml-common.h: kvalues_mxfp4 doubled + e8m0 halved; block_mxfp8). If both agree
bit for bit, the GGUF is the checkpoint and our name mapping is right.
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HOME = Path.home()
HF_DIR = HOME / "llms/ds-flash"
GGUF_DIR = HOME / "llms/UD-Q8_K_XL"
sys.path.insert(0, str(HOME / "forks/llama.cpp/gguf-py"))  # the fork's gguf-py knows MXFP8
import gguf  # noqa: E402

# ---------------------------------------------------------------- HF side (reference semantics)

FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
FP8_BLOCK = 32


def e8m0(u8: torch.Tensor) -> torch.Tensor:
    u8 = u8.view(torch.uint8)
    assert not (u8 == 0xFF).any(), "NaN e8m0 scale"
    return torch.exp2(u8.float() - 127)


class HF:
    def __init__(self, d: Path = HF_DIR):
        self.dir = d
        self.map = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.files = {}

    def has(self, name):
        return name in self.map

    def _f(self, name):
        fn = self.map[name]
        if fn not in self.files:
            self.files[fn] = safe_open(self.dir / fn, framework="pt")
        return self.files[fn]

    def raw(self, name) -> torch.Tensor:
        return self._f(name).get_tensor(name)

    def prefetch(self, names):
        pass  # local files: nothing to do

    def rows(self, name, ranges) -> list[torch.Tensor]:
        sl = self._f(name).get_slice(name)
        return [sl[r0:r1] for r0, r1 in ranges]

    def linear(self, prefix) -> torch.Tensor:
        """`prefix`.weight decoded to float32, applying `prefix`.scale for fp8 / fp4 weights."""
        w = self.raw(prefix + ".weight")
        if w.dtype == torch.float8_e4m3fn:
            s = e8m0(self.raw(prefix + ".scale"))
            o, i = w.shape
            s = s.repeat_interleave(FP8_BLOCK, 0)[:o].repeat_interleave(FP8_BLOCK, 1)[:, :i]
            return w.float() * s
        if w.dtype == torch.int8:  # fp4 e2m1, element 2i in the low nibble
            return fp4_rows(w, self.raw(prefix + ".scale"))
        assert not self.has(prefix + ".scale"), prefix
        return w.float()


HF_1017 = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
ST_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.int8,
             "BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8}


class HFRemote(HF):
    """The 1017 checkpoint read over HTTP range requests: only the bytes of the tensors asked for.

    The local ~/llms/ds-flash is the *base* V4.1-Flash release; the GGUF was converted from the 1017
    revision (same config, code, tokenizer and tensor names; different weight values). Downloading all
    510 GB to check a few layers is unnecessary - each safetensors file is an 8-byte header length, a
    JSON header with byte offsets, then raw data.
    """

    def __init__(self, url: str = HF_1017, index: Path | None = None, threads: int = 16):
        import urllib.request
        self.url, self._ur = url, urllib.request
        import threading
        self.headers, self.cache, self.threads = {}, {}, threads
        self.bytes_fetched, self.lock = 0, threading.Lock()
        idx = json.load(open(index)) if index else json.loads(self._get("model.safetensors.index.json"))
        self.map = idx["weight_map"]

    CHUNK = 8 << 20

    def _get(self, path, start=None, end=None) -> bytes:
        req = self._ur.Request(self.url + path)
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end - 1}")
        for attempt in range(6):
            try:
                with self._ur.urlopen(req, timeout=120) as r:
                    data = r.read()
                if start is None or len(data) == end - start:
                    break
            except OSError:
                pass
            if attempt == 5:
                raise OSError(f"cannot fetch {path} [{start}, {end})")
            time.sleep(2 ** attempt)
        with self.lock:
            self.bytes_fetched += len(data)
        return data

    def _fetch(self, jobs) -> list[bytearray]:
        """Byte ranges [(file, start, end)], fetched as 8 MiB pieces over `threads` connections
        (one connection gets ~13 MB/s from the CDN here; many scale nearly linearly)."""
        from concurrent.futures import ThreadPoolExecutor
        out = [bytearray(e - s) for _, s, e in jobs]
        pieces = [(j, fn, a, min(e, a + self.CHUNK), a - s) for j, (fn, s, e) in enumerate(jobs)
                  for a in range(s, e, self.CHUNK)]

        def run(p):
            j, fn, a, b, o = p
            out[j][o:o + b - a] = self._get(fn, a, b)
        with ThreadPoolExecutor(self.threads) as ex:
            list(ex.map(run, pieces))
        return out

    def _header(self, fn):
        if fn not in self.headers:
            n = struct.unpack("<Q", self._get(fn, 0, 8))[0]
            self.headers[fn] = (8 + n, json.loads(self._get(fn, 8, 8 + n)))
        return self.headers[fn]

    def _meta(self, name):
        fn = self.map[name]
        base, h = self._header(fn)
        m = h[name]
        return fn, base, m["dtype"], m["shape"], m["data_offsets"]

    def prefetch(self, names):
        """Fetch many tensors concurrently; raw() then serves them from memory (once each)."""
        names = [n for n in names if n not in self.cache]
        metas = [self._meta(n) for n in names]
        bufs = self._fetch([(fn, base + a, base + b) for fn, base, _, _, (a, b) in metas])
        for n, (_, _, dt, shape, _), buf in zip(names, metas, bufs):
            self.cache[n] = torch.frombuffer(buf, dtype=ST_DTYPES[dt]).view(shape)

    def raw(self, name) -> torch.Tensor:
        if name not in self.cache:
            self.prefetch([name])
        return self.cache.pop(name)

    def rows(self, name, ranges) -> list[torch.Tensor]:
        """Row ranges [(r0, r1)] of a 2-D tensor, fetched concurrently."""
        fn, base, dt, shape, (a, _) = self._meta(name)
        row = int(np.prod(shape[1:])) * torch.empty(0, dtype=ST_DTYPES[dt]).element_size()
        bufs = self._fetch([(fn, base + a + r0 * row, base + a + r1 * row) for r0, r1 in ranges])
        return [torch.frombuffer(b, dtype=ST_DTYPES[dt]).view(r1 - r0, *shape[1:]) for b, (r0, r1) in zip(bufs, ranges)]


def fp4_rows(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    u = packed.view(torch.uint8)
    v = torch.stack([FP4_TABLE[(u & 0xF).long()], FP4_TABLE[(u >> 4).long()]], dim=-1).flatten(-2)
    s = e8m0(scale).repeat_interleave(32, -1)
    return v * s


def fp8_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Engram table rows: e4m3 values, one e8m0 scale per 32."""
    return q.view(torch.float8_e4m3fn).float() * e8m0(scale).repeat_interleave(32, -1)


# ---------------------------------------------------------------- GGUF side (ggml semantics)

KVALUES_MXFP4 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)


class GGUF:
    """All shards of the split GGUF, tensors addressable by name, data as zero-copy memmaps."""

    def __init__(self, d: Path = GGUF_DIR):
        self.shards = [gguf.GGUFReader(p) for p in sorted(d.glob("*.gguf"))]
        self.tensors = {t.name: t for r in self.shards for t in r.tensors}
        self.kv = self.shards[0].fields

    def field(self, key):
        f = self.kv[key]
        vals = [f.parts[i] for i in f.data]
        if f.types[0] == gguf.GGUFValueType.STRING:
            return bytes(vals[0]).decode()
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            if f.types[-1] == gguf.GGUFValueType.STRING:
                return [bytes(v).decode() for v in vals]
            return [v.item() for v in vals]
        return vals[0].item()

    def type(self, name) -> gguf.GGMLQuantizationType:
        return self.tensors[name].tensor_type

    def ne(self, name) -> list[int]:
        return [int(x) for x in self.tensors[name].shape]

    def raw(self, name) -> np.ndarray:
        return self.tensors[name].data.reshape(-1).view(np.uint8)

    def rows(self, name, r0=0, r1=None) -> torch.Tensor:
        """Rows r0..r1 of a 2-D tensor (or of the flattened outer dims), decoded to float32 [rows, ne0]."""
        t = self.tensors[name]
        ne0 = int(t.shape[0])
        n_rows = int(np.prod([int(x) for x in t.shape[1:]]))
        r1 = n_rows if r1 is None else r1
        typ = t.tensor_type
        block, tsize = gguf.GGML_QUANT_SIZES[typ]
        row_bytes = ne0 // block * tsize
        b = torch.from_numpy(np.array(self.raw(name)[r0 * row_bytes: r1 * row_bytes]))  # copy: memmap is read-only
        n = r1 - r0
        T = gguf.GGMLQuantizationType
        if typ == T.F32:
            return b.view(torch.float32).view(n, ne0).clone()
        if typ == T.BF16:
            return (b.view(torch.int16).int() << 16).view(torch.float32).view(n, ne0)
        if typ == T.MXFP4:  # block: e (1) + qs (16); qs[j] low = element j, high = element j+16
            blk = b.view(n, ne0 // 32, 17)
            e = blk[..., 0].float()
            qs = blk[..., 1:].long()
            codes = torch.cat([qs & 0xF, qs >> 4], dim=-1)
            return (KVALUES_MXFP4[codes] * torch.exp2(e - 128).unsqueeze(-1)).view(n, ne0)
        if typ == T.MXFP8:  # block: e (1) + qs (32 e4m3)
            blk = b.view(n, ne0 // 32, 33)
            e = blk[..., 0].float()
            v = blk[..., 1:].contiguous().view(torch.float8_e4m3fn).float()
            return (v * torch.exp2(e - 127).unsqueeze(-1)).view(n, ne0)
        raise NotImplementedError(typ.name)

    def tensor(self, name) -> torch.Tensor:
        """Whole tensor in HF orientation (ggml ne reversed)."""
        return self.rows(name).view(*reversed(self.ne(name)))


# ---------------------------------------------------------------- name mapping

# per-layer GGUF suffix -> HF suffix (both without "blk.L." / "layers.L.")
LAYER_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "attn_q_a.weight": "attn.wq_a",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b",
    "attn_kv.weight": "attn.wkv",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a",
    "attn_output_b.weight": "attn.wo_b",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_compressor_kv.weight": "attn.compressor.wkv",
    "attn_compressor_gate.weight": "attn.compressor.wgate",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b",
    "indexer.proj.weight": "attn.indexer.weights_proj",
    "indexer.attn_k.weight": "attn.indexer.wk",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "engram_kv.weight": "engram.wkv",
    # engram_gate.weight = engram.q_weight * engram.k_weight (special-cased)
    # ffn_{gate,down,up}_exps.weight = ffn.experts.E.w{1,2,3} stacked (special-cased)
}
EXPS = {"ffn_gate_exps.weight": "w1", "ffn_down_exps.weight": "w2", "ffn_up_exps.weight": "w3"}
GLOBAL_MAP = {"token_embd.weight": "embed.weight", "output.weight": "head.weight", "output_norm.weight": "norm.weight"}


def hf_value(hf: HF, name: str) -> torch.Tensor:
    """The HF tensor that GGUF tensor `name` should equal (not for experts / engram table)."""
    if name in GLOBAL_MAP:
        return hf.raw(GLOBAL_MAP[name]).float()
    _, layer, suffix = name.split(".", 2)
    p = f"layers.{layer}."
    if suffix == "engram_gate.weight":
        return hf.raw(p + "engram.q_weight").float() * hf.raw(p + "engram.k_weight").float()
    src = LAYER_MAP[suffix]
    if hf.has(p + src + ".weight"):
        return hf.linear(p + src)
    return hf.raw(p + src).float()
