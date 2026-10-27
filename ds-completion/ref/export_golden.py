#!/usr/bin/env python3
"""Convert a ds_ref.py golden file (.pt) into a flat format the C++ tests read without torch.

    ~/venvs/ds-ref/bin/python ref/export_golden.py data/ref/hello.pt data/golden/hello

Writes <dir>/data.bin (all tensors back to back) and <dir>/index.txt, one line per tensor:

    <step> <layer> <key> <dtype> <ndim> <dim0> ... <dimN-1> <byte offset>

layer is -1 for per-step values. Shapes are torch order (outermost first; the batch dim of 1 is
dropped), so the innermost dim is ggml's ne[0]. dtype is f32 or i32: activations are stored as
f32 (bf16 widened exactly), ids as i32. Per-step entries: tokens (i32), logits (f32), and in the
header lines "# start_pos <step> <pos>" and "# prompt <ids...>".
"""
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    src, out = sys.argv[1], Path(sys.argv[2])
    g = torch.load(src, weights_only=False)
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# source {src}", "# prompt " + " ".join(map(str, g["prompt_ids"]))]
    off = 0
    with open(out / "data.bin", "wb") as f:
        def put(step, layer, key, t):
            nonlocal off
            t = t.detach()
            if t.is_floating_point():
                a, dt = t.float().numpy(), "f32"
            else:
                a, dt = t.to(torch.int32).numpy(), "i32"
            a = np.ascontiguousarray(a)
            f.write(a.tobytes())
            lines.append(f"{step} {layer} {key} {dt} {a.ndim} {' '.join(map(str, a.shape))} {off}")
            off += a.nbytes

        for s, st in enumerate(g["steps"]):
            lines.append(f"# start_pos {s} {st['start_pos']}")
            put(s, -1, "tokens", torch.tensor(st["tokens"]))
            put(s, -1, "logits", st["logits"])
            for L in sorted(st["layers"]):
                for key, t in sorted(st["layers"][L].items()):
                    if t.dim() > 0 and t.size(0) == 1 and key not in ("router_w", "router_i"):
                        t = t[0]  # batch
                    put(s, L, key, t)
    (out / "index.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(lines)} entries, {off / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
