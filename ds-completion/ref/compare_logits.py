#!/usr/bin/env python3
"""Compare per-step logits of a ds_ref.py golden file against another implementation's.

    ~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/hello.pt data/llama/hello.f32

The other side is a raw float32 file of [steps][n_vocab] (llama-logits --out, later
ds-completion's own dump). Per step: argmax agreement, top-5 overlap, KL(ref || other) of the
softmax, and the max |logit difference| relative to the ref's logit spread.
"""
import functools
import sys

import numpy as np
import torch

print = functools.partial(print, flush=True)


def main():
    ref = torch.load(sys.argv[1], weights_only=False)
    other = torch.from_numpy(np.fromfile(sys.argv[2], dtype=np.float32))
    n_vocab = ref["steps"][0]["logits"].numel()
    other = other.view(-1, n_vocab)
    steps = ref["steps"]
    n = min(len(steps), other.size(0))
    if other.size(0) != len(steps):
        print(f"# note: {len(steps)} reference steps, {other.size(0)} in the other file; comparing {n}")
    kls, agree = [], 0
    print(f"{'step':>4} {'pos':>5} {'ref':>7} {'other':>7} {'top5':>5} {'KL':>10} {'max|d|/std':>11}")
    for i in range(n):
        a, b = steps[i]["logits"].float(), other[i]
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        kl = (la.exp() * (la - lb)).sum().item()
        top_a, top_b = a.topk(5).indices.tolist(), b.topk(5).indices.tolist()
        same = top_a[0] == top_b[0]
        agree += same
        kls.append(kl)
        d = (a - b).abs().max().item() / a.std().item()
        print(f"{i:4d} {steps[i]['start_pos']:5d} {top_a[0]:7d} {top_b[0]:7d} {len(set(top_a) & set(top_b)):5d} "
              f"{kl:10.2e} {d:11.3f}{'' if same else '   <- argmax differs'}")
    print(f"# argmax agrees {agree}/{n}, mean KL {np.mean(kls):.2e}, max KL {np.max(kls):.2e}")


if __name__ == "__main__":
    main()
