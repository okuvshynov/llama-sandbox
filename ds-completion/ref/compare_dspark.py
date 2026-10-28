#!/usr/bin/env python3
"""Our DSpark drafts (ds-test --e2e ... --dspark F --dspark-out ours.bin) against the reference's
(ds_ref.py --dspark: the official forward_spec), step by step on the same token sequence.

    ~/venvs/ds-ref/bin/python ref/compare_dspark.py data/ref/x-dspark.pt ours.bin
"""
import sys

import numpy as np
import torch

ref = torch.load(sys.argv[1], weights_only=False)
steps = [s for s in ref["steps"] if "draft_ids" in s]
B = len(steps[0]["draft_conf"])
V = steps[0]["draft_logits"].shape[-1]
rec = np.dtype([("ids", "<i4", (B + 1,)), ("conf", "<f4", (B,)), ("logits", "<f4", (B, V))])
ours = np.fromfile(sys.argv[2], dtype=rec)
print(f"{len(steps)} reference drafts, {len(ours)} ours; block {B}")
same_ids = 0
for k, (s, o) in enumerate(zip(steps, ours)):
    rl = s["draft_logits"].numpy()
    errs = [np.linalg.norm(o["logits"][j] - rl[j]) / np.linalg.norm(rl[j]) for j in range(B)]
    dc = np.abs(o["conf"] - np.array(s["draft_conf"])).max()
    eq = list(o["ids"]) == s["draft_ids"]
    same_ids += eq
    print(f"step {k + 1:2d}: ids {'same' if eq else 'DIFFER'}  ref {s['draft_ids']}  ours {list(o['ids'])}  "
          f"logits relerr max {max(errs):.1e}  conf |diff| max {dc:.1e}")
print(f"# drafts identical at {same_ids} of {min(len(steps), len(ours))} steps")
