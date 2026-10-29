"""The runtime's indexer selection kernels (hip/kernels.h: k_topk, k_candidate_blocks,
k_mask_candidates) against the reference's own code on the same scores (PLAN.md "Long context").

    ~/venvs/ds-ref/bin/python ref/check_topk.py [--bin bin-hip/test-topk]

Reference semantics (inference/model.py, Indexer.forward at decode, one query):
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
and select_candidate_blocks, copied below verbatim. Continuous random scores (no ties) must give
identical indices / masks. With ties torch's choice is unspecified: there the selected values
must match and our rule (ties to the lower index) is checked on its own; a case whose k-th value is tied
(e.g. several exact zeros of rectified sums) is checked that way automatically.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

TOPK, BLOCK, TOPK_BLOCKS = 512, 8, 2048  # config.json: index_topk, candidate_block_size, candidate_topk_blocks


# --- verbatim from inference/model.py ---------------------------------------------------------
def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]
# ----------------------------------------------------------------------------------------------


def ref_topk(score, k):
    return torch.from_numpy(score).topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values.numpy()


def ref_keep_blocks(score):
    n = len(score)
    keep = select_candidate_blocks(torch.from_numpy(score), n, TOPK_BLOCKS, BLOCK).numpy()
    return np.pad(keep, (0, -n % BLOCK))[::BLOCK].astype(np.uint8)  # per block (all positions of a block agree)


def ref_masked_topk(score, k):
    n = len(score)
    keep = select_candidate_blocks(torch.from_numpy(score), n, TOPK_BLOCKS, BLOCK)
    masked = torch.from_numpy(score).masked_fill(~keep, -torch.inf)
    return masked.topk(k, dim=-1, sorted=False).indices.sort(dim=-1).values.numpy()


def ours_tie_rule(score, k):
    """our specification: the k largest, ties at the k-th value to the lower index, sorted"""
    order = np.lexsort((np.arange(len(score)), -score.astype(np.float64)))  # by score desc, index asc
    return np.sort(order[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(os.path.dirname(__file__), "..", "bin-hip", "test-topk"))
    a = ap.parse_args()
    rng = np.random.default_rng(17)
    cases = []  # (name, mode, n, k, scores, check)

    def scores(n, kind):
        if kind == "normal":
            return rng.standard_normal(n).astype(np.float32)
        if kind == "relu-sum":  # like the indexer: sums of rectified dot products, many small, some 0
            return np.maximum(rng.standard_normal((n, 4)), 0).sum(1).astype(np.float32) * rng.uniform(0.5, 2)
        if kind == "ties":  # few distinct values: ties at the k-th value
            return rng.integers(0, 50, n).astype(np.float32)
        raise ValueError(kind)

    for n in [513, 1100, 4096, 4097, 5000, 16384, 20000, 65536, 262144, 1048576]:
        for kind in ["normal", "relu-sum"]:
            cases.append((f"topk n={n} {kind}", 0, n, min(TOPK, n), scores(n, kind), "exact"))
    for n in [1100, 70000]:
        cases.append((f"topk n={n} ties", 0, n, TOPK, scores(n, "ties"), "ties"))
    for n in [16385, 16391, 20000, 65536, 262144, 1048576]:
        for kind in ["normal", "relu-sum"]:
            cases.append((f"candidates n={n} {kind}", 1, n, 0, scores(n, kind), "exact"))
            cases.append((f"candidates+topk n={n} {kind}", 2, n, TOPK, scores(n, kind), "exact"))

    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(fin, "wb") as f:
            for _, mode, n, k, sc, _ in cases:
                f.write(np.array([mode, n, k, BLOCK, TOPK_BLOCKS], np.int32).tobytes())
                f.write(sc.tobytes())
        subprocess.run([a.bin, fin, fout], check=True)
        out = open(fout, "rb").read()

    off, bad = 0, 0
    for name, mode, n, k, sc, check in cases:
        if mode == 1:
            nb = -(-n // BLOCK)
            got = np.frombuffer(out, np.uint8, nb, off)
            off += nb
            want = ref_keep_blocks(sc)
            ok = np.array_equal(got, want)
            detail = f"{int(got.sum())} blocks kept, {int((got != want).sum())} differ"
        else:
            got = np.frombuffer(out, np.int32, k, off)
            off += 4 * k
            eff = sc if mode == 0 else None
            if eff is not None and k < n:  # a tie at the k-th value: torch's pick is unspecified there
                kth = np.sort(eff)[::-1][k - 1]
                if (eff == kth).sum() > 1 and (np.sort(eff)[::-1][k:] == kth).any():
                    check = "ties"
            if check == "exact":
                want = ref_masked_topk(sc, k) if mode == 2 else ref_topk(sc, k)
                ok = np.array_equal(got, want)
                detail = f"{len(np.setdiff1d(want, got))} of {k} indices differ" + ("" if np.all(np.diff(got) > 0) else ", NOT sorted")
            else:  # ties: the same selected values as torch, and our tie rule exactly
                want = ref_topk(sc, k)
                same_vals = np.array_equal(np.sort(sc[got]), np.sort(sc[want]))
                ok = same_vals and np.array_equal(got, ours_tie_rule(sc, k))
                detail = f"values {'match' if same_vals else 'DIFFER'}, tie rule {'held' if np.array_equal(got, ours_tie_rule(sc, k)) else 'BROKEN'}"
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:40s} {detail}")
    print(f"{len(cases) - bad} of {len(cases)} cases match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
