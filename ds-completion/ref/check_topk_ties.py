"""Are the runtime's indexer disagreements with the reference near-ties? (README step 35)

    bin-hip/ds-test -m $M -g data/golden/X --runtime $F8 --dump-topk data/ds/X-topk.bin
    ~/venvs/ds-ref/bin/python ref/check_topk_ties.py data/golden/X data/ds/X-topk.bin

The golden must come from `ds_ref.py --scores`: per index layer, the scores the reference's
indexer handed to its top-k (bf16, after the candidate mask; -inf = not selectable). For every
query both sides kept, our picks are ranked against those scores:

  - extra = picked by us, not by the reference: its rank in the reference's order (513 = first
    position past the cutoff) and its gap below the reference's 512th score, as a fraction of the
    reference's top-512 range (best - 512th; scores can sit near 0, so not relative to the 512th);
  - masked = an extra the reference could not pick at all (score -inf: outside its candidate
    blocks, or not yet visible). Near-ties never produce these; a different candidate block set
    or a visibility bug does.

Near-ties look like: extras ranked just past 512 with gaps that are a small fraction of the
range, and no masked picks before the candidate consumers (on those, masked picks follow from
the candidate source's own near-ties at its block cutoff).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_golden(d):
    d = Path(d)
    data = np.memmap(d / "data.bin", dtype=np.uint8, mode="r")
    idx, prompt_len, window = {}, None, 128
    for line in open(d / "index.txt"):
        if line.startswith("# prompt"):
            prompt_len = len(line.split()) - 2
            continue
        if line.startswith("#"):
            continue
        f = line.split()
        step, layer, key, dt, nd = int(f[0]), int(f[1]), f[2], f[3], int(f[4])
        shape, off = [int(x) for x in f[5:5 + nd]], int(f[5 + nd])
        n = int(np.prod(shape))
        arr = np.frombuffer(data, dtype=np.float32 if dt == "f32" else np.int32, count=n, offset=off).reshape(shape)
        idx[(step, layer, key)] = arr
    return idx, prompt_len, window


def main():
    gdir, dump = sys.argv[1], sys.argv[2]
    G, T, W = load_golden(gdir)
    raw = np.fromfile(dump, dtype=np.int32)
    per_layer = defaultdict(lambda: {"rows": 0, "jac": 0.0, "extra": 0, "masked": 0, "ranks": [], "gaps": []})
    p, bad_ref = 0, 0
    while p < len(raw):
        s, L, r, n = raw[p:p + 4]
        ours = set(raw[p + 4:p + 4 + n].tolist())
        p += 4 + n
        sc = G.get((s, L, "index_score"))
        if sc is None:
            sys.exit(f"golden has no index_score for step {s} layer {L}: rerun ds_ref.py with --scores")
        sc = sc.reshape(-1, sc.shape[-1])[r].astype(np.float64)
        tk = G[(s, L, "topk")]
        tk = tk.reshape(-1, tk.shape[-1])[r]
        off = T if s == 0 else W
        ref = {int(v) - off for v in tk if v >= 0}
        order = np.argsort(-sc, kind="stable")
        rank = np.empty(len(sc), np.int64)
        rank[order] = np.arange(1, len(sc) + 1)
        k = len(ref)
        srt = np.sort(sc)[::-1]
        kth, span = srt[k - 1], srt[0] - srt[k - 1]
        bad_ref += any(rank[v] > k and sc[v] != kth for v in ref)  # sanity: the reference's picks are its top-k
        a = per_layer[L]
        a["rows"] += 1
        a["jac"] += len(ours & ref) / max(1, len(ours | ref))
        for v in ours - ref:
            a["extra"] += 1
            if v >= len(sc) or not np.isfinite(sc[v]):
                a["masked"] += 1
                continue
            a["ranks"].append(int(rank[v]))
            a["gaps"].append((kth - sc[v]) / span if span > 0 else 0.0)
    print(f"# {gdir}: prompt {T} tokens; the reference's picks are its own top-k on every row: {'yes' if not bad_ref else f'NO ({bad_ref} rows)'}")
    print(f"{'layer':>5} {'rows':>5} {'jaccard':>8} {'extra/row':>9} {'masked':>6} {'rank p50':>8} {'rank p99':>8} {'rank max':>8} {'gap p50':>8} {'gap p99':>8} {'gap max':>8}")
    for L in sorted(per_layer):
        a = per_layer[L]
        rk, gp = np.array(a["ranks"] or [0]), np.array(a["gaps"] or [0.0])
        print(f"{L:5d} {a['rows']:5d} {a['jac'] / a['rows']:8.3f} {a['extra'] / a['rows']:9.1f} {a['masked']:6d} "
              f"{np.percentile(rk, 50):8.0f} {np.percentile(rk, 99):8.0f} {rk.max():8d} "
              f"{np.percentile(gp, 50):8.1e} {np.percentile(gp, 99):8.1e} {gp.max():8.1e}")


if __name__ == "__main__":
    main()
