#!/usr/bin/env python3
"""Compare two lkldtopk files position by position (A/A or A/B).

Both files must carry the SAME token sequences (that is the contract: rescore
consumes collect's token ids verbatim). For each position, computes:

- top-1 agreement between the two stored distributions;
- truncated KL(A||B) over the shared top-K support:
      KL_shared = sum_{i in idsA & idsB} p_A[i] * (logp_A[i] - logp_B[i])
  using exact probabilities (normalizers stored in the files, no
  renormalization). This drops A-mass outside the shared support, so the
  uncovered mass (1 - coverage) is reported alongside as the error-bound
  driver; a full KL-with-bounds tool can refine this later.

Usage: python compare.py a.bin b.bin
"""

import math
import sys

from inspect import read_file, logprobs  # local inspect.py, not stdlib inspect


def compare_seq(sa, sb, label):
    if sa["tokens"] != sb["tokens"]:
        print(f"FAIL [{label}] token sequences differ — files are not comparable")
        return None
    if len(sa["positions"]) != len(sb["positions"]):
        print(f"FAIL [{label}] n_scored differs: {len(sa['positions'])} vs {len(sb['positions'])}")
        return None

    rows = []
    for pa, pb in zip(sa["positions"], sb["positions"]):
        lpa = dict(zip(pa["ids"], logprobs(pa)))
        lpb = dict(zip(pb["ids"], logprobs(pb)))
        shared = lpa.keys() & lpb.keys()
        kl = sum(math.exp(lpa[i]) * (lpa[i] - lpb[i]) for i in shared)
        coverage = sum(math.exp(lpa[i]) for i in shared)
        rows.append({
            "top1_match": pa["ids"][0] == pb["ids"][0],
            "kl": kl,
            "uncovered": 1.0 - coverage,
        })
    return rows


def summarize(rows, label, n_prompt):
    n = len(rows)
    kls = sorted(r["kl"] for r in rows)
    unc = sorted(r["uncovered"] for r in rows)
    agree = sum(r["top1_match"] for r in rows)
    print(f"seq [{label}]: {n} positions ({n_prompt} prompt)")
    print(f"  top-1 agreement: {agree}/{n} ({100.0*agree/n:.2f}%)")
    print(f"  KL(A||B) shared-support: mean={sum(kls)/n:.4e} median={kls[n//2]:.4e} "
          f"p99={kls[int(0.99*(n-1))]:.4e} max={kls[-1]:.4e}")
    print(f"  A-mass outside shared support: mean={sum(unc)/n:.3e} max={unc[-1]:.3e}")
    return agree == n


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    a = read_file(sys.argv[1])
    b = read_file(sys.argv[2])

    print(f"A: {a['model_desc']}")
    print(f"B: {b['model_desc']}")
    if a["n_vocab"] != b["n_vocab"]:
        print(f"WARNING: n_vocab differs ({a['n_vocab']} vs {b['n_vocab']})")
    if len(a["seqs"]) != len(b["seqs"]):
        print(f"FAIL: n_seq differs ({len(a['seqs'])} vs {len(b['seqs'])})")
        return 1

    ok = True
    for sa, sb in zip(a["seqs"], b["seqs"]):
        rows = compare_seq(sa, sb, sa["label"])
        if rows is None:
            ok = False
            continue
        summarize(rows, sa["label"], sa["n_prompt"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
