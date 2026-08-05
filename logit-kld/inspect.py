#!/usr/bin/env python3
"""Reference reader + sanity checker for "lkldtopk" v1 logit files.

This is the format's reference implementation: any rescoring utility (for any
inference framework) should read token sequences the way read_file() does.
Stdlib only.

Usage: python inspect.py file.bin [--dump-tokens]
"""

import math
import struct
import sys


def _read(fp, fmt):
    size = struct.calcsize(fmt)
    data = fp.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of file")
    vals = struct.unpack(fmt, data)
    return vals[0] if len(vals) == 1 else vals


def _read_str(fp):
    n = _read(fp, "<I")
    return fp.read(n).decode("utf-8", errors="replace")


def read_file(path):
    """Parse an lkldtopk v1 file into a dict."""
    with open(path, "rb") as fp:
        magic = fp.read(8)
        if magic != b"lkldtopk":
            raise ValueError(f"bad magic: {magic!r}")
        version = _read(fp, "<I")
        if version != 1:
            raise ValueError(f"unsupported version: {version}")
        n_vocab, top_k, n_seq = _read(fp, "<iii")
        model_desc = _read_str(fp)

        seqs = []
        for _ in range(n_seq):
            label = _read_str(fp)
            n_prompt, n_total, n_scored = _read(fp, "<iii")
            tokens = list(_read(fp, f"<{n_total}i")) if n_total else []
            positions = []
            for _ in range(n_scored):
                max_logit, lse_rest = _read(fp, "<ff")
                ids = list(_read(fp, f"<{top_k}i"))
                logits = list(_read(fp, f"<{top_k}f"))
                positions.append({
                    "max_logit": max_logit,
                    "lse_rest": lse_rest,
                    "ids": ids,
                    "logits": logits,
                })
            seqs.append({
                "label": label,
                "n_prompt": n_prompt,
                "n_total": n_total,
                "tokens": tokens,
                "positions": positions,
            })

        trailing = fp.read(1)
        if trailing:
            raise ValueError("trailing bytes after last sequence")

    return {
        "n_vocab": n_vocab,
        "top_k": top_k,
        "model_desc": model_desc,
        "seqs": seqs,
    }


def logprobs(pos):
    """Exact log-probabilities of the stored top-K entries."""
    lse = pos["max_logit"] + pos["lse_rest"]
    return [l - lse for l in pos["logits"]]


def tail_mass(pos):
    """Probability mass outside the stored top-K (bounds KL truncation error)."""
    return 1.0 - sum(math.exp(lp) for lp in logprobs(pos))


def check(f):
    ok = True
    for seq in f["seqs"]:
        n_prompt, n_total = seq["n_prompt"], seq["n_total"]
        positions = seq["positions"]

        if len(positions) != n_total:
            print(f"FAIL [{seq['label']}] n_scored={len(positions)} != n_total={n_total}")
            ok = False

        tails = []
        for i, pos in enumerate(positions):
            lg = pos["logits"]
            if pos["max_logit"] != lg[0]:
                print(f"FAIL [{seq['label']}] pos {i}: max_logit {pos['max_logit']} != top logit {lg[0]}")
                ok = False
            if any(lg[j] < lg[j + 1] for j in range(len(lg) - 1)):
                print(f"FAIL [{seq['label']}] pos {i}: logits not sorted descending")
                ok = False
            if pos["lse_rest"] < 0 or pos["lse_rest"] > math.log(f["n_vocab"]) + 1e-6:
                print(f"FAIL [{seq['label']}] pos {i}: lse_rest {pos['lse_rest']} out of [0, ln n_vocab]")
                ok = False
            t = tail_mass(pos)
            if t < -1e-5:
                print(f"FAIL [{seq['label']}] pos {i}: top-K mass exceeds 1 (tail={t:.3e})")
                ok = False
            tails.append(t)

        # greedy self-consistency: generated token i+1 must be the argmax at position i
        mismatches = [
            i for i in range(n_prompt - 1, n_total - 1)
            if positions[i]["ids"][0] != seq["tokens"][i + 1]
        ]
        if mismatches:
            print(f"FAIL [{seq['label']}] greedy mismatch at positions {mismatches[:10]}"
                  f"{'...' if len(mismatches) > 10 else ''}")
            ok = False

        tails.sort()
        n = len(tails)
        print(f"seq [{seq['label']}]: n_prompt={n_prompt} n_total={n_total} "
              f"tail mass mean={sum(tails)/n:.3e} p99={tails[int(0.99*(n-1))]:.3e} max={tails[-1]:.3e}")
    return ok


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    f = read_file(args[0])
    print(f"model: {f['model_desc']}")
    print(f"n_vocab={f['n_vocab']} top_k={f['top_k']} n_seq={len(f['seqs'])}")
    if "--dump-tokens" in sys.argv:
        for seq in f["seqs"]:
            print(f"[{seq['label']}] tokens: {seq['tokens']}")
    ok = check(f)
    print("all checks passed" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
