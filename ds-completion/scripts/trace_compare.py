#!/usr/bin/env python3
"""Two --trace files side by side: each span's total time in generation, per verify step.

    ~/venvs/ds-ref/bin/python scripts/trace_compare.py OLD.json[.gz] NEW.json[.gz]

Generation starts at the first verify step; span names lose their " L<layer>" suffix and
"layer N" spans count as "layer".
"""
import collections
import gzip
import json
import re
import sys


def spans(path):
    with (gzip.open(path) if path.endswith(".gz") else open(path)) as f:
        d = json.load(f)
    ev = [e for e in (d["traceEvents"] if isinstance(d, dict) else d) if e.get("ph") == "X"]
    verify = [e for e in ev if e["name"].startswith("verify")]
    t0 = min(e["ts"] for e in verify)
    tot = collections.Counter()
    for e in ev:
        if e["ts"] >= t0:
            n = re.sub(r" L\d+$", "", e["name"])
            tot["layer" if n.startswith("layer ") else n] += e["dur"]
    return len(verify), tot


def main():
    (n0, a), (n1, b) = spans(sys.argv[1]), spans(sys.argv[2])
    print(f"# ms per verify step in generation: {sys.argv[1]} ({n0} steps) vs {sys.argv[2]} ({n1} steps)")
    print(f"{'span':34s} {'old':>9s} {'new':>9s} {'change':>9s}")
    for n in sorted(set(a) | set(b), key=lambda n: -a[n]):
        x, y = a[n] / 1e3 / n0, b[n] / 1e3 / n1
        if max(x, y) >= 0.5:
            print(f"{n[:34]:34s} {x:9.2f} {y:9.2f} {y - x:+9.2f}")


if __name__ == "__main__":
    main()
