#!/usr/bin/env python3
"""Pivot mm-split logs: best e2e time (min of serial/threaded issue) per N x config.

    python3 summarize.py results/split-q4-l40-sclk2.txt [--field e2e|comp|slice] [--speedup]

Prints one table per (tensor, mode). Any row that is not "ok" is marked with '!'
and listed at the end, so a failed check can't hide in a table of numbers.
"""
import argparse
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^  (\d[\d,]*)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+(\S+)\s+(\S+)\s+(.*)$")


def parse(path):
    tables = defaultdict(dict)  # (tensor, mode) -> {(cfg, N): row}
    tensor = mode = None
    for line in open(path):
        if line.startswith("== "):
            tensor = line.split()[1].split(".")[2]
        elif line.strip().startswith("-- mode"):
            mode = line.split()[2]
        else:
            m = ROW.match(line.rstrip("\n"))
            if m and tensor:
                cfg, n = m.group(1), int(m.group(2))
                tables[(tensor, mode)][(cfg, n)] = {
                    "slice": float(m.group(3)),
                    "comp": min(float(m.group(4)), float(m.group(5))),
                    "e2e": min(float(m.group(6)), float(m.group(7))),
                    "ok": m.group(9) == "ok",
                    "vs": m.group(11).strip(),
                }
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--field", default="e2e", choices=["e2e", "comp", "slice"])
    ap.add_argument("--speedup", action="store_true", help="show 1-die time / config time instead of us")
    a = ap.parse_args()
    bad = []
    for path in a.files:
        print(f"### {path}  ({a.field}{', speedup vs 1 die' if a.speedup else ', us'})")
        tables = parse(path)
        for (tensor, mode), rows in tables.items():
            # 1-die rows exist only in row mode; k mode borrows them as its baseline
            base_rows = tables.get((tensor, "row"), rows)
            base = {n: r for (c, n), r in base_rows.items() if "," not in c}
            cfgs = sorted({c for c, _ in base_rows} | {c for c, _ in rows}, key=lambda c: (c.count(","), c))
            ns = sorted({n for _, n in rows})
            print(f"\n{tensor} / {mode}")
            print("  N     " + "".join(f"{c:>12s}" for c in cfgs))
            for n in ns:
                cells = []
                for c in cfgs:
                    r = base.get(n) if "," not in c else rows.get((c, n))
                    if r is None or n not in base:
                        cells.append(f"{'-':>12s}")
                        continue
                    v = base[n][a.field] / r[a.field] if a.speedup else r[a.field]
                    mark = "" if r["ok"] else "!"
                    cells.append(f"{(f'{v:.2f}x' if a.speedup else f'{v:.0f}') + mark:>12s}")
                    if not r["ok"]:
                        bad.append(f"{path}: {tensor}/{mode} dies={c} N={n}: {r['vs']}")
                print(f"  {n:<6d}" + "".join(cells))
        print()
    if bad:
        print("NOT OK rows:")
        for b in sorted(set(bad)):
            print("  " + b)
    else:
        print("all rows ok")


if __name__ == "__main__":
    sys.exit(main())
