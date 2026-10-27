#!/usr/bin/env python3
"""One table across all results/sclk/sclk<level>.txt (written by scripts/sclk-sweep.sh).

    python3 scripts/sclk-summary.py
"""
import glob
import re
import statistics

LAYER = ["attn_q_a", "attn_q_b", "attn_kv", "attn_output_a", "attn_output_b", "ffn_gate_shexp", "ffn_gate_shexp",
         "ffn_down_shexp"]  # gate counted twice: up has its shape


def parse(path):
    r = {"gemv": {}, "kern": {}, "steps": {}, "power": {}}
    section = None
    for line in open(path):
        m = re.match(r"# sclk level (\d+) \((\d+)Mhz\)", line)
        if m:
            r["level"], r["mhz"] = int(m.group(1)), int(m.group(2))
        if line.startswith("== "):
            section = line[3:].strip()
        m = re.match(r"(\S+)\s+\d+\s+\d+\s+(\d)\s+exact\s+\S+\s+([\d.]+)", line)
        if m:
            r["gemv"][(m.group(1), int(m.group(2)))] = float(m.group(3))
        m = re.match(r"(.+?)\s+([\d.]+) us$", line)
        if m and section and "kernels" in section:
            r["kern"][m.group(1).strip()] = float(m.group(2))
        m = re.search(r"GPU \d: ([\d.]+) us per layer", line)
        if m:
            r["fp4"] = float(m.group(1))
        m = re.match(r"# step (\d+): ([\d.]+) ms \(cpu experts ([\d.]+), waiting for the GPU ([\d.]+)[;)]", line)
        if m:
            r["steps"].setdefault(section, []).append(float(m.group(2)))
            # the GPU's part of the step (+ copies): the CPU-expert time is noisy step to step
            r.setdefault("gpu", {}).setdefault(section, []).append(float(m.group(2)) - float(m.group(3)))
        m = re.search(r"dev0 (\d+)/(\d+)\s+dev1 (\d+)/(\d+)\s+dev2 (\d+)/(\d+)\s+dev3 (\d+)/(\d+)", line)
        if m:
            r["power"][section] = [int(x) for x in m.groups()]
    return r


runs = sorted((parse(p) for p in glob.glob("results/sclk/sclk*.txt")), key=lambda r: r.get("level", -1))
runs = [r for r in runs if "level" in r]
cols = [f"L{r['level']} {r['mhz']}" for r in runs]
print(f"{'':44s}" + "".join(f"{c:>12s}" for c in cols))


def row(name, vals, fmt="{:.1f}"):
    print(f"{name:44s}" + "".join(f"{(fmt.format(v) if v is not None else '-'):>12s}" for v in vals))


row("fp8 GEMV per layer, M=1 (us)", [sum(r["gemv"].get((t, 1), 0) for t in LAYER) for r in runs])
row("fp8 GEMV per layer, M=4 (us)", [sum(r["gemv"].get((t, 4), 0) for t in LAYER) for r in runs])
for k in ["empty kernel", "hc_split", "router top-6 of 384", "attn_decode 64 heads, 128 + 512 keys",
          "index_scores 32 x 128, 1100 keys", "topk 512 of 1100"]:
    row(k + " (us)", [r["kern"].get(k) for r in runs])
row("GPU experts, one layer (us)", [r.get("fp4") for r in runs])
for sec, label in [("decode, runtime, dense on die 0, experts on the CPU", "decode step, experts on CPU (ms)"),
                   ("decode, runtime + 13 expert layers on dies 0-3", "decode step, + 13 layers on GPUs (ms)"),
                   ("decode, runtime + experts placed on dies 0-3 (study30)", "decode step, + study30 placement (ms)")]:
    row(label, [statistics.median(r["steps"][sec]) if r["steps"].get(sec) else None for r in runs])
    row("  of which GPU + copies (step - CPU experts)", [statistics.median(r["gpu"][sec]) if r.get("gpu", {}).get(sec) else None for r in runs])
for sec, label in [("fp8 GEMV (die 0, layer 2 shapes)", "peak W dev0, fp8 GEMV bench"),
                   ("GPU experts, one layer (die 0)", "peak W dev0, GPU experts bench"),
                   ("decode, runtime + 13 expert layers on dies 0-3", "peak W dev0..3, decode + 13 layers")]:
    vals = []
    for r in runs:
        p = r["power"].get(sec)
        vals.append(None if p is None else (p[1] if "dev0..3" not in label else "/".join(str(p[i]) for i in (1, 3, 5, 7))))
    w = 18 if "dev0..3" in label else 12
    print(f"{label:44s}" + "".join(f"{(str(v) if v is not None else '-'):>{w}s}" for v in vals))
