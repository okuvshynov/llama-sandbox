#!/usr/bin/env python3
"""Throughput for the three configurations the gate compares.

    python bench.py                                   # decode, CPU-only host
    python bench.py --pp 128 --build-dir build-vk     # prefill too, on the dies

Same three configurations as `gate.py`, so the performance story and the
correctness story describe the same runs:

    stock      llama.cpp as shipped        experts repacked, one graph
    ours-off   -ot exps=MoE, claims off    experts not repacked, one graph
    ours-on    -ot exps=MoE               experts not repacked, ours to compute

Read as two differences rather than three numbers. `stock -> ours-off` is what
llama.cpp's MXFP4 repack is worth, which we forfeit by owning the weights and
cannot get back. `ours-off -> ours-on` is what our compute is worth — the extra
split boundary on the CPU-only host, the four Vega II dies on the Vulkan one.
Quoting only `stock -> ours-on` would fold a compile-time kernel choice into an
architectural claim.

MEASUREMENT DISCIPLINE, ALL OF IT LEARNED HERE (repo CLAUDE.md)

- **`-lm none`**, so weights land in ordinary allocated memory. A single
  mmap-backed timing on Windows is worthless: 1.84 ± 0.02 t/s and 1.04 ± 0.29
  came from the same binary and model minutes apart, differing only in the
  standby list.
- **Every configuration on more than one load.** `-r 5` reports the spread of
  five repetitions *inside* one process, and the load-to-load spread is the
  larger number — 10% against within-run 2-6% on this model. A single load's
  ± is not an error bar for a comparison between configurations.
- **Round-robin, not blocked.** Load 1 of every configuration, then load 2, so
  drift in the machine's state hits all of them alike instead of the one that
  happened to run last.
- **Prefill and decode reported apart**, never averaged: they are bound by
  different things, and for this project by opposite ones. A decode step reads
  6 of 256 experts, so our block is ~20% of its bytes; a 512-token prefill
  touches nearly all of them.

Each run must also prove from its own log where the expert weights went and
whether we computed, and a run that cannot aborts rather than being tabulated.

Stdlib only, as everything else here is.
"""

import argparse
import os
import re
import subprocess
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENGAGED = "MoE: first split has"

# t/s row from llama-bench's markdown table: "| ... | tg32 | 3.73 ± 0.05 |"
ROW = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|")


def placement(text):
    return {line.rsplit("buffer type overridden to", 1)[1].strip()
            for line in text.splitlines()
            if "buffer type overridden to" in line and "_exps.weight" in line}


def run(cfg, exe, model, args, log_path):
    env = dict(os.environ)
    for k in ("GGML_BACKEND_PATH", "MOESERV_DISABLE"):
        env.pop(k, None)
    env.update(cfg["env"])
    cmd = [exe, "-m", model] + args + cfg["args"]
    with open(log_path, "wb") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    if rc != 0:
        sys.stderr.write(text[-3000:])
        raise SystemExit("%s: llama-bench exited %d (see %s)" % (cfg["tag"], rc, log_path))

    got = placement(text)
    if got != cfg["buft"]:
        raise SystemExit("%s: experts went to %s, expected %s — nothing was measured"
                         % (cfg["tag"], sorted(got) or "the default", sorted(cfg["buft"]) or "the default"))
    if (ENGAGED in text) != cfg["engaged"]:
        raise SystemExit("%s: backend %s compute a split, expected the opposite"
                         % (cfg["tag"], "did" if ENGAGED in text else "did not"))

    rows = [(t, float(v), float(sd)) for t, v, sd in ROW.findall(text)]
    if not rows:
        raise SystemExit("%s: no result row in %s" % (cfg["tag"], log_path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=r"D:\llms\ds-v4-flash\UD-Q8_K_XL"
                            r"\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf")
    ap.add_argument("--llama-cpp", default=r"C:\Users\oleksandr\Desktop\llama.cpp")
    ap.add_argument("--build-dir", default="build",
                    help="'build' is the CPU-only baseline host, 'build-vk' has Vulkan")
    ap.add_argument("--backend", default=os.path.join(HERE, "build", "bin", "moeserv.dll"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    # Strings, so llama-bench's own comma syntax works: --pp 8,128,512 sweeps
    # batch size in one load, which is how a per-dispatch cost is told apart
    # from a per-byte one.
    ap.add_argument("--n", default="32", help="tokens to generate (0 skips decode)")
    ap.add_argument("--pp", default="0", help="prompt tokens (0 skips prefill)")
    # Explicit, always. llama-bench defaults -ngl to 99, so on a Vulkan-enabled
    # host every configuration silently offloads the trunk as well — and then
    # `stock` is not a CPU baseline, `ours-off` has attention on the GPU, and the
    # three rows differ in much more than who computes the expert block. Leaving
    # it at 0 isolates the one thing this benchmark is about. `--ngl 99` asks a
    # different and also interesting question (what is the best end-to-end
    # configuration), and the answer belongs in its own table.
    ap.add_argument("--ngl", type=int, default=0, help="layers llama.cpp offloads itself")
    ap.add_argument("--loads", type=int, default=2, help="separate loads per configuration")
    args = ap.parse_args()

    exe = os.path.join(args.llama_cpp, args.build_dir, "bin", "llama-bench.exe")
    for path, what in ((exe, "llama-bench"), (args.model, "model"),
                       (args.backend, "moeserv.dll")):
        if not os.path.exists(path):
            raise SystemExit("missing %s: %s" % (what, path))
    os.makedirs(args.out, exist_ok=True)

    configs = [
        {"tag": "stock",    "args": [],
         "env": {}, "buft": set(), "engaged": False},
        {"tag": "ours-off", "args": ["-ot", "exps=MoE"],
         "env": {"GGML_BACKEND_PATH": args.backend, "MOESERV_DISABLE": "1"},
         "buft": {"MoE"}, "engaged": False},
        {"tag": "ours-on",  "args": ["-ot", "exps=MoE"],
         "env": {"GGML_BACKEND_PATH": args.backend},
         "buft": {"MoE"}, "engaged": True},
    ]
    # -v so the placement lines survive to the log; it changes nothing measured.
    common = ["-t", str(args.threads), "-r", str(args.reps), "-ngl", str(args.ngl),
              "-p", args.pp, "-n", args.n, "-lm", "none", "-v"]
    if args.build_dir != "build":
        # See gate.py: on a Vulkan-enabled build this is the difference between
        # measuring us and measuring op_offload's graph fragmentation.
        common += ["-nopo", "1"]

    print("bench: %s" % os.path.basename(args.model))
    print("  %s\n" % " ".join(common))

    # Keyed by (config, test) — llama-bench emits one row per test, and folding
    # a pp row into a tg mean would average two things bound by different
    # hardware limits.
    results = {}
    tests = []
    for load in range(1, args.loads + 1):
        for c in configs:
            log = os.path.join(args.out, "bench-%s-%d.log" % (c["tag"], load))
            rows = run(c, exe, args.model, common, log)
            for test, val, sd in rows:
                if test not in tests:
                    tests.append(test)
                # flush: a long run is usually redirected to a file, and Python
                # buffers stdout when it is not a terminal — without this the
                # whole log appears at the end and progress is invisible.
                print("  load %d  %-9s %-6s %7.2f +- %.2f" % (load, c["tag"], test, val, sd),
                      flush=True)
                results.setdefault((c["tag"], test), []).append(val)

    for test in tests:
        print("\n%-6s %-10s %-28s %s" % (test, "config", "per load (t/s)", "mean"))
        means, spreads = {}, {}
        for c in configs:
            vals = results.get((c["tag"], test), [])
            if not vals:
                continue
            means[c["tag"]] = statistics.fmean(vals)
            spreads[c["tag"]] = ((max(vals) - min(vals)) / means[c["tag"]] * 100.0
                                 if len(vals) > 1 else float("inf"))
            print("%-6s %-10s %-28s %6.2f   (load-to-load %.1f%%)"
                  % ("", c["tag"], "  ".join("%.2f" % v for v in vals),
                     means[c["tag"]], spreads[c["tag"]]))

        # A delta is only reported as a delta when it is larger than the noise
        # this same run measured. Printing "-4.0%" next to "load-to-load 8.7%"
        # invites exactly the mistake the two-load rule exists to prevent, and
        # the first version of this script did precisely that: every
        # configuration's per-load values interleaved, the two loads ranked them
        # in different orders, and the summary printed three confident
        # percentages anyway.
        print()
        unresolved = 0
        pairs = (("repack        stock -> ours-off ", "stock", "ours-off"),
                 ("our compute   ours-off -> ours-on", "ours-off", "ours-on"),
                 ("net vs stock  stock -> ours-on  ", "stock", "ours-on"))
        for label, a, b in pairs:
            if a not in means or b not in means:
                continue
            delta = (means[b] / means[a] - 1.0) * 100.0
            noise = max(spreads[a], spreads[b])
            if abs(delta) < noise:
                unresolved += 1
                print("  %s: NOT RESOLVED (%+.1f%% vs %.1f%% noise)" % (label, delta, noise))
            else:
                print("  %s: %+.1f%%" % (label, delta))

        if unresolved:
            print("  %d of %d comparisons are below this run's own noise floor."
                  % (unresolved, len(pairs)))
            print("  More loads help only as sqrt(n). Measure on the stub instead,")
            print("  where the spread is ~0.3%, and scale what transfers -- see PLAN.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
