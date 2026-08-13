#!/usr/bin/env python3
"""Decode throughput, CPU only, for the three configurations the gate compares.

    python bench.py                       # ~20 min on DeepSeek-V4-Flash
    python bench.py --loads 3 --n 64

Same three configurations as `gate.py`, so the performance story and the
correctness story describe the same runs:

    stock      llama.cpp as shipped        experts repacked, one graph
    ours-off   -ot exps=MoE, claims off    experts not repacked, one graph
    ours-on    -ot exps=MoE               experts not repacked, our split

Read as two differences rather than three numbers. `stock -> ours-off` is what
llama.cpp's MXFP4 repack is worth, which we forfeit by owning the weights and
cannot get back. `ours-off -> ours-on` is what the extra split boundary costs,
which is ours to fix. Quoting only `stock -> ours-on` would fold a compile-time
kernel choice into an architectural claim.

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
- **Decode only.** `-p 0`. Prefill and decode are bound by different things and
  a table mixing them invites an average of the two.

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
    ap.add_argument("--backend", default=os.path.join(HERE, "build", "bin", "moeserv.dll"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n", type=int, default=32, help="tokens to generate")
    ap.add_argument("--loads", type=int, default=2, help="separate loads per configuration")
    args = ap.parse_args()

    exe = os.path.join(args.llama_cpp, "build", "bin", "llama-bench.exe")
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
    common = ["-t", str(args.threads), "-r", str(args.reps),
              "-p", "0", "-n", str(args.n), "-lm", "none", "-v"]

    print("bench: %s" % os.path.basename(args.model))
    print("  %s\n" % " ".join(common))

    results = {c["tag"]: [] for c in configs}
    for load in range(1, args.loads + 1):
        for c in configs:
            log = os.path.join(args.out, "bench-%s-%d.log" % (c["tag"], load))
            rows = run(c, exe, args.model, common, log)
            for test, val, sd in rows:
                # flush: a 20-minute run is usually redirected to a file, and
                # Python buffers stdout when it is not a terminal — without this
                # the whole log appears at the end and progress is invisible.
                print("  load %d  %-9s %-6s %7.2f +- %.2f" % (load, c["tag"], test, val, sd),
                      flush=True)
                results[c["tag"]].append(val)

    print("\n%-10s %-28s %s" % ("config", "per load (t/s)", "mean"))
    means, spreads = {}, {}
    for c in configs:
        vals = results[c["tag"]]
        means[c["tag"]] = statistics.fmean(vals)
        spreads[c["tag"]] = ((max(vals) - min(vals)) / means[c["tag"]] * 100.0
                             if len(vals) > 1 else float("inf"))
        print("%-10s %-28s %6.2f   (load-to-load %.1f%%)"
              % (c["tag"], "  ".join("%.2f" % v for v in vals),
                 means[c["tag"]], spreads[c["tag"]]))

    # A delta is only reported as a delta when it is larger than the noise this
    # same run measured. Printing "-4.0%" next to "load-to-load 8.7%" invites
    # exactly the mistake the two-load rule exists to prevent, and the first
    # version of this script did precisely that: on the run that produced these
    # numbers, every configuration's per-load values interleaved, load 1 and
    # load 2 ranked them in different orders, and the summary still printed
    # three confident percentages.
    print()
    unresolved = 0
    for label, a, b in (("repack           stock -> ours-off ", "stock", "ours-off"),
                        ("our split        ours-off -> ours-on", "ours-off", "ours-on"),
                        ("net vs stock     stock -> ours-on  ", "stock", "ours-on")):
        delta = (means[b] / means[a] - 1.0) * 100.0
        noise = max(spreads[a], spreads[b])
        if abs(delta) < noise:
            unresolved += 1
            print("  %s: NOT RESOLVED (%+.1f%% vs %.1f%% noise)" % (label, delta, noise))
        else:
            print("  %s: %+.1f%%" % (label, delta))

    if unresolved:
        print("\n%d of 3 comparisons are below this run's own noise floor." % unresolved)
        print("More loads help only as sqrt(n): separating a 4% effect through an")
        print("8% spread needs ~30 loads, which is hours. Measure the per-split")
        print("cost on a small model instead, where the spread is ~0.3%, and")
        print("scale it by the split count -- see PLAN.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
