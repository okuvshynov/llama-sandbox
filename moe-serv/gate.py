#!/usr/bin/env python3
"""The correctness gate: does our backend compute the expert block exactly as
llama.cpp would?

    python gate.py                  # the sharp test, two runs
    python gate.py --vs-stock       # also measure what owning the weights costs
    python gate.py --tol 1e-4       # allow a numerical difference (Vulkan)

Two runs of stock `llama-perplexity` over a fixed corpus, both with `-ot`
placing the routed experts in our buffer, differing only in whether we claim
any operations. Each writes its log-probabilities with `--kl-divergence-base`;
those files are a deterministic function of the logits, so

    identical files  <=>  our compute changed nothing

The harness is llama.cpp itself: no capture format, no graph rebuilt from
metadata, no second definition of the thing under test. What makes it
affordable is `make_stub.py` — the first four layers of DeepSeek-V4-Flash are
~16 GiB and load in seconds, against 150 GiB and minutes.

WHY THE CONTROL IS NOT STOCK llama.cpp

The obvious control is `-ot exps=CPU`, and it is the wrong one. llama.cpp
overrides that to **CPU_REPACK**: MXFP4 experts are rewritten into a blocked
layout at load and multiplied by a different GEMM (repo CLAUDE.md). We cannot
repack — owning the weights is the whole point, and a 137 GiB second copy is
not available — so that comparison differs in weight layout as well as in
backend, and charges the difference to this project. It is worth measuring, and
`--vs-stock` measures it, but it is not the gate.

`MOESERV_DISABLE=1` gives the control that changes exactly one thing: the
weights still land in our buffer, and llama.cpp's CPU backend computes them in
place in one unsplit graph.

ENGAGEMENT IS ASSERTED, NOT ASSUMED

Four separate checks in this project have passed while testing nothing: a
backend that was never loaded, an argument eaten by `cmd`, a pipeline that
killed the process it was measuring, and a grep for a line the tool had
filtered out. So each run must prove from its own log both where the weights
went and whether we computed, and a run that cannot prove it aborts the gate
instead of being compared.

Stdlib only, as everything else here is.
"""

import argparse
import filecmp
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Printed unconditionally by moeserv_backend_graph_compute. Unlike llama.cpp's
# own loader line it survives any verbosity setting, which matters because a
# check that greps for a filtered line reports a false negative.
ENGAGED = "MoE: first split has"

# `-cmoe`'s own regex (LLM_FFN_EXPS_REGEX) pointed at a buffer type by name. No
# '|' here: through `cmd /c` that character is a pipe and the argument never
# arrives.
OT_MOE = "exps=MoE"
OT_CPU = "exps=CPU"


def run(tag, exe, args, log_path, env_extra=None):
    env = dict(os.environ)
    # Scrubbed rather than inherited: a variable left over in the shell would
    # otherwise apply to only some runs and silently make the comparison
    # meaningless.
    for k in ("GGML_BACKEND_PATH", "MOESERV_DISABLE"):
        env.pop(k, None)
    env.update(env_extra or {})
    with open(log_path, "wb") as log:
        rc = subprocess.call([exe] + args, stdout=log, stderr=subprocess.STDOUT, env=env)
    text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    if rc != 0:
        sys.stderr.write(text[-3000:])
        raise SystemExit("%s: llama-perplexity exited %d (see %s)" % (tag, rc, log_path))
    return text


def placement(text):
    """Which buffer type the expert weights actually went to, read back from the
    run's own log rather than inferred from the arguments we passed."""
    seen = set()
    for line in text.splitlines():
        if "buffer type overridden to" in line and "_exps.weight" in line:
            seen.add(line.rsplit("buffer type overridden to", 1)[1].strip())
    return seen


def check(tag, text, want_buft, want_engaged):
    got = placement(text)
    if got != {want_buft}:
        raise SystemExit("%s: experts went to %s, expected %s — nothing was tested"
                         % (tag, sorted(got) or "no buffer we can see", want_buft))
    if (ENGAGED in text) != want_engaged:
        raise SystemExit("%s: backend %s compute a split, expected %s"
                         % (tag, "did" if ENGAGED in text else "did not",
                            "it to" if want_engaged else "it not to"))
    print("  %-6s experts -> %-11s our compute: %s"
          % (tag, want_buft, "yes" if want_engaged else "no"))


def kld_report(text):
    """Print llama.cpp's own comparison table and return the mean KLD."""
    mean = None
    for line in text.splitlines():
        t = line.split("I ")[-1].strip()
        # These labels are padded ("Mean    KLD:"), so match on the pieces
        # rather than on a spelling a format-string tweak would break.
        if "KLD:" in t or t.startswith(("Same top p:", "RMS")):
            print("    %s" % t)
        if t.startswith("Mean") and "KLD:" in t:
            try:
                mean = float(t.split("KLD:")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\llms\stub\ds4-L4.gguf")
    ap.add_argument("--llama-cpp", default=r"C:\Users\oleksandr\Desktop\llama.cpp")
    ap.add_argument("--backend", default=os.path.join(HERE, "build", "bin", "moeserv.dll"))
    ap.add_argument("--corpus", default=os.path.join(HERE, "gate_corpus.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--ctx", type=int, default=512)
    # One thread count for every run, always. ggml partitions a matmul by thread
    # count, so the summation order and therefore the last bits change with it
    # (repo CLAUDE.md); two runs at different counts would differ for a reason
    # that has nothing to do with this backend.
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max mean KLD to accept; 0 requires byte-identical logits")
    ap.add_argument("--vs-stock", action="store_true",
                    help="also measure the gap against repacked stock llama.cpp")
    args = ap.parse_args()

    exe = os.path.join(args.llama_cpp, "build", "bin", "llama-perplexity.exe")
    for path, what in ((exe, "llama-perplexity"), (args.model, "model"),
                       (args.backend, "moeserv.dll"), (args.corpus, "corpus")):
        if not os.path.exists(path):
            raise SystemExit("missing %s: %s" % (what, path))
    os.makedirs(args.out, exist_ok=True)

    # -v only so the placement lines survive to the log; it changes nothing
    # that is computed.
    base = ["-m", args.model, "-f", args.corpus, "-c", str(args.ctx),
            "-t", str(args.threads), "-tb", str(args.threads), "--no-warmup", "-v"]
    dat = lambda n: os.path.join(args.out, "gate-%s.dat" % n)
    log = lambda n: os.path.join(args.out, "gate-%s.log" % n)

    print("gate: %s" % os.path.basename(args.model))

    ctl = run("ctl", exe, base + ["-ot", OT_MOE, "--kl-divergence-base", dat("ctl")],
              log("ctl"), {"GGML_BACKEND_PATH": args.backend, "MOESERV_DISABLE": "1"})
    check("ctl", ctl, "MoE", False)

    moe = run("moe", exe, base + ["-ot", OT_MOE, "--kl-divergence-base", dat("moe")],
              log("moe"), {"GGML_BACKEND_PATH": args.backend})
    check("moe", moe, "MoE", True)

    identical = filecmp.cmp(dat("ctl"), dat("moe"), shallow=False)
    failed = 0
    if identical:
        print("\nPASS: our compute is bit-identical to llama.cpp's on the same weights")
    else:
        print("\nlogits differ from the same-placement control — measuring")
        kl = run("kld", exe, base + ["-ot", OT_MOE, "--kl-divergence",
                                     "--kl-divergence-base", dat("ctl")],
                 log("kld"), {"GGML_BACKEND_PATH": args.backend})
        mean = kld_report(kl)
        if args.tol > 0.0 and mean is not None and mean <= args.tol:
            print("\nPASS: mean KLD %.3e within tolerance %.3e" % (mean, args.tol))
        else:
            print("\nFAIL: expected byte-identical logits"
                  if args.tol <= 0.0 else "\nFAIL: outside tolerance")
            failed = 1

    if args.vs_stock:
        # Not a pass/fail: llama.cpp repacks MXFP4 experts and we cannot, so a
        # difference here is expected and its size is the thing worth knowing.
        # It doubles as the proof that the comparison above can fail at all —
        # the same byte comparison, on a run that genuinely differs.
        print("\nvs stock llama.cpp (experts repacked, ours cannot be):")
        st = run("stock", exe, base + ["-ot", OT_CPU, "--kl-divergence-base", dat("stock")],
                 log("stock"))
        check("stock", st, "CPU_REPACK", False)
        if filecmp.cmp(dat("stock"), dat("moe"), shallow=False):
            print("  identical — which contradicts the repack, so distrust this harness")
            failed = 1
        else:
            kl = run("kld2", exe, base + ["-ot", OT_MOE, "--kl-divergence",
                                          "--kl-divergence-base", dat("stock")],
                     log("kld2"), {"GGML_BACKEND_PATH": args.backend})
            kld_report(kl)

    return failed


if __name__ == "__main__":
    sys.exit(main())
