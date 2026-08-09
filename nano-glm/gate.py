#!/usr/bin/env python3
"""nano-glm correctness gate: named tests, provenance-checked, stdlib only.

    python gate.py                      # smoke (the default)
    python gate.py aa
    python gate.py rpc llamacpp
    python gate.py all
    python gate.py --update-golden      # re-derive testdata/ from scratch

  name       needs             cost    what it establishes
  --------------------------------------------------------------------------
  smoke      model             ~2 min  one 14-token prompt, 18 positions,
                                       byte-identical to the golden
  aa         model            ~15 min  the 5-prompt corpus, byte-identical
  rpc        model + server   ~20 min  the corpus through moe-server equals
                                       the local path, byte for byte
  llamacpp   model + rescore  ~40 min  the corpus re-derived by llama.cpp
                                       agrees at KL == 0 — and *creates* the
                                       golden set

Two questions get conflated whenever a gate is assembled by hand, and they want
opposite responses to a legitimate change. `smoke`, `aa` and `rpc` answer "did
my change alter the output?" — fixed reference, byte comparison, a failure
means *I* did something. `llamacpp` answers "is nano-glm still correct?" — an
independent implementation over the same token ids, a failure means the port is
wrong. Only the first kind is ever re-baselined.

Three setups must agree: llama.cpp (A), nano-glm local (B), nano-glm over RPC
(C). Byte equality is transitive, so two edges always suffice, and B is the hub
because it is the cheapest to reproduce: `llamacpp` is A-B, `rpc` is B-C, and
A-C follows without ever being run.

Why provenance is checked before anything is compared: every confusing hour on
this project has come from a configuration difference wearing the costume of a
code difference — two corrupt model shards, `-t 16` vs `-t 32`, MSVC vs Apple
clang. A reference that cannot state the configuration it was made under is not
a reference; `results/corpus/` had to be thrown away for exactly that, because
nothing in it recorded that it predated the shard repair. So the golden set
carries a fingerprint and this script refuses to compare when it disagrees,
rather than reporting a difference that is not yours.

What counts as a refusal is deliberately narrow — only what changes the bytes
for reasons unrelated to your code:

    compiler, ggml_commit, blas, llamafile, n_threads, model shard size/mtime

`git_rev` is NOT one of them: running at a different git_rev is the entire
point of the gate. `trace` is not either — a -DNANO_EXPERT_TRACE build was
measured byte-identical to a plain one, so it only warns.
"""

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TESTDATA = os.path.join(HERE, "testdata")
LOGIT_KLD = os.path.join(REPO, "logit-kld")
EXE = ".exe" if os.name == "nt" else ""


def lkld_read_file():
    """logit-kld's reference reader for the lkldtopk format.

    Loaded by path rather than by putting logit-kld/ on sys.path: that file is
    named inspect.py, and shadowing the stdlib `inspect` module would break
    whichever library happens to import it — or fail here, depending on which
    got imported first. Loading it under an explicit name has neither problem.
    """
    path = os.path.join(LOGIT_KLD, "inspect.py")
    spec = importlib.util.spec_from_file_location("lkld_inspect", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_file


TESTS = ("smoke", "aa", "rpc", "llamacpp")

# Everything here needs the model. Unit-level invariants — the Hadamard matrix
# being orthonormal, wire-header layout, the routing statistics — belong in a
# C++ test target instead of a flag on the shipping binary; PLAN.md step 10.

# Disagreement here makes a byte comparison meaningless; see the docstring for
# why git_rev and trace are absent.
STRICT_FIELDS = ("compiler", "ggml_commit", "blas", "llamafile", "n_threads")
WARN_FIELDS = ("git_rev", "trace", "ggml_version", "cores_phys", "cores_log")

DEFAULT_MODEL = os.environ.get(
    "NANO_MODEL", r"D:\llms\UD-Q6_K\GLM-5.2-UD-Q6_K-00001-of-00014.gguf")

SMOKE = ["smoke"]
CORPUS = ["01_prose", "02_code", "03_math", "04_history", "05_french"]


class Fail(Exception):
    pass


# ---------------------------------------------------------------------------
# helpers

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log_path):
    """Run to completion, tee to a file, return the text.

    A file rather than a pipe-and-truncate: a long run's interesting line is as
    likely to be in the middle as at the end, and rerunning a 15-minute gate to
    recover a line you discarded is a bad trade (repo CLAUDE.md).
    """
    with open(log_path, "w") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    with open(log_path, "r", errors="replace") as f:
        out = f.read()
    if p.returncode != 0:
        sys.stdout.write(out[-2000:])
        raise Fail("%s exited %d (full log: %s)"
                   % (os.path.basename(cmd[0]), p.returncode, log_path))
    return out


def binary_fingerprint(exe):
    r = subprocess.run([exe, "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail("%s --version failed; is this a build with lib/build_info.h?" % exe)
    fp = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fp[k.strip()] = v.strip()
    return fp


SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def model_fingerprint(model_path):
    """Shard sizes and mtimes.

    Not a hash: 583 GiB per gate run is unaffordable, and sampling a few MiB
    would have missed the ~1.9 KB of real corruption we hit. This does not
    prove the bytes are good — `checksums/` does that, occasionally — but it
    detects "these are not the files the reference was made with", which is the
    failure that actually bit us: the repair replaced two shards, so a stale
    reference now announces itself instead of being silently incomparable.

    The count is asserted against the total the filename declares. A first
    attempt matched shards with `name.split("-0000")[0]`, which silently found
    9 of 14 — `-00001` through `-00009` contain the literal `-0000` and
    `-00010` onward do not. An undercount is invisible: the fingerprint still
    looks plausible and still compares equal to itself. `-of-000NN` says how
    many there should be, so there is no excuse for guessing.
    """
    d = os.path.dirname(model_path) or "."
    base = os.path.basename(model_path)
    m = SHARD_RE.match(base)
    if not m:
        st = os.stat(model_path)
        return {"dir": d, "first": base,
                "shards": [{"name": base, "size": st.st_size, "mtime": int(st.st_mtime)}]}

    stem, total = m.group("stem"), int(m.group("total"))
    shards = []
    for name in sorted(os.listdir(d)):
        s = SHARD_RE.match(name)
        if s and s.group("stem") == stem and int(s.group("total")) == total:
            st = os.stat(os.path.join(d, name))
            shards.append({"name": name, "size": st.st_size, "mtime": int(st.st_mtime)})
    if len(shards) != total:
        raise Fail("found %d shards for %s but the name declares %d — incomplete model?"
                   % (len(shards), stem, total))
    return {"dir": d, "first": base, "shards": shards}


def live_git_rev():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# provenance

def collect_provenance(exe, model_path, threads):
    return {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "git_rev_live": live_git_rev(),
        "n_threads": threads,
        "binary": binary_fingerprint(exe),
        "model": model_fingerprint(model_path),
    }


def check_provenance(golden, current, allow_drift):
    problems, warnings = [], []

    gb, cb = golden.get("binary", {}), current["binary"]
    for k in STRICT_FIELDS:
        if gb.get(k) != cb.get(k):
            problems.append("%s: golden %r, now %r" % (k, gb.get(k), cb.get(k)))
    for k in WARN_FIELDS:
        if gb.get(k) != cb.get(k):
            warnings.append("%s: golden %r, now %r" % (k, gb.get(k), cb.get(k)))
    if golden.get("n_threads") != current["n_threads"]:
        problems.append("n_threads: golden %r, now %r"
                        % (golden.get("n_threads"), current["n_threads"]))

    # Compare shards by name, not path: the same model sits elsewhere on the
    # other machine and that is fine.
    gm = {s["name"]: s for s in golden.get("model", {}).get("shards", [])}
    cm = {s["name"]: s for s in current["model"]["shards"]}
    if set(gm) != set(cm):
        problems.append("shard set differs: golden %d files, now %d" % (len(gm), len(cm)))
    for name in sorted(set(gm) & set(cm)):
        if gm[name]["size"] != cm[name]["size"]:
            problems.append("%s: size %d -> %d" % (name, gm[name]["size"], cm[name]["size"]))
        elif gm[name]["mtime"] != cm[name]["mtime"]:
            problems.append("%s: mtime changed at the same size — replaced or repaired?" % name)

    if golden.get("host") != current["host"]:
        warnings.append("host: golden %r, now %r" % (golden.get("host"), current["host"]))

    if problems and allow_drift:
        warnings = ["ALLOWED DRIFT: " + p for p in problems] + warnings
        problems = []
    return problems, warnings


# ---------------------------------------------------------------------------
# the tests

def load_prompts():
    with open(os.path.join(TESTDATA, "prompts.json")) as f:
        return json.load(f)


def positions_equal(a, b):
    """Exact equality of two lkldtopk files, via the format's reference reader."""
    read_file = lkld_read_file()
    fa, fb = read_file(a), read_file(b)
    if len(fa["seqs"]) != len(fb["seqs"]):
        return False, "sequence count differs"
    for sa, sb in zip(fa["seqs"], fb["seqs"]):
        if sa["tokens"] != sb["tokens"]:
            return False, "token sequences differ"
        if len(sa["positions"]) != len(sb["positions"]):
            return False, "position count differs"
        for i, (pa, pb) in enumerate(zip(sa["positions"], sb["positions"])):
            if (pa["ids"] != pb["ids"] or pa["logits"] != pb["logits"]
                    or pa["max_logit"] != pb["max_logit"] or pa["lse_rest"] != pb["lse_rest"]):
                return False, "position %d differs" % i
    return True, "%d positions identical" % len(fa["seqs"][0]["positions"])


def nano_run(ctx, spec, out_bin, log, extra=()):
    cmd = [ctx.exe, "-m", ctx.model,
           "-T", ",".join(str(t) for t in spec["tokens"]),
           "-n", str(spec["n_predict"]),
           "-t", str(ctx.threads),
           "-o", out_bin] + list(extra)
    return run(cmd, log)


def run_prompts(ctx, names, via_rpc=False, rescore=False, update=False):
    failures = []
    for name in names:
        spec = ctx.prompts[name]
        tag = name + (".rpc" if via_rpc else "")
        out_bin = os.path.join(ctx.work, tag + ".bin")
        print("  [%s] %d prompt tokens + %d generated"
              % (tag, len(spec["tokens"]), spec["n_predict"]))

        # --strict always, never optional: the client defaults to lenient so
        # that a deliberately mixed pairing (Q4_K experts, Q6_K trunk) can run,
        # and the guarantee that a *gate* run is never lenient belongs here
        # rather than in the binary's default.
        extra = ["--moe-addr", ctx.moe_addr, "--strict"] if via_rpc else []
        nano_run(ctx, spec, out_bin, os.path.join(ctx.work, tag + ".log"), extra)
        digest = sha256(out_bin)

        gold = os.path.join(TESTDATA, name + ".bin")
        if update:
            # Staged, not written yet: a divergence found by the rescore below
            # must not leave testdata/ half-replaced with no provenance beside
            # it, which is a worse state than either old or new.
            print("        staged, sha256 %s..." % digest[:16])
            ctx.staged.append((out_bin, gold))
            ctx.runs[name] = {"n_prompt": len(spec["tokens"]),
                              "n_predict": spec["n_predict"],
                              "bytes": os.path.getsize(out_bin), "sha256": digest}
        elif not os.path.exists(gold):
            failures.append("%s: no golden file — run --update-golden" % name)
            print("        NO GOLDEN")
            continue
        elif sha256(gold) == digest:
            print("        bytes identical to golden")
        else:
            ok, detail = positions_equal(gold, out_bin)
            print("        BYTES DIFFER (%s)" % detail)
            failures.append("%s: differs from golden (%s)" % (name, detail))

        if rescore:
            ref = os.path.join(ctx.work, name + ".ref.bin")
            run([ctx.rescore, "-m", ctx.model, "-i", out_bin, "--sim-gen",
                 "-t", str(ctx.threads), "-o", ref],
                os.path.join(ctx.work, name + ".ref.log"))
            ok, detail = positions_equal(out_bin, ref)
            print("        vs llama.cpp rescore: %s (%s)"
                  % ("KL == 0" if ok else "DIVERGES", detail))
            if ok:
                if update:
                    ctx.staged.append((ref, os.path.join(TESTDATA, name + ".ref.bin")))
                    ctx.runs[name]["independent"] = "llama.cpp rescore --sim-gen, KL == 0"
            else:
                failures.append("%s: diverges from llama.cpp (%s)" % (name, detail))
    return failures


def start_server(ctx):
    """Spawn a local moe-server and wait for it to accept connections.

    Stopping it again is the part worth care: a process holding 583 GiB of
    views takes minutes to unmap, and until it does the exe stays locked and a
    rebuild fails with a link error that looks like anything but this (repo
    CLAUDE.md). So the gate waits, and says so if it has to.
    """
    log_path = os.path.join(ctx.work, "moe-server.log")
    log = open(log_path, "w")
    host, port = ctx.moe_addr.rsplit(":", 1)
    proc = subprocess.Popen([ctx.server, "-m", ctx.model, "--host", host,
                             "--port", port, "-t", str(ctx.threads)],
                            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise Fail("moe-server exited during startup (see %s)" % log_path)
        with open(log_path, errors="replace") as f:
            if "listening on" in f.read():
                print("        moe-server up on %s" % ctx.moe_addr)
                return proc, log
        time.sleep(0.5)
    proc.kill()
    raise Fail("moe-server did not come up within 300s (see %s)" % log_path)


def stop_server(proc, log):
    proc.terminate()
    t0 = time.time()
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        print("        WARNING: moe-server still unmapping after 600s; "
              "a rebuild will fail with LNK1104 until it exits")
        proc.kill()
    else:
        dt = time.time() - t0
        if dt > 5:
            print("        moe-server took %.0fs to unmap and exit" % dt)
    log.close()


def test_rpc(ctx, names):
    """The corpus through moe-server must equal the local path, byte for byte."""
    proc = log = None
    if not ctx.external_server:
        proc, log = start_server(ctx)
    try:
        return run_prompts(ctx, names, via_rpc=True)
    finally:
        if proc is not None:
            stop_server(proc, log)


# ---------------------------------------------------------------------------
# main

class Ctx:
    pass


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", default=["smoke"],
                    help="any of: " + ", ".join(TESTS) + ", all  (default: smoke)")
    ap.add_argument("--update-golden", action="store_true",
                    help="run llamacpp over every prompt and rewrite testdata/")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--build", default=os.path.join(HERE, "build", "bin"))
    ap.add_argument("--logit-kld-build", default=os.path.join(LOGIT_KLD, "build", "bin"))
    ap.add_argument("--threads", type=int, default=None,
                    help="default: whatever the golden used, else the binary's default")
    ap.add_argument("--moe-addr", default=None,
                    help="use an already-running backend instead of spawning one "
                         "(this is how the same test runs against another machine)")
    ap.add_argument("--only", default=None,
                    help="comma-separated prompt names to restrict any test to, e.g. "
                         "--only smoke. For iterating: `rpc` over six prompts is 20 "
                         "minutes, over one it is two.")
    ap.add_argument("--work", default=os.path.join(HERE, "results", "gate"))
    ap.add_argument("--allow-drift", action="store_true",
                    help="downgrade provenance refusals to warnings")
    args = ap.parse_args()

    names = list(dict.fromkeys(args.tests))
    if "all" in names:
        names = list(TESTS)
    if args.update_golden:
        names = ["llamacpp"]
    for n in names:
        if n not in TESTS:
            raise Fail("unknown test %r; choose from %s, all" % (n, ", ".join(TESTS)))

    # abspath, not just join: Windows CreateProcess will not resolve a relative
    # path written with forward slashes, so `--build build/bin` would fail with
    # a bare "cannot find the file specified" while the file plainly exists.
    def tool(d, name):
        return os.path.abspath(os.path.join(d, name + EXE))

    ctx = Ctx()
    ctx.exe = tool(args.build, "nano-glm")
    ctx.server = tool(args.build, "moe-server")
    ctx.rescore = tool(args.logit_kld_build, "rescore")
    ctx.model = args.model
    ctx.work = args.work
    ctx.runs = {}
    ctx.staged = []          # (src, dst) pairs, copied into testdata/ only on success
    ctx.external_server = args.moe_addr is not None
    ctx.moe_addr = args.moe_addr or "127.0.0.1:5711"
    if not os.path.exists(ctx.exe):
        raise Fail("no nano-glm at %s (pass --build)" % ctx.exe)
    os.makedirs(ctx.work, exist_ok=True)
    os.makedirs(TESTDATA, exist_ok=True)
    ctx.prompts = load_prompts()

    prov_path = os.path.join(TESTDATA, "provenance.json")
    golden = {}
    if os.path.exists(prov_path):
        with open(prov_path) as f:
            golden = json.load(f)

    ctx.threads = args.threads
    if ctx.threads is None:
        ctx.threads = golden.get("n_threads") or int(binary_fingerprint(ctx.exe)["n_threads"])

    print("=" * 74)
    print("nano-glm gate: %s%s" % (" ".join(names),
                                   "  [--update-golden]" if args.update_golden else ""))
    fp = binary_fingerprint(ctx.exe)
    print("  binary   %s" % ctx.exe)
    print("  build    %s / %s / ggml %s"
          % (fp.get("git_rev"), fp.get("compiler"), fp.get("ggml_commit")))

    current = collect_provenance(ctx.exe, ctx.model, ctx.threads)
    print("  model    %s (%d shards)"
          % (current["model"]["first"], len(current["model"]["shards"])))
    print("  threads  %d" % ctx.threads)

    if golden and not args.update_golden:
        problems, warnings = check_provenance(golden, current, args.allow_drift)
        for w in warnings:
            print("  note     %s" % w)
        if problems:
            print("\nREFUSED: the golden set was made under a different configuration,")
            print("so comparing against it would measure that, not your change.")
            for p in problems:
                print("  - %s" % p)
            print("\nRe-derive with --update-golden, or --allow-drift to compare anyway.")
            return 2
    elif not golden:
        print("  note     no golden set yet — run --update-golden to create one")

    only = args.only.split(",") if args.only else None
    if only:
        for p in only:
            if p not in ctx.prompts:
                raise Fail("unknown prompt %r; have %s" % (p, ", ".join(ctx.prompts)))
        if args.update_golden:
            # A partial golden set would pair a fresh provenance record with
            # stale files it does not describe — the exact defect this whole
            # harness exists to prevent.
            raise Fail("--only cannot be combined with --update-golden")
        print("  only     %s" % " ".join(only))

    def prompts_for(default):
        return only if only else default

    failures = []
    for name in names:
        print()
        if name == "smoke":
            failures += run_prompts(ctx, prompts_for(SMOKE))
        elif name == "aa":
            failures += run_prompts(ctx, prompts_for(SMOKE + CORPUS))
        elif name == "rpc":
            failures += test_rpc(ctx, prompts_for(SMOKE + CORPUS))
        elif name == "llamacpp":
            if not os.path.exists(ctx.rescore):
                raise Fail("llamacpp needs logit-kld's rescore at %s" % ctx.rescore)
            failures += run_prompts(ctx, prompts_for(SMOKE + CORPUS), rescore=True,
                                    update=args.update_golden)

    if args.update_golden:
        if failures:
            print("\nrefusing to write a golden set from a failing run; "
                  "testdata/ is untouched")
        else:
            for src, dst in ctx.staged:
                shutil.copyfile(src, dst)
            current["runs"] = ctx.runs
            current["note"] = ("Golden set: nano-glm local output, every position verified "
                               "against llama.cpp rescore --sim-gen at KL == 0. Regenerate "
                               "with `python gate.py --update-golden`.")
            with open(prov_path, "w") as f:
                json.dump(current, f, indent=1)
            print("\nwrote %s and %d files" % (prov_path, len(ctx.staged)))

    print()
    if failures:
        print("FAILED (%d)" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("PASSED: %s" % " ".join(names))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print("gate: %s" % e)
        sys.exit(3)
