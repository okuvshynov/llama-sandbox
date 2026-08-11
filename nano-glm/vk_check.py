#!/usr/bin/env python3
"""Compare a GPU-assisted moe-server against the CPU one, over identical ids.

    python vk_check.py --gpu-experts 2               # floor, then CPU-vs-GPU
    python vk_check.py --gpu-experts 2 --only smoke
    python vk_check.py --floor-only                  # just the GPU's own noise

Why this exists rather than another `gate.py` test: byte identity ends here.
The expert path runs in different arithmetic on the GPU, so the question stops
being "are the bytes the same" and becomes "is the difference smaller than the
difference the GPU has with itself". Those need different machinery, and mixing
them into gate.py would blur the one thing gate.py is good at.

Three rules this encodes, all of them from PLAN.md step 3 and all of them easy
to get wrong:

  1. **Compare against the same build's CPU run, never a historical file.** A
     golden set carries a fingerprint from another build; a difference against
     it conflates the GPU with everything else that moved.
  2. **Measure the GPU's own reproducibility floor first.** A KL of 3e-4
     against the CPU means nothing until you know whether two GPU runs of the
     same tokens agree to 1e-9 or to 3e-4. The floor is measured by running the
     GPU twice with different batch shapes, which is where ggml's own results
     move around.
  3. **Gate on per-position max, not mean.** One mis-routed token disappears in
     an average over hundreds of positions, and a mis-routed token is exactly
     the failure this code can produce.

The client is always the CPU build: `build\\bin\\nano-glm.exe`. Only the server
changes. That keeps the trunk on the numerics the golden set was made with, so
any difference seen here belongs to the expert path.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
CLIENT = os.path.join(HERE, "build", "bin", "nano-glm" + EXE)
SERVER_CPU = os.path.join(HERE, "build", "bin", "moe-server" + EXE)
SERVER_VK = os.path.join(HERE, "build-vk", "bin", "moe-server" + EXE)
PROMPTS = os.path.join(HERE, "testdata", "prompts.json")
WORK = os.path.join(HERE, "results", "vk")


class Fail(Exception):
    pass


def wait_for_port(host, port, proc, timeout=900):
    """Block until the server accepts, or it dies. A 583 GiB load is slow, and
    a GPU run also uploads experts, so the timeout is generous."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise Fail("moe-server exited during startup (rc=%s)" % proc.returncode)
        s = socket.socket()
        s.settimeout(1.0)
        try:
            s.connect((host, port))
            return time.time() - t0
        except OSError:
            time.sleep(1.0)
        finally:
            s.close()
    raise Fail("moe-server did not come up within %ds" % timeout)


class Server:
    """A moe-server that is reliably gone by the time the next one starts.

    Non-negotiable on this machine: a half-TB process takes minutes to unmap
    its views, and until it does the exe stays locked and the memory stays
    committed. See TESTING.md and the repo CLAUDE.md entry on LNK1104.
    """

    def __init__(self, exe, model, port, extra=(), label=""):
        self.exe, self.model, self.port = exe, model, port
        self.extra, self.label = list(extra), label
        self.proc = self.log = None

    def __enter__(self):
        os.makedirs(WORK, exist_ok=True)
        path = os.path.join(WORK, "server-%s.log" % (self.label or "x"))
        self.log = open(path, "w")
        cmd = [self.exe, "-m", self.model, "--host", "127.0.0.1",
               "--port", str(self.port)] + self.extra
        print("  starting %s" % " ".join(cmd[0:1] + cmd[3:]))
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        secs = wait_for_port("127.0.0.1", self.port, self.proc)
        print("  up in %.0fs (log: %s)" % (secs, path))
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            t0 = time.time()
            self.proc.terminate()
            try:
                self.proc.wait(timeout=900)
            except subprocess.TimeoutExpired:
                print("  WARNING: still unmapping after 900s; killing")
                self.proc.kill()
            dt = time.time() - t0
            if dt > 5:
                print("  server took %.0fs to unmap and exit" % dt)
        if self.log:
            self.log.close()
        return False


def lkld_read_file():
    """logit-kld's reader, loaded by path.

    NOT `sys.path.insert(...); from inspect import read_file` — that module is
    named after a stdlib one and would replace it process-wide, with the
    breakage surfacing wherever some library imports `inspect`, not here.
    """
    import importlib.util
    path = os.path.join(HERE, "..", "logit-kld", "inspect.py")
    spec = importlib.util.spec_from_file_location("lkld_inspect", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_file


def golden_ids(name):
    """Every id the golden file scored: prompt *and* generated.

    This is the fix for a real defect in the first version of this script. It
    ran each configuration with the same *prompt* and let each generate its own
    continuation — but a continuation is greedy, so a perturbation that flips
    one token sends the two runs down different sequences, and then there is
    nothing to compare. It is not hypothetical: `04_history` diverged with
    *both sides on the CPU*, differing only in how a token's 8-term expert sum
    was partitioned.

    Comparing logits requires holding the token ids fixed. Taking them from the
    committed golden means every configuration scores an identical sequence,
    and the sequence does not drift between sessions either.
    """
    f = lkld_read_file()(os.path.join(HERE, "testdata", name + ".bin"))
    return f["seqs"][0]["tokens"]


def run_client(model, ids, out_bin, addr, threads, batch=None):
    # -n 0: score the given ids, generate nothing. Every configuration is then
    # scoring the same sequence, which is what makes the logits comparable.
    cmd = [CLIENT, "-m", model,
           "-T", ",".join(str(t) for t in ids),
           "-n", "0",
           "-t", str(threads),
           "-o", out_bin,
           "--moe-addr", addr]
    # Deliberately NOT --strict: a Vulkan server fails the handshake by design
    # (`vulkan` is in NANO_REPRO_KEYS), and that refusal is the whole reason
    # this script exists instead of `gate.py rpc`.
    if batch is not None:
        cmd += ["-b", str(batch)]
    log = out_bin + ".log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise Fail("client failed (rc=%d), see %s" % (r.returncode, log))
    return out_bin


def compare(a, b):
    """Delegate to logit-kld's compare.py, the reference reader for the format."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "..", "logit-kld", "compare.py"), a, b],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.stderr.strip():
        sys.stderr.write(r.stderr)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"D:\llms\UD-Q6_K\GLM-5.2-UD-Q6_K-00001-of-00014.gguf")
    ap.add_argument("--gpu-experts", type=int, default=2,
                    help="experts per MoE layer on the GPU (31.5 MB each x 75 layers)")
    ap.add_argument("--only", default="smoke", help="comma-separated prompt names")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--port", type=int, default=5713)
    ap.add_argument("--floor-only", action="store_true",
                    help="only measure the GPU's reproducibility floor")
    args = ap.parse_args()

    for exe in (CLIENT, SERVER_CPU, SERVER_VK):
        if not os.path.exists(exe):
            raise Fail("missing %s — build.ps1 and build.ps1 -Vk" % exe)

    with open(PROMPTS) as f:
        prompts = json.load(f)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    for n in names:
        if n not in prompts:
            raise Fail("no such prompt %r (have: %s)" % (n, ", ".join(prompts)))

    os.makedirs(WORK, exist_ok=True)
    vk_extra = ["-t", str(args.threads), "--gpu-experts", str(args.gpu_experts)]

    # ---- 1. the GPU's own reproducibility floor ---------------------------
    # Two runs of the same tokens on the same server, differing only in batch
    # shape. Whatever this shows is the resolution of every later number; a
    # CPU-vs-GPU KL at or below it says nothing at all.
    print("\n=== GPU reproducibility floor (same server, one chunk vs several) ===")
    with Server(SERVER_VK, args.model, args.port, vk_extra, "vk-floor"):
        addr = "127.0.0.1:%d" % args.port
        for name in names:
            # The split batch must be SMALLER than the prompt, or both runs
            # prefill in one chunk and the comparison measures nothing. This
            # bit us once: `smoke` is 14 tokens, so -b 512 against -b 16 gave a
            # flat 0.0 that looked like a beautifully deterministic GPU and was
            # actually two identical runs.
            ids = golden_ids(name)
            n_prompt = len(ids)
            split = max(1, n_prompt // 2)
            if split >= n_prompt:
                raise Fail("%s: cannot split a %d-token prompt" % (name, n_prompt))
            a = run_client(args.model, ids, os.path.join(WORK, name + ".gpu-whole.bin"),
                           addr, args.threads, batch=512)
            a2 = run_client(args.model, ids, os.path.join(WORK, name + ".gpu-whole2.bin"),
                            addr, args.threads, batch=512)
            b = run_client(args.model, ids, os.path.join(WORK, name + ".gpu-split.bin"),
                           addr, args.threads, batch=split)

            # Two questions, and they are not the same one. Determinism asks
            # whether the GPU repeats itself at all; if it does not, every
            # other number here is noise and nothing can be gated. Shape
            # sensitivity asks how much the answer moves for a reason that is
            # not a code change — that is the floor a CPU-vs-GPU figure has to
            # clear before it means anything.
            print("\n[%s] floor 1/2 — determinism: same server, same batch, twice" % name)
            compare(a, a2)
            print("\n[%s] floor 2/2 — shape: prefill in one chunk of %d vs chunks of %d"
                  % (name, n_prompt, split))
            compare(a, b)

    if args.floor_only:
        return 0

    # ---- 2. three runs, so the GPU can be separated from the split --------
    #
    # The middle one is the control, and it is the reason this script is not
    # just "CPU vs GPU". Splitting work across devices changes the summation
    # order regardless of what the second device is: a token's 8 weighted
    # expert rows stop being summed as one sequence. Running that same split
    # onto a second *CPU* device isolates it, because the arithmetic is then
    # identical and any difference is the compaction's alone.
    #
    # Measured once at k=2 and worth keeping in mind: the compaction accounted
    # for most of the CPU-vs-GPU difference. Without this control the whole of
    # it would have been filed against the GPU.
    runs = [
        ("cpu",      SERVER_CPU, ["-t", str(args.threads)],                                  "CPU, no split"),
        ("cpusplit", SERVER_CPU, ["-t", str(args.threads), "--cpu-experts", str(args.gpu_experts)],
                                                                                             "CPU, split onto a 2nd CPU device"),
        ("gpu",      SERVER_VK,  vk_extra,                                                   "CPU, split onto the GPU"),
    ]
    print("\n=== three servers, one client build, identical ids ===")
    for tag, exe, extra, what in runs:
        print("\n-- %s: %s" % (tag, what))
        with Server(exe, args.model, args.port, extra, tag):
            addr = "127.0.0.1:%d" % args.port
            for name in names:
                run_client(args.model, golden_ids(name),
                           os.path.join(WORK, "%s.%s.bin" % (name, tag)),
                           addr, args.threads)

    def path(name, tag):
        return os.path.join(WORK, "%s.%s.bin" % (name, tag))

    for name in names:
        print("\n" + "=" * 70)
        print("[%s] compaction alone   (cpu vs cpusplit) — same arithmetic" % name)
        compare(path(name, "cpu"), path(name, "cpusplit"))
        print("\n[%s] GPU alone          (cpusplit vs gpu) — same partitioning" % name)
        compare(path(name, "cpusplit"), path(name, "gpu"))
        print("\n[%s] end to end         (cpu vs gpu)" % name)
        compare(path(name, "cpu"), path(name, "gpu"))

    print("""
How to read this:
  * `max`, not `mean`. One mis-routed token vanishes in an average, and a
    mis-routed token is the failure this code can produce. compare.py's mean
    can also go negative, which is a top-128 truncation artifact rather than a
    real divergence — median and max are the columns to trust.
  * Nothing is meaningful below the shape floor printed at the top.
  * If `GPU alone` is far above `compaction alone`, suspect the GPU kernels.
    If `compaction alone` is far above the floor, suspect this repo's
    partition/scatter, not the driver.
  * The server log prints the pair split per device, so the share of work that
    actually reached the GPU is measured rather than inferred from k.""")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fail as e:
        print("FAIL: %s" % e, file=sys.stderr)
        sys.exit(1)
