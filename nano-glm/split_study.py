#!/usr/bin/env python3
"""What a slot left on the CPU costs, for a model whose experts nearly fit.

    python split_study.py --model <ds4.gguf>
    python split_study.py --model <ds4.gguf> --ladder 0,0,0,0 1,1,1,1 2,2,1,1

`moe-server --force-split a,b,c,d` sends slot *s* of every token to the device
the pattern names, regardless of what the router chose, remapping the expert
into that device's resident range. **The output is wrong by construction.** The
work has the right shape and cost, which is all a timing needs, and `gate.py`
owns correctness so the two never meet.

Why this study is *not* the one GLM-5.2 got
-------------------------------------------
`OPTIMIZATION.md` used forced placement on GLM-5.2 to reach distributions
residency could not: only ~22% of its experts fit in VRAM, so the real system
could never put every slot on a die, and the interesting question was what full
offload *would* be worth.

DeepSeek-V4-Flash is the opposite case. At 240 of 256 experts resident per layer
(93.75%) a token's 6 slots already land on a die ~5.6 times out of 6, so the
"everything offloaded" end of the ladder is roughly where the real system
already sits, and forcing it there measures almost nothing new.

What is worth measuring here is the other direction: forcing *fewer* slots onto
the dies, to get the marginal cost of a slot the CPU has to serve. That number
is what says whether more residency is worth buying, and it bounds what any
faster interconnect could ever return — if a CPU slot is cheap, moving it to a
die cannot help much no matter how fast the link.

Measurement discipline, all of it paid for elsewhere in this repo
-----------------------------------------------------------------
- **One warm-up pass per configuration, discarded.** A harness-style run pages
  the model in during the measured window; the first pass is not the same
  experiment as the second (repo `CLAUDE.md`).
- **Every repetition printed, never just a mean.** The warm-up curve is signal.
- **Configurations run in ladder order, and the ladder is short.** Each rung
  costs a server restart, which is a 150 GiB load plus a VRAM upload.
- **Two reps is not three.** This file reports what it measured; a difference
  smaller than the spread between reps is not a result. That warning is here
  because this session watched a clean monotone trend across single runs
  dissolve when one configuration was repeated.

Stdlib only, as everything else here is.
"""

import argparse
import os
import re
import subprocess
import sys
import time


def start_server(exe, model, host, port, threads, gpu_experts, gpu_devices,
                 force_split, log_path):
    """Spawn moe-server and wait for it to accept connections.

    Stopping it again is the part that needs care: a process holding a
    half-terabyte of views takes minutes to unmap, and until it does the exe
    stays locked and a rebuild fails with LNK1104, which looks like anything but
    this (repo `CLAUDE.md`). `stop_server` waits, and says so if it has to.
    """
    cmd = [exe, "-m", model, "--host", host, "--port", str(port),
           "-t", str(threads), "--gpu-experts", str(gpu_experts),
           "--gpu-devices", str(gpu_devices)]
    if force_split:
        cmd += ["--force-split", force_split]

    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900          # a VRAM upload is not a model load
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("moe-server exited during startup (see %s)" % log_path)
        with open(log_path, errors="replace") as f:
            if "listening on" in f.read():
                return proc, log
        time.sleep(1.0)
    proc.kill()
    raise SystemExit("moe-server did not come up within 900s (see %s)" % log_path)


def stop_server(proc, log):
    proc.terminate()
    t0 = time.time()
    try:
        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        print("      WARNING: still unmapping after 600s; a rebuild will fail "
              "with LNK1104 until it exits")
        proc.kill()
    else:
        dt = time.time() - t0
        if dt > 5:
            print("      (took %.0fs to unmap and exit)" % dt)
    log.close()


RE_TPS = re.compile(r"n_prompt=\d+ \(([\d.]+) tok/s\), n_gen=\d+ \(([\d.]+) tok/s\)")
RE_RTT = re.compile(r"rtt p50 (\d+) us p90 (\d+) us")


def run_client(exe, model, prompt, n_predict, threads, moe_addr, out_bin, out_log):
    cmd = [exe, "-m", model, "-i", prompt, "-n", str(n_predict),
           "-t", str(threads), "-o", out_bin]
    if moe_addr:
        cmd += ["--moe-addr", moe_addr]
    with open(out_log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit("client failed (rc=%d), see %s" % (rc, out_log))

    text = open(out_log, errors="replace").read()
    m = RE_TPS.search(text)
    if not m:
        raise SystemExit("no throughput line in %s" % out_log)
    pp, tg = float(m.group(1)), float(m.group(2))
    r = RE_RTT.search(text)
    return pp, tg, (int(r.group(1)) if r else 0)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default=os.path.join(here, "testdata-deepseek4", "01_prose.bin"))
    ap.add_argument("--client", default=os.path.join(here, "build", "bin", "nano-glm.exe"))
    ap.add_argument("--server", default=os.path.join(here, "build-vk", "bin", "moe-server.exe"))
    ap.add_argument("--work", default=os.path.join(here, "results", "split-study"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--gpu-experts", type=int, default=240)
    ap.add_argument("--gpu-devices", type=int, default=4)
    ap.add_argument("--reps", type=int, default=2, help="measured passes after one discarded warm-up")
    # Default ladder: 0, 2, 4 and 6 slots left on the CPU out of n_expert_used=6.
    # Four rungs spanning the whole range, because each costs a server restart
    # and a straight line through four points is worth more than two points
    # measured three times.
    ap.add_argument("--ladder", nargs="*",
                    default=["2,2,1,1", "1,1,1,1", "1,1,0,0", "0,0,0,0"])
    ap.add_argument("--skip-natural", action="store_true",
                    help="skip the unforced RPC run (routing decides placement)")
    ap.add_argument("--skip-local", action="store_true",
                    help="skip the CPU-only local run (no server at all)")
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    addr = "%s:%d" % (args.host, args.port)
    rows = []

    def measure(label, moe_addr, tag):
        passes = []
        for i in range(args.reps + 1):
            pp, tg, rtt = run_client(
                args.client, args.model, args.prompt, args.n_predict, args.threads,
                moe_addr,
                os.path.join(args.work, "%s-%d.bin" % (tag, i)),
                os.path.join(args.work, "%s-%d.log" % (tag, i)))
            kind = "warm-up (discarded)" if i == 0 else "pass %d" % i
            print("      %-20s prefill %5.2f t/s   decode %5.3f t/s   rtt p50 %d us"
                  % (kind, pp, tg, rtt))
            if i > 0:
                passes.append((pp, tg, rtt))
        pp = sum(p for p, _, _ in passes) / len(passes)
        tg = sum(t for _, t, _ in passes) / len(passes)
        rtt = sum(r for _, _, r in passes) / len(passes)
        spread = max(t for _, t, _ in passes) - min(t for _, t, _ in passes)
        rows.append((label, pp, tg, rtt, spread))
        return pp, tg

    if not args.skip_local:
        print("\n=== CPU only, no server")
        measure("CPU only (local)", None, "local")

    if not args.skip_natural:
        print("\n=== routing decides placement (no --force-split)")
        proc, log = start_server(args.server, args.model, args.host, args.port,
                                 args.threads, args.gpu_experts, args.gpu_devices,
                                 None, os.path.join(args.work, "server-natural.log"))
        try:
            measure("natural (routing)", addr, "natural")
        finally:
            stop_server(proc, log)

    for pattern in args.ladder:
        on_gpu = sum(int(x) for x in pattern.split(",") if x.strip())
        print("\n=== --force-split %s  (%d of 6 slots on the dies, %d on the CPU)"
              % (pattern, on_gpu, 6 - on_gpu))
        tag = "fs-" + pattern.replace(",", "")
        proc, log = start_server(args.server, args.model, args.host, args.port,
                                 args.threads, args.gpu_experts, args.gpu_devices,
                                 pattern, os.path.join(args.work, "server-%s.log" % tag))
        try:
            measure("--force-split %s (cpu=%d)" % (pattern, 6 - on_gpu), addr, tag)
        finally:
            stop_server(proc, log)

    print("\n%-34s %9s %9s %9s %9s" % ("configuration", "prefill", "decode", "rtt p50", "tg spread"))
    for label, pp, tg, rtt, spread in rows:
        print("%-34s %8.2f  %8.3f  %8.0f  %8.3f" % (label, pp, tg, rtt, spread))
    print("\nDecode is reported to three places because the differences that matter\n"
          "here are a few percent; `tg spread` is max-min across the measured\n"
          "passes, and any difference smaller than it is not a result.")


if __name__ == "__main__":
    sys.exit(main())
