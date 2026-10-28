#!/usr/bin/env python3
"""Adaptive expert cache simulation (PLAN.md "Adaptive expert cache", phase 1): replays routing
logs (ds-completion --log-routing; version 2 logs carry the verify chunks of --spec, rejected
drafts included) against a cache of expert slots with realistic constraints, and sweeps the
aggressiveness knobs.

    ~/venvs/ds-ref/bin/python scripts/cache_sim.py [--quick]

Logs: data/routing/spec/*.bin (scripts/spec-routing.sh: the 8 held-out prompts, --spec,
study30) and data/routing/study/*.bin (the 30 study prompts, plain decoding).

Model, per log (a fresh run each):
  - Slots: 6201 (what the runtime places with the draft loaded: dies 879 + 3 x 1774); the split
    over dies is not modelled (it changes GPU balance, not hits).
  - Warm start: the top of placements/study30.txt for the held-out logs; for the study logs,
    which study30 was counted on, the top of the other 29 logs (leave-one-out).
  - Time is the forward ("step"): one token in plain decoding and prefill, one verify chunk of
    M = 1-6 tokens with --spec. Plain decoding of a held-out prompt is replayed from the
    committed tokens of its --spec log. Prompt tokens drive the cache too (the runtime decodes
    them one by one); every statistic counts generated tokens only.
  - A step's slots hit if their expert is resident when the step starts; its missed slots
    (expert-token pairs) run on the CPU.
  - Step time: g0 + g1 (M - 1) + draft (M > 1) + max(0, c x missed slots - o): g0 and g1 from
    the oracle placement's verify steps (G0_MS, G1_MS); c and o (the CPU work hidden under the
    GPU's) fitted on the 16 measured held-out runs with study30, plain and --spec (printed
    first: ~5% rms). The fit picked missed slots over distinct missed experts (12% rms). Est
    t/s with loads free (fully overlapped) and with every load charged like one missed slot
    (c: its host read of 18.8 MB competes for the same memory bandwidth); the truth is in
    between (phase 2 measures it).
  - Loads: at the end of a step, admitted experts join a queue; loads start at up to `rate` per
    second of modelled time (a token bucket), each evicts a victim at once (it leaves the table
    before its slot is overwritten) and the expert is resident `latency` steps later. Miss
    requests older than 4 steps are dropped; a rebalance replaces the previous one's unissued
    swaps.
  - Admission (k, W): an expert is requested once it has missed k times within the last W
    committed tokens (k = 1: on the first miss, plain LRU). Victims: least recently used.
  - Rebalance (period P, window W): no loads on misses; every P committed tokens, the
    non-resident experts used most in the last W tokens replace the residents used least (a
    swap only if the newcomer was used more), through the same queue and rate.
  - Source: which tokens drive recency, admission and rebalance counts: all forwarded tokens
    (rejected drafts included) or committed ones only (known when the verify step ends).
"""
import argparse
import glob
import heapq
import os
import re
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_stats import pairs, read_log  # noqa: E402

N_EXPERT = 384
SLOTS = 6201
EXPERT_MB = 18.80
DRAFT_MS = 10.5  # measured: 9.6-11.7 ms per drafting step (results/spec-sclk/out2)
# The step's cost with (nearly) every expert on GPUs, from the oracle placement's --spec-log
# (results/spec-heldout/oracle-*.log, mean verify ms by chunk size): 39-41 ms for 1 token, then
# ~15 ms per extra token (54, 72, 88-93, 104, 113-124 ms for 2-6). Fitting these with the miss
# cost confounds them (longer chunks miss more): the fit gave g1 ~26 ms, i.e. --spec slower
# than plain at full hits, or g0 ~19 ms, i.e. plain at full hits far above the oracle's 25 t/s.
G0_MS, G1_MS = 39.0, 15.0
STALE_STEPS = 4
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def read_placement(path):
    keys = []
    for line in open(path):
        if not line.startswith("#"):
            L, e, _ = line.split()
            keys.append(int(L) * N_EXPERT + int(e))
    return np.array(keys)


def gen_time(err_path):
    """(generated tokens, seconds) from a ds-completion stderr"""
    m = re.search(r"generation: (\d+) tokens in ([\d.]+) s", open(err_path).read())
    return int(m.group(1)), float(m.group(2))


class Trace:
    """one run as steps: per step, the slots' pair keys [M, L*k], committed and generated flags"""

    def __init__(self, name, n_pairs, steps):
        self.name, self.n_pairs, self.steps = name, n_pairs, steps

    @staticmethod
    def from_log(path):
        d, nl, k = read_log(path, committed_only=False)
        keys = pairs(d["ids"].astype(np.int64))
        last = {}
        for i, p in enumerate(d["pos"]):
            last[int(p)] = i
        committed = np.zeros(len(d), bool)
        committed[list(last.values())] = True
        gen = d["prompt"] == 0
        bounds = np.flatnonzero(np.diff(d["step"])) + 1
        steps = [(keys[a:b], committed[a:b], gen[a:b]) for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(d)])]
        return Trace(os.path.basename(path).rsplit(".", 1)[0], nl * N_EXPERT, steps)

    def plain(self, n_gen=None):
        """plain decoding of the same text: the committed tokens one per step (the first n_gen
        generated ones)"""
        steps, g = [], 0
        for keys, comm, gen in self.steps:
            for i in np.flatnonzero(comm):
                if gen[i]:
                    if n_gen is not None and g >= n_gen:
                        return Trace(self.name, self.n_pairs, steps)
                    g += 1
                steps.append((keys[i:i + 1], comm[i:i + 1], gen[i:i + 1]))
        return Trace(self.name, self.n_pairs, steps)

    def committed_keys(self):
        return np.concatenate([k[c & g] for k, c, g in self.steps])

    def gen_tokens(self):
        return int(sum((c & g).sum() for _, c, g in self.steps))

    def gen_steps(self):
        return sum(1 for _, _, g in self.steps if g.any())


class TimeModel:
    def __init__(self, c=0.275, o=7.0):
        self.c, self.o = c, o

    def ms(self, M, missed_slots):
        return G0_MS + G1_MS * (M - 1) + (DRAFT_MS if M > 1 else 0.0) + max(0.0, self.c * missed_slots - self.o)


class Stats:
    def __init__(self):
        self.c_slots = self.c_hits = self.a_slots = self.a_hits = 0
        self.steps = self.tokens = self.cpu_experts = self.loads = 0
        self.ms = 0.0

    def add(self, o):
        for f in vars(self):
            setattr(self, f, getattr(self, f) + getattr(o, f))

    def row(self, tm):
        t = max(self.tokens, 1)
        sec = self.ms / 1e3
        charged = sec + self.loads * tm.c / 1e3
        return (f"{self.c_hits / self.c_slots:6.1%} {self.a_hits / self.a_slots:6.1%}  "
                f"{self.cpu_experts / t:6.1f}  {self.loads / t:6.2f} {self.loads * EXPERT_MB / 1e3 / sec:5.2f}  "
                f"{self.tokens / sec:5.1f} {self.tokens / charged:5.1f}")


HEADER = "hit:commit fwd   CPU exp/tok  loads/tok GB/s   t/s:free charged"
# CPU exp/tok: distinct missed experts per step, per committed token (the CPU's weight reads)


def simulate(tr, warm, rate, tm, latency=1, k=1, window=256, source="all", period=0, rb_window=256):
    """rate: loads per second (0 = static, None = unlimited)"""
    n = tr.n_pairs
    C = len(warm)
    resident = np.zeros(n, bool)
    resident[warm] = True
    last_use = np.full(n, -1, np.int64)
    for i, key in enumerate(warm[::-1]):  # warm order: most used first, so the least used go first
        last_use[key] = -C + i
    heap = [(int(last_use[x]), int(x)) for x in warm]
    heapq.heapify(heap)
    pending = np.zeros(n, bool)       # queued or in flight
    queue = deque()                   # (key, victim or -1, step queued)
    flight = deque()                  # (step resident, key)
    misses = {}                       # key -> deque of committed-token times of its misses
    rb_counts = np.zeros(n, np.int64)
    rb_hist = deque()
    t_tok = 0                         # committed tokens so far (prompt included)
    next_rb = period
    bucket = 0.0
    st = Stats()

    def victim():
        while heap:
            s, x = heapq.heappop(heap)
            if not resident[x]:
                continue
            if s < last_use[x]:
                heapq.heappush(heap, (int(last_use[x]), x))
                continue
            return x
        return -1

    for step, (keys, comm, gen) in enumerate(tr.steps):
        while flight and flight[0][0] <= step:
            _, x = flight.popleft()
            resident[x] = True
            pending[x] = False
            last_use[x] = step
            heapq.heappush(heap, (step, x))
        M = len(keys)
        hit = resident[keys]
        cpu = len(np.unique(keys[~hit]))
        step_ms = tm.ms(M, int((~hit).sum()))
        g = gen[:, None]
        if gen.any():
            st.steps += 1
            st.tokens += int((comm & gen).sum())
            st.a_slots += int(g.sum()) * keys.shape[1]
            st.a_hits += int((hit & g).sum())
            cg = (comm & gen)[:, None]
            st.c_slots += int(cg.sum()) * keys.shape[1]
            st.c_hits += int((hit & cg).sum())
            st.cpu_experts += cpu
            st.ms += step_ms
        src = keys if source == "all" else keys[comm]
        last_use[np.unique(src)] = step
        t_tok += int(comm.sum())
        if rate == 0:
            continue
        if period == 0:  # admission on misses
            missed = src[~resident[src]]
            for x in (np.unique(missed) if k == 1 else missed.ravel()):
                x = int(x)
                if pending[x] or resident[x]:
                    continue
                if k > 1:
                    q = misses.setdefault(x, deque())
                    q.append(t_tok)
                    while q and q[0] <= t_tok - window:
                        q.popleft()
                    if len(q) < k:
                        continue
                    del misses[x]
                pending[x] = True
                queue.append((x, -1, step))
        else:  # rebalance: counts over the last rb_window committed tokens
            ks = src.ravel()
            np.add.at(rb_counts, ks, 1)
            rb_hist.append((t_tok, ks))
            while rb_hist and rb_hist[0][0] <= t_tok - rb_window:
                np.add.at(rb_counts, rb_hist.popleft()[1], -1)
            if t_tok >= next_rb:
                next_rb = t_tok + period
                for x, _, _ in queue:
                    pending[x] = False
                queue.clear()
                cand = np.flatnonzero((rb_counts >= 2) & ~resident & ~pending)
                cand = cand[np.argsort(-rb_counts[cand], kind="stable")]
                res = np.flatnonzero(resident)
                res = res[np.lexsort((last_use[res], rb_counts[res]))]
                for c, v in zip(cand, res):
                    if rb_counts[c] <= rb_counts[v]:
                        break
                    pending[c] = True
                    queue.append((int(c), int(v), step))
        # issue loads
        while queue and queue[0][1] < 0 and queue[0][2] < step - STALE_STEPS:
            pending[queue.popleft()[0]] = False
        if rate is None:
            bucket = float("inf")
        else:
            bucket = min(bucket + rate * step_ms / 1e3, 1.0 + rate * 0.2)
        issued = 0
        while queue and bucket >= 1.0:
            x, v, _ = queue.popleft()
            if v < 0 or not resident[v]:
                v = victim()
            if v < 0:
                pending[x] = False
                break
            resident[v] = False
            flight.append((step + latency, x))
            issued += 1
            bucket -= 1.0
        if gen.any():
            st.loads += issued
    return st


def run(traces, warm_of, label, tm, **kw):
    tot = Stats()
    for tr in traces:
        tot.add(simulate(tr, warm_of(tr), tm=tm, **kw))
    print(f"{label:40s} {tot.row(tm)}", flush=True)
    return tot


def calibrate(runs, warm_of):
    """fit c and o (g0, g1 fixed) on the measured generation times of the static (study30) runs:
    grid search, least squares of the relative time error"""
    steps = []
    for kind, tr, (n, sec) in runs:
        res = np.zeros(tr.n_pairs, bool)
        res[warm_of(tr)] = True
        S = np.array([(len(k), int((~res[k]).sum())) for k, c, g in tr.steps if g.any()], float)
        base = (G0_MS + G1_MS * (S[:, 0] - 1) + DRAFT_MS * (S[:, 0] > 1)).sum()
        steps.append((kind, tr.name, S[:, 1], base, sec * 1e3, tr.gen_tokens()))

    def errs(c, o):
        return np.array([(b + np.maximum(0, c * x - o).sum()) / ms - 1 for _, _, x, b, ms, _ in steps])

    grid = [(float(np.sqrt((errs(c, o) ** 2).mean())), c, o)
            for o in np.arange(0, 20.5, 0.5) for c in np.arange(0.15, 0.40, 0.0025)]
    rms, c, o = min(grid)
    tm = TimeModel(c, o)
    print(f"# time model, step = {G0_MS} + {G1_MS} (M - 1) + {DRAFT_MS} draft (M > 1) + max(0, {c:.4f} x missed "
          f"slots - {o:.1f}) ms; c, o fitted on {len(steps)} measured study30 runs: {rms:.1%} rms")
    print("# run                    measured t/s  model t/s")
    for (kind, name, x, b, ms, tok), e in zip(steps, errs(c, o)):
        print(f"#   {kind:5s} {name:14s} {tok / ms * 1e3:8.2f} {tok / (ms * (1 + e)) * 1e3:10.2f}")
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default=f"{ROOT}/placements/study30.txt")
    ap.add_argument("--slots", type=int, default=SLOTS)
    ap.add_argument("--quick", action="store_true", help="fewer configurations")
    a = ap.parse_args()

    study30 = read_placement(a.placement)[:a.slots]
    spec_paths = sorted(glob.glob(f"{ROOT}/data/routing/spec/*.bin"))
    spec = [Trace.from_log(p) for p in spec_paths]
    plain = [t.plain() for t in spec]
    study = [Trace.from_log(p) for p in sorted(glob.glob(f"{ROOT}/data/routing/study/*.bin"))]
    w30 = lambda tr: study30  # noqa: E731

    # calibration: the --spec runs that wrote the logs, and the plain runs of results/spec-sclk/out2
    # (300 tokens: the first 300 committed tokens of the same text)
    cal = [("spec", t, gen_time(p.replace(".bin", ".err"))) for t, p in zip(spec, spec_paths)]
    for t in spec:
        n, sec = gen_time(f"{ROOT}/results/spec-sclk/out2/{t.name}-plain.err")
        cal.append(("plain", t.plain(n), (n, sec)))
    tm = calibrate(cal, w30)

    INF = None
    rates = [0, 25, 50, 100, 200, 400, INF] if not a.quick else [0, 100, INF]
    mid = [50, 100, 200] if not a.quick else [100]

    def r(x):
        return "inf" if x is None else str(x)

    print(f"\n# {a.slots} slots, latency 1 step unless noted; rate: loads per second (0 = static study30)")
    n_tok, n_steps = sum(t.gen_tokens() for t in spec), sum(t.gen_steps() for t in spec)
    print(f"\n## --spec, held-out prompts ({len(spec)} logs, {n_tok} generated tokens in {n_steps} steps, "
          f"{n_tok / n_steps:.2f} tokens per step), warm start study30")
    print(f"{'policy':40s} {HEADER}")
    print("# LRU (load on the first miss), loads from all forwarded tokens")
    for rr in rates:
        run(spec, w30, f"lru rate {r(rr)}", tm, rate=rr)
    print("# LRU, loads from committed tokens only")
    for rr in mid + [INF]:
        run(spec, w30, f"lru rate {r(rr)} src=committed", tm, rate=rr, source="committed")
    print("# admission: k misses within W committed tokens")
    for kk, W in [(2, 64), (2, 256), (3, 256), (4, 1024)]:
        for rr in mid + [INF]:
            run(spec, w30, f"admit k={kk} W={W} rate {r(rr)}", tm, rate=rr, k=kk, window=W)
    print("# rebalance: every P tokens, the top of the last W tokens")
    for P, W in [(16, 256), (64, 256), (256, 1024)]:
        for rr in mid + [INF]:
            run(spec, w30, f"rebalance P={P} W={W} rate {r(rr)}", tm, rate=rr, period=P, rb_window=W)
    print("# load latency in steps (LRU)")
    for lat in [2, 4, 8]:
        for rr in [100, INF]:
            run(spec, w30, f"lru rate {r(rr)} latency {lat}", tm, rate=rr, latency=lat)

    print(f"\n## plain decoding, the same held-out texts (committed tokens), warm start study30")
    print(f"{'policy':40s} {HEADER}")
    for rr in rates:
        run(plain, w30, f"lru rate {r(rr)}", tm, rate=rr)
    for kk, W in [(2, 256), (3, 256)]:
        for rr in mid:
            run(plain, w30, f"admit k={kk} W={W} rate {r(rr)}", tm, rate=rr, k=kk, window=W)
    for P, W in [(16, 256), (64, 256)]:
        for rr in mid:
            run(plain, w30, f"rebalance P={P} W={W} rate {r(rr)}", tm, rate=rr, period=P, rb_window=W)

    if study:
        counts = {t.name: np.bincount(t.committed_keys().ravel(), minlength=t.n_pairs) for t in study}
        allc = sum(counts.values())
        warm = {t.name: np.lexsort((np.arange(len(allc)), -(allc - counts[t.name])))[:a.slots] for t in study}
        wl = lambda tr: warm[tr.name]  # noqa: E731
        print(f"\n## plain decoding, study prompts ({len(study)} logs, {sum(t.gen_tokens() for t in study)} "
              f"generated tokens), warm start: the other 29 logs (leave-one-out)")
        print(f"{'policy':40s} {HEADER}")
        for rr in rates:
            run(study, wl, f"lru rate {r(rr)}", tm, rate=rr)
        for kk, W in [(2, 256), (3, 256)]:
            for rr in mid:
                run(study, wl, f"admit k={kk} W={W} rate {r(rr)}", tm, rate=rr, k=kk, window=W)
        for P, W in [(16, 256), (64, 256)]:
            for rr in mid:
                run(study, wl, f"rebalance P={P} W={W} rate {r(rr)}", tm, rate=rr, period=P, rb_window=W)


if __name__ == "__main__":
    main()
