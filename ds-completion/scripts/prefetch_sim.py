#!/usr/bin/env python3
"""Prefetching the CPU's experts to the GPUs ahead of time, from the router lookahead (README
step 63): a replay of scripts/route-ahead.sh's logs.

    ~/venvs/ds-ref/bin/python scripts/prefetch_sim.py [--load-ms 2.0] [--busy 0.3] [--gpu-ms 0.25] [--load-cost 0.036]

At layer Lt - j of a generation step, the policy picks up to B experts of layer Lt to load from
layer Lt - j's guess; a load goes to the least busy of the 4 links and takes load_ms / (1 - busy)
(busy: the share of the links the cache's own loads take); it is issued only if it can finish before
layer Lt routes (the logged host times of the routings; a load's time is known, so one that
would land late is skipped rather than queued). Layer Lt's CPU time is f(n) = 0.10 + 0.27 n ms
for n distinct non-resident experts (README step 50), its time max(gpu_ms, f): a landed expert
removes itself from n. Each load costs load_cost ms of the step (README step 59, measured live).
The saving is reported against each prompt's generation time without the study (--gen-times);
the logged layer times are scaled by that time over the logged run's own (the study slows a run). Optimistic: the logged
layer times are those without prefetching (faster layers shorten the windows), slots are free.

Log (rt.cpp ra_step_end): header int32 magic n_layer n_expert K; per step int32 M, float end_ms;
per layer uint8 valid, then float routed_ms, int16 chosen[M][6], uint8 resident[X], uint16
uses[X], and for j = 1..K: per token int16 ids[64], float scores[64] (j layers earlier's guess).
"""
import argparse
import glob
import math
import os
import re
import struct

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TOP = 64


def read_log(path):
    with open(path, "rb") as f:
        buf = f.read()
    magic, nl, X, K = struct.unpack_from("<4i", buf, 0)
    assert magic == 0x48415244, path
    o = 16
    steps = []
    while o < len(buf):
        M, tend = struct.unpack_from("<if", buf, o)
        o += 8
        layers = {}
        for L in range(nl):
            valid = buf[o]
            o += 1
            if not valid:
                continue
            (tl,) = struct.unpack_from("<f", buf, o)
            o += 4
            chosen = np.frombuffer(buf, "<i2", M * 6, o).reshape(M, 6)
            o += M * 12
            res = np.frombuffer(buf, "u1", X, o).astype(bool)
            o += X
            uses = np.frombuffer(buf, "<u2", X, o)
            o += 2 * X
            guess = {}
            for j in range(1, K + 1):
                ids = np.empty((M, TOP), np.int16)
                sc = np.empty((M, TOP), np.float32)
                for t in range(M):
                    ids[t] = np.frombuffer(buf, "<i2", TOP, o)
                    o += 2 * TOP
                    sc[t] = np.frombuffer(buf, "<f4", TOP, o)
                    o += 4 * TOP
                guess[j] = (ids, sc)
            layers[L] = (tl, chosen, res, uses, guess)
        steps.append((M, tend, layers))
    return nl, X, K, steps


def gen_ms(err):
    m = re.search(r"generation: (\d+) tokens in ([\d.]+) s", open(err).read())
    return int(m.group(1)), float(m.group(2)) * 1e3


def cpu_ms(n):
    return 0.0 if n == 0 else 0.10 + 0.27 * n


def candidates(policy, ids, sc, res, uses, lam):
    """experts of the target layer to load, best first (resident ones included only by 'score')"""
    best = {}
    votes = {}
    for t in range(ids.shape[0]):
        for i in range(TOP):
            e = int(ids[t, i])
            if e < 0:
                continue
            s = float(sc[t, i])
            if s > best.get(e, -1e30):
                best[e] = s
            if i < 6:
                votes[e] = votes.get(e, 0) + 1
    if policy == "score":
        return sorted(best, key=lambda e: -best[e])
    nonres = [e for e in best if not res[e]]
    if policy == "score-nonres":
        return sorted(nonres, key=lambda e: -best[e])
    if policy == "score+freq":
        return sorted(nonres, key=lambda e: -(best[e] + lam * math.log1p(int(uses[e]))))
    if policy == "votes":
        return sorted([e for e in nonres if votes.get(e, 0)], key=lambda e: (-votes[e], -best[e]))
    raise ValueError(policy)


def simulate(steps, policy, j, B, a, lam=0.0, scale=1.0):
    load = a.load_ms / (1.0 - a.busy)
    tot = dict(saved=0.0, loads=0, landed=0, useful=0, cpu=0, covered=0, layers=0)
    for M, tend, layers in steps:
        links = [0.0] * 4
        order = sorted(layers)
        for Lt in order:
            tl, chosen, res, uses, guess = layers[Lt]
            actual = {int(e) for e in chosen.ravel() if not res[int(e)]}
            tot["cpu"] += len(actual)
            tot["layers"] += 1
            Ls = Lt - j
            if Ls not in layers:
                continue
            issue = layers[Ls][0] * scale
            if policy == "oracle":
                cand = sorted(actual)
            else:
                ids, sc = guess[j]
                if ids[0, 0] < 0:
                    continue
                cand = candidates(policy, ids, sc, res, uses, lam)
            picked = cand[:B]
            landed = set()
            for e in picked:
                if res[e]:  # already on a GPU: the pick is wasted, no load
                    continue
                k = min(range(4), key=lambda i: links[i])
                start = max(links[k], issue)
                if start + load > tl * scale:  # it would land too late: not issued (the scheduler knows the load time)
                    continue
                links[k] = start + load
                tot["loads"] += 1
                landed.add(e)
            tot["landed"] += len(landed)
            cov = actual & landed
            tot["useful"] += len(cov)
            tot["covered"] += len(cov)
            before = max(a.gpu_ms, cpu_ms(len(actual))) if actual else 0.0
            after = max(a.gpu_ms, cpu_ms(len(actual - cov))) if actual else 0.0
            tot["saved"] += before - after
    tot["saved"] -= tot["loads"] * a.load_cost
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=f"{ROOT}/data/route-ahead")
    ap.add_argument("--gen-times", default=f"{ROOT}/results/spec-measure-sclk6-cacheloads800cacheperiod8-head",
                    help="the untraced runs of the same prompts (the study itself slows a run)")
    ap.add_argument("--traced", default=f"{ROOT}/results/route-ahead",
                    help="the logged runs' own outputs: their generation time scales the logged layer times")
    ap.add_argument("--load-ms", type=float, default=2.0)
    ap.add_argument("--busy", type=float, default=0.3)
    ap.add_argument("--gpu-ms", type=float, default=0.25)
    ap.add_argument("--load-cost", type=float, default=0.036)
    a = ap.parse_args()
    runs = []
    for p in sorted(glob.glob(f"{a.logs}/*.bin")):
        name = os.path.basename(p)[:-4]
        nl, X, K, steps = read_log(p)
        n, ms = gen_ms(f"{a.gen_times}/{name}.err")
        _, ms_traced = gen_ms(f"{a.traced}/{name}.err")
        runs.append((name, steps, ms, ms / ms_traced))
    total_ms = sum(r[2] for r in runs)
    n_steps = sum(len(r[1]) for r in runs)
    print("# layer times scaled by untraced / traced generation time: " + " ".join(f"{r[0]} {r[3]:.2f}" for r in runs))
    print(f"# {len(runs)} logs, {n_steps} generation steps; savings against {total_ms / 1e3:.1f} s of untraced generation")
    print(f"# load {a.load_ms} ms / (1 - busy {a.busy}) per link, 4 links; GPU part of a layer {a.gpu_ms} ms; "
          f"{a.load_cost} ms charged per load")
    print(f"{'policy':16s} {'j':>2s} {'B':>2s}  {'loads/step':>10s} {'landed':>7s} {'useful':>7s}  "
          f"{'CPU experts covered':>19s}  {'step time saved':>15s}")
    configs = [("oracle", j, B, 0) for j in (2, 3, 4) for B in (2, 4, 8, 99)]
    for pol in ("score", "score-nonres", "votes"):
        configs += [(pol, j, B, 0) for j in (2, 3, 4) for B in (1, 2, 4, 8)]
    configs += [("score+freq", j, B, lam) for j in (2, 3) for B in (2, 4) for lam in (0.05, 0.1, 0.2)]
    for pol, j, B, lam in configs:
        tot = dict(saved=0.0, loads=0, landed=0, useful=0, cpu=0, covered=0, layers=0)
        for _, steps, _, scale in runs:
            r = simulate(steps, pol, j, B, a, lam, scale)
            for k in tot:
                tot[k] += r[k]
        label = pol + (f" l={lam}" if pol == "score+freq" else "")
        print(f"{label:16s} {j:2d} {B if B < 99 else 'all':>2}  {tot['loads'] / n_steps:10.1f} "
              f"{tot['landed'] / max(1, tot['loads']):7.1%} {tot['useful'] / max(1, tot['loads']):7.1%}  "
              f"{tot['covered'] / max(1, tot['cpu']):19.1%}  {tot['saved'] / total_ms:15.1%}", flush=True)


if __name__ == "__main__":
    main()
