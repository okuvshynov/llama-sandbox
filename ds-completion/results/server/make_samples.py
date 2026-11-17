#!/usr/bin/env python3
"""Two game samples for the endgame benchmark (README step 153): a secret and nine random
guesses with replies under the game's noise (each guessed digit lost with probability 0.2).
The person's own samples are not in this repository; these only exercise the path.

  results/server/make_samples.py     writes results/server/samples/s0.json, s1.json
"""
import json
import random

rng = random.Random(5)
codes = [str(d) for d in range(1000, 10000) if len(set(str(d))) == 4]


def reply(secret, guess):
    if guess == secret:
        return "4 0"
    g = [c if rng.random() >= 0.2 else '?' for c in guess]
    b = sum(a == c for a, c in zip(g, secret))
    w = sum(c in secret for c in g) - b
    return f"{b} {w}"


for i in range(2):
    secret = rng.choice(codes)
    turns = []
    for _ in range(9):
        g = rng.choice(codes)
        turns.append({"guess": g, "reply": reply(secret, g)})
    json.dump({"secret": secret, "turns": turns}, open(f"results/server/samples/s{i}.json", "w"))
