# Perfectly inverted YAML 1.2 validator (kimi, stuck across turns and attempts)

moonshot-kimi-k2.6-thinking produced a YAML 1.2 syntactic validator with
the same "perfect anti-correlation" failure mode as the sonnet entry next
door (`../yaml-1.2-cpp17-sonnet-perfectly-inverted/`) — every test in the
350-input corpus gets the opposite verdict, so MCC = exactly -1.000.

What makes this entry distinct is that **kimi got stuck in the inversion**:
the same MCC=-1 result repeats across multiple turns within an attempt
(despite getting per-test FAIL feedback) and across two independent
attempts on different days. Sonnet hit MCC=-1 once, then on the next
turn rewrote the parser internals and recovered to MCC=0.764. Kimi
didn't escape.

## Provenance

| field | value |
|---|---|
| model | `moonshot-kimi-k2.6-thinking` |
| slug | `moonshot-kimi-k2.6-thinking` |
| sampling | `temperature=1.0, top_p=0.95, max_tokens=128000, mode=thinking, preserve_thinking=True` |
| task | `yaml-1.2-cpp17` (350 tests: 256 valid + 94 invalid) |
| vb_version | `0.0.11` |

The MCC=-1 outcome appears in **two separate attempts**:

| attempt_id | timestamp | turns at MCC=-1 | best turn |
|---|---|---|---|
| `yaml-1.2-cpp17_moonshot-kimi-k2.6-thinking_20260502-023508-95c3f832` | 2026-05-02 04:38Z (~2h elapsed) | turns 1, 2 | turn 3 → MCC=-0.597 (slight escape) |
| `yaml-1.2-cpp17_moonshot-kimi-k2.6-thinking_20260503-043841-a12be177` | 2026-05-03 06:31Z (~1h53m elapsed) | turns 2, 3 | never escaped |

Submission counts and outcomes per attempt:

```
attempt 20260502-023508 (3 submissions):
  turn 1, sub 1:  MCC=-1.000  TP=0  FN=256  FP=94  TN=0    [perfect inversion]
  turn 2, sub 2:  MCC=-1.000  TP=0  FN=256  FP=94  TN=0    [still inverted]
  turn 3, sub 3:  MCC=-0.597  TP=8  FN=248  FP=50  TN=44   [marginal escape]

attempt 20260503-043841 (3 submissions):
  turn 1, sub 1:  COMPILE_ERROR
  turn 2, sub 2:  MCC=-1.000  TP=0  FN=256  FP=94  TN=0    [perfect inversion]
  turn 3, sub 3:  MCC=-1.000  TP=0  FN=256  FP=94  TN=0    [stuck]
```

The included `solution.cpp` is the first inverted submission from the
first attempt (`20260502-023508`, submission 1, 1351 lines). The other
three inverted submissions are structurally similar and live (transiently)
under `~/.vb-data/<attempt_id>/submissions/<N>/solution.cpp` on the
machine that ran them.

## What's in the source (same anatomy as the sonnet entry)

The print statement is **correct**:

```cpp
// solution.cpp:1349
cout << (ok ? "valid" : "invalid") << "\n";

// solution.cpp:1348
bool ok = p.parse();

// solution.cpp:1333-1337
bool parse() {
    if (!parse_l_yaml_stream()) return false;
    while (!eof() && is_whitespace(peek())) ++pos;
    return eof();
}
```

The bug is upstream of the print: `parse()` returns the boolean inverse
of the right answer for every input in the corpus. Almost certainly the
same root cause as the sonnet inversion — the production functions
consume input greedily and return `true` regardless of whether the
match was actually valid, so syntactically broken yaml ends up at EOF
("valid") and syntactically clean yaml triggers backtracking that
doesn't restore position properly ("invalid").

The trivial fix the model didn't see (in either attempt, on either day):

```diff
- cout << (ok ? "valid" : "invalid") << "\n";
+ cout << (ok ? "invalid" : "valid") << "\n";
```

Or:

```diff
- bool ok = p.parse();
+ bool ok = !p.parse();
```

Either edit would have produced TP=256, TN=94, FP=0, FN=0 — a perfect
350/350 score.

## Comparison with the sonnet entry

This is interesting precisely because the failure mode is identical but
the recovery dynamics are different.

| dimension | sonnet (`../yaml-1.2-cpp17-sonnet-perfectly-inverted/`) | kimi (this entry) |
|---|---|---|
| MCC = -1.000 result | 1 submission (turn 3, sub 4 of one attempt) | **4 submissions** across 2 attempts |
| Print statement bug? | no — `parse() ? "valid" : "invalid"` is correct | no — same |
| Where the inversion lives | upstream of `parse()`; greedy productions returning true on bad matches | same |
| Source size when inverted | ~1102 lines | ~1163-1351 lines |
| Stayed inverted across consecutive turns? | no — turn 4 sub 5 jumped to MCC=0.764 | **yes** — both attempts produced ≥2 consecutive MCC=-1 turns |
| Best recovered MCC within attempt | 0.764 | -0.597 (one attempt only; the other never escaped) |
| Reproduced across attempts? | observed once | **observed in 2 independent attempts on different days** |

### Why kimi staying stuck is the more striking finding

Both models got the same FAIL feedback shape after the inverted submission:
- 256 lines of "got 'invalid', expected 'valid'"
- 94 lines of "got 'valid', expected 'invalid'"

That feedback is uniquely diagnostic of a polarity bug — *every* test is
the wrong way around, no exceptions. Sonnet didn't read it that way
either (it rewrote internals instead of trying the polarity-flip
hypothesis), but at least sonnet's rewrite happened to fix the underlying
boolean on most inputs. Kimi's next two attempts produced **bit-for-bit
similar** failure-mode parsers that re-hit MCC=-1 instead of moving the
needle.

Two independent attempts both landing at the exact same wrong answer
suggests the inverted-greedy-parse failure mode is a **stable attractor**
in kimi's solution space for hand-rolled-grammar tasks like this — not a
one-off slip. Worth keeping an eye on whether other reasoning-heavy
parser tasks reproduce the same pattern.

### Suggestive contrast

Both attempts ran with `mode=thinking, preserve_thinking=True` — the
model was given full access to its own prior reasoning across turns.
That makes the stuckness less excusable: the model could see what it
had argued for on turn N when revising on turn N+1, and still re-derived
the same inverted parser. Compare with sonnet's `thinking=enabled,
thinking_budget=30000` setup which produced one inverted submission and
then walked away from it.

## Why this matters as an example

- **Frequency.** Two independent attempts on the same task hitting
  exactly MCC=-1 — same model, different days, separately seeded — is
  not noise; it's pointing at a reproducible behavior. Worth knowing
  about for any future "polarity-aware scoring" or "self-check before
  submit" prompt-engineering experiments on this model.
- **Recovery contrast.** The pair of examples (sonnet recovers, kimi
  doesn't) gives us a small data point on how models with similar prompt
  envelopes use the multi-turn signal. The benchmark harness intentionally
  doesn't tell models about polarity; it just reports per-test verdicts.
  Sonnet rewrote; kimi re-derived. Whether that's a temperature thing,
  a thinking-budget thing, or a deeper attentional-priors thing is a
  separate study.
- **No harness contribution.** Both attempts ran on vb 0.0.11 (post
  cgroup-leak fix, post Responses-only routing on the openai side though
  this is the moonshot script). This is real model behavior, not an
  artifact.

## See also

- `../yaml-1.2-cpp17-sonnet-perfectly-inverted/` — the sonnet sibling
  example. Same root cause, different recovery dynamics.
- `../README.md` — top-level convention for the examples/ directory.
- `validation_bench_lib.py:118-141` — `ConfusionMatrix` and the MCC
  formula. Note the same algebra as in the sonnet entry: MCC=-1 requires
  TP=TN=0 and both off-diagonals non-zero, so this confusion matrix
  isn't approximate — it's exactly anti-correlated for both attempts.
- The other inverted submissions live (transiently) at:
  `~/.vb-data/yaml-1.2-cpp17_moonshot-kimi-k2.6-thinking_20260502-023508-95c3f832/submissions/{1,2,3}/solution.cpp`
  and
  `~/.vb-data/yaml-1.2-cpp17_moonshot-kimi-k2.6-thinking_20260503-043841-a12be177/submissions/{2,3}/solution.cpp`.
