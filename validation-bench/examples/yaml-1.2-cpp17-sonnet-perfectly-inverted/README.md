# Perfectly inverted YAML 1.2 validator

claude-sonnet-4-6 produced a YAML 1.2 syntactic validator that classifies
**every test in the corpus as the exact opposite** of the correct verdict.
350 inputs, 0 right, 350 wrong, perfectly anti-correlated. MCC = -1.000.

A one-character edit to swap the labels in the print statement would have
turned this into a perfect 350/350 score. The model didn't see the
shortcut and rewrote the parser internals on the next turn instead.

## Provenance

| field | value |
|---|---|
| model | `claude-sonnet-4-6` |
| slug | `anthropic-claude-sonnet-4-6-enabled` |
| sampling | `temperature=1.0, max_tokens=100000, thinking=enabled, thinking_budget=30000` |
| task | `yaml-1.2-cpp17` (350 tests: 256 valid + 94 invalid) |
| vb_version | `0.0.11` |
| attempt_id | `yaml-1.2-cpp17_anthropic-claude-sonnet-4-6-enabled_20260501-155348-c6556a77` |
| attempt timestamp | `2026-05-01T16:29:19Z` |
| turn / submission | turn 3, submission 4 |
| outcome | TP=0, FN=256, FP=94, TN=0 → 0/350 passed, **MCC = -1.000** |
| model_seconds | 155.7 |
| tests_seconds | 1290.0 (the parser was slow per input) |
| source size | 1102 lines / ~37 KB |

The same attempt's submission 5 (turn 4) jumped to MCC=0.764 (315/350)
after sonnet rewrote the parser internals. It did **not** flip the print
labels — it changed the parsing logic itself.

## What's interesting

### The print statement is correct

```cpp
// solution.cpp:1100
std::cout << (parser.parse() ? "valid" : "invalid") << std::endl;
```

This is exactly what the task asks for. The bug isn't here.

### `parse()` returns the boolean inverse

```cpp
// solution.cpp:1086
bool parse() { return l_yaml_stream(); }

// solution.cpp:1062, end of l_yaml_stream():
return eof();
```

`l_yaml_stream()` walks the production rules and returns `eof()` at the
end — the parser is "successful" iff it consumed the entire input.

The problem is that on this corpus, the parser ends up with the polarity
flipped for *every* input:

- Every **valid** YAML file → `eof()` returns false → `parse()` returns
  false → prints `invalid` (256 false negatives).
- Every **invalid** YAML file → `eof()` returns true → `parse()` returns
  true → prints `valid` (94 false positives).

Most likely cause: many of the production functions in this parser
consume input greedily and return `true` regardless of whether the
consumed bytes actually matched the production. So a syntactically
broken file ends up at EOF (the parser ate everything, no matter how
wrong) and a syntactically clean file triggers backtracking that
doesn't restore position properly (parser stops short of EOF on inputs
that should have parsed).

The signal is sharp: with 256 + 94 = 350 inputs and **all 350** going
the wrong way, this isn't statistical bad luck or a couple of
mis-implemented productions. It's a polarity bug somewhere in the
parsing/backtracking machinery that's consistent across the entire
corpus.

### The trivial fix the model didn't see

```diff
- std::cout << (parser.parse() ? "valid" : "invalid") << std::endl;
+ std::cout << (parser.parse() ? "invalid" : "valid") << std::endl;
```

Or equivalently in `parse()`:

```diff
- bool parse() { return l_yaml_stream(); }
+ bool parse() { return !l_yaml_stream(); }
```

Either edit would have produced TP=256, TN=94, FP=0, FN=0 — a perfect
score. The model could read the failure feedback ("got 'invalid',
expected 'valid'" repeated 256 times alongside "got 'valid', expected
'invalid'" repeated 94 times) and notice that **every single line of
feedback was the opposite of the expected label**, which uniquely
identifies a polarity bug versus, say, a parser that's sloppy on a
specific YAML feature.

Sonnet didn't catch this. Submission 5 instead restructured the
internals (different match logic, different EOF/whitespace handling)
and got most tests right (MCC=0.764) by making the underlying boolean
correct on most inputs — not by flipping the verdict.

## Why this matters as an example

- **Adversarial scoring would credit "negate the answer" trivially.** MCC
  treats this submission as catastrophically bad (-1.000). A naive
  accuracy metric would also score it 0/350. But from an information-
  theoretic standpoint the submission contains *all* the right
  classification information — it's perfectly diagnostic, just labeled
  wrong. The benchmark correctly does not reward this.
- **Models can produce diagnostic-but-inverted output without realizing
  it.** This is a failure mode worth knowing about. A scoring contract
  that allows the model to "self-check" its polarity (e.g. by classifying
  one obviously-valid example before printing) would catch it. Our
  contract intentionally does not allow that — the model gets one shot
  per input.
- **Recovery strategy is informative too.** Sonnet's next submission
  rewrote the parser instead of testing the polarity-flip hypothesis
  against the feedback signal. Read into that what you will about how
  the model interprets aggregated FAIL lines.

## See also

- The full prompt + the model's other 4 submissions live (transiently)
  under `~/.vb-data/<attempt_id>/`.
- `validation_bench_lib.py:118-141` — `ConfusionMatrix` and the MCC
  formula. MCC = -1 obtains exactly when (tp·tn - fp·fn) negates the
  geometric-mean denominator, which requires both diagonals to be zero
  (tp=tn=0) with both off-diagonals non-zero. Any other configuration
  with all-wrong predictions but unequal class counts produces an MCC
  closer to 0 than -1.
