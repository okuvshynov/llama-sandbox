# Examples

A small museum of interesting model submissions surfaced by validation-bench
runs. Each subdirectory captures one solution that's worth keeping around for
future reference — usually because it illustrates a specific failure mode,
recovery pattern, or quirk of how a model approached a task.

This is *not* a test corpus or a regression baseline. Examples are static
snapshots of one (model, task, attempt) cell at a moment in time; the model's
later outputs and the harness itself will diverge. Treat each entry as a
write-up with the source attached for context, not as something to re-run.

## Layout

Each example lives in its own subdirectory:

```
<short-slug>/
  README.md       # what's interesting about this submission
  solution.<ext>  # the source the model submitted, exactly as it landed
                  # in the per-attempt debug logs (~/.vb-data/<attempt-id>/
                  # submissions/<N>/<source_filename>)
```

The README header records provenance — model, slug, vb_version, attempt_id,
turn, the result row's confusion matrix — so the example is reproducible
in spirit even if the per-attempt debug log under `~/.vb-data/` has been
purged.

## Current examples

- `yaml-1.2-cpp17-kimi-infinite-loop/` — moonshot-kimi-k2.6-thinking
  produced a YAML 1.2 validator that hangs on every input due to an
  unconditional `while (true)` loop in `parse_l_yaml_stream`. Every
  test fails by timeout, no verdict is ever printed, and MCC evaluates
  to exactly -1.000 algebraically (TP=TN=0 with both off-diagonals
  non-zero). The bug is localized to a single five-line function; the
  README points at the exact lines.
- `yaml-1.2-cpp17-gpt-5.5-repro/` — gpt-5.5-xhigh's strongest yaml-1.2-cpp17
  attempt (MCC=0.912, 12 disagreements out of 350) packaged as a
  reproducer. Includes the model's `solution.cpp`, byte-exact input bytes
  for two specific failures (one false-negative case `26DV`, one
  false-positive `H7J7`), and copy-pasteable shell recipes that build the
  solution, install libfyaml as a third-party reference parser, and run
  all three (model / libfyaml / yaml-test-suite ground truth) on each
  case. Lets a reviewer poke at specific disagreements without trusting
  any of our scoring code.
- `yaml-1.2-cpp17-opus-4-7-strongest/` — the strongest single
  yaml-1.2-cpp17 submission across the entire dataset:
  claude-opus-4-7-adaptive at **MCC=0.993, 349/350 passed** (next best
  by any model is gpt-5.5-xhigh at 0.912). Same repro shape as the
  gpt-5.5 entry. The single wrong test (`X38W` — flow mapping with
  anchored-sequence and alias keys) is documented with libfyaml as
  third-party agreement against the model and a YAML 1.2.2 spec
  citation explaining why arbitrary nodes are valid as flow keys.

## Adding an example

1. Create `<slug>/` and copy the source file from
   `~/.vb-data/<attempt-id>/submissions/<N>/<source_filename>` into it.
2. Write `<slug>/README.md` with provenance (use any existing example as a
   template) plus the analysis: what's surprising, where in the source the
   interesting behavior lives, what the model could have done differently.
3. Don't strip or reformat the source — keep it byte-for-byte as the model
   submitted it. Style oddities (one-line functions, terse names, missing
   whitespace) often *are* the example.
