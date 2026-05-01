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

- `yaml-1.2-cpp17-sonnet-perfectly-inverted/` — claude-sonnet-4-6 wrote a
  YAML 1.2 validator whose `parse()` returned the boolean inverse of the
  correct answer for **every** test in the corpus. Confusion matrix
  TP=0/FN=256/FP=94/TN=0 → MCC = exactly -1.000. A one-character edit to
  the print statement would have given MCC = +1.000 (perfect score), but
  the model didn't see the shortcut and instead rewrote the parser
  internals on the next turn.

## Adding an example

1. Create `<slug>/` and copy the source file from
   `~/.vb-data/<attempt-id>/submissions/<N>/<source_filename>` into it.
2. Write `<slug>/README.md` with provenance (use any existing example as a
   template) plus the analysis: what's surprising, where in the source the
   interesting behavior lives, what the model could have done differently.
3. Don't strip or reformat the source — keep it byte-for-byte as the model
   submitted it. Style oddities (one-line functions, terse names, missing
   whitespace) often *are* the example.
