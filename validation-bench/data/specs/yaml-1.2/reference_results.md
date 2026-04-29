# yaml-1.2 reference baselines

These are the scores that off-the-shelf YAML parsers achieve against
this corpus (350 tests: 256 valid / 94 invalid). They establish:

- **Practical ceiling.** The most spec-compliant production parser
  (libfyaml) achieves MCC ≈ 0.986. A model from-scratch implementation
  in C++17 stdlib that scores at or near this is genuinely remarkable;
  scoring above it would mean implementing more strict YAML 1.2
  semantics than any production C library currently does.
- **Realistic upper-band.** Matching libyaml-class compliance
  (MCC ≈ 0.53, what PyYAML/Ruby/JS-libyaml all produce) is what most
  ecosystem deployments actually do.
- **Drift detection on pin bumps.** If `setup.sh` later moves to a newer
  yaml-test-suite commit, re-run the verifier; a sudden ~10pp swing
  from a known-good parser is a flag for either a corpus change or an
  extraction regression.

## Results

| Parser | Version | MCC | TP | FN | FP | TN | Agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **libfyaml** (`fy-tool --yaml-1.2 dump -`) | 0.9.6 | **0.986** | 254 | 2 | 0 | 94 | 348/350 (99.4%) |
| **ruamel.yaml** (`typ=safe`) | 0.17.21 | 0.576 | 196 | 60 | 12 | 82 | 278/350 (79.4%) |
| **PyYAML** (`safe_load_all`) | 6.0.2 (libyaml-backed) | 0.532 | 185 | 71 | 12 | 82 | 267/350 (76.3%) |

Numbers produced by `verify_corpus.py` in this directory.

## Extraction integrity

The verifier also recomputes each test's label by walking
`tests/<id>/error` directly (bypassing `tests.jsonl`) and confirms the
two agree. As of the pinned `yaml-test-suite` commit
`6ad3d2c62885d82fc349026c136ef560838fdf3d`, the result is
**0 mismatches across 350 tests** — `setup.sh` is producing labels that
faithfully reflect upstream's per-test-dir convention.

## libfyaml's two outliers (false negatives)

Both are "unsupported reserved directives" cases:

- `MUS6/05` — input is `%YAM 1.1\n---\n` (typo of `%YAML`).
- `MUS6/06` — input is `%YAMLL 1.1\n---\n`.

Per YAML 1.2 §6.8.1, unsupported reserved directives produce a
**warning** but parsing continues. libfyaml emits the warning and then
fails with a non-zero exit, which is slightly stricter than spec.
This is a known libfyaml strictness rather than a corpus bug.

## PyYAML / ruamel.yaml failure modes

The 70-80 disagreements per Python parser cluster on:

- **Spec-valid inputs they reject:** complex flow indentation
  continuations, aliases as flow-map keys, certain block-scalar
  indentation/whitespace edge cases, BOM handling, tabs in
  whitespace, empty keys in block mappings, trailing-empty-line
  documents.
- **Spec-invalid inputs they accept:** duplicate keys (libyaml allows
  by default), some directive misuse, comment placement after flow
  collection close.

These reflect the gap between libyaml-family lenient parsing and
strict YAML 1.2 — exactly what a benchmark of this difficulty should
surface.

## Reproducing

```bash
# From the validation-bench root, with setup.sh already run:
pip install pyyaml ruamel.yaml          # Python parsers
brew install libfyaml                   # strict-1.2 reference (macOS)
                                        # apt-get install libfyaml-utils  # Debian/Ubuntu

python data/specs/yaml-1.2/verify_corpus.py
```

Add `-v` for per-test disagreement detail. Add `--parser libfyaml` (or
`pyyaml` / `ruamel`) to run a single parser. Add `--skip-integrity` if
you only want parser numbers.
