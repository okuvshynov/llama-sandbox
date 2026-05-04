# Strongest yaml-1.2-cpp17 submission to date — opus-4-7-adaptive (MCC=0.993)

The single highest-scoring submission across the entire `yaml-1.2-cpp17`
dataset: **349/350 tests passed**. Confusion matrix TP=255, FN=1, FP=0,
TN=94 — one valid YAML rejected, zero invalid YAMLs accepted.

The single failure is a flow mapping with anchored-sequence and alias
keys (`X38W`). Both the spec and libfyaml accept it; opus rejects it.

The test inputs under `cases/<TEST_ID>/in.yaml` are vendored from the
[yaml-test-suite](https://github.com/yaml/yaml-test-suite) data branch
(MIT, Copyright 2016–2020 Ingy döt Net). Full notice in
`THIRD-PARTY-NOTICE`.

## Provenance

- model: `claude-opus-4-7` (slug `anthropic-claude-opus-4-7-adaptive`,
  `thinking=adaptive`)
- task: `yaml-1.2-cpp17` (350 tests; 256 valid + 94 invalid)
- vb_version: `0.0.10`
- attempt: `yaml-1.2-cpp17_anthropic-claude-opus-4-7-adaptive_20260430-233833-d8fcf382`,
  turn 2 / submission 3 — strongest single submission across ALL slugs
  on this task (next best by any model is gpt-5.5-xhigh at 0.912)
- per-turn progression on this attempt:
  - turn 0: 333/350 (MCC=0.877)
  - turn 1: 340/350 (MCC=0.931)
  - **turn 2: 349/350 (MCC=0.993)** ← this submission
  - turn 3: 349/350 (MCC=0.993, no further improvement)

## Setup (once)

```bash
brew install gcc libfyaml
```

(On Linux: `apt install g++ libfyaml-utils` or your distro's equivalent.
The homebrew g++ is versioned — `g++-15` at the time of writing — so
don't use plain `g++` on macOS, which resolves to an Apple-clang shim
that lacks `<bits/stdc++.h>`.)

Build the model's submission:

```bash
g++-15 -O2 -o /tmp/solution solution.cpp
```

## Cases

Each `cases/<test_id>/` holds one yaml-test-suite input verbatim. Run
all three verdicts with:

```bash
/tmp/solution         < cases/<test_id>/in.yaml             # model
fy-tool dump --quiet    cases/<test_id>/in.yaml >/dev/null 2>&1 \
    && echo valid || echo invalid                           # libfyaml
```

The third verdict — yaml-test-suite ground truth — is whatever this
README says it is for the case (the test dir contains an `error` file
iff the input is invalid).

### X38W — "Aliases in Flow Objects"

Ground truth: **valid**. Model wrongly rejects it (false negative).
This is the *only* test out of 350 that this submission gets wrong.

```
$ /tmp/solution < cases/X38W/in.yaml
invalid
$ fy-tool dump --quiet cases/X38W/in.yaml >/dev/null 2>&1 && echo valid || echo invalid
valid
```

| source | verdict |
|---|---|
| yaml-test-suite (truth) | valid |
| opus-4-7-adaptive's solution | **invalid** |
| libfyaml | valid |

The 37-byte input is `{ &a [a, &b b]: *b, *a : [c, *b, d]}` — a flow
mapping with two entries:

- key `&a [a, &b b]` (a sequence with anchor `&a`, containing the
  alias-anchor pair) → value `*b` (alias)
- key `*a` (alias to the sequence) → value `[c, *b, d]` (sequence
  containing another alias)

YAML 1.2 explicitly permits arbitrary nodes — including sequences
and aliases — as flow mapping keys. opus's parser appears to require
flow mapping keys to be scalars or limits which kinds of keys are
admissible; both libfyaml and the spec test suite agree the model is
the outlier here.

**Spec rule.** YAML 1.2.2 §7.4.2 "Flow Mappings" makes the
arbitrary-key allowance explicit:

> Note that YAML allows arbitrary nodes to be used as keys.
> In particular, a key may be a sequence or a mapping.
> Thus, without the above restrictions [≤1024 chars, single line],
> practical one-pass parsing would have been impossible to implement.

The relevant production is
`ns-flow-pair-yaml-key-entry(n,c) ::= ns-s-implicit-yaml-key(FLOW-KEY) c-ns-flow-map-separate-value(n,c)`,
where `ns-s-implicit-yaml-key(c) ::= ns-flow-yaml-node(n/a,c) c-s-implicit-json-key`-style filtering applies. `ns-flow-yaml-node` includes
both `c-ns-alias-node` (line `*a`) and `ns-flow-content` which covers
flow sequences (line `&a [a, &b b]`). Spec:
<https://yaml.org/spec/1.2.2/> (navigate to Chapter 7 → 7.4.2).

## Why the entry is worth keeping

This entry pairs with `../yaml-1.2-cpp17-gpt-5.5-repro/` (MCC=0.912,
12 wrong) and `../yaml-1.2-cpp17-kimi-infinite-loop/` (MCC=-1, parser
hangs unconditionally) to bracket the range of model behavior on
this task — from catastrophic failure through high-quality
with-corner-cases through the upper bound of "almost perfect."

The single wrong test is genuinely informative: opus's parser is
otherwise spec-compliant, but it falls over on the
arbitrary-nodes-as-flow-keys rule. That's a corner that requires
careful handling of mutual recursion in the flow-collection
productions, and is one of the trickier parts of the YAML 1.2
grammar.
