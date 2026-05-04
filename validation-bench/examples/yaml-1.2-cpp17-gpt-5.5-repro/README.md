# Reproducing gpt-5.5-xhigh disagreements on yaml-1.2-cpp17

Self-contained recipe for running specific failing test inputs through
the model's own `solution.cpp` and through `libfyaml` (a third-party
YAML 1.2 reference parser), so a reviewer can poke at disagreements
without trusting any of our scoring code.

The test inputs under `cases/<TEST_ID>/in.yaml` are vendored from the
[yaml-test-suite](https://github.com/yaml/yaml-test-suite) data branch
(MIT, Copyright 2016–2020 Ingy döt Net). Full notice in
`THIRD-PARTY-NOTICE`.

## Provenance

- model: `gpt-5.5` (slug `gpt-5.5-xhigh`, `reasoning_effort=xhigh`)
- task: `yaml-1.2-cpp17` (350 tests; 256 valid + 94 invalid)
- vb_version: `0.0.11`
- attempt: `yaml-1.2-cpp17_gpt-5.5-xhigh_20260501-130416-f0e521eb`,
  turn 4 / submission 5 — best gpt-5.5 attempt in the dataset
  (MCC=0.912, TP=251 FN=5 FP=7 TN=87, 12 wrong out of 350)

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
iff the input is invalid; per `data/specs/yaml-1.2/meta.json`).

### 26DV — "Whitespace around colon in mappings"

Ground truth: **valid**. Model wrongly rejects it (false negative).

```
$ /tmp/solution < cases/26DV/in.yaml
invalid
$ fy-tool dump --quiet cases/26DV/in.yaml >/dev/null 2>&1 && echo valid || echo invalid
valid
```

| source | verdict |
|---|---|
| yaml-test-suite (truth) | valid |
| gpt-5.5-xhigh's solution | **invalid** |
| libfyaml | valid |

The input has trailing whitespace after `:` in mapping keys (and one
`top5   :    ` with multi-space padding), which YAML 1.2 explicitly
allows. The model's hand-rolled grammar rejects it; both the spec test
suite and an independent C parser accept it.

**Spec rule.** YAML 1.2.2 §8.2.2 "Block Mappings", production
`c-l-block-map-implicit-value(n) ::= c-mapping-value (s-l+block-node(n,BLOCK-OUT) | (e-node s-l-comments))`.
The `:` (`c-mapping-value`) is followed by either a value node on the
next line or empty content; in both cases the transition from `:` to
the line break goes through `s-separate-in-line ::= s-white+ |
<start-of-line>` (§6.2 "Separation Spaces"), which permits any number
of spaces or tabs. Multi-space padding like `top5   :    ` is
therefore well-formed. Spec: <https://yaml.org/spec/1.2.2/>
(navigate to Chapter 8 → 8.2.2).

### H7J7 — "Node anchor not indented"

Ground truth: **invalid**. Model wrongly accepts it (false positive).

```
$ /tmp/solution < cases/H7J7/in.yaml
valid
$ fy-tool dump --quiet cases/H7J7/in.yaml >/dev/null 2>&1 && echo valid || echo invalid
invalid
```

| source | verdict |
|---|---|
| yaml-test-suite (truth) | invalid |
| gpt-5.5-xhigh's solution | **valid** |
| libfyaml | invalid |

The 21-byte input is `key: &x\n!!map\n  a: b\n` — anchor `&x` declared
on a key whose value lives on the next line, but the value's tag and
content aren't indented under the key. Both the spec and libfyaml
reject this; the model's parser doesn't enforce the indent rule and
accepts it as a well-formed mapping.

**Spec rule.** YAML 1.2.2 §8.2.3 "Block Nodes", production
`s-l+block-collection(n,c) ::= ( s-separate(n+1,c) c-ns-properties(n+1,c) )?  s-l-comments  ( seq-space(n,c) | l+block-mapping(n) )`.
Properties (`c-ns-properties`) on a block collection must appear at
indent `n+1`, where `n` is the parent's indent. Here `key:` is at
column 0 (so `n=0`); the value's properties must be at indent ≥1, but
`!!map` and `&x`'s implied collection appear at column 0. The grammar
fails to match — the input does not parse as a mapping with a
properties-bearing collection value. Spec:
<https://yaml.org/spec/1.2.2/> (navigate to Chapter 8 → 8.2.3).

The two cases together show the failure mode in both directions:
**26DV** is a valid-rejected case, **H7J7** is an invalid-accepted
one. Both are "model is the outlier, two independent reference
implementations agree against it" — distinct from spec corners where
even libfyaml might disagree with the test-suite ground truth.
