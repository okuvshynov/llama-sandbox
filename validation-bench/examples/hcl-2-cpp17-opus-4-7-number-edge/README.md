# Lexer-permissive number literal — opus-4-7-adaptive (MCC=0.985, 1 wrong out of 151)

A near-perfect HCL2 validator from Opus that gets stuck at MCC=0.985
across 4 turns. The single failing test is `invalid/expressions/bad-number.hcl`,
whose entire content is:

```hcl
x = 1.2.3
```

The HashiCorp reference parser rejects this outright (`1.2.3` is not
a valid number literal). Opus's lexer accepts the leading `1.2` as a
float and silently drops on the floor whatever follows — so the
overall validator returns `valid` for this input.

This is one of three Opus attempts on `hcl-2-cpp17`; the other two
attempts diverged on the same lexer behavior. The first attempt
(`...-e1f13855`) reached MCC=1.000 by turn 2; this attempt
(`...-ffe19c81`) and a third (`...-1a74dd4b`) both plateaued at 0.985
without ever fixing the trailing-dot-digit case. Confusion matrix at
the plateau: TP=101, FN=0, FP=1, TN=49 — every valid input accepted,
49/50 invalid inputs rejected, with the one false positive being
`bad-number`.

## Provenance

| field | value |
|---|---|
| model | `claude-opus-4-7` (slug `anthropic-claude-opus-4-7-adaptive`, `thinking=adaptive`) |
| task | `hcl-2-cpp17` (151 tests; 101 valid + 50 invalid) |
| vb_version | (whatever was current on 2026-05-12) |
| attempt | `hcl-2-cpp17_anthropic-claude-opus-4-7-adaptive_20260512-160731-ffe19c81`, turn 1 / submission 2 — first turn to hit MCC=0.985 on this attempt; turns 2 and 3 produced cosmetic edits at the same score |
| outcome | TP=101, FN=0, FP=1, TN=49 → 150/151 passed, **MCC = 0.985** |
| per-turn progression on this attempt | turn 0 (sub 1): 0.927 — turn 1 (sub 2): 0.985 — turn 2 (sub 3): 0.985 — turn 3 (sub 4): 0.985 |

## Setup (once)

The reproducer uses two binaries: Opus's `solution.cpp` (built with
any C++17 toolchain) and the validation-bench HCL oracle (Go binary
under `../../scripts/oracles/hcl-check/`).

```bash
# 1. Build the model's submission. Any modern C++17 compiler works;
#    `g++` from homebrew or Linux distro packages is fine.
g++ -std=c++17 -O2 -o /tmp/opus-hcl solution.cpp

# 2. Build the reference oracle (one-time, requires Go ≥ 1.24).
(cd ../../scripts/oracles/hcl-check && go build -o hcl-check .)
```

## Cases

### bad-number — `x = 1.2.3`

Ground truth: **invalid** (HashiCorp's `hashicorp/hcl/v2/hclparse`
rejects with a diagnostic). Opus accepts as valid (false positive).
This is the *only* test out of 151 that this submission gets wrong.

```
$ /tmp/opus-hcl < cases/bad-number/in.hcl
valid
$ ../../scripts/oracles/hcl-check/hcl-check cases/bad-number/in.hcl; echo "  exit=$?"
  exit=1
```

| source | verdict |
|---|---|
| HashiCorp `hclparse` (truth) | invalid |
| opus-4-7-adaptive's solution | **valid** |

## Where the bug is

`solution.cpp` lines 78–96, in `parseNumber()`:

```cpp
bool parseNumber() {
    if (atEnd() || !isDig((unsigned char)peek())) return false;
    while (!atEnd() && isDig((unsigned char)peek())) pos++;
    if (peek() == '.' && isDig((unsigned char)peek(1))) {
        pos++;
        while (!atEnd() && isDig((unsigned char)peek())) pos++;
    }
    if (peek() == 'e' || peek() == 'E') {
        size_t save = pos;
        pos++;
        if (peek() == '+' || peek() == '-') pos++;
        if (!isDig((unsigned char)peek())) {
            pos = save;
        } else {
            while (!atEnd() && isDig((unsigned char)peek())) pos++;
        }
    }
    return true;
}
```

Walking `x = 1.2.3` through `parseNumber()` after the lexer reaches
position 4 (the `1`):

1. `isDig('1')` → true, enter
2. Consume digits → `pos=5` (after `1`)
3. `peek()=='.' && peek(1)=='2'` → enter fractional branch, `pos++` → `pos=6`
4. Consume more digits → `pos=7` (after `2`)
5. `peek()=='.'` (not `e`/`E`) → skip exponent branch
6. `return true`

Result: `parseNumber()` consumes `1.2` and returns. The remaining `.3`
is left in the input stream for the higher-level expression parser
to deal with. Opus's expression parser then treats the trailing
`.3` as either a no-op or as the start of a (silently-malformed)
attribute traversal, and the document parses as a single attribute
`x = 1.2` followed by trailing junk that doesn't trigger an error.

## What Opus would have needed to do

A spec-compliant lexer must commit to a number-or-not decision and
then refuse a *second* fractional dot. Two equally simple fixes:

1. **Lookahead reject**: after consuming the fractional part, if the
   next char is `.` followed by a digit, the whole token is malformed —
   either return false (and let some other alternative handle it,
   though none will) or set an error flag.
2. **Greedy-then-validate**: lex the maximal contiguous
   `[0-9.eE+-]` run, then validate the result against the number
   grammar in a separate pass. Catches `1.2.3`, `1..2`, `1e`, `1e+`,
   `.1.2` uniformly.

The HashiCorp parser uses approach (1) implicitly via its tokenizer's
state machine: once a fractional `.` has been consumed, a second `.`
in the same token transitions to an error state.

## Why the entry is worth keeping

This is a clean, small example of a lexer subtlety that costs a
single MCC point on an otherwise-perfect submission. The bug is
localized to one function and one missing lookahead check.

It also pairs neatly with `../yaml-1.2-cpp17-opus-4-7-strongest/` —
both are Opus's "almost perfect with one informative miss" entries,
on different specs. The YAML miss was a high-level grammar rule
(arbitrary nodes as flow-mapping keys); the HCL miss is a low-level
lexer corner. Together they bracket the kinds of mistakes a strong
model still makes when writing a parser one-shot.

The corresponding test input also illustrates a useful corpus-design
principle: invalid-by-narrow-margin tests (`1.2.3` differs from the
valid `1.2` by only the trailing `.3`) discriminate strong models
from each other, while invalid-by-wide-margin tests (e.g. random
garbage) only discriminate "writes any parser at all" from "doesn't".
The hcl-2 corpus has more discrimination headroom for tests in this
vein — `1..2`, `.1.2`, `1.2e1.5`, `1e`, `1e+` — would all surface
similar lexer-permissiveness bugs.
