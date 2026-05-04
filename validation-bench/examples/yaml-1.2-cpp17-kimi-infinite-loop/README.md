# Infinite-loop YAML 1.2 validator (kimi-k2.6, MCC = -1.000)

moonshot-kimi-k2.6-thinking produced a YAML 1.2 validator that hangs
on every input — including empty input. The harness's per-test 5-second
timeout fires on all 350 tests, no verdict is ever printed, every test
is scored as a failure.

## Provenance

| field | value |
|---|---|
| model | `moonshot-kimi-k2.6-thinking` |
| task | `yaml-1.2-cpp17` (350 tests; 256 valid + 94 invalid) |
| vb_version | `0.0.11` |
| attempt | `yaml-1.2-cpp17_moonshot-kimi-k2.6-thinking_20260502-023508-95c3f832`, submission 1 |
| outcome | TP=0, FN=256, FP=94, TN=0 → 0/350 passed, **MCC = -1.000** |
| failure breakdown (from `tests.txt`) | 253 `killed by signal 9 (no verdict printed)` + 97 `(exit=125)` |

## How MCC = -1 happens here

The MCC formula is

  MCC = (TP·TN − FP·FN) / sqrt((TP+FP)·(TP+FN)·(TN+FP)·(TN+FN))

When **every** test fails, TP=TN=0 and both off-diagonals (FN, FP) are
non-zero. The numerator becomes `(0·0 − FP·FN) = −FP·FN` and the
denominator simplifies to `sqrt((FP)·(FN)·(FP)·(FN)) = FP·FN`. So
MCC = −1 algebraically — *not* because the parser is "perfectly
anti-correlated" or producing inverted verdicts, but because every
prediction is "no prediction" and the corpus has both classes.

In this submission's case, "every test fails" is realized by the
parser hanging until the harness's per-test `timeout -s KILL 5s`
wrapper kills it (253 of 350) or the parser exiting with rc=125 still
having produced no verdict (97 of 350). Either way, no `valid` /
`invalid` ever lands on stdout.

## Where the bug is

`solution.cpp` lines 1237-1241:

```cpp
bool parse_l_document_prefix() {
    eat(0xFEFF);
    while (parse_l_comment()) {}
    return true;                          // unconditionally true
}
```

Combined with line 1311 (inside `parse_l_yaml_stream`):

```cpp
while (parse_l_document_prefix()) {}      // never terminates
```

`parse_l_document_prefix()` always returns `true` regardless of whether
it consumed any input. The caller's `while (parse_l_document_prefix())`
is therefore an unconditional infinite loop. On any input — including
zero bytes — control never reaches the verdict print at line 1349.
