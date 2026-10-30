#!/usr/bin/env bash
# Gate for chunked prompt processing (README step 37): the prompt in chunks must give the same
# bytes as token by token.
#   - with --canon (placement-invariant: the bigger chunk buffers leave room for fewer expert slots,
#     which changes the fast path's bits): this build at --prefill-chunk 1, 8, 13, 512 against a
#     baseline build that processes the prompt one token at a time;
#   - fast numerics: this build at --prefill-chunk 1 against 512, and 512 without the CPU/GPU
#     overlap (--no-overlap; same placement).
# Prompts: the first 1,200 and 6,000 bytes of prompts/ds-completion-readme-ddf9b43.md (raw text).
#
#   scripts/prefill-gate.sh <baseline ds-completion binary> [-n 32]
set -u
cd "$(dirname "$0")/.."
BASE=$1
N=32
[ "${2:-}" = "-n" ] && N=$3
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
P="--placement placements/study30.txt"
OUT=results/prefill/gate
mkdir -p "$OUT"
fail=0
for bytes in 1200 6000; do
  head -c $bytes prompts/ds-completion-readme-ddf9b43.md > "$OUT/p$bytes.txt"
  run() { "$1" -m "$M" --rt "$F8" $P -f "$OUT/p$bytes.txt" -n "$N" "${@:2}" < /dev/null; }
  tps() { sed -n 's/.*prompt: [0-9]* tokens in [0-9.]* s = \([0-9.]*\) t\/s.*/\1/p' "$1"; }
  run "$BASE" --canon > "$OUT/p$bytes-base-canon.out" 2> "$OUT/p$bytes-base-canon.err"
  line="p$bytes canon vs baseline:"
  for c in 1 8 13 512; do
    run bin-hip/ds-completion --canon --prefill-chunk $c > "$OUT/p$bytes-canon-$c.out" 2> "$OUT/p$bytes-canon-$c.err"
    if cmp -s "$OUT/p$bytes-base-canon.out" "$OUT/p$bytes-canon-$c.out"; then r=same; else r=DIFFERS; fail=1; fi
    line="$line  chunk $c $r ($(tps "$OUT/p$bytes-canon-$c.err") t/s)"
  done
  echo "$line (baseline $(tps "$OUT/p$bytes-base-canon.err") t/s)"
  for c in 1 512; do run bin-hip/ds-completion --prefill-chunk $c > "$OUT/p$bytes-fast-$c.out" 2> "$OUT/p$bytes-fast-$c.err"; done
  run bin-hip/ds-completion --no-overlap > "$OUT/p$bytes-fast-512-no-overlap.out" 2> "$OUT/p$bytes-fast-512-no-overlap.err"
  if cmp -s "$OUT/p$bytes-fast-1.out" "$OUT/p$bytes-fast-512.out" && cmp -s "$OUT/p$bytes-fast-1.out" "$OUT/p$bytes-fast-512-no-overlap.out"; then r=same; else r=DIFFERS; fail=1; fi
  echo "p$bytes fast, chunk 1 vs 512 vs 512 --no-overlap: $r ($(tps "$OUT/p$bytes-fast-1.err") vs $(tps "$OUT/p$bytes-fast-512.err") vs $(tps "$OUT/p$bytes-fast-512-no-overlap.err") t/s)"
done
[ $fail = 0 ] && echo "PASS: every chunked prompt gave the same output" || echo "FAIL"
