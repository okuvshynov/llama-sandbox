#!/usr/bin/env bash
# Speculation with the adaptive cache, logged per step (README step 48): the 8 held-out prompts
# (results/spec-heldout/prompts.txt) and one long reasoning prompt, `--spec --canon
# --cache-loads 200 --cache-wait --spec-log`, study30 warm start, at the current sclk level.
# Per prompt: t/s, tokens per step, and whether the output equals the plain static --canon run of
# README step 32 (canonical numerics: speculation and the cache may not change a byte). Writes
# results/spec-measure[-sclkN]/<name>.{out,err,log}; analysis: scripts/spec_analysis.py.
#
#   [CACHE="--cache-loads 800 --cache-period 8"] scripts/spec-measure.sh
# CACHE: the cache options (default --cache-loads 200); a non-default one writes to
# results/spec-measure[-sclkN]-<options without spaces>/
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
LEVEL=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
CACHE=${CACHE:---cache-loads 200}
OUT=results/spec-measure$([ "$LEVEL" = 2 ] || echo "-sclk$LEVEL")$([ "$CACHE" = "--cache-loads 200" ] || echo "-$(echo $CACHE | tr -d ' -')")
mkdir -p "$OUT"
LONG="long-proof|think|3000|Prove that there are infinitely many primes of the form 4k+3. Then find all pairs of non-negative integers (x, y) with x^2 + y^2 = 2025, and explain how the count follows from the prime factorization of 2025."
{
  echo "# CACHE=\"$CACHE\" scripts/spec-measure.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $LEVEL, $(date -Is)"
  echo "# name: generation t/s, tokens per step, drafts accepted per step, output vs results/cache-test/plain-canon-static"
} | tee "$OUT/summary.txt"
{ cat results/spec-heldout/prompts.txt; echo "$LONG"; } | while IFS='|' read -r name m nt P; do
  [ -z "$name" ] && continue
  extra=()
  [ "$name" = long-proof ] && extra+=(--effort max)  # the held-out ones as in step 32 (default effort)
  ./bin-hip/ds-completion -m "$M" -p "$P" --$m "${extra[@]}" -n "$nt" --rt "$F8" --placement placements/study30.txt \
    --dspark "$D" --spec --canon $CACHE --cache-wait --spec-log "$OUT/$name.log" \
    > "$OUT/$name.out" 2> "$OUT/$name.err" < /dev/null
  tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$OUT/$name.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
  tps_step=$(grep -o '[0-9.]* tokens per step' "$OUT/$name.err" | awk '{print $1}')
  acc=$(grep -o '[0-9.]* drafts accepted per step' "$OUT/$name.err" | awk '{print $1}')
  ref=results/cache-test/plain-canon-static/$name.out
  same=n/a
  [ -f "$ref" ] && { cmp -s "$OUT/$name.out" "$ref" && same=same || same=DIFFERS; }
  echo "$name: ${tps:-?} t/s, ${tps_step:-?} tokens per step, ${acc:-?} accepted, output $same" | tee -a "$OUT/summary.txt"
done
