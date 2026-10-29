#!/usr/bin/env bash
# The adaptive expert cache, live (PLAN.md "Adaptive expert cache", phases 3-5; README step 32):
# the 8 held-out prompts (results/spec-heldout/prompts.txt), study30 warm start, at the current
# sclk level. Per configuration: t/s per prompt and the mean, the share of routed experts that ran
# on GPUs, and whether every output equals the reference: for --canon runs, the plain static
# --canon run (canonical numerics: no configuration may change a byte, --spec included); fast
# runs are only compared with themselves. Cache runs wait for the expert tensors to be
# registered (--cache-wait) so the cache works from the first token. Writes
# results/cache-test[-sclkN]/<config>/ and .../summary.txt (the suffix when the cards are not at
# level 2).
#
#   scripts/cache-test.sh ["mode numerics cache" ...]   e.g. "spec canon 200" "spec fast static"
# (default: all eight configurations below; a canon run is compared with plain-canon-static of
# the same directory when it exists)
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
LEVEL=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
OUT=results/cache-test$([ "$LEVEL" = 2 ] || echo "-sclk$LEVEL")
mkdir -p "$OUT"
CONFIGS=("plain canon static" "plain fast static" "plain canon 100" "spec fast static" "spec canon static"
         "spec canon 50" "spec canon 100" "spec canon 200")
[ $# -gt 0 ] && CONFIGS=("$@")
SUM=$OUT/summary.txt
{
  echo "# scripts/cache-test.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':'), $(date -Is)"
  echo "# config: t/s per prompt ($(cut -d'|' -f1 results/spec-heldout/prompts.txt | tr '\n' ' ')), mean t/s, routed experts on GPUs, outputs equal to the reference"
} | tee "$SUM"
REF=$OUT/plain-canon-static
for cfg in "${CONFIGS[@]}"; do
  read -r mode num cache <<< "$cfg"
  dir=$OUT/$mode-$num-$cache
  mkdir -p "$dir"
  extra=()
  [ "$mode" = spec ] && extra+=(--dspark "$D" --spec)
  [ "$num" = canon ] && extra+=(--canon)
  [ "$cache" != static ] && extra+=(--cache-loads "$cache" --cache-wait)
  line="$(printf '%-20s' "$mode $num $cache")"; tot=0; n=0; same=yes; gpu=0; all=0
  while IFS='|' read -r name m nt P; do
    ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" --placement placements/study30.txt "${extra[@]}" \
      > "$dir/$name.out" 2> "$dir/$name.err" < /dev/null
    tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$dir/$name.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
    read -r g a < <(grep -o 'on GPUs ([0-9]* of [0-9]*)' "$dir/$name.err" | tr -d '()' | awk '{print $3, $5}')
    gpu=$((gpu + ${g:-0})); all=$((all + ${a:-0}))
    if [ "$num" = canon ]; then
      if [ -f "$REF/$name.out" ]; then cmp -s "$dir/$name.out" "$REF/$name.out" || same=NO
      else cmp -s "$dir/$name.out" "results/cache-test/plain-canon-static/$name.out" || same=NO; fi  # level 2's
    fi
    line="$line $(printf '%6s' "$tps")"
    tot=$(awk -v a="$tot" -v b="$tps" 'BEGIN {print a + b}'); n=$((n + 1))
  done < results/spec-heldout/prompts.txt
  [ "$num" = fast ] && same=-
  echo "$line  mean $(awk -v a="$tot" -v n="$n" 'BEGIN {printf "%.2f", a / n}')  on GPUs $(awk -v g="$gpu" -v a="$all" 'BEGIN {printf "%.1f%%", 100 * g / a}')  same $same" | tee -a "$SUM"
done
echo "# done $(date -Is)" | tee -a "$SUM"
