#!/usr/bin/env bash
# Speed of the placement-invariant numerics (--canon) against the fast kernels: the 8 held-out
# prompts (results/spec-heldout/prompts.txt), study30, plain and --spec, at the current sclk
# level. Writes results/canon-speed.txt (per prompt t/s; means).
#
#   scripts/canon-speed.sh
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
OUT=results/canon-speed.txt
TMP=$(mktemp -d)
{
  echo "# scripts/canon-speed.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':'), $(date -Is)"
  echo "# t/s per prompt (held-out set), study30, 16 threads; then the mean"
} | tee "$OUT"
for cfg in "plain fast" "plain canon" "spec fast" "spec canon"; do
  read -r mode num <<< "$cfg"
  extra=()
  [ "$mode" = spec ] && extra+=(--dspark "$D" --spec)
  [ "$num" = canon ] && extra+=(--canon)
  line="$(printf '%-12s' "$mode $num")"; tot=0; n=0
  while IFS='|' read -r name m nt P; do
    ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" --placement placements/study30.txt "${extra[@]}" \
      > "$TMP/$name.out" 2> "$TMP/$name.err" < /dev/null
    tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$TMP/$name.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
    line="$line $(printf '%6s' "$tps")"
    tot=$(awk -v a="$tot" -v b="$tps" 'BEGIN {print a + b}'); n=$((n + 1))
  done < results/spec-heldout/prompts.txt
  echo "$line  mean $(awk -v a="$tot" -v n="$n" 'BEGIN {printf "%.2f", a / n}')" | tee -a "$OUT"
done
rm -rf "$TMP"
