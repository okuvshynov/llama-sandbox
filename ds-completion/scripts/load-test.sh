#!/usr/bin/env bash
# Live cost of background expert loads (PLAN.md "Adaptive expert cache", phase 2; README step 30):
# the 8 held-out prompts (results/spec-heldout/prompts.txt), study30, with --load-test R (random
# experts loaded into 8 spare slots per die that nothing reads, at R loads/s in all). The output
# of every run must equal the R = 0 run's (same spare slots, nothing else changes). Writes
# results/load-test/<config>/ and results/load-test/summary.txt.
#
#   scripts/load-test.sh
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
PROMPTS=results/spec-heldout/prompts.txt
OUT=results/load-test
mkdir -p "$OUT"
SUM=$OUT/summary.txt
echo "# load test, sclk $(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':'), $(date -Is); t/s per prompt, then the mean; 'same' = output identical to the 0-loads run" | tee "$SUM"
# config: mode (spec|plain), rate, path
for cfg in "spec 0 direct" "spec 50 direct" "spec 100 direct" "spec 200 direct" "spec 100 pinned" "plain 0 direct" "plain 100 direct"; do
  read -r mode rate pth <<< "$cfg"
  dir=$OUT/$mode-$rate-$pth
  mkdir -p "$dir"
  extra=()
  [ "$mode" = spec ] && extra+=(--dspark "$D" --spec)
  [ "$pth" = pinned ] && extra+=(--load-pinned)
  line="$(printf '%-18s' "$mode $rate $pth")"
  tot=0; n=0; same=yes
  while IFS='|' read -r name m nt P; do
    ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" --placement placements/study30.txt \
      --load-test "$rate" "${extra[@]}" > "$dir/$name.out" 2> "$dir/$name.err" < /dev/null
    tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$dir/$name.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
    cmp -s "$dir/$name.out" "$OUT/$mode-0-direct/$name.out" || same=NO
    line="$line $(printf '%6s' "$tps")"
    tot=$(awk -v a="$tot" -v b="$tps" 'BEGIN {print a + b}'); n=$((n + 1))
  done < "$PROMPTS"
  echo "$line  mean $(awk -v a="$tot" -v n="$n" 'BEGIN {printf "%.2f", a / n}')  same $same  loads $(grep -h 'load test: [0-9]* loads' "$dir"/*.err | awk '{s += $4} END {print s}')" | tee -a "$SUM"
done
echo "# done $(date -Is)" | tee -a "$SUM"
