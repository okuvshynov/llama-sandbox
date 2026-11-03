#!/usr/bin/env bash
# Router lookahead (README step 63): the 8 held-out prompts with --spec and the tuned cache, each
# layer's router input also scored by the next DS_ROUTE_AHEAD layers' routers. Per prompt the
# runtime's report (results/route-ahead/<name>.err, the "router lookahead" table) and a log of
# every generation step (data/route-ahead/<name>.bin) for scripts/prefetch_sim.py.
#
#   scripts/route-ahead.sh [K]   (K: layers ahead, default 4)
set -u
cd "$(dirname "$0")/.."
K=${1:-4}
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
OUT=results/route-ahead
LOGS=data/route-ahead
mkdir -p "$OUT" "$LOGS"
LEVEL=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
echo "# scripts/route-ahead.sh $K, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $LEVEL, $(date -Is)" | tee "$OUT/summary.txt"
while IFS='|' read -r name m nt P; do
  [ -z "$name" ] && continue
  DS_ROUTE_AHEAD=$K DS_ROUTE_LOG="$LOGS/$name.bin" ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" \
    --placement placements/study30.txt --dspark "$D" --spec --canon --cache-loads 800 --cache-period 8 --cache-wait \
    > "$OUT/$name.out" 2> "$OUT/$name.err" < /dev/null
  same=DIFFERS
  cmp -s "$OUT/$name.out" "results/cache-test/plain-canon-static/$name.out" && same=same
  echo "$name: output $same" | tee -a "$OUT/summary.txt"
  grep -A$((K + 2)) "router lookahead" "$OUT/$name.err" >> "$OUT/summary.txt"
done < results/spec-heldout/prompts.txt
