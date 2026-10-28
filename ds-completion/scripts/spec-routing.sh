#!/usr/bin/env bash
# Routing under speculative decoding, for the cache simulation (PLAN.md "Adaptive expert cache",
# phase 1): the 8 held-out prompts (results/spec-heldout/prompts.txt) with --spec and
# --log-routing, study30 placement, up to N tokens each, into data/routing/spec/<name>.bin
# (every verify chunk logged, rejected drafts included; route_stats.py). Each output must start
# with the 300/500-token output of results/spec-sclk/out2 (logging changes nothing).
#
#   scripts/spec-routing.sh [-n 1024]
set -u
cd "$(dirname "$0")/.."
N=1024
[ "${1:-}" = "-n" ] && N=$2
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
OUT=data/routing/spec
mkdir -p "$OUT"
while IFS='|' read -r name mode n P; do
  ./bin-hip/ds-completion -m "$M" -p "$P" --$mode -n "$N" --rt "$F8" --placement placements/study30.txt \
    --dspark "$D" --spec --log-routing "$OUT/$name.bin" > "$OUT/$name.out" 2> "$OUT/$name.err" < /dev/null
  ref=results/spec-sclk/out2/$name-spec.out
  same=$(cmp -s -n "$(( $(stat -c %s "$ref") - 1 ))" "$ref" "$OUT/$name.out" && echo yes || echo NO)
  echo "$name: $(grep -o 'generation: [0-9]* tokens' "$OUT/$name.err" | cut -d' ' -f2) tokens, $(grep -o '[0-9.]* tokens per step' "$OUT/$name.err"), prefix of out2: $same"
done < results/spec-heldout/prompts.txt
echo "done $(date +%T)"
