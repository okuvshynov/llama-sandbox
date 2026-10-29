#!/usr/bin/env bash
# Gate for the placement-invariant expert numerics (--canon; PLAN.md "Adaptive expert cache",
# phase 3): with --canon the output must not depend on where the experts run. Per prompt, every
# configuration below must produce the same bytes: no placement (all routed experts on the CPU),
# study30, an oracle placement, study30 on dies 1-3 only, and --spec with the draft (whose memory
# changes the placement) at study30 and oracle. Writes results/canon-gate/.
#
#   scripts/canon-gate.sh [-n 128]
set -u
cd "$(dirname "$0")/.."
N=128
[ "${1:-}" = "-n" ] && N=$2
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
OUT=results/canon-gate
mkdir -p "$OUT"
CONFIGS=(
  "cpu-only|"
  "study30|--placement placements/study30.txt"
  "oracle|--placement placements/oracle-code-rust.txt"
  "study30-dies123|--placement placements/study30.txt --moe-dies 1,2,3"
  "study30-spec|--placement placements/study30.txt --dspark $D --spec"
  "oracle-spec5|--placement placements/oracle-code-rust.txt --dspark $D --spec --spec-n 5"
)
PROMPTS=(
  "tcp-udp|chat|What are the differences between TCP and UDP, and when would you choose each?"
  "py-lru|chat|Implement an LRU cache in Python with O(1) get and put, and add a few unit tests."
  "zh-travel|chat|请为第一次去北京的游客写一份三天的旅行计划。"
)
fail=0
for p in "${PROMPTS[@]}"; do
  IFS='|' read -r name mode P <<< "$p"
  first=
  line="$(printf '%-10s' "$name")"
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r cname cargs <<< "$c"
    o=$OUT/$name-$cname
    # shellcheck disable=SC2086
    ./bin-hip/ds-completion -m "$M" -p "$P" --$mode -n "$N" --rt "$F8" --canon $cargs > "$o.out" 2> "$o.err" < /dev/null
    tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$o.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
    if [ -z "$first" ]; then first=$o.out; same=ref; elif cmp -s "$first" "$o.out"; then same=same; else same=DIFFERS; fail=1; fi
    line="$line  $cname ${tps:-?} t/s $same"
  done
  echo "$line"
done
[ $fail = 0 ] && echo "PASS: every configuration gave the same output" || echo "FAIL"
