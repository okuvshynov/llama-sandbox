#!/usr/bin/env bash
# Plain vs speculative decoding at whatever sclk level the cards are set to (set by hand with
# sudo; this script only reads it). The 8 held-out prompts (results/spec-heldout/prompts.txt),
# study30 placement; per run: t/s, mean power of the 4 dies, energy per token; then the median
# verify step of 6 tokens. Writes results/spec-sclk/sclk<level>.txt and the outputs to
# results/spec-sclk/out<level>/ (the clock must not change a single output byte: each output is
# compared with the first level's).
#
#   scripts/spec-sclk.sh
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
PROMPTS=results/spec-heldout/prompts.txt

CARDS=$(for c in /sys/class/drm/card[0-9]; do echo "$(basename "$(readlink -f "$c/device")") ${c##*card}"; done | sort | awk '{print $2}' | tr '\n' ' ')
level() { grep '\*' /sys/class/drm/card$1/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':'; }
L=$(level 0)
for c in 1 2 3; do [ "$(level $c)" = "$L" ] || echo "warning: card$c is at level $(level $c), card0 at $L" >&2; done
mkdir -p results/spec-sclk/out$L
OUT=results/spec-sclk/sclk$L.txt
FIRST=$(ls -d results/spec-sclk/out* | grep -v "out$L\$" | head -1)
: > "$OUT"
log() { echo "$@" | tee -a "$OUT"; }

# run "$@" (stdout to $1, stderr to $2) while sampling the dies' power; prints "t/s watts J/token"
measure() {
  local out=$1 err=$2; shift 2
  local pf
  pf=$(mktemp)
  ( while :; do
      for c in $CARDS; do cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_input; done | tr '\n' ' ' >> "$pf"
      echo >> "$pf"
      sleep 0.1
    done ) &
  local sampler=$!
  "$@" > "$out" 2> "$err"
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  local tps w
  tps=$(grep generation "$err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
  # mean total power of the 4 dies over the whole run (load included: it is the same for both modes)
  w=$(awk '{t = 0; for (c = 1; c <= 4; c++) t += $c / 1e6; s += t; n++} END {printf "%.0f", s / n}' "$pf")
  rm -f "$pf"
  echo "$tps $w $(awk -v t="$tps" -v w="$w" 'BEGIN {printf "%.2f", w / t}')"
}

log "# sclk level $L ($(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $2}')), $(date -Is); study30, 16 threads"
log "# prompt  plain: t/s W J/token  |  spec: t/s W J/token  |  speedup  tokens/step  same output as $FIRST"
while IFS='|' read -r name mode n P; do
  o=results/spec-sclk/out$L
  read -r pt pw pj < <(measure $o/$name-plain.out $o/$name-plain.err \
    ./bin-hip/ds-completion -m "$M" -p "$P" --$mode -n "$n" --rt "$F8" --placement placements/study30.txt)
  read -r st sw sj < <(measure $o/$name-spec.out $o/$name-spec.err \
    ./bin-hip/ds-completion -m "$M" -p "$P" --$mode -n "$n" --rt "$F8" --placement placements/study30.txt --dspark "$D" --spec)
  same=-
  if [ -n "$FIRST" ]; then
    same=$( (cmp -s $o/$name-plain.out $FIRST/$name-plain.out && cmp -s $o/$name-spec.out $FIRST/$name-spec.out) && echo yes || echo NO)
  fi
  log "$(printf '%-13s %6s %4s %6s  |  %6s %4s %6s  |  %s  %s  %s' "$name" "$pt" "$pw" "$pj" "$st" "$sw" "$sj" \
    "$(awk -v a="$pt" -v b="$st" 'BEGIN {printf "%+.0f%%", (b / a - 1) * 100}')" \
    "$(grep -o '[0-9.]* tokens per step' $o/$name-spec.err | cut -d' ' -f1)" "$same")"
done < "$PROMPTS"

# the verify step of 6 tokens at this clock (median over the steps of one run)
./bin-hip/ds-completion -m "$M" -p "Explain the difference between concurrency and parallelism" --chat -n 128 --rt "$F8" \
  --placement placements/study30.txt --dspark "$D" --spec --spec-n 5 --profile 2>&1 >/dev/null | grep "^# verify 6" |
python3 -c '
import re, sys, statistics as st
rows = []
for l in sys.stdin:
    m = re.match(r"# verify 6 tokens: ([\d.]+) ms \(cpu experts ([\d.]+);.*GPU timeline: (.*)", l)
    if m: rows.append((float(m.group(1)), float(m.group(2)), {k: float(v) for k, v in re.findall(r"([a-z_.]+) (\d+\.\d)", m.group(3))}))
print(f"# verify step of 6 tokens (concurrency, {len(rows)} steps): median {st.median(r[0] for r in rows):.1f} ms, host CPU experts {st.median(r[1] for r in rows):.1f} ms")
ks = sorted({k for r in rows for k in r[2]}, key=lambda k: -st.median(r[2].get(k, 0) for r in rows))
print("#   " + "  ".join(f"{k} {st.median(r[2].get(k, 0) for r in rows):.1f}" for k in ks))
' | tee -a "$OUT"
log "# done $(date -Is)"
