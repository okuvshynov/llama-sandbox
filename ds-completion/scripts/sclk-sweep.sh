#!/usr/bin/env bash
# The GPU-specific measurements at whatever sclk level the cards are set to (the level is set by
# hand with sudo; this script only reads it). Writes results/sclk/sclk<level>.txt.
#
#   scripts/sclk-sweep.sh
#
# Per item: the benchmark's own numbers, plus each card's mean and peak power (hwmon
# power1_input, sampled every 100 ms while the item runs).
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
P="Write a haiku about memory bandwidth."

# HIP device n = the n-th GPU in PCI bus order (here dev0..3 = drm card1, card2, card3, card0)
CARDS=$(for c in /sys/class/drm/card[0-9]; do echo "$(basename "$(readlink -f "$c/device")") ${c##*card}"; done | sort | awk '{print $2}' | tr '\n' ' ')
level() { grep '\*' /sys/class/drm/card$1/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':'; }
L=$(level 0)
for c in 1 2 3; do [ "$(level $c)" = "$L" ] || echo "warning: card$c is at level $(level $c), card0 at $L" >&2; done
mkdir -p results/sclk
OUT=results/sclk/sclk$L.txt
: > "$OUT"
log() { echo "$@" | tee -a "$OUT"; }

# run "$@" while sampling power; append the command's filtered output and the power summary
measure() {
  local name=$1 filter=$2; shift 2
  local pf
  pf=$(mktemp)
  ( while :; do
      for c in $CARDS; do cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_input; done | tr '\n' ' ' >> "$pf"
      echo >> "$pf"
      sleep 0.1
    done ) &
  local sampler=$!
  log "== $name"
  "$@" 2>&1 | grep -E "$filter" | tee -a "$OUT"
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  awk '{for (c = 1; c <= 4; c++) { s[c] += $c / 1e6; if ($c / 1e6 > m[c]) m[c] = $c / 1e6 } n++}
       END {printf "  power W (mean/peak) by HIP device: dev0 %.0f/%.0f  dev1 %.0f/%.0f  dev2 %.0f/%.0f  dev3 %.0f/%.0f  (%d samples)\n",
            s[1]/n, m[1], s[2]/n, m[2], s[3]/n, m[3], s[4]/n, m[4], n}' "$pf" | tee -a "$OUT"
  rm -f "$pf"
}

log "# sclk level $L ($(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $2}')), mclk $(grep '\*' /sys/class/drm/card0/device/pp_dpm_mclk | awk '{print $2}'), $(date -Is)"
measure "fp8 GEMV (die 0, layer 2 shapes)" "^attn|^ffn|^engram|^indexer" ./bin-hip/bench-fp8 "$F8" --ms 1,4 --reps 200
measure "runtime kernels alone (die 0)" "us$" ./bin-hip/bench-kernels
measure "GPU experts, one layer (die 0)" "relerr|per layer" ./bin-hip/bench-fp4-moe -m "$M" --layer 20 --reps 300
measure "decode, runtime, dense on die 0, experts on the CPU" "# step 1[3-5]:|generation:" \
  ./bin-hip/ds-completion -m "$M" -p "$P" --chat -n 32 -t 32 --gpu 0 --rt "$F8" --profile
measure "decode, runtime + 13 expert layers on dies 0-3" "# step 1[3-5]:|generation:|uploaded" \
  ./bin-hip/ds-completion -m "$M" -p "$P" --chat -n 32 -t 32 --gpu 0 --rt "$F8" --profile \
  --gpu-moe 0:27,1:36-39,2:32-35,3:28-31
log "# done $(date -Is)"
