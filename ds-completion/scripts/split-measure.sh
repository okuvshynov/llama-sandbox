#!/usr/bin/env bash
# The model in two halves on two machines (README step 111): the back (layers 20-39, the head,
# the draft) on BACK, started over ssh for every prompt; the front (layers 0-19) here, driving
# the speculative loop as scripts/spec-measure.sh does. Per prompt: t/s, tokens per step, and
# whether the output equals the plain static --canon run of README step 32. Writes
# results/split/two-machines/<name>.{out,err,log,back.err}.
#
#   [BACK=oleksandr@192.168.2.2] [PROMPTS="cpp-matrix tcp-udp"] [TRACE=1] scripts/split-measure.sh
# TRACE=1 (README step 112): both halves with --trace, merged into one trace on the front's clock
# (scripts/trace_merge.py): results/trace/two-machines/<name>.json.gz and <name>.clock.txt; the
# other files go there too (a traced run is slower: not a measurement).
# BACK needs the same checkout (built), the same ~/llms paths and the same sclk level; ssh to it
# without a prompt (an agent). PROMPTS: held-out names and long-proof (default: all nine).
set -u
cd "$(dirname "$0")/.."
BACK=${BACK:-oleksandr@192.168.2.2}
HOST=${BACK#*@}
PORT=7001
OUT=results/split/two-machines
[ -n "${TRACE:-}" ] && OUT=results/trace/two-machines
COMMON='-m ~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf --rt ~/llms/ds-1017-fp8/dense-fp8.gguf --placement placements/study30.txt --canon --cache-loads 800 --cache-period 8 --cache-wait'
R="ssh -o BatchMode=yes $BACK"
LEVEL=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
mkdir -p "$OUT"
{
  echo "# BACK=$BACK scripts/split-measure.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $LEVEL (back: $($R "grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk" | awk '{print $1}' | tr -d ':')), $(date -Is)"
  echo "# name: generation t/s, tokens per step, front's expert uses on GPUs, output vs results/cache-test/plain-canon-static"
} | tee "$OUT/summary.txt"
for name in ${PROMPTS:-tcp-udp py-lru cpp-matrix sql-report geometry zh-travel ja-email summary-json long-proof}; do
  IFS='|' read -r n m nt P < <(grep "^$name|" results/spec-heldout/prompts.txt)
  extra=""
  if [ "$name" = long-proof ]; then  # scripts/spec-measure.sh's long reasoning prompt
    IFS='|' read -r n m nt P < <(grep -o 'long-proof|think|3000|[^"]*' scripts/spec-measure.sh | head -1)
    extra="--effort max"
  fi
  $R "mkdir -p /tmp/split && : > /tmp/split/back.err"
  tb="" tf=""
  [ -n "${TRACE:-}" ] && tb="--trace /tmp/split/back.json" tf="--trace $OUT/$name.front.json"
  # -f: ssh goes to the background once the back is started; the back leaves when its front does
  $R -f "cd ~/projects/llama-sandbox/ds-completion && exec ./bin-hip/ds-completion $COMMON --layers 20-39 --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf $tb --serve $PORT > /tmp/split/back.out 2> /tmp/split/back.err < /dev/null"
  $R "for i in \$(seq 1 600); do grep -q 'waiting for the front\|what()' /tmp/split/back.err 2>/dev/null && break; sleep 0.5; done"
  eval ./bin-hip/ds-completion $COMMON --layers 0-19 --remote $HOST:$PORT --spec $tf --spec-log "$OUT/$name.log" -p '"$P"' --$m $extra -n "$nt" > "$OUT/$name.out" 2> "$OUT/$name.err" < /dev/null
  $R "for i in \$(seq 1 120); do grep -q '# trace:\|what()' /tmp/split/back.err && break; ${TRACE:+false &&} grep -q 'the front left' /tmp/split/back.err && break; sleep 0.5; done; cat /tmp/split/back.err" > "$OUT/$name.back.err"
  if [ -n "${TRACE:-}" ]; then
    scp -q -o BatchMode=yes "$BACK:/tmp/split/back.json" "$OUT/$name.back.json"
    python3 scripts/trace_merge.py "$OUT/$name.front.json" "$OUT/$name.back.json" -o "$OUT/$name.json.gz" | tee "$OUT/$name.clock.txt"
    python3 -c 'import os, sys; [os.remove(p) for p in sys.argv[1:]]' "$OUT/$name.front.json" "$OUT/$name.back.json"
  fi
  if [ ! -f results/cache-test/plain-canon-static/$name.out ]; then same="n/a"
  elif cmp -s "$OUT/$name.out" results/cache-test/plain-canon-static/$name.out; then same=same; else same=DIFFERENT; fi
  echo "$name: $(grep -o 'generation: .* = [0-9.]* t/s' "$OUT/$name.err" | grep -o '[0-9.]* t/s$'), $(grep -o '[0-9.]* tokens per step' "$OUT/$name.err"), $(grep -o '[0-9.]*% on GPUs' "$OUT/$name.err"), output $same" | tee -a "$OUT/summary.txt"
done
