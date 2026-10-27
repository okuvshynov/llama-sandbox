#!/usr/bin/env bash
# Routing study: every prompt of prompts/routing-study.tsv through the runtime with
# --log-routing, into data/routing/study/<name>.bin (+ .txt with the completion and timings).
# Prompts already logged are skipped, so an interrupted run can be resumed.
#
#   scripts/routing-study.sh [-n 1024]
# then: ~/venvs/ds-ref/bin/python scripts/route_stats.py data/routing/study/*.bin
set -u
cd "$(dirname "$0")/.."
N=1024
[ "${1:-}" = "-n" ] && N=$2
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
OUT=data/routing/study
mkdir -p "$OUT"
grep -v '^#' prompts/routing-study.tsv | while IFS=$'\t' read -r name category mode prompt; do
  [ -z "$name" ] && continue
  [ -s "$OUT/$name.bin" ] && { echo "skip $name"; continue; }
  if [[ "$prompt" == @* ]]; then  # "@file[:bytes] prefix": prefix (with \n escapes) + the file (its first bytes)
    file=${prompt%% *}; file=${file#@}
    bytes=; [[ "$file" == *:* ]] && { bytes=${file##*:}; file=${file%:*}; }
    prompt="$(printf '%b' "${prompt#* }")$(if [ -n "$bytes" ]; then head -c "$bytes" "$file"; else cat "$file"; fi)"
  fi
  echo "== $name ($category, $mode) $(date +%T)"
  ./bin-hip/ds-completion -m "$M" -p "$prompt" --"$mode" -n "$N" -t 16 --rt "$F8" \
    --log-routing "$OUT/$name.bin.tmp" > "$OUT/$name.txt" 2> "$OUT/$name.err" < /dev/null \
    && mv "$OUT/$name.bin.tmp" "$OUT/$name.bin"
  grep -h "generation" "$OUT/$name.err" | sed 's/.*# prompt/   prompt/'
done
echo "done $(date +%T)"
