#!/usr/bin/env bash
# The HTTP server (server/ds_server.py) with the fastest single-machine settings and everything
# it can log, for a client on this machine such as the pi harness (docs/pi.md; README step 167).
# Ctrl-C stops the server and the engine.
#
#   scripts/serve.sh [more ds_server.py options, e.g. --seed 3 --host 0.0.0.0]
#
# Every start writes a new directory, $DS_LOGS/<date>-<time>/ (default ~/ds-serve-logs: outside
# the repository, the files hold the prompts and the replies):
#   info.txt        the commit, the command, the GPU clock, the page cache, pi's models.json
#   server.log      the server's and the engine's lines (also on the terminal): a line per request,
#                   the engine's "# context:" lines (where a request started, docs/requests.md)
#   requests.jsonl  --log: one JSON line per request (docs/server.md "The log")
#   dump/           --dump: per request its body, the rendered prompt, the raw reply, the message
#   metrics.prom    GET /metrics, taken again every 60 s (the counters start at 0 with every start)
#
# PORT (default 8080): as baseUrl in pi's models.json.
# MAX_CTX (default 262144): the engine's --max-ctx. A position costs ~6.4 KB on GPU 0: 262,144
# take 1.7 GB of the room the home die otherwise gives to experts, 1,000,000 take 6.4 GB.
# contextWindow in pi's models.json (~/.pi/agent/models.json, or under $PI_CODING_AGENT_DIR) must not be larger, or pi does not
# compact a conversation before the engine refuses its prompt: the script warns if it is.
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
PORT=${PORT:-8080}
PI_MODELS=${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/models.json
MAX_CTX=${MAX_CTX:-262144}
# contextWindow of the provider in pi's models.json whose baseUrl ends with the port (empty: none found)
PI_CTX=$(python3 -c 'import json, sys
print(max(m["contextWindow"] for p in json.load(open(sys.argv[1]))["providers"].values() if str(p.get("baseUrl", "")).endswith(":" + sys.argv[2]) for m in p["models"]))' "$PI_MODELS" "$PORT" 2>/dev/null)
[ "$MAX_CTX" -gt 1048576 ] && { echo "MAX_CTX $MAX_CTX: the model's limit is 1048576" >&2; exit 2; }

DIR=${DS_LOGS:-$HOME/ds-serve-logs}/$(date +%Y%m%d-%H%M%S)
mkdir -p "$DIR/dump" || exit 1
ENGINE=(bin-hip/ds-completion -m "$M" --rt "$F8" --placement placements/study30.txt --dspark "$D" --spec --canon
        --cache-loads 800 --cache-period 8 --cache-wait --max-ctx "$MAX_CTX" --requests)
SERVER=(python3 server/ds_server.py --port "$PORT" --log "$DIR/requests.jsonl" --dump "$DIR/dump" "$@" -- "${ENGINE[@]}")
cached_gb=$(awk '/^Cached:/ {printf "%d", $2 / 1048576}' /proc/meminfo)
{
  echo "# scripts/serve.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), $(date -Is)"
  echo "# command: ${SERVER[*]}"
  echo "# max-ctx $MAX_CTX; contextWindow in pi's models.json: ${PI_CTX:-none found} ($PI_MODELS); pi $(pi --version 2>/dev/null | head -1)"
  for c in /sys/class/drm/card[0-9]/device; do
    [ -f "$c/pp_dpm_sclk" ] && echo "# $(basename "$(dirname "$c")"): $(cat "$c/power_dpm_force_performance_level"), sclk $(grep '\*' "$c/pp_dpm_sclk" | tr -d '*')"
  done
  echo "# page cache: $cached_gb GiB (the model files are 488 GiB)"
  [ -f "$PI_MODELS" ] && { echo "# $PI_MODELS:"; sed 's/^/#   /' "$PI_MODELS"; }
} > "$DIR/info.txt"
echo "# logs: $DIR" >&2
# after a reboot the model is read from the disk as it is used, several times slower (README step 155)
[ "$cached_gb" -lt 450 ] && echo "# WARNING: the page cache holds $cached_gb GB, less than the model: first read it with  cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null  (~3 min), or the first requests are slow" >&2
[ -n "$PI_CTX" ] && [ "$PI_CTX" -gt "$MAX_CTX" ] &&
  echo "# WARNING: contextWindow in $PI_MODELS is $PI_CTX, the engine's --max-ctx $MAX_CTX: pi will not compact in time and a longer prompt is refused. Set contextWindow to $MAX_CTX there, or start with MAX_CTX=$PI_CTX" >&2
grep -q auto /sys/class/drm/card[0-9]/device/power_dpm_force_performance_level 2>/dev/null &&
  echo "# WARNING: a GPU's clock is on auto (after a reboot): the speeds will not be the documented ones (docs/reproduce.md, 'GPU clock')" >&2

# /metrics every 60 s, the file replaced whole: what is left after a stop is at most a minute old
(while kill -0 $$ 2>/dev/null && sleep 60; do
   curl -sf --max-time 5 "http://127.0.0.1:$PORT/metrics" -o "$DIR/metrics.tmp" && mv "$DIR/metrics.tmp" "$DIR/metrics.prom"
 done) &
SNAP=$!
trap 'kill $SNAP 2>/dev/null' EXIT
"${SERVER[@]}" 2> >(tee -a "$DIR/server.log" >&2)
rc=$?
echo "# logs: $DIR" >&2
exit $rc
