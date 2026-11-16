#!/usr/bin/env bash
# Draft experts left out for real (README step 125): what do their slots give the main model?
# The 8 held-out prompts with the fastest single-machine settings (study30, --canon, the draft with
# its own draft-length rule, the cache at 800 loads/s). Per prompt, in turn, REPS times:
#   all       every draft expert loaded
#   out       DS_DRAFT_KEEP = the pairs the other seven prompts chose (results/draft-pool/keep/
#             loo-seen/<name>.keep, step 124); the others not loaded, their slots the main model's
#   resident  the same set, the others masked but loaded (DS_DRAFT_KEEP_RESIDENT=1; first rep only):
#             the draft's loss without the slots' gain
# Writes results/draft-pool/leave-out/summary.txt and the records.
#
#   scripts/draft-leave-out.sh
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
OUT=results/draft-pool/leave-out
REPS=${REPS:-2}
LEVEL=$(/usr/bin/grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
MODE=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level)
mkdir -p "$OUT"
{
  echo "# scripts/draft-leave-out.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $([ "$MODE" = manual ] && echo "$LEVEL" || echo "$MODE"), $(date -Is)"
  echo "# config name rep: main experts placed per die | verify steps, drafts accepted | GPU share of expert uses | draft ms, verify ms | generation t/s | output vs results/cache-test/plain-canon-static"
} | tee "$OUT/summary.txt"
for rep in $(seq 1 "$REPS"); do
  while IFS='|' read -r name m nt P; do
    [ -z "$name" ] && continue
    for c in all out resident; do
      [ $c = resident ] && [ "$rep" != 1 ] && continue
      echo "running $c $name $rep ..."
      o=$OUT/$c-$name-$rep
      keep=results/draft-pool/keep/loo-seen/$name.keep
      case $c in
        all) e= ;;
        out) e="DS_DRAFT_KEEP=$keep" ;;
        resident) e="DS_DRAFT_KEEP=$keep DS_DRAFT_KEEP_RESIDENT=1" ;;
      esac
      env $e ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" --placement placements/study30.txt --dspark "$D" --spec --canon \
        --cache-loads 800 --cache-period 8 --cache-wait --record "$o.json" > "$o.out" 2> "$o.err" < /dev/null
      rc=$?
      same=DIFFERS
      cmp -s "$o.out" "results/cache-test/plain-canon-static/$name.out" && same=same
      /usr/bin/grep -q 'the others are left out' "$o.err" && on=out || on=
      python3 - "$o.json" "$c" "$name" "$rep" "$same" "$rc" "$on" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys
path, c, name, rep, same, rc, on = sys.argv[1:8]
try:
    r = json.load(open(path))
except Exception:
    print(f"{c} {name} {rep}: no record (exit code {rc})"); sys.exit()
s, x = r["spec"], r["experts"]
note = "" if rc == "0" else f", exit code {rc}"
if (c == "out") != (on == "out"): note += ", THE SWITCH DID NOT APPLY AS ASKED"
if r["read_bytes"]["gen"]: note += f", read {r['read_bytes']['gen']} bytes in generation"
print(f"{c} {name} {rep}: {' '.join(map(str, x['placed']))} | {s['steps']} {s['accepted']} | {100 * x['gpu_uses'] / (x['gpu_uses'] + x['cpu_uses']):.2f}% | "
      f"{s['draft_ms']:.0f} {s['verify_ms']:.0f} | {r['counters']['gen_tps']:.2f} | {same}{note}")
PY
      rm -f "$o.out" "$o.err"
    done
  done < results/spec-heldout/prompts.txt
done
