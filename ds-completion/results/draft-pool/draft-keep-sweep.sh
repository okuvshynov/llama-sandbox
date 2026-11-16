#!/usr/bin/env bash
# Fewer draft experts (README step 124): what does the draft lose when its router may choose only
# part of its 384 experts? The 8 held-out prompts (results/spec-heldout/prompts.txt), study30,
# --canon, no cache, a fixed draft length (--spec-n 5: every draft verified, so the drafts
# accepted per step measure the draft alone and do not depend on timing). Per prompt:
#   all        every expert (no DS_DRAFT_KEEP)
#   loo-N      the N most chosen (layer, expert) pairs of the OTHER seven prompts' routing logs
#              (results/decode-next/draft-route/, step 87); loo-seen: every pair they chose
#   own-N      the N most chosen of the prompt's own log: the best set of that size for it
# (results/draft-pool/draft_keep.py makes the sets). All experts stay loaded: only the router is masked.
# Writes results/draft-pool/keep/<config>/<name>.{json,out,err}, keep files, and summary.txt.
#
#   results/draft-pool/draft-keep-sweep.sh
set -u
cd "$(dirname "$0")/../.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
R=results/decode-next/draft-route
OUT=results/draft-pool/keep
CONFIGS=${CONFIGS:-"all loo-seen loo-192 loo-144 loo-96 own-144 own-96 own-48"}
LEVEL=$(/usr/bin/grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')
MODE=$(cat /sys/class/drm/card0/device/power_dpm_force_performance_level)
mkdir -p "$OUT"
{
  echo "# results/draft-pool/draft-keep-sweep.sh, commit $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo '+changes'), sclk $([ "$MODE" = manual ] && echo "$LEVEL" || echo "$MODE"), $(date -Is)"
  echo "# config name: kept pairs per draft layer | verify steps, drafts accepted (of 5 per step), accepted per step | generation t/s | output vs results/cache-test/plain-canon-static"
} | tee "$OUT/summary.txt"
for c in $CONFIGS; do
  mkdir -p "$OUT/$c"
  while IFS='|' read -r name m nt P; do
    [ -z "$name" ] && continue
    echo "running $c $name ..."   # progress, before the run
    keep=
    case $c in
      loo-seen) keep=$OUT/$c/$name.keep; python3 results/draft-pool/draft_keep.py 384 $(ls $R/*.log | /usr/bin/grep -v "/$name.log") > "$keep" ;;
      loo-*)    keep=$OUT/$c/$name.keep; python3 results/draft-pool/draft_keep.py "${c#loo-}" $(ls $R/*.log | /usr/bin/grep -v "/$name.log") > "$keep" ;;
      own-*)    keep=$OUT/$c/$name.keep; python3 results/draft-pool/draft_keep.py "${c#own-}" "$R/$name.log" > "$keep" ;;
    esac
    o=$OUT/$c/$name
    env ${keep:+DS_DRAFT_KEEP=$keep} ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" --rt "$F8" --placement placements/study30.txt \
      --dspark "$D" --spec --spec-n 5 --canon --record "$o.json" > "$o.out" 2> "$o.err" < /dev/null
    rc=$?
    same=DIFFERS
    cmp -s "$o.out" "results/cache-test/plain-canon-static/$name.out" && same=same
    python3 - "$o.json" "$c" "$name" "$same" "$rc" "${keep:-}" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys
path, c, name, same, rc, keep = sys.argv[1:7]
try:
    r = json.load(open(path))
except Exception:
    print(f"{c} {name}: no record (exit code {rc})"); sys.exit()
per = "128 128 128"
if keep:
    per = open(keep).readline().split("per layer ")[1].split(";")[0].strip("[]").replace(",", "")
s = r["spec"]
print(f"{c} {name}: {per} | {s['steps']} {s['accepted']} {s['accepted'] / s['steps']:.3f} | {r['counters']['gen_tps']:.2f} | {same}"
      + ("" if rc == "0" else f", exit code {rc}") + ("" if r["read_bytes"]["gen"] == 0 else f", read {r['read_bytes']['gen']} bytes in generation"))
PY
  done < results/spec-heldout/prompts.txt
done
