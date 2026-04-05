#!/bin/bash
set -euo pipefail

# Compare run.sh (per-prompt model loading) vs batch (models loaded once).
# Usage: ./compare-methods.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/build/kv-transfer"
PROMPTS="${SCRIPT_DIR}/prompts"

# --- Config (edit these) -----------------------------------------------------

REF_MODEL="$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
TGT_MODEL="$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf"
REF_TAG="qwen3.5-35b-a3b-ud-q8_k_xl"
TGT_TAG="ud-q6_k_xl"
N_PREDICT=2048
TEMP=0.6
TOP_K=40
TOP_P=0.95
NGL=99

# --- Setup -------------------------------------------------------------------

DIR_INDIVIDUAL=$(mktemp -d)
DIR_BATCH=$(mktemp -d)
trap "echo 'Temp dirs: $DIR_INDIVIDUAL (individual) $DIR_BATCH (batch)'" EXIT

REF_DIR_I="${DIR_INDIVIDUAL}/${REF_TAG}"
TGT_DIR_I="${REF_DIR_I}/${TGT_TAG}"
LOG_DIR_I="${REF_DIR_I}/logs"
mkdir -p "$TGT_DIR_I" "$LOG_DIR_I"

echo "=== Compare: individual commands vs batch ==="
echo "  ref model: $(basename "$REF_MODEL")"
echo "  tgt model: $(basename "$TGT_MODEL")"
echo "  prompts:   $(ls "$PROMPTS"/*.txt | wc -l | tr -d ' ')"
echo "  individual results: $DIR_INDIVIDUAL"
echo "  batch results:      $DIR_BATCH"
echo ""

# --- Method 1: Individual commands (like run.sh) ----------------------------

echo "=============================="
echo "=== Method 1: Individual   ==="
echo "=============================="
echo ""

TIME_I_START=$(date +%s)

for prompt_file in "$PROMPTS"/*.txt; do
    name=$(basename "$prompt_file" .txt)
    ref_bin="${REF_DIR_I}/${name}-ref.bin"
    ref_txt="${REF_DIR_I}/${name}-ref.txt"

    # ref
    echo -n "  [ref] $name... "
    prompt=$(cat "$prompt_file")
    "$BINARY" ref \
        -m "$REF_MODEL" -p "$prompt" \
        -n "$N_PREDICT" --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" \
        -ngl "$NGL" -o "$ref_bin" \
        > /dev/null 2>"${LOG_DIR_I}/${name}-ref.log"
    echo "done"

    # target
    echo -n "  [target] $name... "
    "$BINARY" target \
        -m "$TGT_MODEL" -i "$ref_bin" \
        -o "${TGT_DIR_I}/${name}-target.bin" -ngl "$NGL" \
        > /dev/null 2>"${LOG_DIR_I}/${name}-${TGT_TAG}-target.log"
    echo "done"

    # handoff
    echo -n "  [handoff] $name... "
    "$BINARY" handoff \
        -m-ref "$REF_MODEL" -m-tgt "$TGT_MODEL" -i "$ref_bin" \
        -o "${TGT_DIR_I}/${name}-handoff.bin" -ngl "$NGL" \
        > /dev/null 2>"${LOG_DIR_I}/${name}-${TGT_TAG}-handoff.log"
    echo "done"

    echo ""
done

TIME_I_END=$(date +%s)
TIME_I=$((TIME_I_END - TIME_I_START))

echo "=== Individual: ${TIME_I}s ==="
echo ""

# --- Method 2: Batch command -------------------------------------------------

echo "=============================="
echo "=== Method 2: Batch        ==="
echo "=============================="
echo ""

TIME_B_START=$(date +%s)

"$BINARY" batch \
    -m-ref "$REF_MODEL" -m-tgt "$TGT_MODEL" \
    --ref-tag "$REF_TAG" --tgt-tag "$TGT_TAG" \
    --prompts "$PROMPTS" -o "$DIR_BATCH" \
    -n "$N_PREDICT" --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" \
    -ngl "$NGL" 2>&1 | grep -E '^\s*(---|\[|batch:)'

TIME_B_END=$(date +%s)
TIME_B=$((TIME_B_END - TIME_B_START))

echo ""
echo "=== Batch: ${TIME_B}s ==="
echo ""

# --- Compare results ---------------------------------------------------------

echo "=============================="
echo "=== Comparison             ==="
echo "=============================="
echo ""

REF_DIR_B="${DIR_BATCH}/${REF_TAG}"
TGT_DIR_B="${REF_DIR_B}/${TGT_TAG}"

printf "%-25s  %10s %10s  %10s %10s  %s\n" \
    "prompt" "tgt_kl(I)" "tgt_kl(B)" "hoff_kl(I)" "hoff_kl(B)" "match"
printf "%-25s  %10s %10s  %10s %10s  %s\n" \
    "-------------------------" "----------" "----------" "----------" "----------" "-----"

all_match=true
for prompt_file in "$PROMPTS"/*.txt; do
    name=$(basename "$prompt_file" .txt)

    tgt_i="${TGT_DIR_I}/${name}-target.bin"
    tgt_b="${TGT_DIR_B}/${name}-target.bin"
    hoff_i="${TGT_DIR_I}/${name}-handoff.bin"
    hoff_b="${TGT_DIR_B}/${name}-handoff.bin"

    # get KL from each
    tgt_kl_i=$("$BINARY" compare -f "$tgt_i" 2>/dev/null | grep "KL divergence:" | awk '{print $3}')
    tgt_kl_b=$("$BINARY" compare -f "$tgt_b" 2>/dev/null | grep "KL divergence:" | awk '{print $3}')
    hoff_kl_i=$("$BINARY" compare -f "$hoff_i" 2>/dev/null | grep "KL divergence:" | awk '{print $3}')
    hoff_kl_b=$("$BINARY" compare -f "$hoff_b" 2>/dev/null | grep "KL divergence:" | awk '{print $3}')

    match="OK"
    if [ "$tgt_kl_i" != "$tgt_kl_b" ] || [ "$hoff_kl_i" != "$hoff_kl_b" ]; then
        match="DIFF"
        all_match=false
    fi

    printf "%-25s  %10s %10s  %10s %10s  %s\n" \
        "$name" "$tgt_kl_i" "$tgt_kl_b" "$hoff_kl_i" "$hoff_kl_b" "$match"
done

echo ""
echo "=== Timing ==="
echo "  Individual: ${TIME_I}s"
echo "  Batch:      ${TIME_B}s"
speedup=$(python3 -c "print(f'{$TIME_I/$TIME_B:.2f}x')" 2>/dev/null || echo "N/A")
echo "  Speedup:    ${speedup}"
echo ""

if $all_match; then
    echo "=== All results match ==="
else
    echo "=== Some results differ (expected if sampling is non-deterministic) ==="
fi

# --- Parse perf logs ---------------------------------------------------------

echo ""
echo "=== Individual method: perf breakdown ==="
python3 -c "
import os, re

log_dir = '$LOG_DIR_I'
pat_load  = re.compile(r'load time\s*=\s*([\d.]+)\s*ms')
pat_total = re.compile(r'total time\s*=\s*([\d.]+)\s*ms')

totals = {}
for fn in sorted(os.listdir(log_dir)):
    if not fn.endswith('.log'): continue
    text = open(os.path.join(log_dir, fn)).read()
    loads = pat_load.findall(text)
    tots  = pat_total.findall(text)
    if not loads or not tots: continue
    if '-ref.log' in fn: phase = 'ref'
    elif '-target.log' in fn: phase = 'target'
    elif '-handoff.log' in fn:
        if len(loads) >= 2:
            for p, idx in [('handoff-ref', 0), ('handoff-tgt', 1)]:
                d = totals.setdefault(p, [0, 0, 0])
                d[0] += float(loads[idx]); d[1] += float(tots[idx]); d[2] += 1
        continue
    else: continue
    d = totals.setdefault(phase, [0, 0, 0])
    d[0] += float(loads[0]); d[1] += float(tots[0]); d[2] += 1

print(f\"  {'phase':<15s} {'runs':>5s} {'avg load':>10s} {'avg total':>10s} {'load%':>7s} {'sum':>10s}\")
for phase in ['ref', 'target', 'handoff-ref', 'handoff-tgt']:
    d = totals.get(phase)
    if not d or d[2] == 0: continue
    n = d[2]; al = d[0]/n; at = d[1]/n; pct = d[0]/d[1]*100 if d[1] else 0
    print(f'  {phase:<15s} {n:>5d} {al:>9.0f}ms {at:>9.0f}ms {pct:>6.1f}% {d[1]:>9.0f}ms')
grand_l = sum(d[0] for d in totals.values())
grand_t = sum(d[1] for d in totals.values())
print(f'  Model loading: {grand_l/1000:.0f}s of {grand_t/1000:.0f}s ({grand_l/grand_t*100:.0f}%)')
"
