#!/bin/bash
set -euo pipefail

# ==============================================================================
# kv-transfer experiment runner
#
# This script expects the following variables to be set by the caller:
#   REF_MODEL   — path to reference model GGUF
#   REF_TAG     — short tag for results directory
#   TARGETS     — array of "tag:path" pairs for target models
#   N_PREDICT   — tokens to generate (default: 512)
#   TEMP        — sampling temperature (default: 0.6)
#   TOP_K       — top-k sampling (default: 40)
#   TOP_P       — top-p sampling (default: 0.95)
#   NGL         — GPU layers (default: 99)
#   THREADS     — inference threads (default: not set, llama.cpp default)
#   PROMPT_FILTER — glob pattern to filter prompts (default: *, all prompts)
#                   e.g. PROMPT_FILTER="01_*" for a single prompt
#
# Usage: source config, then call this script. Or use a wrapper like:
#   ./run-qwen3.5-2b.sh
#   PROMPT_FILTER="01_*" ./run-qwen3.5-35b-a3b.sh   # single prompt
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/build/kv-transfer"
PROMPTS="${SCRIPT_DIR}/prompts"
RESULTS="${SCRIPT_DIR}/results"

# defaults
N_PREDICT="${N_PREDICT:-512}"
TEMP="${TEMP:-0.6}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.95}"
NGL="${NGL:-99}"
THREADS="${THREADS:-0}"
PROMPT_FILTER="${PROMPT_FILTER:-*}"

THREAD_ARGS=()
if [ "$THREADS" -gt 0 ] 2>/dev/null; then
    THREAD_ARGS=(-t "$THREADS")
fi

# --- Validate config ---------------------------------------------------------

if [ -z "${REF_MODEL:-}" ] || [ -z "${REF_TAG:-}" ] || [ ${#TARGETS[@]} -eq 0 ]; then
    echo "ERROR: REF_MODEL, REF_TAG, and TARGETS must be set."
    echo "Use a wrapper script like run-qwen3.5-2b.sh"
    exit 1
fi

if [ ! -x "$BINARY" ]; then
    echo "ERROR: kv-transfer not built. Run: cmake -B build && cmake --build build"
    exit 1
fi

REF_DIR="${RESULTS}/${REF_TAG}"
LOG_DIR="${REF_DIR}/logs"
mkdir -p "$REF_DIR" "$LOG_DIR"

# --- Helper: timestamp -------------------------------------------------------

ts() { date "+%H:%M:%S"; }

# --- Helper: read token counts from .bin file --------------------------------

read_token_counts() {
    python3 -c "
import struct, sys
with open(sys.argv[1], 'rb') as f:
    f.read(8)  # magic
    version = struct.unpack('I', f.read(4))[0]
    f.read(4)  # n_vocab
    f.read(4)  # n_prompts
    f.read(4)  # unused
    f.read(16) # temp, top_p, top_k, seed
    f.read(28) # reserved
    path_len = struct.unpack('i', f.read(4))[0]
    f.read(path_len)
    n_tokens = struct.unpack('i', f.read(4))[0]
    n_prompt = struct.unpack('i', f.read(4))[0]
    print(n_tokens, n_prompt)
" "$1"
}

# Get total size in bytes for a model (handles split shards).
model_size_bytes() {
    python3 -c "
import sys, os, glob, re
path = sys.argv[1]
m = re.match(r'(.+)-(\d+)-of-(\d+)(\.gguf)$', path)
if m:
    prefix, _, n_shards, ext = m.groups()
    pattern = prefix + '-*-of-' + n_shards + ext
    total = sum(os.path.getsize(f) for f in sorted(glob.glob(pattern)))
else:
    total = os.path.getsize(path)
print(total)
" "$1"
}

# --- Run ref for each prompt -------------------------------------------------

echo "=== Reference: ${REF_TAG} ==="
for prompt_file in "$PROMPTS"/${PROMPT_FILTER}.txt; do
    name=$(basename "$prompt_file" .txt)
    ref_bin="${REF_DIR}/${name}-ref.bin"

    if [ -f "$ref_bin" ]; then
        echo "  $(ts) [ref] $name — skip (exists)"
        continue
    fi

    log_file="${LOG_DIR}/${name}-ref.log"
    prompt=$(cat "$prompt_file")
    if ! "$BINARY" ref \
        -m "$REF_MODEL" \
        -p "$prompt" \
        -n "$N_PREDICT" --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" -ngl "$NGL" ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} \
        -o "$ref_bin" > /dev/null 2>"$log_file"; then
        echo "  $(ts) [ref] $name — FAILED (see $log_file)"
        exit 1
    fi
    echo "  $(ts) [ref] $name — done"
done
echo ""

# --- Run target + handoff for each target model ------------------------------

for tgt_entry in "${TARGETS[@]}"; do
    tgt_tag="${tgt_entry%%:*}"
    tgt_model="${tgt_entry#*:}"
    tgt_dir="${REF_DIR}/${tgt_tag}"
    mkdir -p "$tgt_dir"

    echo "=== Target: ${tgt_tag} ==="

    for prompt_file in "$PROMPTS"/${PROMPT_FILTER}.txt; do
        name=$(basename "$prompt_file" .txt)
        ref_bin="${REF_DIR}/${name}-ref.bin"
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        # target
        if [ -f "$tgt_bin" ]; then
            echo "  $(ts) [target] $name — skip"
        else
            log_file="${LOG_DIR}/${name}-${tgt_tag}-target.log"
            if ! "$BINARY" target \
                -m "$tgt_model" \
                -i "$ref_bin" \
                -o "$tgt_bin" -ngl "$NGL" ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} > /dev/null 2>"$log_file"; then
                echo "  $(ts) [target] $name — FAILED (see $log_file)"
                exit 1
            fi
            echo "  $(ts) [target] $name — done"
        fi

        # handoff
        if [ -f "$hoff_bin" ]; then
            echo "  $(ts) [handoff] $name — skip"
        else
            log_file="${LOG_DIR}/${name}-${tgt_tag}-handoff.log"
            if ! "$BINARY" handoff \
                -m-ref "$REF_MODEL" \
                -m-tgt "$tgt_model" \
                -i "$ref_bin" \
                -o "$hoff_bin" -ngl "$NGL" ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} > /dev/null 2>"$log_file"; then
                echo "  $(ts) [handoff] $name — FAILED (see $log_file)"
                exit 1
            fi
            echo "  $(ts) [handoff] $name — done"
        fi
    done
    echo ""
done

# --- Compare and collect results ---------------------------------------------

CSV_FILE="${REF_DIR}/summary.csv"
echo "prompt,ref_model,target_model,ref_bytes,target_bytes,n_prompt,n_gen,kl_target,kl_target_p95,kl_target_p99,top1_target,kl_handoff,kl_handoff_p95,kl_handoff_p99,top1_handoff" > "$CSV_FILE"

REF_BYTES=$(model_size_bytes "$REF_MODEL")
echo "=== Results (ref model: $(echo "$REF_BYTES" | awk '{printf "%.0f MB", $1/1048576}')) ==="
printf "%-20s  %-15s  %7s  %7s  %5s  %10s  %8s  %10s  %8s\n" \
    "prompt" "target" "tgt_MB" "prompt" "gen" "KL(tgt)" "top1%" "KL(hoff)" "top1%"
printf "%-20s  %-15s  %7s  %7s  %5s  %10s  %8s  %10s  %8s\n" \
    "--------------------" "---------------" "-------" "-------" "-----" "----------" "--------" "----------" "--------"

for tgt_entry in "${TARGETS[@]}"; do
    tgt_tag="${tgt_entry%%:*}"
    tgt_model="${tgt_entry#*:}"
    tgt_dir="${REF_DIR}/${tgt_tag}"
    TGT_BYTES=$(model_size_bytes "$tgt_model")
    TGT_MB=$(echo "$TGT_BYTES" | awk '{printf "%.0f", $1/1048576}')

    for prompt_file in "$PROMPTS"/${PROMPT_FILTER}.txt; do
        name=$(basename "$prompt_file" .txt)
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        if [ ! -f "$tgt_bin" ] || [ ! -f "$hoff_bin" ]; then
            continue
        fi

        tgt_out=$("$BINARY" compare -f "$tgt_bin" 2>/dev/null) || true
        hoff_out=$("$BINARY" compare -f "$hoff_bin" 2>/dev/null) || true

        tgt_kl=$(echo "$tgt_out" | grep "KL divergence:" | awk '{print $3}' || true)
        if [ -z "$tgt_kl" ]; then
            echo "  [compare] $name ($tgt_tag) — FAILED, skipping"
            continue
        fi

        tgt_p95=$(echo "$tgt_out" | grep "KL p95:" | awk '{print $3}')
        tgt_p99=$(echo "$tgt_out" | grep "KL p99:" | awk '{print $3}')
        tgt_t1=$(echo "$tgt_out" | grep "Top-1 agree:" | awk '{print $3}')
        hoff_kl=$(echo "$hoff_out" | grep "KL divergence:" | awk '{print $3}')
        hoff_p95=$(echo "$hoff_out" | grep "KL p95:" | awk '{print $3}')
        hoff_p99=$(echo "$hoff_out" | grep "KL p99:" | awk '{print $3}')
        hoff_t1=$(echo "$hoff_out" | grep "Top-1 agree:" | awk '{print $3}')

        read n_tokens_bin n_prompt <<< $(read_token_counts "$REF_DIR/${name}-ref.bin")
        n_gen=$((n_tokens_bin - n_prompt))

        tgt_t1_num=$(echo "$tgt_t1" | tr -d '%')
        hoff_t1_num=$(echo "$hoff_t1" | tr -d '%')

        printf "%-20s  %-15s  %7s  %7s  %5s  %10s  %8s  %10s  %8s\n" \
            "$name" "$tgt_tag" "$TGT_MB" "$n_prompt" "$n_gen" "$tgt_kl" "$tgt_t1" "$hoff_kl" "$hoff_t1"

        echo "${name},${REF_TAG},${tgt_tag},${REF_BYTES},${TGT_BYTES},${n_prompt},${n_gen},${tgt_kl},${tgt_p95},${tgt_p99},${tgt_t1_num},${hoff_kl},${hoff_p95},${hoff_p99},${hoff_t1_num}" >> "$CSV_FILE"
    done
done

echo ""
echo "CSV written to: $CSV_FILE"

# --- Decay analysis ----------------------------------------------------------

DECAY_CSV="${REF_DIR}/decay.csv"
echo "=== Decay analysis ==="
echo "prompt,target,n_prompt,window_start,window_end,kl_target,top1_target,kl_handoff,top1_handoff" > "$DECAY_CSV"

for tgt_entry in "${TARGETS[@]}"; do
    tgt_tag="${tgt_entry%%:*}"
    tgt_dir="${REF_DIR}/${tgt_tag}"

    for prompt_file in "$PROMPTS"/${PROMPT_FILTER}.txt; do
        name=$(basename "$prompt_file" .txt)
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        if [ ! -f "$tgt_bin" ] || [ ! -f "$hoff_bin" ]; then
            continue
        fi

        tmp_csv=$(mktemp)
        log_file="${LOG_DIR}/${name}-${tgt_tag}-decay.log"
        "$BINARY" decay \
            --target "$tgt_bin" --handoff "$hoff_bin" \
            --window 64 --csv "$tmp_csv" > /dev/null 2>"$log_file"

        read n_tokens_bin n_prompt_val <<< $(read_token_counts "$REF_DIR/${name}-ref.bin")

        tail -n +2 "$tmp_csv" | while IFS= read -r line; do
            echo "${name},${tgt_tag},${n_prompt_val},${line}" >> "$DECAY_CSV"
        done
        rm -f "$tmp_csv"
        echo "  $(ts) [decay] ${tgt_tag}/${name} — done"
    done
done
echo "Decay CSV written to: $DECAY_CSV"

# --- Timing summary from logs ------------------------------------------------

echo ""
echo "=== Timing summary (from llama_perf logs) ==="
python3 -c "
import os, re, sys

log_dir = sys.argv[1]
if not os.path.isdir(log_dir):
    print('  No logs directory found'); sys.exit(0)

pat_load  = re.compile(r'load time\s*=\s*([\d.]+)\s*ms')
pat_total = re.compile(r'total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens')

totals = {}  # phase -> { load_ms, total_ms, count }

for fn in sorted(os.listdir(log_dir)):
    if not fn.endswith('.log'):
        continue
    path = os.path.join(log_dir, fn)
    text = open(path).read()

    loads  = pat_load.findall(text)
    totals_m = pat_total.findall(text)
    if not loads or not totals_m:
        continue

    if '-ref.log' in fn:
        phase = 'ref'
        load_ms  = float(loads[0])
        total_ms = float(totals_m[0][0])
        n_tok    = int(totals_m[0][1])
    elif '-target.log' in fn:
        phase = 'target'
        load_ms  = float(loads[0])
        total_ms = float(totals_m[0][0])
        n_tok    = int(totals_m[0][1])
    elif '-handoff.log' in fn:
        # two perf prints: ref context, then target context
        if len(loads) >= 2 and len(totals_m) >= 2:
            for sub_phase, idx in [('handoff-ref', 0), ('handoff-tgt', 1)]:
                d = totals.setdefault(sub_phase, {'load_ms': 0, 'total_ms': 0, 'count': 0})
                d['load_ms']  += float(loads[idx])
                d['total_ms'] += float(totals_m[idx][0])
                d['count']    += 1
        continue
    else:
        continue

    d = totals.setdefault(phase, {'load_ms': 0, 'total_ms': 0, 'count': 0})
    d['load_ms']  += load_ms
    d['total_ms'] += total_ms
    d['count']    += 1

if not totals:
    print('  No perf data found in logs'); sys.exit(0)

print(f'  {\"phase\":<15s} {\"runs\":>5s} {\"avg load\":>10s} {\"avg total\":>10s} {\"load %\":>8s} {\"sum total\":>10s}')
print(f'  {\"-\"*15:<15s} {\"-----\":>5s} {\"----------\":>10s} {\"----------\":>10s} {\"--------\":>8s} {\"----------\":>10s}')
for phase in ['ref', 'target', 'handoff-ref', 'handoff-tgt']:
    d = totals.get(phase)
    if not d or d['count'] == 0:
        continue
    n = d['count']
    avg_load  = d['load_ms'] / n
    avg_total = d['total_ms'] / n
    pct = (d['load_ms'] / d['total_ms'] * 100) if d['total_ms'] > 0 else 0
    sum_total = d['total_ms']
    print(f'  {phase:<15s} {n:>5d} {avg_load:>9.0f}s {avg_total:>9.0f}s {pct:>7.1f}% {sum_total:>9.0f}s'.replace('s ', 'ms ').rstrip('s') + 'ms')

grand_total = sum(d['total_ms'] for d in totals.values())
grand_load  = sum(d['load_ms'] for d in totals.values())
print(f'  {\"\":<15s} {\"\":>5s} {\"\":>10s} {\"\":>10s} {\"\":>8s} {\"----------\":>10s}')
print(f'  {\"TOTAL\":<15s} {\"\":>5s} {\"\":>10s} {\"\":>10s} {grand_load/grand_total*100 if grand_total else 0:>7.1f}% {grand_total:>9.0f}ms')
print(f'  Model loading: {grand_load/1000:.0f}s of {grand_total/1000:.0f}s total ({grand_load/grand_total*100 if grand_total else 0:.0f}% overhead)')
" "$LOG_DIR"
