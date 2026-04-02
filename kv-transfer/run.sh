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
#
# Usage: source config, then call this script. Or use a wrapper like:
#   ./run-qwen3.5-2b.sh
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
for prompt_file in "$PROMPTS"/*.txt; do
    name=$(basename "$prompt_file" .txt)
    ref_bin="${REF_DIR}/${name}-ref.bin"

    if [ -f "$ref_bin" ]; then
        echo "  [ref] $name — skip (exists)"
        continue
    fi

    log_file="${LOG_DIR}/${name}-ref.log"
    echo -n "  [ref] $name — running... "
    prompt=$(cat "$prompt_file")
    if ! "$BINARY" ref \
        -m "$REF_MODEL" \
        -p "$prompt" \
        -n "$N_PREDICT" --temp "$TEMP" --top-k "$TOP_K" --top-p "$TOP_P" -ngl "$NGL" "${THREAD_ARGS[@]}" \
        -o "$ref_bin" > /dev/null 2>"$log_file"; then
        echo "FAILED (see $log_file)"
        exit 1
    fi
    echo "done"
done
echo ""

# --- Run target + handoff for each target model ------------------------------

for tgt_entry in "${TARGETS[@]}"; do
    tgt_tag="${tgt_entry%%:*}"
    tgt_model="${tgt_entry#*:}"
    tgt_dir="${REF_DIR}/${tgt_tag}"
    mkdir -p "$tgt_dir"

    echo "=== Target: ${tgt_tag} ==="

    for prompt_file in "$PROMPTS"/*.txt; do
        name=$(basename "$prompt_file" .txt)
        ref_bin="${REF_DIR}/${name}-ref.bin"
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        # target
        if [ -f "$tgt_bin" ]; then
            echo "  [target] $name — skip"
        else
            log_file="${LOG_DIR}/${name}-${tgt_tag}-target.log"
            echo -n "  [target] $name — running... "
            if ! "$BINARY" target \
                -m "$tgt_model" \
                -i "$ref_bin" \
                -o "$tgt_bin" -ngl "$NGL" "${THREAD_ARGS[@]}" > /dev/null 2>"$log_file"; then
                echo "FAILED (see $log_file)"
                exit 1
            fi
            echo "done"
        fi

        # handoff
        if [ -f "$hoff_bin" ]; then
            echo "  [handoff] $name — skip"
        else
            log_file="${LOG_DIR}/${name}-${tgt_tag}-handoff.log"
            echo -n "  [handoff] $name — running... "
            if ! "$BINARY" handoff \
                -m-ref "$REF_MODEL" \
                -m-tgt "$tgt_model" \
                -i "$ref_bin" \
                -o "$hoff_bin" -ngl "$NGL" "${THREAD_ARGS[@]}" > /dev/null 2>"$log_file"; then
                echo "FAILED (see $log_file)"
                exit 1
            fi
            echo "done"
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

    for prompt_file in "$PROMPTS"/*.txt; do
        name=$(basename "$prompt_file" .txt)
        ref_bin="${REF_DIR}/${name}-ref.bin"
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        if [ ! -f "$tgt_bin" ] || [ ! -f "$hoff_bin" ]; then
            continue
        fi

        tgt_out=$("$BINARY" compare -a "$ref_bin" -b "$tgt_bin" 2>/dev/null)
        hoff_out=$("$BINARY" compare -a "$ref_bin" -b "$hoff_bin" 2>/dev/null)

        tgt_kl=$(echo "$tgt_out" | grep "KL divergence:" | awk '{print $3}')
        tgt_p95=$(echo "$tgt_out" | grep "KL p95:" | awk '{print $3}')
        tgt_p99=$(echo "$tgt_out" | grep "KL p99:" | awk '{print $3}')
        tgt_t1=$(echo "$tgt_out" | grep "Top-1 agree:" | awk '{print $3}')
        hoff_kl=$(echo "$hoff_out" | grep "KL divergence:" | awk '{print $3}')
        hoff_p95=$(echo "$hoff_out" | grep "KL p95:" | awk '{print $3}')
        hoff_p99=$(echo "$hoff_out" | grep "KL p99:" | awk '{print $3}')
        hoff_t1=$(echo "$hoff_out" | grep "Top-1 agree:" | awk '{print $3}')

        read n_tokens_bin n_prompt <<< $(read_token_counts "$ref_bin")
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

    for prompt_file in "$PROMPTS"/*.txt; do
        name=$(basename "$prompt_file" .txt)
        ref_bin="${REF_DIR}/${name}-ref.bin"
        tgt_bin="${tgt_dir}/${name}-target.bin"
        hoff_bin="${tgt_dir}/${name}-handoff.bin"

        if [ ! -f "$tgt_bin" ] || [ ! -f "$hoff_bin" ]; then
            continue
        fi

        echo -n "  ${tgt_tag}/${name}... "
        tmp_csv=$(mktemp)
        log_file="${LOG_DIR}/${name}-${tgt_tag}-decay.log"
        "$BINARY" decay \
            --ref "$ref_bin" --target "$tgt_bin" --handoff "$hoff_bin" \
            --window 64 --temp "$TEMP" --csv "$tmp_csv" > /dev/null 2>"$log_file"

        read n_tokens_bin n_prompt_val <<< $(read_token_counts "$ref_bin")

        tail -n +2 "$tmp_csv" | while IFS= read -r line; do
            echo "${name},${tgt_tag},${n_prompt_val},${line}" >> "$DECAY_CSV"
        done
        rm -f "$tmp_csv"
        echo "done"
    done
done
echo "Decay CSV written to: $DECAY_CSV"
