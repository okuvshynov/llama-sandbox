#!/usr/bin/env bash
# ref_batch.sh — run the reference model on all prompts, save logits.
# Run this once per reference model. The output .qmlog files are then reused
# by target_batch.sh for any number of target model variants.
#
# Usage: ./ref_batch.sh <ref_model.gguf> [outdir] [n_predict] [prompts_file]
#   ref_model     path to the reference model (e.g. Q8_K or fp16)
#   outdir        output directory            (default: batch_run)
#   n_predict     tokens to generate          (default: 512)
#   prompts_file  one prompt per line          (default: prompts.txt)
set -euo pipefail

QMATCH=./build/quant-sampling

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ref_model.gguf> [outdir] [n_predict] [prompts_file]" >&2
    exit 1
fi

REF_MODEL="$1"
OUTDIR="${2:-batch_run}"
N_PREDICT="${3:-512}"
PROMPTS_FILE="${4:-prompts.txt}"

if [[ ! -f "$REF_MODEL" ]]; then
    echo "error: ref model not found: $REF_MODEL" >&2
    exit 1
fi
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo "error: prompts file not found: $PROMPTS_FILE" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

# Read prompts into array — while-read is compatible with bash 3.2+ (macOS default)
PROMPTS=()
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    PROMPTS+=("$line")
done < "$PROMPTS_FILE"
N_PROMPTS=${#PROMPTS[@]}

echo "=== Generate: $N_PROMPTS prompts, $N_PREDICT tokens each ==="
echo "    ref model: $REF_MODEL"
echo "    output:    $OUTDIR/ref_NN.qmlog"
echo ""

for ((i=0; i<N_PROMPTS; i++)); do
    num=$(printf "%02d" "$((i+1))")
    out="$OUTDIR/ref_${num}.qmlog"
    echo "--- Prompt $((i+1)) / $N_PROMPTS ---"
    $QMATCH ref \
        -m "$REF_MODEL" \
        -p "${PROMPTS[$i]}" \
        -o "$out" \
        -n "$N_PREDICT" \
        -ngl 99
    echo ""
done

echo "=== Done. Generated $N_PROMPTS reference logit files in $OUTDIR/ ==="
