#!/usr/bin/env bash
# collect_batch.sh — run a quantized model on the reference token sequences,
# collect logits, and write a manifest for compare_batch.sh.
# Run this once per quantized model variant.
#
# Usage: ./collect_batch.sh <quant_model.gguf> <tag> [outdir]
#   quant_model   path to the quantized model (e.g. Q2_K, Q4_K, Q5_K)
#   tag           short label for this variant, used in filenames
#                 (e.g. "q2", "q4", "q5") — must be unique per variant
#   outdir        directory containing ref_NN.qmlog files (default: batch_run)
set -euo pipefail

QMATCH=./build/quant-sampling

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <quant_model.gguf> <tag> [outdir]" >&2
    exit 1
fi

QUANT_MODEL="$1"
TAG="$2"
OUTDIR="${3:-batch_run}"

if [[ ! -f "$QUANT_MODEL" ]]; then
    echo "error: quant model not found: $QUANT_MODEL" >&2
    exit 1
fi
if [[ ! -d "$OUTDIR" ]]; then
    echo "error: outdir not found: $OUTDIR (run generate_batch.sh first)" >&2
    exit 1
fi

# Discover reference files produced by generate_batch.sh
REF_FILES=()
for f in "$OUTDIR"/ref_*.qmlog; do
    [[ -f "$f" ]] && REF_FILES+=("$f")
done
N_PROMPTS=${#REF_FILES[@]}

if [[ $N_PROMPTS -eq 0 ]]; then
    echo "error: no ref_*.qmlog files found in $OUTDIR" >&2
    exit 1
fi

echo "=== Collect [$TAG]: $N_PROMPTS prompts ==="
echo "    quant model: $QUANT_MODEL"
echo "    output:      $OUTDIR/${TAG}_NN.qmlog"
echo ""

for ((i=0; i<N_PROMPTS; i++)); do
    num=$(printf "%02d" "$((i+1))")
    ref="$OUTDIR/ref_${num}.qmlog"
    out="$OUTDIR/${TAG}_${num}.qmlog"
    echo "--- Prompt $((i+1)) / $N_PROMPTS ---"
    $QMATCH collect \
        -m "$QUANT_MODEL" \
        -i "$ref" \
        -o "$out" \
        -ngl 99
    echo ""
done

# Write manifest so compare_batch.sh can find the pairs
MANIFEST="$OUTDIR/manifest_${TAG}.txt"
> "$MANIFEST"
for ((i=0; i<N_PROMPTS; i++)); do
    num=$(printf "%02d" "$((i+1))")
    echo "$OUTDIR/ref_${num}.qmlog  $OUTDIR/${TAG}_${num}.qmlog" >> "$MANIFEST"
done

echo "=== Done. Manifest written to $MANIFEST ==="
echo "    Run: ./compare_batch.sh $TAG"
