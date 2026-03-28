#!/usr/bin/env bash
# target_batch.sh — run a target model on the reference token sequences,
# save logits, and write a manifest for compare_batch.sh.
# Run this once per target model variant.
#
# Usage: ./target_batch.sh <target_model.gguf> <tag> [outdir]
#   quant_model   path to the quantized model (e.g. Q2_K, Q4_K, Q5_K)
#   tag           short label for this variant, used in filenames
#                 (e.g. "q2", "q4", "q5") — must be unique per variant
#   outdir        directory containing ref_NN.qmlog files (default: batch_run)
set -euo pipefail

QMATCH=./build/quant-sampling

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <target_model.gguf> <tag> [outdir]" >&2
    exit 1
fi

TARGET_MODEL="$1"
TAG="$2"
OUTDIR="${3:-batch_run}"

if [[ ! -f "$TARGET_MODEL" ]]; then
    echo "error: target model not found: $TARGET_MODEL" >&2
    exit 1
fi
if [[ ! -d "$OUTDIR" ]]; then
    echo "error: outdir not found: $OUTDIR (run ref_batch.sh first)" >&2
    exit 1
fi

# Discover reference files produced by ref_batch.sh
REF_FILES=()
for f in "$OUTDIR"/ref_*.qmlog; do
    [[ -f "$f" ]] && REF_FILES+=("$f")
done
N_PROMPTS=${#REF_FILES[@]}

if [[ $N_PROMPTS -eq 0 ]]; then
    echo "error: no ref_*.qmlog files found in $OUTDIR" >&2
    exit 1
fi

echo "=== Target [$TAG]: $N_PROMPTS prompts ==="
echo "    target model: $TARGET_MODEL"
echo "    output:      $OUTDIR/${TAG}_NN.qmlog"
echo ""

for ((i=0; i<N_PROMPTS; i++)); do
    num=$(printf "%02d" "$((i+1))")
    ref="$OUTDIR/ref_${num}.qmlog"
    out="$OUTDIR/${TAG}_${num}.qmlog"
    echo "--- Prompt $((i+1)) / $N_PROMPTS ---"
    $QMATCH target \
        -m "$TARGET_MODEL" \
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
