#!/usr/bin/env bash
# A/B KLD comparison between two GGUF quants of the same model.
#
# Phase 1: the ground-truth model greedily generates a continuation for every
# prompt in prompts/ (one collect run per prompt, back-to-back so the model
# stays in the page cache). Phase 2: the test model rescores each file's token
# ids with --sim-gen, so batch shapes match generation exactly — the A/A noise
# floor for matching shapes is zero, and every nat of KL is quant signal.
# Phase 3: per-prompt compare, KL(truth || test).
#
# Override any of these via environment variables.
set -euo pipefail
cd "$(dirname "$0")"

TRUTH_MODEL=${TRUTH_MODEL:-$HOME/projects/llms/GLM-5.2-GGUF-UD-Q6_K/UD-Q6_K_XL/GLM-5.2-UD-Q6_K_XL-00001-of-00016.gguf}
TEST_MODEL=${TEST_MODEL:-$HOME/projects/llms/GLM-5.2-GGUF-UD-Q6_K/UD-Q6_K/GLM-5.2-UD-Q6_K-00001-of-00014.gguf}
PROMPTS_DIR=${PROMPTS_DIR:-prompts}
OUT=${OUT:-results/ab-q6kxl-vs-q6k}
N_PREDICT=${N_PREDICT:-256}
TOP_K=${TOP_K:-128}

mkdir -p "$OUT"

for f in "$PROMPTS_DIR"/*.txt; do
    name=$(basename "$f" .txt)
    echo "=== collect (truth): $name"
    ./build/collect -m "$TRUTH_MODEL" -f "$f" -n "$N_PREDICT" -k "$TOP_K" \
        -o "$OUT/truth_$name.bin" \
        > "$OUT/truth_$name.out.txt" 2> "$OUT/truth_$name.log"
done

for f in "$PROMPTS_DIR"/*.txt; do
    name=$(basename "$f" .txt)
    echo "=== rescore (test, --sim-gen): $name"
    ./build/rescore -m "$TEST_MODEL" -i "$OUT/truth_$name.bin" --sim-gen \
        -o "$OUT/test_$name.bin" 2> "$OUT/test_$name.log"
done

for f in "$PROMPTS_DIR"/*.txt; do
    name=$(basename "$f" .txt)
    echo "=== compare KL(truth || test): $name"
    python3 compare.py "$OUT/truth_$name.bin" "$OUT/test_$name.bin"
done | tee "$OUT/compare.txt"
