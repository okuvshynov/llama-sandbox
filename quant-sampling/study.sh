#!/bin/bash
set -euo pipefail

# ==============================================================================
# quant-sampling study runner
#
# Downloads models from HuggingFace and runs the full ref -> target -> compare
# pipeline. Edit the configuration section below for each model family.
#
# Prerequisites:
#   - quant-sampling binary built (cmake --build build)
#   - huggingface-cli installed (pip install huggingface-hub)
#   - Enough disk space for the models
#
# Usage:
#   ./study.sh                    # run full pipeline
#   ./study.sh --skip-download    # skip model downloads
#   ./study.sh --skip-ref         # skip ref generation (reuse existing ref.bin)
#
# Model config format:
#   For single-file models:
#     "tag:filename.gguf"
#   For split/multi-file models:
#     "tag:pattern:first_shard.gguf"
#   where pattern is passed to huggingface-cli --include (e.g. "Qwen*BF16*")
#   and first_shard is the -00001-of-*.gguf file passed to llama.cpp.
# ==============================================================================

# --- Configuration -----------------------------------------------------------

HF_REPO="unsloth/Qwen3.5-2B-GGUF"

# Reference model: "pattern:first_shard" or just "filename" for single-file
REF_PATTERN="Qwen3.5-2B-BF16.gguf"
REF_SHARD="Qwen3.5-2B-BF16.gguf"

# Target model variants
# Format: "tag:pattern:first_shard" or "tag:filename" for single-file
TARGETS=(
    "iq2_xxs:Qwen3.5-2B-UD-IQ2_XXS.gguf"
    "iq2_m:Qwen3.5-2B-UD-IQ2_M.gguf"
    "q2_k_xl:Qwen3.5-2B-UD-Q2_K_XL.gguf"
    "iq3_xxs:Qwen3.5-2B-UD-IQ3_XXS.gguf"
    "q3_k_xl:Qwen3.5-2B-UD-Q3_K_XL.gguf"
    "iq4_xs:Qwen3.5-2B-IQ4_XS.gguf"
    "q4_k_xl:Qwen3.5-2B-UD-Q4_K_XL.gguf"
    "q5_k_xl:Qwen3.5-2B-UD-Q5_K_XL.gguf"
    "q6_k_xl:Qwen3.5-2B-UD-Q6_K_XL.gguf"
    "q8_k_xl:Qwen3.5-2B-UD-Q8_K_XL.gguf"
)

# Example for a large split model (Qwen3.5-397B):
# HF_REPO="unsloth/Qwen3.5-397B-A17B-GGUF"
# REF_PATTERN="Qwen3.5-397B-A17B-BF16*"
# REF_SHARD="Qwen3.5-397B-A17B-BF16-00001-of-00009.gguf"
# TARGETS=(
#     "iq1_m:Qwen3.5-397B-A17B-UD-IQ1_M*:Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf"
#     "iq2_xxs:Qwen3.5-397B-A17B-UD-IQ2_XXS*:Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"
# )

# Sampling parameters (vendor-recommended for this model)
TEMP=0.6
TOP_P=0.95
TOP_K=40
SEED=42
N_PREDICT=1024

# Inference settings
NGL=99
CTX=8192

# Prompt directory (relative to this script)
PROMPTS="prompts"

# --- Paths -------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="${SCRIPT_DIR}/build/quant-sampling"
MODELS_DIR="${SCRIPT_DIR}/models"
OUTPUT_DIR="${SCRIPT_DIR}/results/qwen3.5-2b"

# --- Parse flags -------------------------------------------------------------

SKIP_DOWNLOAD=false
SKIP_REF=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-ref)      SKIP_REF=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# --- Preflight checks --------------------------------------------------------

if [ ! -x "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY"
    echo "Build first: cmake -B build && cmake --build build"
    exit 1
fi

mkdir -p "$MODELS_DIR" "$OUTPUT_DIR"

# --- Download helper ---------------------------------------------------------

# Parse "tag:pattern:shard" or "tag:filename" -> sets PARSE_PATTERN and PARSE_SHARD
parse_entry() {
    local entry="$1"
    local rest="${entry#*:}"  # everything after first colon

    if [[ "$rest" == *:* ]]; then
        # tag:pattern:shard
        PARSE_PATTERN="${rest%%:*}"
        PARSE_SHARD="${rest#*:}"
    else
        # tag:filename (single file)
        PARSE_PATTERN="$rest"
        PARSE_SHARD="$rest"
    fi
}

download_model() {
    local pattern="$1"
    local shard="$2"
    local dest="${MODELS_DIR}/${shard}"

    if [ -f "$dest" ]; then
        echo "  [skip] $pattern (already exists)"
        return
    fi

    echo "  [download] $pattern"
    huggingface-cli download "$HF_REPO" --include "$pattern" \
        --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
}

# --- Download models ---------------------------------------------------------

if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "=== Downloading models from $HF_REPO ==="
    download_model "$REF_PATTERN" "$REF_SHARD"
    for entry in "${TARGETS[@]}"; do
        parse_entry "$entry"
        download_model "$PARSE_PATTERN" "$PARSE_SHARD"
    done
    echo ""
fi

# --- Run reference model -----------------------------------------------------

REF_BIN="${OUTPUT_DIR}/ref.bin"

if [ "$SKIP_REF" = false ]; then
    echo "=== Running reference model ==="
    "$BINARY" ref \
        -m "${MODELS_DIR}/${REF_SHARD}" \
        -P "${SCRIPT_DIR}/${PROMPTS}" \
        -o "$REF_BIN" \
        -n "$N_PREDICT" \
        --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --seed "$SEED" \
        -ngl "$NGL" -c "$CTX"
    echo ""
else
    if [ ! -f "$REF_BIN" ]; then
        echo "ERROR: --skip-ref but $REF_BIN does not exist"
        exit 1
    fi
    echo "=== Skipping ref (using existing $REF_BIN) ==="
fi

# --- Run target models -------------------------------------------------------

echo "=== Running target models ==="
for entry in "${TARGETS[@]}"; do
    tag="${entry%%:*}"
    parse_entry "$entry"
    target_bin="${OUTPUT_DIR}/target-${tag}.bin"

    if [ -f "$target_bin" ]; then
        echo "  [skip] $tag (${target_bin} already exists)"
        continue
    fi

    echo "  [run] $tag"
    "$BINARY" target \
        -m "${MODELS_DIR}/${PARSE_SHARD}" \
        -i "$REF_BIN" \
        -o "$target_bin" \
        -ngl "$NGL"
done
echo ""

# --- Compare -----------------------------------------------------------------

echo "=== Running comparisons ==="
for entry in "${TARGETS[@]}"; do
    tag="${entry%%:*}"
    target_bin="${OUTPUT_DIR}/target-${tag}.bin"
    csv_file="${OUTPUT_DIR}/${tag}.csv"
    rank_file="${OUTPUT_DIR}/${tag}-rank.csv"

    if [ ! -f "$target_bin" ]; then
        echo "  [skip] $tag (no target bin)"
        continue
    fi

    echo "  [compare] $tag"
    "$BINARY" compare \
        -a "$REF_BIN" \
        -b "$target_bin" \
        --optimize \
        --csv "$csv_file" \
        --rank-csv "$rank_file" \
        2>&1 | tee "${OUTPUT_DIR}/${tag}-compare.log"
    echo ""
done

# --- Copy viewer -------------------------------------------------------------

cp "${SCRIPT_DIR}/results/view.html" "$OUTPUT_DIR/" 2>/dev/null || true

echo "=== Done ==="
echo "Results in: $OUTPUT_DIR"
echo "View: cd $OUTPUT_DIR && python3 -m http.server 8080 --bind 0.0.0.0"
