#!/bin/bash
# Qwen3.5-397B: Q8_K_XL reference vs all lower quants

REF_MODEL="$HOME/projects/llms/qwen3.5-397b/UD-Q8_K_XL/Qwen3.5-397B-A17B-UD-Q8_K_XL-00001-of-00010.gguf"
REF_TAG="qwen3.5-397b-ud-q8_k_xl"

TARGETS=(
    "ud-iq1_m:$HOME/projects/llms/qwen3.5-397b/UD-IQ1_M/Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf"
)

N_PREDICT=512
TEMP=0.6

source "$(dirname "$0")/run.sh"
