#!/bin/bash
# Qwen3.5-397B: Q8_K_XL reference vs all lower quants

REF_MODEL="$HOME/projects/llms/qwen3.5-397b/UD-Q8_K_XL/Qwen3.5-397B-A17B-UD-Q8_K_XL-00001-of-00010.gguf"
REF_TAG="qwen3.5-397b-ud-q8_k_xl"

TARGETS=(
    "ud-iq2_m:$HOME/projects/llms/qwen3.5-397b/UD-IQ2_M/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
    "ud-iq3_xss:$HOME/projects/llms/qwen3.5-397b/UD-IQ3_XXS/Qwen3.5-397B-A17B-UD-IQ3_XXS-00001-of-00004.gguf"
    "ud-iq4_xs:$HOME/projects/llms/qwen3.5-397b/UD-IQ4_XS/Qwen3.5-397B-A17B-UD-IQ4_XS-00001-of-00005.gguf"
)

N_PREDICT=2048
TEMP=0.6

source "$(dirname "$0")/run.sh"
