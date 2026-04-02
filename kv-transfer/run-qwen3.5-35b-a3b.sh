#!/bin/bash
# Qwen3.5-35B-A3B (MoE): Q8_K_XL reference vs lower quants

REF_MODEL="$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
REF_TAG="qwen3.5-35b-a3b-ud-q8_k_xl"

TARGETS=(
    "ud-q2_k_xl:$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q2_K_XL.gguf"
    "ud-q3_k_xl:$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q3_K_XL.gguf"
    "ud-q4_k_xl:$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
    "ud-q5_k_xl:$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
    "ud-q6_k_xl:$HOME/projects/llms/qwen-3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q6_K_XL.gguf"
)

N_PREDICT=512
TEMP=0.6

source "$(dirname "$0")/run.sh"
