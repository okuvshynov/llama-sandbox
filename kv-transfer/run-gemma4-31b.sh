#!/bin/bash
# Gemma4-31B: BF16 reference vs quantized targets

REF_MODEL="$HOME/projects/llms/gemma4/BF16/gemma-4-31B-it-BF16-00001-of-00002.gguf"
REF_TAG="gemma4-31b-bf16"

TARGETS=(
    "ud-q2_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q2_K_XL.gguf"
    "ud-q3_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q3_K_XL.gguf"
    "ud-q4_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q4_K_XL.gguf"
    "ud-q5_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q5_K_XL.gguf"
    "ud-q6_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q6_K_XL.gguf"
    "ud-q8_k_xl:$HOME/projects/llms/gemma4/gemma-4-31B-it-UD-Q8_K_XL.gguf"
)

N_PREDICT=2048
TEMP=1.0
TOP_K=64
TOP_P=0.95

source "$(dirname "$0")/run.sh"
