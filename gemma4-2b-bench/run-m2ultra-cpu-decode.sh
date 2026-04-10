#!/bin/bash
# CPU-only prefill benchmark on Mac Studio M2 Ultra

LLAMA_BENCH="$HOME/projects/forks/llama.cpp/build/bin/llama-bench"
MODEL_DIR="$HOME/projects/llms/gemma4-2b"

$LLAMA_BENCH -m \
    $MODEL_DIR/gemma-4-E2B-it-UD-Q2_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-UD-Q3_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-UD-Q4_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-UD-Q5_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-UD-Q6_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-UD-Q8_K_XL.gguf,$MODEL_DIR/gemma-4-E2B-it-BF16.gguf \
    -n 16,256,1024 -p 0 -ngl 0 -o csv
