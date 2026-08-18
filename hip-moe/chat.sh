#!/bin/bash
# Interactive chat with DeepSeek-V4-Flash on the four Vega II dies, with
# DSpark speculative decoding — the validated 15 t/s configuration
# (hip-moe/README.md, 2026-08-17).
#
#   ./chat.sh                 # interactive llama-cli chat
#   ./chat.sh server          # llama-server instead: web UI + OpenAI API
#                             #   http://127.0.0.1:8090
#
# Placement notes, hard-won:
#   -ncmoe 13 -ts 19/8/8/8   capacity frontier: 30 of 43 expert layers in
#                            HBM, die 0 absorbs the 13 expert-stripped head
#                            layers; ncmoe 12 OOMs under every split.
#   --spec-draft-cpu-moe     the 11 GB drafter's MXFP4 experts stay in host
#                            RAM (~1.5 GB dense on the dies); without this it
#                            tries one 10.2 GiB allocation on die 0 and dies.
#   --spec-draft-n-max 3     measured optimum (1.53x mean; 4+ degrades and
#                            >5 crashes the server — upstream bug).
#   --fit off                everything is pinned; the fitter cannot measure
#                            the drafter sidecar and would misplace it.

MODEL=${MODEL:-$HOME/llms/DS-V4-Flash-0731-UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf}
DRAFT=${DRAFT:-$HOME/llms/dspark/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf}
BIN_DIR=${BIN_DIR:-$HOME/projects/llama.cpp/build-hip/bin}
CTX=${CTX:-8192}

ARGS=(
    -m "$MODEL"
    -ngl 99 -ncmoe 13 -ts 19/8/8/8
    -t 16 -c "$CTX" --fit off
    -md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max 3
    -ngld 99 --spec-draft-cpu-moe
    --jinja
)

if [ "$1" = "server" ]; then
    exec "$BIN_DIR/llama-server" "${ARGS[@]}" --host 127.0.0.1 --port 8090
else
    exec "$BIN_DIR/llama-cli" "${ARGS[@]}" -cnv
fi
