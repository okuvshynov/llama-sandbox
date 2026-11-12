#!/usr/bin/env bash
# An interactive chat session with the runtime's fastest configuration (README step 108):
# one line per turn, an empty line or Ctrl-D ends it. Notes (load, per-turn speed) go to stderr:
# 2>/dev/null hides them and the "> " prompt.
#
#   scripts/chat.sh [--think [--effort max]] [-n 30000] [more ds-completion options]
# --think: with reasoning (printed; earlier turns' reasoning is dropped from the context, README step 109)
set -u
cd "$(dirname "$0")/.."
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
exec ./bin-hip/ds-completion -m "$M" --rt "$F8" --placement placements/study30.txt --dspark "$D" --spec --canon \
  --cache-loads 800 --cache-period 8 --chat -i -n 30000 "$@"
