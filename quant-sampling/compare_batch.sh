#!/usr/bin/env bash
# compare_batch.sh — compute KL divergence and optimize temperature for a
# quantized model variant previously collected by collect_batch.sh.
#
# Usage: ./compare_batch.sh <tag> [outdir] [extra compare-batch args...]
#   tag     the tag used in collect_batch.sh (e.g. "q2", "q4", "q5")
#   outdir  directory containing the manifest (default: batch_run)
#   extra args are forwarded to quant-sampling compare-batch
#           e.g. --optimize --temp-min 1.0 --temp-max 1.5 --temp-step 0.01
set -euo pipefail

QMATCH=./build/quant-sampling

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <tag> [outdir] [extra compare-batch args...]" >&2
    exit 1
fi

TAG="$1"
shift

# Second arg is outdir if it doesn't start with '-', otherwise it's an extra arg
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    OUTDIR="$1"
    shift
else
    OUTDIR="batch_run"
fi

MANIFEST="$OUTDIR/manifest_${TAG}.txt"

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: manifest not found: $MANIFEST" >&2
    echo "       run collect_batch.sh first with tag '$TAG'" >&2
    exit 1
fi

echo "=== Compare [$TAG]: $MANIFEST ==="
echo ""

$QMATCH compare-batch --manifest "$MANIFEST" --optimize "$@"
