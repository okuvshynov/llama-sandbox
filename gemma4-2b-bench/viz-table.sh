#!/bin/bash
# Show 2D table: rows = quant levels, columns = prompt sizes, cells = tps.
# Usage: ./viz-table.sh <csv_file>
# Example: ./viz-table.sh prefill-m2ultra-metal.csv

set -euo pipefail
FILE="${1:?Usage: $0 <csv_file>}"

xan select model_type,n_prompt,avg_ts "$FILE" | \
    xan map 'round(avg_ts) as tps, concat("pp", n_prompt) as pp, replace(model_type, "gemma4 E2B ", "") as quant' | \
    xan select quant,pp,tps | \
    xan pivot pp 'first(tps)' -g quant | \
    xan view --cols 200
