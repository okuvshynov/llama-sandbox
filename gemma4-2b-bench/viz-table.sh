#!/bin/bash
# Show 2D table: rows = quant levels, columns = test sizes, cells = tps.
# With optional baseline comparison showing relative difference.
#
# Usage: ./viz-table.sh <csv_file> [baseline_quant] [size_column]
# Example: ./viz-table.sh prefill-m2ultra-metal.csv              # prefill, pp sizes
#          ./viz-table.sh prefill-m2ultra-metal.csv Q8_0          # with baseline
#          ./viz-table.sh tg-m2ultra-metal.csv "" n_gen           # token gen, tg sizes
#          ./viz-table.sh tg-m2ultra-metal.csv Q8_0 n_gen         # tg with baseline

set -euo pipefail
FILE="${1:?Usage: $0 <csv_file> [baseline_quant] [size_column]}"
BASELINE="${2:-}"
SIZE_COL="${3:-n_prompt}"

# auto-detect prefix for column labels
if [ "$SIZE_COL" = "n_gen" ]; then
    PREFIX="tg"
else
    PREFIX="pp"
fi

pivot_csv() {
    xan select "model_type,$SIZE_COL,avg_ts" "$FILE" | \
        xan map "round(avg_ts) as tps, concat(\"$PREFIX\", $SIZE_COL) as size, replace(model_type, \"gemma4 E2B \", \"\") as quant" | \
        xan select quant,size,tps | \
        xan pivot size 'first(tps)' -g quant
}

if [ -z "$BASELINE" ]; then
    pivot_csv | xan view --cols 200
else
    # add relative diff from baseline row
    pivot_csv | awk -F, -v base="$BASELINE" '
    NR == 1 { print; ncols = NF; next }
    {
        q = $1; gsub(/"/, "", q)
        if (q ~ base) {
            base_row = NR
            for (i = 2; i <= NF; i++) base_val[i] = $i + 0
        }
        rows[NR] = $0
        for (i = 2; i <= NF; i++) vals[NR, i] = $i + 0
        order[NR] = NR
        max_row = NR
    }
    END {
        for (r = 2; r <= max_row; r++) {
            split(rows[r], f, ",")
            printf "%s", f[1]
            for (i = 2; i <= ncols; i++) {
                v = vals[r, i]
                if (base_val[i] > 0 && r != base_row) {
                    pct = (v - base_val[i]) / base_val[i] * 100
                    sign = pct >= 0 ? "+" : ""
                    printf ",%d (%s%d%%)", v, sign, pct
                } else {
                    printf ",%d", v
                }
            }
            printf "\n"
        }
    }' | xan view --cols 200
fi
