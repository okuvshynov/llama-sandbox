#!/usr/bin/env bash
# Progress of the held-out history rerun (scripts/heldout-rerun.sh, README step 72).
#   scripts/heldout-status.sh [LOG]          once
#   watch -n 60 scripts/heldout-status.sh    every minute
# LOG: the rerun's log (default ~/heldout-rerun.log). STALLED: no output file written for
# STALL_MIN minutes (default 15; one prompt takes 1-3 minutes) while the run is still alive.
cd "$(dirname "$0")/.."
LOG=${1:-$HOME/heldout-rerun.log}
STALL_MIN=${STALL_MIN:-15}
RUNS=results/heldout-history/runs
POINTS=results/heldout-history/points.tsv

total=$(grep -v '^#' "$POINTS" | awk -F'\t' '$1 != "" && $2 !~ /\*$/' | wc -l)
done_n=0
for d in "$RUNS"/*/; do
  [ -f "$d/summary.txt" ] && [ "$(grep -c '^[^#].* t/s' "$d/summary.txt")" -ge 8 ] && done_n=$((done_n + 1))
done
echo "points complete: $done_n of $total"

pid=$(pgrep -x heldout-rerun.s 2>/dev/null | head -1)
[ -z "$pid" ] && pid=$(ps -eo pid,args | awk '$0 ~ /scripts\/heldout-rerun\.sh/ && $0 !~ /awk/ {print $1; exit}')
if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
  echo "status: finished ($(grep 'ALL DONE' "$LOG"))"
elif [ -n "$pid" ]; then
  echo "status: running (pid $pid)"
else
  echo "status: NOT RUNNING and not finished; see $LOG"
fi

cur=$(grep -o '^[^ ]* ([0-9a-f]*): running ([0-9:]*)' "$LOG" 2>/dev/null | tail -1)
[ -n "$cur" ] && echo "current point: $cur"
dir=$(ls -td "$RUNS"/*/ 2>/dev/null | head -1)
if [ -n "$dir" ]; then
  echo "  prompts done there: $(grep -c '^[^#].* t/s' "$dir/summary.txt" 2>/dev/null || echo 0) of 8"
  newest=$(ls -t "$dir" | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$dir/$newest") ))
  echo "  last write: $newest, $((age / 60)) min $((age % 60)) s ago"
  if [ -n "$pid" ] && [ $age -gt $((STALL_MIN * 60)) ]; then echo "  STALLED? nothing written for over $STALL_MIN min"; fi
fi
echo "finished points (mean t/s):"
grep 'mean' "$LOG" 2>/dev/null | sed 's/^/  /'
grep -iE 'failed|error' "$LOG" 2>/dev/null | sed 's/^/  PROBLEM: /'
