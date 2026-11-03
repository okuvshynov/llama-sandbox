#!/usr/bin/env bash
# The held-out history (README step 72): rebuild old commits and run the 8 held-out prompts
# (results/spec-heldout/prompts.txt, 300 tokens, --chat) with each point's flags from
# results/heldout-history/points.tsv, at the current sclk level (the history is at sclk 6).
# Per point: results/heldout-history/runs/<step>-<commit>/{summary.txt,<name>.out,<name>.err}.
# scripts/heldout_history.py then collects every point into history.tsv and history.html.
#
#   scripts/heldout-rerun.sh --build [STATUS...]   build the points' worktrees (default: rerun)
#   scripts/heldout-rerun.sh [STATUS|STEP...]      build if needed, then run (default: rerun)
#
# STATUS is a status column value (rerun, optional, measured); STEP a step label (e.g. 21a).
# Worktrees: $WT (default ~/.cache/ds-heldout-history/<commit>), detached at the commit.
# A point whose runs/ directory has a complete summary (8 prompts) is skipped; delete it to rerun.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
WT=${WT:-$HOME/.cache/ds-heldout-history}
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
PL=placements/study30.txt
POINTS=results/heldout-history/points.tsv
PROMPTS=$ROOT/results/spec-heldout/prompts.txt
BUILD_ONLY=0
[ "${1:-}" = --build ] && { BUILD_ONLY=1; shift; }
SEL=("$@")
[ ${#SEL[@]} -eq 0 ] && SEL=(rerun)
LEVEL=$(grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk | awk '{print $1}' | tr -d ':')

selected() {  # step status
  for s in "${SEL[@]}"; do [ "$s" = "$1" ] || [ "$s" = "$2" ] && return 0; done
  return 1
}

build() {  # commit -> the worktree's binary, built once
  local c=$1 dir=$WT/$1
  if [ ! -x "$dir/ds-completion/bin-hip/ds-completion" ]; then
    [ -d "$dir" ] || git -C "$ROOT" worktree add --detach "$dir" "$c" > /dev/null 2>&1 || { echo "$c: worktree failed"; return 1; }
    echo "$c: building ($(date +%T))"
    make -C "$dir/ds-completion" HIP=1 -j16 bin-hip/ds-completion > "$dir/build.log" 2>&1 || { echo "$c: BUILD FAILED, see $dir/build.log"; return 1; }
  fi
  echo "$c: built"
}

mkdir -p "$WT" results/heldout-history/runs
if [ $BUILD_ONLY = 0 ]; then
  [ "$LEVEL" = 6 ] || echo "warning: sclk level $LEVEL, the history is at 6"
  echo "# warming the page cache ($(date +%T))"
  cat ~/llms/UD-Q8_K_XL/*.gguf "$F8" "$D" > /dev/null
fi
grep -v '^#' "$POINTS" | while IFS=$'\t' read -r step commit status flags source summary; do
  [ -z "$step" ] && continue
  selected "$step" "$status" || continue
  case $commit in *'*') echo "$step: $commit is not a commit (measured before it), skipped"; continue ;; esac
  build "$commit" || continue
  [ $BUILD_ONLY = 1 ] && continue
  out=$ROOT/results/heldout-history/runs/$step-$commit
  if [ -f "$out/summary.txt" ] && [ "$(grep -c ' t/s' "$out/summary.txt")" -ge 8 ]; then echo "$step: done before, skipped"; continue; fi
  mkdir -p "$out"
  f=${flags//\$F8/$F8}; f=${f//\$D/$D}; f=${f//\$PL/$PL}
  {
    echo "# scripts/heldout-rerun.sh: step $step, commit $commit (worktree build), sclk $LEVEL, $(date -Is)"
    echo "# ds-completion -m \$M -p <prompt> --chat -n 300 $flags"
    echo "# name: generation t/s, tokens per step, drafts accepted per step, output vs results/cache-test/plain-canon-static"
  } > "$out/summary.txt"
  echo "$step ($commit): running ($(date +%T))"
  while IFS='|' read -r name m nt P; do
    [ -z "$name" ] && continue
    # run from the worktree: the placement file and any relative paths are that commit's own
    (cd "$WT/$commit/ds-completion" && ./bin-hip/ds-completion -m "$M" -p "$P" --$m -n "$nt" $f \
      > "$out/$name.out" 2> "$out/$name.err" < /dev/null)
    tps=$(grep -o 'generation: .* = [0-9.]* t/s' "$out/$name.err" | sed 's/.*= \([0-9.]*\) t\/s/\1/')
    tps_step=$(grep -o '[0-9.]* tokens per step' "$out/$name.err" | awk '{print $1}')
    acc=$(grep -o '[0-9.]* drafts accepted per step' "$out/$name.err" | awk '{print $1}')
    same=DIFFERS
    cmp -s "$out/$name.out" "results/cache-test/plain-canon-static/$name.out" && same=same
    echo "$name: ${tps:-?} t/s, ${tps_step:--} tokens per step, ${acc:--} accepted, output $same" >> "$out/summary.txt"
  done < "$PROMPTS"
  tail -8 "$out/summary.txt" | awk -v s="$step" '{x+=$2} END {printf "%s: mean %.2f t/s\n", s, x/NR}'
done
echo "ALL DONE ($(date +%T))"
