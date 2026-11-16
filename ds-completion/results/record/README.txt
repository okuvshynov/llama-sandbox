# Records written by ds-completion --record (docs/record.md), and the files that quote them.
#
# example.json: one record exactly as the program wrote it, with every field of README step 119.
#   bin-hip/ds-completion -m $M -p "<the cpp-matrix prompt of results/spec-heldout/prompts.txt>" --chat -n 300 --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait --record results/record/example.json
#   the fastest single-machine settings; commit 13ea50c+changes (step 119's change), sclk 6, 2026-11-15
#   Output: the same bytes as results/cache-test/plain-canon-static/cpp-matrix.out.
#   It is the third run of experts.txt. A JSON file holds no comment, so its header is here.
#
# The other files, each with its own header:
#   first.txt       step 115: the first two records (a cold and a warm run)
#   fast.txt        step 115: the fastest settings, once
#   sclk.txt        step 116: the GPU clock setting
#   read-bytes.txt  step 117: bytes read from storage per phase
#   spec.txt        step 118: the speculative loop's totals
#   experts.txt     step 119: the routed experts' uses, placement and cache totals
#   knobs.txt       step 120: the tuning knobs among the arguments (example.json is from before them)
#   timing-options.txt  step 121: the options and DS_* switches that perturb timing, and what two of them cost
