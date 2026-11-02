# Chrome traces of generation (README step 51): open the .json.gz in ui.perfetto.dev ("Open trace file")
# bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait -p <prompt> --chat -n 300 --trace <name>.json
# prompts: tcp-udp, cpp-matrix of results/spec-heldout/prompts.txt; commit d1a2de8 (README step 58); sclk 6 (manual, all four cards); CPU defaults; 2026-11-02; 26.79 and 34.46 t/s traced (28.86 and 36.90 untraced in step 57), outputs identical
# tracks: steps (host): drafts and verify steps; host thread: per layer the wait for the staged block and the CPU's experts
# (args: distinct experts, pairs, tokens); GPU 0 main stream: a "layer N" span per layer with the --profile groups
# nested under it (engram and head outside); GPU 0 experts (second
# stream) and GPU 1-3 experts: each die's expert work per layer (copies included; in generation, args:
# distinct experts, pairs, tokens). GPU times are placed on the host
# clock where host and GPU meet (die 0: the host's per-layer wait for the staged lists; dies 1-3: a reference event
# waited for at each step's start): aligned to ~10-30 us in generation. The prompt's chunks are in it too (issue-time
# anchor there: die 0 can come out a few tenths of a ms early).
