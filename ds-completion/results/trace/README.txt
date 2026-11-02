# Chrome traces of generation (README step 51): open the .json.gz in ui.perfetto.dev ("Open trace file")
# bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait -p <prompt> --chat -n 300 --trace <name>.json
# prompts: tcp-udp, cpp-matrix of results/spec-heldout/prompts.txt; commit cb73732 + layer spans; sclk 6 (manual, all four cards); 2026-11-02; 25.64 and 32.76 t/s, outputs identical
# tracks: steps (host): drafts and verify steps; host thread: per layer the wait for the staged block and the CPU's experts
# (args: distinct experts, pairs, tokens); GPU 0 main stream: a "layer N" span per layer with the --profile groups
# nested under it (engram and head outside); GPU 0 experts (second
# stream) and GPU 1-3 experts: each die's expert work per layer (copies included). GPU times are placed on the host
# clock by a reference event per device at each step's start (aligned to ~0.1 ms). The prompt's chunks are in it too.
