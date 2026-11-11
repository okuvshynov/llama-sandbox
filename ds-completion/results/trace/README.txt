# Chrome traces of generation (README step 51): open the .json.gz in ui.perfetto.dev ("Open trace file")
# bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait -p <prompt> --chat -n 300 --trace <name>.json
# prompts: tcp-udp, cpp-matrix of results/spec-heldout/prompts.txt; commit e23d8fe (README step 98); sclk 6 (manual, all four cards); CPU defaults; 2026-11-11; 33.29 and 47.60 t/s traced (36.85 and 52.41 untraced in step 98), outputs identical
# compare-step58.txt, compare-step69.txt, compare-step73.txt, compare-step74.txt, compare-step84.txt, compare-step86.txt, compare-step88.txt: span totals per verify step against the earlier traces (scripts/trace_compare.py)
# critpath.txt: per verify size, die 0's front, the handoffs to and from the CPU, the CPU's experts, the tail, the time outside
# the layers (split since step 89: before layer 0, the head, the ids to host, the cache step, bookkeeping), the draft (scripts/trace_critpath.py)
# since step 73 the engram rows are fetched on the host track ("engram fetch") while die 0 runs layer 0; no "engram.fetch" group on die 0
# tracks: steps (host): drafts and verify steps; host thread: per layer the wait for the staged block and the CPU's experts
# (args: distinct experts, pairs, tokens) with its phases nested (cpu.inputs, cpu.gate_up, cpu.act, cpu.down; the weight phases
# with the spread between the first and the last thread, step 84), then issuing the copy of its rows and the profile's bookkeeping; GPU 0 main stream: a "layer N" span per layer with the --profile groups (since step 84 ffn.cpu_rows_copy, the CPU's rows coming in)
# nested under it (engram and head outside); GPU 0 experts (second
# stream) and GPU 1-3 experts: each die's expert work per layer (copies included; in generation, args:
# distinct experts, pairs, tokens). GPU times are placed on the host
# clock where host and GPU meet (die 0: the host's per-layer wait for the staged lists; dies 1-3: a reference event
# waited for at each step's start): aligned to ~10-30 us in generation. The prompt's chunks are in it too (issue-time
# anchor there: die 0 can come out a few tenths of a ms early).
# since step 86: steps (host) also commit and its parts, 'draft: issue', and the loop's spans after verify and after commit;
# GPU 0 main stream also the draft ('draft (GPU)', 'draft layer N', draft.* parts); 'GPU d cache loads (host view)': each background load
# since step 87: steps (host) also the verify step's tail ('tail: the head's wait and the ids to host', 'tail: cache step',
# 'tail: profile and trace bookkeeping'); since step 88 the head's group on die 0 includes the argmax
# since step 98 the compressor's two projections run once per chunk: a first 'attn.compressor' group before the per-token ones
