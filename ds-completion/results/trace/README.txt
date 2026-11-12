# Chrome traces of generation (README step 51): open the .json.gz in ui.perfetto.dev ("Open trace file")
# bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --dspark $D --spec --canon --cache-loads 800 --cache-period 8 --cache-wait -p <prompt> --chat -n 300 --trace <name>.json
# prompts: tcp-udp, cpp-matrix of results/spec-heldout/prompts.txt; commit 195bef1 (README step 104; traced in step 105); sclk 6 (manual, all four cards); CPU defaults; 2026-11-11; 34.88 and 49.26 t/s traced (38.89 and 54.03 untraced in step 104), outputs identical
# compare-step58.txt, compare-step69.txt, compare-step73.txt, compare-step74.txt, compare-step84.txt, compare-step86.txt, compare-step88.txt, compare-step98.txt: span totals per verify step against the earlier traces (scripts/trace_compare.py)
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
# since step 104 the fp8 matrices at 3-8 tokens run on the several-input kernel (attn.out, ffn.shared, engram, attn.q); DS_FP8_VIRT=0 gives the 64-lane kernel (results/decode-next/virt-onoff.txt)
# two-machines/ (README step 112): one trace of the model in two halves on two machines, the back's events on the front's clock
#   TRACE=1 PROMPTS="cpp-matrix tcp-udp" scripts/split-measure.sh (front tomb, back shower; commit bf44341 + step 113; sclk 6 on both; 2026-11-12; step 112's were replaced: the back had prompt chunks traced as generation);
#   60.33 and 38.11 t/s traced (62.89 and 42.10 untraced in step 111), outputs identical; <name>.clock.txt: the clock fit (14 samples, ~1 us residuals)
#   process 1 = the front (layers 0-19): the tracks above, with 'send to the back: ...' and 'waiting for the back' on steps (host);
#   process 2 = the back (layers 20-39, the head, the draft): the same tracks, 'receive the stream: ...' on steps (host), and
#   'messages from the front': one 'serve <MESSAGE>' span per message from its header's arrival to the end of its work
