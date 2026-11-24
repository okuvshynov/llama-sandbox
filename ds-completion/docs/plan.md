# ds-completion plan

Status: **done**, **next** (agreed, in order), **planned** (agreed, later), **idea** (not
decided). Today (sclk 6, 16 CPU threads): `--spec` with the adaptive cache (`--canon
--cache-loads 800 --cache-period 8`) 44.57 t/s mean on the 8 held-out prompts (33.9-55.0; README
step 104; every decode change on the same ruler: `results/heldout-history/`). Prompt processing
93 t/s (63 `--canon`; 1,989 tokens, README step 68). Where a token's time goes today: docs/timing.md.

This file is the list of items with their status, the dated plans ("The testing process", "The product and the tools", "Observability of the
server"), the list of ideas still open ("Further optimization ideas") and "Two machines", kept
as history ("today" inside a dated section means its date).
The plans that have been carried out were removed in step 156: the cost model of 2026-10-27,
the plans for batched verification, expert placement, the adaptive cache, long context and
prompt processing, and the first brainstorm with its sections 1-6. Steps and other documents
cite them, and this file, as PLAN.md ("PLAN.md B0", "PLAN.md 'Adaptive expert cache'": it stood in the
project's directory until step 156); the letters A-G of that
brainstorm are in the items' names below, and the text is
`git show 64f5e05:ds-completion/PLAN.md`.

## Overview

Correctness and the base path

| | item | status | one line |
|---|---|---|---|
| 1 | CPU reference forward (ggml Engine) | done | every module of the reference, gated per layer against the official implementation (steps 1-3) |
| 2 | Exact fake quantization | done | fp8/fp4 rounding at the reference's points, so outputs match to ~1e-7 per layer |
| 3 | ggml GPU path | done, removed | dense part on one die through ggml (6 t/s); superseded by the own runtime (steps 4, 11) |
| 4 | fp8 repack of dense weights (B0) | done | lossless 6.4 GB fp8 file instead of 14 GB bf16 |
| 5 | Exact fp8 GEMV (B1) | done | 311 us per layer vs ggml's 832, bit-exact products |
| 6 | Own HIP decode runtime (D3) | done | every op of a token in our kernels on one die: 9.4 -> ~10 t/s |
| 7 | Exact CPU expert kernel (A3) | done | AVX-512 fp4 x fp8, exact; ~85 GB/s at 16 threads in its benchmark, back to back (0.22 ms per expert in a 6-expert call); in the runtime, after its wait for the GPU, a cold call ~0.24-0.32 ms per expert (README steps 77-82) |
| 8 | Exact GPU expert kernel | done | 4 launches, 4-byte weight decode: 1/2/3/6 experts in 72/110/151/281 us (406 GB/s for 6) (README step 17) |
| 9 | Whole expert layers on GPUs | done, replaced | `--gpu-moe`: 15 layers resident on 4 dies -> 12.2 t/s; superseded by item 15 |
| 10 | Prompt through the runtime | done | frees 14 GB on die 0; prompt 6 -> 10 t/s |
| 11 | Remove ggml's GPU path | done | the runtime is self-contained; no fork HIP build needed |
| 12 | GPU clock sweep | done | dense GPU part 38 -> 29 ms from sclk 0 to 8; level 4-6 is the sweet spot |
| 12a | 16 CPU threads, not 32 | done | 32 starved the host thread and the HIP runtime: every configuration +25-50% (README step 18) |

Speed

| | item | status | one line |
|---|---|---|---|
| 13 | Routing statistics (A0) | done | 30 prompts: fixed hot sets transfer poorly (71% at 128 GB), an adaptive LRU cache reaches 95% with ~11 loads per token (README step 13) |
| 14 | HIP graphs + kernel fusion (D) | started | engram per step + experts mapped at load done (~1%); graphs save little on this ROCm, fusion ~37 -> ~22 launches per layer for ~2-3 ms (README step 12) |
| 15 | Per-layer expert split, CPU and all dies concurrently (A1/EP) | done | `--placement`: fixed set from the study; 17.7-21.7 t/s (README steps 14-18) |
| 15a | Expert path fixes: no die waits for the host; faster small-count kernel; 16 threads | done | oracle bound 16 -> ~25 t/s (README steps 15-18) |
| 16 | Adaptive expert cache (A1 + A2) | done; rate and period tuned (step 50) | `--canon --cache-loads R`: frequency rebalance, held-out `--spec` +28% at sclk 2 (25.6 t/s), 29.1 at sclk 8, outputs identical (README steps 28-33; with steps 50-82 37.16 at sclk 6); tuning: "Further optimization ideas" A |
| 17 | Tensor parallelism, 2-way then 4-way (C1/C2) | planned; step 76 tried wo_a/wo_b over 0-1 (+0.6%, not kept) | split the dense GEMVs over dies: ~-6..10 ms per step, matters once experts are mostly on GPUs |
| 18 | Shared expert by TP, routed by EP (C4) | planned | the dense shared expert splits like any other GEMV under TP |
| 19 | DSpark speculative decoding (E1) | done | `--dspark ... --spec`, output identical to plain greedy; held-out prompts +7% to +33% (geometric mean +22%), oracle placement +19% / +71% (README steps 19-26) |
| 20 | Batched prompt processing (F) | done (steps 37-46, 68: 93 / 63 t/s fast / `--canon`, 1,989 tokens) | README steps 37-38: chunks of up to 512, CPU experts batched, CPU/GPU overlap; byte-identical output: 16.8 -> 51.6 t/s (fast), 17.4 -> 40.0 (canon), 1,989 tokens. CPU and GPU now balanced (~33 s each): step 39 (fp8_gemm, bit-identical): 57.1 t/s fast; now the CPU experts are the limit (30 of 36 s). Options and the expert-streaming estimate: "Prompt processing: where it stands and the options" (a section removed in step 156: `git show 64f5e05:ds-completion/PLAN.md`) (P1-P9) |
| 30 | Expert registration stalls the GPU | done | README step 43: not HIP calls but the memory-map lock: one registration call holds it, any mmap/munmap meanwhile waits (up to seconds for the ~48 GB ranges). 64 MB pieces (copies split at piece boundaries): no stall; registration (~95-135 s) still delays the cache's first loads |
| 21 | Dynamic expert cache (A2) | merged into 16 | item 13 showed recent usage beats fixed sets |
| 22 | Faster GPU fp4 expert kernel | partly done | 287 -> 406 GB/s (item 8); the per-row fp8 activation decode is the next cost; matters little while it overlaps |
| 22a | Router + dispatch fusion, copy only what dies need | idea | the per-layer expert overhead (dispatch, copies, waits) is ~7 ms per token at high hit rates |
| 23 | Weight interleaving for TP (C3) | idea | lay out weights so each die's slice is contiguous |
| 24 | fp8 output head | idea | halves the 2.3 ms head, but is no longer exact vs the reference |
| 25 | Clock level by phase | idea | sclk 4-6 for decode with CPU experts, 8 once GPU experts dominate |
| 32 | Further optimization ideas (2026-10-29) | ideas | cache tuning, the dense part of a verify step and TP, speculation tuning, a faster draft: see the section below |
| 31 | Two machines, the model in two halves | done (steps 110-112), then removed (step 178: the link; step 179: the layer range) | layers 0-19 / 20-39 on two Mac Pros over TCP: held-out 44.60 -> 50.85 t/s, outputs identical. One machine is the scope: "Two machines" below |

Features and limits

| | item | status | one line |
|---|---|---|---|
| 26 | Context above 4096 with the runtime | done | 65,536 by default, 1M with `--max-ctx`; README steps 34-35 |
| 27 | Candidate-block selection (>16k tokens) | done (runtime) | the reference's layer-20 path, checked against its own function; the CPU path still stops at 16k |
| 28 | Full 1017 checkpoint | not needed | item 19 runs from the GGUF; exact references read the release by HTTP ranges (README "Which checkpoint") |
| 29 | Sampling beyond greedy | started | temperature and top-p for a minimal server (the person's benchmark). Step 148: the sampler as a library with its test. Step 149: `--temp`, `--top-p`, `--seed` in the plain loop, one seed one output on any placement. Step 150: the speculative loop samples too (the plain loop's outputs; held-out mean 41.8 t/s against 44.8 greedy, not with `--remote`). Step 151: that loss was the draws (~1 ms each and the threads' start); a vector `exp` and draws in turn: 43.1 t/s against 44.9 greedy. Step 152: `--requests`, the engine's side of the server (prompts on stdin, each on an empty context; `docs/requests.md`). Step 153: the HTTP front, `server/ds_server.py` (Anthropic Messages; the model's `encoding.py` renders and parses): the benchmark's script runs against it unchanged. Step 154: a request whose client left is cancelled (`SIGUSR1` to the engine). Sampling with `--spec` on two machines, long listed as open, went with the two machines (step 178). Next: the person's samples, the OpenAI endpoint if wanted. Greedy loops on some long reasoning (README step 48) |
| 33 | Observability of the server | started (steps 1-3 of 4 done, steps 164-166) | a metrics page in the Prometheus format, the engine's choice of context in the protocol and the log, a log of requests as JSON lines: "Observability of the server" below (README step 162) |
| 34 | The product and the tools | the three small steps done (steps 169-171), `fakequant` in two (172), the CPU path as its own program `ds-ref` (173). Open: where `ds-test` belongs, directory moves, a tokenizer without libllama | say and then make true which code is the product and which checks or measures it: "The product and the tools" below |
| 35 | The testing process: the same guarantees in less time | done as a split (steps 186-190): `unit`, `default`, `speed`, `complete`; each to be improved | a full pass is ~3.5 h, most of it loads of the model, with overlapping runs and comparisons written by hand per step: "The testing process" below |
| 36 | The load, rethought | planned (2026-11-23; measured in steps 199-200); the person tests by hand first | a start is 12.9 s of load where the bus allows ~5, and then the cache waits 75-285 s for a registration its loads do not strictly need: "The load" below |

## The load (2026-11-23)

Measured in steps 189, 199 and 200 (`results/register/`). Nothing below is built; the person
runs tests by hand before any of it starts.

**Today, with a well-filled page cache:** start and mapping 1.3 s; dense weights, small
weights and the draft to GPU 0 4.7 s (8.7 GB); the placement's 6,097 experts 7.0 s (114.6 GB at
18.5 GB/s); "loaded in" 12.9 s. Then the registration of the 289 GB of expert tensors in the
background: 75 s when the files came into the page cache by a sequential read, 285 s when the
engine's page faults brought them in. The cache starts no load until it ends. Exit 4.5 s.

**What the machine allows:** 27 GB/s to the four dies (two PCIe 3.0 x16 slots, two dies each;
step 30, at sclk 2): 57 GB and 66 GB per card in 4.2 s and 4.9 s. Registration: 1.0-4.2 GB/s
on page-cache pages by their layout, 4.8 on our own 4 KiB pages, 9.7 on 2 MB huge pages.

**Three things a start has to reach, each with its own need:**

| | needs | today |
|---|---|---|
| able to generate | the dense weights on GPU 0; the CPU kernel reads every expert once the files are mapped | after the whole load, 12.9 s |
| full speed | the placement's experts in their slots | the same moment |
| the cache adapts | loads: direct from registered memory, or by a host copy to a pinned buffer (same rate, 2-3 times the CPU per load) | after the registration, 88-298 s |

**The steps, in the order proposed:**

| | step | expected | check |
|---|---|---|---|
| A | The cache's loads from the first token: the copying path for an expert whose memory is not registered yet, the direct path once it is (the loader already knows per piece) | no correctness check waits for the registration: 47 of `complete`'s 63 starts go from 88-298 s to ~13 s | `default`; `speed` before and after with `--cache-wait` kept there; the speed of the first 75 s of a process measured, since the copying path costs the CPU expert kernel (step 30: noticeable at 400 loads/s; a 300-token run starts ~630/s) |
| B | The upload at the bus rate: two threads per die or direct transfers, and the dense weights to die 0 while experts go to the others | "loaded in" 12.9 -> ~6 s | `default`; the load's phases timed again (`results/register/ts.py`); the bus rate measured again at sclk 6 first |
| C1 | The registration short by the page cache: a script that drops the model's files from the cache and reads them back sequentially (3 min, no root), run by `check.py speed` and `complete` at their start; one line from the engine when it registers below ~2 GB/s | 285 -> 75 s per start whatever filled the cache | one start before and after |
| C2 | The same by a copy that outlives the process: the experts once per boot in shared memory on huge pages | ~30 s; needs root once per boot (`shmem_enabled`), a second loader path, and the logits checks on it | only if 75 s is still too long after A |
| D | Generating before the upload is over: the placement as the cache's first target, the CPU serving what has not arrived | the first token at ~3 s instead of ~6 | left unless wanted: ~4 s for the most code touched |

**Open questions the hand tests may answer:** whether the page-cache layout explains every
boot's rate (step 199 has two points and an inferred mechanism); what the copying path costs at
the cache's real rate; whether the exit's 4.5 s matters to anyone.

Apart from this item: the two crashes of step 198 in the ROCm runtime (the exit abort; the
segfault at a loader thread's first allocation), which may be one fault of concurrent
allocation from the loader threads. B changes that code, so the test of it (stages created one
at a time, many starts) belongs before B.

**After step 201** (`docs/registration-and-stalls.md`): the registration is not needed for correctness and holds 330-540 MB of
page tables on every die, die 0 included. Before A or C2: the engine measured with pinned-only loads (chat, code, a long
prompt). If the registration stays: below HIP, for dies 1-3 (~10-31 s instead of ~39-100, extrapolated). The reserve of
0.75 GB per die is mostly those page tables and stays.

## The testing process (2026-11-22)

The current state, test by test, with terms, needs and times: [testing.md](testing.md) (step 184).

**Why.** After the two-machine code left, gating three commits took ~3.5 h of runs (steps
179-181). A run is ~140 s of load for ~8 s of generation; several runs repeat each other; each
step wrote its own comparison code, and one of them raised a false alarm (step 181).

**What overlaps** (to confirm by reading the matrices side by side):

- the eight held-out greedy `--spec` runs: `results/cleanup/heldout.matrix`, and the `greedy`
  configurations of the two sampling matrices;
- per layer and end to end on one golden set: at 20,000 tokens 32 min and 9 min; the logits'
  bytes are the sharper check, the table says where a difference starts;
- the CPU reference tables after a change that touched only `hip/`;
- a session against stored bytes, and again against fresh runs on the whole conversation
  (10 loads for what the stored bytes already hold);
- each sampling property on three prompts.

**Decided (the person, 2026-11-23, after `docs/testing.md`).** The aims: simplicity and speed
of iteration, and no choosing of tests per change, because that is where mistakes happen. So a
few fixed runs, each one command of `scripts/check.py` with one verdict:

| run | when | what | time | state |
|---|---|---|---|---|
| `unit` | after every change | no model files: the rounding ops, the sampler, the selection kernels, the batched kernels, the server against a fake engine | 30 s | done, step 186 |
| `default` | after every change | the model, loads kept to a minimum. Two verdicts: logits against stored bytes on golden token sequences (`ds-test --e2e`: a 12-token set and a long prompt; it starts in seconds); and the product's outputs in one load: one engine with `--requests` and the full configuration (placement, cache, draft, speculation, `--canon`) takes the held-out prompts, a sampled seed, a conversation that goes on, each reply's hash against a manifest in git. One short `-i` session in a fresh process for the two paths `--requests` does not take | ~5.5 min | done, steps 187-188 |
| `speed` | when a change is about speed (small gains were the common case) | the held-out prompts' tokens per second in fresh processes, as the documents quote it: the second repeat's mean against `tests/data/speed/baseline.tsv` (2%: slower fails, within is "the same"); every output the stored bytes | 36 min, ~28 of them the registration of expert tensors for the cache (step 189) | done as it is measured today, step 190 |
| `complete` | overnight, and before anything is called done | `default`, then: the 20,000-token logits, the request test in both modes, six sessions and ten single runs, the three sampling matrices, a long prompt from a file, and `speed` | ~4 h | done, step 190 |
| localizers | only when a verdict failed | the per-layer table, per token, near-ties, traces: not part of any run; a failed check prints the command to run next | | the hint is in `check.py`'s output |

What this needs, in the order of work: the expected logits and the token ids of the golden
sets in git (a few MB; the 73 GB of golden tensors stay outside, for the localizers); a
manifest of expected hashes; the request test's engine as the default run's second part; the
load itself (139 s) looked at, since it is the floor of `default` and most of `speed`.

Known limits of `default`, accepted: no context past 16k tokens (the 20,000-token set is 9
minutes: `complete`); its speeds are of one process with an adapted cache, comparable between
default runs only; stored logits are bytes of one toolchain, so a ROCm or compiler change is
decided by `complete`'s comparison with the reference and then stored again on purpose.

**To improve, now that the runs exist** (the person, 2026-11-23: iterate after the split is in place): the
registration (loads before it is complete; step 189) and with it `speed`'s 36 minutes; `complete`'s
repeated runs (the held-out greedy runs appear in `speed` and in a sampling matrix; a session and
its ten single runs); the draft against the reference as a check with a verdict; the code
findings of step 182's review.

**A rule from the description** (step 185; testing.md 3.6): a tier runs verdicts (bytes, hashes, a test's
exit code); what says where a difference is (the per-layer table, per token, the traces) runs
when its verdict fails. Where a verdict has nothing behind it (a hash without the ids), the
record or the tool should keep enough to find the first difference.

**To check first:** that outputs through `--requests` are the fresh process's bytes on all
held-out prompts (with `--canon` they should be: the output does not depend on the cache's
state); what cannot vary per request (the placement, `--spec-n`: their own loads); why the
load takes ~140 s and why it was ~116 s at step 151.

## The product and the tools (2026-11-21)

**Why.** The project grew from experiments, and the code that serves requests stands among the
second implementation, the tests and the benchmarks. All of them stay; what is missing is a
line between them that a reader, a build and a packager can rely on.

**The rule.** The product is what runs under `scripts/serve.sh`: `ds-completion --rt`, the
server, `fp8pack`. What checks it (the CPU reference path, the tests, the Python reference) and
the tools (benchmarks, measurement scripts) may depend on it; it should not depend on them.
The map, file by file: `docs/overview.md`, "Code map".

**Small steps, each its own commit, none changing what a program computes:**

| step | what | status |
|---|---|---|
| 1 | the map in `README.md` and `docs/overview.md`, with the places where the rule does not hold yet | done (step 169) |
| 2 | Makefile targets: the product alone, the tests, the benchmarks; `all` stays everything | done (step 170): `make product`, `make tests`, `make tools` |
| 3 | `engram_rows` and `Taps` out of `src/forward.h` into a header of their own, so `hip/` includes nothing of the reference path | done (step 171): `src/engram.{h,cpp}`, `src/taps.h` |

Checked after 2 and 3: the programs build, the short gate and a held-out prompt give the same bytes.

**Deferred until those are done** (each is a decision, not a cleanup):

- `src/fakequant.cpp` in two files (the runtime's host functions; the reference path's graph ops): done, step 172.
- `ds-completion` without `--rt` as a build option or a program of its own (done, step 173: the
  program `ds-ref`; the person chose a program): a serving build would
  then take only the tokenizer and the GGUF reader from llama.cpp (it uses only those; ggml is
  still linked, it comes with `libllama.a`). Is the CPU-only mode part of
  the product? **Decided (the person, 2026-11-21): no, it is a reference tool.** Step 173
  took it out of the product's binary, as a program of its own.
- Moving directories (`src/forward.*`, the analysis scripts): it breaks paths the documents and
  the results' headers name; the moment for it is a move to a repository of its own, if that comes.
- Where `ds-test` belongs: it gates both paths.

## Observability of the server (2026-11-19)

Item 33. From a review of `server/ds_server.py` and the `--requests` loop in
`apps/ds_completion.cpp` after step 161 (README step 162); nothing is built. Two aims: numbers
to watch while the server runs for days (a metrics page), and enough in the logs to say why a
prompt did or did not go on from the context.

**What there is.** The server writes one line per request to stderr (mode, prompt tokens, kept,
prompt seconds, t/s, the wait for the engine, the stop reason) and a progress line every 30 s;
`--dump DIR` writes the request, the prompt, the raw reply and the message. The engine's
protocol has `kept` in its `prompt` line and `kept`, `saved` and the two times in its `end`
line; its own lines per request (speculation, routed experts, speeds) go to the same stderr.

**What is missing.**

- The choice of context cannot be seen. The engine computes how many tokens the prompt shares
  with the context and with the saved point, and takes one of three ways (goes on from the
  context, returns to the saved point, starts empty); it prints `kept=N`. When a prompt does
  not go on, nothing says where the tokens part.
- The server does not read `saved`: a point being set or ended is in no log.
- `GET /health` answers ok without looking at the engine. With the engine gone every request
  is a 500 and the health check is still ok.
- `--dump` writes completed requests only: not the cancelled ones, not the failed or refused.
- The time to the first token is not taken (only when the engine accepted the request).
- The log lines have no time of day, and the engine's lines no request number.
- `--record` is refused with `--requests`, so the record's counters (the speculative loop's
  totals, the routed experts, the cache's loads, bytes read from storage, the GPU clock) are
  not available for a request.

**The steps, in order, one commit each.**

1. **Done (step 164).** **The engine's choice of context, reported.** In the `prompt` line: `ctx=` (the context's
   tokens), `common=` (how many of them the prompt starts with), `how=continue|saved|empty`.
   One stderr line per request with the same; when the prompt parts from the context before
   its end, the index of the first token that differs, the id on each side, and the text of a
   few tokens around it on each side. Both tests read the protocol's lines as key=value pairs,
   so new fields do not break them; `tests/requests_test.py` gains checks of the
   new fields on the cases it already has. First because it is the only step that gives what
   cannot be had today: the others put together what is already logged.
2. **Done (step 165)**, but for `ds_context_tokens`, which waits for step 4 (the server cannot tell
   it exactly from what the engine says today); `ds_engine_load_seconds` was added.
   **`GET /metrics`** in the Prometheus text format, written by hand (the server uses only the
   standard library), from what the server already knows. It does not take the engine's lock,
   so it answers while a request runs. `/health` looks at the engine's process. Tested with
   `tests/fake_engine.py`.

   | metric | kind | labels |
   |---|---|---|
   | `ds_requests_total` | counter | `outcome`: ok, cancelled, bad_request, engine_error |
   | `ds_stop_reason_total` | counter | `reason`: end_turn, tool_use, max_tokens |
   | `ds_prompt_tokens_total` | counter | `kind`: run, kept |
   | `ds_generated_tokens_total` | counter | |
   | `ds_prompt_seconds_total`, `ds_generation_seconds_total`, `ds_queue_wait_seconds_total`, `ds_first_token_seconds_total` | counter | |
   | `ds_context_total` | counter | `how`: continue, saved, empty (from step 1) |
   | `ds_reply_parse_failures_total`, `ds_stream_mismatch_total` | counter | |
   | `ds_engine_up`, `ds_engine_busy`, `ds_requests_waiting` | gauge | |
   | `ds_context_tokens`, `ds_saved_point_tokens` | gauge | |
   | `ds_start_time_seconds` | gauge | |

   No histograms for now (the person, 2026-11-19): nothing collects the metrics yet, and one
   request runs at a time. Speeds are ratios of the counters (generated tokens over
   generation seconds).
3. **Done (step 166).** **A log of requests as JSON lines** (`--log FILE`): one object per request with the time,
   the request's number and message id, every field of today's summary line, `saved`, how the
   context was reached and the time to the first token. The time of day on the stderr lines;
   `--dump` also for cancelled, failed and refused requests.
4. **More of the engine in the `end` line**, for a second set of metrics: the speculative
   loop's steps and accepted drafts, the routed experts on GPUs and on the CPU, bytes read from
   storage (step 155's cold page cache would have shown at once), the GPU clock setting (a
   level, or auto after a reboot). The same numbers `--record` writes, so the code that takes
   them is shared with it.

**Not decided.**

- A comparison of texts in the server (the last prompt and reply against the new prompt: which
  message differs first). Easier to read than token ids, but the tokens are what the engine
  compares; only if step 1's line is not enough.
- The request number in the engine's stderr lines: it needs the number in the request's header
  (a new field of the protocol).
- Setting up a collector (Prometheus) and what to draw: the person's, after step 2.

## Two machines (2026-11-12; removed 2026-11-22)

**Removed** (the person, 2026-11-22; README step 178): for this model and this hardware the aim
is one solid, well-tested and clear single-machine implementation. The two-machine path gave
+14% on short runs and nothing on long ones, was not parallel, was a second branch in every call
of the runtime, and was the least tested path (nothing after step 112 ran on it: not sampling
with `--spec`, the server, records). In three commits: the link (`--remote`, `--serve`,
`src/peer.h`; step 178), the layer range (`--layers`, `--teacher`, the stream in and out; step
179), a last read of the code and the documents (step 180; most documents moved with the two code steps). The last commit with all of
it: `efa4e03`; with the layer range: `519ffd4`. The measurements, the traces and the two
scripts stay under `results/split/` and `results/trace/two-machines/`. What follows is the text
as it stood, for what it says about one machine (the CPU's experts are what a second machine's
GPU memory bought).

**Was done** (README steps 110-112): a second identical Mac Pro; the model cut at layer 20, the only
cut where nothing but the residual stream crosses (compressed caches and index picks stay within
each half; the engram is in the front, the draft and the head in the back); the halves as two
processes over TCP (`src/peer.h`), the front driving; one merged trace. Held-out mean 44.60 ->
50.85 t/s, outputs identical.

**What the split is and is not.** It is GPU room: each machine holds ~88% of its half's expert
pairs (42% before), so the CPU's experts (26.7 ms of a 6-token code step) are gone from both
halves. It is not parallelism: a 6-token code step is front 36.4 ms, then back 36.1 ms, then the
draft 9.35 ms, and one machine idles throughout. A long reasoning run, whose cache already had
97% of the uses on GPUs, runs at the same 57.0 t/s.

**Ideas, none pursued (closed with the removal):**

| idea | what it could give | what it needs |
|---|---|---|
| The draft without waiting for the commit | ~1 ms per step (the draft call queues behind the back's commit, which seeds the draft's rings) | the verify answer carrying enough for the back to commit and draft at once (the back can apply the accept rule itself), or the seed done lazily |
| Overlap between the halves | up to the smaller half of a step: the front runs the next step's layers 0-19 while the back runs this step's 20-39 | the next step's tokens before the back's answer: a guess (the drafts all accepted, the next draft from the back). Steps 106-107 measured such guesses on one machine: right 25-45% of the time, and a draft made without fresh states is weak. To be estimated from the two-machine logs before building |
| The dense part over each machine's idle dies | each half's ~36 ms is one die's dense work | this machine's XGMI pair is not on the second; attention split by heads was studied (step 76) |
| A cache budget per half | the cache was tuned for one machine (800 loads/s every 8 tokens) | a sweep; not parameter tuning for its own sake, since the split changes what is worth loading |
| Sessions and serving on two machines | `scripts/chat.sh` and a server use one machine | the back serving more than one front in a row (a reset of its context), a launcher |

## Further optimization ideas (2026-10-29)

From the verify-step breakdown with the cache (README step 33: per verify step + draft ~122 ms
at sclk 8: dense part ~64%, CPU experts ~20%, draft ~10%, GPU experts ~3% visible). Brief; each
needs its own measurement first. Order (then): after long context (README step 34).

**Decode profile (README step 49, sclk 6, the cache warm):** a verify step is 62 ms (chat, 2
tokens) to 134 ms (code, 6 tokens); the GPU waits 25-38% of it for the CPU's experts (~0.3 ms
each, the single-token kernel reading at ~63 GB/s); then the attention core (per token, ~10% on
code), the router (~4%), launches (at most 7-12%), the small-batch fp8 kernels (~5%). Order:
the CPU's misses first (fewer: cache tuning, A; cheaper: the single-token CPU kernel's memory
bandwidth), then the batched attention core.

**Done since (README steps 50-52):** the CPU kernel's preparation in parallel (+2.6%), the cache
at 800 loads/s and every 8 tokens (+5.6%; held-out mean 30.3 t/s at sclk 6), a per-layer miss
histogram (cost is linear, ~0.27 ms per distinct CPU expert, little hidden: the hit rate is the
target), `--trace` for Perfetto, and a prefetch of the next layer's likely CPU expert measured
and not built (the guess is right ~20-28% of the time). Step 53: the router's top-k in one launch
for all tokens (+4.6%, held-out mean 31.7 t/s). Step 54: co-selection is too weak to place by
(at most 1-3% even with perfect balance). Step 55: a die's expert time is ~30 us per (token,
expert) pair, but reading each expert once for all its tokens (bit-identical) gained only with
heavy sharing (-21%) and lost without (+23%): reverted, documented in `results/fp4-grouped/`.
Step 56: the CPU's verify-step experts on a static schedule (15-22% faster in a benchmark), 2 MB
pages (no change) and CPU settings (performance governor, C6 off: nothing end to end) all
dropped. Step 57: that "fixed cost" was the multi-token dot routine on one-token groups; those
now use the one-token routine (bit-identical): +1.1%, held-out mean 32.0 t/s. Step 59: decayed
counts equal the window, hysteresis margins only cost, a lower candidate threshold (`--cache-min
1`) gains hits but loses ~0.7% live: the policy stays; more GPU slots are the cache's lever.
Step 60: the memory ledger. Experts are 84% of GPU memory; 64k context costs only 21 slots;
the draft's 8 GB on GPU 0 is 7.2 GB of its own mxfp4 experts, its dense part already fp8 (step
61: nothing cheap to free). Step 62: the output head's multi-input GEMV shares activations
through LDS (bit-identical): head 3.3 -> 2.2 ms at 5 inputs, +1.4% held-out (32.5 t/s). Steps
63-64: layer L's router input through the next layers' routers guesses their experts well (one
layer ahead 72% of the chosen in the top 6; 71% of the CPU's experts in 3 guesses vs 20% by use
counts), but loads must start 3-4 layers ahead, where the guesses are weaker: a GPU prefetch
simulates to ~2.5% at best (bound 10%), likely 1-2% real. Not built. Step 65: the multi-input fp8
GEMV re-decodes inputs per lane group for 64-lane rows (bit-identical fixes: ~1% at 3-4 inputs,
none at 6); the GPU rarely waits for the host (<= 3-5%: engram rows, the sum's prelude). Next:
the batched attention core for verify chunks (and prompt chunks), in stages, each bit-identical:
(1) the window ring at window + chunk slots (done, step 66); (2) attn_core in phases with the
small per-token kernels batched (done, step 67); (3) one attention launch per chunk (done, step
68: attn.core 17.7 -> 7.8 ms at 6 tokens, held-out 32.48 -> 34.33 t/s, the prompt +6-12%); (4)
the indexer batched (past ~2k context; open). Step 69: traces regenerated. Step 70: every decode idea re-scored
(README step 70): next the handoffs around the CPU's experts (~9 ms per code step), then the
cross-die latency on the XGMI pair (for attention over dies 0-1), then a CPU-cache prefetch from
the lookahead guesses; verify steps in halves (the prompt's overlap) tried: exact, 10-13% slower.
Step 71: the handoffs: a short forward's copies back from the dies issued before the host waits
(the CPU's experts start ~5 us after staging instead of ~80-190): held-out 34.33 -> 34.98 t/s,
generation after a long prompt +8-10%.
Step 72: the held-out history: every decode change measured on the same 8 prompts at sclk 6
(older commits rebuilt and rerun overnight; `results/heldout-history/`), 10.06 t/s (step 6) to
the current point; each kept step since adds its point.
Step 73: the engram fetch (~1.4 ms of host page faults while the GPU idled at each step's start)
read with pread, after layer 0 is queued, copied on its own stream: a step reaches layer 1 ~1 ms
sooner; held-out 34.98 -> 35.09 t/s (steps 0.4-2.3% faster on 8 of 9 prompts).
Step 74: the indexer skipped while it keeps every compressed position (<= 512 keys; its scores
were computed and dropped): held-out 35.09 -> 35.94 t/s; batching it for long context is F below.
Step 75: the fp8 GEMV at 2-8 inputs (attn.out ~12 ms of a code step at ~270 GB/s): fp16 inputs,
shared memory, fewer registers, a register cap: none faster at M = 4-6; it needs a new work split.
The traces in `results/trace/` were regenerated with step 74's build (c4bd99b).
Step 76: attention's output projections over dies 0-1 (flags in fine-grained memory, ~6-8 us per
hand-off; exact): +0.6%, not kept (~55 us per layer of hand-offs, and die 1's copy of the weights
cost its expert cache 1.41 GiB); `results/decode-next/attn-split.patch`. With the core and q_b
split too (KV state on die 1) est. +4-6%.
Step 77: the CPU expert call from cache (one expert warm in L2 + L3) is only x1.2 faster than
from DRAM (a plain read: x3-6): DRAM is not its limit, so a prefetch to a large LLC or HBM waits
on a faster call. Step 78 takes the call apart.
Step 78: a warm one-token call is ~2x its compute; ~45% of it is reading activation digits the
team just wrote. Per-thread copies: -25% back to back, nothing after the runtime's idle gap
(end to end flat); not kept. After the runtime's idle gap a call loses only 5-10% (libgomp's
workers spin through it). Step 79 (perf counters): the digits read slowly because 16 threads wrote
them (one writer: no penalty; whole lines per thread: -40%); aligned 16-block chunks +0.2% held-out,
not kept. Step 80: an AVX-512 canon_act_blocks (the same bytes, ~12.5x faster) and one writer per
token / pair: a CPU call -5-14% after the gap, held-out 35.94 -> 36.31 t/s. Next on the CPU: a task
scheduler that claims ahead (overlap memory and compute), then a CPU weight layout.
Step 81: a call's first phase loses ~0.03-0.04 ms after the runtime's idle gap (the master's core
idles); C6 off, polled HIP waits and a worker as the writer do not help; a spinning host wait
helped an A/B but was -0.5% held-out. Not kept.
Step 82: the CPU's 32-row tasks handed out in pairs (dynamic, 2): contiguity, not prefetching the
next task, makes a cold call faster; a call with 2-6 experts -12-13% after the gap, held-out
36.31 -> 37.16 t/s (+2.3%). Next: the CPU weight layout (cheaper decode).
Step 83: the CPU weight layout benchmarked (bench only): the same bits, compute x1.8, a cold call
-9% (8 KB prefetch); est. +1.5-2.5%. Not deployed: it needs a repacked ~269 GB expert file as the
only source of expert weights (CPU kernels, the GPU loader's unpack, model loading).
Step 84: traces split the CPU's handoff back (new `scripts/trace_critpath.py`): the copy of the
CPU's rows waited behind the cache's background loads on die 0's copy engine; copied by a kernel
from the pinned buffer instead: held-out 37.16 -> 37.57 t/s. Left: the dies' own small copies
behind their loads (~3 ms per code step; kernels there and smaller load pieces were slower).
Step 85: the background loads paced by the layers (k matrices per layer right after it, or none
during the expert phase): slower or no change; not kept (`results/decode-next/load-gate.patch`).
Step 86: the draft in the traces (its parts, commit, the loads per die): drafts doubled while the
cache loaded into die 0; the home die now keeps its startup experts (dies 1-3 adapt): held-out
37.57 -> 40.82 t/s. Ideas, not built:
- how out of date die 0's set is: per rebalance its experts' share of the uses the best set of
  that size would serve, and the CPU pairs it would have absorbed (ms); if it grows in long runs,
  its loads at the start of the CPU's experts (die 0 idles ~0.8 ms there), one matrix per layer;
- die 0 seeded with the experts used most across kinds of prompts (the stable ones), dies 1-3
  left for the prompt's own;
- the draft's 384 experts (7.2 GB on die 0): measure its router's usage; if skewed, keep the
  used half (~190 slots freed; acceptance may drop, outputs cannot change), first a cache
  simulation of what the slots are worth;
- the draft's window seeded for all of a commit's kept tokens at once (commit's ~1 ms).
Step 87: the draft router's usage: concentrated per prompt, unstable across prompts (top 64 of
seven prompts cover 88% of the eighth's choices, 70% for the worst); fixed pruning not done.
Idea: a draft-expert cache following the prompt (~120 of 384 used per run; ~4-5 GB). The verify
step's tail: the host's argmax of all logits ~0.3-0.8 ms plus their copy ~0.2-0.3 ms.
Step 88: that argmax on the GPU (M ids back): held-out 40.82 -> 41.19 t/s.
Step 89: traces at step 88; the time outside the layers split: ~2-3 ms untraced, almost all the
head (the bf16 head kernel, as the draft's). Sections B and D below updated.
Step 90: the GPU clock: held-out 36.8 / 41.1 / 42.1 t/s at sclk 2 / 6 / 8; ~35% of a token's time at
sclk 6 scales with the clock, ~65% (the CPU's experts, handoffs, host, and HBM-bound GPU work such
as the fp8 GEMVs; step 94's correction) does not.
Step 91: the CPU layout as a copy in RAM: -7-12% per CPU expert in the same run, held-out +1.6%
with all layers, but 269 GiB, ~45 s per start and an unexplained abort; not kept
(`results/cpu-decode/layout-live.patch`).
Step 92: the draft's head over part of the vocabulary, simulated: every deployable subset loses
(32k -8.8%, with the run's own tokens -4.8%; an oracle at most +4%); not built. Open: an abort at
exit, rare (2 in ~40 runs), seen with and without step 91's code (`malloc_consolidate` / `double
free`, right after the expert cache's stats): a teardown bug to find.
Step 93: that abort not repeated in 56 runs (plain, gdb, ASan: no double free on every run); a
race at shutdown, likely; a stack handler and the exit code in the held-out summary kept for the
next one.
Step 94: step 70's verify halves traced: die 0 reads its weights once per half (two 3-token halves
~84 ms against 54.5 for one 6-token step), more than the CPU's time they could hide; closed.
Step 95: llama.cpp upstream reads our GGUF's Engram table (the PR's MXFP8, type 43) as plain fp8
and generates garbage; the fork stays until a newer model is quantized with the merged code
(`docs/llama-cpp-upstream.md`).
Step 96: the exit abort revisited: its window includes the loader's shutdown (step 93 had it after);
three teardown fixes (stop under the mutex, the loader threads' frees one at a time, the expert
tensors unregistered before the unmap); the handler prints the shutdown phase, the thread and the
loads per die. Still open until it either recurs with that output or stays away.
Step 97: the dies' results in smaller copies and the staged block in one: 41.10 -> 41.96 t/s.
The exit abort recurred with step 96's fixes: after main, in the HSA runtime's shutdown (documented,
not worked around).
Step 98: the compressor's projections once per chunk: 41.96 -> 42.52 t/s.
Step 99: the review's four leftovers: none kept (patches in results/decode-next/).
Step 100: traces regenerated at step 98; section B's spans refreshed.
Step 101: the attention output matrices' rows padded by 64 bytes on the GPU: 42.52 -> 43.55 t/s.
Step 102: light launches of exactly 2048 waves sent as two, the router's and compressor's rows
padded: 43.55 -> 44.00 t/s. Expert slots padded, fp8 GEMVs as two launches: not kept.
Step 103: query B and output A do not pay for their multiples of 2048 waves (closed); the
several-input fp8 kernels run in rounds of ~512 waves (measured, nothing changed).
Step 104: the several-input fp8 kernel (64-lane order on 16 / 32 physical lanes): 44.00 -> 44.57 t/s.
Step 105: traces at step 104; the kernel gives what the bench said (+2.0% on code back to back),
the shared expert's part hides behind the CPU's experts; section B's spans refreshed.
Step 106: drafting ahead estimated from the draft's logs: ~+1-2%, least on code; not built.
Step 107: a second draft block before the verify, in shadow: accepted too rarely (every rule
loses 1-4%, an oracle +3.2%); not built.
Step 108: an interactive chat session (`--chat -i`, `scripts/chat.sh`), replies identical to fresh
runs.
Step 109: thinking mode across turns (`Runtime::mark` / `rewind`), replies identical to fresh runs
on the reference's prompts; next towards a server: tool calls, a system prompt, the HTTP side.
Step 110: the second machine reproduces every output at the same speed; the runtime runs a range
of layers (`--layers`, the stream in and out), halves identical to the whole model.
Step 111: the halves as two processes over TCP (`hip/peer.h`), then on the two machines: held-out
mean 44.60 -> 50.85 t/s (+14.0%), outputs identical; the gain is the CPU's experts, nothing on the
long proof or on 1-2-token steps. Next there: the machines take turns (no parallelism), the dense
part of each is one die; the chat launcher and a server on two machines.
Step 112: one trace of both machines (clocks aligned to ~1 us); the documents cover both setups.
Step 113: review round 1 (error paths, checks between the halves, the launcher; the documents).
Step 114: review round 2 (the runtime, kernels and loaders; what it left open is listed in the README's step).
Step 115: testing from here on: a run as one record (`--record`, a first minimal version; the idea
and the next small steps in `docs/record.md`). No test changed yet.
Step 116: the GPU clock setting in the record (a level, or "auto": after a reboot the cards are on
auto and the level moves during a run).
Step 117: bytes read from storage per phase in the record; with the engram table out of the page
cache a new prompt reads its rows from disk and runs ~12% slower once (sql-report 41.2 against 46.8 t/s).
Step 118: the speculative loop's totals in the record (steps, drafts accepted, the steps by drafts
verified, draft and verify time).
Step 119: the routed experts in the record (uses on GPUs and on the CPU, experts placed per die, the
cache's rebalances and loads).
Step 120: the tuning knobs among the record's arguments (`cache_window`, `cache_min`, `cache_wait`,
`spec_n`), defaults written as values.
Step 121: the options and switches that perturb timing in the record (`profile`, `trace`, `spec_log`,
`log_routing`, every `DS_*` variable); `--profile` -8.0%, `DS_FP8_PAD=0` -7.5% on cpp-matrix, clean
runs within ~1%.
Step 122: the cache's loads per die in the record. Step 86's rule (no loads on the home die)
switched off on purpose: 13% of the loads there, -8.1%, drafts +50%, verify steps +3.5%.
Step 123: the draft's 384 experts in slots of the home die's pool (the same bytes, kernels, memory and
speed; drafted ids identical). Next: leaving some out (a router mask for the ones not resident; how
the resident set changes without loads into the home die during a run).
Step 124: fewer draft experts measured with a router mask (`DS_DRAFT_KEEP`), nothing left out yet: the
142 pairs other prompts never chose cost +1.5% steps; a set of 96 chosen for the prompt +1.4%, of 48
+4.6%; a set of 192 chosen without it +11.6% (ja-email halves). Next: leave the unchosen out for real.
Step 125: left out for real (`k_remap_ids`, the freed slots to the main model: 802 -> ~944 experts on the
home die): GPU share +0.1-0.6 points, speed -0.7% over 8 prompts (+1.1% .. -2.9%). Not the default.
Memory on the home die does not limit speed now: no draft-expert cache, no 2-bit draft for its slots.
Step 126: the record's first value, `output.hash` over the printed token ids (from the case study of
steps 123-125: every output comparison there was a `cmp` outside the record).
Step 127: `--record-label` (the caller's names for a run). The scripts sorted by kind (43: 13 sweeps, 2
gates, 5 table makers, ...; 31 with one commit). Next: a reader of records (group, mean, spread, a
condition), then a runner (prompts x configurations x repeats), then file each step's scripts with its results.
Step 128: the reader, `scripts/records.py` (groups by label, means and the spread of repeats, `--vs` prints
only differences outside the spread, `--same` is a gate). On step 125's records: four of eight speed
differences are inside the spread. Next: the runner.
Step 129: the runner, `scripts/run_matrix.py` (a matrix file: prompts, base arguments, configurations,
repeats; a labelled record per run). Step 125's comparison again from a 12-line file: the same result.
Next: file each old step's scripts with its results.
Step 130: done: 33 one-step scripts moved to their results, 11 stay (`scripts/README.md`). New
measurements: a matrix file beside its results, read with `scripts/records.py`.
Step 131: cleanup, part 1: `run()` split: `Args` + `parse_args()`, `write_record()`; behaviour unchanged
(old against new on 8 usage errors and 7 kinds of run). Next: the two generation loops as functions; then
the unadopted studies removed one per commit; `src/rt.cpp` last. The kernels are not touched.
Step 132: part 2: `Turn`, `generate_spec()`, `generate_plain()`, `report_shadow()`; `run()` 268 lines (was
664); eleven kinds of run the same old against new. Next: the unadopted studies, one per commit.
Step 133: the second draft block's study (step 107) out of the code: `DS_DRAFT2_LOG`, `draft_shadow`, `gap`;
the last commit with it: 012bf44. Next: `--load-test`.
Step 134: `--load-test` (step 30) out: the options, the spare slots, the loader's test mode; the last
commit with it: def1a2a. Next: the router lookahead (step 63), the draft's unused hidden-state logs,
`DS_DRAFT_KEEP`. Kept as checks: `DS_ATTN_CHECK`, `DS_EARLY_ROWS`.
Step 135: the router lookahead (step 63) out: 174 lines; the last commit with it: 8798740. The draft's
hidden-state dump stays: `ds-test` and `ref/dspark_check.py` use it. Next: `DS_DRAFT_KEEP`.
Step 136: `DS_DRAFT_KEEP` and the hidden-state dump stay for now. `hip/rt.cpp` part 1: the trace, the
profile's groups and the memory ledger into `src/rt_trace.cpp` (+ `src/rt_internal.h`); speed unchanged.
Next: the two-machine code into its own file, then the constructor in parts.
Step 137: the two-machine code into `hip/rt_peer.cpp`; both halves as two processes on one machine, old
and new builds in every pairing: the whole model's output. Two-machine speed not measured. Next: the
Step 138: the eight held-out prompts after the round, by `run_matrix.py` and `records.py` alone: all outputs
the stored ones' bytes, speed 44.75 -> 44.70 t/s (inside the spread). First reference values:
`results/record/reference/heldout-output.tsv`.
Step 139: programs out of the library directories: `apps/` (the CLI, `ds-inspect`, `fp8pack`), `tests/`,
`bench/`; `src/` and `hip/` are libraries only. All programs rebuilt; the held-out set matches the reference values.
Step 140: `rocprofv3` (rocprofiler-sdk 6.3.4) works on our binary: per-kernel device time, HIP call counts,
copies; -21% to -26% speed, same output. `docs/rocprof.md`. ~2,200 launches per generated token.
Step 141: its hardware counters work (with `hsa-amd-aqlprofile6.3.4`): few per run, 4-7 t/s, same output;
a first per-kernel table (memory unit busy, cache hits, kilobytes fetched). Values not yet checked against a known kernel.
Step 142: checked on `bench-fp8`: `FetchSize` exact (weights + 0.1-0.4%), a miss = one 64-byte line; the
"share of time" counters grow with a kernel's length and do not compare between kernels.
Step 143: from the kernel trace, the home die runs a kernel 50% of a (traced, plain) step; gaps between
back-to-back launches sum to 0.9 ms/token (the bound for fusion); the idle time is the wait for the CPU's
experts (17.7 ms) and a 176 us gap per layer not yet named (6.5 ms). Next: the same on the fastest settings.
Step 144: on the fastest settings the same picture: home die 54% busy, the wait for the CPU's experts 7.2 of
20.0 ms per token (traced), launch boundaries 0.17 ms; the other dies 14-15% busy.
Step 145: the 176 us gap named: the host issuing the three other dies' work (91% of it is the main thread in
HIP calls) before it issues the shared expert. Idle time of the die, not on the critical path by itself.
Optimization is paused here (the person, 2026-11-16). Left open: the loop's untraced host time; the shared
expert issued before the loop; the instruction counters unchecked.
Step 146: a perfect draft-length rule, estimated from `--spec-n 5` logs: 53.6 t/s against 44.8 on the held-out
mean (+19.6%; +29-36% on the three poor prompts, +8-12% on the good ones). A bound, not a plan item.
Step 147 (usability): `--think-show all|line|none`: the reasoning as one moving line on stderr, the answer alone
on stdout; `scripts/chat.sh` uses `line`. The default prints as before.
Step 148: towards a minimal server: a sampler (temperature, top-p; the random number a hash of seed and position)
as a library with its test (`src/sampler.h`, `bin/test-sampler`); no program uses it yet.
Step 149: `--temp`, `--top-p`, `--seed` in the plain loop: greedy unchanged, one seed gives one output on three
placements; not with `--spec` yet.
Step 150: `--temp` with `--spec`: a draft is accepted if it is the sampler's token at its position; outputs equal
the plain loop's; held-out mean 44.76 greedy, 41.79 sampled, as many tokens per step. Open: ~4-6 ms per verify step
(the logits' copy and the draws), sampling on two machines.
Step 151: the ~4-6 ms were the draws (1 ms each, six threads started per step), not the copy (0.3 ms). Vector `exp`,
draws in turn up to the first rejected draft: 2.4 ms per step, held-out 43.06 sampled against 44.94 greedy (-4.2%).
Open: ~0.5 ms per draw, sampling on two machines.
Step 152: `--requests`: one engine answers prompts from stdin, each on an empty context (`mark()` at 0, `rewind()`),
tokens as lines on stdout (`docs/requests.md`); replies equal single runs', with both loops. Next: the HTTP front.
Step 153: `server/ds_server.py`: the Anthropic Messages endpoint over `--requests`, the format by the model's `encoding.py`
(`docs/server.md`). The benchmark's script runs unchanged: a parsed tool call after 43k tokens of reasoning, 19 min.
Open: the OpenAI endpoint, tool results, a live stream, cancel, sampling on two machines.
Step 154: a cancel: `SIGUSR1` stops the engine's request between steps or prompt chunks (`stop=cancel`); the server
sends it when a client closes its connection.
Step 157: tool results in a conversation (`tool_result`, `thinking`, `tool_use` blocks read; a signature that is not empty),
for the pi harness; pi runs two small coding tasks (`results/server/pi.txt`): 101 of 118 s are prompts.
A loop's next prompt = the last prompt + the reply + the new turn. Open: a kept prefix, a live stream.
Step 158: the context kept between requests when a prompt goes on from it (by token ids; replies equal `keep=0`'s).
pi's two tasks 118 -> 63 s. Open: what a new conversation shares with the last (system prompt, tools), a live stream.
Step 160: a second saved point in the runtime, at the end of what two conversations share; from the third conversation
on a new one starts in 0.1-0.3 s, not 14 s (pi's third task: 7.6 s). Open: a live stream; two machines not tested.
Step 161: a stream carries reasoning and text as they are generated, tool calls whole at the end; its message is the
plain request's. Open: the OpenAI endpoint, tool calls streamed in pieces, forced tool use, sampling on two machines.
Step 162: the server's observability reviewed and planned (item 33, "Observability of the server"); nothing built.
Step 163: a trial build against upstream llama.cpp: with the Engram rows decoded by our own code the 8 held-out outputs
are the stored ones'. Documented (`docs/llama-cpp-upstream.md`), not done: the pin stays.
Step 164: item 33's first step: `ctx`, `common`, `how` in the engine's `prompt` line and its `# context:` lines on stderr
(where a prompt and the context part). Next: `GET /metrics`.
Step 165: item 33's second step: `GET /metrics` (counters and gauges, the Prometheus text format, `docs/server.md` "Metrics"),
`GET /health` 503 when the engine is gone. Next: the log of requests as JSON lines.
Step 166: item 33's third step: `--log FILE` (a JSON line per request, 30 keys), the date and time on the server's log lines,
`--dump` for every outcome. Seen in its first run: a request with reasoning and one without share 2 tokens (the effort line
comes first), which ends the saved point. Next: more of the engine in the `end` line.
Step 167: `scripts/serve.sh`: the server with the fastest settings and every log in a directory per start, for the person's
runs with pi; what those logs show decides what comes next.
Step 168: the Engram table decoded by our own code on both paths, with a size check at load: the code builds against
the pinned llama.cpp and against upstream `e626db7ca` with the same gates and held-out outputs. QUICKSTART names upstream;
the PR's commit stays for llama.cpp's own programs. Open: a serving build without ggml's CPU backend.
Step 169: item 34's first step: which code is the product, which checks it and which are tools, written in `README.md`
and `docs/overview.md`'s code map (now in three parts), with the four places where the product still depends on the rest.
Next: Makefile targets.
Step 170: item 34's second step: `make product` (`ds-completion`, `fp8pack`), `make tests`, `make tools`; `make` builds the
same programs as before. The product alone from nothing: 19 s (CPU), 82 s (`HIP=1`), against 58 and 240 s for everything.
Next: `engram_rows` and `Taps` out of `src/forward.h`.
Step 171: item 34's third step: `src/engram.{h,cpp}` and `src/taps.h`; nothing in `hip/` includes `src/forward.h`. Gates and
three held-out outputs the same bytes. Left, for the person to decide: `fakequant.cpp` in two, the CPU-only mode, `ds-test`'s place.
Step 172: the person: the CPU-only mode is a reference tool, not the product. `src/fakequant.{h,cpp}` now without ggml (the
host half), the graph ops in `src/fakequant_graph.{h,cpp}`; gates, the unit test, `fp8pack`'s file and three held-out outputs
the same. Next: `ds-completion` without the reference path (a build option or a program of its own).
Step 173: two programs from `apps/ds_completion.cpp`: `bin-hip/ds-completion` (the product: `--rt` required, no CPU path, no
`Engine` in the binary, 18.5 -> 16.7 MB) and `bin/ds-ref` (the CPU reference path; the old CPU build's bytes on a prompt).
Gates, held-out with and without `--spec`, the request test's 59 checks: the same. ggml is still linked, through libllama.
Step 174: the longer golden sets again (CPU path: 200 and 1,100 tokens; the runtime: 20,000): every per-layer table equal to
the stored one, the 20,000-token logits the bytes of 2026-10-29. Not run against the upstream llama.cpp build.
Step 175: the same and every other check with upstream llama.cpp (`e626db7ca`): all the same. It is now the one pinned
commit; the PR's commit is history, and `llama-logits` (right only with the PR's commit) is out of `make tests`.
Step 176: the checkout in `~/projects/llama.cpp`, the Makefile's `FORK` now `LLAMA_CPP`; the Python reference reads the Engram
table itself (it needed the PR's `gguf-py`, missed in step 175). Checks the same.
Step 177: `~/forks/llama.cpp` deleted; the documents reviewed against steps 168-176: the README's example command, the
"with `--rt`" qualifiers in QUICKSTART, a usage text per program.
Step 178: the two-machine link out of the code (`--remote`, `--serve`, `hip/peer.h`, the runtime's `peer` branches, the
clock samples in `--trace`); the last commit with it: efa4e03. Next: the layer range (`--layers`, `--teacher`, the stream).
Step 179: the layer range out of the runtime and the program (`--layers`, `--teacher`, `--stream-in`, `--stream-out`,
`src/rt_peer.cpp`); the runtime runs layers 0 .. k, all of them outside `ds-test --layers`; the last commit with it: 519ffd4.
Step 180: a last read after the removal: `verify()`'s unused `head` parameter out, two passages of the documents in the past tense.
Step 181: sessions, sampling and the longer golden sets on the build after the removal: all the stored bytes (3 h, 2 h of it
loads of the model). Next: the testing process (item 35).
Step 182: the documents, comments and names against the tree: ~70 corrections of text, the Makefile's `FORK` alias; the code
findings listed in `results/dist/review-182.txt` section 3 for steps of their own. Next: the testing process (item 35).
Step 183: `--requests`: a header that was not understood closes the engine wherever the mistake stands (a bad field after
`bytes=` left the prompt unread); eight header cases in the request test. Next: the testing process (item 35).
Step 184: `docs/testing.md`: the testing as it is, before it changes. Next: item 35 from that page.
Step 185: `docs/testing.md` 3.6: each test gives a verdict or says where a difference is; a tier should run verdicts and
keep the localizers for a failed one.
Step 186: the runs decided: `unit`, `default`, `speed`, `complete`, and localizers on demand; `scripts/check.py unit` (30 s,
no model files). Next: `default`.
Step 187: `check.py default`, the logits half: five `ds-test --e2e` runs against stored sha256, golden sets cut to tokens
and logits in git (`tests/data/e2e/`, 9.5 MB); 105 s with `unit`. Next: the outputs half (one engine, `--requests`).
Step 188: the outputs half (`tests/outputs_check.py`: 12 requests to one engine, one fresh session; token ids stored); `default`
is 13 checks in ~5.5 min. The "load" is 13 s; 105 s more are the registration of expert tensors for the cache. Next: `speed`.
Step 189: the registration measured (2.2-2.9 GB/s whatever the threads or flags; constant within a boot, 116-139 s between
boots); a process that ends first stops it (a chat session 126 -> 21 s). Open, the person's: loads before it is complete,
and how `speed` measures.
Step 190: `check.py speed` and `check.py complete`, both as things are measured today (the person: build the split first,
improve each run after); progress lines of the long checks in the run's own output. The four runs exist.
Step 191: the CPU path's expected logits at 1,102 tokens stored; against the reference 4 of 5 argmaxes (a near-tie on a bf16
set; an fp32 set of that length would decide it).
Step 192: the CPU reference path is to be removed (the person): first the rounding test moved to the runtime's host functions.
Next: its checks and `ds-ref` out; then `src/forward.*`, `src/fakequant_graph.*`, the CPU mode of `ds-test`.
Step 193: `ds-ref` and the CPU path's checks removed; `apps/ds_completion.cpp` builds one program (`default`: 11 checks, 5 min).
Next: `src/forward.*`, `src/fakequant_graph.*`, the CPU mode of `ds-test`.
Step 194: done: the sources and `ds-test`'s CPU mode removed (793 + 138 lines); `ds-test` is the runtime's test, HIP=1 only.
The CPU reference path is gone: one implementation, checked against the official reference.
Step 195: the directories (the person): every test under `tests/`, data under `tests/data/`. Next: `hip/` into `src/`; `bench/` to `tools/`.
Step 196: `hip/` merged into `src/` (the engine, 34 files). Next: `bench/` to `tools/`.
Step 197: `bench/` is `tools/`, with `ds_inspect`. The directories follow the three groups: product (`src/`, `apps/`, `server/`), `tests/`, `tools/`.
Step 198: `complete` whole for the first time: 16 of 18 pass, 6 h 43 min; no output differs; two crashes in the ROCm runtime (the exit abort; a segfault at a
loader thread's first allocation, perhaps the same fault); registration 285 s per start in this boot. Next: the registration.
Step 199: the slow registration is the page cache's layout: filled by the engine's page faults 285 s, after a drop and a sequential `cat` 75 s
(1.0 -> 4.2 GB/s). To decide: a script for it run by `speed` and `complete`; the engine's hint; loads before the registration is complete.
Step 200: the load studied: 12.9 s (the bus allows ~5), then 75-285 s of registration the cache waits for; huge pages register at 9.7 GB/s.
Proposed: cache loads at once by the copying path; the upload at the bus rate; the registration short. The person decides.
Then: written up as item 36, "The load"; the person tests by hand first.
Step 201: the registration and the stalls studied (`docs/registration-and-stalls.md`): page tables of 330-540 MB per die; HIP's per-device locks; a path below HIP
(lock for dies 1-3, ROCr's copy) ~3x faster in a probe; loads into die 0 hold its kernels up every ~93 us (cause open), loads into die 1 cost it ~6%.
To decide: pinned-only loads measured first; die 1 last for loads.
Step 202: from the kernel: the registration is 854,000 page-table jobs on the copy engine's queue (one per 75 us); the stalls show nothing on the host (no driver work, no interrupt inside a held-up kernel) and do not depend on the copy's piece size.
Step 203: measured for this item: pinned-only loads -6.6% on the held-out prompts (nothing on the long prompt; mechanism open), no loads before the registration ends -34.2%, so loads through the pinned buffer until the registration is done are worth having; the reserve at 0.3 GB without registration: no gain today; the model on 2 MB pages of the hugetlb pool registers in 22-37 s with page tables near zero (474 GiB reserved after a boot: untried). Next: why pinned loads cost what they do.

**Triage (2026-11-07, after README step 90).** A 6-token code step (~89 ms traced): die 0 waits
~25 ms for the CPU's experts (the part the GPU clock does not touch, step 90), die 0's dense front
~54 ms (fp8 GEMVs ~25), the draft 8.4, the tail after the CPU's rows 6.3; outside the layers ~2-3
ms untraced (step 89). Die 0 and the CPU still take turns.

Promising, in order:
1. Tried (step 91, not kept): +1.6% with all layers as a copy in RAM, too costly for the gain.
   Only as a repacked expert file that is the only source (the GPU loader reading it), if ever.
2. Closed (step 92, simulated): a vocabulary subset for the draft's head loses at every size.
3. Closed (step 94): verify halves double die 0's weight reads (+~30 ms per code step), more
   than the CPU's 25-30 ms they could hide.
4. Done (step 97): the tail was the dies' worst-case copies, not their loads: 3 rows per token
   first, the staged block in one copy, +2.1%. With it the compressor batched (step 98, +1.3%).
   The review's leftovers (step 99): hc kernels fused, the compressor's state per chunk (prompt
   +1.1%, decode 0), a gather stream per die: tried, not kept; the draft's head in fp8 waits for
   the multi-input fp8 kernel.
5. A simulation of ~190 more slots on die 0 (`results/cache_sim.py`): what a draft-expert cache
   would buy, before building one.
6. Commit's 0.86 ms wait: the draft's window seeded for all kept tokens at once (~1%).

Step 101: the fp8 matrices with rows a multiple of 4096 bytes apart had read at half speed
(cache collisions); padded by 64 bytes: +2.4%. Step 102 closed the same check elsewhere: only the
router gains (padded), the head, hc projection and expert slots do not. It also found that a
launch of 2017-2048 waves takes ~48 us whatever it does (`bin-hip/bench-wave-slots`): the light
ones (compressor, a 4-token quantize) go out as two launches, +1.0% together. Query B and output A,
on multiples of 2048 waves, do not pay for it (step 103).

Done (step 104, +1.3%): the fp8 GEMVs at 3-8 tokens on 16 / 32 physical lanes carrying the 64
partial sums; attention output A 113 -> 79 us at 6 tokens. The runtime gets what the bench
measured (step 105). Left there: the prompt kernel on the large matrices, 2-token calls.
Was: the fp8 GEMVs at 6 tokens (step 101's probe: the multiplies fit, the
activations' loads per lane and step are the cost; a kernel that keeps the 64-lane order of
additions on an 8- or 16-lane work split; local variants failed in 65 and 75; step 103: at
several inputs a launch just over a multiple of ~512 waves pays a whole extra round, query A
640 waves, the shared expert's gate / up 1152, down 640: ~1-1.5% of a step in those tails);
drafting ahead while the verify runs: estimated in step 106 at ~+1-2% (the guess holds in 25% of
steps, the draft needs the verify's last layers, code steps have no sixth guess); not built.

Backlog (less likely): cache policy tweaks (59 lost; 63-64 2.5% simulated), an LLC prefetch (77:
x1.2 ceiling), CPU micro-fixes around the idle gap (78, 79, 81), load pacing (85), attention over
dies 0-1 / TP (76: +0.6%), small-op fusion (65: launches not the limit), speculation tuning and
trees (94% of the hindsight bound), the fast path's activation (held-out runs `--canon`), PGO and
compile-time shapes, the batched indexer (long context), die 0's staleness metric (long runs).

**A. Tuning the expert cache** (the CPU experts' ~25 ms)
- Hysteresis: a newcomer must beat its victim by a margin, so the cache settles in long runs
  instead of churning at the noisy boundary; window length / decayed counts; the default rate
  (100-200 loads/s). Simulate first (`results/cache_sim.py`), then live.
- Place newcomers so each layer stays spread over the dies (today a newcomer takes its victim's
  slot).
- More slots: compressed caches in fp8/fp4 (part of "Long context") and anything else freed on
  die 0.
- The fast path's activation (`fq_fp8_host`, scalar, written in parallel 8-block tasks): the
  canonical path's step 80 (AVX-512, one writer per token / pair) has no fast-path counterpart;
  the CPU's fixed per-layer latency (~0.03-0.04 ms after the idle gap, step 81).
- Hardware (discussed, not planned): DDR5 or a multi-socket box with experts striped by rows
  across sockets (exact under the canonical numerics).

**B. The dense part of a verify step** (updated after README step 105: cpp-matrix's die 0 spans
in step 104's traces, means over all its verify steps, ms per step: ffn.shared 8.3 (while the CPU
computes: hidden), attn.out 7.7 (11.5 at step 98), attn.core 5.7, attn.q 5.1, hc 5.0, the wait for
dies 1-3's experts 3.8, attn.compressor 2.5, router 2.4, the head 2.2, dispatch 2.1, attn.kv 1.7,
indexer 1.2, engram 1.1; die 0 waits 30.7 for the CPU's experts. A 6-token step, medians: 84.8 ms,
front 48.2, the wait for the CPU 26.7, the tail 4.7, outside the layers 5.0; the draft 8.1 more)
- Done: the router's top-k in one launch (step 53), the head through LDS (step 62), the batched
  attention core (steps 66-68: 17.7 -> ~6 ms), the engram fetch off the critical path (step 73),
  the indexer skipped while it picks every position (step 74), the head's argmax on the GPU
  (step 88), the dies' results in smaller copies and the staged block in one (step 97), the
  compressor's projections once per chunk (step 98).
- Tried, not kept: verify steps in two halves (step 70), the fp8 GEMV for 2-8 inputs as local
  variants (steps 65, 75), attention's output over the XGMI pair 0-1 (step 76), the hc kernels
  fused, the compressor's state per chunk, a gather stream per die (step 99).
- Left: the fp8 GEMVs at 6 tokens (attn.out, shared expert, q: ~25 ms, weights read at ~270 GB/s
  against ~580 at one token) need a restructured kernel: more rows per wave without holding M
  decoded inputs (step 75's conclusion).
- Kernel fusion of the small ops (hc, norms, quantize; item 14): launches are not the limit
  (step 65: the host is ahead; step 99: the hc kernels fused took the same time).
- Tensor parallelism (items 17, 18, 23): step 76 split attention's output over the pair 0-1 and
  lost to the exchange; not before the kernels above.

**C. Tuning speculation**
- The draft-length rule refits its verify-cost line online, so with the cache it already sees
  cheaper verifies; check that it now picks longer drafts and whether the backoff still pays.
- A tiny tree (top-2 at the first draft slot): the sandbox's offline study (`dspark-tree`, on
  V4-Flash) found trees buy little (+0.2 accepted tokens for one extra draft token, +0.7 for 25x
  the budget), so only the smallest trees are worth a test.
- Per-prompt-type expectations: code accepts ~2x more than chat; nothing to tune, but the
  held-out mix decides the average.
- Momentum (trust drafts more after recent successes): checked and rejected (README step 47):
  past success adds nothing beyond the confidence head; rechecked live with the cache (step 48):
  the same, and the head stays calibrated.
- The rule is at 94% of a hindsight lower bound per step (step 48): little left in draft-length
  choices; the cost side (verify step, draft) is where speculation still gains.

**D. A faster draft** (updated after README step 88: 8.3 ms of GPU time per call on die 0, GPU-bound
(step 86); 3 layers, fp8 dense weights (step 61), 128 mxfp4 experts top-3 per layer, all resident)
- Its parts per call (step 86's traces): the head 2.2 (the main model's bf16 head, 1.32 GB, for 5
  tokens), routed experts 1.7, the Markov part 1.0, attn.out 0.8, attn.core 0.75, attn.proj 0.6,
  shared expert 0.5, hc 0.3.
- Done: fp8 dense weights (an exact repack at load), the head through LDS (step 62), no cache
  loads into its die (step 86: drafts had doubled while die 0 loaded).
- The head over a subset of the vocabulary (the most frequent ~32k tokens by a row-index list,
  from prompts outside the held-out set): ~2.2 -> ~0.6 ms; outputs cannot change (the verify
  decides), its confidence is its own projection (not the logits); acceptance may drop on chat
  in other languages.
- Its experts in less memory (step 87: concentrated per prompt, not across prompts): a draft
  expert cache following the prompt (~4-5 GB of the 7.2 freed for the main model); a cache
  simulation first.
- Overlap: start drafting as soon as the verify's accepted prefix is known (a few ms), or draft
  ahead assuming every draft is accepted, kept only if it was; on another die it needs the
  draft's weights there.
- Batching it with the verify (one forward for both) is the larger structural idea, and the
  hardest.

**E. Build-level specialization** (2026-11-01, after README step 45's -O2 finding)
- PGO (`-fprofile-generate`, a prompt or two, `-fprofile-use`) for the host code. Expected
  small (0-5%): the hot CPU loops are hand-written AVX-512 whose instruction mix the numerics
  fix; PGO helps inlining, branch layout and unroll choices, which mattered once (-O2) and is
  fixed. Risk: GCC fuses multiply-adds by default (`-ffp-contract=fast`) and PGO changes
  inlining, so plain C++ numeric code (SiLU, fake quantization) could change bits between
  builds; the canonical numerics must stay bit-identical to the GPU's, so every PGO build
  would need the gates. First `-ffp-contract=off` on the numeric files (bits independent of the
  optimization level), then a ~30-minute experiment (speed and output bytes).
- Compile-time shapes (E = 5120, F = 2304, 6 experts per token) as template constants in the CPU
  kernels: full unrolling without runtime bounds; cheap, deterministic per build.
- The GPU side has no practical PGO in ROCm's device compiler; its equivalent is the kernel
  sweeps already done (README steps 39-40).

**F. The indexer batched over a verify step's tokens** (2026-11-03, long context only; README step 74)
- Today the indexer (layers 2, 8, 14, 20, 24, 28, 32, 36) runs one token at a time inside
  `attn_prep`: ~6 launches per token, `idx_q_b` (fp8) and `idx_proj` (bf16) read again for every
  token, ~76 us per token and source layer at short context (~3.4 ms of a 6-token code step).
  Step 74 skips it while every compressed position is picked (<= 512: up to ~1,024 tokens on
  layers 2-19, 512 on 20-39), so this only matters beyond that, where scoring also grows with
  the context (`k_index_scores` over n / ratio keys, then `k_topk`).
- Batched: the two projections, the rope and the fake quantization for all M tokens in one
  launch each (they do not depend on the compressor's per-token state; `q_b` is already batched
  the same way); one scores kernel for M queries with each token's own key count (a group
  completed mid-chunk adds a key for the later tokens only), reading every key once for all M;
  then the top-k per token (M blocks in one launch), the candidate blocks likewise.
- Exactness: each token's scores and picks must equal the one-token path bit for bit (the
  batched gemv rows already are; the scores kernel needs the same per-key reduction order, the
  top-k the same tie rule); gate with an `attn_check`-style comparison and the 6,106-token
  prompt.
- Measure first: decode t/s against position (step 34's 0.3k / 7.4k / 29.6k points) and the
  indexer's share of a verify step there.
