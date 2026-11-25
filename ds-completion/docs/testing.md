# Testing: the current state (2026-11-23, step 184)

**Since step 186** the tests are being gathered into a few fixed runs of `scripts/check.py`:
section 7. Sections 1 to 6 describe the tests themselves and the practice before that.

What checks this engine today: the terms, every test with what it tests, what it is compared
against, what it needs and how long it takes. This page describes; it changes nothing. The plan
to make the process shorter is [plan.md](plan.md), "The testing process" (item 35), and it
starts from this page.

Times are from this machine at sclk 6 with the model in the page cache. "Measured at step N"
names the results file's step; times without a step were measured for this page
(`results/dist/testing-times.txt`).

## 1. What "correct" means here, and the chain of evidence

The claim: the product (`bin-hip/ds-completion --rt ...`) computes what DeepSeek's official
implementation computes for this checkpoint. Nothing compares the two directly on a real
prompt at full speed; the claim rests on a chain, each link checked by its own test:

| link | claim | checked by |
|---|---|---|
| 1 | the GGUF files hold the official checkpoint's weights, tensor by tensor | `ref/check_weights.py` |
| 2 | the official code, run on the CPU from those files, is the reference | `ref/ds_ref.py` (the official `model.py` unmodified, on pure-torch stand-ins for its GPU kernels, `ref/kernel_cpu.py`) |
| 3 | our rounding functions are the reference's, bit for bit | `test-fakequant` |
| 4 | each layer of our code, given the reference's input, gives the reference's output | `ds-test` per layer, on the golden sets |
| 5 | the layers together give the reference's logits | `ds-test --e2e` and `ref/compare_logits.py` |
| 6 | the selection kernels, the sampler and the draft are the reference's or meet their own specification | `test-topk` with `ref/check_topk.py`, `test-sampler`, `ref/compare_dspark.py` |
| 7 | everything faster than "one token at a time on one placement" changes no output: prompt chunks, speculation, the expert cache, sessions, requests, the server | outputs against stored outputs (sections 3.3 to 3.5) |

Links 1 to 6 compare against the reference. Link 7 compares the product against itself: the
plain greedy output was established once, and every faster path must print the same bytes.

## 2. Terms

- **The reference.** The official implementation (`inference/model.py` of the release, in
  `~/llms/ds-flash/`), run by `ref/ds_ref.py` in the Python environment `~/venvs/ds-ref`. Slow
  (minutes per forward) and large (~500 GB of RAM at 20,000 tokens). Not to be confused with
  the next term.
- **The CPU reference path** (history; in older steps and results). Our own first
  implementation of the model, in C++ on ggml graphs (`src/forward.*`, the class `Engine`, the
  program `bin/ds-ref`, `bin/ds-test` without `--runtime`); not the official code. Removed in
  steps 192-194: the runtime is compared with the reference directly. The last commit with it
  is `bb42f51`; its stored tables (`results/test-*.txt` without `-runtime`) stay as history.
- **The runtime.** The product's implementation on the GPUs (`src/rt.cpp` and what it uses, the class `Runtime`; in `hip/` until step 196).
- **A golden set** (`data/golden/<name>/`, made from `data/ref/<name>.pt`). One run of the
  reference on one prompt, recorded: for every forward (the prompt, then each decoded token)
  and every layer, the layer's input and its intermediate and output tensors, the routed
  experts' ids, the indexer's selection, and the logits. `ref/export_golden.py` turns the
  reference's `.pt` file into a flat `data.bin` + `index.txt` that the C++ tests read without
  torch. A golden set is fixed data: making one needs the reference, using one does not.
- **A variant** of the reference. The real reference computes in bf16 and rounds activations
  to fp8 / fp4 ("fake quantization"). One bf16-against-f32 difference before a rounding moves
  a value by a whole fp8 step, so against the real reference a correct implementation shows
  2-10% error per layer, which hides bugs. So the reference is also run in **fp32** (`-fp32`
  in a set's name: activations in fp32, rounding kept) and without rounding (**`-nq`**, "no
  quant"), and each of our modes is compared with its match, where agreement is ~1e-7 to 1e-3
  and a structural bug stands out.
- **Teacher-forced, per layer.** Each layer is given the reference's own input for that layer
  and its output is compared with the reference's. Errors cannot accumulate from layer to
  layer, so a row of the table tests one layer.
- **End to end (e2e).** The whole model on the golden set's token sequence; the logits of
  every step are written to a file and compared.
- **A table** (`results/test-*.txt`, `results/rt-*.txt`). `ds-test`'s output: one row per
  step and layer, the relative error of each intermediate, the share of tokens with the same
  routed experts, the share of queries with the same selection. `ds-test` has **no pass or
  fail**: it prints numbers. The gate is that a new table equals the stored one in every
  column but the last (milliseconds).
- **Stored bytes, a stored hash.** An output file or an FNV-1a hash of the output's token ids
  kept in git from an earlier run. "The stored bytes" in a step means `cmp` found no
  difference.
- **Byte-identical output and identical logits** are different strengths. Greedy decoding maps
  many logits to one token, so identical text does not show identical arithmetic. Comparing
  logits files byte for byte does.
- **`--canon`.** The numerics mode in which an expert gives the same bits wherever it runs
  (CPU or any GPU). With it the output does not depend on the placement or on the expert
  cache's state, which is what makes "the same bytes" a valid check across configurations.
  Without it (the "fast" mode) outputs can differ between placements.
- **The held-out prompts.** Eight prompts (`results/spec-heldout/prompts.txt`: chat, code,
  SQL, math with thinking, Chinese, Japanese, JSON; 300 tokens each, 98 to 345 generated) that
  were not among the 30 prompts of the routing study the placement comes from. The standard
  set for output and speed.
- **A record** (`--record FILE`, [record.md](record.md)). A run as one JSON object: its
  arguments, the output's hash, speeds, counters. `scripts/run_matrix.py` makes them for
  prompts x configurations x repeats from a **matrix** file; `scripts/records.py` reads them.
- **A verdict and a localizer.** A test's result is either a verdict (same or not, pass or
  fail) or an account of where a difference is (which layer, which token, how far off). Few
  tests give both: section 3.6.
- **A gate.** Whatever a step ran to show its change broke nothing. Until now each step chose
  its gates and wrote its comparisons by hand.
- **The first pass after a boot** is not a measurement: prompts read from storage while
  generating and run 7-14% slower. Outputs are unaffected.

## 3. The tests

Columns: **needs**: M = the model's nine GGUF shards (474 GiB, in the page cache), F8 =
`dense-fp8.gguf`, D = the draft's GGUF, GPU = the four dies with ROCm, G = golden sets under
`data/golden/` (not in git), py = `~/venvs/ds-ref` (torch), ref = the official code and
checkpoint access. **verdict**: "auto" = the test itself exits non-zero on a failure;
"by hand" = someone compares its output with a stored file.

### 3.1 Without the model (seconds)

| test | target | compared with | needs | time | verdict |
|---|---|---|---|---|---|
| `bin/test-fakequant` | the host rounding functions the runtime calls (`src/fakequant.cpp`: `fq_fp8_host`, `fq_fp8_split`) and the scalar fp4 rules, put together as the reference does; no ggml (step 192; before it, the reference path's graph ops only) | `ref/kernel_cpu.py`'s outputs on 45,440 values per function, with ties of both grids (`tests/data/fq/`, made by `ref/make_fq_test.py`): bit-identical | nothing (`tests/data/fq/` is in git since step 186) | < 1 s | auto |
| `bin/test-sampler` | `src/sampler.*`: temperature, top-p, one seed gives one draw sequence, distributions (chi-square) | its own specification | nothing | 19 s (80 s until step 186: the comparison sorted the whole vocabulary for every draw) | auto |
| `ref/check_topk.py` (runs `bin-hip/test-topk`) | the indexer's selection kernels: top-k, candidate blocks | torch's `topk` and the reference's `select_candidate_blocks` on 46 cases | GPU, py | 3 s | auto |
| `bin-hip/bench-kernels --bitwise` | the batched (M-token) kernels | the one-token kernels: bit-identical | GPU | 1 s | auto |
| `python3 tests/test_server.py` | `server/ds_server.py`: prompts rendered, replies parsed, streams, refusals, cancels, metrics, logs | canned replies from `tests/fake_engine.py` | the model's `encoding.py` (`~/llms/ds-flash/encoding/`) | 8 s | auto |

### 3.2 Against the reference, on golden sets

The golden sets (all in `data/golden/`, 73 GB together, not in git):

| set | prompt | variant | steps | made for |
|---|---|---|---|---|
| `hello` | 12 tokens, chat | real (bf16 + rounding) | 17 | the first comparison; errors of 2-10% by construction |
| `hello-fp32` | 12 tokens | fp32 + rounding | 5 | the standard small set |
| `hello-fp32-nq` | 12 tokens | fp32, no rounding | 5 | the arithmetic without the rounding. Not used since step 194: only the CPU reference path could run without rounding |
| `hello-fp32-dspark` | 12 tokens | fp32 + rounding, with the draft | 13 | the draft's guesses and confidences |
| `readme200` | 202 tokens | real | 9 | past the 128-token window (the ring buffer) |
| `readme200-fp32-nq` | 202 tokens | fp32, no rounding | 5 | the same, exact variant (not used since step 194) |
| `readme1100` | 1,102 tokens | real | 5 | compressed attention with many blocks |
| `readme20k-s` | 20,002 tokens | real, with the indexer's scores | 9 | long context: candidate blocks, selection past 16k (65 GB) |

| test | target | compared with | needs | time | verdict |
|---|---|---|---|---|---|
| `bin-hip/ds-test -g hello-fp32 --runtime F8` | every layer of the runtime (one die, experts on the CPU) | the reference's tensors, then the table of the previous build | M, F8, GPU, G | 5 s | by hand |
| the same with `--layers 0-5`, `--layers 3,20` | a runtime of some layers (the test's own mode) | the previous build's table | as above | seconds | by hand |
| `... --runtime F8 --e2e FILE`, then `ref/compare_logits.py` | the runtime end to end | a stored logits file byte for byte (`data/ds/`, not in git); the reference's logits (argmax, KL) | as above, py | 5 s + 2 s | by hand |
| `bin-hip/ds-test -g readme20k-s --runtime F8 --dump-topk FILE` | the runtime per layer at 20,000 tokens, the selection | `results/rt-readme20k.txt`; the dump byte for byte with `data/ds/readme20k-topk.bin` | M, F8, GPU, G | 32 min (step 181) | by hand |
| `... -g readme20k-s --runtime F8 --e2e FILE`, `compare_logits.py` | the runtime end to end at 20,000 tokens | `data/ds/readme20k-rt.f32` byte for byte; the reference: argmax 9 of 9 | as above, py | 9 min (step 181) | by hand |
| `ref/check_topk_ties.py` | whether selection differences at 20,000 tokens are near-ties | the reference's scores in the golden set | G, the dump, py | not timed here | by hand |
| `ds-test --e2e ... --dspark D --dspark-out`, `ref/compare_dspark.py` | the draft's guesses | the reference's `forward_spec` (`hello-fp32-dspark`) | M, F8, D, GPU, G, py | not timed here; last run at step 19, as far as the steps record | by hand |
| `ref/check_weights.py` | the GGUF is the checkpoint; the Engram table | the release on Hugging Face by HTTP ranges | M, py, network | 7 min for `--layers 0 --experts 0` (2.9 GB fetched, step 176); the 9-layer check ~6 GB | auto |

Making a golden set (`ref/ds_ref.py`, then `ref/export_golden.py`): minutes for the short
ones; ~42 min and ~500 GB of RAM for `readme20k-s`. Needed only for a new prompt, a new
checkpoint or a change in the reference.

### 3.3 The product's outputs against stored outputs

Each run of `ds-completion` here is one process: ~140 s before it generates (with `--cache-wait`; ~116 s
until step 151, the cause of the change not known) for 8-10 s of generation. Step 188 found what
the 140 s are: the model loads in ~13 s; registering the 289 GB of expert tensors for the cache's
loads takes ~105 s more at the start with `--cache-wait` (until step 189 also at the exit without it;
a run that ends first now stops it). Without
`--cache-wait` a load is 10-35 s (the placement's upload; step 181's plain sampling runs);
without experts on GPUs ~9 s.

| test | target | compared with | needs | time | verdict |
|---|---|---|---|---|---|
| held-out matrix: `scripts/run_matrix.py results/cleanup/heldout.matrix --keep-out` | the fastest configuration: speculation, the adaptive cache, `--canon`, prompt chunks | `results/cache-test/plain-canon-static/<prompt>.out` with `cmp`: plain greedy decoding on a fixed placement | M, F8, D, GPU | 21 min per repeat (8 loads); twice, as usual: 41 min | by hand (`cmp` per file) |
| a long prompt: 9,000 bytes of `docs/steps.md`, `--canon -f FILE -n 40` | prompt chunks of 512 at 2,731 tokens | the previous build's output and hash | M, F8, GPU | ~2 min per run | by hand |
| `bin/ds-ref -p "The three largest cities in Ukraine are" -n 24 --record` | removed in step 193 (the CPU reference path as a program) | a stored hash (`fnv1a64:3a40e97fd9b90430`, step 173) | M | 13 s | by hand |
| sessions: `scripts/chat.sh` on the questions of `results/chat/` and `results/chat/think/` | chat and thinking over three turns, rewinds, the context kept between turns | the stored replies (`*.out`) | M, F8, D, GPU | 6 sessions, 20 min (step 181) | by hand |
| a session against fresh runs: `ds-completion -f <whole conversation>` | a session's reply k equals a single run on the conversation before it | the session's stored reply | as above | 10 runs, 24 min (step 181) | by hand |
| sampling, plain loop: `results/sampler/plain.matrix` on 3 prompts | one seed gives one output across repeats, placements and the cache | the stored records' `output.hash` | M, F8, GPU | 18 runs, 15 min (step 181) | by hand |
| sampling with speculation: `results/sampler/step151/spec.matrix` on 3 prompts | the speculative loop draws what the plain loop draws; fixed draft lengths | step 151's records | M, F8, D, GPU | 15 runs, 38 min (step 181) | by hand |
| the same on the held-out prompts: `results/sampler/spec-heldout.matrix` | greedy and one seed on all eight | step 151's records | as above | 16 runs, 41 min (step 181) | by hand |

### 3.4 The request protocol and the server

| test | target | compared with | needs | time | verdict |
|---|---|---|---|---|---|
| `python3 tests/requests_test.py` | `--requests` on one engine: replies greedy and sampled, a short `n`, mistakes in values, cancels, a context kept or saved between requests, the engine's account of it | each reply's hash with the stored hash of a single run (`results/record/reference/heldout-output.tsv`, `results/sampler/spec-heldout/`) | M, F8, D, GPU | 12 min, 2 of them the load (step 183); 67 checks | auto |
| the same with `--no-spec` | the plain loop behind requests | as above | M, F8, GPU | ~10 min | auto |
| `... --headers-only` | headers that cannot be followed | the protocol's rule | M, F8, GPU | 66 s (8 engines without experts on GPUs) | auto |
| `tests/test_server.py` | see 3.1 | | | 8 s | auto |
| pi on the server (`docs/pi.md`) | a real client, tools, long conversations | by eye; `results/server/pi*.txt` | everything, pi | by hand, last at step 161 | by hand |

### 3.5 Speed and timing (measurements, not tests)

| what | target | compared with | time |
|---|---|---|---|
| the held-out matrix's second repeat | generation speed, mean of eight prompts (44.6-45.1 t/s) | the previous step's mean; run-to-run 1-3% | part of 3.3's 41 min |
| `--trace FILE` on one prompt, `scripts/trace_critpath.py` | that a trace is still written and read; where a step's time goes | the previous trace (`scripts/trace_compare.py`) | ~3 min |
| the benchmarks (`tools/`) | one kernel or the loader | their own earlier numbers | seconds to minutes each, run when their code changes |

### 3.6 Two kinds of result: a verdict, and where it went wrong

The tests above answer two different questions, and most answer only one of them.

- **A verdict** (a detector): same or not, pass or fail. It is cheap to read and can be very
  sharp (two logits files equal byte for byte), but when it fails it says little: "the
  output differs at byte 412".
- **Where** (a localizer): which layer, which part of it, which token, which step, how far
  off. It is what one reads after a verdict failed, and it costs more to run and to read. By
  itself it gives no verdict: someone has to say what counts as wrong.

| test | a verdict | where |
|---|---|---|
| the unit tests, the kernels' `--bitwise`, the server test, the request test | yes | the name of the check or case that failed |
| `cmp` of an output with stored bytes (held-out, sessions, the long prompt) | yes | the first byte that differs, nothing about why |
| a stored `output.hash` (sampling, the request test, the long prompt) | yes | nothing: not even the first token that differs |
| `ds-test --e2e` logits against stored bytes | yes, the sharpest: the arithmetic, not what was sampled from it | the step, with `compare_logits.py`: argmax, KL, largest difference |
| `ds-test` per layer (the table) | none of its own (by hand: equal to the stored table) | the step, the layer, and the part: attention in / out, FFN in / out, the stream out, the router's weights, the experts chosen, the selection |
| `ds-test --per-token` | none | as above per prompt token |
| `ds-test --determinism` | yes (two engines, bit for bit) | the layer and tensor |
| `ds-test --runtime --placement FILE` | none | as the table, with experts on GPUs: a placement's effect per layer |
| `ref/check_topk_ties.py` | none | whether a selection that differs is a near-tie, and by how much |
| `ref/dspark_check.py` | none | the draft itself, or the hidden states the main model hands it |
| `--trace` with `trace_critpath.py` / `trace_compare.py`, `--profile`, `--spec-log` | none | where a step's time went; which drafts were accepted |
| `scripts/records.py` on two sets of records | a comparison outside the repeats' spread | the prompt and configuration, not the cause |
| the engine's `# context:` lines, the server's `--log` and `--dump` | none | what a request did and why |

Read as a ladder, from the cheapest verdict down to the cause:

| a verdict fails | then | then |
|---|---|---|
| an output's bytes or hash | `--e2e` logits on a golden set, both paths: is the arithmetic off, or the loop around it (speculation, cache, chunks, session)? | arithmetic: the per-layer table, then `--per-token`; the loop: `--spec-log`, the `# context:` lines, the same run without `--spec` / without the cache |
| logits against stored bytes | `compare_logits.py`: at the reference's noise level or a real difference? | the per-layer table on the same set |
| the selection (the table's `topk` column) | `check_topk_ties.py` | `check_topk.py` on the kernel alone |
| the draft's acceptance or speed with `--spec` | `--spec-log`, `compare_dspark.py` | `dspark_check.py` |
| speed | `trace_critpath.py` against the previous trace | `--profile`, the benchmark of the kernel named |

What this shows about the current practice:

- **Localizers are run as gates.** The per-layer tables are produced at every step and compared
  with stored tables by hand, although the byte comparison of the logits is the sharper verdict
  on the same data and the table is needed only when that fails. At 20,000 tokens that is 32
  minutes of localizer for a question the 9-minute verdict answers.
- **Some verdicts have no localizer behind them.** A sampled run's hash that differs says
  nothing more: the record keeps the hash, not the ids, so finding the first token that
  differs means running both programs again with their outputs kept. The same for a session
  whose bytes differ: which turn is visible, which step of it is not.
- **Some localizers have no verdict in front of them.** Nothing checks the draft against the
  reference unless someone runs `compare_dspark.py`; nothing says a trace is "wrong".
- **The ladder is in people's heads** (and now here), not in a tool: after a failed `cmp`
  the next command is chosen by whoever is at the keyboard.

## 4. What each test needs

| needed | what it is | in git | who needs it |
|---|---|---|---|
| the model (M), 474 GiB in the page cache | `~/llms/UD-Q8_K_XL/*.gguf` | no | everything in 3.2 to 3.5 |
| `dense-fp8.gguf` (F8) | made once by `bin/fp8pack` | no | every run of the runtime |
| the draft (D) | `~/llms/dspark/*.gguf` | no | speculation: held-out, sessions, sampling with `--spec`, the request test |
| the GPUs, ROCm 6.3.4, the clock set (sclk 6, by hand with sudo after a boot) | | | everything on `bin-hip/`; the clock matters for speed only |
| golden sets (G) | `data/golden/`, 73 GB; `data/ref/*.pt` behind them | **no** (`data/` is ignored) | `ds-test` |
| stored logits and dumps | `data/ds/*.f32`, `readme20k-topk.bin` | **no** | the byte-for-byte end-to-end checks |
| `tests/data/fq/` | inputs and expected outputs of the rounding test | yes (since step 186; `data/fq/` before) | `test-fakequant` |
| stored tables | `results/test-*.txt`, `results/rt-readme20k.txt` | yes | `ds-test` gates |
| stored outputs, hashes and records | `results/cache-test/plain-canon-static/`, `results/record/reference/heldout-output.tsv`, `results/sampler/`, `results/chat/` | yes | 3.3, 3.4 |
| `~/venvs/ds-ref` (torch, numpy) | the reference's environment | no (`ref/requirements.txt`) | making golden sets, `compare_logits.py`, `check_topk.py`, `check_weights.py`, the draft's comparison |
| the official code and `encoding.py` | `~/llms/ds-flash/` | no | making golden sets; `tests/test_server.py` and the server (`encoding.py`) |
| the network | Hugging Face | | `ref/check_weights.py` only |
| the previous build's programs | built in a second checkout or copied before a change | | every "against the previous build" comparison |

Between tests: `compare_logits.py` needs `ds-test --e2e`'s file; `check_topk_ties.py` needs
`ds-test --dump-topk`'s; the sampling matrix with speculation is compared with records that
came from the plain matrix's settings (step 149 before 150); the request test needs the
held-out outputs and the sampling records to exist in `results/`.

What a test does **not** need is as useful: nothing in 3.1 needs the model; `ds-test` on the
short sets needs neither the draft nor a placement and starts in seconds, because it maps the
model and uploads only the dense weights.

## 5. What was run when (the practice of steps 174 to 183)

There is no fixed suite. Each step listed its gates in its results file. The sets that
recurred:

| name used here | contents | time | used at |
|---|---|---|---|
| the usual gate of a runtime change | the three `ds-test --runtime` tables and `--e2e` on `hello-fp32` against the previous build; `ds-ref`'s hash; the long prompt; the held-out matrix twice; a trace; the request test; the server test | 55-60 min, 41 of them the held-out matrix | steps 178, 179 |
| the short gate | both builds; the held-out matrix once | 25 min | step 180 |
| the rest | sessions, session against fresh runs, three sampling matrices, the CPU tables, the 20,000-token set per layer and end to end | 3 h 04 min, about 2 h of it 52 loads | step 181 (and step 174 for the golden sets) |
| no run | text only: the disassembly equals the previous build's | 1 min | step 182 |
| a protocol change | the header cases; the request test; the server test | 14 min | step 183 |

A full pass of everything in section 3 except making golden sets, the draft's comparison,
`check_weights.py` and pi: about 4 hours.

## 6. What is known to be thin, double or by hand

Stated as facts about the current state; what to do about them is the plan's subject.

- **By hand.** Every comparison marked "by hand" above was done with `cmp`, `diff` or a script
  written for that step. One such script gave a false alarm (step 181).
- **No single command** runs any of the sets of section 5, and none says pass or fail overall.
- **Loads dominate.** Of the 4 hours, generation is a few minutes; a run of the product is
  ~140 s of load.
- **The same runs under several names.** The eight greedy held-out runs with speculation and
  the cache appear in `results/cleanup/heldout.matrix` and as the `greedy` configuration of
  `results/sampler/spec-heldout.matrix`, and three of them again in
  `results/sampler/step151/spec.matrix`.
- **Per layer and end to end at 20,000 tokens** take 32 and 9 minutes on the same data.
- **Not in git:** the whole golden sets and the stored logits files. A clean checkout cannot
  make `ds-test`'s per-layer tables until they are made again with the reference. (Since step 187 the
  end-to-end runs need only the cut sets in `tests/data/e2e/`, and hashes take the place of stored logits files.) (The rounding test's data: in git since step 186.)
- **Not covered by any test today:** `-i` beyond the six stored sessions; `--think-show`
  (checked once, step 147); `--profile`, `--log-routing` and `--spec-log` outputs; the
  metrics and logs of the real server (only against the fake engine); a context above 20,000
  tokens against the reference; the fast (non-`--canon`) mode's outputs across placements,
  which are not expected to be identical; the draft against the reference since step 19.
- **Seen, not explained:** the load is 139 s where step 151 recorded 116 s. (`test-sampler`'s
  80 s: explained and cut to 19 s in step 186.)

## 7. The runs (from step 186)

Decided after this page was written (plan.md, "The testing process"): no choosing of tests per
change. A few fixed runs, each one command with one verdict and an exit code:

| run | command | when | contents | time |
|---|---|---|---|---|
| unit | `scripts/check.py unit` | after every change | everything of 3.1: no model files | 30 s (47 s with a build) |
| default | `scripts/check.py default` | after every change | `unit`; the logits of the runtime at 12, 202 and 1,102 tokens, each file's sha256 against `tests/data/e2e/expected.txt` (step 187); the product's outputs: one engine with the fastest settings takes the held-out prompts, two sampled, a second turn, and one fresh session, every reply's token ids against `tests/data/outputs/expected.json` (step 188) | ~5.5 min |
| speed | `scripts/check.py speed` | a change about speed | the eight held-out prompts, each in a fresh process with the fastest settings, twice; the second repeat's mean against `tests/data/speed/baseline.tsv`: slower by more than 2% fails, within 2% is "the same", faster says so. Also: every output the stored bytes, the clock level the baseline's, the second repeat not reading from storage (step 190) | 36 min |
| complete | `scripts/check.py complete` | overnight, and before anything is called done | `default`; the logits at 20,000 tokens; the request test with and without speculation; six sessions and ten single runs on their conversations; the three sampling matrices; a 2,731-token prompt from a file; `speed` (step 190) | ~4 h estimated; the first whole run 6 h 43 min, in a boot where the registration took 285 s per engine start (step 198) |
| localizers | by hand | when a verdict failed | section 3.6's "where" column | |

Before the checks of a run with the model, `check.py` prepares the page cache
(`scripts/model_cache.py`, step 204): what is not cached is read sequentially, a probe of 6 s
(`bin-hip/reg-probe`) says how long the expert tensors would take to register, and a shard that
registers below 2 GB/s is dropped and read again when the run's starts (1, 16, 69) would wait
longer for it than the reading takes. This keeps a run out of step 198's state, where every start
waited 285 s. It is a preparation without a verdict. Times below are from boots where a start
waited 105-140 s; right after a boot with the model read once, a start waits ~40-55 s and
`default` took 259 s.

`check.py` builds first (`make`, `make HIP=1`: a stale program is a wrong verdict), prints a
line when a check starts and, for the checks that take minutes, their own lines as they come
(indented), then one line per check with its verdict and time, keeps every check's output under `data/check/`, and for a failed check prints
where its output is and what to run next. A check whose needed file is missing fails; it is
not skipped.

`default`'s logits checks read cut golden sets that are in git (`tests/data/e2e/<name>/`: per step
the token ids and the reference's logits, made by `ref/trim_golden.py`), so they need only the
model and `dense-fp8.gguf`. `scripts/check.py default --store` replaces the expected hashes: for
a change meant to move the logits, after the difference was compared with the reference.

**Not in any run** (by hand, when their subject changes): what gives no verdict, i.e. the
localizers of 3.6 (the per-layer tables, `--per-token`, `check_topk_ties.py`, traces) and the
draft against the reference (`compare_dspark.py`: numbers to read); `ref/check_weights.py`,
which reads the checkpoint from Hugging Face (when the model files change); pi on the real
server.

`--store` on `default`, `speed` or `complete` makes what the checks gave the new expectation:
the logits' hashes, the outputs' token ids (never a greedy reply that is not the reference's),
the speed baseline. For a change that is meant to move them.

What `unit` needs outside the repository: ROCm and the GPUs (two checks run kernels),
`~/venvs/ds-ref` (torch, for the reference's own top-k) and the model's `encoding.py` in
`~/llms/ds-flash/encoding/` (the chat format the server test renders with). No weights.

