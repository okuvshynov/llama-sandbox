# `--record`: a run as one record

**Status (2026-11-16, README steps 115-122, 126-127):** the idea, and a first minimal `--record` in
`ds-completion`. No test uses it yet and no existing script changed. The record is meant to grow
field by field, each addition a small change; the testing procedure follows once the record has
shown what it can see.

## Why

The project has about 60 scripts and programs that verify, measure or simulate something. Each
was written for one step, and each gate script repeats the same loop: run a few prompts under a
few configurations, read numbers off the `# ...` lines on stderr with `grep` and `sed`, compare
outputs with `cmp`, print a table. That served 114 steps. For what comes next it is more than a
reader can hold, and three things in it are fragile:

- the numbers a script reports are scraped from text meant for people;
- what a run actually did (which options took effect, whether the page cache was warm) is not
  written down anywhere a later check can read;
- every script decides for itself what "the same" and "failed" mean.

## The idea

One primitive: **a run**. It has inputs and outputs, and `ds-completion --record FILE` writes
them down as one JSON object. A test is then a run (or several) plus a check of the records. The
engine never judges its own record.

Inputs, in three groups:

| group | examples | may it change the logits? |
|---|---|---|
| identity | the checkpoint, the prompt's token ids, the tokens to generate, the numerics (`--canon`) | yes, by definition |
| free | placement, cache settings, speculation and its draft, prompt chunk, the layer split, threads, the GPU clock, the machine | no: this is the contract `--canon` makes |
| provenance | commit, build flags, ROCm and compiler versions, the `DS_*` switches | it should not; a toolchain change can |

Outputs, in three groups:

| group | examples |
|---|---|
| values (the same on every run) | a hash of the emitted token ids, later of the logits |
| counters (vary run to run) | load time, prompt and generation t/s, tokens per verify step, the share of experts on GPUs |
| status | the exit code, the configuration as it took effect, whether the run was cold |

Checks then read as statements over records:

- **identity:** records with the same identity inputs have the same values, whatever their free
  and provenance inputs;
- **speed:** two records that differ in one input, their counters compared per prompt against
  the spread of repeated runs;
- **fidelity:** a record against the reference's, with a tolerance (this one needs the logits,
  not a hash);
- **a golden** is a stored record.

## Scope

Decided in the discussion this document comes from:

- The engine only. Sampling, serving and chat sit a layer above it and are not covered here.
- Hardware dependence is accepted: the tests need the model and the GPUs, as the code does.
- Forward, not backward: the old steps stay reproducible through docs/reproduce.md and git; new
  testing covers what changes next.
- The changes expected next, and what each may break:

| change | values | counters |
|---|---|---|
| small performance work | must not change | the point |
| refactoring, cleanup | must not change | must not get worse |
| another speculation (draft) module | must not change: the main model verifies | the point: tokens per step, draft time |
| parameter tuning (cache, speculation) | must not change | the point |
| toolchain (ROCm, compiler, kernel) | may change: check against the reference, then store new goldens | a new baseline |
| a new checkpoint | new goldens, through the whole reference chain | a new baseline |

- Two identical machines are available: one can run a known build while the other runs a
  candidate, and both must produce the same values.
- The model in two halves (README step 111) is one more free input. Not covered by the first
  version: `--record` refuses `--serve`.

## What the first version writes

`--record FILE` with a single prompt (not with `-i`, `--serve` or `--teacher`). A whole record
as the program wrote it, the fastest single-machine settings (`results/record/example.json`):

```json
{
  "schema": 1,
  "args": {"mode": "chat", "effort": 75, "n": 300, "threads": 16, "runtime": true, "canon": true, "spec": true,
           "placement": "placements/study30.txt", "cache_loads": 800, "cache_period": 8},
  "prompt": {"tokens": 25, "hash": "fnv1a64:32138dd9ac19a294"},
  "env": {"gpu_sclk": [6, 6, 6, 6]},
  "read_bytes": {"load": 0, "prompt": 0, "gen": 0},
  "spec": {"steps": 66, "plain_steps": 0, "accepted": 233, "verified": [0, 3, 7, 12, 10, 34], "draft_ms": 514.0, "verify_ms": 4921.8},
  "experts": {"gpu_uses": 72804, "cpu_uses": 12156, "dies": [0, 1, 2, 3], "placed": [801, 1765, 1765, 1765],
               "cache": {"rebalances": 32, "swaps_planned": 3857, "loads_started": 3686, "loads_published": 3621}},
  "counters": {"load_s": 116.699, "prompt_s": 0.6114, "prompt_tps": 40.889, "gen_tokens": 300, "gen_s": 5.4982, "gen_tps": 54.564}
}
```

- `args`: fourteen of the options, not all of them yet, as they took effect: an option not
  given has its default (never a missing key), and `n` is the limit after `--max-ctx` capped it.
  `spec_n` 0 (the default): the number of drafts verified is chosen per step. Since step 120
  with the knobs a sweep varies: `cache_window`, `cache_min`, `cache_wait`, `spec_n`. Since step
  121 with the options that add work to a run, on or off: `profile`, `trace`, `spec_log`,
  `log_routing`. Since step 149 with the sampler's settings: `temp` (0: greedy), `top_p`,
  `seed`. The example above is from before all three.
- `env.ds` (step 121): every environment variable named `DS_*` with its value, `{}` if none:
  the experiment switches. A record with one set, or with `profile` on, is not a clean timing
  (`--profile` -8%, `DS_FP8_PAD=0` -7.5% on the example's run).
- `label` (step 127; not in the example above): `--record-label key=value[,key=value...]`, the
  option repeatable: the names a run goes by for whoever started it (configuration, prompt
  name, repeat), as strings in the order given; `{}` without the option. The program writes
  them back and never reads them. A key twice, a part without `=`, or a label without
  `--record` is refused before the load.
- `prompt.hash`: FNV-1a, 64 bits, over the prompt's token ids after the chat template, each id
  as four bytes, least significant first. Two runs with the same hash ran the same tokens,
  whatever text or options produced them. A prompt's name belongs to whoever starts the run; the
  engine does not know it.
- `env.gpu_sclk` (step 116): the GPU clock setting, one value per AMD card: the level (0-8)
  where it is fixed, `"auto"` (or another mode's name) where the driver moves it with the load,
  `"unknown"` if it could not be read. As it was when the run started; a setting changed by
  hand during a run is not seen. After a reboot every card is on auto.
- `read_bytes` (step 117): what the process had fetched from storage in each phase
  (`/proc/self/io`): this process only, all its threads, only reads that reached the disk. 0: the
  phase did not wait for storage. A prompt not run since a reboot read 72 MB during its
  generation (its engram rows came from the disk) and ran 12% slower than its repeat.
  `null`: not available.
- `spec` (step 118; `null` without `--spec`): the speculative loop's totals. `steps`: verify
  steps; `plain_steps`: those without a draft; `accepted`: drafts the main model agreed with;
  `verified`: the steps by the number of drafts verified, from 0; `draft_ms`, `verify_ms`: the
  time in the draft and in the verify steps. Totals only: tokens per step is `gen_tokens / steps`.
- `experts` (step 119; `null` on the CPU path): `gpu_uses` and `cpu_uses`, routed-expert uses
  by where they ran over the whole run (rejected drafts included); `dies` and `placed`, the dies
  that hold experts and how many each held after startup; `cache`, the adaptive cache's totals
  (`null` without `--cache-loads`). Since step 122 with `loads_started_by_die`, in the order of
  `dies`: the home die takes no loads (README step 86), so its place holds 0. With that rule
  switched off on purpose it held 13% of the loads and the run was 8% slower.
- `output` (step 126; not in the example above), the first of the values:
  `{"tokens": 300, "hash": "fnv1a64:..."}`, the prompt's hash over the ids of the tokens the run
  printed, in order, without the end-of-sequence id that ends a run. Equal hashes: the same
  tokens were chosen; the logits may still differ.
- `counters`: the numbers of the `# loaded in ...` and `# prompt: ... generation: ...` lines.

The file is opened before the model loads and written after the generation. A run that fails or
is killed in between leaves it **empty**, so an empty record means "did not finish" and a missing
directory is reported in the first second, not after the load.

## The first two records

The same 20-token prompt twice in a row, 32 tokens generated, every expert on the CPU
(`results/record/first.txt`; the cards on auto). The page cache happened to be empty before the first run:

| | run 1 (cold) | run 2 (warm) |
|---|---|---|
| `prompt.hash` | `ca691ad07d0dd8ee` | `ca691ad07d0dd8ee` |
| `load_s` | 92.7 | 4.0 |
| `prompt_tps` | 18.98 | 20.46 |
| `gen_tps` | 13.72 | 14.52 |

The record's numbers equal the stderr lines', and the two outputs are the same bytes. The cold
run shows in `load_s`, and its generation was 5.5% slower too, after the load had finished. That
is one pair, not a measurement; but nothing in this record says "cold" except a load time someone
has to know is unusual. Item 3 below is for that.

A third record, of the fastest single-machine settings (`results/record/fast.txt`: cpp-matrix,
300 tokens, placement, draft, `--canon`, the cache at 800 loads/s): 47.7 t/s, the output the
stored one's bytes. Step 114 measured 54.86 t/s on this prompt at clock level 6. Here the cards
were on auto, which this record did not hold (step 116 added it), and the same command an hour
later gave 55.05 t/s (`results/record/sclk.txt`): a 15% spread between two runs that no field
explained then. Step 117's field gives a likely cause, after the fact: the first run of a text
after a reboot reads its engram rows from the disk. On auto each card was seen at levels 1, 5 and 8 within one run. Neither does the
record hold what stderr
printed next to it: 68 verify steps, 4.41 tokens per step, 85.7% of expert uses on GPUs, 3,663
cache loads. And tokens per step was 4.55 and 4.62 in two earlier runs of this prompt: the draft
length follows measured times, so the steps of two runs differ while their outputs do not.

## Records in git

A record is committed when something in the repository points at it; otherwise it stays out.

- **Evidence:** the records a README step quotes. In `results/record/`, a new file per step,
  never overwritten. Until a record carries its own commit and date, it sits in a text file
  under the usual `#` header, or as a `.json` file whose header is in `results/record/README.txt`.
- **References:** stored values that checks compare against. Values only, no counters, since
  counters differ on every run; changed only by a commit that says why. The first (README step
  138): `results/record/reference/heldout-output.tsv`, the held-out prompts' output hashes.
- **One-off runs** (comparisons, sweeps, repeated timings): not in git; `data/` is ignored
  already. One that ends up deciding something moves to `results/` in that commit.

## Next, each a small change

In the order I would take them; the order is not decided.

1. The rest of the configuration as it took effect: the `DS_*` switches and the options that
   perturb timing are done (step 121), the tuning knobs too (step 120). Left: the draft's file,
   the remaining options, the commit the binary was built from, the host, the input files'
   sizes and dates. (The GPU clock setting: done,
   step 116.)
2. Proof that a requested feature ran: done with steps 118-119 (`spec.steps`, `experts.placed`,
   `experts.cache.loads_published`). A check must still compare them with the arguments. A feature that silently did not apply looks like "no
   difference".
3. Cold runs: done (step 117), as bytes read from storage per phase; major page faults do not
   measure it (one fault for 64 MiB). Still open on the machine's side: how much of each model
   file is cached when a run starts, which says whether the machine is ready, not what a run read.
4. The counters the engine already prints: done (speculation in step 118, experts and the cache
   in step 119).
5. Values: a hash of the emitted token ids: done (step 126, `output`). Then the drafted ids
   (see the case study below), then the logits. On the speculative path the
   logits stay on the GPU (README step 88), so a checksum would be computed there, next to the
   argmax, and its cost measured with the record itself. Its definition (which positions, which
   bits, in what order) must be fixed before any golden is stored.
6. Which paths a run took: the window wrapped, the largest verify step, cache evictions, experts
   used per die. A check that never reaches a path cannot fail on it.

Then, before any test script: **break things on purpose and see whether the records show it.**
(Begun in step 121: `DS_FP8_PAD=0` and `--profile` each show as ~8% and are named in the record.
Step 122: a one-line change that lets the cache load into the home die again shows as loads on
that die, -8.1%, and the time in `draft_ms` and `verify_ms`.)
Slowdowns with a known size need no code change (`DS_FP8_PAD=0`, README step 101;
`DS_STALL_SPLIT=0`, step 102; `-t 32`, step 18; a lower clock level; part of the model evicted
from the page cache). Bugs should hide on rare paths: a reordered sum in one kernel (logits
change, tokens do not), a fault only in 6-token verify steps, only for one die's experts, only
after a cache eviction, in `commit`'s rollback, only past the 128-token window. Every one that
escapes names a missing field or a missing prompt.

## A case study: steps 123-125 (the draft's experts in the pool, fewer of them, left out)

Three steps done with the record as it was after step 122, 116 runs. What it gave, and what
had to be done around it.

**It gave:** cold runs (`read_bytes` named the first sweep's and one odd run of step 125);
proof that the treatment applied (`experts.placed` 802 -> 944); step 124's whole result
(`spec.steps`, `spec.accepted`); the switch in every record (`env.ds`).

**Done outside it:**

| needed | how it was done | missing |
|---|---|---|
| the output is the same (104 runs) | `cmp` with stored files | a hash of the printed ids: step 126 |
| the draft is the same | `DS_DRAFT_TOKEN_LOG`, `awk` to drop the times, `cmp` | a hash of the drafted ids |
| left out equals masked | `cmp` of two routing logs | the same, or a hash of the draft's routing |
| did the switch apply? | `grep` for a sentence on stderr | the draft's state: its file, where its experts are, how many are loaded |
| which prompt, configuration, repeat | file names | a label given by whoever starts the run |
| which of three binaries | by hand | the commit or the binary's hash |
| which keep set | `env.ds` has its path only | a hash of each input file |
| memory moved from the draft to the slots | stderr's startup table | the memory per die |

**On the checking side:** no tool reads records. Two scripts wrote a text summary from them
and two more parsed it with regular expressions, which is the scraping the record was to end.
The spread of repeated runs was quoted from memory (~1%), not computed, and most of step 125's
per-prompt differences are inside it. A table printed verify time per step where the steps
verified different numbers of drafts.

**One claim it could not support:** that the home die's new slots hold experts a prompt seldom
needs. The record has expert uses on GPUs and on the CPU, not per die.

In the order taken: the output hash (done, step 126); a label (done, step 127); a tool that
reads records (done, step 128: `scripts/records.py`, groups, means, the spread of repeats, a
comparison that prints only differences outside the spread, a condition) and one that runs a
matrix of them (done, step 129: `scripts/run_matrix.py`) (README step 127 has the plan for the scripts); the drafted ids' hash; the draft's state
and uses per die; the binary's and the input files' identity; memory per die and verify time
by drafts verified.
