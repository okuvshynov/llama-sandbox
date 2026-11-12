# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM): on one
machine, or split over two of them for more speed. Both setups print the same bytes.

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts: the most-used ones on the four GPUs, the rest on the CPU (our own exact
  AVX-512 kernel); with the adaptive cache the GPUs follow the experts in use.
- The prompt goes through in chunks of up to 512 tokens; generation one token at a time, or a
  few at once with speculative decoding. Greedy, one sequence, text only.
- With two machines the model is cut in the middle: layers 0-19 on the first (the **front**),
  layers 20-39 with the head and the draft on the second (the **back**), each with its own four
  GPUs for its half's experts.

Words used below: a **die** is one of the four GPU chips (two per Vega II Duo card). **sclk** is
the GPU clock level, 0-8. A **placement** is a file ranking the routed experts by use; the top
ones that fit are put on the GPUs at startup (`placements/study30.txt`, from a 30-prompt study).
The **held-out prompts** are the 8 prompts `scripts/spec-measure.sh` runs, none of them used for
the placement. `--canon` makes the output independent of where an expert runs.

## What it needs

- **System:** Ubuntu 22.04 with the T2 kernel (`6.10.5-1-t2-jammy` here), booted with
  `pcie_ports=native intel_iommu=on iommu=pt pm_async=off pci=realloc pci-stub.ids=106b:2005`;
  ROCm 6.3.4 in `/opt/rocm-6.3.4`; the user in the `video` and `render` groups. (The cards' XGMI
  bridges are not needed: the runtime never copies between dies directly.)
- **The GPU clock level** is set outside the program, as root, and again after a reboot; it
  changes the speed only, never the output. The numbers below are at sclk 6 unless marked:
  ```bash
  for c in /sys/class/drm/card[0-3]/device; do
    echo manual | sudo tee $c/power_dpm_force_performance_level
    echo 6 | sudo tee $c/pp_dpm_sclk          # level 0..8
  done
  cat /sys/class/drm/card0/device/pp_dpm_sclk   # the line with * is the level in use
  ```
- **Model files** (in `~/llms/`; 488 GB, run from the page cache, so the RAM must hold them):
  - `UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf` (474 GB; Unsloth's
    [DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF));
  - `dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf` (8.5 GB): the DSpark draft for `--spec`;
  - `ds-1017-fp8/dense-fp8.gguf` (6.8 GB): made from the model by `bin/fp8pack` (below).

  Copies of files this size have come out silently corrupted on this hardware before, so check
  them: `cd ~/llms && grep -v '^#' ~/projects/llama-sandbox/checksums/DS-V4.1-Flash-1017.sha256 | sha256sum -c`
  (paths relative to the model directory; `dense-fp8.gguf` after making it). After a copy, check
  again after a reboot: a check right after it may only read the page cache.
- **llama.cpp at the DeepSeek-V4.1 PR commit**, not upstream: upstream reads these GGUF files
  silently wrong and generates garbage (their Engram table's type 43 is the PR's MXFP8, upstream's
  plain fp8; [docs/llama-cpp-upstream.md](docs/llama-cpp-upstream.md)). Used for ggml, GGUF
  reading and the tokenizer, as static libraries in `~/forks/llama.cpp/build/`.
- **For two machines:** a second Mac Pro set up the same way (the same system, the same three
  model files at the same `~/llms/` paths with the same checksums, this repository at the same
  commit, built; run the checksum line on both), the same GPU clock level on both, and a network
  between them (10 GbE here; a step moves 64 KB per token). See "Two machines" below.

## Build and prepare (once)

```bash
# llama.cpp, pinned by hash (the head of PR #30817, the code these GGUF files were made with)
git clone https://github.com/ggml-org/llama.cpp ~/forks/llama.cpp
cd ~/forks/llama.cpp
git fetch origin b72050d03ae14bb3cb5138ecefab3f3f65157d31
git checkout -b deepseek41 b72050d03ae14bb3cb5138ecefab3f3f65157d31
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32

cd ~/projects/llama-sandbox/ds-completion
make && make HIP=1                     # -> bin/ (CPU) and bin-hip/ (GPU); FORK=<dir> for another llama.cpp
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf   # lossless fp8 repack, 6.8 GB, ~20 s
```

`dense-fp8.gguf` comes out byte for byte the same on every build (its checksum is in the manifest),
so it doubles as a check of the build and the model files.

## Run

Run from this directory (the placement path is relative). Which setting is fastest depends on
the job (sclk 6; README steps 32, 37-102):

**Long generation** (reasoning, long answers): speculative decoding + the adaptive expert cache.
~44.6 t/s mean on the held-out prompts at sclk 6 (34-55; README step 104); a long reasoning run
gains most (57.0 t/s on the held-out long proof). The cache first registers the experts' memory with the GPUs (~95-135 s in the
background; generation runs at the static placement's speed meanwhile), so it pays off for runs
of a few minutes or more:

```bash
cd ~/projects/llama-sandbox/ds-completion
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt \
    --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --spec \
    --canon --cache-loads 800 --cache-period 8 \
    -f prompt.txt --think --effort max -n 60000
```

**Short runs and long prompts with short answers:** without the cache (it would not start
loading before the run ends) and without `--canon` (the prompt runs ~48% faster: 93 t/s vs
63 at sclk 6, README step 68). Speculative decoding
gives the same bytes as plain decoding, +22% on generation on average (more on code and math,
least on Chinese/Japanese):

```bash
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt \
    --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --spec \
    -p "Write a haiku about memory bandwidth." --chat -n 64
```

Without `--dspark ... --spec`: 17.7-21.7 t/s generation (at sclk 2), and 8.5 GB more GPU 0 memory for
experts. Without `--placement` every routed expert runs on the CPU (faster start, ~12 t/s).
A higher GPU clock helps both: held-out with the cache 36.8 / 41.1 / 42.1 t/s at sclk 2 / 6 / 8
when last swept (README step 90; sclk 6 is 44.6 since); the clock is set outside the program (sudo).

After a memory-heavy job (anything that pushed the 474 GB model out of the page cache), the
first run reads it back from disk; `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` first (2-4 min)
keeps that out of a timed run.

Options:

- `-m FILE`: the model's first GGUF shard. `--rt FILE`: the GPU runtime, with the fp8 file.
- `-p "text"` or `-f prompt.txt`: the prompt, inline or from a file (as is).
- `--chat`: one user turn, no thinking.
- `-i` / `--interactive` (with `--chat` or `--think`, and `--rt`): an interactive session, one user line per turn on
  one context; `-p` is then optional (the first turn), `-n` is per reply (reasoning included). An
  empty line ends it. With `--think`, earlier turns' reasoning is dropped from the context, as the
  reference does. `scripts/chat.sh [--think [--effort max]]` runs it with the fast configuration
  and `-n 30000` (README steps 108, 109).
- `--think`: with thinking. `--effort N`: its reasoning effort, 1-100 (default 75), or `low` /
  `high` / `max` (50 / 75 / 100), as the official encoding; the model card's benchmarks use 100.
- neither: the prompt is raw text after BOS.
- `-n`: max new tokens.
- `--max-ctx N`: the context's size (default 65,536 with `--rt`; "Limits" below).
- `--layers a-b` (with `--rt`): only that range of layers ("Two machines" below). A range must
  hold the layers its own layers read from, which leaves layer 20 as the only cut (checked;
  README step 110).
- `--serve PORT`: the back: a range up to the last layer; waits for one front, then exits.
- `--remote HOST:PORT`: the front: a range from layer 0; connects to the back.
- `--teacher` with `--stream-out FILE` / `--stream-in FILE`: a test of a range against the whole
  model on one machine (the values between two layers written to and read from a file).
- `--prefill-chunk N`: prompt tokens per forward (default and most 512, the fastest for
  long prompts in the fast numerics). The output does not depend on it.
- `--no-overlap`: prompt chunks without the CPU/GPU overlap (slower; for comparisons).
- `-t`: CPU threads; the default 16 (one per physical core) is best with `--rt`. 32 leaves no
  CPU for the host thread and the HIP runtime and is much slower (README step 18).
- `--profile`: a per-step time breakdown, the CPU's misses per layer and what they cost (README
  step 50).
- `--trace FILE`: a Chrome trace of every step (open in ui.perfetto.dev; README step 51). On two
  machines each half writes its own; `scripts/trace_merge.py FRONT BACK -o MERGED` makes one
  trace on the front's clock (`TRACE=1 scripts/split-measure.sh` does all of it; README step 112).
- `--gpu D`: the die for the runtime (default 0).
- `--placement FILE`: routed experts on GPUs, the file's top-ranked ones that fit (10-30 s upload).
- `--moe-dies 0,1,2,3`: the dies that hold experts (default all four).
- `--dspark FILE`: the DSpark draft GGUF (8.5 GB on GPU 0). Alone: shadow mode, drafts are
  measured but not used.
- `--spec`: speculative decoding (needs `--dspark`, or `--remote` to a back that has it); the number of guesses verified per step
  is chosen from the draft's confidence. `--spec-n N` fixes it; `--spec-log FILE` writes
  per-step data.
- `--canon` ([NUMERICS.md](NUMERICS.md)): placement-invariant expert numerics: the output no longer depends on where the
  experts run (any placement, any `--moe-dies`, with or without the draft, with or without the
  cache give the same bytes); generation ~2% slower, the prompt ~32% (93 -> 63 t/s; its CPU expert
  kernel is slower).
- `--cache-loads R`: the adaptive expert cache (README step 32): the GPUs keep the experts used
  most in the last 256 tokens, loading up to R experts per second in the background (800 with
  `--cache-period 8` is the fastest measured, README step 50; 0 = the static placement). Use with `--canon`, or the output depends on load
  timing. It starts loading once the model's expert memory is registered with the GPUs (~95-135
  s in the background, longer if the model must come back from disk; README step 43);
  `--cache-wait` waits for that before the prompt. `--cache-period N` (16) and
  `--cache-window N` (256) set how often it rebalances and over how many tokens it counts;
  `--cache-min N` (2) the uses a newcomer needs (1 was ~0.7% slower, README step 59).
- `--load-test R [--load-pinned]`: a measurement aid: loads random experts into spare GPU slots
  at R loads/s in the background, as the cache would (README step 30); output unchanged.

- `--log-routing FILE`: the router's choices per token and layer (the routing studies).
- `--no-quant` (CPU path only): without the reference's fp8/fp4 rounding (a comparison).

Without `--rt` (or with `bin/ds-completion`) everything runs on the CPU.

## Two machines

The model in two halves (README steps 110-112): **~50.9 t/s** mean on the held-out prompts
(35-63) against 44.6 on one machine, the same bytes. The second machine is set up as the first
("What it needs", "Build and prepare"). Start the back there first:

```bash
# on the second machine: the back (layers 20-39, the head, the draft)
cd ~/projects/llama-sandbox/ds-completion
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt --canon --cache-loads 800 --cache-period 8 \
    --layers 20-39 --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --serve 7001
```

It loads (~1 min), prints `waiting for the front`, serves one front and exits when that front
is done. Then, on the first machine:

```bash
# the front (layers 0-19): the one-machine command without --dspark, plus --layers and --remote
cd ~/projects/llama-sandbox/ds-completion
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt --canon --cache-loads 800 --cache-period 8 \
    --layers 0-19 --remote <second machine>:7001 --spec \
    -f prompt.txt --think --effort max -n 60000        # prompt.txt: your own file
```

- **The front is used as the whole model is:** `-p` / `-f`, `--chat` / `--think`, `--spec`. An
  interactive session: the front's command with `--chat -i` (or `--think -i`) in place of `-f ...`
  (`scripts/chat.sh` runs one machine only).
- **What must agree:** the layer cut (checked when the front connects: nothing else is), the
  model files and the build (not checked: the checksum line and the same commit on both), and
  `--canon` on both halves. The placement and cache options may differ; with `--canon` they do
  not change the output.
- **What to expect:** the gain is the routed experts that ran on the CPU: each machine holds
  ~88% of its half's experts on GPUs instead of ~42%. Per held-out prompt: +12% to +27% on code,
  math, SQL and JSON, +8% on chat, +2.5% on Chinese and Japanese (1-2 tokens per step), and
  nothing on a long reasoning run whose cache had already adapted (57.0 t/s either way). The
  cache's registration takes ~30 s per half (half the experts each).
- **The network:** the back listens on every interface at the port given (7001 is only this
  page's choice; open it in a firewall between the two); plain TCP, no authentication, no
  encryption: a private network only.
- **If something goes wrong:** a front that finds no back ends with `peer: cannot connect to
  ...`; if one side dies the other ends with `peer: the connection closed ...`; a back still
  waiting for a front stops with Ctrl-C. In every case start the back again before the next run.
- `scripts/split-measure.sh` runs the held-out set this way (it starts the back over ssh, so it
  needs `ssh` to the second machine without a password prompt); `TRACE=1` gives one merged trace
  of both machines (README step 112).

## Current results (2026-11-12)

The two top rows were measured with `--cache-wait` (the run starts after the cache's
registration); without it a short run starts at the static placement's speed.

| mode | command | decode | KL vs reference*: median / mean without the near-tie step |
|---|---|---|---|
| **two machines, the model in two halves** | the two commands of "Two machines" above | **35.0-62.9 t/s, mean 50.9** at sclk 6 (300 tokens, held-out prompts; README step 111) | output byte-identical to every row down to "own runtime" with `--canon` |
| **+ adaptive expert cache** | `... --placement placements/study30.txt --dspark ... --spec --canon --cache-loads 800 --cache-period 8` | **33.9-55.0 t/s, mean 44.6** at sclk 6 (300 tokens, held-out prompts; README steps 50, 53, 57, 62, 68, 71, 73, 74, 80, 82, 84, 86, 88, 97, 98, 101, 102, 104; `--cache-loads 200`: 25.6 at sclk 2, 28.0 at sclk 6) | output byte-identical to the plain `--canon` run |
| + speculative decoding | `... --placement placements/study30.txt --dspark ... --spec` | 16-24 t/s (sclk 2; 300 tokens, held-out prompts; +22% geometric mean) | output byte-identical to the row below |
| own runtime + experts placed on GPUs | `bin-hip/ds-completion --rt ... --placement placements/study30.txt` | 17.7-21.7 t/s (sclk 2; 128 tokens) | same numerics: per-layer gate identical to below |
| own runtime, dense on GPU 0 | `bin-hip/ds-completion --rt ...` | 12.0-12.5 t/s (sclk 2) | 5.9e-6 / 4.8e-3 (earlier run 1.7e-5 / 6.3e-4) |
| CPU only | `bin/ds-completion` | 3.5 t/s | 1.3e-5 / 3.9e-3 |
| llama.cpp fork (llama-cli, CPU) | - | 3.5 t/s | 2.1e-5 / 1.3e-3 |

\* vs the official reference implementation run on the CPU (README "Step 2"): chat prompt
"Hello, how are you?", 17 steps. Every mode agrees with the reference on the top token at 16
of 17 steps. The miss is one near-tie (step 12), which dominates any plain mean, so the table
leaves it out.
Means over all 17 steps: 1.8e-2 to 2.5e-2 for every mode.

- **The runtime is also gated per layer:** attention and FFN output within ~1e-7 of the fp32 +
  fake-quant reference. It has no inexact op left, and repeated runs are bit-identical.
- **Prompt processing** (`--rt`, chunks of 512, the CPU's experts overlapped with the GPU's
  work; README steps 37-45, 68): 93 t/s, 63 with `--canon` (1,989 tokens; 97 / 73 at 6,106; sclk 6),
  output byte-identical to token-by-token processing; ~7.7 t/s on the CPU path. The
  runtime's mean KL varies run to run with fp8 rounding flips (README step 10); every exact mode
  scatters the same way.
- **Where a step goes on two machines** (README step 112's merged trace): a 6-token code step
  with its draft ~83 ms (~93 on one machine, below): the front's layers ~36 ms, the back's ~36
  ms, the draft ~9 ms, the network ~0.5 ms; neither half waits for the CPU's experts.
- **Where a step goes** (`--spec`, cache, sclk 6; the traces of README step 105): a 6-token
  code verify step ~85 ms (medians): die 0's dense part ~48 ms, its wait for the CPU's experts
  ~27 ms, its wait for the other dies ~5 ms, outside the layers ~5 ms; the draft before it ~8 ms
  more. The prompt: the CPU's experts and the GPU's work overlap and the slower
  sets the pace (TIMING.md).
- **GPU memory with `--rt`:** 6.41 GiB fp8 + 1.75 GiB small weights + ~1 GB of prompt-chunk
  buffers on die 0; the rest holds experts (and the 8.5 GB draft with `--dspark`).

## Limits

- `-i` sessions: one line per turn, no system prompt, no tool calls.
- Two machines: the back serves one front and exits; the connection is plain TCP without
  authentication or encryption; only the layer cut is checked between the halves; the two
  machines take turns, so the gain is bounded by what the CPU's experts cost on one.
- Context with `--rt`: 65,536 tokens by default (prompt + generation), up to the model's
  1,048,576 with `--max-ctx N` (~6.4 KB of GPU 0 memory per position). A 30k-token prompt takes
  ~5-8 min at 63-97 t/s (it took ~30 min token by token). Without `--rt` (the CPU path): 4,096 by
  default, at most 16k (no candidate-block selection there).
- The dense part uses one GPU; the others only hold experts with `--placement`.
- At exit the process sometimes aborts (signal 6, exit code 134) inside the GPU runtime's
  shutdown, after `main` has returned: the output and the timing lines are complete by then
  (README step 96; unexplained).
- Sampling is greedy only. Greedy decoding can fall into a repetition loop on long reasoning
  (README step 48: a proof prompt repeats one line after ~800 tokens; the model's behaviour, not
  a bug).

Details: [TIMING.md](TIMING.md) (where the time goes, one page), [OVERVIEW.md](OVERVIEW.md) (a 20-minute summary), [README.md](README.md) (all measurements and checks), [PLAN.md](PLAN.md) (what's next).
