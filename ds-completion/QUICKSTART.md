# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM).

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts: the most-used ones on the four GPUs, the rest on the CPU (our own exact
  AVX-512 kernel); with the adaptive cache the GPUs follow the experts in use.
- The prompt goes through in chunks of up to 512 tokens; generation one token at a time, or a
  few at once with speculative decoding. Greedy, one sequence, text only.

## What it needs

- **System:** Ubuntu 22.04 with the T2 kernel (`6.10.5-1-t2-jammy` here), booted with
  `pcie_ports=native intel_iommu=on iommu=pt pm_async=off pci=realloc pci-stub.ids=106b:2005`;
  ROCm 6.3.4 in `/opt/rocm-6.3.4`; the user in the `video` and `render` groups; the Vega II Duo
  bridges seated so the dies pair over XGMI as 0-1 and 2-3. The GPU clock level is set with sudo
  outside the program (all four cards; the numbers below are at sclk 6).
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
~44.0 t/s mean on the held-out prompts at sclk 6 (34-54; README step 102); a long reasoning run
gains most (56.0 t/s on the held-out long proof). The cache first registers the experts' memory with the GPUs (~95-135 s in the
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

Without `--dspark ... --spec`: 17.7-21.7 t/s generation, and 8.5 GB more GPU 0 memory for
experts. Without `--placement` every routed expert runs on the CPU (faster start, ~12 t/s).
A higher GPU clock helps both: held-out with the cache 36.8 / 41.1 / 42.1 t/s at sclk 2 / 6 / 8
when last swept (README step 90; sclk 6 is 44.0 since); the clock is set outside the program (sudo).

After a memory-heavy job (anything that pushed the 474 GB model out of the page cache), the
first run reads it back from disk; `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` first (2-4 min)
keeps that out of a timed run.

Options:

- `-p "text"` or `-f prompt.txt`: the prompt, inline or from a file (as is).
- `--chat`: one user turn, no thinking.
- `--think`: with thinking. `--effort N`: its reasoning effort, 1-100 (default 75), or `low` /
  `high` / `max` (50 / 75 / 100), as the official encoding; the model card's benchmarks use 100.
- neither: the prompt is raw text after BOS.
- `-n`: max new tokens.
- `--prefill-chunk N`: prompt tokens per forward (default and most 512, the fastest for
  long prompts in the fast numerics). The output does not depend on it.
- `--no-overlap`: prompt chunks without the CPU/GPU overlap (slower; for comparisons).
- `-t`: CPU threads; the default 16 (one per physical core) is best with `--rt`. 32 leaves no
  CPU for the host thread and the HIP runtime and is much slower (README step 18).
- `--profile`: a per-step time breakdown, the CPU's misses per layer and what they cost (README
  step 50).
- `--trace FILE`: a Chrome trace of every step (open in ui.perfetto.dev; README step 51).
- `--gpu D`: the die for the runtime (default 0).
- `--placement FILE`: routed experts on GPUs, the file's top-ranked ones that fit (10-30 s upload).
- `--moe-dies 0,1,2,3`: the dies that hold experts (default all four).
- `--dspark FILE`: the DSpark draft GGUF (8.5 GB on GPU 0). Alone: shadow mode, drafts are
  measured but not used.
- `--spec`: speculative decoding (needs `--dspark`); the number of guesses verified per step
  is chosen from the draft's confidence. `--spec-n N` fixes it; `--spec-log FILE` writes
  per-step data.
- `--canon`: placement-invariant expert numerics: the output no longer depends on where the
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

Without `--rt` (or with `bin/ds-completion`) everything runs on the CPU.

With `--placement placements/study30.txt` the four GPUs hold the routed experts used most in
the 30-prompt routing study (43% of all, 10-30 s upload); 17.7-21.7 t/s on the test prompts (README step 18).

## Current results (2026-11-11)

| mode | command | decode | KL vs reference*: median / mean without the near-tie step |
|---|---|---|---|
| **+ adaptive expert cache** | `... --placement placements/study30.txt --dspark ... --spec --canon --cache-loads 800 --cache-period 8` | **33.9-53.8 t/s, mean 44.0** at sclk 6 (300 tokens, held-out prompts; README steps 50, 53, 57, 62, 68, 71, 73, 74, 80, 82, 84, 86, 88, 97, 98, 101, 102; `--cache-loads 200`: 25.6 at sclk 2, 28.0 at sclk 6) | output byte-identical to the plain `--canon` run |
| + speculative decoding | `... --placement placements/study30.txt --dspark ... --spec` | 16-24 t/s (300 tokens, held-out prompts; +22% geometric mean) | output byte-identical to the row below |
| own runtime + experts placed on GPUs | `bin-hip/ds-completion --rt ... --placement placements/study30.txt` | 17.7-21.7 t/s (128 tokens) | same numerics: per-layer gate identical to below |
| own runtime, dense on GPU 0 | `bin-hip/ds-completion --rt ...` | 12.0-12.5 t/s | 5.9e-6 / 4.8e-3 (earlier run 1.7e-5 / 6.3e-4) |
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
- **Where a step goes** (`--spec`, cache, sclk 6; the traces of README step 100, before steps
  101-102 took ~3% more off the dense part): a 6-token code verify step ~87 ms (medians):
  die 0's dense part before the CPU's experts ~52 ms, its wait for them ~26 ms, its wait for the
  other dies ~4 ms, outside the layers ~5 ms; the draft before it ~8.3 ms more. The prompt: the CPU's experts and the GPU's work overlap and the slower
  sets the pace (TIMING.md).
- **GPU memory with `--rt`:** 6.41 GiB fp8 + 1.75 GiB small weights + ~1 GB of prompt-chunk
  buffers on die 0; the rest holds experts (and the 8.5 GB draft with `--dspark`).

## Limits

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
