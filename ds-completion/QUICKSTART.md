# ds-completion quickstart

DeepSeek-V4.1-Flash-1017 text completion on the Mac Pro (4x Vega II, 755 GB RAM), on one
machine. (Until step 178 the model could also be split over two such machines: +14% on the
held-out prompts, the same bytes; README steps 110-112 and 178.)

- Dense weights (attention, shared experts, head) run on one GPU through our own HIP kernels,
  in exact fp8.
- Routed experts: the most-used ones on the four GPUs, the rest on the CPU (our own exact
  AVX-512 kernel); with the adaptive cache the GPUs follow the experts in use.
- The prompt goes through in chunks of up to 512 tokens; generation one token at a time, or a
  few at once with speculative decoding. Greedy by default, sampled with `--temp`; one sequence,
  text only.
- As a program for one prompt, as an interactive session (`-i`), or behind an HTTP server
  with the Anthropic Messages API ("A server" below).

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
  Set it right after a boot, before any GPU program runs (`scripts/set_vega_freq.sh` is this
  loop): once, set after a program had run, the level was in effect but the starred line read 1
  whenever the GPUs idled, and the records labelled the runs wrongly (README steps 178-179).
- **Model files** (in `~/llms/`; 488 GiB = 524 GB, run from the page cache, so the RAM must hold them):
  - `UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf` (474 GiB = 508 GB; Unsloth's
    [DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF));
  - `dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf` (8.5 GB): the DSpark draft for `--spec`;
  - `ds-1017-fp8/dense-fp8.gguf` (6.8 GB): made from the model by `bin/fp8pack` (below).

  Copies of files this size have come out silently corrupted on this hardware before, so check
  them: `cd ~/llms && grep -v '^#' ~/projects/llama-sandbox/checksums/DS-V4.1-Flash-1017.sha256 | sha256sum -c`
  (paths relative to the model directory; `dense-fp8.gguf` after making it). After a copy, check
  again after a reboot: a check right after it may only read the page cache.
- **llama.cpp**, upstream (its `master` branch), pinned to one commit: `e626db7ca` (2026-11-20),
  the one every check was last run with (step 175). Used for ggml, GGUF reading and the
  tokenizer, as static libraries in `~/projects/llama.cpp/build/` (`make LLAMA_CPP=<dir>` for another
  place; the Python reference reads the same variable from the environment). No fork and no
  pull request is needed. Do not run upstream's *own* programs (`llama-cli`) on these GGUF files:
  they read the Engram table's type as another one and generate garbage without an error; we
  decode that table ourselves (step 168, [docs/llama-cpp-upstream.md](docs/llama-cpp-upstream.md)).

## Build and prepare (once)

```bash
# llama.cpp: upstream master at the commit this was tested with (step 175)
git clone https://github.com/ggml-org/llama.cpp ~/projects/llama.cpp
cd ~/projects/llama.cpp
git checkout e626db7ca
cmake -B build -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32

cd ~/projects/llama-sandbox/ds-completion
make && make HIP=1                     # -> bin/ (CPU: fp8pack, ds-ref, tests) and bin-hip/ (GPU: ds-completion); LLAMA_CPP=<dir> for a llama.cpp checkout elsewhere
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf   # lossless fp8 repack, 6.8 GB, ~20 s
```

`make` builds every program: the tests and the benchmarks too. To serve or to chat, the product
alone is enough and about three times faster to build: `make HIP=1 product && bin-hip/fp8pack -m $M -o ...`
(`ds-completion` and `fp8pack`; `make tests`, with the CPU reference program `ds-ref`, and `make tools` are the other two groups, README
"The product and the tools around it").

`dense-fp8.gguf` comes out byte for byte the same on every build (its checksum is in the manifest),
so it doubles as a check of the build and the model files.

## Check

    scripts/check.py default   # ~5.5 min: builds; the checks without the model; logits against stored hashes;
                               # the product's outputs (held-out prompts, sampling, a second turn, a session) against stored ids
    scripts/check.py unit      # only the part without the model, 30 s

`default` after every change: it ends with "all checks passed" or names what failed and what
to run next. What there is beyond it (long context, the sessions and sampling matrices,
speed), test by test: [docs/testing.md](docs/testing.md).

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

Without `--dspark ... --spec`: 17.7-21.7 t/s generation (at sclk 2), and 8.0 GB more GPU 0 memory for
experts. Without `--placement` every routed expert runs on the CPU (faster start, ~12 t/s).
A higher GPU clock helps both: held-out with the cache 36.8 / 41.1 / 42.1 t/s at sclk 2 / 6 / 8
when last swept (README step 90; sclk 6 is 44.6 since); the clock is set outside the program (sudo).

After a memory-heavy job (anything that pushed the 474 GiB model out of the page cache), the
first run reads it back from disk; `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` first (2-4 min)
keeps that out of a timed run. After a reboot too: until the engram table is cached, each new
text reads its rows from disk and runs ~12% slower once (README step 117; `--record` shows it
as `read_bytes`).

Options:

- `-m FILE`: the model's first GGUF shard. `--rt FILE`: the GPU runtime, with the fp8 file.
- `-p "text"` or `-f prompt.txt`: the prompt, inline or from a file (as is).
- `--chat`: one user turn, no thinking.
- `-i` / `--interactive` (with `--chat` or `--think`): an interactive session, one user line per turn on
  one context; `-p` is then optional (the first turn), `-n` is per reply (reasoning included). An
  empty line ends it. With `--think`, earlier turns' reasoning is dropped from the context, as the
  reference does. `scripts/chat.sh [--think [--effort max]]` runs it with the fast configuration
  and `-n 30000` (README steps 108, 109).
- `--think`: with thinking. `--effort N`: its reasoning effort, 1-100 (default 75), or `low` /
  `high` / `max` (50 / 75 / 100), as the official encoding; the model card's benchmarks use 100.
- `--think-show all|line|none` (with `--think`): where the reasoning goes. `all` (default): on stdout,
  before the answer. `line`: one moving line on stderr (a token count and its last characters),
  erased when the answer starts; stdout has the answer alone. `none`: hidden. `scripts/chat.sh`
  uses `line`. README step 147.
- `--temp T [--top-p P] [--seed S]`: sample the next token instead of taking the most likely
  one (the default, `--temp 0`). The model card's settings are `--temp 1 --top-p 0.95`. One seed
  gives one output, whatever the placement (with `--canon`) and with or without `--spec`. README steps 148-151.
- neither: the prompt is raw text after BOS.
- `--requests` (in place of a prompt): prompts from stdin, one after another, each
  with the reply of an empty context; the reply's tokens as lines on stdout. For a server's front. The
  protocol: `docs/requests.md`. README steps 152, 158.
  `python3 server/ds_server.py [--port 8080] -- <this command with --requests>` puts an HTTP server
  with the Anthropic Messages API in front of it (`docs/server.md`, README step 153).
- `-n`: max new tokens.
- `--max-ctx N`: the context's size (default 65,536; "Limits" below).
- `--prefill-chunk N`: prompt tokens per forward (default and most 512, the fastest for
  long prompts in the fast numerics). The output does not depend on it.
- `--no-overlap`: prompt chunks without the CPU/GPU overlap (slower; for comparisons).
- `-t`: CPU threads; the default 16 (one per physical core) is best. 32 leaves no
  CPU for the host thread and the HIP runtime and is much slower (README step 18).
- `--profile`: a per-step time breakdown, the CPU's misses per layer and what they cost (README
  step 50).
- `--trace FILE`: a Chrome trace of every step (open in ui.perfetto.dev; README step 51).
- Time per kernel on the device, HIP call counts, memory copies: AMD's `rocprofv3` in front of the
  command ([docs/rocprof.md](docs/rocprof.md), README steps 140-141). It slows generation by 21-26%;
  with hardware counters (`--pmc`) to 4-7 t/s.
- `--gpu D`: the die for the runtime (default 0).
- `--placement FILE`: routed experts on GPUs, the file's top-ranked ones that fit (10-30 s upload).
- `--moe-dies 0,1,2,3`: the dies that hold experts (default all four).
- `--dspark FILE`: the DSpark draft GGUF (an 8.5 GB file, 8.0 GB on GPU 0). Alone: shadow mode, drafts are
  measured but not used.
- `--spec`: speculative decoding (needs `--dspark`); the number of guesses verified per step
  is chosen from the draft's confidence. `--spec-n N` fixes it; `--spec-log FILE` writes
  per-step data.
- `--canon` ([docs/numerics.md](docs/numerics.md)): placement-invariant expert numerics: the output no longer depends on where the
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

- `--log-routing FILE`: the router's choices per token and layer (the routing studies).
- `--record FILE`: the run as one JSON object: a few of its options, the prompt's token count and
  hash, the GPU clock setting (each card's level, or `"auto"`), the bytes read from storage during
  the load, the prompt and the generation (0: nothing waited for the disk), with `--spec` the
  speculative loop's totals (steps, drafts accepted, steps by drafts verified, draft and verify
  time), the routed experts' uses on GPUs and on the CPU, the experts placed per die and the
  cache's totals, a hash of the printed token ids, load time, prompt and generation t/s.
  `--record-label key=value[,key=value...]` (repeatable): names for the run, written back as
  `label` and never read by the program. Many runs: `scripts/run_matrix.py MATRIX` starts
  prompts x configurations x repeats with a record each, `scripts/records.py` reads them
  (README steps 128-129). A first version, to grow
  ([docs/record.md](docs/record.md), README steps 115-119). One prompt only (not with `-i`); a run that does not
  finish leaves the file empty.
- `--no-quant` (`bin/ds-ref` only): without the reference's fp8/fp4 rounding (a comparison).

`ds-completion` needs `--rt` (step 173). The model on the CPU alone, through ggml graphs, is the
reference path and a program of its own: `bin/ds-ref -m $M -p "..."` (built by `make`; the same
options for the prompt, the modes, `-n`, `-t`, sampling and `--record`; none of the runtime's).
Until step 173 this was `bin/ds-completion`, or `bin-hip/ds-completion` without `--rt`.

## A server

`server/ds_server.py` (Python 3, standard library only) starts the engine with `--requests`
and serves the Anthropic Messages API (README steps 152-154; [docs/server.md](docs/server.md)).
Everything after `--` is the engine's command, the command above with `--requests`
in place of the prompt:

```bash
cd ~/projects/llama-sandbox/ds-completion
python3 server/ds_server.py --port 8080 -- \
    bin-hip/ds-completion -m $M --rt ~/llms/ds-1017-fp8/dense-fp8.gguf \
    --placement placements/study30.txt \
    --dspark ~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf --spec \
    --canon --cache-loads 800 --cache-period 8 --cache-wait --requests
```

`scripts/serve.sh` is this command with every log kept (the request log, dumps, `/metrics`
every minute) in a new directory under `~/ds-serve-logs/` per start, and `--max-ctx 262144`
(`MAX_CTX` for another; it warns if pi's `models.json` asks for more).

It prints `listening 8080` when the model is loaded (~2.5 min). A client: the Anthropic SDK with
`base_url="http://127.0.0.1:8080"` (or `ANTHROPIC_BASE_URL`) and any API key; or the pi coding
harness with `server/pi-models.json` (the setup: [docs/pi.md](docs/pi.md)).

- **What it reads of a request:** user and assistant text turns, a system prompt, tools,
  `thinking`, `output_config.effort` (`low` 50, `medium` 62, `high` 75, `xhigh` 88, `max` 100),
  `max_tokens`, `temperature`, `top_p`, `stream`. The model's own `encoding.py`
  (`--encoding`, default `~/llms/ds-flash/encoding/encoding.py`) renders the prompt and parses
  the reply, tool calls included.
- **Sampling:** temperature 1 and top-p 0.95 unless the request says otherwise, and one seed
  for every request (`--seed`, default 0): the same request gives the same reply.
- **One request at a time;** others wait. A stream carries the reasoning and the text as they are
  generated, tool calls whole at the end, pings in the silences. A request whose client leaves is cancelled.
- **A conversation with tool calls:** earlier replies (reasoning, text, tool calls) and tool
  results are read. A request that goes on from the last one (its prompt, its reply, a new
  turn) runs only the new tokens; one that shares its start with the last ones (the system
  prompt and tools) runs from a point saved there; any other from its first token.
- **Refused:** forced tool use (`tool_choice` `any` / `tool`), images.
- **The log** (stderr): a line per finished request, and every 30 s the tokens so far of a
  running one. `--log FILE` appends a JSON line per request (time, outcome, tokens, seconds,
  where the engine started); `--dump DIR` keeps each request's prompt, raw reply and message.
  The engine's `# context:` lines say where each request started and where its prompt and
  the context part (step 164).
- **Metrics and health:** `GET /metrics` has counters and gauges in the Prometheus text format
  (requests by outcome, tokens run / kept / generated, seconds, the engine up and busy;
  `docs/server.md` "Metrics", step 165); `GET /health` is 200 while the engine lives, 503 when
  it is gone.
- **The network:** it listens on `127.0.0.1`; `--host 0.0.0.0` for other machines. No
  authentication, no encryption: a private network only.
- A reasoning request can take long: one of the benchmark this was written for ran 19 minutes
  (43k tokens at 38.5 t/s; README step 153).

The engine's side alone, for another front: `--requests`, [docs/requests.md](docs/requests.md).

## Current results (each row from the step it names; read through again 2026-11-23, step 182)

The two top rows were measured with `--cache-wait` (the run starts after the cache's
registration); without it a short run starts at the static placement's speed. At sclk 6 unless
a row says otherwise (the clock does not matter to the CPU rows).

| mode | command | decode | KL vs reference*: median / mean without the near-tie step |
|---|---|---|---|
| **speculative decoding + adaptive expert cache** | `... --placement placements/study30.txt --dspark ... --spec --canon --cache-loads 800 --cache-period 8` | **33.9-55.0 t/s, mean 44.6** (300 tokens, held-out prompts; README step 104) | output byte-identical to plain decoding with `--canon` |
| the same, sampled | `... --temp 1 --top-p 0.95 --seed 1` | mean 43.1 against 44.9 greedy in the same session (README step 151) | the plain loop's output for the same seed |
| dense on GPU 0, every routed expert on the CPU | `bin-hip/ds-completion --rt ...` without `--placement` | 12.0-12.5 t/s (sclk 2) | 5.9e-6 / 4.8e-3 (earlier run 1.7e-5 / 6.3e-4) |
| CPU only, the reference path | `bin/ds-ref` (`bin/ds-completion` until step 173) | 3.5 t/s | 1.3e-5 / 3.9e-3 |
| llama.cpp with the DeepSeek-V4.1 PR (llama-cli, CPU; steps 2-3) | - | 3.5 t/s | 2.1e-5 / 1.3e-3 |

The steps between these rows (the static placement, speculation without the cache) and their
numbers: docs/overview.md, section 2.

\* vs the official reference implementation run on the CPU (README "Step 2"): chat prompt
"Hello, how are you?", 17 steps. Every mode agrees with the reference on the top token at 16
of 17 steps. The miss is one near-tie (step 12), which dominates any plain mean, so the table
leaves it out.
Means over all 17 steps: 1.8e-2 to 2.5e-2 for every mode.

- **The runtime is also gated per layer:** attention and FFN output within ~1e-7 of the fp32 +
  fake-quant reference. It has no inexact op left, and repeated runs are bit-identical.
- **Prompt processing** (chunks of 512, the CPU's experts overlapped with the GPU's
  work; README steps 37-45, 68): 93 t/s, 63 with `--canon` (1,989 tokens; 97 / 73 at 6,106; sclk 6),
  output byte-identical to token-by-token processing; ~7.7 t/s with `bin/ds-ref`. The
  runtime's mean KL varies run to run with fp8 rounding flips (README step 10); every exact mode
  scatters the same way.
- **Where a step goes** (`--spec`, cache, sclk 6; the traces of README steps 105 and 155): a 6-token
  code verify step ~85 ms (medians): die 0's dense part ~48 ms, its wait for the CPU's experts
  ~27 ms, its wait for the other dies ~5 ms, outside the layers ~5 ms; the draft before it ~8 ms
  more. The prompt: the CPU's experts and the GPU's work overlap and the slower
  sets the pace (TIMING.md).
- **GPU memory:** on die 0 the fp8 weights (6.85 GB), the small weights (1.9 GB) and
  activations and scratch (0.8 GB); the rest holds experts (and the draft's 8.0 GB with
  `--dspark`; `results/gpu-memory/`).

## Limits

- `-i` sessions: one line per turn, no system prompt, no tool calls.
- Context: 65,536 tokens by default (prompt + generation), up to the model's
  1,048,576 with `--max-ctx N` (~6.4 KB of GPU 0 memory per position). A 30k-token prompt takes
  ~5-8 min at 63-97 t/s (it took ~30 min token by token). `bin/ds-ref` (the CPU reference path): 4,096 by
  default, at most 16k (no candidate-block selection there).
- The dense part uses one GPU; the others only hold experts with `--placement`.
- At exit the process sometimes aborts (signal 6, exit code 134) inside the GPU runtime's
  shutdown, after `main` has returned: the output and the timing lines are complete by then
  (README steps 93 and 96; unexplained; seen again once in 16 runs in step 175).
- Greedy decoding (the default) can fall into a repetition loop on long reasoning (README
  step 48: a proof prompt repeats one line after ~800 tokens; the model's behaviour, not a
  bug). Sampling (`--temp 1 --top-p 0.95`, the model card's settings) is the way around it.
- The server: one request at a time, no forced tool use, no images, one context and one saved
  point kept between requests, tool calls streamed only when complete, the Anthropic API only.

Details: [docs/timing.md](docs/timing.md) (where the time goes, one page), [docs/overview.md](docs/overview.md) (a 20-minute summary), [docs/steps.md](docs/steps.md) (all measurements and checks, step by step), [docs/plan.md](docs/plan.md) (what's next).
