# Reproducing ds-completion

What is needed to rebuild the setup, re-run every experiment and regenerate every file in
`results/`. How the work was approached and why each step followed the last: [docs/process.md](process.md).
What each step found: [docs/steps.md](steps.md).

Items marked **(to confirm)** are not yet pinned down; they are listed at the end.

## 1. Machine

| | |
|---|---|
| computer | Mac Pro 2019 |
| CPU | Intel Xeon W-3245, 16 cores / 32 threads, AVX-512 (VNNI) |
| RAM | 755 GB (~112 GB/s measured, 16 threads) |
| GPUs | 2x Radeon Pro Vega II Duo = 4 dies (gfx906), 32 GB HBM2 each, PCIe 3.0 (MPX) |
| GPU links | Infinity Fabric (XGMI) within each card: dies 0-1 and 2-3 (bridge reseated; see the sibling repo `macpro-2019-vega-if`: ROCm crashes on cross-card runtime copies, so nothing here relies on peer copies) |

The RAM must hold the whole GGUF (474 GiB = 508 GB) in the page cache next to the work: everything runs
from mmap, and the routed experts are read from it on every token.

**The second machine** (README steps 110-112; not used by any program since step 178, which
removed the model in two halves; hostnames `tomb`, the first, and `shower`): the
same computer, CPU, RAM and GPUs, without the XGMI bridges (unused); 10 GbE between the two
(direct or switched, MTU 1500, ping ~0.2 ms). For those steps it had the same software (section 2), the same model
files at the same paths with the same sha256 (`results/split/sha256.txt`), this repository at
the same commit and built (section 4), the same sclk level; what it has now: section 7. The one-machine results do not need
it; it reproduced them to the byte at the same speed (44.33 against 44.60 t/s held-out).

## 2. Software

| | version |
|---|---|
| OS | Ubuntu 22.04 LTS, kernel `6.10.5-1-t2-jammy` (t2linux, for the Mac's T2 chip) |
| ROCm | 6.3.4 in `/opt/rocm-6.3.4` (HIP 6.3.42134; gfx906 is past its support window but works) |
| compiler | g++ 11.4.0; hipcc (AMD clang 18, from ROCm 6.3.4) |
| llama.cpp | upstream (`master`) at `e626db7caadee98c6174b1c2dc69f3f43075faa0` (2026-11-20), in `~/projects/llama.cpp` (`~/forks/llama.cpp` until step 175; `make LLAMA_CPP=<dir>`, `FORK=<dir>` in older commands, for another place (`FORK` did nothing from step 176 to step 182: the Makefile assigned the variable to itself)); CPU build in `build/` (static libraries), QUICKSTART's two `cmake` lines. Used for ggml, GGUF reading and the tokenizer. Every check was run with it in step 175 (`results/dist/upstream.txt`). |
| llama.cpp, until step 167 | `~/forks/llama.cpp` on branch `deepseek41`: commit `b72050d03` "model : add DeepSeek-V4.1-Flash (deepseek41)" (Unsloth; the head of llama.cpp PR #30817) on top of upstream `797de63b2`. The results of steps 1 to 167 and step 174 were made with this build, and they are what "the fork" means in older steps and results files. Still needed for one thing: llama.cpp's own implementation on these files (the CPU baseline `llama-cli` of steps 2-3, `bin/llama-logits`), which upstream reads wrong ([llama-cpp-upstream.md](llama-cpp-upstream.md)). |
| Python (reference) | `ref/weights.py` imports `gguf-py` from the llama.cpp checkout (`LLAMA_CPP` in the environment, default `~/projects/llama.cpp`; upstream's works since step 176). venv `~/venvs/ds-ref`, Python 3.10.12: `pip install -r ref/requirements.txt` (a `pip freeze`: torch 2.14.1+cpu, numpy 2.2.6, safetensors 0.8.0, sympy 1.14.0, tokenizers 0.23.3, huggingface_hub 2.3.0, ...) |
| perf (counters, step 79) | Ubuntu's `linux-tools-generic` (perf 5.15) run as `/usr/lib/linux-tools/5.15.0-202-generic/perf` (the `/usr/bin/perf` wrapper refuses the 6.10 kernel); user-space core events with `perf_event_paranoid` 2 |

GPU clock: experiments state their sclk level (6 for every measurement since step 48, and in steps 42-44; step 90
sweeps 2 / 6 / 8; before that 2, 1204 MHz). It is set by
hand, as root, per card:

```bash
for c in /sys/class/drm/card[0-3]/device; do
  echo manual | sudo tee $c/power_dpm_force_performance_level
  echo 6 | sudo tee $c/pp_dpm_sclk          # level 0..8 (999..1730 MHz)
done
```

Warm page cache: the model is memory-mapped, so after a reboot the first run reads experts from
NVMe and is several times slower (README step 12). Measure only with the files cached, e.g. after
one full run.

## 3. Model files

| file | source | size |
|---|---|---|
| `~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf` | [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) of deepseek-ai/DeepSeek-V4.1-Flash-1017 (GGUF metadata: `quantized_by Unsloth`, `repo_url https://huggingface.co/unsloth`, base model `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`) (revision not recorded; the checksum manifest below pins the bytes) | 474 GB |
| `~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf` | Unsloth's DSpark draft head GGUF (arch `dflash`, metadata as above) (revision not recorded; the checksum manifest below pins the bytes) | 8.5 GB |
| `~/llms/ds-flash/` | the base DeepSeek-V4.1-Flash safetensors (HF commit `314113f8`); only for weight provenance checks. The GGUF matches the 1017 release (HF commit `1bae0e10`) instead, which `ref/check_weights.py --hf remote` reads by HTTP range requests (README "Which checkpoint") | 510 GB |
| `~/llms/ds-1017-fp8/dense-fp8.gguf` | made here: `bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf` (lossless fp8 repack of the dense weights, ~20 s; README step 5) | 6.8 GB |

Checksums of all eleven files: `../checksums/DS-V4.1-Flash-1017.sha256` (QUICKSTART.md's
`sha256sum -c` line). Run it on every machine, and again after a reboot following a copy: a check
right after a copy may only read the page cache.

## 4. Build

```bash
# llama.cpp: upstream, a CPU build with static libraries in build/, at the commit and with the cmake lines of QUICKSTART.md "Build and prepare"
cd ~/projects/llama-sandbox/ds-completion
make            # bin/: fp8pack, the unit tests and the CPU benchmarks (links llama.cpp's static CPU libraries)
make HIP=1      # bin-hip/: ds-completion (the product: the GPU runtime), ds-test, the GPU benchmarks
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
```

Run from this directory (placement files are relative paths). How to run: [QUICKSTART.md](../QUICKSTART.md).

## 5. Data that is not in git (`data/`, ~14 GB; ~115 GB with the 20k-token run of README step 35)

| directory | what | how to regenerate | cost |
|---|---|---|---|
| `data/ref/` | reference runs (`.pt`): the official model math on the CPU (`ref/ds_ref.py`) | `~/venvs/ds-ref/bin/python ref/ds_ref.py --prompt "..." --chat --n-decode 16 [--fp32] [--no-quant] --out data/ref/x.pt` (prompts: README step 2; the 20k-token run: step 35) | minutes to ~1 h (20k tokens: 42 min, ~500 GB of RAM, a 35 GB file) |
| `data/golden/` | per-layer golden activations and logits for the gates | `ref/export_golden.py data/ref/x.pt data/golden/x` | minutes |
| `data/ds/`, `data/llama/` | our and llama.cpp's logits for the comparisons | `bin-hip/ds-test ... --runtime $F8 --e2e` (before step 194 also `bin/ds-test ... --e2e`, the CPU reference path); `bin/llama-logits` (README steps 2-3; `make bin/llama-logits LLAMA_CPP=<a checkout of the PR's commit>`, not part of `make` since step 175) | minutes |
| `tests/fq/` (in git since step 186; `data/fq/` before) | fake-quant test vectors | `ref/make_fq_test.py` | seconds |
| `data/routing/study/` | routing logs of the 30 study prompts (plain): inputs of `placements/study30.txt` and the simulations | `results/routing-study.sh` (prompts in `prompts/routing-study.tsv`) | ~1 h |
| `data/routing/oracle/` | routing logs of the 2 oracle prompts | README step 15 | minutes |
| `data/routing/spec/` | routing logs of the 8 held-out prompts under `--spec` | `results/spec-routing.sh` | ~15 min |
| `data/routing/prefill/` | two long prompts (`head -c 24000 prompts/ds-completion-readme-ddf9b43.md`, `head -c 32000 hip/rt.cpp` at e3542ac) and their routing logs | `--log-routing` (README step 42) | ~7 min |

The routing logs (48 MB) are cheap to keep and should be archived with any publication: the
placement and every simulation number depend on them, and regenerating them only reproduces
them exactly with the same binary and placement.

## 6. Which command produced which result

Every number in the documents comes from a file in `results/`. Speed numbers depend on the sclk level
and vary ~1-3% run to run; correctness results (identical outputs, 0 differing values) must
reproduce exactly. Up to step 26 many runs were typed by hand: the table names the program and
the step (`docs/steps.md`) gives the settings (prompt, tokens, threads, placement), but not always a
script; from step 27 on every result has a script. The commit that added a file:
`git log -- results/<file>`. Convention from now on: a result file starts with a `#` header
naming the command or script, the commit, the clock level and the date (summary and table
files from step 31 on; earlier ones have none, and raw `.out` / `.err` outputs are described by
their directory's summary or README.txt). **Not in git:** `.log` files (the per-step logs of
`--spec-log` and of the `DS_*_LOG` switches; the repository ignores `*.log`), so a clone has
none of them, and `results/decode-next/draft2/` and `draft-route/` are empty there: the row's
command writes them again. **Scripts** (since step 130): one written for a single step lies
with its results and the rows below name it there; `scripts/` has the tools still in use
(`scripts/README.md`). A header in an older results file may still say `scripts/<name>`.

Diagnostic switches read from the environment (each explained at its `getenv` and in the step
that used it; none changes the output): `DS_EARLY_ROWS`,
`DS_ATTN_CHECK` (`hip/rt.cpp`), `DS_DRAFT_PROFILE`, `DS_DRAFT_ROUTE_LOG` (`hip/dspark.cpp`),
`DS_DRAFT_TOKEN_LOG` (`apps/ds_completion.cpp`), `DS_DRAFT_KEEP`, `DS_DRAFT_KEEP_RESIDENT` (`hip/dspark.cpp`, steps 124-125; with the placement they change what is loaded and so the speed, never the output), `DS_FP8_PAD`, `DS_FP8_VIRT`,
`DS_STALL_SPLIT` (`hip/fp8_gemv.*`), `DS_F32_PAD` (`hip/kernels.hip`). A `DS_*` name next to a
`git apply results/...patch` below exists only with that patch applied.

Unless stated: sclk 2, 16 CPU threads (32 before README step 18), the study30 placement.

| step | results | produced by |
|---|---|---|
| 1 | `check-weights-1017.txt` | `ref/check_weights.py --experts all --hf remote` |
| 2 | `ref-*.txt`, `compare-hello-ref-vs-fork.txt` | `ref/ds_ref.py`, `bin/llama-logits`, `ref/compare_logits.py` |
| 3, 6 | `test-*.txt`, `compare-hello-ref-vs-ds*.txt` | `bin[-hip]/ds-test` (per layer, `--e2e`), `ref/compare_logits.py` |
| 4 | `bench-ggml-bf16.txt` | the ggml GPU path (removed in step 11; recoverable from git history) |
| 5 | `fp8pack.txt`, `bench-fp8*.txt` | `bin/fp8pack`, `bin-hip/bench-fp8 $F8 [--sweep]` |
| 7 | `bench-cpu-moe-*.txt` | `bin/bench-cpu-moe -m $M` (variants: rows per task, prefetch, SMT) |
| 8, 17 | `gpu-moe-e2e.txt`, `bench-fp4-moe-v1.txt` | `bin-hip/ds-completion --gpu-moe ...` (since removed), `bin-hip/bench-fp4-moe` |
| 9 | `sclk/` | `results/sclk/sclk-sweep.sh`, `results/sclk/sclk-summary.py` (per step: the speed table at the levels) |
| 10-11 | `rt-prefill-*.txt`, `rt-cleanup-speed.txt` | `bin-hip/ds-completion --rt` |
| 13 | `routing-study*.txt` | `scripts/route_stats.py data/routing/study/*.bin [--cross prompts/routing-study.tsv]` |
| 15 | `oracle-placement.txt` | `scripts/route_stats.py --emit-placement`, then the runtime per placement |
| 18 | `threads-sweep.txt` | `bin-hip/ds-completion -t N` |
| 19 | `dspark-shadow.txt` | `bin-hip/ds-completion --dspark $D` (shadow mode) |
| 20 | `spec-phase1.txt`, `fp8-batch-bitwise.txt` | `--dspark --spec` vs `--dspark`; `bin-hip/bench-fp8 --ms ...` |
| 21-23 | `spec-draft-length.txt` | `--spec-n 1..5` |
| 24 | `bench-fp8-multi.txt` | `bin-hip/bench-fp8 --ms 1,2,...,8` |
| 25 | `spec-conf/`, `spec-conf-calibration.txt` | `--spec-log`, `results/spec-conf/spec_conf.py --cal ... --cost ...` |
| 26 | `spec-heldout/`, `spec-profiles.txt` | the held-out prompts (`spec-heldout/prompts.txt`) plain and `--spec`; `--spec-n 5 --profile` |
| 27 | `spec-sclk/` | `results/spec-sclk/spec-sclk.sh` at sclk 0, 2, 8 |
| 28 | `canon-phase0.txt` | `bin/bench-cpu-moe -t 16 --tokens 14 --check-layers 2,20,39`, `bin-hip/bench-fp4-moe --layer 20 --reps 300` |
| 29 | `cache-sim.txt` | `results/spec-routing.sh`, then `results/cache_sim.py` |
| 30 | `bench-expert-load.txt`, `load-test/` | `bin-hip/bench-expert-load -m $M`, `results/load-test.sh` (with a build of commit def1a2a: `--load-test` was removed in step 134) |
| 31-32 | `canon-gate.txt`, `canon-gate/` | `results/canon-gate/canon-gate.sh` (rerun for steps 32 and 45) |
| 32 | `cache-test/`, `cache-test-sclk8/` | `results/cache-test/cache-test.sh` (at sclk 8: `results/cache-test/cache-test.sh "spec fast static" "spec canon 200"`); the CPU kernel at 1, 2, 6 experts: `bin/bench-cpu-moe -m $M -t 16 --tokens 8 --check-layers 2 -k N` |
| 33 | `verify-breakdown.txt` | `bin-hip/ds-completion ... --dspark $D --spec --spec-n 5 --canon --profile [--cache-loads 200 --cache-wait]` (command in the file's header), medians of the second half |
| 34 | `check-topk.txt` | `~/venvs/ds-ref/bin/python ref/check_topk.py` (runs `bin-hip/test-topk`) |
| 34 | `long-context/` | `bin-hip/ds-completion ... --max-ctx N -p "$(cat README.md) ..."` (commands in `long-context/summary.txt`) |
| 35 | `ref-readme20k.txt`, `ref-readme20k-scores.txt` | `ref/ds_ref.py --file prompts/ds-completion-readme-ddf9b43.md --prompt-tokens 20000 --n-decode 8 --max-seq-len 20480 --keep-last 32 [--scores]` (~42 min, ~500 GB peak), then `ref/export_golden.py` (only the `--scores` run is kept) |
| 35 | `rt-readme20k.txt`, `rt-readme20k-e2e.txt`, `compare-readme20k-ref-vs-rt.txt` | `bin-hip/ds-test -m $M -g data/golden/readme20k-s --runtime $F8 --dump-topk data/ds/readme20k-topk.bin` (~35 min); `... -g data/golden/readme20k-s --runtime $F8 --e2e data/ds/readme20k-rt.f32`; `ref/compare_logits.py data/ref/readme20k-s.pt data/ds/readme20k-rt.f32` (the e2e and compare files name `readme20k`, the first run without `--scores`: bit-identical in every tensor it has, since deleted) |
| 35 | `topk-ties-readme20k.txt` | `~/venvs/ds-ref/bin/python ref/check_topk_ties.py data/golden/readme20k-s data/ds/readme20k-topk.bin` |
| 37 | `prefill/gate.txt`, `prefill/gate/` | `results/prefill/prefill-gate.sh <baseline ds-completion>` (baseline: a build of commit 73b6891, token-by-token prompt) |
| 37 | `prefill/profile.txt` | `bin-hip/ds-completion ... -f <prompt> -n 1 --profile --prefill-chunk N` (command and stages in the file) |
| 37 | `prefill/bench-cpu-moe-batch.txt` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --batch T` |
| 38 | `prefill/overlap.txt` | `bin-hip/ds-completion ... -f results/prefill/gate/pN.txt -n 32 [--no-overlap] [--canon] [--profile]` (commands in the file) |
| 39 | `prefill/bench-fp8-gemm.txt` | `bin-hip/bench-fp8 $F8 --gemm --ms 256 --sweep`, then `--gemm --ms 64,512` |
| 39 | `prefill/gemm-e2e.txt` | `bin-hip/ds-completion ... -f results/prefill/gate/pN.txt -n 32 [--canon] [--prefill-chunk 1] [--profile]` (commands in the file) |
| 40 | `prefill/bench-fp8-gemm-tuning.txt` | experimental builds (variants removed; described in the file); `prefill/bench-fp8-gemm.txt` rerun with the kept choice: `bin-hip/bench-fp8 $F8 --gemm --ms 64,256,512` |
| 41 | `prefill/bench-cpu-moe-batch.txt` (rerun), `prefill/canon-e2e.txt` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --batch T`; `bin-hip/ds-completion ... --canon -f results/prefill/gate/pN.txt -n 32 [--profile]` |
| 42 | `prefill/stream-sim.txt` | routing logs of `data/routing/prefill/{readme24k,rtcpp32k}.txt` (`--log-routing`, commands in the file), then `results/prefill/prefill_stream_sim.py [--cpu-scale 1.65] [--load-rate 1000] [--canon]`; the calibration run is the same prompt with `--profile` |
| 43 | `prefill/bench-register.txt` | `bin-hip/bench-register $SHARD7 --gb 8 --pieces 0,1024,256,64 --faults $SHARD6` |
| 43 | `prefill/bench-expert-load-pieces.txt`, `prefill/register-stall.txt` | `bin-hip/bench-expert-load -m $M`; the runtime commands in the file |
| 44 | `prefill/stream-attempt.txt` | the removed P2 code at each stage (commands and stages in the file; the code is not in the repository) |
| 45 | `prefill/o3.txt`, `canon-gate.txt` + `canon-gate/` (rerun) | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --replay data/routing/prefill/readme24k.bin`; `bin-hip/ds-completion ... --profile` (commands in the file); `results/canon-gate/canon-gate.sh` |
| 46 | `prefill/chunk1024.txt` | `kChunk = 1024` (reverted), then the commands in the file |
| 47 | `spec-momentum.txt` | `~/venvs/ds-ref/bin/python results/spec_momentum.py 'results/spec-conf/*-n5.log'`; docs/timing.md quotes steps 47-82 (generation, prompt, the cache's loads) |
| 48 | `spec-measure-sclk6/` (summary, per-prompt logs, `analysis.txt`) | `scripts/spec-measure.sh`, then `scripts/spec_analysis.py --max-steps 400 results/spec-measure-sclk6/*.log` |
| 48 | `spec-loop-check.txt`, `engram-copy-fix.txt` | commands in the files (the old build: a worktree at 73b6891) |
| 49 | `decode-profile-sclk6.txt` | the command in the file under `LD_PRELOAD` of `results/hip_count_shim.c` (`gcc -O2 -shared -fPIC -o shim.so results/hip_count_shim.c -ldl`), `scripts/profile_verify.py`, `bin-hip/bench-launch` |
| 50 | `cpu-moe-parallel-act.txt` | `bin/bench-cpu-moe -m $M -t 16 --tokens 8 --check-layers 2 -k N`; `... --check-layers "" --batch 6` |
| 50 | `cache-tuning-sclk6.txt`, `spec-measure-sclk6-cacheloads200cacheperiod16/`, `spec-measure-sclk6-cacheloads800cacheperiod8/` | the two-prompt command in the file with each setting; `CACHE="--cache-loads 200 --cache-period 16" scripts/spec-measure.sh`, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 50 | `miss-histogram-sclk6.txt` | the command in the file (`--profile`) |
| 51 | `trace/` | `bin-hip/ds-completion ... --spec --canon --cache-loads 800 --cache-period 8 --cache-wait --chat -n 300 --trace <name>.json` (in `trace/README.txt`), then `gzip -9` |
| 52 | `prefetch-prediction-sclk6.txt` | the `--profile` output of the step 51 runs (commands in `trace/README.txt`) |
| 53 | `router-batched.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-router/`, `prefill/router-batched-ab.txt` | `bin-hip/bench-kernels`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (page cache warm first: `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null`; the run writes the step-50 directory, copied); the A/B commands in the file (old build: a worktree at 97844c8) |
| 54 | `coselect.txt` | `~/venvs/ds-ref/bin/python results/coselect.py data/routing/study/*.bin --placement placements/study30.txt --trace results/trace/cpp-matrix.json.gz` |
| 55 | `fp4-grouped/` | `git apply results/fp4-grouped/attempt.patch` on da19a4f, `make HIP=1 bin-hip/bench-fp4-moe`, `bin-hip/bench-fp4-moe -m $M` (variant 2: `kCanonR` in `hip/fp4_moe.hip`; variant 1: `canon_block` instead of the decoded form in the grouped kernels) |
| 56 | `cpu-decode/` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --decode 1,2,4,6` [`--gap-us 1500`, `OMP_*` env, CPU settings as in the file]; `... --replay data/routing/prefill/readme24k.bin [--static]`; the runtime A/B used a `--cpu-static` flag since removed (`batch_sched = kStatic` for calls of up to 16 tokens) |
| 57 | `cpu-decode/per-call.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-nx1/` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --decode 1,2,6 --steps 20` (the per-call lines), `--batch 1`, `--batch 3`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (writes the step-50 directory; copied) |
| 58 | `trace/` (regenerated), `timing-sclk6/` | the commands in `trace/README.txt`, then `gzip -9`; the `--profile` comparison and `timing-sclk6/verify-profile.txt`: the same command with `--profile` instead of `--trace`, then `scripts/profile_verify.py`; `timing-sclk6/prompt-profile.txt`: the command in the file |
| 59 | `cache-decay/` | `~/venvs/ds-ref/bin/python results/cache_sim.py --decay` (`sweep.txt`); `threshold.txt`: `cs.run(..., rate=800, period=8, [half_life=H], min_count=M)` over the same logs (see its header); `live-min.txt`: the command in the file with `--cache-min 2` and `1` |
| 60 | `gpu-memory/` | the command in `gpu-memory/README.txt` at each `--max-ctx` (the ledger is printed at startup) |
| 61 | (none: a reading of `hip/dspark.cpp`) | the draft's loader: `fp8(...)` for its dense matrices, `fp4_upload(...)` for its experts |
| 62 | the load rate in docs/timing.md: `loads started` and `generation:` in `spec-measure-sclk6-cacheloads800cacheperiod8-head/*.err` | (from the files) |
| 62 | `head-lds.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-head/` | `bin-hip/bench-kernels --bitwise`; the variants: `gemv_head_launch<R, WAVES>` in `hip/kernels.hip`; `DS_DRAFT_PROFILE=1 bin-hip/ds-completion ...` (the draft's parts); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (copied) |
| 63 | `route-ahead/` (`summary.txt`, `pooled.txt`, per prompt `.err`) | with a build of commit 8798740 (the switches were removed in step 135): `results/route-ahead/route-ahead.sh 4` (`DS_ROUTE_AHEAD=4 DS_ROUTE_LOG=...`; logs in `data/route-ahead/`, ~185 MB); `pooled.txt`: the per-prompt tables weighted by their counts |
| 64 | `route-ahead/prefetch-sim.txt` | `~/venvs/ds-ref/bin/python results/route-ahead/prefetch_sim.py` on `data/route-ahead/*.bin` (from step 63's run) |
| 65 | `decode-next/fp8-multi.txt`, `fp8-lds64.patch`, `host-lag.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,4,6 [--gemm \| --sweep]` (the 8-wave variant: `git apply results/decode-next/fp8-lds64.patch`, `DS_FP8_LDS64=8`); the lag: the command in `host-lag.txt` (`--trace` prints it) |
| 66 | `attn-batch/stage1.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-ring/` | `results/prefill/prefill-gate.sh`-style runs, a 20,000-byte prompt against a build of 67453f6, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (commands in the file; the spec run copied) |
| 67 | `attn-batch/stage2.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-phases/` | the same gates (commands in the file; the spec run copied) |
| 68 | `attn-batch/stage3.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-attnbatch/` | `DS_ATTN_CHECK=1 bin-hip/ds-completion ...` (prints the comparison at exit), the prompt gate, the long prompt against a build of 6e7126a, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (copied), the profile with `--profile` and `scripts/profile_verify.py` |
| 69 | `trace/` (regenerated), `trace/compare-step58.txt` | the commands in `trace/README.txt`, then `gzip -9 -n`; the comparison: `scripts/trace_compare.py` on step 58's traces (`git show d7bbabd:ds-completion/results/trace/<name>.json.gz`) and the new ones |
| 70 | `decode-next/verify-halves.txt`, `verify-halves.patch` | the commands in the file; the layer medians: `results/trace/*.json.gz` (step 69) |
| 71 | `decode-next/handoff.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-handoff/` | the gate commands in the file (prompt gate, the 20,000-byte prompt, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`, copied); the host timer was a temporary patch (three clock reads in `moe_finish`), the trace medians: `--trace` as in `trace/README.txt` |
| 72 | `heldout-history/` | `results/heldout-history/heldout-rerun.sh --build rerun optional`, `results/heldout-history/heldout-rerun.sh rerun` (and `optional`), then `python3 results/heldout-history/heldout_history.py`; points and flags in `points.tsv`; progress: `results/heldout-history/heldout-status.sh` |
| 73 | `decode-next/engram.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-engram/`, `trace/` (regenerated), `trace/compare-step69.txt` | the gate commands in the file (as step 71's); the probe was a temporary patch (getrusage around `fetch_engram_rows`); traces: the commands in `trace/README.txt`, `gzip -9 -n`, `scripts/trace_compare.py` against `git show b3b107b:ds-completion/results/trace/<name>.json.gz` |
| 74 | `decode-next/indexer.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-indexer/` | the gate commands in the file (step 73's against step 73's outputs); the trace split per layer: `results/trace/cpp-matrix.json.gz`, `attn.indexer L<n>` spans |
| 75 | `decode-next/fp8-multi-step75.txt`, `trace/` (regenerated), `trace/compare-step73.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,4,5,6,8` (and `--bitwise`) with each variant as a temporary change to `hip/fp8_gemv.hip`; register counts: `hipcc ... -c hip/fp8_gemv.hip -Rpass-analysis=kernel-resource-usage`; traces: the commands in `trace/README.txt`, compared with `git show 8138b22:ds-completion/results/trace/<name>.json.gz` |
| 76 | `decode-next/attn-split.txt`, `decode-next/attn-split.patch` | on 4b1a52c: `git apply results/decode-next/attn-split.patch`, `make HIP=1`, `bin-hip/bench-xdie 6`, `bin-hip/bench-xdie 2`, `bin-hip/bench-xdie proj`; end to end `scripts/spec-measure.sh`'s command for one prompt with and without `--attn-split`, and the same with `--trace` compared by `scripts/trace_compare.py` |
| 77 | `cpu-decode/warm.txt` (part 1) | `make HIP=1 bin-hip/bench-cpu-moe`, `bin-hip/bench-cpu-moe -m $M -t 16 --check-layers "" --warm 1,2,6 --steps 20` |
| 78 | `cpu-decode/decode-bench.txt`, `cpu-decode/call.txt`, `cpu-decode/warm.txt` (part 2), `cpu-decode/private-copies-ab.txt`, `cpu-decode/call-gap.txt`, `cpu-decode/private-copies.patch` | `make bin/bench-fp4-decode`; `OMP_PROC_BIND=spread OMP_PLACES=cores bin/bench-fp4-decode 16 5120` and `... 16 2304`; `bin/bench-fp4-decode call 16` and `call 16 4k`; `call 16 2m 1500` / `-1500` and with `OMP_WAIT_POLICY=passive` / `active`; `bin-hip/bench-cpu-moe ... --warm 1,2,6` and `--warm 1 --gap-us 1500`; the A/B: `git apply results/cpu-decode/private-copies.patch` on e1b9fbc, the command in the file for cpp-matrix and tcp-udp with and without it, after `cat` of the model files to warm the page cache |
| 79 | `cpu-decode/counters.txt`, `cpu-decode/aligned16-ab.txt`, `cpu-decode/aligned16.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-aligned16/` | `git apply results/cpu-decode/aligned16.patch` on 3f1ca2c, `make bin/bench-fp4-decode`; `bin/bench-fp4-decode call 16 2m 0 <once|rewrite|copy|single|aligned16> 2000`, and under `perf stat -x, -e cycles:u,instructions:u,<4 events>` with 2000 and 0 calls (groups in the file); the A/B and `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` with the patch built |
| 80 | `cpu-decode/act-single.txt`, `cpu-decode/act-single-ab.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-actsingle/` | `bin/bench-fp4-decode act`; `bin/bench-fp4-decode call 16 2m <0|1500> <once|rewrite|single> <2000|1000>`; `bin-hip/bench-cpu-moe -m $M -t 16 --batch 1` (3, 8) and `--warm 1,2,6 --steps 20`; the A/B against a build of 3f1ca2c; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 81 | `cpu-decode/after-gap.txt`, `cpu-decode/spin-wait.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-spinwait/` | `bin/bench-fp4-decode call 16 2m <0|1500|-1500> <single|worker> 1000`; C6 entries from `/sys/devices/system/cpu/cpu*/cpuidle/state3/usage`, C6 off by writing 1 to `state3/disable` (root); end to end with `HSA_ENABLE_INTERRUPT=0`, and with `git apply results/cpu-decode/spin-wait.patch` and `DS_SPIN_WAIT=1` (alternating, and `DS_SPIN_WAIT=1 CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`) |
| 82 | `cpu-decode/sched.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-taskpair/` | `bin/bench-fp4-decode sched 0` and `sched 1500` (`SCHEDS=a,b` for some); the real kernel: `bin-hip/bench-cpu-moe -m $M -t 16 --check-layers "" --warm 1,2,6 --steps 20` and `--decode 1,2,6 --steps 10 --layers 4` with `kTaskPair` 1 (or the old static loops) vs 2 in `src/cpu_moe.cpp`; `--batch 1` (3, 8); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 83 | `cpu-decode/layout.txt` | `make bin/bench-fp4-decode`, `bin/bench-fp4-decode layout 1500`, and with `BENCH_PF64=0` (4096, 16384, 32768) for the prefetch distance |
| 84 | `decode-next/copy-engine.txt`, `trace/` (regenerated), `trace/compare-step74.txt`, `trace/critpath.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-kcopy/` | traces: the command in `trace/README.txt` (and `--cache-loads 0` without `--cache-wait` for the no-load case), `scripts/trace_compare.py`, `scripts/trace_critpath.py`, the copy distribution from the `ffn.cpu_rows_copy` spans (the file's method); the variants as temporary changes in `hip/rt.cpp` / `hip/expert_loader.cpp`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 85 | `decode-next/load-gate.txt`, `decode-next/load-gate.patch` | `git apply results/decode-next/load-gate.patch` on ecb5af9, `make HIP=1`; the trace command of `trace/README.txt` with `DS_LOAD_GATE=1` (and `DS_LOAD_PER_LAYER=2` / `3`) or `DS_LOAD_GATE=2`; `scripts/trace_compare.py` against step 84's traces |
| 86 | `decode-next/draft.txt`, `trace/` (regenerated), `trace/compare-step84.txt`, `trace/critpath.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-homefixed/` | traces: the command in `trace/README.txt` (and `--cache-loads 0` without `--cache-wait`; step 84's build for the slow drafts), `scripts/trace_critpath.py` (its cycle section), `scripts/trace_compare.py`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 87 | `decode-next/draft-route.txt`, `decode-next/draft-route/`, `decode-next/verify-tail.txt` | `DS_DRAFT_ROUTE_LOG=<name>.log` with the held-out command for each of the 8 prompts, `results/decode-next/draft_route_stats.py <logs>`; the tail: the trace command of `trace/README.txt` (spans `tail: ...`) |
| 88 | `decode-next/gpu-argmax.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-gpuargmax/` | the trace command of `trace/README.txt` before and after (spans `tail: ...`); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 89 | `trace/` (regenerated), `trace/compare-step86.txt`, `trace/critpath.txt` | the command in `trace/README.txt` for tcp-udp and cpp-matrix, `gzip -9 -n`; `scripts/trace_compare.py` against `git show b3d3b24:ds-completion/results/trace/<name>.json.gz`; `scripts/trace_critpath.py` on both |
| 90 | `decode-next/sclk-levels.txt`, `spec-measure-cacheloads800cacheperiod8/` (sclk 2), `spec-measure-sclk6-cacheloads800cacheperiod8-sclkcmp/`, `spec-measure-sclk8-cacheloads800cacheperiod8/` | after a reboot `cat` the model files to `/dev/null`; at each level (set on all four cards) `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (level 6's copied, the tracked directory checked out again) |
| 91 | `cpu-decode/layout-live.txt`, `cpu-decode/layout-live.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8cpulayoutall/` | on 880b0b6: `git apply results/cpu-decode/layout-live.patch`, `make && make HIP=1`; `bin/bench-fp4-decode layout` (bits); the trace command of `trace/README.txt` with `--cpu-layout even`, `--cpu-layout odd`, `--cpu-layout even --cpu-layout-control` for cpp-matrix and tcp-udp, `scripts/trace_cpu_copy.py` (created by the patch); `CACHE="--cache-loads 800 --cache-period 8 --cpu-layout all" scripts/spec-measure.sh` (memory: /proc/meminfo sampled during it) |
| 92 | `decode-next/draft-vocab.txt`, `decode-next/draft-vocab/` | `DS_DRAFT_TOKEN_LOG=results/decode-next/draft-vocab/<name>.log` with `scripts/spec-measure.sh`'s command for each held-out prompt and long-proof; `~/venvs/ds-ref/bin/python results/decode-next/draft_vocab_sim.py results/decode-next/draft-vocab results/decode-next/draft-vocab` (calibration: `data/routing/study/*.bin`, step 13, not in git) |
| 93 | `decode-next/exit-abort.txt` | `scripts/spec-measure.sh`'s command for ja-email / py-lru in a loop; under `gdb -q -batch -ex "handle all nostop noprint pass" -ex "handle SIGABRT SIGSEGV SIGBUS stop print" -ex "run <args> > out 2> err < /dev/null" -ex "thread apply all bt 40" bin-hip/ds-completion` (the prompt by `-f`); ASan: `make HIP=1 BIN=bin-asan bin-asan/ds-completion CXXFLAGS="<CXXFLAGS with -O1 -fno-omit-frame-pointer -fsanitize=address>"`, run with `ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:halt_on_error=1:log_path=<prefix>` |
| 94 | `decode-next/verify-halves-trace.txt` | `git apply results/decode-next/verify-halves.patch` on a519a9d, `make HIP=1`; the trace command of `trace/README.txt` with `DS_VERIFY_HALVES=1` (and untraced) for cpp-matrix and tcp-udp; `results/decode-next/trace_overlap.py` and `results/decode-next/trace_overlap.py --by-size` on those traces and step 88's |
| 95 | `dist/llama-cpp-fork.txt`, `docs/llama-cpp-upstream.md` | the fork from a fresh clone: the commands in `docs/llama-cpp-upstream.md`, then `make FORK=<clone> BIN=bin-fresh && make HIP=1 FORK=<clone> BIN=bin-fresh-hip`, `bin-fresh/fp8pack` (cmp with `dense-fp8.gguf`), the held-out command for cpp-matrix and tcp-udp; `llama-completion -p "The three largest cities in Ukraine are" -n 32 --temp 0 --top-k 1 -t 16 -no-cnv` with the fork, the fresh build and upstream `a4d4f2fd4`; the type-43 tensors with each version's `gguf-py` (`GGUFReader`) |
| 96 | `decode-next/exit-abort-step96.txt` | `bin-hip/bench-loader-exit 1500` and `bin-hip/bench-loader-exit 1500 serial`; the held-out command for ja-email and py-lru (the `# expert loader: ... unregistered` line); the handler: the same command with `-n 2000`, `kill -ABRT <pid>` during generation |
| 97 | `decode-next/local-review.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-gather/` | the same at step 97's commit; the second-copy path: `DS_EARLY_ROWS=1` with the held-out command (the `# short forwards:` line); the tables: step 88's traces (`results/trace/`) |
| 98 | `decode-next/local-review.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-compressor/` | `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 99 | `decode-next/review-leftovers.txt`, `decode-next/{hc-mix,compressor-chunk,gather-streams}.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-cmpchunk/` | each patch on 5f21709 (`git apply`): `bin-hip/bench-kernels` (hc lines); `scripts/spec-measure.sh` with the tuned cache; the prompt gate's `--canon` runs against `results/prefill/gate/p*-base-canon.out`; the held-out command for cpp-matrix, tcp-udp, ja-email |
| 100 | `trace/` (`cpp-matrix.json.gz`, `tcp-udp.json.gz`, `compare-step88.txt`, `critpath.txt`) | the command in `trace/README.txt` for both prompts; `scripts/trace_compare.py <step 88's trace> <new>` (step 88's: `git show 2509593:ds-completion/results/trace/<name>.json.gz`); `scripts/trace_critpath.py <trace>` |
| 101 | `decode-next/row-stride.txt`, `decode-next/fp8-probe.txt` (+ `.patch`), `spec-measure-sclk6-cacheloads800cacheperiod8-rowpad/` | `bin-hip/bench-row-stride`; `[DS_FP8_PAD=0] bin-hip/bench-fp8 $F8 --ms 1,2,3,4,6`, `--bitwise`, `--gemm --ms 13,64`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`; the probe: the patch on 58341df, the loop in the file's header |
| 102 | `decode-next/wave-slots.txt`, `decode-next/pool-pad.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-stall/` | `bin-hip/bench-wave-slots [64\|256] [split]`; `[DS_STALL_SPLIT=0] bin-hip/bench-kernels --stride`, `--bitwise`; the launch count: the build lines in `results/decode-next/hip_launch_waves_shim.cpp`, then `results/decode-next/launch_waves.py LOG bin-hip/ds-completion`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`; the slots: the patch, `DS_T_POOLPAD=256` on py-lru and cpp-matrix |
| 103 | `decode-next/wave-rounds.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,6 --waves --reps 500` |
| 104 | `decode-next/fp8-virt.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-virt/` | `bin-hip/bench-fp8 $F8 --ms 2,3,4,5,6,7,8 --virt`; `[DS_FP8_VIRT=0] bin-hip/bench-fp8 $F8 --ms 2,3,4,5,6,8`; `--bitwise --ms 2,3,4,5,6,7,8`; `--gemm --ms 13,64`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 105 | `trace/` (`cpp-matrix.json.gz`, `tcp-udp.json.gz`, `compare-step98.txt`, `critpath.txt`), `decode-next/virt-onoff.txt` | the command in `trace/README.txt` for both prompts, and again with `DS_FP8_VIRT=0`; `scripts/trace_compare.py <old> <new>` (step 98's traces: `git show cb5b695:ds-completion/results/trace/<name>.json.gz`); `scripts/trace_critpath.py <trace>`; untraced: the held-out command on cpp-matrix and geometry with and without `DS_FP8_VIRT=0` |
| 106 | `decode-next/draft-ahead.txt` | `python3 results/decode-next/draft_ahead_sim.py` (over step 92's `decode-next/draft-vocab/` logs) |
| 107 | `decode-next/draft2.txt`, `decode-next/draft2/` | with a build of commit 012bf44 (the switch was removed in step 133): `DS_DRAFT2_LOG=results/decode-next/draft2/<name>.log` with `scripts/spec-measure.sh`'s command for each held-out prompt; `python3 results/decode-next/draft2_sim.py results/decode-next/draft2 results/spec-measure-sclk6-cacheloads800cacheperiod8-virt` |
| 108 | `chat/` | the commands in `chat/README.txt`: a piped three-turn `scripts/chat.sh` session, then `ds-completion -f chat/whole.txt` with the same options |
| 109 | `chat/think/` | the commands in `chat/think/README.txt`: `scripts/chat.sh --think` sessions from the `.q` files, `results/chat/chat_ref_prompts.py` (needs the reference's `encoding.py`), then `ds-completion -f` on each `-ref<k>.txt` |
| 110 | (`--layers`, `--teacher` and the stream files were removed in step 179: `split/teacher.txt` needs a build of `519ffd4` or earlier) `split/sha256.txt`, `split/teacher.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-shower/`, `-tomb-baseline/` | on each machine the `sha256sum` line in `split/sha256.txt` and `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (the second machine: the repo, `~/forks/llama.cpp` built with `-DBUILD_SHARED_LIBS=OFF`, the fp8 file, the same `~/llms` paths); the three `--teacher` commands in `split/teacher.txt` |
| 111 | `split/two-machines/` | with a build of `efa4e03` or earlier (`--remote` and `--serve` were removed in step 178; the script is now `results/split/two-machines/split-measure.sh`): `BACK=oleksandr@192.168.2.2 scripts/split-measure.sh` (the back machine prepared as in step 110, the same commit built there, ssh without a prompt); the one-machine side: `spec-measure-sclk6-cacheloads800cacheperiod8-tomb-baseline/` of step 110. On one machine: the back with `--layers 20-39 --gpu 2 --moe-dies 2,3 --dspark $D --serve 7001`, the front with `--layers 0-19 --gpu 0 --moe-dies 0,1 --remote 127.0.0.1:7001 --spec` |
| 112 | `trace/two-machines/` | as step 111's row, a build of `efa4e03` or earlier (the merge script is now `results/trace/two-machines/trace_merge.py`): `TRACE=1 PROMPTS="cpp-matrix tcp-udp" scripts/split-measure.sh` (both halves with `--trace`, then `scripts/trace_merge.py FRONT BACK -o MERGED`; the per-step table: the spans named in README step 112, medians over the 6-token and 2-token verify steps) |
| 113 | `trace/two-machines/` (collected again) | the command of step 112; the session check: `scripts/chat.sh --think --effort 75 --cache-wait -n 45 < results/chat/think/th45.q`, and the same questions to a front (`--layers 0-19 --remote HOST:7001 --spec --think --effort 75 -i -n 45`) |
| 114 | - (no results file) | the checks named in README step 114: the held-out command of `scripts/spec-measure.sh` per prompt, with `--profile`, the session of step 113, `--spec` against plain without `--canon`, `--prefill-chunk 512` against `7` on a long prompt, `PROMPTS="cpp-matrix summary-json" scripts/split-measure.sh` |
| 115 | `record/first.txt`, `record/fast.txt` | the command in each file's header; `first.txt`: twice in a row (run 1 cold: the model out of the page cache; any run after a reboot) |
| 116 | `record/sclk.txt` | the reader: a directory of made-up `card<N>/device/{vendor,power_dpm_force_performance_level,pp_dpm_sclk}` files and a three-line program printing `json_array(gpu_sclk(dir))` from `apps/record.h`; the run: the command of `record/fast.txt` with `--record`, while a loop prints the starred line of each card's `pp_dpm_sclk` every 0.1 s |
| 117 | `record/read-bytes.txt` | the two commands in the file with `--record`; the evicted file: `python3 -c "import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)" ~/llms/ds-1017-fp8/dense-fp8.gguf`; the cold prompts need the engram table out of the page cache (after a reboot, before the model is read through) |
| 118 | `record/spec.txt` | the three commands named in the file, each with `--record` |
| 119 | `record/experts.txt`, `record/example.json` (its header: `record/README.txt`) | the three commands named in `experts.txt`, each with `--record`; `example.json` is the third one's record |
| 120 | `record/knobs.txt` | the command in the file's header, without and with the knobs it names |
| 121 | `record/timing-options.txt` | the two commands in the file with `--record`, plain and with the switches and options it names |
| 122 | `record/loads-by-die.txt` | the command of `record/fast.txt` with `--record`, with this build and with a build that has the one-line change in the file's header |
| 123 | `draft-pool/in-pool.txt` | the command in the file with a build of 566de25 and with this one; the ids: each token log without its time columns (`awk -F'\|' '{split($1,a," "); print a[1], a[2], "\|" $2 "\|" $3}'`), compared with `cmp`; then the command of `record/fast.txt` |
| 124 | `draft-pool/keep/` (`summary.txt`, `table.txt`, per configuration each prompt's `.keep` set and record) | `results/draft-pool/draft-keep-sweep.sh` (needs step 87's `decode-next/draft-route/*.log`: `*.log` is not in git; made with `DS_DRAFT_ROUTE_LOG` and the held-out command), then `results/draft-pool/draft_keep_table.py` |
| 125 | `draft-pool/leave-out/` (`summary.txt`, `table.txt`, the records) | `results/draft-pool/draft-leave-out.sh` (needs step 124's `draft-pool/keep/loo-seen/*.keep`), then `results/draft-pool/draft_leave_out_table.py`; left out against masked: the first command of `draft-pool/in-pool.txt` with `DS_DRAFT_KEEP=results/draft-pool/keep/loo-seen/geometry.keep`, with and without `DS_DRAFT_KEEP_RESIDENT=1`, and `DS_DRAFT_ROUTE_LOG` |
| 126 | `record/output-hash.txt` | the five commands in the file; the recomputation: FNV-1a 64 (`apps/record.h`) over the third `\|` field of each `DS_DRAFT_TOKEN_LOG` line, after the first printed id and without the last id |
| 127 | `record/label.txt` | the commands in the file; the scripts' count by kind: read from each file's header, `git log --oneline -- scripts/<file>` for its commits |
| 128 | `record/reader.txt` | the commands in the file (`scripts/records.py` over `draft-pool/leave-out/*.json`; step 126's three records are made by the commands of `record/output-hash.txt`) |
| 129 | `record/runner/` (`leave-out.matrix`, `leave-out/` with its `matrix.txt` and 8 records, `runner.txt`) | `scripts/run_matrix.py results/record/runner/leave-out.matrix --prompts tcp-udp,geometry --out <a new directory>` (needs step 124's `draft-pool/keep/loo-seen/*.keep`), then the `scripts/records.py` command in `runner.txt` |
| 130 | (no results file: scripts moved) | `git show --stat` of the step's commit lists every move; the checks: `bash -n` on each `.sh`, `python3 -m py_compile` on each `.py`, `python3 results/draft-pool/draft_keep_table.py \| diff - <(tail -n +4 results/draft-pool/keep/table.txt)` and the same for `draft_leave_out_table.py` (`tail -n +2`, `leave-out/table.txt`) |
| 131 | `cleanup/run-split-1.txt` | the commands in the file with a build of e4a04ca and with this one; the records compared without their times |
| 132 | `cleanup/run-split-2.txt` | the commands in the file with a build of 60433e8 and with this one |
| 133 | `cleanup/study-draft2.txt` | the commands in the file with a build of 012bf44 and with this one |
| 134 | `cleanup/study-load-test.txt` | the commands in the file with a build of def1a2a and with this one |
| 135 | `cleanup/study-route-ahead.txt` | the commands in the file with a build of 8798740 and with this one |
| 136 | `cleanup/rt-split-trace.txt` | the commands in the file with a build of 6425e03 and with this one |
| 137 | `cleanup/rt-split-peer.txt` | the three commands in the file (the back first; the front once the back prints "waiting for the front"), with a build of 35561d0 and with this one |
| 138 | `cleanup/heldout.txt`, `cleanup/heldout/` (its `matrix.txt`, 16 records), `record/reference/heldout-output.tsv` | `scripts/run_matrix.py results/cleanup/heldout.matrix --keep-out --out <a new directory>`, `cmp` of each `.out` with `cache-test/plain-canon-static/<prompt>.out`, then the two `scripts/records.py` commands in `heldout.txt` |
| 139 | `cleanup/layout.txt` | `touch apps/* tests/* bench/*`, `make`, `make HIP=1`; `scripts/run_matrix.py results/cleanup/heldout.matrix --reps 1 --out <a new directory>`, each record against `record/reference/heldout-output.tsv`; the `scripts/records.py` command of the file |
| 140 | `rocprof/first-run.txt` | `sudo apt install rocprofiler-sdk6.3.4`, then the three commands in the file; its tables from the `*_stats.csv` files and `k_kernel_trace.csv` as `docs/rocprof.md` describes (the traces themselves, ~160 MB, are not kept) |
| 141 | `rocprof/counters.txt` | `sudo apt install hsa-amd-aqlprofile6.3.4`; the probe and the four model runs as the file gives them; the table: per kernel the mean `Counter_Value` of `p_counter_collection.csv` over the launches from the first `gemv_partial_kernel<unsigned short, ...>` on |
| 142 | `rocprof/calibration.txt` | the five commands in the file; per tensor the median `Counter_Value` of the `fp8_gemv_kernel` launches with its instance and grid size |
| 143 | `rocprof/busy.txt`, `rocprof/busy.py` | the `--kernel-trace` run of step 140, then `results/rocprof/busy.py D/k_kernel_trace.csv` |
| 144 | `rocprof/busy-fast.txt` | the two commands in the file, then `results/rocprof/busy.py D/k_kernel_trace.csv 300` (the sums by kind: the gaps of 100 us and more on die 1 grouped by the kernel that follows) |
| 145 | `rocprof/gap-calls.txt`, `rocprof/gap_calls.py` | the `--runtime-trace` run of step 140, then the two commands in the file |
| 146 | `spec-oracle/oracle.txt`, `spec-oracle/runs/`, `spec-oracle/oracle.matrix`, `spec-oracle/oracle.py` | `scripts/run_matrix.py results/spec-oracle/oracle.matrix --out <a new directory>` (the matrix names its log files under `results/spec-oracle/runs/`: change both), then `results/spec-oracle/oracle.py <the directory>` |
| 147 | `usability/think-show.txt` | the commands in the file, with a build of 0bcb0d2 and with this one; the terminal runs through `script -qec "stty cols N; ..." FILE` |
| 148 | `sampler/test-sampler.txt` | `make bin/test-sampler && bin/test-sampler` (CPU only, a few seconds) |
| 149 | `sampler/plain.txt`, `sampler/plain/` (records), `sampler/plain.matrix` | `scripts/run_matrix.py results/sampler/plain.matrix --prompts tcp-udp,cpp-matrix,geometry` (18 runs, 14 min), then the `scripts/records.py` command in the file; the repeat by hand as the file says |
| 150 | `sampler/spec.txt`, `sampler/spec/`, `sampler/spec-heldout/` (records, spec logs), `sampler/spec.matrix`, `sampler/spec-heldout.matrix`, `sampler/spec_summary.py` | `scripts/run_matrix.py results/sampler/spec.matrix --prompts tcp-udp,cpp-matrix,geometry` (15 runs, ~12 min; needs step 149's records), `scripts/run_matrix.py results/sampler/spec-heldout.matrix` (16 runs, ~12 min), then `results/sampler/spec_summary.py` |
| 151 | `sampler/step151/summary.txt`, `summary.py`, `spec.matrix`, `spec/`, `spec-heldout/` (records, spec logs), `where.txt`, `phases.cpp`, `test-sampler.txt` | `scripts/run_matrix.py results/sampler/step151/spec.matrix --prompts tcp-udp,cpp-matrix,geometry` (15 runs, ~13 min), `scripts/run_matrix.py results/sampler/spec-heldout.matrix --out results/sampler/step151/spec-heldout` (16 runs, ~14 min), then `results/sampler/step151/summary.py` (needs step 150's records); `where.txt`: by hand as its header says (traces with `--trace`, a temporary clock in `generate_spec`, `phases.cpp`); `bin/test-sampler` |
| 152 | `requests/requests_test.txt`, `requests/requests_test-plain.txt`, `requests/requests_test.py` | `python3 results/requests/requests_test.py` and the same with `--no-spec` (~4 min each, 2 of them the load; needs the stored outputs of steps 138 and 150) |
| 153 | `server/test_server.txt`, `server/endgame.txt`, `server/endgame-message.json.gz`, `server/samples/`, `server/make_samples.py` | `python3 server/test_server.py` (no model, seconds); the samples: `python3 results/server/make_samples.py`; `endgame.txt`: by hand, the two commands in its header (the server with `--dump`, then the benchmark's script from `~/projects/endgame` with `anthropic` 0.125.0; ~21 min with the load) |
| 154 | `requests/requests_test.txt`, `requests/requests_test-plain.txt` (again, with the cancel checks), `server/test_server.txt` (again), `server/cancel.txt` | `python3 results/requests/requests_test.py` and with `--no-spec` (~5 min each); `python3 server/test_server.py`; `cancel.txt`: by hand, the commands in its header |
| 155 | `trace/` (`cpp-matrix.json.gz`, `tcp-udp.json.gz`, `compare-step104.txt`, `critpath.txt`) | the page cache warm (`cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null` after a reboot), then the command in `trace/README.txt` for both prompts, `gzip -9 -n`; `scripts/trace_compare.py <old> <new>` (step 104's traces: `git show f99081b:ds-completion/results/trace/<name>.json.gz`); `scripts/trace_critpath.py <trace>` |
| 157 | `server/test_server.txt` (again, with a tool loop's second request), `server/pi.txt` | `python3 server/test_server.py`; `pi.txt`: by hand, the commands in its header (pi built from `~/projects/pi`, `server/pi-models.json` as its `models.json`; ~2 min of model load, ~2 min of tasks) |
| 158 | `requests/requests_test.txt`, `requests/requests_test-plain.txt` (again, with the kept-context checks), `server/test_server.txt` (again), `server/pi-kept.txt` | `python3 results/requests/requests_test.py` and with `--no-spec` (~7 min each); `python3 server/test_server.py`; `pi-kept.txt`: by hand, the commands in the header of `server/pi.txt` |
| 160 | `requests/requests_test.txt`, `requests/requests_test-plain.txt` (again, with the saved-point checks), `server/test_server.txt` (again), `server/pi-saved.txt` | `python3 results/requests/requests_test.py` and with `--no-spec` (~10 min each); `python3 server/test_server.py`; `pi-saved.txt`: by hand, the commands in the header of `server/pi.txt`, three tasks |
| 161 | `server/test_server.txt` (again), `server/stream_check.py`, `server/stream.txt`, `server/pi-stream.txt` | `python3 server/test_server.py`; `results/server/stream_check.py http://127.0.0.1:PORT` against a running server (the model); `pi-stream.txt`: by hand, the commands in the header of `server/pi.txt` with `--mode json` |
| 163 | `dist/trunk-trial.txt`, `dist/trunk-trial.patch` | upstream llama.cpp at `e626db7ca` built with the fork's two `cmake` lines; `git apply ds-completion/results/dist/trunk-trial.patch` on 40f345e; `make FORK=<upstream> BIN=bin-trunk && make HIP=1 FORK=<upstream> BIN=bin-trunk-hip`; the held-out command per prompt (`scripts/spec-measure.sh`'s, with `--cache-loads 800 --cache-period 8`, without `--spec-log`), `cmp` with `cache-test/plain-canon-static/<name>.out`; `bin-trunk/fp8pack` and `cmp` with `dense-fp8.gguf` |
| 164 | `requests/requests_test.txt`, `requests/requests_test-plain.txt` (again, with the checks of `ctx`, `common`, `how` and of the engine's `# context:` lines) | `python3 results/requests/requests_test.py` and with `--no-spec` (~10 min each) |
| 165 | `server/test_server.txt` (again, with the metrics and health checks), `server/metrics.txt`, `server/metrics_check.py` | `python3 server/test_server.py` (no model); `python3 results/server/metrics_check.py SERVER_LOG` (the model: ~1 min of load, ~1 min of requests), its output and the log's `# server` and `# context` lines |
| 166 | `server/test_server.txt` (again, with the `--log` and `--dump` checks), `server/log.txt`, `server/log_check.py` | `python3 server/test_server.py` (no model); `python3 results/server/log_check.py SERVER_LOG` (the model: ~1 min of load, ~1 min of requests), its output and the log's `# server` and `# context` lines |
| 168 | `dist/trunk.txt` | upstream llama.cpp at `e626db7ca` built with QUICKSTART's two `cmake` lines; `make && make HIP=1` (the pin) and `make FORK=<upstream> BIN=bin-trunk && make HIP=1 FORK=<upstream> BIN=bin-trunk-hip`; per build `ds-test -m $M -g data/golden/hello-fp32` with and without `--runtime $F8`, each also with `--e2e <file>`, the files compared with `cmp` (the build before the change: `git stash` or a checkout of 9a57d22); `scripts/run_matrix.py results/cleanup/heldout.matrix --keep-out --out <dir>` per build (for trunk a copy of the matrix with a `bin` line), outputs against `results/cache-test/plain-canon-static/` with `cmp`; `bin-trunk/fp8pack`; the commands are in the file (~75 min, most of it the three held-out runs) |
| 170 | `dist/make-groups.txt` | `make -n -B [HIP=1] [product\|tests\|tools]` for the lists; `time make -j1 [HIP=1] BIN=<new dir> [product]` for the times; `cmp` of the programs of two directories, and of one object built twice to one path and once to another (the commands are in the file) |
| 171 | `dist/forward-split.txt` | `make && make HIP=1`; `ds-test -m $M -g data/golden/hello-fp32` with and without `--runtime $F8`, each also with `--e2e <file>`, against the same of the commit before (`cmp`; the tables row by row without the ms column); `scripts/run_matrix.py results/cleanup/heldout.matrix --prompts cpp-matrix,ja-email,summary-json --reps 1 --keep-out --out <dir>` and `cmp` with `results/cache-test/plain-canon-static/`; `python3 server/test_server.py` |
| 172 | `dist/fakequant-split.txt` | `make && make HIP=1`; `bin/test-fakequant data/fq`; the gates and the three held-out prompts as step 171's row; `bin/fp8pack -m $M -o <file>` and `cmp` with `dense-fp8.gguf` |
| 173 | `dist/ds-ref.txt`, `requests/requests_test.txt` (again) | `make && make HIP=1`; `bin/ds-ref -m $M -p "The three largest cities in Ukraine are" -n 24 --record <file>` against the same with the CPU `ds-completion` of 1c121ba (`cmp`, the records' `output.hash`); the gates and the three held-out prompts as step 171's row; two prompts without `--spec` (a copy of the held-out matrix with `base --rt $F8 --placement placements/study30.txt --canon`); `python3 results/requests/requests_test.py` (~10 min); `python3 server/test_server.py`; `nm -C <program> \| grep -c ' Engine::'` |
| 174 | `gates-long.txt` | the six `ds-test` commands in the file (CPU: seconds to 3 min; the runtime at 20,000 tokens: ~31 min per layer, ~8 min end to end), each table against the stored one named beside it; `cmp` of the logits and the top-k dump with `data/ds/readme20k-rt.f32` and `data/ds/readme20k-topk.bin`; `~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/readme20k-s.pt <file>` |
| 175 | `dist/upstream.txt` | `~/forks/llama.cpp` at upstream `e626db7ca`, built with QUICKSTART's two `cmake` lines; `make clean && make && make HIP=1`; then the commands in the file: the unit tests, `fp8pack` and `cmp`, the seven `ds-test` tables and the two `--e2e` runs, `bin/ds-ref`, `scripts/run_matrix.py results/cleanup/heldout.matrix --keep-out --out <dir>` (twice per prompt) and `cmp` with `results/cache-test/plain-canon-static/`, `results/requests/requests_test.py`, `server/test_server.py` (~90 min in all) |
| 176 | `dist/llama-cpp-dir.txt` | llama.cpp at `e626db7ca` in `~/projects/llama.cpp`, built with QUICKSTART's two `cmake` lines; `make clean && make && make HIP=1`; the commands in the file (step 175's without the 20,000-token set, ~50 min); `~/venvs/ds-ref/bin/python ref/check_weights.py --layers 0 --experts 0` (7 min, 2.9 GB from Hugging Face) |
| 177 | `dist/review-177.txt` | by hand: the comparisons named in the file; `make && make HIP=1`; the two programs without arguments; the four commands of its part 4 and `server/test_server.py` |
| 178 | `dist/two-machines-link.txt` | `make && make HIP=1`; a second checkout of `efa4e03` with `make HIP=1 tests` for the base `ds-test`; then the commands in the file (~55 min, 42 of them the held-out prompts twice): `ds-test --runtime` per layer and `--e2e` on both builds (`cmp`; the tables without the ms column), `bin/ds-ref --record`, the three `--teacher` runs, `scripts/run_matrix.py results/cleanup/heldout.matrix --reps 2 --keep-out --out <dir>` and `cmp` with `results/cache-test/plain-canon-static/`, a 60-token `--trace` and `scripts/trace_critpath.py`, `results/requests/requests_test.py`, `server/test_server.py`. The clock: sampled with `grep '\*' /sys/class/drm/card0/device/pp_dpm_sclk` four times a second during a run |
| 179 | `dist/layer-range.txt` | before the change: `make HIP=1` at `519ffd4` and copies of `bin-hip/ds-test` and `bin-hip/ds-completion` kept aside (the base); after it `make && make HIP=1`; then the commands in the file (~60 min, 41 of them the held-out prompts twice): the three `ds-test --runtime` tables and `--e2e` on both builds, `bin/ds-ref --record`, the long prompt on both programs three times (`head -c 9000 docs/steps.md` as the prompt file), `scripts/run_matrix.py results/cleanup/heldout.matrix --reps 2 --keep-out --out <dir>` and `cmp` with `results/cache-test/plain-canon-static/`, a 60-token `--trace` and `scripts/trace_critpath.py`, `results/requests/requests_test.py`, `server/test_server.py` |
| 180 | `dist/two-machines-last-read.txt` | the `grep` in the file; `make && make HIP=1`; `scripts/run_matrix.py results/cleanup/heldout.matrix --reps 1 --keep-out --out <dir>` and `cmp` of each output with `results/cache-test/plain-canon-static/<prompt>.out` (~20 min) |
| 181 | `dist/after-removal-gates.txt`, `dist/after-removal/` (`run.sh`, `cmp_table.py`, `cmp_session.py`, `cmp_hash.py`, `run-log.txt`) | `mkdir -p data/step181 && cp results/dist/after-removal/{run.sh,cmp_*.py} data/step181/ && data/step181/run.sh > data/step181/run.log 2>&1` (3 h 04 min: sessions 44, sampling 95, golden sets 45; `run.sh` finds the repository two directories above itself) |
| 182 | `dist/review-182.txt` | the two checks by program are described in the file; the readings were by hand (four read-only subagents); `make && make HIP=1`, `objdump -d` of the two programs against the previous build's, `python3 server/test_server.py` |
| 183 | `requests/requests_test.txt` (again, with the header cases), `requests/headers-before.txt` | `make HIP=1`; `python3 results/requests/requests_test.py` (~12 min); `python3 results/requests/requests_test.py --headers-only` (~80 s), and the same with `DS_BIN=<a ds-completion built at 0aa67be>` for the file of the program before; `python3 server/test_server.py` |
| 184 | `dist/testing-times.txt` | the commands in the file, each once; the loads from `--record` files of step 181's runs; `data/ref/<name>.pt`'s `variant` field with `~/venvs/ds-ref/bin/python` |
| 186 | `dist/check-unit.txt` | `scripts/check.py unit`; the negative control: one float of `tests/fq/fp8.f32` changed (`struct.pack` at offset 4000), the run, the file restored; `bin/test-sampler` timed before and after the change |
| 187 | `dist/check-default-logits.txt`, `tests/e2e/` (the cut golden sets, `expected.txt`) | `python3 ref/trim_golden.py data/golden/<set> tests/e2e/<set>` for the three sets (needs the whole sets under `data/golden/`); `scripts/check.py default --store` once, then `scripts/check.py default` (~2 min); the control: `hip/rt.cpp` line 981 `eps` -> `eps + 1e-3f`, the run, the file restored |
| 188 | `dist/check-default-outputs.txt`, `tests/outputs/expected.json` | `python3 tests/outputs_check.py --store` once (~4 min), then `scripts/check.py default` (~5.5 min); the control as step 187's; the session timed with four sets of options (the command in the file) |
| 189 | `register/registration.txt`, `register/reg_threads.cpp`, `register/reg_flags.cpp` | the `g++` line in the file for each program; `./reg_threads <shard 3> 6`, `./reg_flags <shard 4>`; the registration times of stored runs: `grep -rhoE 'expert tensors registered in [0-9.]+ s' results` and `load_s` of the records with `cache_wait`; `scripts/chat.sh --think --think-show all --effort 75 -n 20 < results/chat/think/th20.q` timed before and after; `scripts/check.py default` |
| 190 | `dist/check-speed-complete.txt`, `tests/speed/baseline.tsv`, `tests/e2e/readme20k-s/`, `tests/long-prompt.txt` | `scripts/check.py speed --store` (36 min; then `python3 tests/speed_check.py --out <the run's runs/> --store` after the limit was corrected); `python3 ref/trim_golden.py data/golden/readme20k-s tests/e2e/readme20k-s`; `head -c 9000 docs/steps.md > tests/long-prompt.txt` (at this commit's parent); `scripts/check.py complete` (~4 h; not yet run to its end) |
| 191 | `tests/e2e/expected.txt` (the entry `cpu/readme1100`) | `bin/ds-test -m $M -g tests/e2e/readme1100 --e2e <file>` twice, `sha256sum`; `~/venvs/ds-ref/bin/python ref/compare_logits.py data/ref/readme1100.pt <file>` |
| 192 | `tests/fq/` (two more rows) | `~/venvs/ds-ref/bin/python ref/make_fq_test.py`; `make bin/test-fakequant && bin/test-fakequant`; the controls: `sed 's/nearbyintf(v \/ q)/roundf(v \/ q)/' src/fakequant.cpp > x.cpp; g++ -O2 -std=c++17 -Isrc -o t tests/test_fakequant.cpp x.cpp; ./t` (and `1e-4f` to `1e-3f`) |
| 193 | (no results file: the checks are the result) | `scripts/check.py default`; `python3 tests/record_check.py fnv1a64:df516e7200ad14c8 bin-hip/ds-completion -m $M --rt $F8 --placement placements/study30.txt --canon -f tests/long-prompt.txt -n 40`; `ds-ref` itself: `git checkout bb42f51 && make bin/ds-ref` |
| 194 | (no results file) | `scripts/check.py default`; `bin-hip/ds-test -m $M -g data/golden/hello-fp32 --runtime $F8` from this commit and from `90b4613`, the last column cut (`awk '{NF=NF-1; print}'`), `cmp`; `bin/bench-cpu-moe -m $M --tokens 2 --check-layers 0,20` |
| 195 | (a move; no results file) | `scripts/check.py default`; `python3 tests/requests_test.py --headers-only`. Paths in rows above this one are those of their commits: `tests/e2e` etc. are `tests/data/...` now, `server/test_server.py` and `results/requests/requests_test.py` are in `tests/` |
| 196 | (a move; no results file) | `scripts/check.py default`. In rows above this one `hip/x` is `src/x` now |
| 167 | `server/serve.txt` | by hand: `DS_LOGS=<a directory> scripts/serve.sh`, one request with `curl` (the one of `docs/pi.md`, section 1), a minute's wait for `metrics.prom`, Ctrl-C; the file's parts are named in it |

## 7. Open items (to confirm)

- The Hugging Face revisions of the two Unsloth GGUFs and the draft's repository (the sha256
  manifest pins the bytes meanwhile). Settled since: llama.cpp is upstream at one commit
  (section 2; until step 167 the head of PR #30817, `b72050d03ae14bb3cb5138ecefab3f3f65157d31`).
- The second machine (`shower`) is as it was at step 167: llama.cpp in `~/forks/llama.cpp` at the
  PR's commit, and an older build of this repository. No program uses two machines since step
  178; as a second machine for A/B runs of one-machine builds it needs the checkout and build of
  QUICKSTART and this repository at the commit under test, `make HIP=1`.
- Turn the hand-typed runs a write-up relies on (steps 4-26) into scripts.
- An archive of `data/routing/` next to any publication.
