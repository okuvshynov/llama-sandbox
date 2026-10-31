# Reproducing ds-completion

What is needed to rebuild the setup, re-run every experiment and regenerate every file in
`results/`. How the work was approached and why each step followed the last: [PROCESS.md](PROCESS.md).
What each step found: [README.md](README.md).

Items marked **(to confirm)** are not yet pinned down; they are listed at the end.

## 1. Machine

| | |
|---|---|
| computer | Mac Pro 2019 |
| CPU | Intel Xeon W-3245, 16 cores / 32 threads, AVX-512 (VNNI) |
| RAM | 755 GB (~112 GB/s measured, 16 threads) |
| GPUs | 2x Radeon Pro Vega II Duo = 4 dies (gfx906), 32 GB HBM2 each, PCIe 3.0 (MPX) |
| GPU links | Infinity Fabric (XGMI) within each card: dies 0-1 and 2-3 (bridge reseated; see the sibling repo `macpro-2019-vega-if`: ROCm crashes on cross-card runtime copies, so nothing here relies on peer copies) |

The RAM must hold the whole GGUF (474 GB) in the page cache next to the work: everything runs
from mmap, and the routed experts are read from it on every token.

## 2. Software

| | version |
|---|---|
| OS | Ubuntu 22.04 LTS, kernel `6.10.5-1-t2-jammy` (t2linux, for the Mac's T2 chip) |
| ROCm | 6.3.4 in `/opt/rocm-6.3.4` (HIP 6.3.42134; gfx906 is past its support window but works) |
| compiler | g++ 11.4.0; hipcc (AMD clang 18, from ROCm 6.3.4) |
| llama.cpp fork | `~/forks/llama.cpp`, branch `deepseek41`: commit `b72050d03` "model : add DeepSeek-V4.1-Flash (deepseek41)" (Unsloth) on top of upstream `797de63b2`, plus one uncommitted line in `ggml/src/ggml-cuda/vendors/hip.h` (`#define __ballot_sync(mask, var) __ballot(var)`, for the HIP build). CPU build in `build/` (static libraries). Used for ggml, GGUF reading, the tokenizer and the CPU baseline. **(to confirm: where the branch is published)** |
| Python (reference) | venv `~/venvs/ds-ref`, Python 3.10.12: `pip install -r ref/requirements.txt` (a `pip freeze`: torch 2.14.1+cpu, numpy 2.2.6, safetensors 0.8.0, sympy 1.14.0, tokenizers 0.23.3, huggingface_hub 2.3.0, ...) |

GPU clock: experiments state their sclk level (default here: level 2, 1204 MHz). It is set by
hand, as root, per card:

```bash
for c in /sys/class/drm/card[0-3]/device; do
  echo manual | sudo tee $c/power_dpm_force_performance_level
  echo 2 | sudo tee $c/pp_dpm_sclk          # level 0..8 (999..1730 MHz)
done
```

Warm page cache: the model is memory-mapped, so after a reboot the first run reads experts from
NVMe and is several times slower (README step 12). Measure only with the files cached, e.g. after
one full run.

## 3. Model files

| file | source | size |
|---|---|---|
| `~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-0000{1..9}-of-00009.gguf` | Unsloth's GGUF of deepseek-ai/DeepSeek-V4.1-Flash-1017 (GGUF metadata: `quantized_by Unsloth`, `repo_url https://huggingface.co/unsloth`, base model `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017`) **(to confirm: exact repo and revision)** | 474 GB |
| `~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf` | Unsloth's DSpark draft head GGUF (arch `dflash`, metadata as above) **(to confirm: exact repo and revision)** | 8.5 GB |
| `~/llms/ds-flash/` | the base DeepSeek-V4.1-Flash safetensors (HF commit `314113f8`); only for weight provenance checks. The GGUF matches the 1017 release (HF commit `1bae0e10`) instead, which `ref/check_weights.py --hf remote` reads by HTTP range requests (README "Which checkpoint") | 510 GB |
| `~/llms/ds-1017-fp8/dense-fp8.gguf` | made here: `bin/fp8pack -m $M -o ~/llms/ds-1017-fp8/dense-fp8.gguf` (lossless fp8 repack of the dense weights, ~20 s; README step 5) | 6.8 GB |

## 4. Build

```bash
# the fork: a CPU build with static libraries in build/ **(to confirm: its exact CMake options)**
cd ~/projects/llama-sandbox/ds-completion
make            # bin/: CPU programs (links the fork's static CPU libraries)
make HIP=1      # bin-hip/: the same plus the own GPU runtime and the GPU benchmarks
M=~/llms/UD-Q8_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf
F8=~/llms/ds-1017-fp8/dense-fp8.gguf
D=~/llms/dspark/dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf
```

Run from this directory (placement files are relative paths). How to run: [QUICKSTART.md](QUICKSTART.md).

## 5. Data that is not in git (`data/`, ~14 GB; ~115 GB with the 20k-token run of README step 35)

| directory | what | how to regenerate | cost |
|---|---|---|---|
| `data/ref/` | reference runs (`.pt`): the official model math on the CPU (`ref/ds_ref.py`) | `~/venvs/ds-ref/bin/python ref/ds_ref.py --prompt "..." --chat --n-decode 16 [--fp32] [--no-quant] --out data/ref/x.pt` (prompts: README step 2; the 20k-token run: step 35) | minutes to ~1 h (20k tokens: 42 min, ~500 GB of RAM, a 35 GB file) |
| `data/golden/` | per-layer golden activations and logits for the gates | `ref/export_golden.py data/ref/x.pt data/golden/x` | minutes |
| `data/ds/`, `data/llama/` | our and the fork's logits for the comparisons | `bin/ds-test ... --e2e`, `bin/llama-logits` (README steps 2-3) | minutes |
| `data/fq/` | fake-quant test vectors | `ref/make_fq_test.py` | seconds |
| `data/routing/study/` | routing logs of the 30 study prompts (plain): inputs of `placements/study30.txt` and the simulations | `scripts/routing-study.sh` (prompts in `prompts/routing-study.tsv`) | ~1 h |
| `data/routing/oracle/` | routing logs of the 2 oracle prompts | README step 15 | minutes |
| `data/routing/spec/` | routing logs of the 8 held-out prompts under `--spec` | `scripts/spec-routing.sh` | ~15 min |

The routing logs (48 MB) are cheap to keep and should be archived with any publication: the
placement and every simulation number depend on them, and regenerating them only reproduces
them exactly with the same binary and placement.

## 6. Which command produced which result

Every number in README comes from a file in `results/`. Speed numbers depend on the sclk level
and vary ~1-3% run to run; correctness results (identical outputs, 0 differing values) must
reproduce exactly. Up to step 26 many runs were typed by hand: the table names the program and
the README step gives the settings (prompt, tokens, threads, placement), but not always a
script; from step 27 on every result has a script. The commit that added a file:
`git log -- results/<file>`. Convention from now on: a result file starts with a `#` header
naming the command or script, the commit, the clock level and the date.

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
| 9 | `sclk/` | `scripts/sclk-sweep.sh`, `scripts/sclk-summary.py` (per step: the speed table at the levels) |
| 10-11 | `rt-prefill-*.txt`, `rt-cleanup-speed.txt` | `bin-hip/ds-completion --rt` |
| 13 | `routing-study*.txt` | `scripts/route_stats.py data/routing/study/*.bin [--cross prompts/routing-study.tsv]` |
| 15 | `oracle-placement.txt` | `scripts/route_stats.py --emit-placement`, then the runtime per placement |
| 18 | `threads-sweep.txt` | `bin-hip/ds-completion -t N` |
| 19 | `dspark-shadow.txt` | `bin-hip/ds-completion --dspark $D` (shadow mode) |
| 20 | `spec-phase1.txt`, `fp8-batch-bitwise.txt` | `--dspark --spec` vs `--dspark`; `bin-hip/bench-fp8 --ms ...` |
| 21-23 | `spec-draft-length.txt` | `--spec-n 1..5` |
| 24 | `bench-fp8-multi.txt` | `bin-hip/bench-fp8 --ms 1,2,...,8` |
| 25 | `spec-conf/`, `spec-conf-calibration.txt` | `--spec-log`, `scripts/spec_conf.py --cal ... --cost ...` |
| 26 | `spec-heldout/`, `spec-profiles.txt` | the held-out prompts (`spec-heldout/prompts.txt`) plain and `--spec`; `--spec-n 5 --profile` |
| 27 | `spec-sclk/` | `scripts/spec-sclk.sh` at sclk 0, 2, 8 |
| 28 | `canon-phase0.txt` | `bin/bench-cpu-moe -t 16 --tokens 14 --check-layers 2,20,39`, `bin-hip/bench-fp4-moe --layer 20 --reps 300` |
| 29 | `cache-sim.txt` | `scripts/spec-routing.sh`, then `scripts/cache_sim.py` |
| 30 | `bench-expert-load.txt`, `load-test/` | `bin-hip/bench-expert-load -m $M`, `scripts/load-test.sh` |
| 31-32 | `canon-gate.txt`, `canon-gate/` | `scripts/canon-gate.sh` (rerun for step 32) |
| 34 | `check-topk.txt` | `~/venvs/ds-ref/bin/python ref/check_topk.py` (runs `bin-hip/test-topk`) |
| 41 | `prefill/bench-cpu-moe-batch.txt` (rerun), `prefill/canon-e2e.txt` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --batch T`; `bin-hip/ds-completion ... --canon -f results/prefill/gate/pN.txt -n 32 [--profile]` |
| 40 | `prefill/bench-fp8-gemm-tuning.txt` | experimental builds (variants removed; described in the file); `prefill/bench-fp8-gemm.txt` rerun with the kept choice: `bin-hip/bench-fp8 $F8 --gemm --ms 64,256,512` |
| 39 | `prefill/bench-fp8-gemm.txt` | `bin-hip/bench-fp8 $F8 --gemm --ms 256 --sweep`, then `--gemm --ms 64,512` |
| 39 | `prefill/gemm-e2e.txt` | `bin-hip/ds-completion ... -f results/prefill/gate/pN.txt -n 32 [--canon] [--prefill-chunk 1] [--profile]` (commands in the file) |
| 38 | `prefill/overlap.txt` | `bin-hip/ds-completion ... -f results/prefill/gate/pN.txt -n 32 [--no-overlap] [--canon] [--profile]` (commands in the file) |
| 37 | `prefill/gate.txt`, `prefill/gate/` | `scripts/prefill-gate.sh <baseline ds-completion>` (baseline: a build of commit 73b6891, token-by-token prompt) |
| 37 | `prefill/profile.txt` | `bin-hip/ds-completion ... -f <prompt> -n 1 --profile --prefill-chunk N` (command and stages in the file) |
| 37 | `prefill/bench-cpu-moe-batch.txt` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --batch T` |
| 35 | `ref-readme20k.txt`, `ref-readme20k-scores.txt` | `ref/ds_ref.py --file prompts/ds-completion-readme-ddf9b43.md --prompt-tokens 20000 --n-decode 8 --max-seq-len 20480 --keep-last 32 [--scores]` (~42 min, ~500 GB peak), then `ref/export_golden.py` (only the `--scores` run is kept) |
| 35 | `rt-readme20k.txt`, `rt-readme20k-e2e.txt`, `compare-readme20k-ref-vs-rt.txt` | `bin-hip/ds-test -m $M -g data/golden/readme20k-s --runtime $F8 --dump-topk data/ds/readme20k-topk.bin` (~35 min); `... -g data/golden/readme20k-s --runtime $F8 --e2e data/ds/readme20k-rt.f32`; `ref/compare_logits.py data/ref/readme20k-s.pt data/ds/readme20k-rt.f32` (the e2e and compare files name `readme20k`, the first run without `--scores`: bit-identical in every tensor it has, since deleted) |
| 35 | `topk-ties-readme20k.txt` | `~/venvs/ds-ref/bin/python ref/check_topk_ties.py data/golden/readme20k-s data/ds/readme20k-topk.bin` |
| 34 | `long-context/` | `bin-hip/ds-completion ... --max-ctx N -p "$(cat README.md) ..."` (commands in `long-context/summary.txt`) |
| 33 | `verify-breakdown.txt` | `bin-hip/ds-completion ... --dspark $D --spec --spec-n 5 --canon --profile [--cache-loads 200 --cache-wait]` (command in the file's header), medians of the second half |
| 32 | `cache-test/`, `cache-test-sclk8/` | `scripts/cache-test.sh` (at sclk 8: `scripts/cache-test.sh "spec fast static" "spec canon 200"`); the CPU kernel at 1, 2, 6 experts: `bin/bench-cpu-moe -m $M -t 16 --tokens 8 --check-layers 2 -k N` |

## 7. Open items (to confirm)

- The fork branch `deepseek41` exists only locally: publish it (or record the upstream PR it
  came from) and commit the `hip.h` line, or the CPU baseline and tokenizer cannot be rebuilt.
- The exact Hugging Face repos and revisions of the two Unsloth GGUFs.
- The fork's CMake options for its CPU build.
- Turn the hand-typed runs a write-up relies on (steps 4-26) into scripts.
- An archive of `data/routing/` next to any publication.
