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
| perf (counters, step 79) | Ubuntu's `linux-tools-generic` (perf 5.15) run as `/usr/lib/linux-tools/5.15.0-202-generic/perf` (the `/usr/bin/perf` wrapper refuses the 6.10 kernel); user-space core events with `perf_event_paranoid` 2 |

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
| `data/routing/prefill/` | two long prompts (`head -c 24000 prompts/ds-completion-readme-ddf9b43.md`, `head -c 32000 hip/rt.cpp` at e3542ac) and their routing logs | `--log-routing` (README step 42) | ~7 min |

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
| 31-32 | `canon-gate.txt`, `canon-gate/` | `scripts/canon-gate.sh` (rerun for steps 32 and 45) |
| 34 | `check-topk.txt` | `~/venvs/ds-ref/bin/python ref/check_topk.py` (runs `bin-hip/test-topk`) |
| 101 | `decode-next/row-stride.txt`, `decode-next/fp8-probe.txt` (+ `.patch`), `spec-measure-sclk6-cacheloads800cacheperiod8-rowpad/` | `bin-hip/bench-row-stride`; `[DS_FP8_PAD=0] bin-hip/bench-fp8 $F8 --ms 1,2,3,4,6`, `--bitwise`, `--gemm --ms 13,64`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`; the probe: the patch on 58341df, the loop in the file's header |
| 102 | `decode-next/wave-slots.txt`, `decode-next/pool-pad.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-stall/` | `bin-hip/bench-wave-slots [64\|256] [split]`; `[DS_STALL_SPLIT=0] bin-hip/bench-kernels --stride`, `--bitwise`; the launch count: the build lines in `scripts/hip_launch_waves_shim.cpp`, then `scripts/launch_waves.py LOG bin-hip/ds-completion`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`; the slots: the patch, `DS_T_POOLPAD=256` on py-lru and cpp-matrix |
| 103 | `decode-next/wave-rounds.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,6 --waves --reps 500` |
| 104 | `decode-next/fp8-virt.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-virt/` | `bin-hip/bench-fp8 $F8 --ms 2,3,4,5,6,7,8 --virt`; `[DS_FP8_VIRT=0] bin-hip/bench-fp8 $F8 --ms 2,3,4,5,6,8`; `--bitwise --ms 2,3,4,5,6,7,8`; `--gemm --ms 13,64`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 105 | `trace/` (`cpp-matrix.json.gz`, `tcp-udp.json.gz`, `compare-step98.txt`, `critpath.txt`), `decode-next/virt-onoff.txt` | the command in `trace/README.txt` for both prompts, and again with `DS_FP8_VIRT=0`; `scripts/trace_compare.py <old> <new>` (step 98's traces: `git show cb5b695:ds-completion/results/trace/<name>.json.gz`); `scripts/trace_critpath.py <trace>`; untraced: the held-out command on cpp-matrix and geometry with and without `DS_FP8_VIRT=0` |
| 106 | `decode-next/draft-ahead.txt` | `python3 scripts/draft_ahead_sim.py` (over step 92's `decode-next/draft-vocab/` logs) |
| 100 | `trace/` (`cpp-matrix.json.gz`, `tcp-udp.json.gz`, `compare-step88.txt`, `critpath.txt`) | the command in `trace/README.txt` for both prompts; `scripts/trace_compare.py <step 88's trace> <new>` (step 88's: `git show 2509593:ds-completion/results/trace/<name>.json.gz`); `scripts/trace_critpath.py <trace>` |
| 99 | `decode-next/review-leftovers.txt`, `decode-next/{hc-mix,compressor-chunk,gather-streams}.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-cmpchunk/` | each patch on 5f21709 (`git apply`): `bin-hip/bench-kernels` (hc lines); `scripts/spec-measure.sh` with the tuned cache; the prompt gate's `--canon` runs against `results/prefill/gate/p*-base-canon.out`; the held-out command for cpp-matrix, tcp-udp, ja-email |
| 98 | `decode-next/local-review.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-compressor/` | `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 97 | `decode-next/local-review.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-gather/` | the same at step 97's commit; the second-copy path: `DS_EARLY_ROWS=1` with the held-out command (the `# short forwards:` line); the tables: step 88's traces (`results/trace/`) |
| 96 | `decode-next/exit-abort-step96.txt` | `bin-hip/bench-loader-exit 1500` and `bin-hip/bench-loader-exit 1500 serial`; the held-out command for ja-email and py-lru (the `# expert loader: ... unregistered` line); the handler: the same command with `-n 2000`, `kill -ABRT <pid>` during generation |
| 95 | `dist/llama-cpp-fork.txt`, `docs/llama-cpp-upstream.md` | the fork from a fresh clone: the commands in `docs/llama-cpp-upstream.md`, then `make FORK=<clone> BIN=bin-fresh && make HIP=1 FORK=<clone> BIN=bin-fresh-hip`, `bin-fresh/fp8pack` (cmp with `dense-fp8.gguf`), the held-out command for cpp-matrix and tcp-udp; `llama-completion -p "The three largest cities in Ukraine are" -n 32 --temp 0 --top-k 1 -t 16 -no-cnv` with the fork, the fresh build and upstream `a4d4f2fd4`; the type-43 tensors with each version's `gguf-py` (`GGUFReader`) |
| 94 | `decode-next/verify-halves-trace.txt` | `git apply results/decode-next/verify-halves.patch` on a519a9d, `make HIP=1`; the trace command of `trace/README.txt` with `DS_VERIFY_HALVES=1` (and untraced) for cpp-matrix and tcp-udp; `scripts/trace_overlap.py` and `scripts/trace_overlap.py --by-size` on those traces and step 88's |
| 93 | `decode-next/exit-abort.txt` | `scripts/spec-measure.sh`'s command for ja-email / py-lru in a loop; under `gdb -q -batch -ex "handle all nostop noprint pass" -ex "handle SIGABRT SIGSEGV SIGBUS stop print" -ex "run <args> > out 2> err < /dev/null" -ex "thread apply all bt 40" bin-hip/ds-completion` (the prompt by `-f`); ASan: `make HIP=1 BIN=bin-asan bin-asan/ds-completion CXXFLAGS="<CXXFLAGS with -O1 -fno-omit-frame-pointer -fsanitize=address>"`, run with `ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:halt_on_error=1:log_path=<prefix>` |
| 92 | `decode-next/draft-vocab.txt`, `decode-next/draft-vocab/` | `DS_DRAFT_TOKEN_LOG=results/decode-next/draft-vocab/<name>.log` with `scripts/spec-measure.sh`'s command for each held-out prompt and long-proof; `~/venvs/ds-ref/bin/python scripts/draft_vocab_sim.py results/decode-next/draft-vocab results/decode-next/draft-vocab` (calibration: `data/routing/study/*.bin`, step 13, not in git) |
| 91 | `cpu-decode/layout-live.txt`, `cpu-decode/layout-live.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8cpulayoutall/` | on 880b0b6: `git apply results/cpu-decode/layout-live.patch`, `make && make HIP=1`; `bin/bench-fp4-decode layout` (bits); the trace command of `trace/README.txt` with `--cpu-layout even`, `--cpu-layout odd`, `--cpu-layout even --cpu-layout-control` for cpp-matrix and tcp-udp, `scripts/trace_cpu_copy.py`; `CACHE="--cache-loads 800 --cache-period 8 --cpu-layout all" scripts/spec-measure.sh` (memory: /proc/meminfo sampled during it) |
| 90 | `decode-next/sclk-levels.txt`, `spec-measure-cacheloads800cacheperiod8/` (sclk 2), `spec-measure-sclk6-cacheloads800cacheperiod8-sclkcmp/`, `spec-measure-sclk8-cacheloads800cacheperiod8/` | after a reboot `cat` the model files to `/dev/null`; at each level (set on all four cards) `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (level 6's copied, the tracked directory checked out again) |
| 89 | `trace/` (regenerated), `trace/compare-step86.txt`, `trace/critpath.txt` | the command in `trace/README.txt` for tcp-udp and cpp-matrix, `gzip -9 -n`; `scripts/trace_compare.py` against `git show b3d3b24:ds-completion/results/trace/<name>.json.gz`; `scripts/trace_critpath.py` on both |
| 88 | `decode-next/gpu-argmax.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-gpuargmax/` | the trace command of `trace/README.txt` before and after (spans `tail: ...`); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 87 | `decode-next/draft-route.txt`, `decode-next/draft-route/`, `decode-next/verify-tail.txt` | `DS_DRAFT_ROUTE_LOG=<name>.log` with the held-out command for each of the 8 prompts, `scripts/draft_route_stats.py <logs>`; the tail: the trace command of `trace/README.txt` (spans `tail: ...`) |
| 86 | `decode-next/draft.txt`, `trace/` (regenerated), `trace/compare-step84.txt`, `trace/critpath.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-homefixed/` | traces: the command in `trace/README.txt` (and `--cache-loads 0` without `--cache-wait`; step 84's build for the slow drafts), `scripts/trace_critpath.py` (its cycle section), `scripts/trace_compare.py`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 85 | `decode-next/load-gate.txt`, `decode-next/load-gate.patch` | `git apply results/decode-next/load-gate.patch` on ecb5af9, `make HIP=1`; the trace command of `trace/README.txt` with `DS_LOAD_GATE=1` (and `DS_LOAD_PER_LAYER=2` / `3`) or `DS_LOAD_GATE=2`; `scripts/trace_compare.py` against step 84's traces |
| 84 | `decode-next/copy-engine.txt`, `trace/` (regenerated), `trace/compare-step74.txt`, `trace/critpath.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-kcopy/` | traces: the command in `trace/README.txt` (and `--cache-loads 0` without `--cache-wait` for the no-load case), `scripts/trace_compare.py`, `scripts/trace_critpath.py`, the copy distribution from the `ffn.cpu_rows_copy` spans (the file's method); the variants as temporary changes in `hip/rt.cpp` / `hip/expert_loader.cpp`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 83 | `cpu-decode/layout.txt` | `make bin/bench-fp4-decode`, `bin/bench-fp4-decode layout 1500`, and with `BENCH_PF64=0` (4096, 16384, 32768) for the prefetch distance |
| 82 | `cpu-decode/sched.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-taskpair/` | `bin/bench-fp4-decode sched 0` and `sched 1500` (`SCHEDS=a,b` for some); the real kernel: `bin-hip/bench-cpu-moe -m $M -t 16 --check-layers "" --warm 1,2,6 --steps 20` and `--decode 1,2,6 --steps 10 --layers 4` with `kTaskPair` 1 (or the old static loops) vs 2 in `src/cpu_moe.cpp`; `--batch 1` (3, 8); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 81 | `cpu-decode/after-gap.txt`, `cpu-decode/spin-wait.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-spinwait/` | `bin/bench-fp4-decode call 16 2m <0|1500|-1500> <single|worker> 1000`; C6 entries from `/sys/devices/system/cpu/cpu*/cpuidle/state3/usage`, C6 off by writing 1 to `state3/disable` (root); end to end with `HSA_ENABLE_INTERRUPT=0`, and with `git apply results/cpu-decode/spin-wait.patch` and `DS_SPIN_WAIT=1` (alternating, and `DS_SPIN_WAIT=1 CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`) |
| 80 | `cpu-decode/act-single.txt`, `cpu-decode/act-single-ab.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-actsingle/` | `bin/bench-fp4-decode act`; `bin/bench-fp4-decode call 16 2m <0|1500> <once|rewrite|single> <2000|1000>`; `bin-hip/bench-cpu-moe -m $M -t 16 --batch 1` (3, 8) and `--warm 1,2,6 --steps 20`; the A/B against a build of 3f1ca2c; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 79 | `cpu-decode/counters.txt`, `cpu-decode/aligned16-ab.txt`, `cpu-decode/aligned16.patch`, `spec-measure-sclk6-cacheloads800cacheperiod8-aligned16/` | `git apply results/cpu-decode/aligned16.patch` on 3f1ca2c, `make bin/bench-fp4-decode`; `bin/bench-fp4-decode call 16 2m 0 <once|rewrite|copy|single|aligned16> 2000`, and under `perf stat -x, -e cycles:u,instructions:u,<4 events>` with 2000 and 0 calls (groups in the file); the A/B and `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` with the patch built |
| 78 | `cpu-decode/decode-bench.txt`, `cpu-decode/call.txt`, `cpu-decode/warm.txt` (part 2), `cpu-decode/private-copies-ab.txt`, `cpu-decode/call-gap.txt`, `cpu-decode/private-copies.patch` | `make bin/bench-fp4-decode`; `OMP_PROC_BIND=spread OMP_PLACES=cores bin/bench-fp4-decode 16 5120` and `... 16 2304`; `bin/bench-fp4-decode call 16` and `call 16 4k`; `call 16 2m 1500` / `-1500` and with `OMP_WAIT_POLICY=passive` / `active`; `bin-hip/bench-cpu-moe ... --warm 1,2,6` and `--warm 1 --gap-us 1500`; the A/B: `git apply results/cpu-decode/private-copies.patch` on e1b9fbc, the command in the file for cpp-matrix and tcp-udp with and without it, after `cat` of the model files to warm the page cache |
| 77 | `cpu-decode/warm.txt` (part 1) | `make HIP=1 bin-hip/bench-cpu-moe`, `bin-hip/bench-cpu-moe -m $M -t 16 --check-layers "" --warm 1,2,6 --steps 20` |
| 76 | `decode-next/attn-split.txt`, `decode-next/attn-split.patch` | on 4b1a52c: `git apply results/decode-next/attn-split.patch`, `make HIP=1`, `bin-hip/bench-xdie 6`, `bin-hip/bench-xdie 2`, `bin-hip/bench-xdie proj`; end to end `scripts/spec-measure.sh`'s command for one prompt with and without `--attn-split`, and the same with `--trace` compared by `scripts/trace_compare.py` |
| 75 | `decode-next/fp8-multi-step75.txt`, `trace/` (regenerated), `trace/compare-step73.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,4,5,6,8` (and `--bitwise`) with each variant as a temporary change to `hip/fp8_gemv.hip`; register counts: `hipcc ... -c hip/fp8_gemv.hip -Rpass-analysis=kernel-resource-usage`; traces: the commands in `trace/README.txt`, compared with `git show 8138b22:ds-completion/results/trace/<name>.json.gz` |
| 74 | `decode-next/indexer.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-indexer/` | the gate commands in the file (step 73's against step 73's outputs); the trace split per layer: `results/trace/cpp-matrix.json.gz`, `attn.indexer L<n>` spans |
| 73 | `decode-next/engram.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-engram/`, `trace/` (regenerated), `trace/compare-step69.txt` | the gate commands in the file (as step 71's); the probe was a temporary patch (getrusage around `fetch_engram_rows`); traces: the commands in `trace/README.txt`, `gzip -9 -n`, `scripts/trace_compare.py` against `git show b3b107b:ds-completion/results/trace/<name>.json.gz` |
| 72 | `heldout-history/` | `scripts/heldout-rerun.sh --build rerun optional`, `scripts/heldout-rerun.sh rerun` (and `optional`), then `python3 scripts/heldout_history.py`; points and flags in `points.tsv`; progress: `scripts/heldout-status.sh` |
| 71 | `decode-next/handoff.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-handoff/` | the gate commands in the file (prompt gate, the 20,000-byte prompt, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh`, copied); the host timer was a temporary patch (three clock reads in `moe_finish`), the trace medians: `--trace` as in `trace/README.txt` |
| 70 | `decode-next/verify-halves.txt`, `verify-halves.patch` | the commands in the file; the layer medians: `results/trace/*.json.gz` (step 69) |
| 69 | `trace/` (regenerated), `trace/compare-step58.txt` | the commands in `trace/README.txt`, then `gzip -9 -n`; the comparison: `scripts/trace_compare.py` on step 58's traces (`git show d7bbabd:ds-completion/results/trace/<name>.json.gz`) and the new ones |
| 68 | `attn-batch/stage3.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-attnbatch/` | `DS_ATTN_CHECK=1 bin-hip/ds-completion ...` (prints the comparison at exit), the prompt gate, the long prompt against a build of 6e7126a, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (copied), the profile with `--profile` and `scripts/profile_verify.py` |
| 67 | `attn-batch/stage2.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-phases/` | the same gates (commands in the file; the spec run copied) |
| 66 | `attn-batch/stage1.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-ring/` | `scripts/prefill-gate.sh`-style runs, a 20,000-byte prompt against a build of 67453f6, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (commands in the file; the spec run copied) |
| 65 | `decode-next/fp8-multi.txt`, `fp8-lds64.patch`, `host-lag.txt` | `bin-hip/bench-fp8 $F8 --ms 1,2,3,4,6 [--gemm \| --sweep]` (the 8-wave variant: `git apply results/decode-next/fp8-lds64.patch`, `DS_FP8_LDS64=8`); the lag: the command in `host-lag.txt` (`--trace` prints it) |
| 64 | `route-ahead/prefetch-sim.txt` | `~/venvs/ds-ref/bin/python scripts/prefetch_sim.py` on `data/route-ahead/*.bin` (from step 63's run) |
| 63 | `route-ahead/` (`summary.txt`, `pooled.txt`, per prompt `.err`) | `scripts/route-ahead.sh 4` (`DS_ROUTE_AHEAD=4 DS_ROUTE_LOG=...`; logs in `data/route-ahead/`, ~185 MB); `pooled.txt`: the per-prompt tables weighted by their counts |
| 62 | the load rate in TIMING.md: `loads started` and `generation:` in `spec-measure-sclk6-cacheloads800cacheperiod8-head/*.err` | (from the files) |
| 62 | `head-lds.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-head/` | `bin-hip/bench-kernels --bitwise`; the variants: `gemv_head_launch<R, WAVES>` in `hip/kernels.hip`; `DS_DRAFT_PROFILE=1 bin-hip/ds-completion ...` (the draft's parts); `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (copied) |
| 61 | (none: a reading of `hip/dspark.cpp`) | the draft's loader: `fp8(...)` for its dense matrices, `fp4_upload(...)` for its experts |
| 60 | `gpu-memory/` | the command in `gpu-memory/README.txt` at each `--max-ctx` (the ledger is printed at startup) |
| 59 | `cache-decay/` | `~/venvs/ds-ref/bin/python scripts/cache_sim.py --decay` (`sweep.txt`); `threshold.txt`: `cs.run(..., rate=800, period=8, [half_life=H], min_count=M)` over the same logs (see its header); `live-min.txt`: the command in the file with `--cache-min 2` and `1` |
| 58 | `trace/` (regenerated), `timing-sclk6/` | the commands in `trace/README.txt`, then `gzip -9`; the `--profile` comparison and `timing-sclk6/verify-profile.txt`: the same command with `--profile` instead of `--trace`, then `scripts/profile_verify.py`; `timing-sclk6/prompt-profile.txt`: the command in the file |
| 57 | `cpu-decode/per-call.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-nx1/` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --decode 1,2,6 --steps 20` (the per-call lines), `--batch 1`, `--batch 3`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (writes the step-50 directory; copied) |
| 56 | `cpu-decode/` | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --decode 1,2,4,6` [`--gap-us 1500`, `OMP_*` env, CPU settings as in the file]; `... --replay data/routing/prefill/readme24k.bin [--static]`; the runtime A/B used a `--cpu-static` flag since removed (`batch_sched = kStatic` for calls of up to 16 tokens) |
| 55 | `fp4-grouped/` | `git apply results/fp4-grouped/attempt.patch` on da19a4f, `make HIP=1 bin-hip/bench-fp4-moe`, `bin-hip/bench-fp4-moe -m $M` (variant 2: `kCanonR` in `hip/fp4_moe.hip`; variant 1: `canon_block` instead of the decoded form in the grouped kernels) |
| 54 | `coselect.txt` | `~/venvs/ds-ref/bin/python scripts/coselect.py data/routing/study/*.bin --placement placements/study30.txt --trace results/trace/cpp-matrix.json.gz` |
| 53 | `router-batched.txt`, `spec-measure-sclk6-cacheloads800cacheperiod8-router/`, `prefill/router-batched-ab.txt` | `bin-hip/bench-kernels`; `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` (page cache warm first: `cat ~/llms/UD-Q8_K_XL/*.gguf > /dev/null`; the run writes the step-50 directory, copied); the A/B commands in the file (old build: a worktree at 97844c8) |
| 52 | `prefetch-prediction-sclk6.txt` | the `--profile` output of the step 51 runs (commands in `trace/README.txt`) |
| 51 | `trace/` | `bin-hip/ds-completion ... --spec --canon --cache-loads 800 --cache-period 8 --cache-wait --chat -n 300 --trace <name>.json` (in `trace/README.txt`), then `gzip -9` |
| 50 | `cpu-moe-parallel-act.txt` | `bin/bench-cpu-moe -m $M -t 16 --tokens 8 --check-layers 2 -k N`; `... --check-layers "" --batch 6` |
| 50 | `cache-tuning-sclk6.txt`, `spec-measure-sclk6-cacheloads200cacheperiod16/`, `spec-measure-sclk6-cacheloads800cacheperiod8/` | the two-prompt command in the file with each setting; `CACHE="--cache-loads 200 --cache-period 16" scripts/spec-measure.sh`, `CACHE="--cache-loads 800 --cache-period 8" scripts/spec-measure.sh` |
| 50 | `miss-histogram-sclk6.txt` | the command in the file (`--profile`) |
| 49 | `decode-profile-sclk6.txt` | the command in the file under `LD_PRELOAD` of `scripts/hip_count_shim.c` (`gcc -O2 -shared -fPIC -o shim.so scripts/hip_count_shim.c -ldl`), `scripts/profile_verify.py`, `bin-hip/bench-launch` |
| 48 | `spec-measure-sclk6/` (summary, per-prompt logs, `analysis.txt`) | `scripts/spec-measure.sh`, then `scripts/spec_analysis.py --max-steps 400 results/spec-measure-sclk6/*.log` |
| 48 | `spec-loop-check.txt`, `engram-copy-fix.txt` | commands in the files (the old build: a worktree at 73b6891) |
| 47 | `spec-momentum.txt` | `~/venvs/ds-ref/bin/python scripts/spec_momentum.py 'results/spec-conf/*-n5.log'`; TIMING.md quotes steps 47-82 (generation, prompt, the cache's loads) |
| 46 | `prefill/chunk1024.txt` | `kChunk = 1024` (reverted), then the commands in the file |
| 45 | `prefill/o3.txt`, `canon-gate.txt` + `canon-gate/` (rerun) | `bin/bench-cpu-moe -m $M -t 16 --check-layers "" --replay data/routing/prefill/readme24k.bin`; `bin-hip/ds-completion ... --profile` (commands in the file); `scripts/canon-gate.sh` |
| 44 | `prefill/stream-attempt.txt` | the removed P2 code at each stage (commands and stages in the file; the code is not in the repository) |
| 43 | `prefill/bench-register.txt` | `bin-hip/bench-register $SHARD7 --gb 8 --pieces 0,1024,256,64 --faults $SHARD6` |
| 43 | `prefill/bench-expert-load-pieces.txt`, `prefill/register-stall.txt` | `bin-hip/bench-expert-load -m $M`; the runtime commands in the file |
| 42 | `prefill/stream-sim.txt` | routing logs of `data/routing/prefill/{readme24k,rtcpp32k}.txt` (`--log-routing`, commands in the file), then `scripts/prefill_stream_sim.py [--cpu-scale 1.65] [--load-rate 1000] [--canon]`; the calibration run is the same prompt with `--profile` |
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
