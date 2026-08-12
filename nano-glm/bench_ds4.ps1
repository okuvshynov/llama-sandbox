# DeepSeek-V4-Flash: what the expert-parallel split costs, and what llama.cpp's
# weight repacking is worth. PLAN.md step 14, reasons 2 and 3.
#
#   .\bench_ds4.ps1 -Repack        # experiment 1: llama-bench, repack ON vs OFF
#   .\bench_ds4.ps1 -Split         # experiment 2: llama.cpp vs nano-glm vs RPC
#   .\bench_ds4.ps1 -Vulkan        # experiment 3: llama-bench, sweeping -ngl
#
# Three experiments, deliberately not one, because they answer different
# questions and only compose if measured apart:
#
#   1. **The kernel handicap.** `GGML_CPU_REPACK` rewrites quantized weights
#      into a blocked layout at load and runs a different GEMM against them.
#      nano-glm mmaps weights as they sit in the file and cannot follow, so this
#      is a cost it pays by construction, not a bug to fix. Two llama-bench
#      binaries from one source, differing in that flag alone.
#
#   2. **The architecture.** llama.cpp with everything local, against nano-glm
#      with its routed experts behind a socket. Measured with repack OFF on both
#      sides so the kernels match and the difference is the *split* — experiment
#      1 is what translates the answer back to llama.cpp as shipped.
#
#   3. **The GPUs.** This machine has four Vega II dies at 31.73 GiB each —
#      126.9 GiB against a 150.7 GiB model, so roughly five sixths of it can be
#      resident and `-ngl` is the knob that says how much. Sweeping it turns
#      "does the GPU help" into a curve, which is the only form the answer can
#      take when the model does not fit: every layer offloaded is a layer the
#      CPU no longer streams, and the last few layers cannot go anywhere.
#
#      Read this one with a caveat attached. llama.cpp's Vulkan backend has no
#      kernel for either of DeepSeek-V4's two custom ops — `DSV4_HC_COMB`
#      (hyper-connections, every layer) or `LIGHTNING_INDEXER` (the ratio-4
#      layers); CPU, CUDA and Metal have both, Vulkan has neither. The scheduler
#      falls those nodes back to the host, so an offloaded layer still pays a
#      round trip on the residual stream. The curve is therefore a floor for
#      this architecture on Vulkan, not the hardware's ceiling.
#
# Why not one number: quoting "nano-glm is X% of llama.cpp" without saying which
# kernels either side used would fold a compile flag into an architectural
# claim, and this repo has already paid for that class of mistake twice.
#
# Measurement discipline, all of it learned the hard way (repo CLAUDE.md):
#   - never a single mmap-backed timing on Windows: 1.84 +/- 0.02 t/s and
#     1.04 +/- 0.29 came from the same binary and model minutes apart. `-r 5`,
#     and `-lm none` where the tool has it.
#   - prefill and decode separately: they are bound by different things.
#   - a discarded warm-up run for the tools that have no `-r`, because a
#     harness-style run pages the model in *during* the measured window.

[CmdletBinding()]
param(
    [switch] $Repack,
    [switch] $Split,
    [switch] $Vulkan,
    [string] $Model = "D:\llms\ds-v4-flash\UD-Q8_K_XL\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf",
    [int]    $Threads = 16,
    [int]    $Reps = 5,
    [int[]]  $Ngl = @(0, 8, 16, 24, 32, 36),
    [string] $MoeAddr = ""
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$results = Join-Path $here "results"

if (-not $Repack -and -not $Split -and -not $Vulkan) { throw "pick -Repack, -Split or -Vulkan" }

if ($Repack) {
    foreach ($v in @("on", "off")) {
        $exe = Join-Path $results "bench-builds\repack-$v\bin\llama-bench.exe"
        if (-not (Test-Path $exe)) { throw "missing $exe - run .\build_bench.ps1" }
        $log = Join-Path $results "ds4-bench-repack-$v.log"
        Write-Host "`n=== llama-bench, repack=$v" -ForegroundColor Cyan
        # -lm none: weights into ordinary allocated memory rather than mmap, so
        # the standby-list state of the machine stops being a variable. This is
        # the fix for the +/-27% spread recorded in CLAUDE.md.
        & $exe -m $Model -t $Threads -r $Reps -p 128 -n 32 -lm none 2>&1 |
            Tee-Object -FilePath $log
    }
}

if ($Split) {
    $golden = Join-Path $here "testdata-deepseek4\01_prose.bin"
    if (-not (Test-Path $golden)) { throw "missing $golden" }

    # Identical work on both sides: the same token ids, prompt in one chunk then
    # single-token steps. `rescore --sim-gen` and `nano-glm` differ in
    # implementation and in nothing else, which is what makes the ratio mean
    # something. A tool-shaped comparison (llama-bench vs nano-glm) would not.
    $rescore = Join-Path $here "..\logit-kld\build\bin\rescore.exe"
    $nano    = Join-Path $here "build\bin\nano-glm.exe"

    # Two passes each, both kept. Neither tool has llama-bench's `-r`, and a
    # harness-style run pages the model in *during* the measured window — the
    # first pass is the warm-up and the second is the number. Keeping both is
    # what makes the warm-up curve visible instead of averaged in.
    #
    # Through `cmd /c` rather than called directly: both tools write progress to
    # stderr, and under `$ErrorActionPreference = "Stop"` any stderr from a
    # native command becomes a terminating error even on exit 0. build.ps1 does
    # the same for the same reason.
    $log = Join-Path $results "ds4-bench-split.log"
    if (Test-Path $log) { Remove-Item $log }

    foreach ($pass in 1, 2) {
        Add-Content $log "=== pass $pass"
        Write-Host "`
=== pass $pass" -ForegroundColor DarkGray

        $rlog = Join-Path $results "bench-rescore.out"
        cmd /c "`"$rescore`" -m `"$Model`" -i `"$golden`" --sim-gen -t $Threads -o `"$results\bench-rescore.bin`" > `"$rlog`" 2>&1"
        $line = (Select-String -Path $rlog -Pattern "scored" | Select-Object -Last 1).Line
        Add-Content $log "  llama.cpp rescore --sim-gen : $line"
        Write-Host "  llama.cpp rescore : $line"

        $nlog = Join-Path $results "bench-nano.out"
        cmd /c "`"$nano`" -m `"$Model`" -i `"$golden`" -n 32 -t $Threads -o `"$results\bench-nano.bin`" > `"$nlog`" 2>&1"
        $line = (Select-String -Path $nlog -Pattern "n_prompt=" | Select-Object -Last 1).Line
        Add-Content $log "  nano-glm local              : $line"
        Write-Host "  nano-glm local    : $line"

        if ($MoeAddr) {
            cmd /c "`"$nano`" -m `"$Model`" -i `"$golden`" -n 32 -t $Threads --moe-addr $MoeAddr -o `"$results\bench-nano-rpc.bin`" > `"$results\bench-nano-rpc.out`" 2>&1"
            $line = (Select-String -Path "$results\bench-nano-rpc.out" -Pattern "n_prompt=" | Select-Object -Last 1).Line
            Add-Content $log "  nano-glm via moe-server     : $line"
            Write-Host "  nano-glm via RPC  : $line"
        }
    }
    Write-Host "`
wrote $log" -ForegroundColor Green
}

if ($Vulkan) {
    $exe = Join-Path $results "bench-builds\vulkan\bin\llama-bench.exe"
    if (-not (Test-Path $exe)) { throw "missing $exe - run .\build_bench.ps1 -Vulkan" }
    $log = Join-Path $results "ds4-bench-vulkan.log"
    if (Test-Path $log) { Remove-Item $log }

    # `-ngl` alone turned out to be the wrong knob and the sweep that found that
    # out is preserved as the first four rows. Two flags matter more:
    #
    #   -nopo 1   disables `op_offload`, which otherwise hands any host-resident
    #             matmul to a higher-priority backend once its batch reaches 32
    #             (ggml-backend.cpp:959, ggml-vulkan.cpp:18511). With two ops
    #             Vulkan cannot run at all — DSV4_HC_COMB every layer,
    #             LIGHTNING_INDEXER on the ratio-4 layers — each hand-off comes
    #             straight back, and prefill fragments into ~1800 graph splits.
    #             It costs 3.3x of prefill at -ngl 0, where *nothing* is
    #             deliberately offloaded, and one flag recovers all of it.
    #
    #   -ncmoe N  keeps the routed experts of the first N layers on the CPU
    #             while everything else follows -ngl. On this model 3.19 of the
    #             3.46 GiB per layer is experts, so `-ngl 99 -ncmoe 43` puts
    #             every layer's *attention* on the GPU for 13 GiB and leaves all
    #             141 GiB of experts on the host. That is the best decode
    #             configuration measured, and it uses a tenth of the VRAM of the
    #             best -ngl one.
    #
    # One invocation per configuration rather than llama-bench's own list
    # syntax, for two reasons. It reloads the model for each anyway (what gets
    # allocated where changes), so nothing is saved by batching; and the
    # interesting configurations sit near the VRAM limit, where the failure mode
    # is an abort during load. Separately invoked, the run that does not fit
    # costs only itself and the ones already done are on disk.
    #
    # Through `cmd /c` for the reason the -Split block gives: llama.cpp writes
    # to stderr, and under `$ErrorActionPreference = "Stop"` that is a
    # terminating error even on exit 0.
    $configs = @()
    foreach ($n in $Ngl) { $configs += @{ tag = "ngl$n"; args = "-ngl $n" } }
    $configs += @{ tag = "ngl0-nopo";     args = "-ngl 0 -nopo 1" }
    $configs += @{ tag = "ngl32-nopo";    args = "-ngl 32 -nopo 1" }
    $configs += @{ tag = "ncmoe43";       args = "-ngl 99 -ncmoe 43" }
    $configs += @{ tag = "ncmoe43-nopo";  args = "-ngl 99 -ncmoe 43 -nopo 1" }
    $configs += @{ tag = "ncmoe36-nopo";  args = "-ngl 99 -ncmoe 36 -nopo 1" }
    # -ts is SLASH-separated. llama-bench reads a comma as "run this again with
    # the next value", so `-ts 30,3,3,7` is four single-device runs that put
    # everything on Vulkan0 — and it fails with an OOM naming the device you
    # meant to leave empty, which reads as a capacity problem and is a parsing
    # one. These splits exist because -ncmoe makes per-layer size uneven (3.46
    # GiB carrying experts, 0.27 without) while llama.cpp splits layers evenly,
    # so without them the expert-carrying tail piles onto the last die.
    $configs += @{ tag = "ncmoe30-ts";    args = "-ngl 99 -ncmoe 30 -nopo 1 -ts 30/3/3/7" }
    $configs += @{ tag = "ncmoe24-ts";    args = "-ngl 99 -ncmoe 24 -nopo 1 -ts 24/6/6/7" }
    $configs += @{ tag = "ncmoe19-ts";    args = "-ngl 99 -ncmoe 19 -nopo 1 -ts 19/8/8/8" }

    foreach ($c in $configs) {
        Write-Host "`n=== llama-bench vulkan, $($c.args)" -ForegroundColor Cyan
        $out = Join-Path $results "bench-vulkan-$($c.tag).out"
        # -v so the load-time buffer sizes ("Vulkan0 model buffer size = ...")
        # land in the per-run file: with four devices and a model that does not
        # fit, where the weights actually went is half the measurement.
        cmd /c "`"$exe`" -m `"$Model`" -t $Threads -r $Reps -p 128 -n 32 -lm none $($c.args) -v > `"$out`" 2>&1"
        $rc = $LASTEXITCODE

        Add-Content $log "=== $($c.args)  (exit $rc)"
        if ($rc -ne 0) {
            Write-Host "  FAILED (exit $rc) - see $out" -ForegroundColor Yellow
            $why = (Select-String -Path $out -Pattern "error|failed|unable|out of memory" |
                    Select-Object -Last 3).Line
            $why | ForEach-Object { Add-Content $log "    $_" }
            continue
        }
        $rows = (Select-String -Path $out -Pattern "^\|.*\|\s*(pp|tg)" ).Line
        $rows | ForEach-Object { Add-Content $log "  $_"; Write-Host "  $_" }
        $bufs = (Select-String -Path $out -Pattern "model buffer size|KV self size|compute buffer size").Line
        $bufs | ForEach-Object { Add-Content $log "    $_" }
        # The split count is the diagnostic, not a curiosity: it tracked prefill
        # inversely across every configuration measured (1780 splits -> 5.37
        # t/s, 10 splits -> 16.68) and is what identified op_offload as the
        # cause. Record it beside the timing or the next reader repeats the
        # afternoon that found it.
        $spl = (Select-String -Path $out -Pattern "graph splits|graph nodes").Line | Select-Object -First 2
        $spl | ForEach-Object { Add-Content $log "    $_" }
    }
    Write-Host "`nwrote $log" -ForegroundColor Green
}
