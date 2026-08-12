# DeepSeek-V4-Flash: what the expert-parallel split costs, and what llama.cpp's
# weight repacking is worth. PLAN.md step 14, reasons 2 and 3.
#
#   .\bench_ds4.ps1 -Repack        # experiment 1: llama-bench, repack ON vs OFF
#   .\bench_ds4.ps1 -Split         # experiment 2: llama.cpp vs nano-glm vs RPC
#
# Two experiments, deliberately not one, because they answer different questions
# and only compose if measured apart:
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
    [string] $Model = "D:\llms\ds-v4-flash\UD-Q8_K_XL\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf",
    [int]    $Threads = 16,
    [int]    $Reps = 5,
    [string] $MoeAddr = ""
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$results = Join-Path $here "results"

if (-not $Repack -and -not $Split) { throw "pick -Repack or -Split" }

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
