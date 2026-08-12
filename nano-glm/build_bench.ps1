# Build two llama-bench binaries that differ in exactly one thing:
# GGML_CPU_REPACK. Nothing else about them may vary, which is why they are built
# here from one source tree with one command rather than reusing whatever is in
# llama.cpp/build — that tree is the user's, its age is unknown, and "same
# source, same flags" is the entire content of the comparison.
#
#   .\build_bench.ps1                 # both variants into results/bench-builds/
#   .\build_bench.ps1 -Vulkan         # a third: Vulkan offload, repack ON
#
# GGML_CPU_REPACK rewrites quantized weights into a blocked layout at load and
# runs a different GEMM against them. nano-glm mmaps weights as they sit in the
# file and cannot follow, so `logit-kld` builds with it OFF to keep the
# *correctness* comparison meaningful (logit-kld/CMakeLists.txt). That makes the
# question "what does it cost?" a real one rather than a curiosity: the answer
# is the size of the handicap nano-glm accepts by construction.
#
# `-Vulkan` answers a different question: this machine has four Vega II dies at
# 31.73 GiB each, 126.9 GiB against a 150.7 GiB model, so most of DeepSeek-V4
# *could* be resident. It builds with repack ON — llama.cpp exactly as shipped —
# because the point is what the GPUs add to the tool a user would actually run,
# and because `-ngl 0` on that binary should then reproduce the repack-on CPU
# number, which is the control that says the Vulkan build changed nothing else.

[CmdletBinding()]
param(
    [string] $LlamaCppDir = "C:\Users\oleksandr\Desktop\llama.cpp",
    [switch] $Vulkan,
    [switch] $Clean
)

$ErrorActionPreference = "Stop"
$out = Join-Path $PSScriptRoot "results\bench-builds"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found - install Visual Studio" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$ninja = Get-Command ninja -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $ninja) { $ninja = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }

# name -> the two flags that define it. Everything else is held fixed below.
$variants = if ($Vulkan) { @{ "vulkan" = @("ON", "ON") } }
            else         { [ordered]@{ "repack-on" = @("ON", "OFF"); "repack-off" = @("OFF", "OFF") } }

foreach ($name in $variants.Keys) {
    $repack, $vk = $variants[$name]
    $build = Join-Path $out $name
    if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

    # GGML_NATIVE and the build type match llama.cpp's own default build, so the
    # only difference between the trees — and between any of them and what
    # llama.cpp ships — is the repack flag and, for the third, the Vulkan one.
    $cfg = "cmake -S `"$LlamaCppDir`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
           "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DGGML_NATIVE=ON -DGGML_METAL=OFF " +
           "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON " +
           "-DLLAMA_BUILD_SERVER=OFF -DGGML_CPU_REPACK=$repack -DGGML_VULKAN=$vk"

    Write-Host "`nconfiguring $name (repack=$repack vulkan=$vk) -> $build" -ForegroundColor Cyan
    cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($name)" }

    # The Vulkan variant compiles a few thousand shaders through
    # vulkan-shaders-gen before anything else links; it is minutes, not seconds.
    cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" --target llama-bench -j 2>&1" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "build failed ($name)" }

    Write-Host "built $build\bin\llama-bench.exe" -ForegroundColor Green
}
