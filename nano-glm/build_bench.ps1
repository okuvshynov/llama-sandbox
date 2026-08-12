# Build two llama-bench binaries that differ in exactly one thing:
# GGML_CPU_REPACK. Nothing else about them may vary, which is why they are built
# here from one source tree with one command rather than reusing whatever is in
# llama.cpp/build — that tree is the user's, its age is unknown, and "same
# source, same flags" is the entire content of the comparison.
#
#   .\build_bench.ps1                 # both variants into results/bench-builds/
#
# GGML_CPU_REPACK rewrites quantized weights into a blocked layout at load and
# runs a different GEMM against them. nano-glm mmaps weights as they sit in the
# file and cannot follow, so `logit-kld` builds with it OFF to keep the
# *correctness* comparison meaningful (logit-kld/CMakeLists.txt). That makes the
# question "what does it cost?" a real one rather than a curiosity: the answer
# is the size of the handicap nano-glm accepts by construction.

[CmdletBinding()]
param(
    [string] $LlamaCppDir = "C:\Users\oleksandr\Desktop\llama.cpp",
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

foreach ($repack in @("ON", "OFF")) {
    $build = Join-Path $out ("repack-" + $repack.ToLower())
    if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

    # GGML_NATIVE and the build type match llama.cpp's own default build, so the
    # only difference between the two trees — and between either of them and
    # what llama.cpp ships — is the repack flag.
    $cfg = "cmake -S `"$LlamaCppDir`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
           "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DGGML_NATIVE=ON -DGGML_METAL=OFF " +
           "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON " +
           "-DLLAMA_BUILD_SERVER=OFF -DGGML_CPU_REPACK=$repack"

    Write-Host "`nconfiguring repack=$repack -> $build" -ForegroundColor Cyan
    cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed (repack=$repack)" }

    cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" --target llama-bench -j 2>&1" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "build failed (repack=$repack)" }

    Write-Host "built $build\bin\llama-bench.exe" -ForegroundColor Green
}
