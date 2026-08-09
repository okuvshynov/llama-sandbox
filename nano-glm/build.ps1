# Windows build for nano-glm (and, with -Project logit-kld, for the sibling
# verification tools). cl.exe is not on PATH by default and CMake needs a
# generator; this locates the MSVC environment and VS's bundled ninja the same
# way ../colibri and ../vk-test do, so a build is one command.
#
#   .\build.ps1                          # build nano-glm
#   .\build.ps1 -Project logit-kld       # build collect + rescore
#   .\build.ps1 -Clean                   # reconfigure from scratch
#   .\build.ps1 -Trace                   # routing-trace variant, into build-trace\
#
# Executables and the ggml/llama DLLs both land in <project>\build\bin\.
#
# -Trace is a separate build *tree* on purpose: NANO_EXPERT_TRACE changes how
# the graph is allocated (lib/expert_trace.h), so the untraced binary has to
# stay around to byte-compare against.

[CmdletBinding()]
param(
    [string] $Project     = "nano-glm",
    [string] $LlamaCppDir = "C:\Users\oleksandr\Desktop\llama.cpp",
    [switch] $Clean,
    [switch] $Trace
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $repo $Project
if (-not (Test-Path $src)) { throw "no such subproject: $src" }
if (-not (Test-Path (Join-Path $LlamaCppDir "ggml\CMakeLists.txt"))) {
    throw "llama.cpp not found at $LlamaCppDir (pass -LlamaCppDir)"
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found - install Visual Studio" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "no VS install with the C++ toolchain found" }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

# CMake needs an explicit path to ninja: vcvars64 does not put VS's copy on PATH.
$ninja = Get-Command ninja -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $ninja) { $ninja = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }
if (-not (Test-Path $ninja)) { throw "ninja not found (looked for $ninja)" }

$build = Join-Path $src ($(if ($Trace) { "build-trace" } else { "build" }))
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

# cmake and cl must see the same environment, so both steps run under vcvars in
# one cmd invocation. The redirect silences vcvars' banner without hiding errors.
# CMAKE_EXPORT_COMPILE_COMMANDS must be a CACHE variable, not just any value:
# llama.cpp sets it as a normal variable inside ggml's directory scope, which
# exports ggml's own sources but not this project's. Passing -D here makes it
# global, so build/compile_commands.json covers our targets too and VS Code's
# C/C++ extension can resolve the ggml includes (see ../.vscode/).
$cfg = "cmake -S `"$src`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
       "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DLLAMA_CPP_DIR=`"$LlamaCppDir`" " +
       "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
if ($Trace) { $cfg += " -DNANO_EXPERT_TRACE=ON" }
# The inner 2>&1 is cmd's, not PowerShell's: llama.cpp's CMakeLists writes
# informational lines to stderr, and under $ErrorActionPreference = "Stop" any
# stderr from a native command becomes a terminating error even on exit 0.
cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" -j 2>&1"
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`nbuilt into $build\bin" -ForegroundColor Green
Get-ChildItem "$build\bin\*.exe" | ForEach-Object { Write-Host "  $($_.Name)" }
