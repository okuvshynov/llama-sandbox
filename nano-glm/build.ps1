# Windows build for nano-glm (and, with -Project logit-kld, for the sibling
# verification tools). cl.exe is not on PATH by default and CMake needs a
# generator; this locates the MSVC environment and VS's bundled ninja the same
# way ../colibri and ../vk-test do, so a build is one command.
#
#   .\build.ps1                          # build nano-glm
#   .\build.ps1 -Project logit-kld       # build collect + rescore
#   .\build.ps1 -Clean                   # reconfigure from scratch
#
# Executables and the ggml/llama DLLs both land in <project>\build\bin\.

[CmdletBinding()]
param(
    [string] $Project     = "nano-glm",
    [string] $LlamaCppDir = "C:\Users\oleksandr\Desktop\llama.cpp",
    [switch] $Clean
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

$build = Join-Path $src "build"
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

# cmake and cl must see the same environment, so both steps run under vcvars in
# one cmd invocation. The redirect silences vcvars' banner without hiding errors.
$cfg = "cmake -S `"$src`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
       "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DLLAMA_CPP_DIR=`"$LlamaCppDir`""
# The inner 2>&1 is cmd's, not PowerShell's: llama.cpp's CMakeLists writes
# informational lines to stderr, and under $ErrorActionPreference = "Stop" any
# stderr from a native command becomes a terminating error even on exit 0.
cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" -j 2>&1"
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Write-Host "`nbuilt into $build\bin" -ForegroundColor Green
Get-ChildItem "$build\bin\*.exe" | ForEach-Object { Write-Host "  $($_.Name)" }
