# Build moeserv.dll. Mirrors ../nano-glm/build.ps1's toolchain discovery so
# both projects compile with the same MSVC, which matters here for the same
# reason it matters there: FP contraction differs between compilers and this
# library's output is compared byte for byte against a llama.cpp built with one
# of them (repo CLAUDE.md).
#
#   .\build.ps1              # into build\bin\moeserv.dll
#   .\build.ps1 -Clean       # reconfigure from scratch

[CmdletBinding()]
param(
    [string] $LlamaCppDir = "C:\Users\oleksandr\Desktop\llama.cpp",
    [switch] $Clean
)

$ErrorActionPreference = "Stop"
$here  = $PSScriptRoot
$build = Join-Path $here "build"

if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found - install Visual Studio" }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$ninja  = Get-Command ninja -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $ninja) { $ninja = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" }

# CMAKE_EXPORT_COMPILE_COMMANDS must be passed as -D (a cache variable):
# llama.cpp sets it as a normal variable inside ggml's directory scope, which
# exports ggml's own sources but not this project's — build/compile_commands.json
# then exists, looks plausible, and covers zero moe-serv files. -D makes it
# global so VS Code's C/C++ extension can resolve our includes (../.vscode/).
$cfg = "cmake -S `"$here`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
       "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DLLAMA_CPP_DIR=`"$LlamaCppDir`" " +
       "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# Through cmd /c because cmake writes progress to stderr and
# $ErrorActionPreference = "Stop" turns any native stderr into a terminating
# error, even on exit 0.
cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1" | Out-Null
if ($LASTEXITCODE -ne 0) {
    cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1"
    throw "cmake configure failed"
}

cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" --target moeserv -j 2>&1" | Tee-Object -Variable out | Out-Null
if ($LASTEXITCODE -ne 0) { $out | Write-Host; throw "build failed" }

$dll = Join-Path $build "bin\moeserv.dll"
if (-not (Test-Path $dll)) { throw "expected $dll" }
Write-Host "built $dll" -ForegroundColor Green
Write-Host ""
Write-Host "  python gate.py              # correctness, against the stub model" -ForegroundColor DarkGray
Write-Host "  python gate.py --vs-stock   # and what owning the weights costs" -ForegroundColor DarkGray
