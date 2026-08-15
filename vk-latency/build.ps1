# Build vk-latency.exe. Toolchain discovery mirrors ../moe-serv/build.ps1 —
# not for FP reasons here (nothing is compared bit for bit), just so every
# project on this machine builds the same way.
#
#   .\build.ps1              # into build\bin\vk-latency.exe
#   .\build.ps1 -Clean       # reconfigure from scratch

[CmdletBinding()]
param(
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

$cfg = "cmake -S `"$here`" -B `"$build`" -G Ninja -DCMAKE_BUILD_TYPE=Release " +
       "-DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

# Through cmd /c because cmake writes progress to stderr and
# $ErrorActionPreference = "Stop" turns any native stderr into a terminating
# error, even on exit 0.
cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1" | Out-Null
if ($LASTEXITCODE -ne 0) {
    cmd /c "`"$vcvars`" >nul 2>&1 && $cfg 2>&1"
    throw "cmake configure failed"
}

cmd /c "`"$vcvars`" >nul 2>&1 && cmake --build `"$build`" -j 2>&1" | Tee-Object -Variable out | Out-Null
if ($LASTEXITCODE -ne 0) { $out | Write-Host; throw "build failed" }

$exe = Join-Path $build "bin\vk-latency.exe"
if (-not (Test-Path $exe)) { throw "expected $exe" }
Write-Host "built $exe" -ForegroundColor Green
