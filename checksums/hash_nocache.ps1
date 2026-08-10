<#
.SYNOPSIS
  SHA-256 of a file read with FILE_FLAG_NO_BUFFERING, bypassing the page cache.

.DESCRIPTION
  Windows-only by nature, and the reason it exists is a failure mode this
  machine has produced twice in different directions:

    August 2026  the page cache held *good* pages while the disk held corrupt
                 ones, so hashing right after the copy passed and only a reboot
                 exposed the damage.
    This time    the disk holds good data and a *cached page* is corrupt, so an
                 ordinary hash fails on a file that is actually intact.

  An ordinary Get-FileHash cannot tell those apart, because it reads through
  the cache in both cases. This reads from the device, so a mismatch here means
  the bytes on the platter are wrong and a mismatch only in the buffered hash
  means memory or a driver scribbled the cache.

.EXAMPLE
  .\hash_nocache.ps1 D:\llms\UD-Q6_K\GLM-5.2-UD-Q6_K-00002-of-00014.gguf
#>
param([Parameter(Mandatory=$true)][string] $Path,
      [int] $ChunkMiB = 8)

$sig = @'
using System;
using System.Runtime.InteropServices;
public class NoCacheIO {
  [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
  public static extern IntPtr CreateFileW(string p, uint acc, uint share, IntPtr sa, uint disp, uint flags, IntPtr tmpl);
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern bool ReadFile(IntPtr h, IntPtr buf, uint n, out uint got, IntPtr ov);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
'@
if (-not ("NoCacheIO" -as [type])) { Add-Type -TypeDefinition $sig }

$GENERIC_READ = [uint32]2147483648
$OPEN_EXISTING = [uint32]3
$FLAG_NO_BUFFERING = [uint32]536870912   # 0x20000000

$size = (Get-Item $Path).Length
$chunk = $ChunkMiB * 1MB
$sector = 4096

$h = [NoCacheIO]::CreateFileW($Path, $GENERIC_READ, [uint32]1, [IntPtr]::Zero,
                              $OPEN_EXISTING, $FLAG_NO_BUFFERING, [IntPtr]::Zero)
if ($h -eq [IntPtr]::new(-1)) { throw "CreateFile failed: $([Runtime.InteropServices.Marshal]::GetLastWin32Error())" }

# NO_BUFFERING requires a sector-aligned buffer, which AllocHGlobal does not
# promise; over-allocate and round the pointer up.
$raw = [Runtime.InteropServices.Marshal]::AllocHGlobal($chunk + $sector)
$buf = [IntPtr](([int64]$raw + $sector - 1) -band -$sector)
$sha = [Security.Cryptography.SHA256]::Create()
$managed = New-Object byte[] $chunk

try {
    $done = [int64]0
    while ($done -lt $size) {
        # Reads must be whole sectors, so the tail is over-read and trimmed.
        $want = [Math]::Min([int64]$chunk, $size - $done)
        $toRead = [uint32](([int64]$want + $sector - 1) -band -$sector)
        $got = 0
        if (-not [NoCacheIO]::ReadFile($h, $buf, $toRead, [ref]$got, [IntPtr]::Zero)) {
            throw "ReadFile failed at $done : $([Runtime.InteropServices.Marshal]::GetLastWin32Error())"
        }
        if ($got -eq 0) { break }
        $use = [int][Math]::Min([int64]$got, $want)
        [Runtime.InteropServices.Marshal]::Copy($buf, $managed, 0, $use)
        $null = $sha.TransformBlock($managed, 0, $use, $null, 0)
        $done += $use
    }
    $null = $sha.TransformFinalBlock((New-Object byte[] 0), 0, 0)
    $hex = ([BitConverter]::ToString($sha.Hash) -replace '-','').ToLower()
    "{0}  {1}" -f $hex, (Split-Path $Path -Leaf)
}
finally {
    [Runtime.InteropServices.Marshal]::FreeHGlobal($raw)
    [NoCacheIO]::CloseHandle($h) | Out-Null
}
