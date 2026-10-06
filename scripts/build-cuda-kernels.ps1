# Regenerates plugins/cuda/kernels/kernels.ptx from kernels.cu with nvcc (needs the CUDA toolkit
# and MSVC Build Tools). The PTX is committed, so building the Rust crate never needs nvcc.
#
# The PTX targets a low virtual architecture (compute_75) so the driver JIT can compile it for any
# newer GPU (Ampere, Ada, Blackwell sm_120...). The toolkit may be newer than the installed driver, so
# the emitted `.version` is clamped to $PtxVersion (the ISA the oldest supported driver accepts).
param(
    [string]$Cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    [string]$Arch = "compute_75",
    [string]$PtxVersion = "8.0"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dir = Join-Path $root "plugins/cuda/kernels"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) {
    $vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
}
$nvcc = Join-Path $Cuda "bin\nvcc.exe"
$src = Join-Path $dir "kernels.cu"
$ptx = Join-Path $dir "kernels.ptx"
cmd /c "`"$vcvars`" >nul && `"$nvcc`" -ptx -arch=$Arch -O3 -fmad=true -lineinfo -o `"$ptx`" `"$src`""
if ($LASTEXITCODE -ne 0) { throw "nvcc failed" }
$text = [IO.File]::ReadAllText($ptx)
$text = [regex]::Replace($text, '(?m)^\.version\s+[\d.]+', ".version $PtxVersion")
[IO.File]::WriteAllText($ptx, $text.Replace("`r`n", "`n"))
Write-Host "wrote $ptx"
Select-String -Path $ptx -Pattern '^\.(version|target)'
