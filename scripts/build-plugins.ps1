# Builds every plugin under plugins/<name> and copies the shared library into plugins/bin.
# Convention: plugins/<name> is the crate `pytorches-plugin-<name>` producing
# pytorches_plugin_<name>.{dll,so,dylib}.
param([string]$Profile = "release")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$bin = Join-Path $root "plugins/bin"
New-Item -ItemType Directory -Force $bin | Out-Null
$flag = if ($Profile -eq "release") { "--release" } else { "" }
Get-ChildItem (Join-Path $root "plugins") -Directory | Where-Object { Test-Path "$($_.FullName)/Cargo.toml" } | ForEach-Object {
    $name = $_.Name
    Write-Host "building plugin: $name"
    cargo build $flag -p "pytorches-plugin-$name" --manifest-path (Join-Path $root "Cargo.toml")
    if ($LASTEXITCODE -ne 0) { throw "plugin $name failed to build" }
    $lib = Join-Path $root "target/$Profile/pytorches_plugin_$($name.Replace('-','_')).dll"
    Copy-Item $lib $bin -Force
}
Get-ChildItem $bin
