# SPDX-License-Identifier: GPL-3.0-or-later
# Measure training speed, GPU memory and data-loader speed for a config.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\benchmark.ps1 -Config configs\m1_46m.toml
[CmdletBinding()]
param([string]$Config = 'configs\m1_46m.toml')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\setup.ps1 first.' }
Push-Location $root
try {
    & $python -I -m mouse_moe.benchmark --config (Resolve-Path (Join-Path $root $Config))
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
