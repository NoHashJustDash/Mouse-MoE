# SPDX-License-Identifier: GPL-3.0-or-later
# Start or resume a training run.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train.ps1 -Config configs\m1_46m.toml
#   ... -Resume        continue runs\<name>\checkpoints\latest.pt
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Config,
    [switch]$Resume,
    [int]$MaxSteps = 0
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts\setup.ps1 first.' }
$arguments = @('-I', '-m', 'mouse_moe.train', '--config', (Resolve-Path (Join-Path $root $Config)))
if ($Resume) { $arguments += '--resume' }
if ($MaxSteps -gt 0) { $arguments += @('--max-steps', $MaxSteps) }
Push-Location $root
try {
    & $python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
