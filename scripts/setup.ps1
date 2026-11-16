# SPDX-License-Identifier: GPL-3.0-or-later
# One-command training environment: creates .venv, installs pinned PyTorch (CUDA 12.8)
# and tools, verifies the GPU, then runs the test suite.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup.ps1
#   ... -Full   also run the tests that need downloaded data, TensorRT, lc0 and BT4
#
# Either way no selected test may be skipped.
# Requires python.org Python 3.12 (py launcher or default per-user/all-users path).
[CmdletBinding()]
param([switch]$SkipTests, [switch]$Full)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$root = Split-Path -Parent $PSScriptRoot
$torchIndex = 'https://download.pytorch.org/whl/cu128'
$torchPin = 'torch==2.9.1+cu128'

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $File $($Arguments -join ' ')" }
}

function Find-Python312 {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $resolved = & $launcher.Source -3.12 -I -c 'import sys; print(sys.executable)'
            if ($LASTEXITCODE -eq 0 -and $resolved) { $candidates = @($resolved) + $candidates }
        } catch {
            Write-Host 'py launcher has no Python 3.12; checking default install paths.'
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -match 'WindowsApps') { continue }
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $ok = & $candidate -I -c 'import sys; print(sys.version_info[:2] == (3, 12))'
            if ($LASTEXITCODE -eq 0 -and $ok -eq 'True') { return $candidate }
        }
    }
    throw 'Python 3.12 from python.org was not found. Install it (https://www.python.org/downloads/release/python-31210/), then rerun.'
}

Push-Location $root
try {
    $python = Find-Python312
    Write-Host "Using $python"
    $venvPython = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Checked $python @('-I', '-m', 'venv', '.venv')
    }
    Invoke-Checked $venvPython @('-I', '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', $torchIndex, $torchPin)
    Invoke-Checked $venvPython @('-I', '-m', 'pip', 'install', '--only-binary=:all:', '-r', 'requirements-train.txt')
    Invoke-Checked $venvPython @('-I', '-m', 'pip', 'install', '--no-build-isolation', '--no-deps', '-e', '.')
    Invoke-Checked $venvPython @('-I', '-m', 'pip', 'check')
    $lock = & $venvPython -I -m pip freeze --exclude-editable
    if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed' }
    ($lock -join "`n") + "`n" | Out-File -FilePath (Join-Path $root 'requirements-train.lock') -Encoding ascii

    $gpu = & $venvPython -I -c @'
import torch
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
major, minor = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
assert f"sm_{major}{minor}" in arches, f"this PyTorch build lacks sm_{major}{minor}: {arches}"
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print(f"{torch.__version__} | {torch.cuda.get_device_name(0)} | sm_{major}{minor} | CUDA {torch.version.cuda}")
'@
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed' }
    Write-Host "GPU OK: $gpu"

    if (-not $SkipTests) {
        Invoke-Checked $venvPython @('-I', '-m', 'ruff', 'check', 'src', 'tests')
        Invoke-Checked $venvPython @('-I', '-m', 'mypy', '--strict', 'src/mouse_moe/policy_map.py', 'src/mouse_moe/v6.py',
            'src/mouse_moe/shards.py', 'src/mouse_moe/dataset.py', 'src/mouse_moe/config.py',
            'src/mouse_moe/budget.py', 'src/mouse_moe/validate_data.py')
        $report = Join-Path $root 'test-report.xml'
        $selection = if ($Full) { @() } else { @('-m', 'not data and not tensorrt and not lc0') }
        & $venvPython -I -m pytest -p no:cacheprovider -rs @selection "--junitxml=$report"
        if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
        $suite = ([xml](Get-Content -LiteralPath $report -Raw)).testsuites.testsuite
        if ([int]$suite.skipped -ne 0) { throw "$($suite.skipped) test(s) were skipped; the environment is incomplete" }
        Write-Host "All $($suite.tests) tests passed, none skipped."
    }
    Write-Host 'Environment ready. Next: see docs\2-getting-started.md.'
} finally {
    Pop-Location
}
