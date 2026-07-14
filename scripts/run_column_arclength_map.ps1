<#
.SYNOPSIS
  Guarded production smoke and 50x50 fail-fast column-arclength map.
#>
[CmdletBinding()]
param(
    [int]$WaitForPid = 0,
    [string]$OutRoot = "outputs/column_arclength_diagnostic"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($WaitForPid -gt 0) {
    Write-Host "Waiting for PID $WaitForPid before starting arclength smoke..."
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
}

$Common = @(
    "scripts/run_gain_map.py",
    "--executor", "inprocess", "--mode", "warmstart",
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--inproc-fail-fast",
    "--column-arclength-recovery",
    "--column-arclength-ds", "0.02",
    "--column-arclength-max-steps", "80",
    "--fold-skip-patience", "4",
    "--inproc-schur-cache-size", "2",
    "--signal-detuning-mhz", "100",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "6",
    "--pump-mode-count", "10", "--nt", "40",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--no-signal-spectrum",
    "--ipm-dir", "outputs/ipm_python_design",
    "--pump-power-min-dbm", "-35", "--pump-power-max-dbm", "-23",
    "--overwrite"
)

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$Smoke = Join-Path $OutRoot "smoke_8p2117ghz"
$Full = Join-Path $OutRoot "map_50x50_m35_m23_7p5_8p5"

Write-Host "Running one-column arclength smoke: $Smoke"
& python @Common `
    --n-power 61 --n-frequency 1 --frequency-chunk-size 0 `
    --pump-freq-min-ghz 8.21173469384667 `
    --pump-freq-max-ghz 8.21173469384667 `
    --outdir $Smoke
if ($LASTEXITCODE -ne 0) { throw "arclength smoke failed with exit $LASTEXITCODE" }

$Recovered = @(Import-Csv (Join-Path $Smoke "map_points.csv") | Where-Object {
    $_.status -eq "PASS" -and $_.pump_predictor -match "arclength"
})
if ($Recovered.Count -eq 0) {
    throw "smoke produced no arclength-recovered PASS; full map not started"
}
Write-Host "Smoke recovered $($Recovered.Count) arclength target(s); starting full map."

& python @Common `
    --n-power 50 --n-frequency 50 --frequency-chunk-size 10 `
    --pump-freq-min-ghz 7.5 --pump-freq-max-ghz 8.5 `
    --signal-offset-count-per-side 5 --signal-offset-step-mhz 250 `
    --outdir $Full
if ($LASTEXITCODE -ne 0) { throw "full arclength map failed with exit $LASTEXITCODE" }

