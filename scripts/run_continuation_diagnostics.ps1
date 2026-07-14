<#
.SYNOPSIS
  Controlled continuation diagnostics on reduced 2c pump-map domains.

.DESCRIPTION
  Runs four gated phases:
    sentinel - 41x3 control grid with skipping disabled in the key references.
    period   - 41x17 grid spanning one observed frequency period.
    fold     - pump-only pseudo-arclength fold locator at three frequencies.
    branch   - small post-fold pseudo-arclength map probe; run only after fold.
    spectrum - one-period branch map with per-cell spectra; run last.

  Use -DryRun to inspect exact commands and -Only to select configuration ids.
#>
[CmdletBinding()]
param(
    [ValidateSet("sentinel", "period", "fold", "branch", "spectrum")]
    [string]$Phase = "sentinel",
    [string]$OutRoot = "outputs/continuation_diagnostics",
    [string[]]$Only = @(),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = "python"
$RunGain = "scripts/run_gain_map.py"
$Circuit = "outputs/ipm_python_design"

$Base = @(
    "--executor", "inprocess", "--mode", "warmstart",
    "--circuit-dir", $Circuit,
    "--pump-power-min-dbm", "-31", "--pump-power-max-dbm", "-23",
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--inproc-schur-cache-size", "2",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--pump-mode-count", "10", "--nt", "40",
    "--signal-detuning-mhz", "100",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "1",
    "--no-signal-spectrum", "--overwrite"
)

$SentinelGrid = @(
    "--n-power", "41", "--n-frequency", "3",
    "--pump-freq-min-ghz", "7.7857142857",
    "--pump-freq-max-ghz", "8.1530612245",
    "--frequency-chunk-size", "0"
)

$PeriodGrid = @(
    "--n-power", "41", "--n-frequency", "17",
    "--pump-freq-min-ghz", "7.9693877551",
    "--pump-freq-max-ghz", "8.3571428571"
)

$SentinelConfigs = @(
    @{ id = "s00_exhaustive"; flags = @("--traversal", "column", "--fold-skip-patience", "0") },
    @{ id = "s01_failfast_noskip"; flags = @("--traversal", "column", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s02_failfast_p2"; flags = @("--traversal", "column", "--inproc-fail-fast", "--fold-skip-patience", "2") },
    @{ id = "s03_failfast_p4"; flags = @("--traversal", "column", "--inproc-fail-fast", "--fold-skip-patience", "4") },
    @{ id = "s04_altparent_ff"; flags = @("--traversal", "column", "--predictor", "power_secant", "--recovery", "alt_parent", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s05_bridge_ff"; flags = @("--traversal", "column", "--predictor", "power_secant", "--recovery", "bridge", "--bridge-mode", "adaptive", "--bridge-steps", "2", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s06_backbone_copy_ff"; flags = @("--traversal", "backbone", "--predictor", "copy", "--recovery", "none", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s07_portfolio_best_ff"; flags = @("--traversal", "backbone", "--predictor", "portfolio", "--portfolio-policy", "best", "--recovery", "none", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s08_portfolio_ranked_ff"; flags = @("--traversal", "backbone", "--predictor", "portfolio", "--portfolio-policy", "ranked", "--recovery", "none", "--inproc-fail-fast", "--fold-skip-patience", "0") },
    @{ id = "s09_portfolio_bridge_ff"; flags = @("--traversal", "backbone", "--predictor", "portfolio", "--portfolio-policy", "best", "--recovery", "bridge", "--bridge-mode", "adaptive", "--bridge-steps", "2", "--inproc-fail-fast", "--fold-skip-patience", "0") }
)

$PeriodConfigs = @(
    @{ id = "p00_exhaustive"; flags = @("--traversal", "column", "--fold-skip-patience", "0", "--frequency-chunk-size", "4") },
    @{ id = "p01_failfast_noskip"; flags = @("--traversal", "column", "--inproc-fail-fast", "--fold-skip-patience", "0", "--frequency-chunk-size", "4") },
    @{ id = "p02_failfast_p4"; flags = @("--traversal", "column", "--inproc-fail-fast", "--fold-skip-patience", "4", "--frequency-chunk-size", "4") },
    @{ id = "p03_altparent_ff_p3"; flags = @("--traversal", "column", "--predictor", "power_secant", "--recovery", "alt_parent", "--inproc-fail-fast", "--fold-skip-patience", "3", "--frequency-chunk-size", "0") },
    @{ id = "p04_bridge_ff_p3"; flags = @("--traversal", "column", "--predictor", "power_secant", "--recovery", "bridge", "--bridge-mode", "adaptive", "--bridge-steps", "2", "--inproc-fail-fast", "--fold-skip-patience", "3", "--frequency-chunk-size", "0") },
    @{ id = "p05_backbone_bridge_single"; flags = @("--traversal", "backbone", "--predictor", "portfolio", "--recovery", "bridge", "--bridge-mode", "adaptive", "--bridge-steps", "2", "--inproc-fail-fast", "--fold-skip-patience", "3", "--frequency-chunk-size", "0") },
    @{ id = "p06_backbone_bridge_chunked"; flags = @("--traversal", "backbone", "--predictor", "portfolio", "--recovery", "bridge", "--bridge-mode", "adaptive", "--bridge-steps", "2", "--inproc-fail-fast", "--fold-skip-patience", "3", "--local-traversal-chunks", "--frequency-chunk-size", "4") }
)

function Invoke-Run([string]$Id, [string[]]$Grid, [string[]]$Flags) {
    $selected = @($Only -join ",").Split(",", [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($selected.Count -gt 0 -and $selected -notcontains $Id) { return }
    $out = Join-Path $OutRoot $Id
    $cmd = @($Python, $RunGain) + $Base + $Grid + $Flags + @("--outdir", $out)
    Write-Host ">> $Id" -ForegroundColor Cyan
    Write-Host ("   " + ($cmd -join " ")) -ForegroundColor DarkGray
    if ($DryRun) { return }
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "$Id failed with exit code $LASTEXITCODE" }
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
}

if ($Phase -eq "sentinel") {
    foreach ($cfg in $SentinelConfigs) {
        Invoke-Run $cfg.id $SentinelGrid $cfg.flags
    }
}

if ($Phase -eq "period") {
    foreach ($cfg in $PeriodConfigs) {
        Invoke-Run $cfg.id $PeriodGrid $cfg.flags
    }
}

if ($Phase -eq "fold") {
    Invoke-Run "f00_fold_follow" @(
        "--n-power", "1", "--n-frequency", "3",
        "--pump-freq-min-ghz", "7.7857142857",
        "--pump-freq-max-ghz", "8.1530612245",
        "--frequency-chunk-size", "0"
    ) @("--fold-follow", "--traversal", "column")
}

if ($Phase -eq "branch") {
    Invoke-Run "a00_arclength_probe" @(
        "--n-power", "21", "--n-frequency", "3",
        "--pump-power-min-dbm", "-29", "--pump-power-max-dbm", "-23",
        "--pump-freq-min-ghz", "7.7857142857",
        "--pump-freq-max-ghz", "8.1530612245",
        "--frequency-chunk-size", "0"
    ) @(
        "--traversal", "backbone", "--predictor", "portfolio",
        "--portfolio-policy", "best", "--recovery", "none",
        "--fold-policy", "arclength", "--inproc-fail-fast",
        "--fold-skip-patience", "0"
    )
}

if ($Phase -eq "spectrum") {
    Invoke-Run "z00_arclength_spectrum" @(
        "--n-power", "21", "--n-frequency", "9",
        "--pump-power-min-dbm", "-29", "--pump-power-max-dbm", "-23",
        "--pump-freq-min-ghz", "7.9693877551",
        "--pump-freq-max-ghz", "8.3571428571",
        "--local-traversal-chunks", "--frequency-chunk-size", "3"
    ) @(
        "--traversal", "backbone", "--predictor", "portfolio",
        "--portfolio-policy", "best", "--recovery", "none",
        "--fold-policy", "arclength", "--inproc-fail-fast",
        "--fold-skip-patience", "0", "--signal-spectrum",
        "--sidebands", "6", "--signal-workers", "1"
    )
}
