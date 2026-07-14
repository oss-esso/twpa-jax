<#
.SYNOPSIS
  Controlled methods 31-37 recovery comparison on the requested 50x20 pump map.

.DESCRIPTION
  Runs one intra-cell continuation method at a time. Fail-fast is disabled so a
  failed warm solve retries the plain parent and then invokes the selected
  continuation method. Patience 4 bounds repeated recovery work before the rest
  of a column is skipped. Each completed map is plotted into its own output
  directory. The final analysis compares status/gain against method 31 and
  records the first attempted pump error in every frequency column.
#>
[CmdletBinding()]
param(
    [string]$OutRoot = "outputs/intracell_continuation_31_37_recovery_p4_sb6",
    [string[]]$Only = @(),
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Common = @(
    "--executor", "inprocess", "--mode", "warmstart",
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--fold-skip-patience", "4",
    "--inproc-schur-cache-size", "2",
    "--signal-detuning-mhz", "100",
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--skip-baselines", "--sidebands", "6", "--signal-workers", "6",
    "--pump-mode-count", "10", "--nt", "40",
    "--inproc-max-newton", "16", "--inproc-solve-deadline-s", "14",
    "--inproc-fallback-fixed-steps", "8",
    "--inproc-continuation-deadline-s", "45",
    "--n-power", "50", "--n-frequency", "20",
    "--frequency-chunk-size", "10", "--no-signal-spectrum",
    "--circuit-dir", "outputs/ipm_python_design",
    "--pump-power-min-dbm", "-32", "--pump-power-max-dbm", "-20",
    "--pump-freq-min-ghz", "8.1", "--pump-freq-max-ghz", "8.3"
)

$Methods = @(
    @{ id = "m31_fixed";             mode = "fixed" },
    @{ id = "m32_adaptive_copy";     mode = "adaptive_copy" },
    @{ id = "m33_adaptive_secant";   mode = "adaptive_secant" },
    @{ id = "m34_adaptive_tangent";  mode = "adaptive_tangent" },
    @{ id = "m35_affine";            mode = "affine" },
    @{ id = "m36_ptc";               mode = "ptc" },
    @{ id = "m37_arclength";         mode = "arclength" }
)

$Selected = @()
if ($Only.Count -gt 0) { $Selected = ($Only -join ",").Split(",").Trim() }
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$Log = Join-Path $OutRoot "campaign_log.csv"
if (-not (Test-Path $Log)) {
    "id,status,run_s,started" | Set-Content -Encoding utf8 $Log
}

foreach ($Method in $Methods) {
    if ($Selected.Count -gt 0 -and $Selected -notcontains $Method.id) { continue }
    $Out = Join-Path $OutRoot $Method.id
    if ($Resume -and (Test-Path (Join-Path $Out "map_summary.json"))) {
        Write-Host "SKIP $($Method.id) (complete)" -ForegroundColor DarkGray
        continue
    }
    $Run = @("python", "scripts/run_gain_map.py") + $Common + @(
        "--inproc-continuation", $Method.mode, "--outdir", $Out
    )
    if (-not $Resume) { $Run += "--overwrite" }
    Write-Host "`nRUN $($Method.id): $($Run -join ' ')" -ForegroundColor Cyan
    if ($DryRun) { continue }
    $Started = (Get-Date).ToString("s")
    $Watch = [System.Diagnostics.Stopwatch]::StartNew()
    & $Run[0] $Run[1..($Run.Length - 1)]
    $RunCode = $LASTEXITCODE
    $Watch.Stop()
    if ($RunCode -ne 0) {
        "$($Method.id),failed,$([int]$Watch.Elapsed.TotalSeconds),$Started" |
            Add-Content -Encoding utf8 $Log
        throw "$($Method.id) failed with exit $RunCode"
    }
    $Plot = @(
        "python", "scripts/plot_gain_map.py", "--run-dir", $Out,
        "--outdir", $Out, "--ipm-dir", "outputs/__maps_only_no_circuit__"
    )
    & $Plot[0] $Plot[1..($Plot.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "plot for $($Method.id) failed" }
    "$($Method.id),ok,$([int]$Watch.Elapsed.TotalSeconds),$Started" |
        Add-Content -Encoding utf8 $Log
}

if (-not $DryRun) {
    python scripts/analyze_intracell_campaign.py $OutRoot --reference m31_fixed
    if ($LASTEXITCODE -ne 0) { throw "campaign analysis failed" }
}
