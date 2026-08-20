# Overnight 30x30 gain-map campaign for the canonical IPM coupler-count family.
#
# Runs designs/ipm_2c, ipm_3c, ipm_7c, ipm_20c sequentially, smallest first, so
# a run that overruns still leaves the cheap designs finished. Each design gets
# its own output subdirectory and its own log; a design that fails does not stop
# the ones after it.
#
# Usage:
#   powershell.exe -ExecutionPolicy Bypass -File .\scripts\run_coupler_count_30x30.ps1
#   ... -Only ipm_2c,ipm_3c        run a subset
#   ... -DryRun                    print the commands without running them

param(
    [string]$OutputRoot = "outputs\coupler_count_30x30",
    [int]$Workers = 2,
    [int]$NPower = 30,
    [int]$NFrequency = 30,
    [double]$PowerMinDbm = -30.0,
    [double]$PowerMaxDbm = -20.0,
    [double]$FreqMinGhz = 7.0,
    [double]$FreqMaxGhz = 8.0,
    [string[]]$Only = @(),
    [switch]$NoQuantumEfficiency,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repoRoot

# Smallest first: 16k, 26k, 65k, 192k elements.
$designs = @("ipm_2c", "ipm_3c", "ipm_7c", "ipm_20c")
if ($Only.Count -gt 0) {
    $designs = $designs | Where-Object { $Only -contains $_ }
}
if ($designs.Count -eq 0) {
    Write-Host "no designs selected"
    exit 1
}

$designRoot = Join-Path $OutputRoot "designs"
$mapRoot = Join-Path $OutputRoot "maps"
$logRoot = Join-Path $OutputRoot "logs"
foreach ($dir in @($mapRoot, $logRoot)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

$startedAt = Get-Date
Write-Host ("campaign start: " + $startedAt.ToString("yyyy-MM-dd HH:mm:ss"))
Write-Host ("grid: {0}x{1} power [{2}, {3}] dBm freq [{4}, {5}] GHz workers={6}" -f `
    $NPower, $NFrequency, $PowerMinDbm, $PowerMaxDbm, $FreqMinGhz, $FreqMaxGhz, $Workers)
Write-Host ("designs: " + ($designs -join ", "))

$results = @()
foreach ($design in $designs) {
    $circuitDir = Join-Path $designRoot $design
    if (-not (Test-Path -LiteralPath $circuitDir)) {
        Write-Host ("[{0}] SKIP: circuit directory not built: {1}" -f $design, $circuitDir)
        $results += [pscustomobject]@{ design = $design; status = "NO_CIRCUIT"; minutes = 0 }
        continue
    }

    $runDir = Join-Path $mapRoot $design
    $logPath = Join-Path $logRoot ($design + ".log")

    $mapArgs = @(
        "workflows\run_gain_map_and_plots.py",
        "--design", $circuitDir,
        "--output-dir", $runDir,
        "--fast",
        "--n-power", $NPower,
        "--n-frequency", $NFrequency,
        "--pump-power-min-dbm", $PowerMinDbm,
        "--pump-power-max-dbm", $PowerMaxDbm,
        "--pump-freq-min-ghz", $FreqMinGhz,
        "--pump-freq-max-ghz", $FreqMaxGhz,
        "--frequency-workers", $Workers,
        "--frequency-chunk-size", $Workers,
        "--signal-backend", "direct",
        "--overwrite"
    )
    if ($NoQuantumEfficiency) {
        $mapArgs += "--no-quantum-efficiency"
    }

    Write-Host ""
    Write-Host ("[{0}] start {1}" -f $design, (Get-Date).ToString("HH:mm:ss"))
    Write-Host ("[{0}] log: {1}" -f $design, $logPath)
    if ($DryRun) {
        Write-Host ("python " + ($mapArgs -join " "))
        continue
    }

    $designStart = Get-Date
    & python @mapArgs *>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    $minutes = [math]::Round(((Get-Date) - $designStart).TotalMinutes, 1)

    if ($exitCode -eq 0) {
        Write-Host ("[{0}] DONE in {1} min" -f $design, $minutes)
        $results += [pscustomobject]@{ design = $design; status = "OK"; minutes = $minutes }
    } else {
        Write-Host ("[{0}] FAILED exit={1} after {2} min; continuing" -f $design, $exitCode, $minutes)
        $results += [pscustomobject]@{ design = $design; status = "FAILED($exitCode)"; minutes = $minutes }
    }
}

if ($DryRun) {
    exit 0
}

$totalMinutes = [math]::Round(((Get-Date) - $startedAt).TotalMinutes, 1)
Write-Host ""
Write-Host ("campaign end: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss") + " total " + $totalMinutes + " min")
$results | Format-Table -AutoSize | Out-String | Write-Host

$summaryPath = Join-Path $OutputRoot "campaign_summary.csv"
$results | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
Write-Host ("summary: " + $summaryPath)
