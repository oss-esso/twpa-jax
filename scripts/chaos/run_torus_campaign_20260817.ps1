# Unattended torus-branch campaign, 2026-08-17.
# PowerShell 5.1 compatible.  The Python controller enforces the 7.5 hour
# budget, writes run.log and all point artifacts atomically, and runs Stage G
# after a graceful deadline stop.

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..\..')

$scriptName = 'run_torus_campaign_20260817.ps1'
$currentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
$parentProcessId = if ($currentProcess) {
    [int]$currentProcess.ParentProcessId
} else {
    0
}
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.ProcessId -ne $parentProcessId -and
    $_.CommandLine -and $_.CommandLine -match (
        '(?i)-File\s+["'']?.*' + [regex]::Escape($scriptName)
    )
}
if ($existing) {
    $pids = ($existing | ForEach-Object { $_.ProcessId }) -join ', '
    throw "An existing $scriptName process is running (PID $pids). Refusing a second campaign."
}

$outputRoot = Join-Path (Get-Location) 'outputs\chaos\torus_campaign_20260817'
if (-not (Test-Path $outputRoot)) {
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
}

& python scripts\chaos\torus_campaign_20260817.py `
    --campaign `
    --output-root $outputRoot `
    --rss-limit-gb 6.0
exit $LASTEXITCODE
