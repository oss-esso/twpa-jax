param(
    [switch]$Force,
    [string]$OutputDir = ""
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
Set-Location $repoRoot

# Native Windows JAX is CPU-only. Prefer the project-local WSL2 environment
# when it is available so this entry point can still launch the CUDA session.
$wslDistribution = "Ubuntu-22.04"
$wslPython = "/home/edo/twpa_jax_venv/bin/python"
$useWsl = $false
if ($env:OS -eq "Windows_NT") {
    & wsl -d $wslDistribution -- test -x $wslPython 2>$null
    $useWsl = ($LASTEXITCODE -eq 0)
}

function Convert-ToSessionPath([string]$Path) {
    if (-not $useWsl) {
        return $Path
    }
    $normalized = $Path -replace "\\", "/"
    if ($normalized -notmatch "^([A-Za-z]):/(.*)$") {
        throw "Unable to convert path for WSL2: $Path"
    }
    return ("/mnt/{0}/{1}" -f $matches[1].ToLowerInvariant(), $matches[2])
}

function Invoke-SessionPython([string]$Module, [string[]]$Arguments) {
    if ($useWsl) {
        & wsl -d $wslDistribution -- $wslPython -m $Module @Arguments
    } else {
        & python -m $Module @Arguments
    }
    $script:LastSessionPythonExitCode = [int]$LASTEXITCODE
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = Join-Path $repoRoot (Join-Path "outputs\chaos\gpu_sessions" ("gpu_" + $stamp))
} else {
    $runDir = [System.IO.Path]::GetFullPath($OutputDir)
}
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$preflightPath = Join-Path $runDir "preflight.json"
$precisionPath = Join-Path $runDir "precision.json"
$benchmarkPath = Join-Path $runDir "benchmark.csv"
$summaryPath = Join-Path $runDir "summary.txt"
$fixturePath = Join-Path $repoRoot "tests\data\fdtd_reference\ipm_2c_7p9.npz"

$fixtureSessionPath = Convert-ToSessionPath $fixturePath
$preflightSessionPath = Convert-ToSessionPath $preflightPath
$precisionSessionPath = Convert-ToSessionPath $precisionPath
$benchmarkSessionPath = Convert-ToSessionPath $benchmarkPath

Write-Host ("GPU session output: " + $runDir)
if ($useWsl) {
    Write-Host ("[runtime] using WSL2 " + $wslDistribution + " CUDA environment")
} else {
    Write-Host "[runtime] using native Windows Python"
}

if ($Force -or -not (Test-Path -LiteralPath $preflightPath)) {
    Write-Host "[preflight] running"
    Invoke-SessionPython "scripts.chaos.gpu_preflight" @(
        "--fixture", $fixtureSessionPath,
        "--output", $preflightSessionPath
    )
    $stageCode = $script:LastSessionPythonExitCode
    if ($stageCode -ne 0) {
        Write-Host ("[preflight] FAILED with exit code " + $stageCode)
        exit $stageCode
    }
} else {
    Write-Host "[preflight] existing output; skipped"
}

$preflight = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
if ($preflight.status -ne "PASS") {
    Write-Host "[preflight] output is not PASS; stopping"
    exit 1
}

if ($Force -or -not (Test-Path -LiteralPath $precisionPath)) {
    Write-Host "[precision] running"
    Invoke-SessionPython "scripts.chaos.measure_kernel_precision" @(
        "--jax-device", "gpu",
        "--output", $precisionSessionPath
    )
    $stageCode = $script:LastSessionPythonExitCode
    if ($stageCode -ne 0) {
        Write-Host ("[precision] FAILED with exit code " + $stageCode)
        exit $stageCode
    }
} else {
    Write-Host "[precision] existing output; skipped"
}

$precision = Get-Content -LiteralPath $precisionPath -Raw | ConvertFrom-Json
if ($precision.verdict -ne "GO") {
    Write-Host ("[precision] verdict is " + $precision.verdict + "; stopping")
    exit 1
}

if ($Force -or -not (Test-Path -LiteralPath $benchmarkPath)) {
    Write-Host "[benchmark] running"
    Invoke-SessionPython "scripts.chaos.benchmark_batched_fdtd" @(
        "--output", $benchmarkSessionPath
    )
    $stageCode = $script:LastSessionPythonExitCode
    if ($stageCode -ne 0) {
        Write-Host ("[benchmark] FAILED with exit code " + $stageCode)
        exit $stageCode
    }
} else {
    Write-Host "[benchmark] existing output; skipped"
}

$rows = Import-Csv -LiteralPath $benchmarkPath | Where-Object {
    $_.status -eq "complete" -and $_.throughput_steps_s -ne ""
}
if ($null -eq $rows -or @($rows).Count -eq 0) {
    Write-Host "[summary] no completed benchmark rows"
    exit 1
}
$winner = @($rows | Sort-Object { [double]$_.throughput_steps_s } -Descending)[0]
$summaryLines = @(
    "GPU session: $runDir",
    "Winning configuration by measured throughput:",
    ("backend={0}; device={1}; batch={2}; solve_kind={3}; dtype={4}" -f `
        $winner.backend, $winner.jax_device, $winner.batch, $winner.solve_kind, $winner.dtype),
    ("throughput_steps_s={0}; runtime_s={1}; peak_vram_bytes={2}" -f `
        $winner.throughput_steps_s, $winner.runtime_s, $winner.peak_vram_bytes)
)
$summaryLines | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summaryLines | ForEach-Object { Write-Host $_ }
exit 0
