param(
    [switch]$Force,
    [string]$OutputDir = ""
)

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)
Set-Location $repoRoot

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

Write-Host ("GPU session output: " + $runDir)

if ($Force -or -not (Test-Path -LiteralPath $preflightPath)) {
    Write-Host "[preflight] running"
    & python -m scripts.chaos.gpu_preflight `
        --fixture (Join-Path $repoRoot "tests\data\fdtd_reference\ipm_2c_7p9.npz") `
        --output $preflightPath
    $stageCode = $LASTEXITCODE
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
    & python -m scripts.chaos.measure_kernel_precision `
        --jax-device gpu `
        --output $precisionPath
    $stageCode = $LASTEXITCODE
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
    & python -m scripts.chaos.benchmark_batched_fdtd `
        --output $benchmarkPath
    $stageCode = $LASTEXITCODE
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
