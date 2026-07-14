<#
.SYNOPSIS
  Pump/gain-map continuation-method campaign on the 2c design.

  Runs a curated set of traversal / predictor / recovery / fold-policy / intra-cell
  continuation configurations sequentially. After each map it (1) plots with
  scripts/plot_gain_map.py (maps + top-3 candidate S21 sweeps) and (2) reclaims
  disk with scripts/prune_map_solutions.py (purge whole point dirs, keep top-100).

  Band and grid mirror the current production run
  outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10:
  50x50, -35..-23 dBm x 7.5..8.5 GHz, per-cell signal spectrum, sidebands 10.

  Frequency-crossing traversals (backbone / nearest / serpentine / floodfill)
  run single-process (the script/CLI forces --frequency-chunk-size 0); the
  column baseline keeps 10-column chunking.

.NOTES
  Runtime estimate (50x50, spectrum, ~half fold, candidates + prune each):
    - column / simple traversals ....... ~60-90 min/run
    - portfolio / bridge / combined .... ~80-120 min/run
    - arclength fold-policy ............ ~110-150 min/run
    - fold-follow diagnostic ........... ~10-20 min
    Full list (~16 runs) ~ 20-26 h. Comment out rows in $configs to trim.

  Disk estimate: peak ~2-3 GB per map before prune; after prune ~0.1-0.2 GB
  retained/run (top-100 solutions + map arrays/spectrum + plots). Each map is
  pruned before the next starts, so peak stays ~3 GB. Final ~2-3 GB for 16 runs.

.PARAMETER OutRoot   Parent folder for all run outputs.
.PARAMETER DryRun    Print the commands without executing.
.PARAMETER Only      Comma-separated config ids to run (default: all).
#>
[CmdletBinding()]
param(
    [string]$OutRoot = "outputs/campaign_continuation_methods",
    [switch]$DryRun,
    [string[]]$Only = @(),
    [int]$NPower = 50,
    [int]$NFreq = 50,
    [switch]$NoPlot,
    # Skip configs already logged "ok" in campaign_log.csv. Use after a crash
    # (this box takes recurring Kernel-Power/Event-41 resets) to continue the
    # campaign without redoing completed maps. Partial/failed runs re-run
    # (--overwrite clears their dir).
    [switch]$Resume,
    # Concurrent gain-solve factorizations. 6 (reference) needs ~2.5-3 GB free;
    # each worker adds ~300 MB peak. On a loaded desktop (<~5 GB free) 6 OOMs the
    # SuperLU gain solve even on easy cells -- drop to 2 (peak ~1 GB). Pure
    # concurrency, results identical.
    [int]$SignalWorkers = 6
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python  = "python"
$Circuit = "outputs/ipm_python_design"   # 2c design
$RunGain = "scripts/run_gain_map.py"
$Plot    = "scripts/plot_gain_map.py"
$Prune   = "scripts/prune_map_solutions.py"

# Flags shared by every map run. These are EXACTLY the reference production
# command's flags (outputs/solver_spectrum_2c_recover_m35_m23_7p5_8p5_50x50_s20_sb10),
# minus --ipm-dir/range/outdir. Do not hand-trim: an earlier subset that dropped
# --inproc-solve-deadline-s 14 let stiff high-power cells finish their pump solve
# and then OOM the gain factorization (MemoryError in gssv). The 14 s deadline
# aborts those cells before the gain solve. Per-config `flags` add only the
# traversal / predictor / recovery deltas.
$Common = @(
    "--executor", "inprocess", "--mode", "warmstart",
    "--circuit-dir", $Circuit,
    "--n-power", "$NPower", "--n-frequency", "$NFreq",
    "--pump-power-min-dbm", "-35", "--pump-power-max-dbm", "-23",
    "--pump-freq-min-ghz", "7.5", "--pump-freq-max-ghz", "8.5",
    "--inproc-pump-backend", "schur_cpu_mt",
    "--inproc-preconditioner", "real_coupled_fast",
    "--inproc-fold-predictor", "secant",
    "--fold-skip-patience", "4",
    "--inproc-schur-cache-size", "2",
    "--inproc-max-newton", "16",
    "--inproc-solve-deadline-s", "14",
    "--pump-mode-count", "10", "--nt", "40",
    "--signal-detuning-mhz", "100",
    # direct, exactly as the reference production command. Do NOT switch to schur:
    # the signal Schur partition densifies the retained block (elimination fill-in)
    # and uses MORE memory than the sparse direct matrix -> OOMs the gain solve.
    "--signal-backend", "direct", "--signal-solver", "superlu",
    "--sidebands", "10", "--signal-workers", "$SignalWorkers",
    # Non-spectrum: one gain solve per cell at --signal-detuning-mhz (100), not an
    # 11-point ladder -> ~10x faster gain, so far more method combos per campaign.
    # Compare configs on the single-point gain / fold-skip maps. Re-run a chosen
    # config with --signal-spectrum later for candidate S21 sweeps.
    "--no-signal-spectrum",
    "--signal-offset-count-per-side", "5", "--signal-offset-step-mhz", "500",
    "--frequency-chunk-size", "10",
    "--overwrite"
)

# Curated configurations. `flags` are the deltas vs the shared band/grid.
# `id` -> test-matrix mapping is in the comment.
$configs = @(
    # --- Controls (current production) -------------------------------------
    @{ id = "c04_baseline_prod";        flags = @("--traversal","column","--inproc-fold-predictor","secant") }                                   # 0.4/0.5
    @{ id = "c03_warm_copy";            flags = @("--traversal","column","--inproc-fold-predictor","none") }                                     # 0.3
    # --- Traversal (Phase 1) -----------------------------------------------
    @{ id = "t01_nearest_copy";         flags = @("--traversal","nearest","--predictor","copy") }                                               # 1,4
    @{ id = "t02_backbone_secant";      flags = @("--traversal","backbone","--backbone-direction","center_out","--predictor","power_secant") }  # 2,3,24
    @{ id = "t03_backbone_ltr";         flags = @("--traversal","backbone","--backbone-direction","ltr","--predictor","power_secant") }         # backbone dir
    @{ id = "t13_serpentine";           flags = @("--traversal","serpentine","--predictor","power_secant") }                                    # 5
    @{ id = "t14_floodfill_portfolio";  flags = @("--traversal","floodfill","--predictor","portfolio","--portfolio-policy","best") }            # 6,7
    # --- Predictors (Phase 2) ----------------------------------------------
    @{ id = "p04_backbone_freqsec";     flags = @("--traversal","backbone","--predictor","freq_secant") }                                       # 10,25
    @{ id = "p05_backbone_corner";      flags = @("--traversal","backbone","--predictor","corner") }                                            # 12,26
    @{ id = "p06_backbone_plane";       flags = @("--traversal","backbone","--predictor","plane") }                                             # 13
    @{ id = "p07_backbone_portfolio";   flags = @("--traversal","backbone","--predictor","portfolio","--portfolio-policy","best") }             # 16,27
    @{ id = "p08_portfolio_ranked";     flags = @("--traversal","backbone","--predictor","portfolio","--portfolio-policy","ranked") }           # 16
    # --- Recovery + fold policy (Phase 3) ----------------------------------
    @{ id = "r10_baseline_bridge";      flags = @("--traversal","column","--recovery","bridge","--bridge-mode","adaptive") }                    # 19,20
    @{ id = "r11_combined_best";        flags = @("--traversal","backbone","--predictor","portfolio","--recovery","bridge","--bridge-mode","adaptive") }  # 28 (expected best)
    @{ id = "r12_combined_foldpolicy";  flags = @("--traversal","backbone","--predictor","portfolio","--recovery","ladder","--fold-policy","combined") }  # 44-47
    # --- Advanced continuation (Phase 4/5) ---------------------------------
    @{ id = "a15_arclength_fold";       flags = @("--traversal","backbone","--predictor","portfolio","--recovery","ladder","--fold-policy","arclength") } # 37,48
)

function Invoke-Step([string]$desc, [string[]]$cmd) {
    Write-Host ">> $desc" -ForegroundColor Cyan
    Write-Host ("   " + ($cmd -join " ")) -ForegroundColor DarkGray
    if ($DryRun) { return $true }
    & $cmd[0] $cmd[1..($cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        return $false
    }
    return $true
}

$onlySet = @()
if ($Only.Count -gt 0) { $onlySet = ($Only -join ",").Split(",").Trim() }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$logCsv = Join-Path $OutRoot "campaign_log.csv"
if (-not (Test-Path $logCsv)) {
    "id,status,run_s,total_s,started" | Out-File -Encoding utf8 $logCsv
}

# -Resume: ids already completed (any log row status=ok) are skipped.
$doneSet = @()
if ($Resume) {
    # Real completion only: status ok AND nonzero runtime (skips DryRun ok,0,0 rows).
    $doneSet = @(Import-Csv $logCsv |
                 Where-Object { $_.status -eq "ok" -and [int]$_.run_s -gt 0 } |
                 Select-Object -ExpandProperty id -Unique)
    if ($doneSet.Count -gt 0) {
        Write-Host "Resume: skipping completed $($doneSet -join ', ')" -ForegroundColor DarkCyan
    }
}

foreach ($cfg in $configs) {
    $id = $cfg.id
    if ($onlySet.Count -gt 0 -and ($onlySet -notcontains $id)) { continue }
    if ($doneSet -contains $id) { Write-Host "SKIP $id (already ok)" -ForegroundColor DarkGray; continue }
    $out = Join-Path $OutRoot $id
    Write-Host "`n============================================================" -ForegroundColor Yellow
    Write-Host "CONFIG $id -> $out" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    $started = (Get-Date).ToString("s")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $runCmd  = @($Python, $RunGain) + $Common + $cfg.flags + @("--outdir", $out)
    $ok = Invoke-Step "run map" $runCmd
    $runS = [int]$sw.Elapsed.TotalSeconds

    if ($ok -and -not $NoPlot) {
        # Plot: maps + top-3 candidate S21 sweeps (candidates are the slow part).
        $plotCmd = @($Python, $Plot, "--run-dir", $out, "--ipm-dir", $Circuit,
                     "--top-k", "3", "--save-pdf")
        $ok = Invoke-Step "plot (maps + top-3 candidates)" $plotCmd
    }
    #if ($ok) {
    
    #    # Prune: purge whole point dirs, keep the top-100 solutions for re-plot.
    #    $pruneCmd = @($Python, $Prune, $out, "--top-k", "100",
    #                  "--purge-point-dirs", "--apply")
    #    $ok = Invoke-Step "prune (keep top-100, purge point dirs)" $pruneCmd
    #}

    $sw.Stop()
    $status = if ($ok) { "ok" } else { "failed" }
    if (-not $DryRun) {
        "$id,$status,$runS,$([int]$sw.Elapsed.TotalSeconds),$started" | Out-File -Append -Encoding utf8 $logCsv
    }
    Write-Host "CONFIG $id : $status  run=${runS}s total=$([int]$sw.Elapsed.TotalSeconds)s" -ForegroundColor Green
}

# Optional diagnostic (not a gain map): fold curve vs frequency via arclength.
if ($onlySet.Count -eq 0 -or $onlySet -contains "fold_follow") {
    $foldOut = Join-Path $OutRoot "fold_follow"
    $foldCmd = @($Python, $RunGain) + $Common + @(
        "--fold-follow", "--no-signal-spectrum", "--traversal", "column",
        "--outdir", $foldOut)
    Invoke-Step "fold-follow diagnostic" $foldCmd | Out-Null
}

Write-Host "`nCampaign complete. Log: $logCsv" -ForegroundColor Green
Write-Host "Compare gain/skip maps under $OutRoot/<id>/plots/maps and candidate tables." -ForegroundColor Green
