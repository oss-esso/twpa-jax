# Overnight chaos campaign, 2026-08-17.  Closes the gaps left open by the
# 2026-08-16 run.  Every stage writes per point and skips completed points, so
# this is resumable: rerun the same command and it continues where it stopped.
#
#   Stage 1  ~35 min  guarcello rebuilt at record_stride 4   (gap 4)
#   Stage 2  ~58 min  jc_jtwpa dense onset                   (gap 3)
#   Stage 3  ~68 min  ipm_2c_fixed window + onset            (gaps 2, 3)
#   Stage 4  ~67 min  jc_fqjtwpa at 2100 periods, 4th device (gap 2)
#   Stage 5  ~84 min  dt/2 controls inside the torus windows
#   Stage 6  ~15 min  reductions and figures (CPU only, re-runnable)
#
# Total approximately 5.5 h wall at 3 workers.
#
# Windows PowerShell 5.1: no '&&', no '||', no ternary.  Conditional chaining
# is ';' followed by 'if ($?) { ... }'.
#
# Run from the repository root:
#     powershell -ExecutionPolicy Bypass -File scripts\chaos\run_overnight_campaign_20260817.ps1

$ErrorActionPreference = 'Continue'

Set-Location (Join-Path $PSScriptRoot '..\..')
$RUN = 'scripts/chaos/run_phaseB_overnight.py'
$LOG = 'outputs/chaos/campaign_20260817.log'

if (-not (Test-Path 'outputs/chaos')) { New-Item -ItemType Directory -Force 'outputs/chaos' | Out-Null }

function Say([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Write-Output $line
    Add-Content -Path $LOG -Value $line -Encoding utf8
}

function Invoke-Stage {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Say "--- $Label"
    $started = Get-Date
    & python $RUN @Arguments 2>&1 | Add-Content -Path $LOG -Encoding utf8
    $code = $LASTEXITCODE
    $minutes = ((Get-Date) - $started).TotalMinutes
    Say ("--- {0} done (exit {1}, {2:N1} min)" -f $Label, $code, $minutes)
}

Say '=== campaign start ==='

# ---------------------------------------------------------------------------
# Stage 1.  guarcello, rebuilt with a usable record stride.
#
# record_stride 20 gave guarcello only 6.23 stored samples per pump period,
# below the eight-sample guard for a stroboscopic section, so every earlier
# guarcello point silently fell back to the Poincare upward crossings -- a
# different observable in different units.  Stride 4 gives 155.6 samples per
# period, verified 2026-08-17.
#
# A NEW output root is mandatory: at the old root the completed stride-20
# points are found by the resume check and every point would be skipped.
# guarcello costs ~1.3 min/point at 2100 periods, so the full range is
# affordable at 0.5 dB and the onset at 0.05 dB.
# ---------------------------------------------------------------------------
Invoke-Stage 'stage 1a: guarcello full range, stride 4, 51 points' @(
    '--output', 'outputs/chaos/phaseB_guarcello_s4',
    '--devices', 'guarcello', '--control-linspace', '-70', '-45', '51',
    '--periods', '2100', '--dt-norm', '0.01', '--record-stride', '4', '--workers', '3'
)

Invoke-Stage 'stage 1b: guarcello onset 0.05 dB, 29 points' @(
    '--output', 'outputs/chaos/phaseB_guarcello_s4',
    '--devices', 'guarcello', '--control-linspace', '-54.8', '-53.4', '29',
    '--periods', '2100', '--dt-norm', '0.01', '--record-stride', '4', '--workers', '3'
)

# ---------------------------------------------------------------------------
# Stage 2.  jc_jtwpa dense onset.
#
# The torus appears at about -29.44 dBm, where D2 first saturates at 1.  The
# existing grid steps 0.035 dB there, which gave beta = 6.13 / 3.57 / 1.73 as
# the fit window narrowed: a sigmoid, not a power law.  0.025 dB across
# -29.50..-29.30 puts nine points inside the growth region itself.
#
# Same root and settings as the existing 31 points, so these extend one curve
# rather than starting a second.
# ---------------------------------------------------------------------------
Invoke-Stage 'stage 2: jc_jtwpa onset -29.50..-29.30, 9 points' @(
    '--output', 'outputs/chaos/phaseB_jtwpa_2100',
    '--devices', 'jc_jtwpa', '--control-linspace', '-29.50', '-29.30', '9',
    '--periods', '2100', '--dt-norm', '0.01', '--workers', '3'
)

# ---------------------------------------------------------------------------
# Stage 3.  ipm_2c_fixed window edges and onset.
#
# 2c has 0.005 spacing, which resolved the transition but leaves its regular
# window bounded by only four points and its onset by one.  0.0025 across
# 0.5875..0.6100 doubles the resolution where the window opens and closes.
# ---------------------------------------------------------------------------
Invoke-Stage 'stage 3: ipm_2c_fixed 0.5875..0.6100 at 0.0025, 10 points' @(
    '--output', 'outputs/chaos/phaseB_2c_gap',
    '--devices', 'ipm_2c_fixed', '--control-linspace', '0.5875', '0.6100', '10',
    '--periods', '2100', '--dt-norm', '0.01', '--workers', '3'
)

# ---------------------------------------------------------------------------
# Stage 4.  jc_fqjtwpa, the fourth device, at a usable settling budget.
#
# Its 87 existing pump-only points ran 600 periods against a ~1050-period
# settling floor and were excluded from every classification.  2100 periods
# across its transition makes it a fourth independent test of the
# period-1 -> torus -> chaos route.
# ---------------------------------------------------------------------------
Invoke-Stage 'stage 4: jc_fqjtwpa -32.0..-30.8, 13 points, 2100 periods' @(
    '--output', 'outputs/chaos/phaseB_fqjtwpa_2100',
    '--devices', 'jc_fqjtwpa', '--control-linspace', '-32.0', '-30.8', '13',
    '--periods', '2100', '--dt-norm', '0.01', '--workers', '3'
)

# ---------------------------------------------------------------------------
# Stage 5.  Timestep controls inside the torus windows.
#
# The 2026-08-16 dt/2 controls sampled the transition endpoints, not the torus
# window, so nothing yet shows the WINDOW is timestep-converged rather than a
# discretisation artifact.  Separate roots per device, again so the resume
# check does not match the dt = 0.01 points.
# ---------------------------------------------------------------------------
Invoke-Stage 'stage 5a: guarcello dt=0.005 inside the window, 4 points' @(
    '--output', 'outputs/chaos/phaseB_dt005_guarcello_s4',
    '--devices', 'guarcello', '--control-linspace', '-54.2', '-53.6', '4',
    '--periods', '2100', '--dt-norm', '0.005', '--record-stride', '4', '--workers', '3'
)

Invoke-Stage 'stage 5b: jc_jtwpa dt=0.005 inside the window, 2 points' @(
    '--output', 'outputs/chaos/phaseB_dt005_jtwpa_window',
    '--devices', 'jc_jtwpa', '--control-linspace', '-29.2333', '-29.0926', '2',
    '--periods', '2100', '--dt-norm', '0.005', '--workers', '3'
)

Invoke-Stage 'stage 5c: ipm_2c_fixed dt=0.005 inside the window, 2 points' @(
    '--output', 'outputs/chaos/phaseB_dt005_2c_window',
    '--devices', 'ipm_2c_fixed', '--control-linspace', '0.5950', '0.6050', '2',
    '--periods', '2100', '--dt-norm', '0.005', '--workers', '3'
)

# ---------------------------------------------------------------------------
# Stage 6.  Reductions and figures.  CPU only, minutes, re-runnable at will --
# nothing here needs the overnight window, it is here so the morning starts
# with the answers rather than with a queue.
# ---------------------------------------------------------------------------
Say '--- stage 6a: nonlinear diagnostics'
$reductions = @(
    @('outputs/chaos/phaseB_guarcello_s4', 'guarcello',    'outputs/chaos/nonlinear_guarcello_s4'),
    @('outputs/chaos/phaseB_jtwpa_2100',   'jc_jtwpa',     'outputs/chaos/nonlinear_jtwpa_torus'),
    @('outputs/chaos/phaseB_2c_gap',       'ipm_2c_fixed', 'outputs/chaos/nonlinear_2c_gap'),
    @('outputs/chaos/phaseB_fqjtwpa_2100', 'jc_fqjtwpa',   'outputs/chaos/nonlinear_fqjtwpa')
)
foreach ($job in $reductions) {
    & python scripts/chaos/run_nonlinear_diagnostics.py `
        --campaign $job[0] --devices $job[1] --output $job[2] 2>&1 |
        Add-Content -Path $LOG -Encoding utf8
}
Say '--- stage 6a done'

Say '--- stage 6b: figures, four devices'
& python scripts/chaos/plot_nonlinear_diagnostics.py `
    --device jc_jtwpa     outputs/chaos/nonlinear_jtwpa_torus/jc_jtwpa.json `
    --device ipm_2c_fixed outputs/chaos/nonlinear_2c_gap/ipm_2c_fixed.json `
    --device guarcello    outputs/chaos/nonlinear_guarcello_s4/guarcello.json `
    --device jc_fqjtwpa   outputs/chaos/nonlinear_fqjtwpa/jc_fqjtwpa.json `
    --output outputs/chaos/figures_20260817 2>&1 | Add-Content -Path $LOG -Encoding utf8
Say '--- stage 6b done'

# The lambda_1 estimator is expected to report NOT_ESTABLISHED until its delay
# rule is replaced with a mutual-information criterion.  This records the gate
# rather than gating the campaign on it.
Say '--- stage 6c: lambda_1 estimator reference gate (informational)'
& python scripts/chaos/lyapunov_kantz.py `
    --campaign outputs/chaos/phaseB_jtwpa_2100 --devices jc_jtwpa `
    --output outputs/chaos/lyapunov_kantz 2>&1 | Add-Content -Path $LOG -Encoding utf8
Say '--- stage 6c done'

Say '=== campaign complete ==='
