from __future__ import annotations

import json

import pytest

from twpa_solver.multitone.resources import ResourceLimitExceeded

from scripts import run_compression
from scripts.run_compression import (
    SMALL_SIGNAL_FLOOR_TOL_DB,
    _build_multitone_basis,
    _effective_p1db_current,
    _frequency_worker_limit,
    _first_kinetic_threshold_current,
    _interpolate_p1db_current,
    _resolve_attenuation,
    _resolve_pump_current_a,
    _resolve_signal_current_bracket_a,
    _small_signal_floor_delta_db,
    build_parser,
    main,
)


def test_pump_power_dbm_rejects_explicit_current() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir", "unused", "--signal-ghz", "4.5",
                "--pump-current-a", "1e-6", "--pump-power-dbm", "-20.0",
            ]
        )


@pytest.mark.parametrize("order", [2, 10, 17, -3])
def test_imd_order_validation(order: int) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([
            "--output-dir", "unused", "--signal-ghz", "4.5",
            "--imd-max-order", str(order),
        ])


def test_imd_order_three_extends_matched_basis_by_one_tone() -> None:
    base_args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "6.0", "--multitone-sidebands", "10"]
    )
    plain = _build_multitone_basis(base_args, list(range(1, 20, 2)), 2.0 * 3.141592653589793 * 7e9, 2.0 * 3.141592653589793 * 1e9)
    imd_args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "6.0", "--multitone-sidebands", "10", "--imd-max-order", "3"]
    )
    extended = _build_multitone_basis(imd_args, list(range(1, 20, 2)), 2.0 * 3.141592653589793 * 7e9, 2.0 * 3.141592653589793 * 1e9)
    assert extended.n_tones == plain.n_tones + 1
    assert len(imd_args._imd_tones_added) == 1


def test_resolve_pump_current_prefers_explicit_current_over_power_and_default() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--pump-current-a", "3.7e-06"]
    )
    current, source = _resolve_pump_current_a(args, default_current=9.0e-06)
    assert current == pytest.approx(3.7e-06)
    assert source == "explicit_current"


def test_resolve_pump_current_falls_back_to_circuit_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    current, source = _resolve_pump_current_a(args, default_current=9.0e-06)
    assert current == pytest.approx(9.0e-06)
    assert source == "circuit_metadata"


def test_resolve_pump_current_raises_without_any_source() -> None:
    parser = build_parser()
    args = parser.parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    with pytest.raises(ValueError, match="pump-current-a.*pump-power-dbm"):
        _resolve_pump_current_a(args, default_current=None)


def test_resolve_pump_current_from_power_dbm_matches_manual_conversion() -> None:
    from twpa_solver.loss import pump_loss_model
    from twpa_solver.ports import port_current_from_power_a

    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "unused", "--signal-ghz", "4.5",
            "--pump-power-dbm", "-20.0", "--pump-freq-ghz", "7.1",
            "--circuit-dir", "designs/ipm_2c_fixed",
        ]
    )
    current, source = _resolve_pump_current_a(args, default_current=None)
    assert source == "pump_power_dbm"
    atten_db = pump_loss_model().attenuation_db(7.1)
    on_chip_power_w = 1.0e-3 * 10.0 ** ((-20.0 - atten_db) / 10.0)
    expected = port_current_from_power_a(on_chip_power_w, 50.0, convention="legacy_traveling_wave")
    assert current == pytest.approx(expected)


def test_signal_power_dbm_requires_both_bounds() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir", "unused", "--signal-ghz", "4.5",
                "--signal-power-min-dbm", "-60.0",
            ]
        )


def test_signal_power_dbm_rejects_explicit_current() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--output-dir", "unused", "--signal-ghz", "4.5",
                "--signal-power-min-dbm", "-60.0", "--signal-power-max-dbm", "-20.0",
                "--signal-current-min-a", "1e-10",
            ]
        )


def test_resolve_signal_bracket_defaults_when_nothing_given() -> None:
    parser = build_parser()
    args = parser.parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    min_a, max_a, source = _resolve_signal_current_bracket_a(args)
    assert (min_a, max_a) == pytest.approx((1e-12, 1e-9))
    assert source == "explicit_current"


def test_resolve_signal_bracket_prefers_explicit_current() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "unused", "--signal-ghz", "4.5",
            "--signal-current-min-a", "3e-10", "--signal-current-max-a", "7e-8",
        ]
    )
    min_a, max_a, source = _resolve_signal_current_bracket_a(args)
    assert (min_a, max_a) == pytest.approx((3e-10, 7e-8))
    assert source == "explicit_current"


def test_resolve_signal_bracket_from_power_dbm_matches_manual_conversion() -> None:
    from twpa_solver.loss import signal_line_loss_model
    from twpa_solver.ports import port_current_from_power_a

    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir", "unused", "--signal-ghz", "7.2",
            "--signal-power-min-dbm", "-60.0", "--signal-power-max-dbm", "-20.0",
            "--circuit-dir", "designs/ipm_2c_fixed",
        ]
    )
    min_a, max_a, source = _resolve_signal_current_bracket_a(args)
    assert source == "signal_power_dbm"
    atten_db = signal_line_loss_model().attenuation_db(7.2)
    expected_min = port_current_from_power_a(
        1.0e-3 * 10.0 ** ((-60.0 - atten_db) / 10.0), 50.0, convention="legacy_traveling_wave"
    )
    expected_max = port_current_from_power_a(
        1.0e-3 * 10.0 ** ((-20.0 - atten_db) / 10.0), 50.0, convention="legacy_traveling_wave"
    )
    assert min_a == pytest.approx(expected_min)
    assert max_a == pytest.approx(expected_max)


def test_effective_p1db_prefers_kinetic_threshold() -> None:
    assert _effective_p1db_current(2.0e-6, 1.5e-6) == (None, "THRESHOLD_CROSSED")
    assert _effective_p1db_current(2.0e-6, None) == (2.0e-6, "SMOOTH_COMPRESSION")
    assert _effective_p1db_current(None, None) == (None, "NOT_REACHED")


def test_first_kinetic_threshold_ignores_failed_points() -> None:
    points = [
        {"status": "SOLVER_FAILED", "max_current_over_ic": 2.0, "signal_current_a": 1.0e-6},
        {"status": "VALID_SOLVED", "max_current_over_ic": 0.9, "signal_current_a": 2.0e-6},
        {"status": "VALID_SOLVED", "max_current_over_ic": 1.1, "signal_current_a": 3.0e-6},
    ]
    assert _first_kinetic_threshold_current(points) == pytest.approx(3.0e-6)


def test_multitone_preconditioner_defaults_exact_and_accepts_sector() -> None:
    parser = build_parser()
    default = parser.parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    sector = parser.parse_args(
        [
            "--output-dir",
            "unused",
            "--signal-ghz",
            "4.5",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    )
    assert default.multitone_preconditioner == "real_coupled_fast"
    assert default.multitone_basis == "matched"
    assert default.multitone_sidebands == 2
    assert default.multitone_backend == "auto"
    assert default.pump_port is None
    assert sector.multitone_preconditioner == "floquet_sector"
    assert default.signal_ghz == 4.5
    assert default.p1db_power_tol_db == pytest.approx(0.1)


def test_signal_frequency_is_required() -> None:
    with pytest.raises(SystemExit):
        main(["--output-dir", "unused"])


def _worker_args(*extra: str) -> object:
    return build_parser().parse_args(
        [
            "--output-dir", "unused", "--fixture", "jtwpa",
            "--signal-ghz", "6.6", "--pump-freq-ghz", "7.12",
            "--pump-current-a", "3.7e-06", "--signal-workers", "4",
            *extra,
        ]
    )


def test_frequency_workers_scale_down_as_the_tone_basis_grows(monkeypatch) -> None:
    """Worker count follows the real per-worker footprint, not a constant.

    A fixed per-worker estimate either idles cores on a small basis or
    overcommits on a large one; jtwpa at S=10 peaks near 4.14 GB per worker,
    which a 3.0 GB constant underestimates into swap.
    """
    monkeypatch.setattr(run_compression, "available_memory_gb", lambda: 1024.0)
    # Request far more workers than memory allows, so the footprint is what
    # binds rather than the request itself.
    budget = ("--signal-workers", "64", "--resource-budget-gb", "64")

    small = _worker_args("--multitone-sidebands", "2", *budget)
    large = _worker_args("--multitone-sidebands", "10", *budget)

    assert _frequency_worker_limit(small, 64) > _frequency_worker_limit(large, 64)


def test_frequency_workers_never_exceed_request_or_task_count(monkeypatch) -> None:
    monkeypatch.setattr(run_compression, "available_memory_gb", lambda: 1024.0)
    args = _worker_args("--multitone-sidebands", "2", "--resource-budget-gb", "1024")

    assert _frequency_worker_limit(args, 10) <= 4
    assert _frequency_worker_limit(args, 2) <= 2


def test_frequency_workers_are_capped_by_free_memory_not_just_budget(
    monkeypatch,
) -> None:
    """A generous budget must not override what the machine actually has free."""
    args = _worker_args("--multitone-sidebands", "10", "--resource-budget-gb", "1024")
    peak_gb = run_compression._estimate_worker_footprint(args).peak_gb
    headroom_gb = run_compression._memory_headroom_gb(peak_gb)

    monkeypatch.setattr(run_compression, "available_memory_gb", lambda: 1024.0)
    plentiful = _frequency_worker_limit(args, 10)
    # Enough for exactly one worker plus the reserve, and no more.
    monkeypatch.setattr(
        run_compression,
        "available_memory_gb",
        lambda: peak_gb + headroom_gb + 0.1,
    )
    scarce = _frequency_worker_limit(args, 10)

    assert scarce == 1
    assert plentiful > scarce


def test_frequency_workers_reserve_headroom_below_free_memory(monkeypatch) -> None:
    """Worker count must not consume every free byte at launch.

    Free memory is sampled once, but the sweep runs for hours against whatever
    else the machine is doing, so a count that exactly fits at launch ends up
    swapping later.
    """
    args = _worker_args("--multitone-sidebands", "10", "--resource-budget-gb", "1024")
    footprint = run_compression._estimate_worker_footprint(args)
    peak_gb = footprint.peak_gb
    # Enough free memory for exactly 3 workers with nothing to spare.
    monkeypatch.setattr(
        run_compression, "available_memory_gb", lambda: 3.0 * peak_gb
    )

    assert _frequency_worker_limit(args, 10) < 3


def test_driver_builds_one_preconditioner_for_the_whole_run(
    tmp_path, monkeypatch
) -> None:
    """All problems in a run share one cached preconditioner.

    The driver builds several problems that differ only in source path. The
    coupled matrix, its scatter map, and its factors depend on neither the
    source nor the signal power, so giving each problem its own cache keeps one
    full preconditioner alive per problem -- at S=10 that is ~2.6 GB apiece and
    exhausts memory before any worker starts.
    """
    from twpa_solver.pump.backends import fast_coupled

    built = []
    original = fast_coupled.FastCoupledPreconditioner

    class _Counted(original):  # type: ignore[misc,valid-type]
        def __init__(self, problem, **kwargs: object) -> None:
            built.append(problem)
            super().__init__(problem, **kwargs)

    monkeypatch.setattr(fast_coupled, "FastCoupledPreconditioner", _Counted)

    assert main(
        [
            "--output-dir", str(tmp_path),
            "--signal-ghz", "4.75",
            "--n-signal-power", "2",
            "--multitone-sidebands", "2",
        ]
    ) == 0
    assert len(built) == 1, f"expected one preconditioner, built {len(built)}"


def test_factor_backend_flag_reaches_the_preconditioner(tmp_path) -> None:
    """The flag must actually select the backend, not just parse.

    The preconditioner is built several call layers below the driver and, for a
    frequency sweep, in a different process; the setting travels by environment
    variable, so a plain argument-passing test would not catch it being dropped.
    """
    from twpa_solver.pump.backends import fast_coupled

    backends = []
    original = fast_coupled.FastCoupledPreconditioner

    class _Recorded(original):  # type: ignore[misc,valid-type]
        def refactor(self, tangent) -> None:
            super().refactor(tangent)
            backends.append(self.last_factor_backend)

    monkeypatch_target = fast_coupled.FastCoupledPreconditioner
    fast_coupled.FastCoupledPreconditioner = _Recorded
    try:
        assert main(
            [
                "--output-dir", str(tmp_path),
                "--signal-ghz", "4.75",
                "--n-signal-power", "2",
                "--multitone-sidebands", "2",
                "--factor-backend", "banded",
            ]
        ) == 0
    finally:
        fast_coupled.FastCoupledPreconditioner = monkeypatch_target

    assert backends, "preconditioner was never refactored"
    assert set(backends) == {"banded"}, f"backends used: {set(backends)}"


def test_factor_backend_defaults_to_the_sparse_solver() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5"]
    )
    assert args.factor_backend == "pardiso"


def test_banded_backend_raises_the_worker_cap(monkeypatch) -> None:
    """A smaller per-worker peak must translate into more workers."""
    monkeypatch.setattr(run_compression, "available_memory_gb", lambda: 7.0)
    sparse = _worker_args("--multitone-sidebands", "10", "--resource-budget-gb", "1024")
    banded = _worker_args(
        "--multitone-sidebands", "10", "--resource-budget-gb", "1024",
        "--factor-backend", "banded",
    )

    assert _frequency_worker_limit(banded, 8) > _frequency_worker_limit(sparse, 8)


def test_auto_backend_uses_banded_only_when_it_adds_a_worker(monkeypatch) -> None:
    monkeypatch.setattr(run_compression, "available_memory_gb", lambda: 7.0)
    args = _worker_args(
        "--multitone-sidebands", "10",
        "--resource-budget-gb", "1024",
        "--factor-backend", "auto",
    )
    args.n_signal_freq = 4
    assert run_compression._select_factor_backend(args, 4) == "banded"
    args.n_signal_freq = 1
    assert run_compression._select_factor_backend(args, 1) == "pardiso"


def test_sweep_refuses_to_start_when_one_worker_cannot_fit(monkeypatch) -> None:
    """Refuse rather than swap when even a single worker exceeds free memory.

    Flooring the count at one worker is not a safeguard when one worker is
    itself too big: the solve does not degrade gracefully under memory
    pressure, it takes the machine down.
    """
    args = _worker_args("--multitone-sidebands", "10", "--resource-budget-gb", "1024")
    peak_gb = run_compression._estimate_worker_footprint(args).peak_gb
    monkeypatch.setattr(
        run_compression, "available_memory_gb", lambda: 0.5 * peak_gb
    )

    with pytest.raises(ResourceLimitExceeded, match="a single worker needs"):
        _frequency_worker_limit(args, 10)


def test_memory_overcommit_flag_permits_a_run_that_does_not_fit(
    monkeypatch,
) -> None:
    args = _worker_args(
        "--multitone-sidebands", "10",
        "--resource-budget-gb", "1024",
        "--allow-memory-overcommit",
    )
    peak_gb = run_compression._estimate_worker_footprint(args).peak_gb
    monkeypatch.setattr(
        run_compression, "available_memory_gb", lambda: 0.5 * peak_gb
    )

    assert _frequency_worker_limit(args, 10) == 1


def test_frequency_workers_fall_back_to_one_when_footprint_is_unknown(
    monkeypatch,
) -> None:
    """An unreadable circuit must not silently license full concurrency."""
    def _boom(_args: object) -> None:
        raise ValueError("circuit metadata unavailable")

    monkeypatch.setattr(run_compression, "_estimate_worker_footprint", _boom)
    args = _worker_args("--resource-budget-gb", "1024")

    assert _frequency_worker_limit(args, 10) == 1


def test_p1db_interpolation_is_logarithmic_in_current() -> None:
    points = [
        {"signal_current_a": 1.0e-9, "compression_db": 0.5, "status": "VALID_SOLVED"},
        {"signal_current_a": 1.0e-8, "compression_db": 1.5, "status": "VALID_SOLVED"},
    ]
    assert _interpolate_p1db_current(points) == pytest.approx(10.0 ** -8.5)


def test_p1db_refinement_can_be_disabled_to_reach_interpolation_fallback() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir", "unused", "--signal-ghz", "4.5",
            "--p1db-power-tol-db", "0",
        ]
    )
    assert args.p1db_power_tol_db == 0.0
    points = [
        {"signal_current_a": 1.0e-9, "compression_db": 0.5, "status": "VALID_SOLVED"},
        {"signal_current_a": 1.0e-8, "compression_db": 1.5, "status": "VALID_SOLVED"},
    ]
    assert _interpolate_p1db_current(points) == pytest.approx(10.0 ** -8.5)


def test_small_signal_floor_delta_flags_non_flat_grid() -> None:
    """A synthetic already-compressed grid: gain already dropping between the
    two lowest currents means the sweep never reached the flat small-signal
    region, so G0 (read from points[0]) is biased high.
    """
    points = [
        {"status": "VALID_SOLVED", "gain_vs_off_db": 10.0},
        {"status": "VALID_SOLVED", "gain_vs_off_db": 9.5},
        {"status": "VALID_SOLVED", "gain_vs_off_db": 8.0},
    ]
    delta = _small_signal_floor_delta_db(points)
    assert delta == pytest.approx(0.5)
    assert delta >= SMALL_SIGNAL_FLOOR_TOL_DB


def test_small_signal_floor_delta_passes_flat_grid() -> None:
    points = [
        {"status": "VALID_SOLVED", "gain_vs_off_db": 10.000},
        {"status": "VALID_SOLVED", "gain_vs_off_db": 9.980},
        {"status": "VALID_SOLVED", "gain_vs_off_db": 6.000},
    ]
    delta = _small_signal_floor_delta_db(points)
    assert delta == pytest.approx(0.02, abs=1e-9)
    assert delta < SMALL_SIGNAL_FLOOR_TOL_DB


def test_small_signal_floor_delta_none_with_insufficient_points() -> None:
    assert _small_signal_floor_delta_db([]) is None
    assert _small_signal_floor_delta_db(
        [{"status": "VALID_SOLVED", "gain_vs_off_db": 10.0}]
    ) is None
    assert _small_signal_floor_delta_db(
        [{"status": "FAIL", "gain_vs_off_db": float("nan")},
         {"status": "VALID_SOLVED", "gain_vs_off_db": 10.0}]
    ) is None


def test_small_signal_floor_delta_skips_failed_points() -> None:
    """Only VALID_SOLVED points count -- a failed point in between must not
    be compared against."""
    points = [
        {"status": "VALID_SOLVED", "gain_vs_off_db": 10.0},
        {"status": "FAIL", "gain_vs_off_db": float("nan")},
        {"status": "VALID_SOLVED", "gain_vs_off_db": 9.99},
    ]
    delta = _small_signal_floor_delta_db(points)
    assert delta == pytest.approx(0.01, abs=1e-9)


def test_non_flat_starting_grid_reports_g0_grid_not_flat(tmp_path) -> None:
    """Real solve, not a synthetic points list: the JPA fixture's gain vs.
    signal current is non-monotone (rises from ~7.40 dB at very low current
    to a ~12 dB peak before collapsing into compression), so a sweep that
    starts at 4e-11 A instead of near-zero begins already on the rising
    flank -- the grid never reaches the flat small-signal region, and the
    guard must catch that rather than reporting a P1dB read off a biased G0.
    """
    assert main([
        "--output-dir", str(tmp_path),
        "--fixture", "jpa",
        "--pump-freq-ghz", "4.75001",
        "--pump-current-a", "1.13e-08",
        "--pump-current-jc-scale", "1.0",
        "--signal-ghz", "4.75",
        "--source-port", "1", "--pump-port", "1", "--out-port", "1",
        "--n-signal-power", "3",
        "--signal-current-min-a", "4e-11",
        "--signal-current-max-a", "2e-10",
        "--attenuation-db", "0",
        "--multitone-basis", "matched", "--multitone-sidebands", "2",
        "--recovery", "ladder",
    ]) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    assert summary["status"] == "G0_GRID_NOT_FLAT"
    assert summary["small_signal_floor_flat"] is False
    assert summary["small_signal_floor_delta_db"] >= SMALL_SIGNAL_FLOOR_TOL_DB
    assert summary["p1db"] is None


def _jpa_gain_args(tmp_path, n_points: int) -> list[str]:
    """The exp20 jpa operating point, which really does compress.

    The default fixture point has no gain, so it can never exercise the
    refinement branch -- P1dB is suppressed there.
    """
    return [
        "--output-dir", str(tmp_path),
        "--fixture", "jpa",
        "--pump-freq-ghz", "4.75001",
        "--pump-current-a", "1.13e-08",
        "--pump-current-jc-scale", "1.0",
        "--signal-ghz", "4.75",
        "--source-port", "1", "--pump-port", "1", "--out-port", "1",
        "--n-signal-power", str(n_points),
        "--signal-current-min-a", "1e-12",
        "--signal-current-max-a", "3e-08",
        "--attenuation-db", "0",
        "--multitone-basis", "matched", "--multitone-sidebands", "2",
        "--recovery", "ladder",
    ]


def test_refined_run_also_reports_the_interpolated_p1db(tmp_path) -> None:
    """Both numbers must come out of one sweep.

    The refined-versus-interpolated delta decides whether published sweeps
    need re-running. Reading the two halves off two separate runs would fold
    run-to-run variation into a comparison that has none, so the driver keeps
    the interpolated value after refinement overwrites the reported P1dB.
    """
    assert main(_jpa_gain_args(tmp_path, 9) + ["--p1db-power-tol-db", "0.1"]) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())

    assert summary["p1db_method"] == "refined"
    assert summary["p1db_interpolated_dbm"] is not None
    # The refined value is a real solve, the interpolated one a log-linear
    # guess between grid points ~11 dB apart; they must not be the same number.
    assert summary["p1db"] != pytest.approx(summary["p1db_interpolated_dbm"])


def test_interpolated_p1db_is_the_reported_one_when_refinement_is_off(
    tmp_path,
) -> None:
    """With refinement disabled the two fields must agree exactly."""
    assert main(_jpa_gain_args(tmp_path, 9) + ["--p1db-power-tol-db", "0"]) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())

    assert summary["p1db_method"] == "interpolated"
    assert summary["p1db"] == pytest.approx(summary["p1db_interpolated_dbm"])


def test_power_convention_defaults_to_legacy_traveling_wave() -> None:
    args = build_parser().parse_args(["--output-dir", "unused", "--signal-ghz", "4.5"])
    assert args.power_convention == "legacy_traveling_wave"


def test_legacy_power_convention_shifts_p1db_by_exactly_6p02_db(tmp_path) -> None:
    """Norton vs legacy differ only by the fixed 6.0206 dB relabel.

    Gain is a ratio of ratios (pump-on over pump-off), so it is invariant
    under the power convention; only the reported dBm axis moves.
    """
    norton_dir = tmp_path / "norton"
    legacy_dir = tmp_path / "legacy"
    args = _jpa_gain_args(legacy_dir, 5) + ["--p1db-power-tol-db", "0"]
    assert main(args) == 0
    assert main(
        [a if a != str(legacy_dir) else str(norton_dir) for a in args]
        + ["--power-convention", "norton"]
    ) == 0

    norton_summary = json.loads((norton_dir / "compression_summary.json").read_text())
    legacy_summary = json.loads((legacy_dir / "compression_summary.json").read_text())

    assert norton_summary["power_convention"] == "norton"
    assert legacy_summary["power_convention"] == "legacy_traveling_wave"
    assert legacy_summary["p1db"] == pytest.approx(
        norton_summary["p1db"] + 6.0205999133, abs=1e-6
    )
    assert norton_summary["small_signal_gain_vs_off_db"] == pytest.approx(
        legacy_summary["small_signal_gain_vs_off_db"]
    )


def test_stop_after_p1db_matches_the_full_sweep_but_skips_the_tail(
    tmp_path,
) -> None:
    """Stopping right after the 1 dB crossing must not change the P1dB.

    Both the interpolated and refined values only ever need the two points
    straddling the crossing -- everything past it is deep-saturation tail
    that the full sweep also solves but neither P1dB path reads.
    """
    full_dir = tmp_path / "full"
    early_dir = tmp_path / "early"
    full_args = _jpa_gain_args(full_dir, 9)
    early_args = _jpa_gain_args(early_dir, 9)

    assert main(full_args + ["--p1db-power-tol-db", "0.1"]) == 0
    assert main(
        early_args + ["--p1db-power-tol-db", "0.1", "--stop-after-p1db"]
    ) == 0

    full_summary = json.loads((full_dir / "compression_summary.json").read_text())
    early_summary = json.loads((early_dir / "compression_summary.json").read_text())

    assert full_summary["p1db_method"] == "refined"
    assert early_summary["p1db_method"] == "refined"
    assert early_summary["p1db"] == pytest.approx(full_summary["p1db"])
    assert early_summary["n_requested_power_points"] < full_summary[
        "n_requested_power_points"
    ]


def test_no_gain_operating_point_suppresses_compression(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--signal-ghz",
            "4.5",
            "--n-signal-power",
            "2",
        ]
    ) == 0
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    points = (tmp_path / "compression_points.csv").read_text()
    assert summary["status"] == "NO_GAIN_AT_OPERATING_POINT"
    assert summary["p1db"] is None
    assert "nan" in points


def test_multitone_preconditioner_rejects_unknown_name() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--output-dir",
                "unused",
                "--signal-ghz",
                "4.5",
                "--multitone-preconditioner",
                "unknown",
            ]
        )


def test_three_tone_guard_names_missing_pump_modes() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "unused",
            "--signal-ghz",
            "4.5",
            "--multitone-basis",
            "three_tone",
        ]
    )
    with pytest.raises(ValueError, match=r"pump_modes=\[1, 3\].*missing=\[3\]"):
        _build_multitone_basis(args, [1, 3], 10.0, 1.0)


def test_fixture_and_circuit_attenuation_defaults_are_distinct() -> None:
    parser = build_parser()
    fixture = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--fixture", "jpa"]
    )
    circuit = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--circuit-dir", "design"]
    )
    explicit = parser.parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--fixture", "jpa", "--attenuation-db", "7"]
    )
    assert _resolve_attenuation(fixture) == (0.0, "fixture_default_zero")
    assert _resolve_attenuation(circuit)[1] == "signal_line_loss_model"
    assert _resolve_attenuation(explicit) == (7.0, "explicit")


def test_run_compression_uses_sector_preconditioner(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--n-signal-power",
            "2",
            "--signal-ghz",
            "4.5",
            "--multitone-preconditioner",
            "floquet_sector",
        ]
    ) == 0
    summary = json.loads(
        (tmp_path / "compression_summary.json").read_text()
    )
    assert summary["multitone_preconditioner"] == "floquet_sector"


def test_run_compression_smoke_writes_artifacts(tmp_path) -> None:
    assert main(
        [
            "--output-dir",
            str(tmp_path),
            "--signal-ghz",
            "4.5",
            "--n-signal-power",
            "5",
        ]
    ) == 0
    assert (tmp_path / "compression_points.csv").exists()
    assert (tmp_path / "compression_arrays.npz").exists()
    summary = json.loads((tmp_path / "compression_summary.json").read_text())
    points = (tmp_path / "compression_points.csv").read_text()
    assert summary["stability_status"] == "NOT_CHECKED"
    assert summary["multitone_preconditioner"] == "real_coupled_fast"
    assert summary["multitone_backend"] == "full"
    assert summary["pump_port"] == 1
    assert "pump_depletion_db" in points
    assert "compression_model_depletion_only" in points
    assert "power_balance_rel_err" in points
    assert "hb_residual_rel" in points
    assert "max_power_balance_rel_err" in summary
    assert "manley_rowe_photon_flux" in points
    assert "manley_rowe_rel_err" in points
    assert "external_manley_rowe_rel_err" in points
    assert "max_manley_rowe_rel_err" in summary
    assert "p1db_method" in summary
    assert "signal_s21_real" in points
    assert "pump_s21_real" in points
    assert "idler_s21_real" in points


def test_spatial_profile_flag_is_explicit() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5"]
    )
    assert args.spatial_profiles is False


def test_stability_check_is_default_off() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5"]
    )
    assert args.check_stability is False
    assert build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5", "--check-stability"]
    ).check_stability is True
