from __future__ import annotations

import json

import pytest

from twpa_solver.multitone.resources import ResourceLimitExceeded

from scripts import run_compression
from scripts.run_compression import (
    _build_multitone_basis,
    _frequency_worker_limit,
    _interpolate_p1db_current,
    _resolve_attenuation,
    build_parser,
    main,
)


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
    assert _resolve_attenuation(circuit)[1] == "themis_default_loss_model"
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
    assert "signal_s21_real" in points
    assert "pump_s21_real" in points
    assert "idler_s21_real" in points


def test_spatial_profile_flag_is_explicit() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "unused", "--signal-ghz", "4.5"]
    )
    assert args.spatial_profiles is False
