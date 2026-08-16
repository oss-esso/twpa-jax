"""Run the matched-circuit 2c RCSJ stability column.

The finite RCSJ resistance is a numerical regularizer.  It is not a device
property.  Every persisted transient is seeded from a production HB checkpoint
computed on the same stamped circuit at the same resistance ratio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map as gain_map  # noqa: E402
from scripts.chaos import attractor_classify as classifier  # noqa: E402
from twpa_solver.core import (  # noqa: E402
    load_circuit,
    save_circuit,
    stamp_rcsj_shunt,
)


CIRCUIT_DIR = ROOT / "designs" / "ipm_2c_fixed"
DEFAULT_OUTDIR = ROOT / "outputs" / "chaos" / "2c_rcsj_stability_column"
PUMP_GHZ = 7.9
SIGNAL_GHZ = 7.4
PUMP_PORT = 4
SOURCE_PORT = 1
OUT_PORT = 2
FULL_RESIDUAL_GATE = 1.0e-8
JUNCTION_CAPACITANCE_F = 145.0e-15
TARGET_CURRENTS_A = (
    5.2326e-6,
    5.5233e-6,
    6.1047e-6,
    6.3954e-6,
    6.6861e-6,
    7.2675e-6,
    7.5582e-6,
    7.8489e-6,
    8.4303e-6,
    8.7210e-6,
)
TD_CURRENTS_A = (6.1047e-6, 6.3954e-6, 6.6861e-6, 7.2675e-6)
RATIOS = (1.0e4, 1.0e5, 1.0e6)
PRIMARY_RATIO = 1.0e5
CSV_FIELDS = (
    "arm",
    "circuit_identity",
    "resistance_ratio",
    "damping_per_period",
    "achieved_pump_current_a",
    "pump_power_instrument_dbm",
    "gain_vs_off_db",
    "r_j",
    "production_hb_full_residual_rel",
    "hill_max_abs_lambda",
    "hill_root_frequency_ghz",
    "envelope_slope_per_period",
    "max_abs_phi_rad",
    "status",
    "runtime_s",
    "source_path",
    "checkpoint_path",
)


def atomic_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact by atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the combined measurement table by atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def ratio_slug(ratio: float) -> str:
    return f"r{ratio:.0e}".replace("+", "")


def _hash_array(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.view(np.uint8))


def circuit_identity(circuit: Any) -> str:
    """Return a content identity for all matrices and nonlinear branches."""
    digest = hashlib.sha256()
    for name in ("C", "G", "K", "Bphi"):
        matrix = sp.csr_matrix(getattr(circuit, name))
        digest.update(name.encode("ascii"))
        _hash_array(digest, matrix.indptr)
        _hash_array(digest, matrix.indices)
        _hash_array(digest, matrix.data)
    _hash_array(digest, np.asarray(circuit.Ic, dtype=np.float64))
    return digest.hexdigest()


def continuation_currents(
    targets: tuple[float, ...] = TARGET_CURRENTS_A,
    *,
    max_step_db: float = 0.25,
) -> list[tuple[float, bool]]:
    """Insert geometric bridge currents while preserving every target exactly."""
    if not targets or max_step_db <= 0.0:
        raise ValueError("targets and max_step_db must be positive")
    points: list[tuple[float, bool]] = [(float(targets[0]), True)]
    for left, right in zip(targets, targets[1:]):
        delta_db = 20.0 * math.log10(right / left)
        segments = max(1, int(math.ceil(delta_db / max_step_db)))
        for segment in range(1, segments):
            fraction = segment / segments
            points.append((left * (right / left) ** fraction, False))
        points.append((float(right), True))
    return points


def max_continuation_step_db(points: list[tuple[float, bool]]) -> float:
    values = [point[0] for point in points]
    return max(20.0 * math.log10(right / left) for left, right in zip(values, values[1:]))


def load_records(outdir: Path) -> list[dict[str, Any]]:
    path = outdir / "records.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def persist_records(outdir: Path, rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            str(row["arm"]),
            float(row["resistance_ratio"]),
            float(row["achieved_pump_current_a"]),
        )
    )
    atomic_json(outdir / "records.json", {"rows": rows})
    atomic_csv(outdir / "rcsj_stability_column.csv", rows)


def replace_record(rows: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = (
        record["arm"],
        float(record["resistance_ratio"]),
        float(record["achieved_pump_current_a"]),
    )
    rows[:] = [
        row
        for row in rows
        if (
            row["arm"],
            float(row["resistance_ratio"]),
            float(row["achieved_pump_current_a"]),
        )
        != key
    ]
    rows.append(record)


def find_record(
    rows: list[dict[str, Any]], arm: str, ratio: float, current: float
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["arm"] == arm
            and float(row["resistance_ratio"]) == float(ratio)
            and math.isclose(
                float(row["achieved_pump_current_a"]),
                current,
                rel_tol=2.0e-9,
                abs_tol=1.0e-18,
            )
        ):
            return row
    return None


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _fdtd_rows() -> list[dict[str, Any]]:
    root = ROOT / "outputs" / "chaos" / "phaseB_signal" / "ipm_2c_fixed"
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        current = _number(row.get("pump_current_peak_a_achieved"))
        if current is None:
            continue
        if any(math.isclose(current, target, rel_tol=2.0e-9) for target in TARGET_CURRENTS_A):
            row["_source_path"] = str(path.resolve())
            candidates.append(row)
    selected: list[dict[str, Any]] = []
    for target in TARGET_CURRENTS_A:
        matches = [
            row
            for row in candidates
            if math.isclose(
                float(row["pump_current_peak_a_achieved"]),
                target,
                rel_tol=2.0e-9,
                abs_tol=1.0e-18,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"arm A requires one retained result at {target:.7e} A; got {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def install_arm_a(outdir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    circuit = load_circuit(CIRCUIT_DIR)
    identity = circuit_identity(circuit)
    fdtd = _fdtd_rows()
    for source in fdtd:
        record = {
            "arm": "A_FDTD_LOSSLESS",
            "circuit_identity": identity,
            "resistance_ratio": math.inf,
            "damping_per_period": 0.0,
            "achieved_pump_current_a": float(source["pump_current_peak_a_achieved"]),
            "pump_power_instrument_dbm": _number(source.get("pump_power_instrument_dbm")),
            "gain_vs_off_db": _number(source.get("gain_vs_off_db")),
            "r_j": _number(source.get("r_j")),
            "production_hb_full_residual_rel": None,
            "hill_max_abs_lambda": None,
            "hill_root_frequency_ghz": None,
            "envelope_slope_per_period": None,
            "max_abs_phi_rad": None,
            "status": "MEASURED_RETAINED",
            "runtime_s": _number(source.get("runtime_s")),
            "source_path": source["_source_path"],
            "checkpoint_path": None,
        }
        replace_record(rows, record)
        persist_records(outdir, rows)
    return fdtd


def build_variant(outdir: Path, ratio: float) -> tuple[Path, Any, str, float]:
    """Create or validate one finite-RCSJ circuit variant."""
    base = load_circuit(CIRCUIT_DIR)
    variant = outdir / "circuits" / ratio_slug(ratio)
    expected, parameters = stamp_rcsj_shunt(
        base,
        ratio,
        junction_capacitance_f=JUNCTION_CAPACITANCE_F,
        pump_frequency_hz=PUMP_GHZ * 1.0e9,
    )
    expected_identity = circuit_identity(expected)
    if variant.exists():
        actual = load_circuit(variant)
        if circuit_identity(actual) != expected_identity:
            raise ValueError(f"existing RCSJ variant has the wrong identity: {variant}")
    else:
        variant.parent.mkdir(parents=True, exist_ok=True)
        temporary = variant.with_name(f"{variant.name}.tmp_{os.getpid()}")
        save_circuit(expected, temporary)
        temporary.replace(variant)
    damping = float(np.median(parameters.damping_per_pump_period))
    return variant, parameters, expected_identity, damping


def _hb_args(circuit_dir: Path, outdir: Path) -> argparse.Namespace:
    cli = [
        "--mode",
        "warmstart",
        "--executor",
        "inprocess",
        "--circuit-dir",
        str(circuit_dir),
        "--outdir",
        str(outdir),
        "--n-power",
        "1",
        "--n-frequency",
        "1",
        "--pump-freq-min-ghz",
        str(PUMP_GHZ),
        "--pump-freq-max-ghz",
        str(PUMP_GHZ),
        "--signal-ghz",
        str(SIGNAL_GHZ),
        "--no-signal-spectrum",
        "--power-convention",
        "legacy_traveling_wave",
        "--pump-port",
        str(PUMP_PORT),
        "--source-port",
        str(SOURCE_PORT),
        "--out-port",
        str(OUT_PORT),
        "--pump-mode-policy",
        "positive_odd_jc",
        "--pump-mode-count",
        "10",
        "--harmonics",
        "19",
        "--nt",
        "40",
        "--sidebands",
        "6",
        "--continuation-steps",
        "20",
        "--pump-full-residual-gate",
        str(FULL_RESIDUAL_GATE),
        "--inproc-pump-backend",
        "schur_cpu_mt",
        "--inproc-preconditioner",
        "real_coupled_fast",
        "--log-level",
        "INFO",
    ]
    return gain_map.parse_args(cli)


def _point_power_dbm(current: float, args: argparse.Namespace) -> float:
    return gain_map.peak_current_to_power_dbm(current, PUMP_GHZ, args)


def _pump_report(row: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    pump_dir_value = row.get("pump_dir")
    if not pump_dir_value:
        return None, None
    pump_dir = Path(str(pump_dir_value))
    if not pump_dir.is_absolute():
        pump_dir = ROOT / pump_dir
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return pump_dir, None
    return pump_dir, json.loads(report_path.read_text(encoding="utf-8"))


def _hb_record(
    arm: str,
    ratio: float,
    damping: float,
    identity: str,
    row: dict[str, Any],
    runtime_s: float,
) -> dict[str, Any]:
    pump_dir, report = _pump_report(row)
    passed = row.get("status") == "PASS" and report is not None
    metadata = (report or {}).get("metadata", {})
    residual = _number(metadata.get("production_hb_full_residual_rel"))
    status = str(row.get("status") or "HB_NO_STATUS")
    if passed and (residual is None or residual > FULL_RESIDUAL_GATE):
        passed = False
        status = "FAIL_FULL_HARMONIC_RESIDUAL"
    if passed and (report or {}).get("final_status") != "VALID_CONVERGED":
        passed = False
        status = str((report or {}).get("final_status") or "HB_INVALID_REPORT")
    source_path = str((pump_dir / "pump_report.json").resolve()) if report else str(
        row.get("pump_dir") or ""
    )
    record = {
        "arm": arm,
        "circuit_identity": identity,
        "resistance_ratio": ratio,
        "damping_per_period": damping,
        "achieved_pump_current_a": float(row["pump_current_peak_a"]),
        "pump_power_instrument_dbm": float(row["pump_power_dbm"]),
        "gain_vs_off_db": _number(row.get("gain_vs_off_db")) if passed else None,
        "r_j": _number(row.get("pump_branch_current_max_over_ic")) if passed else None,
        "production_hb_full_residual_rel": residual if passed else None,
        "hill_max_abs_lambda": None,
        "hill_root_frequency_ghz": None,
        "envelope_slope_per_period": None,
        "max_abs_phi_rad": None,
        "status": "VALID_CONVERGED" if passed else status,
        "runtime_s": runtime_s,
        "source_path": source_path,
        "checkpoint_path": str(pump_dir.resolve()) if passed and pump_dir else None,
    }
    if passed and pump_dir is not None:
        sidecar = {
            "circuit_identity": identity,
            "resistance_ratio": ratio,
            "checkpoint_path": str(pump_dir.resolve()),
            "circuit_dir": str(Path(str(metadata.get("circuit_dir", ""))).resolve()),
            "production_hb_full_residual_rel": residual,
        }
        atomic_json(pump_dir / "rcsj_fixture_identity.json", sidecar)
    return record


def run_hb_arm(
    outdir: Path,
    rows: list[dict[str, Any]],
    *,
    arm: str,
    ratio: float,
    circuit_dir: Path,
    identity: str,
    damping: float,
) -> None:
    if all(find_record(rows, arm, ratio, current) is not None for current in TARGET_CURRENTS_A):
        return
    arm_dir = outdir / "hb" / arm.lower() / ratio_slug(ratio)
    args = _hb_args(circuit_dir, arm_dir)
    points_with_roles = continuation_currents()
    points = [
        gain_map.GridPoint(
            index=index,
            i_power=index,
            j_freq=0,
            power_dbm=_point_power_dbm(current, args),
            pump_freq_ghz=PUMP_GHZ,
            current_a=current,
        )
        for index, (current, _is_target) in enumerate(points_with_roles)
    ]
    atomic_json(
        arm_dir / "protocol.json",
        {
            "arm": arm,
            "resistance_ratio": ratio,
            "circuit_identity": identity,
            "circuit_dir": str(circuit_dir.resolve()),
            "target_currents_a": TARGET_CURRENTS_A,
            "continuation_currents_a": [point.current_a for point in points],
            "max_continuation_step_db": max_continuation_step_db(points_with_roles),
            "full_residual_gate": FULL_RESIDUAL_GATE,
            "attenuation_override": None,
        },
    )
    engine = gain_map.InProcessEngine(args)
    started = time.perf_counter()
    hb_rows = gain_map.run_warm_pass_inprocess(points, arm_dir / "pass", engine)
    total_runtime = time.perf_counter() - started
    elapsed_per_row = total_runtime / max(len(hb_rows), 1)
    targets = {round(current, 16) for current in TARGET_CURRENTS_A}
    for hb_row in hb_rows:
        current = float(hb_row["pump_current_peak_a"])
        if round(current, 16) not in targets:
            continue
        record = _hb_record(arm, ratio, damping, identity, hb_row, elapsed_per_row)
        replace_record(rows, record)
        persist_records(outdir, rows)


def gate_g1_g2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    currents = TARGET_CURRENTS_A[:4]
    gain_b_minus_a: list[float] = []
    gain_c_minus_b: list[float] = []
    rj_c_minus_b: list[float] = []
    missing: list[str] = []
    for current in currents:
        arm_a = find_record(rows, "A_FDTD_LOSSLESS", math.inf, current)
        arm_b = find_record(rows, "B_HB_LOSSLESS", math.inf, current)
        arm_c = find_record(rows, "C_HB_RCSJ", PRIMARY_RATIO, current)
        for name, row in (("A", arm_a), ("B", arm_b), ("C", arm_c)):
            if row is None or row.get("gain_vs_off_db") is None:
                missing.append(f"{name}@{current:.7e}")
        if arm_a and arm_b and arm_a.get("gain_vs_off_db") is not None and arm_b.get("gain_vs_off_db") is not None:
            gain_b_minus_a.append(float(arm_b["gain_vs_off_db"]) - float(arm_a["gain_vs_off_db"]))
        if arm_b and arm_c and arm_b.get("gain_vs_off_db") is not None and arm_c.get("gain_vs_off_db") is not None:
            gain_c_minus_b.append(float(arm_c["gain_vs_off_db"]) - float(arm_b["gain_vs_off_db"]))
        if arm_b and arm_c and arm_b.get("r_j") is not None and arm_c.get("r_j") is not None:
            rj_c_minus_b.append(float(arm_c["r_j"]) - float(arm_b["r_j"]))
    g1_max = max((abs(value) for value in gain_b_minus_a), default=None)
    g2_gain_max = max((abs(value) for value in gain_c_minus_b), default=None)
    g2_rj_max = max((abs(value) for value in rj_c_minus_b), default=None)
    return {
        "G1": {
            "status": "NOT_MEASURED" if missing else ("PASS" if g1_max is not None and g1_max <= 1.5 else "FAIL"),
            "threshold_db": 1.5,
            "max_abs_gain_difference_db": g1_max,
            "differences_db": gain_b_minus_a,
            "missing": missing,
        },
        "G2": {
            "status": "NOT_MEASURED" if missing else (
                "PASS"
                if g2_gain_max is not None
                and g2_rj_max is not None
                and g2_gain_max <= 0.1
                and g2_rj_max <= 0.02
                else "FAIL"
            ),
            "gain_threshold_db": 0.1,
            "r_j_threshold": 0.02,
            "max_abs_gain_difference_db": g2_gain_max,
            "max_abs_r_j_difference": g2_rj_max,
            "gain_differences_db": gain_c_minus_b,
            "r_j_differences": rj_c_minus_b,
            "missing": missing,
        },
    }


def fixture_integrity(record: dict[str, Any], circuit_dir: Path) -> dict[str, Any]:
    checkpoint = Path(str(record.get("checkpoint_path") or ""))
    sidecar_path = checkpoint / "rcsj_fixture_identity.json"
    if not checkpoint.is_dir() or not sidecar_path.exists():
        return {"passed": False, "reason": "missing checkpoint or fixture sidecar"}
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_identity = circuit_identity(load_circuit(circuit_dir))
    checks = {
        "circuit_identity": sidecar.get("circuit_identity") == actual_identity,
        "resistance_ratio": float(sidecar.get("resistance_ratio"))
        == float(record["resistance_ratio"]),
        "checkpoint_path": Path(str(sidecar.get("checkpoint_path"))).resolve()
        == checkpoint.resolve(),
        "circuit_dir": Path(str(sidecar.get("circuit_dir"))).resolve()
        == circuit_dir.resolve(),
        "full_residual_gate": float(sidecar.get("production_hb_full_residual_rel"))
        <= FULL_RESIDUAL_GATE,
    }
    return {"passed": all(checks.values()), "checks": checks, "sidecar": str(sidecar_path)}


def run_hill(outdir: Path, rows: list[dict[str, Any]], ratio: float, circuit_dir: Path) -> None:
    targets = [find_record(rows, "C_HB_RCSJ", ratio, current) for current in TARGET_CURRENTS_A]
    if any(record is None or record.get("checkpoint_path") is None for record in targets):
        raise RuntimeError(f"Hill route lacks validated RCSJ checkpoints at R/Rn={ratio:g}")
    hill_path = outdir / "hill" / ratio_slug(ratio) / "hill.json"
    if not hill_path.exists():
        command = [
            sys.executable,
            str(ROOT / "scripts" / "floquet_stability_sweep.py"),
            "--circuit-dir",
            str(circuit_dir),
            "--pump-dir",
            *[str(record["checkpoint_path"]) for record in targets if record is not None],
            "--pump-freq-ghz",
            str(PUMP_GHZ),
            "--sidebands",
            "4",
            "--gamma-nt",
            "1024",
            "--loss-model",
            "current_complex_c",
            "--n-points",
            "700",
            "--mode-spacing-mhz",
            "241.7",
            "--top-k",
            "8",
            "--refine-complex",
            "--out",
            str(hill_path),
        ]
        hill_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = hill_path.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not hill_path.exists():
            raise RuntimeError(f"Hill route failed at R/Rn={ratio:g}; see {log_path}")
    payload = json.loads(hill_path.read_text(encoding="utf-8"))
    sweeps = payload["target"]
    if isinstance(sweeps, dict):
        sweeps = [sweeps]
    if len(sweeps) != len(TARGET_CURRENTS_A):
        raise ValueError("Hill output point count does not match the prescribed column")
    for current, sweep in zip(TARGET_CURRENTS_A, sweeps):
        record = find_record(rows, "C_HB_RCSJ", ratio, current)
        if record is None:
            raise AssertionError("missing arm C record during Hill reduction")
        roots = [root for root in sweep.get("complex_resonances", []) if root.get("converged")]
        if not roots:
            record["status"] = "HILL_NO_CONVERGED_ROOT"
            record["hill_max_abs_lambda"] = None
            record["hill_root_frequency_ghz"] = None
        else:
            root = max(roots, key=lambda item: float(item["floquet"]["magnitude"]))
            record["hill_max_abs_lambda"] = float(root["floquet"]["magnitude"])
            record["hill_root_frequency_ghz"] = float(root["signal_ghz_real"])
            record["runtime_s"] = float(record.get("runtime_s") or 0.0) + float(
                sweep.get("runtime_s", 0.0)
            ) + float(sweep.get("refine_runtime_s", 0.0))
        setting_index = TARGET_CURRENTS_A.index(current)
        setting_path = hill_path.with_name(
            f"{hill_path.stem}.setting_{setting_index:02d}.json"
        )
        record["source_path"] = str(setting_path.resolve())
        persist_records(outdir, rows)


def run_td(
    outdir: Path,
    rows: list[dict[str, Any]],
    ratio: float,
    circuit_dir: Path,
    identity: str,
    damping: float,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for current in TD_CURRENTS_A:
        existing = find_record(rows, "D_TD_RCSJ", ratio, current)
        if existing is not None and existing.get("status") not in (None, ""):
            continue
        checkpoint_record = find_record(rows, "C_HB_RCSJ", ratio, current)
        if checkpoint_record is None:
            raise RuntimeError(f"missing arm C checkpoint record at {current:.7e} A")
        audit = fixture_integrity(checkpoint_record, circuit_dir)
        audits.append({"ratio": ratio, "current_a": current, **audit})
        if not audit["passed"]:
            raise RuntimeError(
                f"G0 fixture mismatch at R/Rn={ratio:g}, current={current:.7e} A: {audit}"
            )
        checkpoint = Path(str(checkpoint_record["checkpoint_path"]))
        point_dir = outdir / "td" / ratio_slug(ratio) / f"i_{current:.7e}".replace("+", "")
        summary_path = point_dir / "summary.json"
        runtime_path = point_dir / "runtime.json"
        if not summary_path.exists():
            command = [
                sys.executable,
                str(ROOT / "scripts" / "h1_transient_branch_transfer.py"),
                "--circuit-dir",
                str(circuit_dir),
                "--checkpoint",
                str(checkpoint),
                "--outdir",
                str(point_dir),
                "--freq-ghz",
                str(PUMP_GHZ),
                "--pump-port",
                str(PUMP_PORT),
                "--out-port",
                str(OUT_PORT),
                "--target-current-a",
                repr(current),
                "--initialization-mode",
                "hb_periodic",
                "--ramp-periods",
                "0",
                "--hold-periods",
                "800",
                "--samples-per-period",
                "64",
                "--method",
                "implicit_trapezoid",
                "--max-step",
                repr(2.0 * math.pi / 64.0),
                "--min-step-theta",
                repr(1.0 / 64.0),
                "--compact-output",
                "--compact-sample-count",
                "512",
                "--compact-history-states",
                "2048",
                "--checkpoint-periods",
                "10",
                "--skip-projection",
            ]
            point_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with (point_dir / "stdout.log").open("w", encoding="utf-8") as stdout:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stdout,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            atomic_json(
                runtime_path,
                {"runtime_s": time.perf_counter() - started, "return_code": completed.returncode},
            )
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
        if not summary_path.exists():
            record = {
                "arm": "D_TD_RCSJ",
                "circuit_identity": identity,
                "resistance_ratio": ratio,
                "damping_per_period": damping,
                "achieved_pump_current_a": current,
                "pump_power_instrument_dbm": checkpoint_record["pump_power_instrument_dbm"],
                "gain_vs_off_db": None,
                "r_j": None,
                "production_hb_full_residual_rel": checkpoint_record[
                    "production_hb_full_residual_rel"
                ],
                "hill_max_abs_lambda": None,
                "hill_root_frequency_ghz": None,
                "envelope_slope_per_period": None,
                "max_abs_phi_rad": None,
                "status": f"TD_FAILED_RETURN_CODE_{runtime.get('return_code', 'UNKNOWN')}",
                "runtime_s": _number(runtime.get("runtime_s")),
                "source_path": str((point_dir / "stdout.log").resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
            }
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            envelope = summary.get("envelope_classification") or {}
            compact_path = point_dir / "td_compact.npz"
            compact = np.load(compact_path, allow_pickle=False) if compact_path.exists() else None
            max_abs_phi = (
                float(np.nanmax(compact["max_abs_phi"]))
                if compact is not None and "max_abs_phi" in compact.files
                else None
            )
            r_j = (
                float(np.nanmax(compact["max_abs_sin_phi"]))
                if compact is not None and "max_abs_sin_phi" in compact.files
                else None
            )
            status = str(summary.get("final_status") or summary.get("classification") or "TD_NO_STATUS")
            if max_abs_phi is not None and max_abs_phi > 5.0:
                status = "BLOWUP"
            record = {
                "arm": "D_TD_RCSJ",
                "circuit_identity": identity,
                "resistance_ratio": ratio,
                "damping_per_period": damping,
                "achieved_pump_current_a": current,
                "pump_power_instrument_dbm": checkpoint_record["pump_power_instrument_dbm"],
                "gain_vs_off_db": None,
                "r_j": r_j,
                "production_hb_full_residual_rel": checkpoint_record[
                    "production_hb_full_residual_rel"
                ],
                "hill_max_abs_lambda": None,
                "hill_root_frequency_ghz": None,
                "envelope_slope_per_period": _number(envelope.get("slope_per_period")),
                "max_abs_phi_rad": max_abs_phi,
                "status": status,
                "runtime_s": _number(runtime.get("runtime_s")),
                "source_path": str(summary_path.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
            }
        replace_record(rows, record)
        persist_records(outdir, rows)
    audit_path = outdir / "g0_fixture_audit.json"
    previous_audits = []
    if audit_path.exists():
        previous_audits = json.loads(
            audit_path.read_text(encoding="utf-8")
        ).get("audits", [])
    keys = {
        (float(item["ratio"]), float(item["current_a"]))
        for item in audits
    }
    combined_audits = [
        item
        for item in previous_audits
        if (float(item["ratio"]), float(item["current_a"])) not in keys
    ] + audits
    atomic_json(audit_path, {"audits": combined_audits})
    return audits


def classifier_audit(outdir: Path) -> list[dict[str, Any]]:
    sources = _fdtd_rows()
    baseline_q_even = float(np.mean([float(row["q_even"]) for row in sources[:5]]))
    baseline_q_dc = float(np.mean([float(row["q_dc"]) for row in sources[:5]]))
    audits: list[dict[str, Any]] = []
    for row in sources:
        point_dir = Path(row["_source_path"]).parent
        branch = np.load(point_dir / "poincare_branches.npz", allow_pickle=False)["upward"]
        spectrum = np.load(point_dir / "spectrum.npz", allow_pickle=False)
        result = classifier.classify_details(
            branch,
            spectrum_frequencies_hz=np.asarray(spectrum["frequency_hz"]),
            drive_hz=PUMP_GHZ * 1.0e9,
            period_multiple_value=int(row["period_multiple"]),
            q_even=float(row["q_even"]),
            q_dc=float(row["q_dc"]),
            baseline_q_even=baseline_q_even,
            baseline_q_dc=baseline_q_dc,
            symmetry_floor_factor=20.0,
            half_integer_line_db=float(row["half_integer_line_db"]),
            half_integer_floor_db=float(row["half_integer_floor_db"]),
            half_integer_gate_db=18.0,
            cluster_tolerance=0.03,
            cluster_tolerance_decay=2.503,
        )
        audits.append(
            {
                "current_a": float(row["pump_current_peak_a_achieved"]),
                "power_instrument_dbm": float(row["pump_power_instrument_dbm"]),
                "verdict": result.verdict,
                "reason": result.reason,
                "period_multiple": result.period_multiple,
                "q_even": result.q_even,
                "q_dc": result.q_dc,
                "poincare_clusters": result.poincare_clusters,
                "sigma_vprime_ps": result.sigma_vprime_ps,
                "spectral_period_doubling": result.spectral_period_doubling,
                "spectral_period_disagreement": result.spectral_period_disagreement,
                "pump_referred_fp_half_db": float(row["pump_referred_fp_half_db"]),
                "pump_referred_3fp_half_db": float(row["pump_referred_3fp_half_db"]),
                "source_path": row["_source_path"],
            }
        )
    implicated = "lambda_to_minus_1_half_pump_2T_basis" if any(
        item["verdict"] in {"PERIOD_DOUBLING", "PERIOD_DOUBLING_ONSET"}
        for item in audits
    ) else "no_route_implicated"
    atomic_json(
        outdir / "classifier_audit.json",
        {
            "baseline_q_even": baseline_q_even,
            "baseline_q_dc": baseline_q_dc,
            "phase5_table_branch_implicated": implicated,
            "ansatz_enabled": False,
            "rows": audits,
        },
    )
    return audits


def _crossing(
    records: list[dict[str, Any]], value_key: str, threshold: float
) -> dict[str, Any] | None:
    usable = sorted(
        [
            row
            for row in records
            if row.get(value_key) is not None
            and row.get("pump_power_instrument_dbm") is not None
            and row.get("status") != "BLOWUP"
        ],
        key=lambda row: float(row["pump_power_instrument_dbm"]),
    )
    for left, right in zip(usable, usable[1:]):
        y0 = float(left[value_key]) - threshold
        y1 = float(right[value_key]) - threshold
        if y0 <= 0.0 < y1:
            x0 = float(left["pump_power_instrument_dbm"])
            x1 = float(right["pump_power_instrument_dbm"])
            fraction = -y0 / (y1 - y0)
            return {
                "power_dbm": x0 + fraction * (x1 - x0),
                "bracket_dbm": [x0, x1],
                "values": [float(left[value_key]), float(right[value_key])],
            }
    return None


def evaluate_gates(outdir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = gate_g1_g2(rows)
    g0_path = outdir / "g0_fixture_audit.json"
    g0_rows = json.loads(g0_path.read_text(encoding="utf-8")).get("audits", []) if g0_path.exists() else []
    gates["G0"] = {
        "status": "PASS" if g0_rows and all(item.get("passed") for item in g0_rows) else "NOT_MEASURED",
        "audited_runs": len(g0_rows),
        "all_passed": bool(g0_rows) and all(item.get("passed") for item in g0_rows),
    }
    rung_results: dict[str, Any] = {}
    for ratio in RATIOS:
        arm_c = [
            row
            for row in rows
            if row["arm"] == "C_HB_RCSJ" and float(row["resistance_ratio"]) == ratio
        ]
        arm_d = [
            row
            for row in rows
            if row["arm"] == "D_TD_RCSJ" and float(row["resistance_ratio"]) == ratio
        ]
        first_four = sorted(arm_c, key=lambda row: float(row["achieved_pump_current_a"]))[:4]
        lambda_values = [row.get("hill_max_abs_lambda") for row in first_four]
        g3_pass = (
            len(lambda_values) == 4
            and all(value is not None for value in lambda_values)
            and float(lambda_values[0]) < 0.999
            and not all(abs(float(value) - 1.0) <= 1.0e-9 for value in lambda_values)
            and all(float(right) >= float(left) for left, right in zip(lambda_values, lambda_values[1:]))
        )
        hill_crossing = _crossing(arm_c, "hill_max_abs_lambda", 1.0)
        td_crossing = _crossing(arm_d, "envelope_slope_per_period", 1.0e-5)
        difference = (
            abs(float(hill_crossing["power_dbm"]) - float(td_crossing["power_dbm"]))
            if hill_crossing and td_crossing
            else None
        )
        rung_results[ratio_slug(ratio)] = {
            "G3": {
                "status": "PASS" if g3_pass else ("FAIL" if len(first_four) == 4 else "NOT_MEASURED"),
                "lowest_threshold": 0.999,
                "first_four_max_abs_lambda": lambda_values,
                "monotone": bool(lambda_values) and all(
                    right is not None and left is not None and float(right) >= float(left)
                    for left, right in zip(lambda_values, lambda_values[1:])
                ),
            },
            "hill_crossing": hill_crossing,
            "td_crossing": td_crossing,
            "G4": {
                "status": "PASS" if difference is not None and difference <= 0.5 else (
                    "FAIL" if hill_crossing or td_crossing else "NOT_MEASURED"
                ),
                "threshold_db": 0.5,
                "absolute_crossing_difference_db": difference,
            },
        }
    gates["rungs"] = rung_results
    crossings = [
        rung_results[ratio_slug(ratio)]["hill_crossing"]
        for ratio in RATIOS
    ]
    powers = [item["power_dbm"] if item else None for item in crossings]
    trend_available = all(power is not None for power in powers)
    lossless_bracket = (-24.25, -23.4211)
    distance = [
        0.0
        if lossless_bracket[0] <= float(power) <= lossless_bracket[1]
        else min(abs(float(power) - edge) for edge in lossless_bracket)
        for power in powers
    ] if trend_available else []
    converging = trend_available and all(
        right <= left for left, right in zip(distance, distance[1:])
    )
    gates["G5"] = {
        "status": "PASS" if converging and distance[-1] == 0.0 else (
            "FAIL" if trend_available else "NOT_MEASURED"
        ),
        "ratios": RATIOS,
        "hill_crossing_powers_dbm": powers,
        "distance_to_lossless_td_bracket_db": distance,
        "lossless_td_bracket_dbm": lossless_bracket,
        "trend_toward_lossless_limit": converging,
        "fit_performed": False,
    }
    classifier_path = outdir / "classifier_audit.json"
    gates["G6"] = {
        "status": "PASS" if classifier_path.exists() else "NOT_MEASURED",
        "source_path": str(classifier_path.resolve()),
        "ansatz_enabled": False,
    }
    source_paths = [str(row.get("source_path") or "") for row in rows]
    distinct = all(source_paths) and len(source_paths) == len(set(source_paths))
    gates["source_path_assertion"] = {
        "status": "PASS" if distinct else "FAIL",
        "rows": len(source_paths),
        "distinct_paths": len(set(source_paths)),
    }
    atomic_json(outdir / "gate_report.json", gates)
    return gates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--stage",
        choices=("prepare", "primary", "all", "report"),
        default="all",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"twpa_solver={__import__('twpa_solver').__file__}", flush=True)
    rows = load_records(outdir)
    fdtd = install_arm_a(outdir, rows)
    classifier_audit(outdir)
    if args.stage == "report":
        evaluate_gates(outdir, rows)
        return 0
    if args.stage == "prepare":
        evaluate_gates(outdir, rows)
        return 0

    base = load_circuit(CIRCUIT_DIR)
    base_identity = circuit_identity(base)
    run_hb_arm(
        outdir,
        rows,
        arm="B_HB_LOSSLESS",
        ratio=math.inf,
        circuit_dir=CIRCUIT_DIR,
        identity=base_identity,
        damping=0.0,
    )
    variants: dict[float, tuple[Path, str, float]] = {}
    for ratio in RATIOS:
        variant, _parameters, identity, damping = build_variant(outdir, ratio)
        variants[ratio] = (variant, identity, damping)
    variant, identity, damping = variants[PRIMARY_RATIO]
    run_hb_arm(
        outdir,
        rows,
        arm="C_HB_RCSJ",
        ratio=PRIMARY_RATIO,
        circuit_dir=variant,
        identity=identity,
        damping=damping,
    )
    primary_gates = gate_g1_g2(rows)
    atomic_json(outdir / "primary_gate_report.json", primary_gates)
    if primary_gates["G2"]["status"] != "PASS":
        evaluate_gates(outdir, rows)
        print(json.dumps(primary_gates, indent=2), flush=True)
        print("G2 did not pass; G3-G5 were not started.", flush=True)
        return 2
    if args.stage == "primary":
        evaluate_gates(outdir, rows)
        return 0

    for ratio in RATIOS:
        variant, identity, damping = variants[ratio]
        run_hb_arm(
            outdir,
            rows,
            arm="C_HB_RCSJ",
            ratio=ratio,
            circuit_dir=variant,
            identity=identity,
            damping=damping,
        )
        run_hill(outdir, rows, ratio, variant)
        run_td(outdir, rows, ratio, variant, identity, damping)
    gates = evaluate_gates(outdir, rows)
    if args.stage in {"all", "report"}:
        atomic_json(
            outdir / "report.json",
            {
                "circuit": str(CIRCUIT_DIR.resolve()),
                "pump_ghz": PUMP_GHZ,
                "pump_port": PUMP_PORT,
                "source_port": SOURCE_PORT,
                "out_port": OUT_PORT,
                "attenuation_model": "A10; no override",
                "attenuation_db_at_7p9_ghz": float(fdtd[0]["loss_attenuation_db"]),
                "power_convention": "legacy_traveling_wave",
                "rcsj_interpretation": "numerical regularizer; no ratio is a device property",
                "new_ansatz_enabled": False,
                "gates": gates,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
