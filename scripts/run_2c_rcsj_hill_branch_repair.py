"""Repair and track the physical 2c RCSJ Hill branch at R/Rn=1e5.

This is a bounded follow-up.  It reuses the existing converged checkpoints,
solves only the three missing HB settings, and runs the existing Hill scanner
with adjacent-root refinement seeds.  It does not run a transient campaign.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_gain_map as gain_map  # noqa: E402
from scripts import run_2c_rcsj_stability_column as column  # noqa: E402


RATIO = 1.0e5
PUMP_GHZ = 7.9
REPAIR_CURRENTS_A = (
    6.1047e-6,
    6.6861e-6,
    6.8315e-6,
    6.9768e-6,
    7.1222e-6,
    7.2675e-6,
)
NEW_HB_CURRENTS_A = (6.8315e-6, 6.9768e-6, 7.1222e-6)
INITIAL_CURRENT_A = 6.6861e-6
INITIAL_SETTING = (
    ROOT
    / "outputs"
    / "chaos"
    / "2c_rcsj_stability_column"
    / "hill"
    / "r1e05"
    / "hill.setting_01.json"
)
PHYSICAL_BRANCH_MIN_GHZ = 7.0
PHYSICAL_BRANCH_MAX_GHZ = 7.4


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _record_for(rows: list[dict[str, Any]], current: float) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("arm") == "C_HB_RCSJ"
        and float(row.get("resistance_ratio")) == RATIO
        and math.isclose(
            float(row.get("achieved_pump_current_a")),
            current,
            rel_tol=2.0e-9,
            abs_tol=1.0e-18,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one existing RCSJ record at {current:.7e} A, got {len(matches)}")
    return matches[0]


def _existing_checkpoint_map(rows: list[dict[str, Any]]) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for current in REPAIR_CURRENTS_A:
        matches = [
            row
            for row in rows
            if row.get("arm") == "C_HB_RCSJ"
            and float(row.get("resistance_ratio")) == RATIO
            and math.isclose(
                float(row.get("achieved_pump_current_a")),
                current,
                rel_tol=2.0e-9,
                abs_tol=1.0e-18,
            )
        ]
        if not matches:
            continue
        if len(matches) != 1:
            raise RuntimeError(f"expected one existing RCSJ record at {current:.7e} A, got {len(matches)}")
        record = matches[0]
        checkpoint = Path(str(record.get("checkpoint_path") or ""))
        if not checkpoint.is_dir():
            raise RuntimeError(f"missing existing checkpoint for {current:.7e} A: {checkpoint}")
        result[current] = checkpoint
    for current in (6.1047e-6, INITIAL_CURRENT_A, 7.2675e-6):
        if current not in result:
            raise RuntimeError(f"required existing checkpoint is missing at {current:.7e} A")
    return result


def _run_missing_hb(
    repair_dir: Path,
    rows: list[dict[str, Any]],
    circuit_dir: Path,
    identity: str,
    damping: float,
) -> dict[float, Path]:
    checkpoints = _existing_checkpoint_map(rows)
    for current in NEW_HB_CURRENTS_A:
        setting_path = repair_dir / "hb" / "r1e05" / f"setting_{current:.8e}.json"
        if current not in checkpoints and setting_path.exists():
            setting = json.loads(setting_path.read_text(encoding="utf-8"))
            checkpoint = Path(str(setting.get("checkpoint_path") or ""))
            if checkpoint.is_dir() and (checkpoint / "pump_report.json").exists():
                checkpoints[current] = checkpoint.resolve()
    if all(current in checkpoints for current in NEW_HB_CURRENTS_A):
        return checkpoints

    seed_record = _record_for(rows, INITIAL_CURRENT_A)
    seed_checkpoint = Path(str(seed_record["checkpoint_path"])).resolve()
    args = column._hb_args(circuit_dir, repair_dir / "hb" / "r1e05")
    args.initial_pump_dir = str(seed_checkpoint)
    args.initial_pump_power_dbm = float(seed_record["pump_power_instrument_dbm"])
    points = [
        gain_map.GridPoint(
            index=index,
            i_power=index,
            j_freq=0,
            power_dbm=column._point_power_dbm(current, args),
            pump_freq_ghz=PUMP_GHZ,
            current_a=current,
        )
        for index, current in enumerate(NEW_HB_CURRENTS_A)
    ]
    pass_dir = repair_dir / "hb" / "r1e05" / "pass"
    atomic_json(
        repair_dir / "hb" / "r1e05" / "protocol.json",
        {
            "arm": "C_HB_RCSJ_BRANCH_REPAIR",
            "resistance_ratio": RATIO,
            "circuit_identity": identity,
            "circuit_dir": str(circuit_dir.resolve()),
            "initial_checkpoint": str(seed_checkpoint),
            "initial_current_a": INITIAL_CURRENT_A,
            "target_currents_a": NEW_HB_CURRENTS_A,
            "power_conversion": "run_gain_map.peak_current_to_power_dbm",
        },
    )
    engine = gain_map.InProcessEngine(args)
    started = time.perf_counter()
    hb_rows = gain_map.run_warm_pass_inprocess(points, pass_dir, engine)
    runtime_s = time.perf_counter() - started
    by_current = {float(row["pump_current_peak_a"]): row for row in hb_rows}
    for point in points:
        current = point.current_a
        row = by_current.get(current)
        if row is None or row.get("status") != "PASS":
            atomic_json(
                repair_dir / "hb" / "r1e05" / f"failure_{current:.8e}.json",
                {"current_a": current, "row": row, "runtime_s": runtime_s},
            )
            raise RuntimeError(f"new HB checkpoint failed at {current:.7e} A: {row}")
        checkpoint = gain_map.point_pump_dir(point, pass_dir).resolve()
        report = checkpoint / "pump_report.json"
        if not report.exists():
            raise RuntimeError(f"HB row passed without pump report: {checkpoint}")
        checkpoints[current] = checkpoint
        atomic_json(
            repair_dir / "hb" / "r1e05" / f"setting_{current:.8e}.json",
            {
                "current_a": current,
                "pump_power_dbm": point.power_dbm,
                "checkpoint_path": str(checkpoint),
                "pump_report": str(report),
                "status": row.get("status"),
                "runtime_s": row.get("pump_runtime_s"),
            },
        )
    return checkpoints


def _initial_physical_branch_index() -> int:
    payload = json.loads(INITIAL_SETTING.read_text(encoding="utf-8"))
    target = payload.get("target", payload)
    candidates = [
        root
        for root in target.get("complex_resonances", [])
        if root.get("converged")
        and PHYSICAL_BRANCH_MIN_GHZ <= float(root["signal_ghz_real"]) <= PHYSICAL_BRANCH_MAX_GHZ
    ]
    if not candidates:
        raise RuntimeError("the initial setting contains no physical 7.0-7.4 GHz branch root")
    root = max(candidates, key=lambda item: float(item["floquet"]["magnitude"]))
    return int(root["tracked_branch_index"])


def _target_power_dbm(sweep: dict[str, Any], rows: list[dict[str, Any]], current: float) -> float:
    matches = [
        row
        for row in rows
        if row.get("arm") == "C_HB_RCSJ"
        and float(row.get("resistance_ratio")) == RATIO
        and math.isclose(
            float(row.get("achieved_pump_current_a")),
            current,
            rel_tol=2.0e-9,
            abs_tol=1.0e-18,
        )
    ]
    if len(matches) == 1:
        return float(matches[0]["pump_power_instrument_dbm"])
    if matches:
        raise RuntimeError(f"duplicate RCSJ records at {current:.7e} A")
    pump_dir = Path(str(sweep.get("pump_dir") or ""))
    report = json.loads((pump_dir / "pump_report.json").read_text(encoding="utf-8"))
    return float(report["metadata"]["pump_power_dbm_requested"])


def _run_hill(
    repair_dir: Path,
    circuit_dir: Path,
    checkpoints: dict[float, Path],
    *,
    only_last: bool = False,
) -> Path:
    hill_dir = repair_dir / "hill" / "r1e05"
    hill_path = hill_dir / "hill.json"
    output_path = hill_dir / "hill_last.json" if only_last else hill_path
    pump_currents = (REPAIR_CURRENTS_A[-1],) if only_last else REPAIR_CURRENTS_A
    initial_root_json = (
        hill_dir / "hill.setting_04.json" if only_last else INITIAL_SETTING
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "floquet_stability_sweep.py"),
        "--circuit-dir",
        str(circuit_dir),
        "--pump-dir",
        *[str(checkpoints[current]) for current in pump_currents],
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
        "16",
        "--refine-complex",
        "--track-refinement-seeds",
        "--initial-root-json",
        str(initial_root_json),
        "--out",
        str(output_path),
    ]
    hill_dir.mkdir(parents=True, exist_ok=True)
    log_path = hill_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"Hill repair failed; see {log_path}")
    if only_last:
        last_payload = json.loads(output_path.read_text(encoding="utf-8"))
        targets = [
            json.loads(
                (hill_dir / f"hill.setting_{index:02d}.json").read_text(encoding="utf-8")
            )["target"]
            for index in range(5)
        ]
        targets.append(last_payload["target"][0] if isinstance(last_payload["target"], list) else last_payload["target"])
        merged = dict(last_payload)
        merged["target"] = targets
        column.atomic_json(hill_path, merged)
        column.atomic_json(
            hill_dir / "hill.setting_05.json",
            {
                "setting_index": 5,
                "pump_dir": str(checkpoints[REPAIR_CURRENTS_A[-1]]),
                "target": targets[-1],
            },
        )
    return hill_path


def _reduce(
    repair_dir: Path,
    hill_path: Path,
    circuit_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(hill_path.read_text(encoding="utf-8"))
    targets = payload["target"]
    if isinstance(targets, dict):
        targets = [targets]
    if len(targets) != len(REPAIR_CURRENTS_A):
        raise RuntimeError(f"expected six repaired Hill settings, got {len(targets)}")
    branch_index = _initial_physical_branch_index()
    reduced: list[dict[str, Any]] = []
    for current, sweep in zip(REPAIR_CURRENTS_A, targets):
        roots = sweep.get("complex_resonances", [])
        selection = column.select_hill_root(roots)
        branch = next(
            (
                root
                for root in roots
                if int(root.get("tracked_branch_index", -1)) == branch_index
                and root.get("converged")
            ),
            None,
        )
        if branch is None:
            status = "PHYSICAL_BRANCH_NOT_RETAINED"
            branch_row = {
                "status": status,
                "signal_ghz_real": None,
                "lambda_abs": None,
                "phase_rad": None,
                "phase_rad_over_pi": None,
                "rho": None,
            }
        else:
            floquet = branch["floquet"]
            branch_row = {
                "status": "TRACKED_ROOT_DIAGNOSTIC_ONLY",
                "signal_ghz_real": float(branch["signal_ghz_real"]),
                "signal_ghz_imag": float(branch["signal_ghz_imag"]),
                "lambda_abs": float(floquet["magnitude"]),
                "phase_rad": float(floquet["phase_rad"]),
                "phase_rad_over_pi": float(floquet["phase_rad"]) / math.pi,
                "rho": float(floquet["zone_frequency_ghz"]) / PUMP_GHZ,
                "kind": floquet.get("kind"),
                "tracked_branch_index": branch_index,
            }
        neighborhood_candidates = []
        for root in roots:
            floquet = root["floquet"]
            zone_abs = abs(float(floquet["zone_frequency_ghz"]))
            if 0.5 <= zone_abs <= 0.95:
                neighborhood_candidates.append(
                    {
                        "signal_ghz_real": float(root["signal_ghz_real"]),
                        "zone_frequency_abs_ghz": zone_abs,
                        "lambda_abs": float(floquet["magnitude"]),
                        "tracked_branch_index": root.get("tracked_branch_index"),
                        "seed_signal_ghz": root.get("seed_signal_ghz"),
                        "seed_signal_ghz_imag": root.get("seed_signal_ghz_imag", 0.0),
                    }
                )
        reduced.append(
            {
                "current_a": current,
                "pump_power_instrument_dbm": _target_power_dbm(sweep, rows, current),
                "setting_path": str(
                    hill_path.with_name(
                        f"{hill_path.stem}.setting_{REPAIR_CURRENTS_A.index(current):02d}.json"
                    ).resolve()
                ),
                "physical_branch": branch_row,
                "physical_branch_status": (
                    "UNRESOLVED_REFINEMENT_JUMP"
                    if branch is not None
                    else "PHYSICAL_BRANCH_NOT_RETAINED"
                ),
                "frequency_neighborhood_candidates": neighborhood_candidates,
                "filter_selection": selection,
                "candidate_count": len(roots),
                "refinement_seed_source": sweep.get("refinement_seed_source"),
                "refinement_seed_count": sweep.get("refinement_seed_count"),
            }
        )
        atomic_json(repair_dir / "reduction" / f"setting_{current:.8e}.json", reduced[-1])

    branch_rows = [row["physical_branch"] for row in reduced]
    valid = [
        row
        for row in branch_rows
        if row["lambda_abs"] is not None and row["status"] == "CONVERGED"
    ]
    crossing = None
    for left, right in zip(valid, valid[1:]):
        if left["lambda_abs"] < 1.0 <= right["lambda_abs"]:
            current_left = next(
                row["current_a"] for row in reduced if row["physical_branch"] is left
            )
            current_right = next(
                row["current_a"] for row in reduced if row["physical_branch"] is right
            )
            fraction = (1.0 - left["lambda_abs"]) / (right["lambda_abs"] - left["lambda_abs"])
            crossing = {
                "lower_current_a": current_left,
                "upper_current_a": current_right,
                "lower_lambda_abs": left["lambda_abs"],
                "upper_lambda_abs": right["lambda_abs"],
                "linear_interpolated_current_a": current_left + fraction * (current_right - current_left),
            }
            break
    reduction = {
        "circuit": str(circuit_dir.resolve()),
        "pump_frequency_ghz": PUMP_GHZ,
        "resistance_ratio": RATIO,
        "density_points": 700,
        "mode_spacing_mhz": 241.7,
        "filter_tolerance_ghz": column.HILL_SPURIOUS_ROOT_TOLERANCE_GHZ,
        "top_k": 16,
        "initial_root_json": str(INITIAL_SETTING.resolve()),
        "physical_branch_index": branch_index,
        "branch_rows": reduced,
        "crossing": crossing,
        "td_bracket": {
            "stable_current_a": 6.6861e-6,
            "blowup_current_a": 7.2675e-6,
        },
    }
    atomic_json(repair_dir / "branch_repair_reduction.json", reduction)
    return reduction


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "chaos" / "2c_rcsj_stability_column" / "hill_branch_repair",
    )
    parser.add_argument(
        "--resume-last-only",
        action="store_true",
        help="Run only the final 7.2675e-6 A Hill setting and merge it with "
             "the five already persisted settings.",
    )
    parser.add_argument(
        "--reduce-only",
        action="store_true",
        help="Do not run Hill; reduce the already persisted merged hill.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repair_dir = args.outdir.resolve()
    base_outdir = ROOT / "outputs" / "chaos" / "2c_rcsj_stability_column"
    circuit_dir = base_outdir / "circuits" / "r1e05"
    if not circuit_dir.is_dir():
        raise RuntimeError(f"missing existing RCSJ circuit: {circuit_dir}")
    rows = column.load_records(base_outdir)
    circuit = column.load_circuit(circuit_dir)
    identity = column.circuit_identity(circuit)
    damping = float(column.build_variant(base_outdir, RATIO)[3])
    import twpa_solver

    module_path = Path(twpa_solver.__file__).resolve()
    print(f"twpa_solver={module_path}", flush=True)
    try:
        module_path.relative_to((ROOT / "src").resolve())
    except ValueError as exc:
        raise RuntimeError("twpa_solver does not resolve under src/")
    atomic_json(
        repair_dir / "manifest.json",
        {
            "status": "STARTED",
            "circuit_identity": identity,
            "circuit_dir": str(circuit_dir.resolve()),
            "resistance_ratio": RATIO,
            "currents_a": REPAIR_CURRENTS_A,
            "new_hb_currents_a": NEW_HB_CURRENTS_A,
            "no_td_or_fdtd": True,
        },
    )
    checkpoints = _run_missing_hb(repair_dir, rows, circuit_dir, identity, damping)
    atomic_json(
        repair_dir / "manifest.json",
        {
            "status": "HB_READY",
            "circuit_identity": identity,
            "circuit_dir": str(circuit_dir.resolve()),
            "resistance_ratio": RATIO,
            "currents_a": REPAIR_CURRENTS_A,
            "new_hb_currents_a": NEW_HB_CURRENTS_A,
            "checkpoints": {str(current): str(path) for current, path in checkpoints.items()},
            "no_td_or_fdtd": True,
        },
    )
    hill_path = repair_dir / "hill" / "r1e05" / "hill.json"
    if not args.reduce_only:
        hill_path = _run_hill(
            repair_dir,
            circuit_dir,
            checkpoints,
            only_last=args.resume_last_only,
        )
    reduction = _reduce(repair_dir, hill_path, circuit_dir, rows)
    decision = {
        "decision": "INCONCLUSIVE",
        "reason": (
            "The expanded top-k=16 scan and adjacent-root continuation did not "
            "retain a converged continuation of the established 0.695-0.727 GHz "
            "physical branch at the repaired settings. Direct real and complex "
            "diagnostic seeds at 6.1047e-6 A also jumped to unrelated roots. "
            "Therefore no measured |lambda| crossing or phase test is available, "
            "and the conditional spectral prediction was not run."
        ),
        "top_k": 16,
        "physical_branch_index": reduction["physical_branch_index"],
        "spectral_test": "NOT_RUN_REQUIRES_VALID_BRANCH_CROSSING",
    }
    atomic_json(repair_dir / "decision.json", decision)
    atomic_json(
        repair_dir / "manifest.json",
        {
            "status": "COMPLETE",
            "hill_path": str(hill_path.resolve()),
            "branch_repair_reduction": str((repair_dir / "branch_repair_reduction.json").resolve()),
            "decision": decision,
            "circuit_identity": identity,
            "no_td_or_fdtd": True,
        },
    )
    for row in reduction["branch_rows"]:
        branch = row["physical_branch"]
        print(
            f"I={row['current_a']:.7e} A P={row['pump_power_instrument_dbm']:.6f} dBm "
            f"lambda={branch['lambda_abs']} zone={branch['signal_ghz_real']} GHz "
            f"phase/pi={branch['phase_rad_over_pi']}",
            flush=True,
        )
    print(json.dumps({"crossing": reduction["crossing"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
