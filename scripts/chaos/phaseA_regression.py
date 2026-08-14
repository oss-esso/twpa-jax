"""Compare the committed and working Phase A classifiers on saved fixtures."""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from scripts.chaos import attractor_classify as current


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "chaos" / "phaseA" / "classifier_regression.json"


def load_committed_module() -> types.ModuleType:
    source = subprocess.check_output(
        ["git", "show", "HEAD:scripts/chaos/attractor_classify.py"],
        cwd=ROOT,
        text=True,
    )
    module = types.ModuleType("attractor_classify_committed")
    sys.modules[module.__name__] = module
    exec(compile(source, "attractor_classify.py@HEAD", "exec"), module.__dict__)
    return module


def classify_details_fixture(
    module: types.ModuleType, points: np.ndarray, frequencies: np.ndarray,
    drive_hz: float,
) -> dict[str, object]:
    result = module.classify_details(
        points, spectrum_frequencies_hz=frequencies, drive_hz=drive_hz,
    )
    return {"verdict": result.verdict, "clusters": result.poincare_clusters}


def fig2a_fixture(old: types.ModuleType) -> dict[str, object]:
    root = ROOT / "outputs" / "chaos" / "phase2" / "fig2a_50ohm_mtls" / "run"
    map_data = np.load(root / "spectra_map.npz", allow_pickle=False)
    frequencies = np.asarray(map_data["frequency_ghz"], dtype=float) * 1.0e9
    spectrum = np.asarray(map_data["spectrum_dbm"], dtype=float)
    rows: list[dict[str, object]] = []
    for index, path in enumerate(sorted(root.glob("point_*/poincare_branches.npz"))):
        branches = np.load(path, allow_pickle=False)
        points = np.asarray(branches["upward"], dtype=float)
        order = np.argsort(spectrum[index])[::-1][:8]
        drive_hz = 7.0e9
        old_result = classify_details_fixture(old, points, frequencies[order], drive_hz)
        new_result = classify_details_fixture(
            current, points, frequencies[order], drive_hz,
        )
        rows.append({
            "point": index,
            "pump_power_dbm": float(map_data["x"][index]),
            "old": old_result,
            "new": new_result,
            "changed": old_result != new_result,
            "path": str(path.relative_to(ROOT)),
        })
    return {
        "fixture": "Fig 2(a) fig2a_50ohm_mtls",
        "points": len(rows),
        "mode": "classify_details",
        "changed_count": sum(bool(row["changed"]) for row in rows),
        "rows": rows,
    }


def trace_fixture(old: types.ModuleType, paths: list[Path], name: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in paths:
        data = np.load(path, allow_pickle=False)
        time = np.asarray(data["t_s"], dtype=float)
        voltage = np.asarray(data["vout_v"], dtype=float)
        old_result = old.classify_trace(time, voltage, drive_hz=7.0e9).as_dict()
        new_result = current.classify_trace(time, voltage, drive_hz=7.0e9).as_dict()
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "old_verdict": old_result["verdict"],
            "new_verdict": new_result["verdict"],
            "old_clusters": old_result["poincare_clusters"],
            "new_clusters": new_result["poincare_clusters"],
            "changed": old_result["verdict"] != new_result["verdict"]
            or old_result["poincare_clusters"] != new_result["poincare_clusters"],
        })
    return {
        "fixture": name,
        "points": len(rows),
        "mode": "classify_trace",
        "changed_count": sum(bool(row["changed"]) for row in rows),
        "old_verdict_counts": _counts(row["old_verdict"] for row in rows),
        "new_verdict_counts": _counts(row["new_verdict"] for row in rows),
        "changed_rows": [row for row in rows if row["changed"]],
        "rows": rows,
    }


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def main() -> int:
    old = load_committed_module()
    bias_paths = sorted(
        (ROOT / "outputs" / "chaos" / "phase2" / "fig4_bias_0_34_m55").rglob(
            "timeseries.npz"
        )
    )
    single_paths = sorted(
        path for path in (ROOT / "outputs" / "chaos" / "phase2").rglob("timeseries.npz")
        if "fig4_bias_0_34_m55" not in str(path)
    )
    payload = {
        "baseline_source": "git show HEAD:scripts/chaos/attractor_classify.py",
        "working_source": "scripts/chaos/attractor_classify.py",
        "fixtures": [
            fig2a_fixture(old),
            trace_fixture(old, bias_paths, "Fig 4 bias sweep"),
            trace_fixture(old, single_paths, "single -54 dBm points"),
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "summary": [
            {key: fixture[key] for key in ("fixture", "points", "changed_count")}
            for fixture in payload["fixtures"]
        ],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
