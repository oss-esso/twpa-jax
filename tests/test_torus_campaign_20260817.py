from __future__ import annotations

import json

import numpy as np

from scripts.chaos.torus_campaign_20260817 import atomic_json


def test_campaign_atomic_json_writes_parent_and_valid_payload(tmp_path) -> None:
    target = tmp_path / "nested" / "point.json"

    atomic_json(target, {"status": "complete", "values": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "status": "complete",
        "values": [1, 2, 3],
    }
    assert not list(target.parent.glob("*.tmp"))


def test_campaign_result_serialization_accepts_numpy_scalars(tmp_path) -> None:
    target = tmp_path / "numpy.json"

    atomic_json(target, {"value": np.float64(1.25)})

    assert json.loads(target.read_text(encoding="utf-8"))["value"] == 1.25
