"""Run and plot a KIMPA 3WM gain map using peak I/Ic as the pump axis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_kimpa_gain_map import main


if __name__ == "__main__":
    raise SystemExit(main())
