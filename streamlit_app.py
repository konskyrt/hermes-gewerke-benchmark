"""Hermes Gewerke-Benchmark — cross-schedule trade duration dashboard.

Streamlit Community Cloud entry point. Deploy this file as the main script.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(
    str(ROOT / "schedule_benchmark" / "dashboard" / "app.py"),
    run_name="__main__",
)
