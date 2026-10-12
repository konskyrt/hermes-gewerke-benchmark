"""BuildSync Gewerke-Benchmark - Streamlit Community Cloud entry point.

One page: a card per Gewerk comparing every schedule that contains it -
phases by project and the Gewerk's flow over time (m³, LKW, m², progress).
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "app_pages" / "gewerk_comparison.py"),
               run_name="__main__")
