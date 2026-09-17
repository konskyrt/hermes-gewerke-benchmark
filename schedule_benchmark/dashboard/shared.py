"""Shared paths, colours and data loaders for the Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PACKAGE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PACKAGE_DIR / "out"
REPO_ROOT = PACKAGE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GEWERK_COLORS = {
    "KO": "#8d6e63", "ENV": "#546e7a", "INT": "#ffb300", "SN": "#26a69a",
    "HZ": "#e53935", "KT": "#29b6f6", "LF": "#7e57c2", "EL": "#fdd835",
    "SPR": "#d81b60", "AUTO": "#43a047", "SEC": "#5c6bc0", "GAS": "#fb8c00",
    "SPEC": "#00897b", "TRANS": "#6d4c41", "EQUIP": "#ec407a", "SITE": "#9ccc65",
    "OTH": "#bdbdbd",
}

MONTAGE_COLORS = {
    "GROB": "#6d4c41",
    "FEIN": "#fb8c00",
    "END": "#43a047",
}
MONTAGE_ORDER = ["Grobmontage", "Feinmontage", "Endmontage"]

DC30_COLOR = "#1565c0"
PEER_COLOR = "#90a4ae"


def ensure_build() -> bool:
    if (OUT_DIR / "tasks_all.parquet").is_file():
        return True
    st.error(
        "No build output found. Run `python -m schedule_benchmark.src.build` first."
    )
    return False


@st.cache_data(show_spinner=False)
def load_tasks() -> pd.DataFrame:
    return pd.read_parquet(OUT_DIR / "tasks_all.parquet")


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    path = OUT_DIR / f"{name}.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_coverage() -> dict:
    path = OUT_DIR / "coverage.json"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def days(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.0f} d".replace(",", "'")
