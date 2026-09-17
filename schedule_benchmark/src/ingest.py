"""Read every registered schedule file into one frame of canonical raw columns.

Ingest only renames and carries provenance. Type coercion, hierarchy and the
classification axes all happen later, in normalize.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .config import enabled_sources, load_yaml, schedule_root

# Canonical raw column names a source may map onto. Anything not listed here is
# dropped at ingest, which is what silently discards ELP V6's duplicated
# `BKP.1`/`Bereich.1`/... columns.
RAW_FIELDS = [
    "task_id",
    "task_name",
    "task_category",
    "duration",
    "start",
    "finish",
    "wbs",
    "wbs_level",
    "full_path",
    "parent_name",
    "is_summary_raw",
    "is_milestone_raw",
    "is_critical_raw",
    "building",
    "floor_raw",
    "zone",
    "room",
    "gewerk_raw",
    "phase_raw",
    "bkp",
    "contractor",
    "contractor_2",
    "procurement_package",
    "crew_size",
    "usage",
    "pct_complete",
    "takt_nr",
    "description_en",
    "material_hint",
    "keywords",
    "predecessors",
]


def _read_excel(path: Path, source: dict[str, Any]) -> pd.DataFrame:
    sheet = source.get("sheet", 0)
    return pd.read_excel(path, sheet_name=sheet, dtype=object)


def _read_csv_semicolon(path: Path, source: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        dtype=object,
        engine="python",
        keep_default_na=True,
        na_values=["NA", "N/A", ""],
    )


READERS = {"excel": _read_excel, "csv_semicolon": _read_csv_semicolon}


def read_source(source: dict[str, Any], root: Path) -> pd.DataFrame:
    """Read one registered file and rename its columns to the canonical raw names."""
    path = root / source["file"]
    if not path.is_file():
        raise FileNotFoundError(f"{source['schedule_id']}: missing source file {path}")

    reader = READERS[source["reader"]]
    raw = reader(path, source)

    missing = [
        src_col
        for src_col in source.get("columns", {}).values()
        if src_col not in raw.columns
    ]
    if missing:
        raise KeyError(
            f"{source['schedule_id']}: columns declared in sources.yaml are absent "
            f"from {path.name}: {missing}"
        )

    frame = pd.DataFrame(index=raw.index)
    for field, src_col in source.get("columns", {}).items():
        if field not in RAW_FIELDS:
            raise KeyError(
                f"{source['schedule_id']}: '{field}' is not a canonical raw field. "
                f"Add it to RAW_FIELDS or fix sources.yaml."
            )
        frame[field] = raw[src_col]

    for field, value in source.get("constants", {}).items():
        frame[field] = value

    for field in RAW_FIELDS:
        if field not in frame.columns:
            frame[field] = pd.NA

    frame = frame[RAW_FIELDS]
    frame.insert(0, "schedule_id", source["schedule_id"])
    frame.insert(1, "project", source["project"])
    frame.insert(2, "schedule_label", source["label"])
    frame.insert(3, "source_file", path.name)
    frame.insert(4, "row_index", range(len(frame)))
    frame["duration_unit"] = source.get("duration_unit", "mpp")
    frame["parser"] = source.get("parser", pd.NA)
    return frame


def content_fingerprint(frame: pd.DataFrame) -> str:
    """Order-insensitive hash of the columns that carry meaning.

    Used to confirm the duplicate-file claim in sources.yaml without depending
    on column order, which is the only thing that differs between the two
    Zentrum Bären exports.
    """
    cols = ["task_id", "task_name", "start", "finish", "duration", "wbs", "gewerk_raw"]
    subset = frame[cols].astype(str).sort_values(cols).reset_index(drop=True)
    return hashlib.sha256(subset.to_csv(index=False).encode("utf-8")).hexdigest()


def check_duplicates(cfg: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Verify every `duplicate_of` claim. Returns one report row per claim."""
    reports = []
    for source in cfg["sources"]:
        twin_id = source.get("duplicate_of")
        if not twin_id:
            continue
        twin = next(s for s in cfg["sources"] if s["schedule_id"] == twin_id)
        mine = content_fingerprint(read_source(source, root))
        theirs = content_fingerprint(read_source(twin, root))
        reports.append(
            {
                "schedule_id": source["schedule_id"],
                "duplicate_of": twin_id,
                "identical": mine == theirs,
                "fingerprint": mine,
                "twin_fingerprint": theirs,
            }
        )
    return reports


def ingest_all(verbose: bool = True) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    cfg = load_yaml("sources.yaml")
    root = schedule_root(cfg)

    frames = []
    for source in enabled_sources(cfg):
        frame = read_source(source, root)
        if verbose:
            print(f"  ingested {source['schedule_id']:<16} {len(frame):>6} rows")
        frames.append(frame)

    duplicate_reports = check_duplicates(cfg, root)
    for report in duplicate_reports:
        verdict = "confirmed identical" if report["identical"] else "DIFFERS"
        if verbose:
            print(
                f"  duplicate check {report['schedule_id']} vs "
                f"{report['duplicate_of']}: {verdict}"
            )

    combined = pd.concat(frames, ignore_index=True)
    return combined, duplicate_reports
