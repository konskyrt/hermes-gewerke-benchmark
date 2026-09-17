"""Turn the ingested raw frame into the canonical task record.

Handles the four date dialects, three duration dialects, three hierarchy
dialects and the floor-label zoo found across the eleven schedules.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- booleans

_TRUE = {"yes", "ja", "y", "j", "true", "1", "1.0"}
_FALSE = {"no", "nein", "n", "false", "0", "0.0", "", "nan", "none", "<na>"}


def to_bool(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


# ------------------------------------------------------------------- dates

_GERMAN_MONTHS = {
    "januar": "01", "februar": "02", "märz": "03", "maerz": "03", "april": "04",
    "mai": "05", "juni": "06", "juli": "07", "august": "08", "september": "09",
    "oktober": "10", "november": "11", "dezember": "12",
}

_DE_DATE = re.compile(r"^(\d{1,2})\.?\s+([A-Za-zäöüÄÖÜ]+)\s+(\d{4})")
_MPP_DATE = re.compile(r"^(?:[A-Za-z]{3}\s+)?(\d{1,2})/(\d{1,2})/(\d{2,4})$")


def parse_date(value: Any) -> pd.Timestamp | None:
    """Parse the date formats in use, in cheapest-first order.

    - MS Project text export: "Thu 22/9/22" (day first, two-digit year)
    - ISO / Excel datetime:   "2026-03-16T08:00:00" or a real datetime
    - German long form:       "16 März 2022 08:00"
    - Plain ISO date:         "2021-03-12"
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else pd.Timestamp(value).normalize()
    if isinstance(value, float) and np.isnan(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "na"}:
        return None

    match = _MPP_DATE.match(text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        if year < 100:
            year += 2000
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None

    match = _DE_DATE.match(text)
    if match:
        day, month_name, year = match.groups()
        month = _GERMAN_MONTHS.get(month_name.strip().lower())
        if month:
            try:
                return pd.Timestamp(f"{year}-{month}-{int(day):02d}")
            except ValueError:
                return None

    parsed = pd.to_datetime(text, errors="coerce", format="ISO8601")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else pd.Timestamp(parsed).normalize()


# --------------------------------------------------------------- durations

_DURATION_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)?)")


def parse_duration(value: Any, unit: str) -> float | None:
    """Read a duration in working days.

    `mpp`     "1345 dys?" / "0 dys"
    `de_tage` "1312 Tage"
    `days_d`  "325 d"
    `days_int` a bare number
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if pd.isna(value) else float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na"}:
        return None
    match = _DURATION_NUMBER.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", "."))
    except ValueError:
        return None


# ------------------------------------------------------------------ floors

_FLOOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "Untergeschoss 02", "Obergeschoss 01", "Erdgeschoss", "Dachgeschoss"
    (re.compile(r"^unter\s*geschoss\s*(\d+)", re.I), "UG"),
    (re.compile(r"^ober\s*geschoss\s*(\d+)", re.I), "OG"),
    (re.compile(r"^dach\s*geschoss\s*(\d*)", re.I), "DG"),
    (re.compile(r"^erd\s*geschoss", re.I), "EG"),
    # "1.UG", "2.OG", "10. OG"
    (re.compile(r"^(\d+)\s*\.\s*ug\b", re.I), "UG"),
    (re.compile(r"^(\d+)\s*\.\s*og\b", re.I), "OG"),
    (re.compile(r"^(\d+)\s*\.\s*dg\b", re.I), "DG"),
    # "UG04", "OG14", "EG00", "DG1", "UG 2", and the bare "U01" form
    (re.compile(r"^ug?\s*(\d*)$", re.I), "UG"),
    (re.compile(r"^og?\s*(\d*)$", re.I), "OG"),
    (re.compile(r"^dg\s*(\d*)$", re.I), "DG"),
    (re.compile(r"^eg\s*\d*$", re.I), "EG"),
    # ELP "G0", "G01", "G2"; BäreTower / UH1 "E07"
    (re.compile(r"^g\s*0*$", re.I), "EG"),
    (re.compile(r"^g\s*0?(\d+)$", re.I), "OG"),
    (re.compile(r"^e\s*0*$", re.I), "EG"),
    (re.compile(r"^e\s*0?(\d+)$", re.I), "OG"),
]

_MEZZANINE = re.compile(r"^(g\s*0*z|zg|zwischengeschoss)", re.I)
_ROOF = re.compile(r"^(dach|attika|roof)", re.I)
_MULTI = re.compile(r"[+&,/]|\bbis\b|(?<=\d)\s*-\s*(?=[a-zA-Z]*\d)")
# "7.+8.OG", "9.+10.OG": the keyword trails the numbers, so the range has to be
# read before the string is split on its separator.
_MULTI_TRAILING_KEYWORD = re.compile(
    r"^(\d+)\s*\.?\s*[+&,/]\s*\d+\s*\.?\s*(og|ug|dg)\b", re.I
)
_LEADING_JUNK = re.compile(r"^[^0-9A-Za-zÄÖÜäöü]+")


def parse_floor(value: Any) -> tuple[str | None, float | None, bool]:
    """Return (canonical label, ordinal, spans_multiple_floors).

    Ordinal is signed storey number: UG04 is -4, EG00 is 0, OG14 is 14. A
    mezzanine sits at 0.5 and a roof level above the tallest tower in the set.
    """
    if value is None or value is pd.NA:
        return None, None, False
    text = _LEADING_JUNK.sub("", str(value).strip())
    if not text or text.lower() in {"nan", "none", "na", "-", "unknown", "alle", "all"}:
        return None, None, False

    spans_multiple = bool(_MULTI.search(text))

    match = _MULTI_TRAILING_KEYWORD.match(text)
    if match:
        number, kind = int(match.group(1)), match.group(2).upper()
        sign = -1 if kind == "UG" else 1
        offset = 90 if kind == "DG" else 0
        return f"{kind}{number:02d}", float(offset + sign * number), True

    if spans_multiple:
        # "OG02-OG03", "U01 / U02", "UG02-UG01": classify by the first token and
        # keep the flag so cycle-time metrics can exclude them.
        text = re.split(r"[+&,/]|\bbis\b|-", text)[0].strip()
        if not text:
            return None, None, True

    if _MEZZANINE.match(text):
        return "ZG", 0.5, spans_multiple
    if _ROOF.match(text):
        return "DA", 90.0, spans_multiple

    for pattern, kind in _FLOOR_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        digits = match.group(1) if match.groups() and match.group(1) else ""
        if kind == "EG":
            return "EG00", 0.0, spans_multiple
        if not digits:
            # A bare "OG"/"UG" names no storey, so it carries no ordinal.
            return None, None, spans_multiple
        number = int(digits)
        if kind == "UG":
            return f"UG{number:02d}", float(-number), spans_multiple
        if kind == "OG":
            return f"OG{number:02d}", float(number), spans_multiple
        if kind == "DG":
            return f"DG{number:02d}", 90.0 + number, spans_multiple

    return None, None, spans_multiple


# --------------------------------------------------------------- hierarchy


def _split_path(value: Any) -> list[str] | None:
    if value is None or value is pd.NA:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    separator = "//" if "//" in text else (" > " if " > " in text else None)
    if separator is None:
        return [text]
    return [part.strip() for part in text.split(separator) if part.strip()]


_PATH_PREFIX = re.compile(r"^\d+\s+")


def _clean_path_segment(segment: str) -> str:
    """Drop the leading task id pRED prefixes each path segment with."""
    return _PATH_PREFIX.sub("", segment).strip()


def _ancestors_from_wbs(frame: pd.DataFrame) -> pd.Series:
    """Ancestor name chain for schedules with a dotted WBS column."""
    wbs = frame["wbs"].astype("string").str.strip()
    name_by_wbs = {}
    for code, name in zip(wbs, frame["task_name"].astype("string")):
        if code and code is not pd.NA and code not in name_by_wbs:
            name_by_wbs[code] = name

    paths = []
    for code in wbs:
        if code is pd.NA or not code:
            paths.append([])
            continue
        parts = code.split(".")
        chain = []
        for depth in range(1, len(parts)):
            ancestor = ".".join(parts[:depth])
            name = name_by_wbs.get(ancestor)
            if name and name is not pd.NA:
                chain.append(str(name).strip())
        paths.append(chain)
    return pd.Series(paths, index=frame.index)


def _ancestors_from_parent_name(frame: pd.DataFrame, max_depth: int = 12) -> pd.Series:
    """Ancestor chain for schedules that only name their parent (UH1).

    Names are not guaranteed unique, so the first summary row carrying a name
    wins and self-references are cut.
    """
    names = frame["task_name"].astype("string").str.strip()
    parents = frame["parent_name"].astype("string").str.strip()
    parent_of_name: dict[str, str] = {}
    for name, parent in zip(names, parents):
        if name is pd.NA or not name or name in parent_of_name:
            continue
        if parent is not pd.NA and parent and parent != name:
            parent_of_name[name] = parent

    paths = []
    for parent in parents:
        chain: list[str] = []
        cursor = parent
        seen: set[str] = set()
        while (
            cursor is not pd.NA
            and cursor
            and cursor not in seen
            and len(chain) < max_depth
        ):
            seen.add(cursor)
            chain.append(str(cursor))
            cursor = parent_of_name.get(cursor, pd.NA)
        paths.append(list(reversed(chain)))
    return pd.Series(paths, index=frame.index)


def _hierarchy_for_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    """Ancestor path, depth and leaf flag for one schedule."""
    has_full_path = frame["full_path"].notna().any()
    has_wbs = frame["wbs"].notna().any()

    if has_full_path:
        raw_paths = frame["full_path"].map(_split_path)
        # The last segment of a full path is the task itself.
        ancestors = raw_paths.map(
            lambda parts: [_clean_path_segment(p) for p in parts[:-1]] if parts else []
        )
    elif has_wbs:
        ancestors = _ancestors_from_wbs(frame)
    else:
        ancestors = _ancestors_from_parent_name(frame)

    depth = ancestors.map(len) + 1
    declared_level = pd.to_numeric(frame["wbs_level"], errors="coerce")
    depth = declared_level.fillna(pd.Series(depth, index=frame.index)).astype(int)

    if has_wbs and not has_full_path:
        codes = frame["wbs"].astype("string").str.strip()
        known = {c for c in codes if c is not pd.NA and c}
        has_children = codes.map(
            lambda code: any(
                other.startswith(f"{code}.") for other in known
            )
            if code is not pd.NA and code
            else False
        )
    else:
        parent_names = {
            name
            for chain in ancestors
            for name in chain
        }
        has_children = frame["task_name"].astype("string").str.strip().isin(parent_names)

    return pd.DataFrame(
        {
            "ancestors": ancestors,
            "wbs_level": depth,
            "has_children": has_children.fillna(False).astype(bool),
        },
        index=frame.index,
    )


# ---------------------------------------------------------------- assembly


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw ingest frame -> canonical task record, one row per task."""
    out = raw.copy()

    out["start"] = out["start"].map(parse_date)
    out["finish"] = out["finish"].map(parse_date)
    out["duration_days"] = [
        parse_duration(value, unit)
        for value, unit in zip(out["duration"], out["duration_unit"])
    ]

    calendar = (out["finish"] - out["start"]).dt.days + 1
    out["calendar_days"] = calendar.where(calendar >= 0)
    # Prefer the schedule's own duration; fall back to the calendar span.
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    out["duration_days"] = out["duration_days"].fillna(out["calendar_days"])

    out["is_summary"] = out["is_summary_raw"].map(to_bool)
    out["is_milestone"] = out["is_milestone_raw"].map(to_bool)
    out["is_critical"] = out["is_critical_raw"].map(to_bool).fillna(False)

    category = out["task_category"].astype("string").str.upper().str.strip()
    out["is_summary"] = out["is_summary"].fillna(category.eq("SUMMARY TASK"))

    hierarchy = pd.concat(
        [
            _hierarchy_for_schedule(group)
            for _, group in out.groupby("schedule_id", sort=False)
        ]
    ).reindex(out.index)
    out["ancestors"] = hierarchy["ancestors"]
    out["wbs_level"] = hierarchy["wbs_level"]
    out["has_children"] = hierarchy["has_children"]

    out["is_summary"] = out["is_summary"].fillna(out["has_children"]).astype(bool)
    # A zero-length task that is not a container is a milestone, whatever the
    # source called it.
    out["is_milestone"] = (
        out["is_milestone"]
        .fillna((out["duration_days"].fillna(-1) == 0) & ~out["has_children"])
        .astype(bool)
    )
    out["is_leaf"] = ~out["has_children"]
    # A row the source itself calls a summary is a rollup bar even when nothing
    # in the file links a child to it - USB, Frankfurt Four and the pRED files
    # all carry blocks of contract-level bars whose detail sits in a separate
    # branch. Counting them as work double-counts entire project spans as
    # single tasks (USB's "Baustelleneinrichtung" alone reads 1'461 days).
    out["is_executable"] = out["is_leaf"] & ~out["is_milestone"] & ~out["is_summary"]

    floors = out["floor_raw"].map(parse_floor)
    out["floor_norm"] = [f[0] for f in floors]
    out["floor_ordinal"] = [f[1] for f in floors]
    out["floor_spans_multiple"] = [f[2] for f in floors]

    out["crew_size"] = pd.to_numeric(out["crew_size"], errors="coerce")
    out["pct_complete"] = pd.to_numeric(out["pct_complete"], errors="coerce")
    # Burckhardt stores completion as a 0-1 fraction, everyone else as 0-100.
    fraction = out["pct_complete"].le(1) & out["pct_complete"].gt(0)
    out.loc[fraction, "pct_complete"] = out.loc[fraction, "pct_complete"] * 100

    for field in ["zone", "room", "building", "contractor", "usage", "keywords"]:
        out[field] = out[field].astype("string").str.strip().replace({"": pd.NA, "-": pd.NA})

    out["task_name"] = out["task_name"].astype("string").str.strip()
    out["task_id"] = out["task_id"].astype("string").str.strip()
    out["uid"] = out["schedule_id"] + ":" + out["row_index"].astype(str)

    return out


def renormalize_floors(frame: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the floor fields after a parser has filled in `floor_raw`.

    pRED carries no floor column at all - its storeys come out of the task
    names, which the pRED parser only reads once normalization has run.
    """
    out = frame.copy()
    floors = out["floor_raw"].map(parse_floor)
    out["floor_norm"] = [f[0] for f in floors]
    out["floor_ordinal"] = [f[1] for f in floors]
    spans = pd.Series([f[2] for f in floors], index=out.index)
    if "pred_floor_spans_multiple" in out:
        spans = spans | out["pred_floor_spans_multiple"].fillna(False).astype(bool)
    out["floor_spans_multiple"] = spans
    return out


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-schedule sanity table, printed by build.py."""
    rows = []
    for schedule_id, group in frame.groupby("schedule_id", sort=False):
        rows.append(
            {
                "schedule_id": schedule_id,
                "tasks": len(group),
                "leaf": int(group["is_leaf"].sum()),
                "summary": int(group["is_summary"].sum()),
                "milestone": int(group["is_milestone"].sum()),
                "max_depth": int(group["wbs_level"].max()),
                "dated": int(group["start"].notna().sum()),
                "with_duration": int(group["duration_days"].notna().sum()),
                "with_floor": int(group["floor_norm"].notna().sum()),
                "start": group["start"].min(),
                "finish": group["finish"].max(),
            }
        )
    return pd.DataFrame(rows)
