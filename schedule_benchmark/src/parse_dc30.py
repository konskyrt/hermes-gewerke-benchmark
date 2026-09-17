"""Structural parser for the DC30 data-center schedule.

DC30 is the richest source in the set. Three things are unique to it and worth
pulling apart properly:

- `Gewerk` is a numbered contractor package, "12 / Kälte" - the number is the
  package, the text is the trade.
- `Full_Path` is a fixed hierarchy, "Bauphase > M31 > Ausbau > EG0 > BA1 > Aktivität".
- `Inbound_Material` lists the components a task consumes with their units,
  "Kabeltrassen (Stück), Schaltwartenboden (m²)". No other schedule has this.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

PACKAGE = re.compile(r"^\s*(\d+)\s*/\s*(.+?)\s*$")
MATERIAL_ITEM = re.compile(r"^\s*(.+?)\s*\(([^)]+)\)\s*$")

# Keywords that describe where or when, not what - excluded when a keyword has
# to stand in for a component name.
_NON_COMPONENT_KEYWORDS = re.compile(
    r"^(\d+\.(OG|UG)|EG|DG\d*|OG\d*|UG\d*|Flur|Nebenräume|Montage|Installation|"
    r"Lieferung|Einbau|Ausführung|Verkabelung|Anschluss|Vorbereitung|Reinigung|"
    r"Prüfung|Abnahme|Feininstallation|Rohinstallation|innen|außen|aussen)$",
    re.I,
)

_ACTIVITY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"Reinigung|Baureinigung", re.I), "Reinigung", "Reinigung"),
    (re.compile(r"Prüfung|Abnahme|Messung|Inspektion|Test", re.I), "Prüfung", "Pruefung"),
    (re.compile(r"Inbetriebnahme|Einregulierung|IBN", re.I), "Inbetriebnahme", "Inbetriebnahme"),
    (re.compile(r"Rückbau|Demontage|Abbruch", re.I), "Rückbau", "Rueckbau"),
    (re.compile(r"Lieferung|Anlieferung|Lift\s*&\s*Shif", re.I), "Lieferung", "Lieferung"),
    (re.compile(r"Vorbereitung|Vorarbeiten|AVOR", re.I), "Vorbereitung", "Vorbereitung"),
    (re.compile(r"Isolierung|Dämmung", re.I), "Isolierung", "Montage"),
    (re.compile(r"Verkabelung|Anschluss|Anschlüsse", re.I), "Verkabelung", "Montage"),
    (re.compile(r"Montage|Ausführung|Einbau|Verlegung|Installation|Setzen", re.I), "Montage", "Montage"),
]


def parse_materials(value: Any) -> list[tuple[str, str | None]]:
    """"Kabeltrassen (Stück), Schaltwartenboden (m²)" -> [(item, unit), ...]"""
    if pd.isna(value):
        return []
    items: list[tuple[str, str | None]] = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = MATERIAL_ITEM.match(chunk)
        if match:
            unit = match.group(2).strip()
            items.append((match.group(1).strip(), _UNIT_ALIASES.get(unit.lower(), unit)))
        else:
            items.append((chunk, None))
    return items


# Description_EN is machine-written English and uses a small, regular verb set.
_ACTIVITY_RULES_EN: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bclean", re.I), "Reinigung", "Reinigung"),
    (re.compile(r"\b(test|inspect|commission\w*\s+check|verify|measure)", re.I), "Prüfung", "Pruefung"),
    (re.compile(r"\bcommission", re.I), "Inbetriebnahme", "Inbetriebnahme"),
    (re.compile(r"\b(dismantl|demolish|remov)", re.I), "Rückbau", "Rueckbau"),
    (re.compile(r"\b(deliver|supply|transport)", re.I), "Lieferung", "Lieferung"),
    (re.compile(r"\b(prepare|preparation)", re.I), "Vorbereitung", "Vorbereitung"),
    (re.compile(r"\b(insulat)", re.I), "Isolierung", "Montage"),
    (re.compile(r"\b(wire|cabl|connect|terminat)", re.I), "Verkabelung", "Montage"),
    (re.compile(r"\b(install|mount|erect|lay|apply|fit|assembl|paint|plaster|route)", re.I), "Montage", "Montage"),
]

# The same unit written two ways in the same column.
_UNIT_ALIASES = {"stk": "Stück", "stck": "Stück", "st": "Stück", "pcs": "Stück"}


def _activity(text: str) -> tuple[str | None, str | None]:
    for pattern, label, group in _ACTIVITY_RULES:
        if pattern.search(text):
            return label, group
    return None, None


def _activity_en(text: str) -> tuple[str | None, str | None]:
    for pattern, label, group in _ACTIVITY_RULES_EN:
        if pattern.search(text):
            return label, group
    return None, None


def _component_group(materials: list[tuple[str, str | None]], keywords: Any) -> str | None:
    """The component a task acts on: its first material, else its first keyword."""
    if materials:
        return materials[0][0]
    if pd.isna(keywords):
        return None
    for token in str(keywords).split(","):
        token = token.strip()
        if token and not _NON_COMPONENT_KEYWORDS.match(token):
            return token
    return None


def parse(frame: pd.DataFrame) -> pd.DataFrame:
    """Add DC30-derived columns to the normalized rows of the DC30 schedule."""
    out = frame.copy()

    packages = out["gewerk_raw"].map(
        lambda v: PACKAGE.match(str(v)) if pd.notna(v) else None
    )
    out["package_code"] = [m.group(1) if m else None for m in packages]
    out["gewerk_raw"] = [
        m.group(2) if m else (v if pd.notna(v) else None)
        for m, v in zip(packages, out["gewerk_raw"])
    ]

    # Bauphase > Site_Area > Ausbau|Ausbau TGA > Floor > Zone > Activity
    levels = out["full_path"].map(
        lambda v: [p.strip() for p in str(v).split(" > ")] if pd.notna(v) else []
    )
    out["dc30_bauphase"] = [parts[0] if len(parts) > 0 else None for parts in levels]
    out["dc30_work_type"] = [parts[2] if len(parts) > 2 else None for parts in levels]

    materials = out["material_hint"].map(parse_materials)
    out["material_items"] = materials.map(lambda items: [i[0] for i in items])
    out["material_units"] = materials.map(
        lambda items: [i[1] for i in items if i[1]]
    )
    out["material_count"] = materials.map(len)

    out["component_group"] = [
        _component_group(items, keywords)
        for items, keywords in zip(materials, out["keywords"])
    ]
    out["component_source"] = out["component_group"].notna().map(
        {True: "parse", False: None}
    )

    # Roughly half of DC30's task names are bare noun phrases ("Stromschiene",
    # "EMV"). Description_EN carries the verb for those, so it is the fallback.
    activities = [
        _activity(str(name)) if pd.notna(name) else (None, None)
        for name in out["task_name"]
    ]
    activities = [
        found
        if found[0]
        else (_activity_en(str(description)) if pd.notna(description) else (None, None))
        for found, description in zip(activities, out["description_en"])
    ]
    out["activity_raw"] = [a[0] for a in activities]
    out["activity_code"] = [a[1] for a in activities]
    out["activity_source"] = ["parse" if a[0] else None for a in activities]

    # Every DC30 row sits under "Bauphase", and the work type separates the
    # building fit-out from the technical fit-out.
    out["work_stream"] = "Ausfuehrung"
    out["gewerk_source"] = out["gewerk_raw"].notna().map({True: "column", False: None})
    return out
