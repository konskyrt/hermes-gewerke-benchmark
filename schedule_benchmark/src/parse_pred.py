"""Structural parser for the three Roche pRED schedules.

pRED carries no trade column, but its leaf task names are a code:

    B07-OG01-B.05-BAU_Onboarding (Kick-Off, ...) Teppichboden
    B04-T.02.01a-SANITÄR-Schächte_Modellbearbeitung
    B07-OG05-T.09.02-EMSR_Hauptleitungen (inkl. Feldgeräte) E1

Reading right to left: an activity after the last underscore, a discipline word
before it, a dotted work-package code, optional floor tokens, and the building.
The package code doubles as a component group - `pathString` spells the codes
out ("137 B.05-Natursteinböden inkl. Unterlagsböden"), so the descriptions are
harvested from the schedule itself rather than hand-maintained here.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Dotted package codes (B.05, T.02.01a, T.09.02, P.01) and the rare dotless
# variant that appears as "-B09-". The dot is required for the dotted form,
# otherwise the "B07" building prefix matches.
PACKAGE_CODE = re.compile(r"\b([BTPE]\.\d{1,2}[a-z]?(?:\.\d{1,2}[a-z]?)*)\b")
PACKAGE_CODE_DOTLESS = re.compile(r"-([BTPE]\d{2}[a-z]?)-")

# "137 B.05-Natursteinböden inkl. Unterlagsböden" in a pathString segment. The
# dot is mandatory here too: without it "E3-..." in a task name would read as a
# package code and drag the whole name in as its description.
PACKAGE_IN_PATH = re.compile(
    r"^\d+\s+([BTPE]\.\d{1,2}[a-z]?(?:\.\d{1,2}[a-z]?)*)\s*-\s*(.+)$"
)
MAX_PACKAGE_DESCRIPTION = 90

BUILDING = re.compile(r"^(B\d{2})\b")

# A storey names a level; an area names a vertical or site region that spans
# levels. They land in different canonical fields.
STOREY_TOKEN = re.compile(r"^(UG\s*\d*|OG\s*\d*|EG|DG\s*\d*|Dach|Attika)$", re.I)
AREA_TOKEN = re.compile(
    r"^(Schächte|Schacht|Treppenhaus|Treppenhäuser|Fassade|Velokeller|Areal|"
    r"Aussen|Kern|Auditorium|Nord|Süd|Ost|West)$",
    re.I,
)

# The three installation stages pRED sequences its TGA/EMSR work in.
INSTALL_STAGE = re.compile(r"\bE([123])\b(?!\s*\w)")

# Activity vocabulary, longest phrase first so "WMP Prüflauf" wins over "WMP".
_ACTIVITY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"WMP\s*Prüflauf", re.I), "WMP Prüflauf", "Pruefung"),
    (re.compile(r"WMP\s*Erstellung", re.I), "WMP Erstellung", "Planung"),
    (re.compile(r"WMP\s*AVOR", re.I), "WMP AVOR", "Beschaffung"),
    (re.compile(r"DKS[\s-]*WMP", re.I), "DKS-WMP", "Planung"),
    (re.compile(r"\bAVOR\b", re.I), "AVOR", "Beschaffung"),
    (re.compile(r"Bestellung|Lieferung|Beschaffung", re.I), "Bestellung/Lieferung", "Lieferung"),
    (re.compile(r"Vorfertigung|Produktion", re.I), "Vorfertigung", "Vorfertigung"),
    (re.compile(r"Modell\s*bearbeitung", re.I), "Modellbearbeitung", "Planung"),
    (re.compile(r"Onboarding|Kick[\s-]*Off", re.I), "Onboarding", "Vorbereitung"),
    (re.compile(r"Bemusterung", re.I), "Bemusterung", "Vorbereitung"),
    (re.compile(r"Dichtheitsprüfung|Druckprüfung|Prüflauf|Prüfung|Test|Abnahme", re.I), "Prüfung", "Pruefung"),
    (re.compile(r"Inbetriebnahme|Einregulierung", re.I), "Inbetriebnahme", "Inbetriebnahme"),
    (re.compile(r"Rückbau|Abbruch|Demontage", re.I), "Rückbau", "Rueckbau"),
    (re.compile(r"Reinigung", re.I), "Reinigung", "Reinigung"),
    (re.compile(r"Feinmontage|Montage|montieren|Einbau|einbauen|verlegen|stellen|"
                r"Installation|Verkabelung|Anschluss|schliessen|spicken|Dämmung|"
                r"Leitungen|Halterungen|Anstrich|Erstellung", re.I), "Montage", "Montage"),
    (re.compile(r"Freigabe", re.I), "Freigabe", "Koordination"),
]

# pathString level 2 is the discipline branch.
_BRANCH_TO_DISCIPLINE = {
    "bau": "BAU",
    "tga": "TGA",
    "emsr": "EMSR",
    "labor": "LABOR",
    "labor (compound stores)": "LABOR",
    "elektro | ga": "ELEKTRO",
    "baustelleneinrichtung | logistik": "LOGISTIK",
    "sondernutzung": "SONDERNUTZUNG",
    "freigabeprozess (dks-wmp) meilensteine": "FREIGABE",
    "meilensteine": "MEILENSTEIN",
    "vertragstermine baumeister": "MEILENSTEIN",
    "ausbau geschosse_start | mechanical complete": "MEILENSTEIN",
}

_PATH_PREFIX = re.compile(r"^\d+\s+")


def _strip_id(segment: str) -> str:
    return _PATH_PREFIX.sub("", segment).strip()


def build_package_dictionary(paths: pd.Series) -> dict[str, str]:
    """code -> human description, harvested from every pathString segment.

    The longest description wins, because the same code shows up both as
    "T.02.01a-SANITÄR- WMP" and as "T.02.01a-Sanitär WMP".
    """
    dictionary: dict[str, str] = {}
    for value in paths.dropna():
        for segment in str(value).split("//"):
            match = PACKAGE_IN_PATH.match(segment.strip())
            if not match:
                continue
            code, description = match.group(1), match.group(2).strip()
            description = re.sub(r"\s+WMP$", "", description).strip(" -")
            description = description.replace("&amp;", "&")
            if not description or len(description) > MAX_PACKAGE_DESCRIPTION:
                continue
            existing = dictionary.get(code)
            if existing is None or len(description) > len(existing):
                dictionary[code] = description
    return dictionary


def _package_from_path(value: Any) -> str | None:
    """Deepest package code named anywhere in a row's pathString."""
    if pd.isna(value):
        return None
    found = None
    for segment in str(value).split("//"):
        match = PACKAGE_IN_PATH.match(segment.strip())
        if match:
            found = match.group(1)
    return found


def _activity(text: str) -> tuple[str | None, str | None]:
    for pattern, label, group in _ACTIVITY_RULES:
        if pattern.search(text):
            return label, group
    return None, None


def _split_name(name: str) -> dict[str, Any]:
    """Pull building, floor tokens, package code and discipline out of a name."""
    result: dict[str, Any] = {
        "building": None,
        "floor_token": None,
        "area_token": None,
        "floor_spans_multiple": False,
        "package_code": None,
        "discipline": None,
    }

    head, _, tail = name.partition("_")
    # Everything before the first underscore is the location/package code; if
    # there is no underscore the whole name is treated as the head.
    segment = head if tail else name

    match = BUILDING.match(segment)
    if match:
        result["building"] = match.group(1)
        segment = segment[match.end():].lstrip("-")

    code_match = PACKAGE_CODE.search(segment) or PACKAGE_CODE_DOTLESS.search(
        f"-{segment}-"
    )
    if code_match:
        result["package_code"] = code_match.group(1)
        before = segment[: code_match.start(1)].strip("- ")
        after = segment[code_match.end(1):].strip("- ")
        _assign_location(result, before.split("-"))
        if after:
            # "SANITÄR-Schächte" or "HEIZUNG-KÄLTE": the discipline is the
            # leading run of upper-case words.
            words = after.split("-")
            discipline_words = []
            for word in words:
                if word.upper() == word and any(c.isalpha() for c in word):
                    discipline_words.append(word)
                else:
                    break
            result["discipline"] = (
                "-".join(discipline_words) if discipline_words else words[0]
            )
    else:
        _assign_location(result, segment.split("-"))

    return result


def _assign_location(result: dict[str, Any], parts: list[str]) -> None:
    """Sort the tokens ahead of the package code into storeys and areas."""
    storeys = [p.strip() for p in parts if STOREY_TOKEN.match(p.strip())]
    areas = [p.strip() for p in parts if AREA_TOKEN.match(p.strip())]
    if storeys:
        result["floor_token"] = storeys[0]
        result["floor_spans_multiple"] = len(storeys) > 1
    if areas:
        result["area_token"] = "-".join(areas)


def parse(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pRED-derived columns to the normalized rows of one pRED schedule."""
    out = frame.copy()
    packages = build_package_dictionary(out["full_path"])

    branch_l1: list[str | None] = []
    branch_l2: list[str | None] = []
    for value in out["full_path"]:
        parts = [
            _strip_id(p)
            for p in (str(value).split("//") if pd.notna(value) else [])
            if p.strip()
        ]
        branch_l1.append(parts[1] if len(parts) > 1 else None)
        branch_l2.append(parts[2] if len(parts) > 2 else None)

    parsed = [_split_name(str(n)) if pd.notna(n) else {} for n in out["task_name"]]

    # Most EMSR and TGA rows carry no code in their own name, but their
    # pathString does: "...//2242 T.09.02-Doppelboden inkl. Teppich//...".
    path_codes = [_package_from_path(value) for value in out["full_path"]]

    out["pred_branch"] = branch_l1
    out["package_code"] = [
        p.get("package_code") or path_code
        for p, path_code in zip(parsed, path_codes)
    ]
    out["component_group"] = [
        packages.get(code) if code else None for code in out["package_code"]
    ]

    name_discipline = [
        (p.get("discipline") or "").strip() or None for p in parsed
    ]
    branch_discipline = [
        _BRANCH_TO_DISCIPLINE.get(str(b).lower()) if b else None for b in branch_l2
    ]
    # The name is more specific ("SANITÄR" beats "TGA"); the branch is the
    # fallback for rows whose name carries no code.
    disciplines = [
        (name or branch) for name, branch in zip(name_discipline, branch_discipline)
    ]
    out["pred_discipline"] = disciplines
    out["pred_discipline_branch"] = branch_discipline

    # "BAU", "TGA" and "EMSR" name a department, not a trade. For those rows the
    # work-package description is what the gewerk rules can actually act on.
    generic = {"BAU", "TGA", "EMSR", "LABOR", "GU", "GP"}
    out["gewerk_raw"] = [
        f"{discipline}: {component}"
        if discipline in generic and component and pd.notna(component)
        else discipline
        for discipline, component in zip(disciplines, out["component_group"])
    ]

    activities = [_activity(str(n)) if pd.notna(n) else (None, None) for n in out["task_name"]]
    out["activity_raw"] = [a[0] for a in activities]
    out["activity_code"] = [a[1] for a in activities]
    out["activity_source"] = ["parse" if a[0] else None for a in activities]

    def _stage(name: Any) -> str | None:
        if pd.isna(name):
            return None
        match = INSTALL_STAGE.search(str(name))
        return f"E{match.group(1)}" if match else None

    out["install_stage"] = [_stage(n) for n in out["task_name"]]

    # Anything with an installation stage is site work regardless of which
    # branch of the WBS it hangs under.
    is_execution = out["install_stage"].notna() | out["pred_branch"].astype(
        "string"
    ).str.contains("Ausführung", case=False, na=False)
    out["work_stream"] = pd.Series(
        ["Ausfuehrung" if flag else "Beschaffung" for flag in is_execution],
        index=out.index,
        dtype="string",
    )

    floor_tokens = pd.Series(
        [p.get("floor_token") for p in parsed], index=out.index, dtype="string"
    )
    area_tokens = pd.Series(
        [p.get("area_token") for p in parsed], index=out.index, dtype="string"
    )
    out["floor_raw"] = out["floor_raw"].astype("string").fillna(floor_tokens)
    out["zone"] = out["zone"].astype("string").fillna(area_tokens)
    # A parent named "Obergeschoss 03" carries the floor for rows whose own name
    # has no location token.
    parent_floor = out["parent_name"].astype("string")
    out["floor_raw"] = out["floor_raw"].fillna(
        parent_floor.where(
            parent_floor.str.match(r"^(Ober|Unter|Erd|Dach)geschoss", case=False, na=False)
        )
    )
    out["pred_floor_spans_multiple"] = [
        bool(p.get("floor_spans_multiple")) for p in parsed
    ]
    # pRED's WBS groups by procurement package, not by phase: its site work
    # hangs under "Beschaffung | AVOR" alongside the lead times it belongs to.
    # Phase has to come from the task itself, never from its ancestors.
    out["phase_ancestors_reliable"] = False

    out["gewerk_source"] = out["gewerk_raw"].notna().map({True: "parse", False: None})
    out["component_source"] = out["component_group"].notna().map({True: "parse", False: None})
    return out
