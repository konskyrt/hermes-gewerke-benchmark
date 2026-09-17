"""Assign the classification axes: gewerk, phase, task group, component, montage.

Deterministic throughout. Every assignment records which mechanism produced it
in a `*_source` column - `column`, `parse`, `rule`, `cache` or `llm` - so the
coverage report can say how much of the answer is actually evidence.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Any

import pandas as pd

from .config import TAXONOMY_PATH, load_yaml

GEWERK_UNRESOLVED = None
FALLBACK_GEWERK = "OTH"


# ------------------------------------------------------------------ helpers


def normalize_text(value: Any) -> str:
    """Lower case, collapse whitespace, drop the umlaut diacritics."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def _compile(rules: list[dict[str, Any]], key: str) -> list[tuple[re.Pattern[str], str]]:
    """Compile a rule list into (pattern, value) pairs, umlauts folded away."""
    compiled = []
    for rule in rules:
        pattern = normalize_text(rule["match"])
        # The rule files spell alternatives as "(ä|ae)"; after folding both
        # branches read "ae", which is harmless but noisy, so collapse them.
        pattern = pattern.replace("(ae|ae)", "ae").replace("(oe|oe)", "oe")
        pattern = pattern.replace("(ue|ue)", "ue").replace("(ss|ss)", "ss")
        compiled.append((re.compile(pattern), rule[key]))
    return compiled


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, str]:
    with open(TAXONOMY_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def gewerk_names() -> dict[str, str]:
    return {code: name for code, name in load_taxonomy().items() if "." not in code}


# ------------------------------------------------------------------- gewerk


class GewerkClassifier:
    def __init__(self) -> None:
        cfg = load_yaml("gewerk_map.yaml")
        self.unresolved = {normalize_text(v) for v in cfg.get("unresolved", [])}
        self.exact = {normalize_text(k): v for k, v in cfg.get("exact", {}).items()}
        self.contractor = {
            normalize_text(k): v for k, v in cfg.get("contractor", {}).items()
        }
        self.keywords = _compile(cfg.get("keywords", []), "gewerk")
        self.weak_keywords = _compile(cfg.get("weak_keywords", []), "gewerk")
        self.bkp = {str(k): v for k, v in cfg.get("bkp", {}).items()}
        self.valid = set(gewerk_names())

    @staticmethod
    def _first_hit(rules: list[tuple[re.Pattern[str], str]], text: str) -> str | None:
        if not text:
            return None
        for pattern, code in rules:
            if pattern.search(text):
                return code
        return None

    def _by_keyword(self, text: str) -> str | None:
        return self._first_hit(self.keywords, text)

    def _by_bkp(self, value: Any) -> str | None:
        text = str(value).strip() if value is not None and value is not pd.NA else ""
        digits = re.match(r"^(\d{3})", text)
        if not digits:
            return None
        return self.bkp.get(digits.group(1))

    def classify_row(
        self,
        gewerk_raw: Any,
        task_name: Any,
        bkp: Any,
        contractor: Any,
        ancestors: Any,
    ) -> tuple[str | None, str | None]:
        """Returns (gewerk code, source) or (None, None) when nothing matched."""
        raw = normalize_text(gewerk_raw)
        has_trade_column = bool(raw) and raw not in self.unresolved

        if has_trade_column:
            if raw in self.exact:
                return self.exact[raw], "column"
            hit = self._by_keyword(raw)
            if hit:
                return hit, "column"

        firm = normalize_text(contractor)
        if firm in self.contractor:
            return self.contractor[firm], "column"
        if raw in self.contractor:
            return self.contractor[raw], "column"

        name_text = normalize_text(task_name)
        hit = self._by_keyword(name_text)
        if hit:
            return hit, "rule"

        # Inherit from the nearest ancestor that names a trade. A task called
        # "Etage 3" under "Elektro Ausbau" is electrical work.
        ancestor_names = (
            [normalize_text(n) for n in reversed(list(ancestors))]
            if isinstance(ancestors, (list, tuple))
            else []
        )
        for text in ancestor_names:
            hit = self._by_keyword(text)
            if hit:
                return hit, "rule"

        hit = self._by_bkp(bkp)
        if hit:
            return hit, "rule"

        # Nothing specific anywhere: settle for the phase or department word.
        if has_trade_column:
            hit = self._first_hit(self.weak_keywords, raw)
            if hit:
                return hit, "column"
        for text in [name_text, *ancestor_names]:
            hit = self._first_hit(self.weak_keywords, text)
            if hit:
                return hit, "rule"

        return GEWERK_UNRESOLVED, None


# -------------------------------------------------------------------- phase


class PhaseClassifier:
    def __init__(self) -> None:
        cfg = load_yaml("phase_rules.yaml")
        self.phases = cfg["phases"]
        self.explicit = {normalize_text(k): v for k, v in cfg.get("explicit", {}).items()}
        self.strong = _compile(cfg.get("strong", []), "phase")
        self.weak = _compile(cfg.get("weak", []), "phase")
        self.gewerk_default = cfg.get("gewerk_default", {})

    @staticmethod
    def _first_hit(rules: list[tuple[re.Pattern[str], str]], text: str) -> str | None:
        if not text:
            return None
        for pattern, code in rules:
            if pattern.search(text):
                return code
        return None

    def _in_ancestors(
        self, rules: list[tuple[re.Pattern[str], str]], ancestors: Any
    ) -> str | None:
        if not isinstance(ancestors, (list, tuple)):
            return None
        for name in reversed(list(ancestors)):
            hit = self._first_hit(rules, normalize_text(name))
            if hit:
                return hit
        return None

    def classify_row(
        self,
        phase_raw: Any,
        task_name: Any,
        ancestors: Any,
        gewerk: str | None,
        work_stream: Any = None,
        ancestors_reliable: bool = True,
    ) -> tuple[str, str]:
        """`ancestors_reliable` is False where the WBS groups by procurement
        package rather than by phase, as pRED's "Beschaffung | AVOR" branch does
        for its site work as well as its lead times."""
        raw = normalize_text(phase_raw)
        if raw and raw in self.explicit:
            return self.explicit[raw], "column"

        # The leaf is the more specific statement, so it is read first.
        hit = self._first_hit(self.strong, normalize_text(task_name))
        if hit:
            return hit, "rule"

        stream = normalize_text(work_stream)
        if stream and stream in self.explicit:
            return self.explicit[stream], "parse"

        if ancestors_reliable:
            hit = self._in_ancestors(self.strong, ancestors)
            if hit:
                return hit, "rule"

        if gewerk and gewerk in self.gewerk_default:
            return self.gewerk_default[gewerk], "rule"

        hit = self._first_hit(self.weak, normalize_text(task_name))
        if hit:
            return hit, "rule"
        if ancestors_reliable:
            hit = self._in_ancestors(self.weak, ancestors)
            if hit:
                return hit, "rule"

        return "OTHER", "fallback"


# --------------------------------------------------------------- task group


class ActivityClassifier:
    def __init__(self) -> None:
        cfg = load_yaml("activity_rules.yaml")
        self.groups = cfg["groups"]
        self.keywords = _compile(cfg.get("keywords", []), "group")

    def _by_keyword(self, text: str) -> str | None:
        if not text:
            return None
        for pattern, code in self.keywords:
            if pattern.search(text):
                return code
        return None

    def classify_row(
        self, task_name: Any, ancestors: Any, is_milestone: bool
    ) -> tuple[str | None, str | None]:
        if is_milestone:
            return "Meilenstein", "rule"
        hit = self._by_keyword(normalize_text(task_name))
        if hit:
            return hit, "rule"
        if isinstance(ancestors, (list, tuple)) and ancestors:
            hit = self._by_keyword(normalize_text(ancestors[-1]))
            if hit:
                return hit, "rule"
        return None, None


# ---------------------------------------------------------------- component


class ComponentClassifier:
    def __init__(self) -> None:
        cfg = load_yaml("component_map.yaml")
        self.any_rules = _compile(cfg.get("any", []), "code")
        self.by_gewerk = {
            gewerk: _compile(rules, "code")
            for gewerk, rules in cfg.get("by_gewerk", {}).items()
        }
        self.taxonomy = load_taxonomy()

    def classify_row(
        self, component_group: Any, task_name: Any, gewerk: str | None
    ) -> tuple[str | None, str | None, str | None]:
        """Returns (component code, component name, source)."""
        texts = [normalize_text(component_group), normalize_text(task_name)]
        sources = ["parse", "rule"]

        for text, source in zip(texts, sources):
            if not text:
                continue
            for pattern, code in self.any_rules:
                if pattern.search(text):
                    return code, self.taxonomy.get(code), source

        rules = self.by_gewerk.get(gewerk or "", [])
        for text, source in zip(texts, sources):
            if not text:
                continue
            for pattern, code in rules:
                if pattern.search(text):
                    return code, self.taxonomy.get(code), source

        if gewerk:
            fallback = f"{gewerk}.99"
            if fallback in self.taxonomy:
                return fallback, self.taxonomy[fallback], "fallback"
        return None, None, None


# ----------------------------------------------------------------- montage


class MontageClassifier:
    """Grobmontage / Feinmontage / Endmontage within a gewerk."""

    # Activities that are site installation and therefore eligible for a stage.
    PRODUCTIVE = {
        "Montage",
        "Vorfertigung",
        "Rueckbau",
        "Reinigung",
        "Lieferung",
    }

    def __init__(self) -> None:
        cfg = load_yaml("montage_rules.yaml")
        self.stages = cfg["stages"]
        self.explicit = {
            normalize_text(k): v for k, v in cfg.get("explicit", {}).items()
        }
        self.keywords = _compile(cfg.get("keywords", []), "stage")

    def _by_keyword(self, text: str) -> str | None:
        if not text:
            return None
        for pattern, code in self.keywords:
            if pattern.search(text):
                return code
        return None

    def classify_row(
        self,
        task_name: Any,
        ancestors: Any,
        install_stage: Any,
        activity_code: Any,
        phase_code: Any = None,
    ) -> tuple[str | None, str | None]:
        stage_raw = normalize_text(install_stage)
        if stage_raw and stage_raw in self.explicit:
            return self.explicit[stage_raw], "parse"

        hit = self._by_keyword(normalize_text(task_name))
        if hit:
            return hit, "rule"

        if isinstance(ancestors, (list, tuple)):
            for name in reversed(list(ancestors)):
                hit = self._by_keyword(normalize_text(name))
                if hit:
                    return hit, "rule"

        # Installation work with no stage cue: infer from the construction phase.
        # Rohbau and earthworks are rough-in; envelope and fit-out are final fix;
        # TGA installation is the distribution step in between.
        if activity_code in self.PRODUCTIVE:
            phase_default = {
                "EARTH": "GROB",
                "SHELL": "GROB",
                "DEMO": "GROB",
                "MEP": "FEIN",
                "ENVELOPE": "END",
                "FITOUT": "END",
                "COMM": "END",
                "HANDOVER": "END",
            }
            if phase_code in phase_default:
                return phase_default[phase_code], "rule"

        return None, None


# ----------------------------------------------------------------- pipeline


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply all five axes. Parser-supplied values are respected, not overwritten."""
    out = frame.copy()
    for column in [
        "gewerk_source",
        "activity_code",
        "activity_source",
        "component_group",
        "component_source",
        "work_stream",
        "install_stage",
    ]:
        if column not in out.columns:
            out[column] = pd.NA
    if "phase_ancestors_reliable" not in out.columns:
        out["phase_ancestors_reliable"] = True
    out["phase_ancestors_reliable"] = out["phase_ancestors_reliable"].fillna(True)

    gewerk_clf = GewerkClassifier()
    phase_clf = PhaseClassifier()
    activity_clf = ActivityClassifier()
    component_clf = ComponentClassifier()
    montage_clf = MontageClassifier()

    gewerks, gewerk_sources = [], []
    for raw, name, bkp, contractor, ancestors in zip(
        out["gewerk_raw"], out["task_name"], out["bkp"], out["contractor"], out["ancestors"]
    ):
        code, source = gewerk_clf.classify_row(raw, name, bkp, contractor, ancestors)
        gewerks.append(code)
        gewerk_sources.append(source)
    out["gewerk_code"] = gewerks
    out["gewerk_source"] = gewerk_sources
    out["gewerk_name"] = [gewerk_names().get(code) if code else None for code in gewerks]

    phases, phase_sources = [], []
    for raw, name, ancestors, gewerk, stream, reliable in zip(
        out["phase_raw"],
        out["task_name"],
        out["ancestors"],
        out["gewerk_code"],
        out["work_stream"],
        out["phase_ancestors_reliable"],
    ):
        code, source = phase_clf.classify_row(
            raw, name, ancestors, gewerk, stream, bool(reliable)
        )
        phases.append(code)
        phase_sources.append(source)
    out["phase_code"] = phases
    out["phase_source"] = phase_sources
    out["phase_name"] = [phase_clf.phases[code]["name"] for code in phases]
    out["phase_order"] = [phase_clf.phases[code]["order"] for code in phases]

    activities, activity_sources = [], []
    for existing, existing_source, name, ancestors, milestone in zip(
        out["activity_code"], out["activity_source"], out["task_name"], out["ancestors"], out["is_milestone"]
    ):
        if pd.notna(existing):
            activities.append(existing)
            activity_sources.append(existing_source if pd.notna(existing_source) else "parse")
            continue
        code, source = activity_clf.classify_row(name, ancestors, bool(milestone))
        activities.append(code)
        activity_sources.append(source)
    out["activity_code"] = activities
    out["activity_source"] = activity_sources
    out["activity_name"] = [
        activity_clf.groups[code]["name"] if code in activity_clf.groups else None
        for code in activities
    ]
    out["activity_productive"] = [
        activity_clf.groups[code]["productive"] if code in activity_clf.groups else None
        for code in activities
    ]

    codes, names, sources = [], [], []
    for group, name, gewerk in zip(out["component_group"], out["task_name"], out["gewerk_code"]):
        code, label, source = component_clf.classify_row(group, name, gewerk)
        codes.append(code)
        names.append(label)
        sources.append(source)
    out["component_code"] = codes
    out["component_name"] = names
    out["component_source"] = sources

    montage_codes, montage_sources = [], []
    for name, ancestors, install_stage, activity, phase in zip(
        out["task_name"],
        out["ancestors"],
        out["install_stage"],
        out["activity_code"],
        out["phase_code"],
    ):
        code, source = montage_clf.classify_row(
            name, ancestors, install_stage, activity, phase
        )
        montage_codes.append(code)
        montage_sources.append(source)
    out["montage_code"] = montage_codes
    out["montage_source"] = montage_sources
    out["montage_name"] = [
        montage_clf.stages[code]["name"] if code in montage_clf.stages else None
        for code in montage_codes
    ]
    out["montage_order"] = [
        montage_clf.stages[code]["order"] if code in montage_clf.stages else None
        for code in montage_codes
    ]

    return out


def apply_fallback_gewerk(frame: pd.DataFrame) -> pd.DataFrame:
    """Last resort: everything still unresolved becomes OTH.

    Run after the LLM stage so the coverage report can distinguish "we know it
    is Sonstige" from "we never found out".
    """
    out = frame.copy()
    missing = out["gewerk_code"].isna()
    out.loc[missing, "gewerk_code"] = FALLBACK_GEWERK
    out.loc[missing, "gewerk_source"] = "fallback"
    out["gewerk_name"] = out["gewerk_code"].map(gewerk_names())
    return out
