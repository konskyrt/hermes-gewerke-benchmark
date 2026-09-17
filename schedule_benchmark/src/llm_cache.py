"""Signature cache and Groq fallback for the gewerk axis.

Tasks are deduplicated by a hash of their name, the same signature Hermes
already uses, so two prior classification runs seed this cache for free:

- USB / NBK2, 394 signatures covering 3,209 tasks
- ELP V6, 1,039 signatures with gewerk and component classes

The rule table always wins over a cached label. Disagreements are not silently
dropped - they are written to `out/rule_vs_llm.csv`, which is the only free
accuracy check available on `gewerk_map.yaml`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .classify import gewerk_names
from .config import REPO_ROOT, ensure_out_dir

CACHE_PATH = "llm_cache.json"

# Prior Hermes runs whose signatures transfer directly.
WARM_START_SOURCES = [
    {
        "label": "usb_gewerk_run",
        "path": REPO_ROOT
        / "backend/data/projects/test/processed"
        / "schedule_gewerk_session_1780583885270.json",
        "shape": "list",
    },
    {
        "label": "elp_v6_gewerk_run",
        "path": REPO_ROOT
        / "backend/data/schedule_classifications/ELP_new_schedule_220426.converted/output"
        / "ELP_new_schedule_220426.converted_20260429_122917_gewerks.json",
        "shape": "list",
    },
]

# qwen3.6 emits a reasoning trace before its answer. At 30 items per call the
# JSON gets truncated by the token limit and Groq rejects the whole response,
# so batches are kept small and split once more on failure.
BATCH_SIZE = 12
MIN_BATCH_SIZE = 3
MAX_TOKENS = 4096
INTER_BATCH_SLEEP = 2.0


def signature(task_name: Any) -> str:
    """Hash of a task name, byte-compatible with the Hermes schedule pipeline."""
    text = "" if task_name is None or pd.isna(task_name) else str(task_name).strip()
    attributes = {"task_name_raw": text}
    payload = json.dumps(sorted(attributes.items()), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain(value: Any) -> str | None:
    """JSON-safe scalar. pandas NA and numpy scalars are not serializable."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return str(value)


def load_cache() -> dict[str, dict[str, Any]]:
    path = ensure_out_dir() / CACHE_PATH
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        print(f"  cache at {path} is unreadable ({error}); starting a fresh one")
        return {}


def save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """Write via a temp file so an interrupted run cannot corrupt the cache."""
    path = ensure_out_dir() / CACHE_PATH
    temp = path.with_suffix(".json.tmp")
    payload = {
        key: {field: _plain(value) for field, value in entry.items()}
        for key, entry in cache.items()
    }
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1, sort_keys=True)
    temp.replace(path)


def warm_start(cache: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Fold the prior runs into the cache. Returns per-source counts added."""
    valid = set(gewerk_names())
    added: dict[str, int] = {}

    for source in WARM_START_SOURCES:
        path = Path(source["path"])
        if not path.is_file():
            added[source["label"]] = 0
            continue

        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        if isinstance(records, dict):
            records = records.get("results", [])

        count = 0
        for record in records:
            gewerk = record.get("gewerk")
            if gewerk not in valid:
                continue
            if record.get("validation_status") not in (None, "passed"):
                continue
            key = record.get("signature") or signature(record.get("concatenated_text"))
            if key in cache:
                continue
            cache[key] = {
                "gewerk": gewerk,
                "task_name": _plain(record.get("concatenated_text")),
                "reasoning": _plain(record.get("reasoning")),
                "origin": source["label"],
            }
            count += 1
        added[source["label"]] = count

    return added


def audit_rules(frame: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare every rule-assigned gewerk against a cached LLM label.

    The rule wins - this only reports where the two disagree, so the rule table
    can be corrected deliberately rather than drifting.
    """
    rows = []
    resolved = frame[frame["gewerk_code"].notna()]
    for name, mine, source, schedule in zip(
        resolved["task_name"],
        resolved["gewerk_code"],
        resolved["gewerk_source"],
        resolved["schedule_id"],
    ):
        entry = cache.get(signature(name))
        if not entry or entry["gewerk"] == mine:
            continue
        rows.append(
            {
                "task_name": name,
                "schedule_id": schedule,
                "rule_gewerk": mine,
                "rule_source": source,
                "cached_gewerk": entry["gewerk"],
                "cached_origin": entry.get("origin"),
                "cached_reasoning": entry.get("reasoning"),
            }
        )

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit
    return (
        audit.groupby(
            ["task_name", "rule_gewerk", "cached_gewerk", "rule_source", "cached_origin"],
            dropna=False,
        )
        .agg(tasks=("schedule_id", "size"), schedules=("schedule_id", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
        .sort_values("tasks", ascending=False)
    )


def apply_cache(frame: pd.DataFrame, cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Fill unresolved gewerks from the cache."""
    out = frame.copy()
    missing = out["gewerk_code"].isna()
    if not missing.any():
        return out

    filled_codes, filled_sources = [], []
    for name, code, source in zip(out["task_name"], out["gewerk_code"], out["gewerk_source"]):
        if pd.notna(code):
            filled_codes.append(code)
            filled_sources.append(source)
            continue
        entry = cache.get(signature(name))
        if entry:
            filled_codes.append(entry["gewerk"])
            filled_sources.append("cache" if entry.get("origin") else "llm")
        else:
            filled_codes.append(None)
            filled_sources.append(None)

    out["gewerk_code"] = filled_codes
    out["gewerk_source"] = filled_sources
    out["gewerk_name"] = out["gewerk_code"].map(gewerk_names())
    return out


def pending_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    """Distinct task names still without a gewerk, largest first."""
    # A task with no name carries nothing for the model to read.
    missing = frame[frame["gewerk_code"].isna() & frame["task_name"].notna()].copy()
    if missing.empty:
        return missing.assign(signature=[], tasks=[])
    missing["signature"] = missing["task_name"].map(signature)
    return (
        missing.groupby("signature")
        .agg(
            task_name=("task_name", "first"),
            tasks=("uid", "size"),
            schedules=("schedule_id", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values("tasks", ascending=False)
    )


# ---------------------------------------------------------------- Groq call

SYSTEM_PROMPT = """Du bist Experte für Schweizer/Deutsche Bauterminpläne und ordnest \
Vorgangsnamen einem Gewerk zu.

GEWERKE:
KO=Konstruktion (Rohbau, Beton, Mauerwerk, Stahlbau, Fundament, Decke, Treppe)
ENV=Gebäudehülle (Fassade, Dach, Fenster, Abdichtung, Sonnenschutz)
INT=Ausbau (Gipser, Maler, Boden, Türen, Trennwände, Decken, Schreiner)
SN=Sanitär (Sanitärrohre, Apparate, Entwässerung)
HZ=Heizung (Wärmeerzeuger, Heizrohre, Heizkörper)
KT=Kälte (Kältemaschinen, Kaltwasser, Kühldecken)
LF=Lüftung (Lüftungsgeräte, Luftkanäle, Klima, RLT)
EL=Elektro (Kabel, Trassen, Beleuchtung, Verteilungen, Blitzschutz)
SPR=Sprinkler (Löschanlagen, Sprinklerrohre, Gaslöschanlagen)
AUTO=Automation (MSR, Gebäudeautomation, DDC, Leittechnik)
SEC=Sicherheit (Brandmelde, Zutritt, Video, Einbruch, Schliessanlagen)
GAS=Gas (Gasleitungen, Medizinalgase)
SPEC=Spezialmedien (Druckluft, Vakuum, Reinstwasser, Prozessmedien)
TRANS=Transport (Aufzüge, Fahrtreppen, Rohrpost, Krane)
EQUIP=Ausstattung (Mobiliar, Küchen, Labor, Medizintechnik, IT)
SITE=Umgebung (Erdbau, Baugrube, Aussenanlagen, Werkleitungen)
OTH=Sonstige (Planung, Bauleitung, Meilensteine, Reinigung, Gerüst, mehrere Gewerke)

REGELN:
- Ein Vorgang ohne klaren Gewerkbezug (Meilenstein, Koordination, Planung) ist OTH.
- Ein Vorgang, der mehrere Gewerke gleichzeitig umfasst, ist OTH.
- Entscheide nach dem Bauteil, an dem gearbeitet wird, nicht nach der Tätigkeit.

Antworte NUR mit JSON: {"results":[{"id":0,"gewerk":"EL","reasoning":"kurz"}]}"""


async def _call_groq(client: Any, model: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    listing = "\n".join(f"{item['id']}: {item['task_name']}" for item in batch)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ordne jedem Vorgang ein Gewerk zu:\n{listing}"},
        ],
        temperature=0.1,
        top_p=0.8,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return payload.get("results", [])


async def _classify_batch(
    client: Any, model: str, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One call, halving the batch and retrying if the response was truncated."""
    try:
        return await _call_groq(client, model, batch)
    except Exception as error:  # noqa: BLE001 - split and retry rather than lose the batch
        if len(batch) <= MIN_BATCH_SIZE:
            print(f"    dropped {len(batch)} signatures: {error}")
            return []
        middle = len(batch) // 2
        first = await _classify_batch(client, model, batch[:middle])
        await asyncio.sleep(INTER_BATCH_SLEEP)
        second = await _classify_batch(client, model, batch[middle:])
        return first + second


async def _run_llm(
    pending: pd.DataFrame,
    model: str,
    api_key: str,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=api_key)
    valid = set(gewerk_names())
    items = [
        {"id": index, "task_name": row.task_name, "signature": row.signature}
        for index, row in enumerate(pending.itertuples())
    ]

    results: dict[str, dict[str, Any]] = {}
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        answers = await _classify_batch(client, model, batch)

        by_id = {item["id"]: item for item in batch}
        for answer in answers:
            item = by_id.get(answer.get("id"))
            gewerk = answer.get("gewerk")
            if not item or gewerk not in valid:
                continue
            results[item["signature"]] = {
                "gewerk": gewerk,
                "task_name": _plain(item["task_name"]),
                "reasoning": _plain(answer.get("reasoning")),
                "origin": None,
            }
        print(
            f"  batch {start // BATCH_SIZE + 1}/{(len(items) - 1) // BATCH_SIZE + 1}: "
            f"{len(results)} resolved"
        )
        # Checkpoint after every batch: this run takes minutes and its results
        # are the only thing in the pipeline that costs money to reproduce.
        if cache is not None:
            cache.update(results)
            save_cache(cache)
        if start + BATCH_SIZE < len(items):
            await asyncio.sleep(INTER_BATCH_SLEEP)

    return results


def run_llm_fallback(
    pending: pd.DataFrame, cache: dict[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    """Classify the remaining signatures with Groq, if a key is configured.

    Returns an empty mapping when no key is available, so the pipeline stays
    runnable offline; those rows just fall through to OTH.
    """
    if pending.empty:
        return {}

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / "backend" / ".env")
    except ImportError:
        pass

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  GROQ_API_KEY not set - skipping LLM fallback")
        return {}

    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b")
    print(f"  classifying {len(pending)} signatures with {model}")
    return asyncio.run(_run_llm(pending, model, api_key, cache))
