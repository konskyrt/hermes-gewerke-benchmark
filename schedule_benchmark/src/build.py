"""Run the whole pipeline and write everything the dashboard reads.

    python -m schedule_benchmark.src.build            # rules + cache only
    python -m schedule_benchmark.src.build --llm      # also call Groq for the rest
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import classify, cluster, llm_cache, metrics, parse_dc30, parse_pred
from .config import ensure_out_dir
from .ingest import ingest_all
from .normalize import normalize, renormalize_floors, summarize

PARSERS = {"pred": parse_pred.parse, "dc30": parse_dc30.parse}

# Columns holding python lists, which parquet keeps but CSV would flatten to
# unparseable strings.
LIST_COLUMNS = ["ancestors", "material_items", "material_units"]

# Raw source values kept only until normalization has read them. They hold
# mixed types by nature ("23 dys?" next to 23) and parquet cannot type them.
RAW_COLUMNS = [
    "duration",
    "is_summary_raw",
    "is_milestone_raw",
    "is_critical_raw",
    "task_category",
    "duration_unit",
    "parser",
    "contractor_2",
    "predecessors",
]


def prepare_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw columns and give every remaining object column one type."""
    out = frame.drop(columns=[c for c in RAW_COLUMNS if c in frame], errors="ignore")
    for column in out.columns:
        if column in LIST_COLUMNS or out[column].dtype != object:
            continue
        out[column] = out[column].astype("string")
    return out


def apply_parsers(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("schedule_id", sort=False):
        parser_name = group["parser"].iloc[0]
        parser = PARSERS.get(parser_name) if pd.notna(parser_name) else None
        parts.append(parser(group) if parser else group)
    return pd.concat(parts, ignore_index=True)


def coverage_report(
    frame: pd.DataFrame,
    duplicate_reports: list[dict[str, Any]],
    warm_start_counts: dict[str, int],
    llm_resolved: int,
    disagreements: int,
) -> dict[str, Any]:
    """Per-schedule, per-axis resolution rates broken down by mechanism."""
    axes = {
        "gewerk": ("gewerk_code", "gewerk_source"),
        "phase": ("phase_code", "phase_source"),
        "activity": ("activity_code", "activity_source"),
        "component": ("component_code", "component_source"),
        "montage": ("montage_code", "montage_source"),
        "stage": ("stage_code", "stage_source"),
    }

    def block(subset: pd.DataFrame) -> dict[str, Any]:
        work = subset[subset["is_executable"]]
        out: dict[str, Any] = {}
        for axis, (code_column, source_column) in axes.items():
            sources = subset.loc[subset[code_column].notna(), source_column]
            out[axis] = {
                "resolved": float(subset[code_column].notna().mean()),
                "resolved_work_tasks": float(work[code_column].notna().mean())
                if not work.empty
                else None,
                "by_source": {
                    str(key): int(value) for key, value in sources.value_counts().items()
                },
            }
        out["fields"] = {
            "start": float(subset["start"].notna().mean()),
            "duration": float(subset["duration_days"].notna().mean()),
            "floor": float(subset["floor_norm"].notna().mean()),
            "zone": float(subset["zone"].notna().mean()),
            "crew_size": float(subset["crew_size"].notna().mean()),
            "contractor": float(subset["contractor"].notna().mean()),
            "critical_flag": float(subset["is_critical"].fillna(False).mean()),
        }
        out["tasks"] = int(len(subset))
        out["work_tasks"] = int(len(work))
        return out

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": block(frame),
        "by_schedule": {
            str(schedule_id): block(group)
            for schedule_id, group in frame.groupby("schedule_id", sort=False)
        },
        "duplicate_checks": duplicate_reports,
        "warm_start": warm_start_counts,
        "llm_signatures_resolved": llm_resolved,
        "rule_vs_cache_disagreements": disagreements,
    }


def build(use_llm: bool = False) -> pd.DataFrame:
    out_dir = ensure_out_dir()

    print("1/7 ingest")
    raw, duplicate_reports = ingest_all()

    print("2/7 normalize")
    frame = normalize(raw)

    print("3/7 structural parsers (pRED, DC30)")
    frame = renormalize_floors(apply_parsers(frame))
    print(summarize(frame).to_string(index=False))

    print("4/7 classify")
    frame = classify.classify(frame)

    print("5/7 signature cache")
    cache = llm_cache.load_cache()
    warm_start_counts = llm_cache.warm_start(cache)
    for label, count in warm_start_counts.items():
        print(f"  {label}: +{count} signatures")

    audit = llm_cache.audit_rules(frame, cache)
    if not audit.empty:
        audit.to_csv(out_dir / "rule_vs_llm.csv", index=False, encoding="utf-8-sig")
        print(f"  {len(audit)} rule/cache disagreements -> out/rule_vs_llm.csv")

    frame = llm_cache.apply_cache(frame, cache)
    pending = llm_cache.pending_signatures(frame)
    print(f"  {len(pending)} signatures still unresolved")

    llm_resolved = 0
    if use_llm and not pending.empty:
        results = llm_cache.run_llm_fallback(pending, cache)
        cache.update(results)
        llm_resolved = len(results)
        frame = llm_cache.apply_cache(frame, cache)
    save_cache_and_report(cache, pending, out_dir)

    frame = classify.apply_fallback_gewerk(frame)

    print("6/7 cluster task names")
    frame = cluster.assign_clusters(frame)
    frame, cluster_profile = cluster.apply_stage_inference(frame)
    work = frame[frame["is_executable"]]
    print(
        f"  montage stage: {work['montage_code'].notna().mean():.1%} from rules, "
        f"{work['stage_code'].notna().mean():.1%} once clusters fill the gaps"
    )

    print("7/7 metrics")
    write_outputs(frame, out_dir, cluster_profile)

    report = coverage_report(
        frame, duplicate_reports, warm_start_counts, llm_resolved, len(audit)
    )
    with open(out_dir / "coverage.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    overall = report["overall"]
    print()
    for axis in ["gewerk", "phase", "activity", "component", "montage", "stage"]:
        print(
            f"  {axis:<10} {overall[axis]['resolved']:.1%} of all tasks, "
            f"{overall[axis]['resolved_work_tasks']:.1%} of work tasks"
        )
    print(f"\nwrote {len(frame)} tasks to {out_dir}")
    return frame


def save_cache_and_report(
    cache: dict[str, dict[str, Any]], pending: pd.DataFrame, out_dir: Any
) -> None:
    llm_cache.save_cache(cache)
    if not pending.empty:
        pending.to_csv(out_dir / "pending_signatures.csv", index=False, encoding="utf-8-sig")


def write_outputs(
    frame: pd.DataFrame, out_dir: Any, cluster_profile: pd.DataFrame | None = None
) -> None:
    stored = prepare_for_storage(frame)
    stored.to_parquet(out_dir / "tasks_all.parquet", index=False)

    flat = stored.copy()
    for column in LIST_COLUMNS:
        if column in flat:
            flat[column] = flat[column].map(
                lambda value: " | ".join(str(v) for v in value)
                if isinstance(value, (list, tuple))
                else value
            )
    flat.to_csv(out_dir / "tasks_all.csv", index=False, encoding="utf-8-sig")

    tables = {
        "metrics_gewerk": metrics.by_axis(frame, "gewerk"),
        "metrics_phase": metrics.by_axis(frame, "phase"),
        "metrics_activity": metrics.by_axis(frame, "activity"),
        "metrics_component": metrics.by_axis(frame, "component"),
        "metrics_montage": metrics.by_axis(frame, "montage"),
        "metrics_stage": metrics.by_axis(frame, "stage"),
        "metrics_stage_all": metrics.by_axis(frame, "stage", per_schedule=False),
        "metrics_gewerk_stage": metrics.gewerk_stage(frame, per_schedule=True),
        "metrics_gewerk_stage_all": metrics.gewerk_stage(frame, per_schedule=False),
        "metrics_gewerk_phase": metrics.gewerk_phase(frame, per_schedule=True),
        "metrics_gewerk_phase_all": metrics.gewerk_phase(frame, per_schedule=False),
        "metrics_phase_all": metrics.by_axis(frame, "phase", per_schedule=False),
        "metrics_gewerk_montage": metrics.gewerk_montage(frame, per_schedule=True),
        "metrics_gewerk_montage_all": metrics.gewerk_montage(frame, per_schedule=False),
        "metrics_cluster": metrics.by_axis(frame, "cluster"),
        "metrics_cluster_all": metrics.by_axis(frame, "cluster", per_schedule=False),
        "metrics_cluster_stage": metrics.cluster_stage(frame, per_schedule=True),
        "metrics_cluster_stage_all": metrics.cluster_stage(frame, per_schedule=False),
        "cluster_profile": cluster_profile,
        "task_clusters": metrics.task_cluster_export(frame),
        "metrics_gewerk_all": metrics.by_axis(frame, "gewerk", per_schedule=False),
        "cycle_times": metrics.cycle_times(frame),
        "schedule_overview": metrics.schedule_overview(frame),
        "gewerk_matrix": metrics.gewerk_matrix(frame),
        "sequence": metrics.sequence_table(frame),
        "pred_split": metrics.pred_split(frame),
        "dc30_materials": metrics.dc30_materials(frame),
    }
    for name, table in tables.items():
        if table is None or table.empty:
            print(f"  {name}: empty, skipped")
            continue
        include_index = name == "gewerk_matrix"
        table.to_csv(out_dir / f"{name}.csv", index=include_index, encoding="utf-8-sig")
        print(f"  {name}: {len(table)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="call Groq for signatures the rules could not resolve",
    )
    args = parser.parse_args()
    build(use_llm=args.llm)


if __name__ == "__main__":
    main()
