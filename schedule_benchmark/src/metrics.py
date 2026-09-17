"""Time and productivity metrics per gewerk, phase, task group and component.

Three duration measures are reported side by side, because they answer
different questions and disagree by a lot:

- `task_days`    sum of task durations. Double-counts parallel work, so it
                 reads as effort rather than time.
- `span_days`    first start to last finish. What the trade occupied on the
                 programme, including its idle gaps.
- `active_days`  calendar days on which at least one of the trade's tasks was
                 running. Span minus the gaps - the honest "how long were they
                 actually on site".

Only leaf, non-milestone tasks are counted, so a summary bar is never added to
the children it summarizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Axes the per-schedule and cross-schedule tables are cut along.
AXES = {
    "gewerk": ("gewerk_code", "gewerk_name"),
    "phase": ("phase_code", "phase_name"),
    "activity": ("activity_code", "activity_name"),
    "component": ("component_code", "component_name"),
    "montage": ("montage_code", "montage_name"),
    # Same three stages as "montage", but including the stages inferred from
    # the task-name clusters. Use this one when coverage matters more than
    # keeping to direct evidence.
    "stage": ("stage_code", "stage_name"),
    "cluster": ("cluster_id", "cluster_label"),
}


def executable(frame: pd.DataFrame) -> pd.DataFrame:
    """Leaf tasks that represent real work: no containers, no milestones."""
    return frame[frame["is_executable"] & frame["duration_days"].notna()].copy()


def active_days(starts: pd.Series, finishes: pd.Series) -> float:
    """Calendar days covered by the union of [start, finish] intervals."""
    pairs = [
        (s, f)
        for s, f in zip(starts, finishes)
        if pd.notna(s) and pd.notna(f) and f >= s
    ]
    if not pairs:
        return float("nan")
    pairs.sort()
    total = 0
    current_start, current_end = pairs[0]
    for start, end in pairs[1:]:
        if start <= current_end + pd.Timedelta(days=1):
            current_end = max(current_end, end)
        else:
            total += (current_end - current_start).days + 1
            current_start, current_end = start, end
    total += (current_end - current_start).days + 1
    return float(total)


def peak_concurrency(starts: pd.Series, finishes: pd.Series) -> float:
    """Most tasks of this group running on any single day."""
    events: list[tuple[pd.Timestamp, int]] = []
    for start, finish in zip(starts, finishes):
        if pd.isna(start) or pd.isna(finish) or finish < start:
            continue
        events.append((start, 1))
        events.append((finish + pd.Timedelta(days=1), -1))
    if not events:
        return float("nan")
    events.sort()
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return float(peak)


def _project_window(frame: pd.DataFrame) -> tuple[pd.Timestamp, float]:
    starts = frame["start"].dropna()
    finishes = frame["finish"].dropna()
    if starts.empty or finishes.empty:
        return pd.NaT, float("nan")
    origin = starts.min()
    span = (finishes.max() - origin).days
    return origin, float(span) if span > 0 else float("nan")


def _group_metrics(
    group: pd.DataFrame, origin: pd.Timestamp, project_span: float
) -> dict[str, float]:
    durations = group["duration_days"].dropna()
    starts, finishes = group["start"], group["finish"]

    span = float("nan")
    if starts.notna().any() and finishes.notna().any():
        span = float((finishes.max() - starts.min()).days + 1)

    # Where in the programme this group sits, 0 at project start and 1 at
    # project finish. This is what makes trade sequencing comparable between
    # projects of different length.
    start_position = finish_position = float("nan")
    if pd.notna(origin) and not np.isnan(project_span) and project_span > 0:
        if starts.notna().any():
            start_position = (starts.min() - origin).days / project_span
        if finishes.notna().any():
            finish_position = (finishes.max() - origin).days / project_span

    crew = group["crew_size"]
    person_days = float("nan")
    if crew.notna().any():
        person_days = float((group["duration_days"] * crew).sum(skipna=True))

    return {
        "tasks": int(len(group)),
        "task_days": float(durations.sum()),
        "median_task_days": float(durations.median()) if not durations.empty else float("nan"),
        "mean_task_days": float(durations.mean()) if not durations.empty else float("nan"),
        "p90_task_days": float(durations.quantile(0.9)) if not durations.empty else float("nan"),
        "max_task_days": float(durations.max()) if not durations.empty else float("nan"),
        "span_days": span,
        "active_days": active_days(starts, finishes),
        "peak_concurrency": peak_concurrency(starts, finishes),
        "first_start": starts.min(),
        "last_finish": finishes.max(),
        "start_position": start_position,
        "finish_position": finish_position,
        "critical_tasks": int(group["is_critical"].fillna(False).sum()),
        "critical_share": float(group["is_critical"].fillna(False).mean()),
        "person_days": person_days,
        "floors": int(group["floor_norm"].nunique()),
        "zones": int(group["zone"].nunique()),
        "contractors": int(group["contractor"].nunique()),
    }


def by_axis(frame: pd.DataFrame, axis: str, per_schedule: bool = True) -> pd.DataFrame:
    """Metric table for one classification axis."""
    code_column, name_column = AXES[axis]
    work = executable(frame)
    if work.empty:
        return pd.DataFrame()

    keys = ["schedule_id", "project", "schedule_label"] if per_schedule else []
    rows = []
    scope = work.groupby("schedule_id", sort=False) if per_schedule else [(None, work)]

    for _, subset in scope:
        origin, project_span = _project_window(subset)
        for value, group in subset.groupby(code_column, dropna=False, sort=False):
            if pd.isna(value):
                continue
            row = {key: group[key].iloc[0] for key in keys}
            row[axis] = value
            row[f"{axis}_name"] = group[name_column].iloc[0]
            row.update(_group_metrics(group, origin, project_span))
            row["project_span_days"] = project_span
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Share of the schedule's total task-days, so trades stay comparable across
    # schedules of very different size.
    if per_schedule:
        totals = table.groupby("schedule_id")["task_days"].transform("sum")
    else:
        totals = table["task_days"].sum()
    table["task_days_share"] = table["task_days"] / totals
    return table.sort_values("task_days", ascending=False).reset_index(drop=True)


def gewerk_stage(frame: pd.DataFrame, per_schedule: bool = True) -> pd.DataFrame:
    """Drill each gewerk into Grobmontage / Feinmontage / Endmontage.

    Same shape as `gewerk_montage`, but reads `stage_code`, so the stages the
    task-name clusters filled in are included. That is the difference between
    covering ~62% and ~79% of work tasks. `inferred_task_days` records how much
    of each bar rests on that inference.

    Only tasks with a stage are counted, so the segments do not add up to the
    gewerk's full task days - the rest is planning, procurement and testing.
    """
    work = executable(frame)
    work = work[work["stage_code"].notna() & work["gewerk_code"].notna()]
    if work.empty:
        return pd.DataFrame()

    keys = ["schedule_id", "project", "schedule_label"] if per_schedule else []
    rows = []
    scope = work.groupby("schedule_id", sort=False) if per_schedule else [(None, work)]

    for _, subset in scope:
        origin, project_span = _project_window(subset)
        for key, group in subset.groupby(
            ["gewerk_code", "gewerk_name", "stage_code", "stage_name"],
            dropna=False,
            sort=False,
        ):
            row = {column: group[column].iloc[0] for column in keys}
            row["gewerk"] = key[0]
            row["gewerk_name"] = key[1]
            row["stage"] = key[2]
            row["stage_name"] = key[3]
            row.update(_group_metrics(group, origin, project_span))
            row["inferred_task_days"] = float(
                group.loc[group["stage_source"] == "cluster", "duration_days"].sum()
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if per_schedule:
        totals = table.groupby(["schedule_id", "gewerk"])["task_days"].transform("sum")
    else:
        totals = table.groupby("gewerk")["task_days"].transform("sum")
    table["stage_share_of_gewerk"] = table["task_days"] / totals.replace(0, pd.NA)
    table["stage_order"] = table["stage"].map({"GROB": 10, "FEIN": 20, "END": 30})
    sort_keys = ["schedule_id", "gewerk", "stage_order"] if per_schedule else [
        "gewerk", "stage_order"
    ]
    return table.sort_values(sort_keys).reset_index(drop=True)


def gewerk_phase(frame: pd.DataFrame, per_schedule: bool = True) -> pd.DataFrame:
    """Drill each gewerk into the construction phases it works in.

    Every work task carries both a gewerk and a phase, so unlike the montage
    drill these bars do add up to the gewerk's full task days.
    """
    work = executable(frame)
    work = work[work["gewerk_code"].notna() & work["phase_code"].notna()]
    if work.empty:
        return pd.DataFrame()

    keys = ["schedule_id", "project", "schedule_label"] if per_schedule else []
    rows = []
    scope = work.groupby("schedule_id", sort=False) if per_schedule else [(None, work)]

    for _, subset in scope:
        origin, project_span = _project_window(subset)
        for key, group in subset.groupby(
            ["gewerk_code", "gewerk_name", "phase_code", "phase_name"],
            dropna=False,
            sort=False,
        ):
            row = {column: group[column].iloc[0] for column in keys}
            row["gewerk"] = key[0]
            row["gewerk_name"] = key[1]
            row["phase"] = key[2]
            row["phase_name"] = key[3]
            row["phase_order"] = group["phase_order"].iloc[0]
            row.update(_group_metrics(group, origin, project_span))
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if per_schedule:
        gewerk_totals = table.groupby(["schedule_id", "gewerk"])["task_days"].transform("sum")
        phase_totals = table.groupby(["schedule_id", "phase"])["task_days"].transform("sum")
    else:
        gewerk_totals = table.groupby("gewerk")["task_days"].transform("sum")
        phase_totals = table.groupby("phase")["task_days"].transform("sum")
    table["phase_share_of_gewerk"] = table["task_days"] / gewerk_totals.replace(0, pd.NA)
    table["gewerk_share_of_phase"] = table["task_days"] / phase_totals.replace(0, pd.NA)
    sort_keys = ["schedule_id", "gewerk", "phase_order"] if per_schedule else [
        "gewerk", "phase_order"
    ]
    return table.sort_values(sort_keys).reset_index(drop=True)


def cluster_stage(frame: pd.DataFrame, per_schedule: bool = True) -> pd.DataFrame:
    """Task-name clusters drilled into Grobmontage / Feinmontage / Endmontage.

    Uses `stage_code`, so a task staged by cluster inference counts here. The
    `inferred_task_days` column says how much of each bar rests on inference
    rather than on a parsed field or a keyword rule.
    """
    work = executable(frame)
    work = work[work["stage_code"].notna() & work["cluster_id"].notna()]
    if work.empty:
        return pd.DataFrame()

    keys = ["schedule_id", "project", "schedule_label"] if per_schedule else []
    rows = []
    scope = work.groupby("schedule_id", sort=False) if per_schedule else [(None, work)]

    for _, subset in scope:
        origin, project_span = _project_window(subset)
        for key, group in subset.groupby(
            ["cluster_id", "cluster_label", "stage_code", "stage_name"],
            dropna=False,
            sort=False,
        ):
            row = {column: group[column].iloc[0] for column in keys}
            row["cluster_id"] = int(key[0])
            row["cluster_label"] = key[1]
            row["stage"] = key[2]
            row["stage_name"] = key[3]
            row.update(_group_metrics(group, origin, project_span))
            inferred = group[group["stage_source"] == "cluster"]
            row["inferred_task_days"] = float(inferred["duration_days"].sum())
            row["top_gewerk"] = (
                group["gewerk_name"].mode().iloc[0]
                if not group["gewerk_name"].mode().empty
                else pd.NA
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if per_schedule:
        totals = table.groupby(["schedule_id", "cluster_id"])["task_days"].transform("sum")
    else:
        totals = table.groupby("cluster_id")["task_days"].transform("sum")
    table["stage_share_of_cluster"] = table["task_days"] / totals.replace(0, pd.NA)
    table["inferred_share"] = table["inferred_task_days"] / table["task_days"].replace(0, pd.NA)
    table["stage_order"] = table["stage"].map({"GROB": 10, "FEIN": 20, "END": 30})
    sort_keys = ["schedule_id", "cluster_id", "stage_order"] if per_schedule else [
        "cluster_id", "stage_order"
    ]
    return table.sort_values(sort_keys).reset_index(drop=True)


def task_cluster_export(frame: pd.DataFrame) -> pd.DataFrame:
    """One flat row per task: its cluster, gewerk, phase and montage stage.

    This is the single cross-project file - every schedule in one table, so a
    cluster can be followed from one programme to the next.
    """
    columns = [
        "project", "schedule_id", "schedule_label", "task_id", "task_name",
        "cluster_id", "cluster_label",
        "gewerk_code", "gewerk_name", "gewerk_source",
        "phase_code", "phase_name",
        "activity_code", "activity_name",
        "montage_code", "montage_name", "montage_source",
        "stage_code", "stage_name", "stage_source",
        "floor_norm", "zone", "start", "finish", "duration_days",
        "is_executable",
    ]
    out = frame[[c for c in columns if c in frame]].copy()
    return out.sort_values(
        ["cluster_id", "project", "schedule_id"], na_position="last"
    ).reset_index(drop=True)


def cycle_times(frame: pd.DataFrame) -> pd.DataFrame:
    """Takt: how long one gewerk needs for one floor-zone, repeated.

    Only groups with at least three comparable occurrences are reported, and
    tasks whose floor label spans several storeys are dropped, since they are
    not one cycle.
    """
    work = executable(frame)
    work = work[work["floor_norm"].notna() & ~work["floor_spans_multiple"].fillna(False)]
    if work.empty:
        return pd.DataFrame()

    work = work.copy()
    work["cell"] = (
        work["floor_norm"].astype(str) + "|" + work["zone"].fillna("-").astype(str)
    )

    rows = []
    for (schedule, gewerk), group in work.groupby(["schedule_id", "gewerk_code"], sort=False):
        per_cell = group.groupby("cell").agg(
            cell_days=("duration_days", "sum"),
            cell_tasks=("duration_days", "size"),
            cell_start=("start", "min"),
            cell_finish=("finish", "max"),
        )
        if len(per_cell) < 3:
            continue
        # Rhythm: the typical gap between one cell starting and the next.
        starts = per_cell["cell_start"].dropna().sort_values()
        takt_gap = float(starts.diff().dt.days.median()) if len(starts) > 2 else float("nan")
        rows.append(
            {
                "schedule_id": schedule,
                "schedule_label": group["schedule_label"].iloc[0],
                "project": group["project"].iloc[0],
                "gewerk": gewerk,
                "gewerk_name": group["gewerk_name"].iloc[0],
                "cells": int(len(per_cell)),
                "median_cell_days": float(per_cell["cell_days"].median()),
                "mean_cell_days": float(per_cell["cell_days"].mean()),
                "p10_cell_days": float(per_cell["cell_days"].quantile(0.1)),
                "p90_cell_days": float(per_cell["cell_days"].quantile(0.9)),
                "median_tasks_per_cell": float(per_cell["cell_tasks"].median()),
                "takt_gap_days": takt_gap,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # How repeatable the trade is: 1.0 means every cell took the same time.
    table["consistency"] = 1 - (
        (table["p90_cell_days"] - table["p10_cell_days"]).abs()
        / table["median_cell_days"].replace(0, np.nan)
    ).clip(0, 1)
    return table.sort_values(["schedule_id", "median_cell_days"], ascending=[True, False])


def schedule_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per schedule: the headline numbers for the comparison tab."""
    rows = []
    for schedule_id, group in frame.groupby("schedule_id", sort=False):
        work = executable(group)
        origin, span = _project_window(group)
        rows.append(
            {
                "schedule_id": schedule_id,
                "schedule_label": group["schedule_label"].iloc[0],
                "project": group["project"].iloc[0],
                "source_file": group["source_file"].iloc[0],
                "tasks": int(len(group)),
                "work_tasks": int(len(work)),
                "milestones": int(group["is_milestone"].sum()),
                "summaries": int(group["is_summary"].sum()),
                "first_start": group["start"].min(),
                "last_finish": group["finish"].max(),
                "project_span_days": span,
                "project_span_months": span / 30.44 if pd.notna(span) else float("nan"),
                "task_days": float(work["duration_days"].sum()),
                "median_task_days": float(work["duration_days"].median()) if not work.empty else float("nan"),
                "gewerke": int(work["gewerk_code"].nunique()),
                "floors": int(group["floor_norm"].nunique()),
                "zones": int(group["zone"].nunique()),
                "buildings": int(group["building"].nunique()),
                "contractors": int(group["contractor"].nunique()),
                "critical_share": float(work["is_critical"].fillna(False).mean()) if not work.empty else float("nan"),
                "has_crew_size": bool(group["crew_size"].notna().any()),
                "has_materials": bool(group.get("material_count", pd.Series(dtype=float)).fillna(0).gt(0).any())
                if "material_count" in group
                else False,
            }
        )
    return pd.DataFrame(rows)


def gewerk_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Gewerk x schedule share of task-days: the core cross-project comparison."""
    table = by_axis(frame, "gewerk", per_schedule=True)
    if table.empty:
        return table
    matrix = table.pivot_table(
        index=["gewerk", "gewerk_name"],
        columns="schedule_id",
        values="task_days_share",
        aggfunc="sum",
    )
    return (matrix * 100).round(1)


def sequence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalized start position of each gewerk in each schedule.

    Two projects of very different length can be laid over each other on this
    0-1 axis, which is how trade sequencing becomes comparable at all.
    """
    table = by_axis(frame, "gewerk", per_schedule=True)
    if table.empty:
        return table
    columns = [
        "schedule_id",
        "schedule_label",
        "gewerk",
        "gewerk_name",
        "start_position",
        "finish_position",
        "task_days_share",
        "span_days",
        "active_days",
    ]
    return table[columns].sort_values(["schedule_id", "start_position"])


def gewerk_montage(frame: pd.DataFrame, per_schedule: bool = True) -> pd.DataFrame:
    """Drill each gewerk into Grobmontage / Feinmontage / Endmontage.

    Only tasks that carry a montage stage are counted, so a gewerk's three
    stages will not add up to its total task days - the gap is work that is
    not installation (planning, testing, delivery, ...).
    """
    work = executable(frame)
    work = work[work["montage_code"].notna() & work["gewerk_code"].notna()]
    if work.empty:
        return pd.DataFrame()

    keys = ["schedule_id", "project", "schedule_label"] if per_schedule else []
    group_keys = keys + ["gewerk_code", "gewerk_name", "montage_code", "montage_name"]
    rows = []
    scope = work.groupby("schedule_id", sort=False) if per_schedule else [(None, work)]

    for _, subset in scope:
        origin, project_span = _project_window(subset)
        for key, group in subset.groupby(
            ["gewerk_code", "gewerk_name", "montage_code", "montage_name"],
            dropna=False,
            sort=False,
        ):
            row = {column: group[column].iloc[0] for column in keys}
            row["gewerk"] = key[0]
            row["gewerk_name"] = key[1]
            row["montage"] = key[2]
            row["montage_name"] = key[3]
            row.update(_group_metrics(group, origin, project_span))
            rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if per_schedule:
        totals = table.groupby(["schedule_id", "gewerk"])["task_days"].transform("sum")
    else:
        totals = table.groupby("gewerk")["task_days"].transform("sum")
    table["stage_share_of_gewerk"] = table["task_days"] / totals.replace(0, pd.NA)
    order = {"GROB": 10, "FEIN": 20, "END": 30}
    table["montage_order"] = table["montage"].map(order)
    return table.sort_values(
        ["schedule_id", "gewerk", "montage_order"] if per_schedule else ["gewerk", "montage_order"]
    ).reset_index(drop=True)


def pred_split(frame: pd.DataFrame) -> pd.DataFrame:
    """pRED only: procurement lead time against site production, per gewerk.

    Kept separate because averaging the two together is exactly the mistake
    this schedule invites - most of its rows are waiting, not building.
    """
    work = executable(frame[frame["schedule_id"].str.startswith("pred_")])
    if work.empty or "work_stream" not in work:
        return pd.DataFrame()
    table = (
        work.groupby(["schedule_id", "gewerk_code", "gewerk_name", "work_stream"], dropna=False)
        .agg(
            tasks=("uid", "size"),
            task_days=("duration_days", "sum"),
            median_task_days=("duration_days", "median"),
        )
        .reset_index()
    )
    return table.sort_values(["schedule_id", "task_days"], ascending=[True, False])


def dc30_materials(frame: pd.DataFrame) -> pd.DataFrame:
    """DC30 only: component throughput by material and unit.

    DC30 is the one schedule that says what each task consumes, so it is the
    only place a duration can be tied to a quantity unit.
    """
    subset = frame[frame["schedule_id"] == "dc30"]
    if subset.empty or "material_items" not in subset:
        return pd.DataFrame()

    rows = []
    for items, units, gewerk, gewerk_name, days, floor, zone in zip(
        subset["material_items"],
        subset["material_units"],
        subset["gewerk_code"],
        subset["gewerk_name"],
        subset["duration_days"],
        subset["floor_norm"],
        subset["zone"],
    ):
        if not isinstance(items, (list, tuple)):
            continue
        for index, item in enumerate(items):
            rows.append(
                {
                    "material": item,
                    "unit": units[index] if isinstance(units, (list, tuple)) and index < len(units) else None,
                    "gewerk": gewerk,
                    "gewerk_name": gewerk_name,
                    "duration_days": days,
                    "floor_norm": floor,
                    "zone": zone,
                }
            )
    if not rows:
        return pd.DataFrame()

    exploded = pd.DataFrame(rows)
    return (
        exploded.groupby(["gewerk", "gewerk_name", "material", "unit"], dropna=False)
        .agg(
            tasks=("duration_days", "size"),
            task_days=("duration_days", "sum"),
            median_task_days=("duration_days", "median"),
        )
        .reset_index()
        .sort_values("task_days", ascending=False)
    )
