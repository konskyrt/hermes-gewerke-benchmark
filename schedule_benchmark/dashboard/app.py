"""Cross-schedule benchmark dashboard.

    streamlit run schedule_benchmark/dashboard/app.py

Reads only what `python -m schedule_benchmark.src.build` wrote to `out/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PACKAGE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PACKAGE_DIR / "out"
sys.path.insert(0, str(PACKAGE_DIR.parent))

from schedule_benchmark.src import metrics  # noqa: E402
from schedule_benchmark.dashboard.dc30_analysis import render_dc30_difference  # noqa: E402

st.set_page_config(
    page_title="Hermes Gewerke-Benchmark", page_icon="", layout="wide"
)

GEWERK_COLORS = {
    "KO": "#8d6e63", "ENV": "#546e7a", "INT": "#ffb300", "SN": "#26a69a",
    "HZ": "#e53935", "KT": "#29b6f6", "LF": "#7e57c2", "EL": "#fdd835",
    "SPR": "#d81b60", "AUTO": "#43a047", "SEC": "#5c6bc0", "GAS": "#fb8c00",
    "SPEC": "#00897b", "TRANS": "#6d4c41", "EQUIP": "#ec407a", "SITE": "#9ccc65",
    "OTH": "#bdbdbd",
}


# --------------------------------------------------------------- data access


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


# ------------------------------------------------------------------ helpers


def days(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.0f} d".replace(",", "'")


def gewerk_bar(table: pd.DataFrame, value: str, title: str, label: str) -> go.Figure:
    # One flat colour on purpose: the gewerk is already named on the y-axis, so
    # a per-gewerk palette here would be colour with no legend to explain it.
    ordered = table.sort_values(value, ascending=True)
    figure = px.bar(
        ordered,
        x=value,
        y="gewerk_name",
        orientation="h",
        color_discrete_sequence=["#546e7a"],
        title=title,
        labels={value: label, "gewerk_name": ""},
        text=ordered[value].map(lambda v: f"{v:,.0f}".replace(",", "'")),
    )
    figure.update_layout(showlegend=False, height=max(320, 26 * len(table)))
    figure.update_traces(textposition="outside", cliponaxis=False)
    return figure


def gewerk_name_colors(table: pd.DataFrame) -> dict[str, str]:
    """Colour map keyed by gewerk name, so legends read "Elektro", not "EL"."""
    pairs = table.dropna(subset=["gewerk", "gewerk_name"]).drop_duplicates("gewerk")
    return {
        row.gewerk_name: GEWERK_COLORS.get(row.gewerk, "#bdbdbd")
        for row in pairs.itertuples()
    }


def duration_explainer() -> None:
    st.caption(
        "**Task days** sums every task's duration, so parallel work is counted twice "
        "over - read it as effort. **Span** is first start to last finish, including "
        "idle gaps. **Active days** counts only the calendar days on which the trade "
        "had at least one task running, which is the closest thing to time on site."
    )


def axis_table(table: pd.DataFrame, axis: str) -> pd.DataFrame:
    columns = {
        f"{axis}_name": axis.title(),
        "tasks": "Tasks",
        "task_days": "Task days",
        "task_days_share": "Share",
        "median_task_days": "Median task",
        "span_days": "Span",
        "active_days": "Active days",
        "peak_concurrency": "Peak parallel",
        "critical_share": "On crit. path",
        "first_start": "First start",
        "last_finish": "Last finish",
    }
    available = {k: v for k, v in columns.items() if k in table}
    view = table[list(available)].rename(columns=available)
    if "Share" in view:
        view["Share"] = (view["Share"] * 100).round(1)
    if "On crit. path" in view:
        view["On crit. path"] = (view["On crit. path"] * 100).round(0)
    for column in ["Task days", "Median task", "Span", "Active days", "Peak parallel"]:
        if column in view:
            view[column] = view[column].round(0)
    for column in ["First start", "Last finish"]:
        if column in view:
            view[column] = pd.to_datetime(view[column], errors="coerce").dt.date
    return view


# The only three colours in this section. Same three everywhere they appear.
STAGE_COLORS = {
    "Grobmontage": "#6d4c41",
    "Feinmontage": "#fb8c00",
    "Endmontage": "#43a047",
}
STAGE_ORDER = ["Grobmontage", "Feinmontage", "Endmontage"]


def gewerk_stage_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "gewerk_name": "Gewerk",
        "stage_name": "Montage stage",
        "tasks": "Tasks",
        "task_days": "Task days",
        "stage_share_of_gewerk": "Share of gewerk",
        "median_task_days": "Median task",
        "span_days": "Span",
        "active_days": "Active days",
        "first_start": "First start",
        "last_finish": "Last finish",
    }
    available = {k: v for k, v in columns.items() if k in table}
    view = table[list(available)].rename(columns=available)
    if "Share of gewerk" in view:
        view["Share of gewerk"] = (view["Share of gewerk"] * 100).round(1)
    for column in ["Task days", "Median task", "Span", "Active days"]:
        if column in view:
            view[column] = view[column].round(0)
    for column in ["First start", "Last finish"]:
        if column in view:
            view[column] = pd.to_datetime(view[column], errors="coerce").dt.date
    return view


def render_gewerk_stage_section(schedule_id: str | None) -> None:
    """Headline view: every gewerk as a bar, drilled into Grobmontage,
    Feinmontage and Endmontage."""
    if schedule_id is None:
        table = load_table("metrics_gewerk_stage_all")
        stages = load_table("metrics_stage_all")
        suffix = " (all schedules)"
    else:
        table = load_table("metrics_gewerk_stage")
        table = table[table["schedule_id"] == schedule_id] if not table.empty else table
        stages = load_table("metrics_stage")
        stages = stages[stages["schedule_id"] == schedule_id] if not stages.empty else stages
        suffix = ""

    st.subheader("Duration of each gewerk and montage stage")
    st.caption(
        "Every bar is one gewerk, split into the three montage stages: "
        "**Grobmontage** (rough-in, main runs), **Feinmontage** (distribution) "
        "and **Endmontage** (final fix). The colours mean the same thing in "
        "every chart on this page and are named in the legend on the right."
    )
    duration_explainer()
    if table.empty:
        st.info("No montage stages classified for this selection.")
        return

    gewerk_order = (
        table.groupby("gewerk_name")["task_days"].sum().sort_values(ascending=True).index.tolist()
    )
    orders = {"gewerk_name": gewerk_order, "stage_name": STAGE_ORDER}

    st.plotly_chart(
        px.bar(
            table,
            x="task_days",
            y="gewerk_name",
            color="stage_name",
            orientation="h",
            barmode="stack",
            category_orders=orders,
            color_discrete_map=STAGE_COLORS,
            hover_data=["tasks", "median_task_days", "active_days"],
            title=f"Chart of task days by gewerk, drilled into montage stages{suffix}",
            labels={
                "task_days": "Task days",
                "gewerk_name": "",
                "stage_name": "Montage stage",
            },
            height=max(420, 30 * len(gewerk_order)),
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            table,
            x="stage_share_of_gewerk",
            y="gewerk_name",
            color="stage_name",
            orientation="h",
            barmode="stack",
            category_orders=orders,
            color_discrete_map=STAGE_COLORS,
            title="Chart of montage-stage share within each gewerk",
            labels={
                "stage_share_of_gewerk": "Share of the gewerk's staged days",
                "gewerk_name": "",
                "stage_name": "Montage stage",
            },
            height=max(420, 30 * len(gewerk_order)),
        ),
        width="stretch",
    )
    if not stages.empty:
        right.plotly_chart(
            px.bar(
                stages,
                x="stage_name",
                y="task_days",
                color="stage_name",
                category_orders={"stage_name": STAGE_ORDER},
                color_discrete_map=STAGE_COLORS,
                title=f"Chart of task days by montage stage{suffix}",
                labels={
                    "task_days": "Task days",
                    "stage_name": "Montage stage",
                },
                height=420,
            ),
            width="stretch",
        )

    if "inferred_task_days" in table:
        inferred = table["inferred_task_days"].sum() / max(table["task_days"].sum(), 1)
        st.caption(
            "Only tasks with an identifiable stage are counted, so the segments "
            "do not add up to a gewerk's full task days - the remainder is "
            f"planning, procurement and testing. {inferred * 100:.0f}% of the "
            "staged days here come from the task-name clustering rather than "
            "from a parsed field or a keyword rule."
        )

    gewerk_options = sorted(table["gewerk_name"].dropna().unique().tolist())
    picked = st.multiselect(
        "Filter gewerke in the table",
        options=gewerk_options,
        default=gewerk_options,
        key=f"gewerk_stage_filter_{schedule_id or 'all'}",
    )
    filtered = table[table["gewerk_name"].isin(picked)] if picked else table
    st.markdown("##### Table of gewerk × montage-stage time metrics")
    st.dataframe(
        gewerk_stage_table(filtered.sort_values("task_days", ascending=False)),
        hide_index=True,
        width="stretch",
        height=420,
    )


# ------------------------------------------------------------ combined view


def render_combined(tasks: pd.DataFrame) -> None:
    overview = load_table("schedule_overview")
    st.subheader("What is in the benchmark")

    columns = st.columns(5)
    columns[0].metric("Schedules", len(overview))
    columns[1].metric("Projects", overview["project"].nunique())
    columns[2].metric("Tasks", f"{len(tasks):,}".replace(",", "'"))
    columns[3].metric(
        "Work tasks", f"{int(overview['work_tasks'].sum()):,}".replace(",", "'")
    )
    columns[4].metric(
        "Calendar covered",
        f"{pd.to_datetime(overview['first_start']).min():%Y} - "
        f"{pd.to_datetime(overview['last_finish']).max():%Y}",
    )

    st.markdown("##### Table of schedules in the benchmark")
    view = overview.copy()
    view["first_start"] = pd.to_datetime(view["first_start"]).dt.date
    view["last_finish"] = pd.to_datetime(view["last_finish"]).dt.date
    view["project_span_months"] = view["project_span_months"].round(1)
    view["task_days"] = view["task_days"].round(0)
    st.dataframe(
        view[
            [
                "schedule_label", "tasks", "work_tasks", "milestones", "gewerke",
                "floors", "zones", "contractors", "first_start", "last_finish",
                "project_span_months", "task_days",
            ]
        ].rename(
            columns={
                "schedule_label": "Schedule", "tasks": "Tasks",
                "work_tasks": "Work tasks", "milestones": "Milestones",
                "gewerke": "Gewerke", "floors": "Floors", "zones": "Zones",
                "contractors": "Firms", "first_start": "Start",
                "last_finish": "Finish", "project_span_months": "Months",
                "task_days": "Task days",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    render_gewerk_stage_section(schedule_id=None)

    st.divider()
    st.subheader("Where the time goes, by gewerk")
    duration_explainer()

    combined = load_table("metrics_gewerk_all")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            gewerk_bar(
                combined,
                "task_days",
                "Chart of task days by gewerk across all schedules",
                "Task days",
            ),
            width="stretch",
        )
    with right:
        share = combined.nlargest(10, "task_days")
        st.plotly_chart(
            px.pie(
                share,
                values="task_days",
                names="gewerk_name",
                title="Chart of top 10 gewerke by share of task days",
                color="gewerk",
                color_discrete_map=GEWERK_COLORS,
                hole=0.45,
            ),
            width="stretch",
        )

    st.markdown("##### Table of gewerk time metrics across all schedules")
    st.dataframe(axis_table(combined, "gewerk"), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Is the mix the same everywhere?")
    st.caption(
        "Each column is one schedule; each cell is that gewerk's share of the "
        "schedule's total task days. Reading across a row shows how differently "
        "the same trade is weighted from project to project."
    )
    matrix = load_table("gewerk_matrix")
    if not matrix.empty:
        matrix = matrix.set_index(["gewerk", "gewerk_name"])
        st.plotly_chart(
            px.imshow(
                matrix,
                labels={"x": "", "y": "", "color": "% of task days"},
                color_continuous_scale="YlOrBr",
                aspect="auto",
                text_auto=".0f",
                title="Chart of gewerk mix by schedule (% of task days)",
                height=520,
            ),
            width="stretch",
        )

    st.divider()
    render_programme_compare(overview)

    st.divider()
    st.subheader("Takt: how long one gewerk needs per floor-zone")
    st.caption(
        "Only gewerke that repeat across at least three floor-zone cells are shown. "
        "Consistency is 1.0 when every cell took the same time; a low value means "
        "the trade's cycle is not yet stable."
    )
    cycles = load_table("cycle_times")
    if not cycles.empty:
        top = cycles.nlargest(28, "cells")
        st.plotly_chart(
            px.scatter(
                top,
                x="median_cell_days",
                y="consistency",
                color="gewerk_name",
                color_discrete_map=gewerk_name_colors(top),
                size="cells",
                size_max=30,
                hover_data=["schedule_label", "gewerk_name", "cells", "takt_gap_days"],
                labels={
                    "median_cell_days": "Median days per floor-zone",
                    "consistency": "Cycle consistency",
                },
                title="Graph of floor-zone cycle time vs consistency by gewerk",
                height=460,
            ),
            width="stretch",
        )
        st.markdown("##### Table of floor-zone cycle times (takt)")
        st.dataframe(
            cycles[
                [
                    "schedule_label", "gewerk_name", "cells", "median_cell_days",
                    "p10_cell_days", "p90_cell_days", "takt_gap_days", "consistency",
                ]
            ]
            .round(2)
            .rename(
                columns={
                    "schedule_label": "Schedule", "gewerk_name": "Gewerk",
                    "cells": "Floor-zones", "median_cell_days": "Median days/cell",
                    "p10_cell_days": "P10", "p90_cell_days": "P90",
                    "takt_gap_days": "Takt gap", "consistency": "Consistency",
                }
            ),
            hide_index=True,
            width="stretch",
            height=320,
        )

    st.divider()
    st.subheader("Phases and task groups")
    left, right = st.columns(2)
    phases = load_table("metrics_phase")
    if not phases.empty:
        pivot = phases.pivot_table(
            index="phase_name", columns="schedule_label", values="task_days_share", aggfunc="sum"
        ).fillna(0)
        left.plotly_chart(
            px.imshow(
                (pivot * 100).round(0),
                labels={"color": "% of task days"},
                color_continuous_scale="Blues",
                aspect="auto",
                text_auto=".0f",
                title="Chart of phase share per schedule",
                height=430,
            ),
            width="stretch",
        )
    activities = load_table("metrics_activity")
    if not activities.empty:
        pivot = activities.pivot_table(
            index="activity_name", columns="schedule_label", values="task_days_share", aggfunc="sum"
        ).fillna(0)
        right.plotly_chart(
            px.imshow(
                (pivot * 100).round(0),
                labels={"color": "% of task days"},
                color_continuous_scale="Greens",
                aspect="auto",
                text_auto=".0f",
                title="Chart of task-group share per schedule",
                height=430,
            ),
            width="stretch",
        )

    st.divider()
    render_coverage()


def render_programme_compare(overview: pd.DataFrame) -> None:
    st.subheader("When each gewerk was on the programme")
    st.caption(
        "Position 0 is the day the schedule starts, 1 the day it finishes. "
        "Pick two or more schedules to overlay their trade sequences. "
        "Normalizing this way lets a two-year fit-out and a seven-year hospital "
        "be compared on sequence rather than on length."
    )

    sequence = load_table("sequence")
    if sequence.empty:
        st.info("No sequence data available.")
        return

    labels = overview["schedule_label"].tolist()
    default = labels[: min(4, len(labels))]
    selected = st.multiselect(
        "Schedules to compare",
        options=labels,
        default=default,
        key="programme_compare_schedules",
        help="Select any combination of schedules. The chart and table update to match.",
    )
    if not selected:
        st.warning("Select at least one schedule to compare.")
        return

    filtered = sequence[sequence["schedule_label"].isin(selected)]
    figure = px.scatter(
        filtered,
        x="start_position",
        y="gewerk_name",
        color="schedule_label",
        size="task_days_share",
        size_max=26,
        labels={
            "start_position": "Normalized start (0 = project start, 1 = finish)",
            "gewerk_name": "",
            "schedule_label": "Schedule",
        },
        title="Graph of when each gewerk started on the selected programmes",
        height=620,
    )
    figure.update_xaxes(range=[-0.05, 1.05])
    st.plotly_chart(figure, width="stretch")

    # Gantt-style comparison on the 0–1 axis (numeric bars, not datetimes).
    timeline_rows = []
    for label in selected:
        side = filtered[filtered["schedule_label"] == label].dropna(
            subset=["start_position", "finish_position"]
        )
        for _, row in side.iterrows():
            start = float(row["start_position"])
            finish = float(row["finish_position"])
            timeline_rows.append(
                {
                    "schedule_label": label,
                    "gewerk_name": row["gewerk_name"],
                    "start": start,
                    "width": max(finish - start, 0.01),
                    "task_days_share": row["task_days_share"],
                }
            )
    if timeline_rows:
        bars = pd.DataFrame(timeline_rows)
        gewerk_order = sorted(bars["gewerk_name"].unique().tolist())
        figure = go.Figure()
        for label in selected:
            chunk = bars[bars["schedule_label"] == label]
            figure.add_trace(
                go.Bar(
                    name=label,
                    y=chunk["gewerk_name"],
                    x=chunk["width"],
                    base=chunk["start"],
                    orientation="h",
                    opacity=0.7,
                    hovertemplate=(
                        "%{y}<br>" + label
                        + "<br>start %{base:.0%}<br>width %{x:.0%}<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            barmode="overlay",
            title="Chart of gewerk programme windows for the selected schedules",
            xaxis=dict(range=[-0.05, 1.05], title="Normalized programme position (0–1)"),
            yaxis=dict(
                categoryorder="array",
                categoryarray=list(reversed(gewerk_order)),
                title="",
            ),
            height=max(360, 30 * len(gewerk_order)),
            legend_title_text="Schedule",
        )
        st.plotly_chart(figure, width="stretch")

    st.markdown("##### Table of gewerk programme positions for the selected schedules")
    show = filtered[
        [
            "schedule_label",
            "gewerk_name",
            "start_position",
            "finish_position",
            "task_days_share",
            "span_days",
            "active_days",
        ]
    ].copy()
    show["start_position"] = (show["start_position"] * 100).round(1)
    show["finish_position"] = (show["finish_position"] * 100).round(1)
    show["task_days_share"] = (show["task_days_share"] * 100).round(1)
    st.dataframe(
        show.rename(
            columns={
                "schedule_label": "Schedule",
                "gewerk_name": "Gewerk",
                "start_position": "Start %",
                "finish_position": "Finish %",
                "task_days_share": "Share %",
                "span_days": "Span days",
                "active_days": "Active days",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )


def render_coverage() -> None:
    coverage = load_coverage()
    if not coverage:
        return
    st.subheader("How much of this is evidence?")
    st.caption(
        "Every classification records how it was reached. `column` means the "
        "schedule already said so, `parse` means it was decoded from a structured "
        "task name or path, `rule` means a keyword table, `cache`/`llm` means a "
        "language model, and `fallback` means nothing matched."
    )

    rows = []
    for schedule_id, block in coverage.get("by_schedule", {}).items():
        row = {"schedule_id": schedule_id, "tasks": block["tasks"]}
        for axis in ["gewerk", "phase", "activity", "component", "montage"]:
            if axis not in block:
                continue
            resolved = block[axis]["resolved_work_tasks"]
            row[axis] = round(resolved * 100, 1) if resolved is not None else None
        for field, value in block["fields"].items():
            row[field] = round(value * 100, 1)
        rows.append(row)

    st.markdown("##### Table of classification coverage by schedule")
    rename = {
        "schedule_id": "Schedule", "tasks": "Tasks",
        "gewerk": "Gewerk %", "phase": "Phase %",
        "activity": "Task group %", "component": "Component %",
        "montage": "Montage %",
        "start": "Dated %", "duration": "Duration %", "floor": "Floor %",
        "zone": "Zone %", "crew_size": "Crew %", "contractor": "Firm %",
        "critical_flag": "Critical %",
    }
    st.dataframe(
        pd.DataFrame(rows).rename(columns=rename),
        hide_index=True,
        width="stretch",
    )

    sources = coverage["overall"]["gewerk"]["by_source"]
    st.caption(
        "Gewerk assignments overall: "
        + ", ".join(f"**{count:,}** by `{source}`".replace(",", "'") for source, count in sources.items())
        + f". {coverage.get('rule_vs_cache_disagreements', 0)} of them disagree with a "
        "label an earlier LLM run gave the same task name; those are listed in "
        "`out/rule_vs_llm.csv`."
    )


# ------------------------------------------------------------ schedule view


def render_schedule(tasks: pd.DataFrame, schedule_id: str) -> None:
    subset = tasks[tasks["schedule_id"] == schedule_id]
    work = metrics.executable(subset)
    overview = load_table("schedule_overview")
    row = overview[overview["schedule_id"] == schedule_id].iloc[0]

    st.subheader(row["schedule_label"])
    st.caption(f"`{row['source_file']}`")

    columns = st.columns(6)
    columns[0].metric("Tasks", f"{int(row['tasks']):,}".replace(",", "'"))
    columns[1].metric("Work tasks", f"{int(row['work_tasks']):,}".replace(",", "'"))
    columns[2].metric("Gewerke", int(row["gewerke"]))
    columns[3].metric("Duration", f"{row['project_span_months']:.0f} months")
    columns[4].metric("Task days", days(row["task_days"]))
    columns[5].metric("Median task", days(row["median_task_days"]))

    gewerk = load_table("metrics_gewerk")
    gewerk = gewerk[gewerk["schedule_id"] == schedule_id]
    if gewerk.empty:
        st.info("No classified work tasks in this schedule.")
        return

    st.divider()
    render_gewerk_stage_section(schedule_id)

    st.divider()
    if schedule_id.startswith("pred_"):
        render_pred_banner(subset)
    if schedule_id == "dc30":
        render_dc30_difference()
        st.divider()
        render_dc30_panel(subset)
        st.divider()

    st.markdown("#### Time per gewerk")
    duration_explainer()
    left, right = st.columns([3, 2])
    left.plotly_chart(
        gewerk_bar(gewerk, "task_days", "Chart of task days by gewerk", "Task days"),
        width="stretch",
    )
    right.plotly_chart(
        gewerk_bar(
            gewerk[gewerk["active_days"].notna()],
            "active_days",
            "Chart of active days on site by gewerk",
            "Active days",
        ),
        width="stretch",
    )
    st.markdown("##### Table of gewerk time metrics")
    st.dataframe(axis_table(gewerk, "gewerk"), hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### When each gewerk was on the programme")
    timeline = gewerk.dropna(subset=["first_start", "last_finish"]).copy()
    if not timeline.empty:
        timeline["first_start"] = pd.to_datetime(timeline["first_start"])
        timeline["last_finish"] = pd.to_datetime(timeline["last_finish"])
        figure = px.timeline(
            timeline.sort_values("first_start"),
            x_start="first_start",
            x_end="last_finish",
            y="gewerk_name",
            color_discrete_sequence=["#546e7a"],
            hover_data=["tasks", "task_days", "active_days", "peak_concurrency"],
            title="Chart of when each gewerk was on the programme",
            height=max(300, 28 * len(timeline)),
        )
        figure.update_yaxes(autorange="reversed", title="")
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "The bar spans first start to last finish. A long bar with few active "
            "days means the trade came back repeatedly rather than working through."
        )

    render_components(schedule_id)
    render_extras(subset)

    with st.expander(f"Browse all {len(subset):,} tasks".replace(",", "'")):
        columns = [
            "task_name", "gewerk_name", "montage_name", "stage_name", "stage_source",
            "cluster_id", "cluster_label", "phase_name", "activity_name",
            "component_name", "floor_norm", "zone", "start", "finish",
            "duration_days", "is_critical", "contractor",
        ]
        st.dataframe(
            subset[[c for c in columns if c in subset]],
            hide_index=True,
            width="stretch",
            height=420,
        )


def render_components(schedule_id: str) -> None:
    components = load_table("metrics_component")
    components = components[components["schedule_id"] == schedule_id]
    if components.empty:
        return
    st.divider()
    st.markdown("#### Component groups")
    top = components.nlargest(20, "task_days")
    st.plotly_chart(
        px.bar(
            top,
            x="task_days",
            y="component_name",
            orientation="h",
            labels={"task_days": "Task days", "component_name": ""},
            title="Chart of task days by component group",
            height=max(320, 24 * len(top)),
        ).update_yaxes(categoryorder="total ascending"),
        width="stretch",
    )


def render_extras(subset: pd.DataFrame) -> None:
    """Fields only some schedules carry: crew size, procurement package, takt."""
    panels = []
    if subset["crew_size"].notna().any():
        panels.append("crew")
    if subset["procurement_package"].notna().any():
        panels.append("package")
    if "takt_nr" in subset and subset["takt_nr"].notna().any():
        takt = pd.to_numeric(subset["takt_nr"], errors="coerce")
        if takt.gt(0).any():
            panels.append("takt")
    if not panels:
        return

    st.divider()
    st.markdown("#### Extra parameters in this schedule")
    columns = st.columns(len(panels))

    for column, panel in zip(columns, panels):
        if panel == "crew":
            crew = subset[subset["crew_size"].notna()].copy()
            crew["person_days"] = crew["duration_days"] * crew["crew_size"]
            by_gewerk = (
                crew.groupby("gewerk_name")["person_days"].sum().nlargest(12).reset_index()
            )
            column.plotly_chart(
                px.bar(
                    by_gewerk,
                    x="person_days",
                    y="gewerk_name",
                    orientation="h",
                    title="Chart of person-days by gewerk (crew × duration)",
                    labels={"person_days": "Person-days", "gewerk_name": ""},
                    height=360,
                ).update_yaxes(categoryorder="total ascending"),
                width="stretch",
            )
        elif panel == "package":
            packages = (
                subset[subset["procurement_package"].notna()]
                .groupby("procurement_package")["duration_days"]
                .agg(["size", "sum"])
                .nlargest(12, "sum")
                .reset_index()
            )
            packages.columns = ["Package", "Tasks", "Task days"]
            column.markdown("##### Table of largest procurement packages")
            column.dataframe(packages, hide_index=True, width="stretch", height=360)
        elif panel == "takt":
            takt = subset.copy()
            takt["takt_nr"] = pd.to_numeric(takt["takt_nr"], errors="coerce")
            takt = takt[takt["takt_nr"] > 0]
            by_takt = (
                takt.groupby("takt_nr")
                .agg(tasks=("uid", "size"), task_days=("duration_days", "sum"))
                .reset_index()
            )
            column.plotly_chart(
                px.bar(
                    by_takt,
                    x="takt_nr",
                    y="task_days",
                    title="Chart of task days per Takt",
                    labels={"takt_nr": "Takt", "task_days": "Task days"},
                    height=360,
                ),
                width="stretch",
            )


# ------------------------------------------------- special-case panels


def render_pred_banner(subset: pd.DataFrame) -> None:
    work = metrics.executable(subset)
    if work.empty or "work_stream" not in work:
        return
    procurement = work[work["work_stream"] == "Beschaffung"]
    site = work[work["work_stream"] == "Ausfuehrung"]
    share = len(procurement) / len(work) if len(work) else 0

    st.warning(
        f"**Read this schedule differently.** {share:.0%} of its work tasks "
        f"({len(procurement):,} of {len(work):,}) are procurement and submittal "
        "steps - onboarding, model work, DKS/WMP approval loops, ordering and "
        "prefabrication - not site production. Averaging the two together would "
        "make every trade look far slower than it builds. The two streams are "
        "kept apart everywhere below.".replace(",", "'")
    )

    columns = st.columns(4)
    columns[0].metric("Procurement tasks", f"{len(procurement):,}".replace(",", "'"))
    columns[1].metric("Site tasks", f"{len(site):,}".replace(",", "'"))
    columns[2].metric(
        "Median procurement step", days(procurement["duration_days"].median())
    )
    columns[3].metric("Median site task", days(site["duration_days"].median()))

    split = load_table("pred_split")
    split = split[split["schedule_id"] == subset["schedule_id"].iloc[0]]
    if not split.empty:
        st.plotly_chart(
            px.bar(
                split,
                x="task_days",
                y="gewerk_name",
                color="work_stream",
                orientation="h",
                barmode="group",
                title="Chart of procurement lead time vs site production per gewerk",
                labels={
                    "task_days": "Task days",
                    "gewerk_name": "",
                    "work_stream": "Stream",
                },
                color_discrete_map={"Beschaffung": "#90a4ae", "Ausfuehrung": "#ef6c00"},
                height=max(340, 30 * split["gewerk_name"].nunique()),
            ).update_yaxes(categoryorder="total ascending"),
            width="stretch",
        )

    if "install_stage" in work and work["install_stage"].notna().any():
        stages = (
            work[work["install_stage"].notna()]
            .groupby("install_stage")
            .agg(
                tasks=("uid", "size"),
                task_days=("duration_days", "sum"),
                median_days=("duration_days", "median"),
            )
            .reset_index()
        )
        st.caption(
            "pRED sequences its site work in three installation stages: **E1** = "
            "Grobmontage, **E2** = Feinmontage, **E3** = Endmontage. That is the "
            "schedule's own takt structure, mapped onto the common montage axis."
        )
        st.markdown("##### Table of pRED installation stages (E1 / E2 / E3)")
        st.dataframe(
            stages.rename(
                columns={
                    "install_stage": "Stage", "tasks": "Tasks",
                    "task_days": "Task days", "median_days": "Median task",
                }
            ).round(1),
            hide_index=True,
            width="stretch",
        )


def render_dc30_panel(subset: pd.DataFrame) -> None:
    st.info(
        "**The only schedule that says what each task consumes.** DC30 lists "
        "inbound material with quantity units per task, alongside English "
        "descriptions and German keywords. That makes it the one place a duration "
        "can be tied to a unit of work."
    )

    materials = load_table("dc30_materials")
    if materials.empty:
        return

    columns = st.columns(3)
    columns[0].metric("Distinct materials", materials["material"].nunique())
    columns[1].metric("Units in use", materials["unit"].nunique())
    columns[2].metric(
        "Tasks with material", int(subset["material_count"].fillna(0).gt(0).sum())
    )

    left, right = st.columns([3, 2])
    top = materials.nlargest(20, "task_days")
    left.plotly_chart(
        px.bar(
            top,
            x="task_days",
            y="material",
            color="gewerk_name",
            color_discrete_map=gewerk_name_colors(top),
            orientation="h",
            title="Chart of task days by inbound material",
            labels={"task_days": "Task days", "material": ""},
            height=560,
        ).update_yaxes(categoryorder="total ascending"),
        width="stretch",
    )
    by_unit = (
        materials.groupby("unit")
        .agg(materials=("material", "nunique"), task_days=("task_days", "sum"))
        .reset_index()
        .sort_values("task_days", ascending=False)
    )
    right.plotly_chart(
        px.pie(
            by_unit,
            values="task_days",
            names="unit",
            title="Chart of task days by quantity unit",
            hole=0.45,
        ),
        width="stretch",
    )

    with st.expander("All materials by gewerk"):
        st.markdown("##### Table of inbound materials by gewerk")
        st.dataframe(
            materials.rename(
                columns={
                    "gewerk_name": "Gewerk", "material": "Material", "unit": "Unit",
                    "tasks": "Tasks", "task_days": "Task days",
                    "median_task_days": "Median task",
                }
            )[["Gewerk", "Material", "Unit", "Tasks", "Task days", "Median task"]].round(1),
            hide_index=True,
            width="stretch",
            height=420,
        )


# --------------------------------------------------------------------- main


def main() -> None:
    if not (OUT_DIR / "tasks_all.parquet").is_file():
        st.error(
            "No build output found. Run `python -m schedule_benchmark.src.build` first."
        )
        return

    tasks = load_tasks()
    overview = load_table("schedule_overview")

    st.title("Hermes Gewerke-Benchmark")
    st.caption(
        f"{len(overview)} construction schedules from {overview['project'].nunique()} "
        f"projects, harmonized onto one gewerk, phase, task-group, montage-stage "
        f"and component taxonomy. Combined findings first, then one tab per "
        f"schedule."
    )

    labels = ["Combined"] + overview["schedule_label"].tolist()
    tabs = st.tabs(labels)

    with tabs[0]:
        render_combined(tasks)
    for tab, schedule_id in zip(tabs[1:], overview["schedule_id"]):
        with tab:
            render_schedule(tasks, schedule_id)


main()
