"""How different DC30 is from every other schedule in the benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from schedule_benchmark.dashboard.shared import (
    DC30_COLOR,
    MONTAGE_ORDER,
    PEER_COLOR,
    days,
    load_table,
    load_tasks,
)


def _share_vector(frame: pd.DataFrame, code_col: str, name_col: str, value_col: str = "task_days") -> pd.Series:
    totals = frame.groupby(code_col, dropna=True)[value_col].sum()
    if totals.sum() <= 0:
        return pd.Series(dtype=float)
    shares = totals / totals.sum()
    names = (
        frame.dropna(subset=[code_col])
        .drop_duplicates(code_col)
        .set_index(code_col)[name_col]
    )
    shares.index = shares.index.map(lambda c: names.get(c, c))
    return shares


def _l1_distance(a: pd.Series, b: pd.Series) -> float:
    idx = sorted(set(a.index) | set(b.index))
    left = a.reindex(idx, fill_value=0.0)
    right = b.reindex(idx, fill_value=0.0)
    return float((left - right).abs().sum())


def _peer_median_share(gewerk_metrics: pd.DataFrame) -> pd.Series:
    peers = gewerk_metrics[gewerk_metrics["schedule_id"] != "dc30"]
    if peers.empty:
        return pd.Series(dtype=float)
    pivot = peers.pivot_table(
        index="gewerk_name", columns="schedule_id", values="task_days_share", aggfunc="sum"
    ).fillna(0.0)
    return pivot.median(axis=1)


def render_dc30_difference() -> None:
    """DC30-vs-everyone analysis, rendered inside the DC30 schedule tab."""
    tasks = load_tasks()
    overview = load_table("schedule_overview")
    gewerk = load_table("metrics_gewerk")
    phase = load_table("metrics_phase")
    montage = load_table("metrics_montage")
    sequence = load_table("sequence")

    dc_row = overview[overview["schedule_id"] == "dc30"]
    peers = overview[overview["schedule_id"] != "dc30"]
    if dc_row.empty:
        st.error("DC30 is not in the build output.")
        return
    dc = dc_row.iloc[0]

    st.markdown("#### How different is DC30 from the other schedules?")
    st.caption(
        "DC30 is a data-center fit-out schedule. This section measures how far it "
        "sits from the other ten schedules on duration, trade mix, phase mix, "
        "montage stages and the fields only DC30 carries."
    )

    # ---------------------------------------------------------------- KPIs
    st.markdown("##### Headline differences")
    cols = st.columns(5)
    cols[0].metric(
        "Programme length",
        f"{dc['project_span_months']:.0f} months",
        delta=f"{dc['project_span_months'] - peers['project_span_months'].median():+.0f} vs peer median",
        delta_color="inverse",
    )
    cols[1].metric(
        "Median task",
        days(dc["median_task_days"]),
        delta=f"{dc['median_task_days'] - peers['median_task_days'].median():+.0f} d vs peer median",
        delta_color="inverse",
    )
    cols[2].metric(
        "Work tasks",
        f"{int(dc['work_tasks']):,}".replace(",", "'"),
        delta=f"{int(dc['work_tasks'] - peers['work_tasks'].median()):+,} vs peer median".replace(",", "'"),
    )
    cols[3].metric("Milestones", int(dc["milestones"]), help="DC30 has no milestone rows.")
    cols[4].metric(
        "Tasks with material",
        f"{int((tasks.loc[tasks.schedule_id == 'dc30', 'material_count'].fillna(0) > 0).sum()):,}".replace(",", "'"),
        help="No other schedule in the set carries inbound material quantities.",
    )

    st.info(
        "**What makes DC30 structurally unique.** It is the only schedule with "
        "`Inbound_Material` quantity units, English task descriptions and numbered "
        "contractor packages (`12 / Kälte`). It is also the shortest programme in "
        "the set (~14 months vs a peer median of ~"
        f"{peers['project_span_months'].median():.0f} months), has **no milestones "
        "and no summary tasks**, and is dominated by TGA rather than Ausbau."
    )

    # -------------------------------------------------------- distance board
    st.divider()
    st.markdown("##### How far is each schedule from DC30?")
    st.caption(
        "Distance is the L1 distance between gewerk task-day share vectors "
        "(0 = identical mix, 2 = completely disjoint). Smaller = more similar."
    )

    dc_gewerk = gewerk[gewerk["schedule_id"] == "dc30"]
    dc_vec = dc_gewerk.set_index("gewerk_name")["task_days_share"]
    distance_rows = []
    for schedule_id, group in gewerk[gewerk["schedule_id"] != "dc30"].groupby("schedule_id"):
        label = group["schedule_label"].iloc[0]
        peer_vec = group.set_index("gewerk_name")["task_days_share"]
        distance_rows.append(
            {
                "schedule_id": schedule_id,
                "schedule_label": label,
                "gewerk_distance": _l1_distance(dc_vec, peer_vec),
            }
        )
    distances = pd.DataFrame(distance_rows).sort_values("gewerk_distance")

    # Phase distance too.
    dc_phase = phase[phase["schedule_id"] == "dc30"].set_index("phase_name")["task_days_share"]
    phase_dist = []
    for schedule_id, group in phase[phase["schedule_id"] != "dc30"].groupby("schedule_id"):
        phase_dist.append(
            {
                "schedule_id": schedule_id,
                "phase_distance": _l1_distance(
                    dc_phase, group.set_index("phase_name")["task_days_share"]
                ),
            }
        )
    distances = distances.merge(pd.DataFrame(phase_dist), on="schedule_id", how="left")
    distances["combined"] = (
        distances["gewerk_distance"] + distances["phase_distance"].fillna(0)
    ) / 2

    left, right = st.columns([3, 2])
    left.plotly_chart(
        px.bar(
            distances.sort_values("gewerk_distance", ascending=True),
            x="gewerk_distance",
            y="schedule_label",
            orientation="h",
            title="Chart of gewerk-mix distance from DC30 (L1)",
            labels={"gewerk_distance": "L1 distance", "schedule_label": ""},
            color="gewerk_distance",
            color_continuous_scale="Reds",
            height=max(320, 28 * len(distances)),
        ).update_layout(showlegend=False, coloraxis_showscale=False),
        width="stretch",
    )
    right.markdown("###### Table of distance from DC30")
    right.dataframe(
        distances.sort_values("gewerk_distance")[
            ["schedule_label", "gewerk_distance", "phase_distance", "combined"]
        ]
        .round(3)
        .rename(
            columns={
                "schedule_label": "Schedule",
                "gewerk_distance": "Gewerk L1",
                "phase_distance": "Phase L1",
                "combined": "Average",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )

    closest = distances.nsmallest(1, "gewerk_distance").iloc[0]
    farthest = distances.nlargest(1, "gewerk_distance").iloc[0]
    st.caption(
        f"Closest gewerk mix: **{closest['schedule_label']}** "
        f"(L1 {closest['gewerk_distance']:.2f}). "
        f"Farthest: **{farthest['schedule_label']}** "
        f"(L1 {farthest['gewerk_distance']:.2f})."
    )

    # ----------------------------------------------------------- gewerk mix
    st.divider()
    st.markdown("##### Trade mix: DC30 against the peer median")
    st.caption(
        "Peer median is the median task-day share of each gewerk across the other "
        "ten schedules. DC30 is an electrical / refrigeration data-center fit-out; "
        "most peers are mixed building programmes dominated by Ausbau and Konstruktion."
    )

    peer_median = _peer_median_share(gewerk)
    compare = pd.DataFrame(
        {
            "gewerk_name": sorted(set(dc_vec.index) | set(peer_median.index)),
        }
    )
    compare["DC30"] = compare["gewerk_name"].map(dc_vec).fillna(0) * 100
    compare["Peer median"] = compare["gewerk_name"].map(peer_median).fillna(0) * 100
    compare["Delta pp"] = compare["DC30"] - compare["Peer median"]
    compare = compare.sort_values("DC30", ascending=False)

    melted = compare.melt(
        id_vars="gewerk_name",
        value_vars=["DC30", "Peer median"],
        var_name="Series",
        value_name="share",
    )
    st.plotly_chart(
        px.bar(
            melted,
            x="share",
            y="gewerk_name",
            color="Series",
            orientation="h",
            barmode="group",
            category_orders={"gewerk_name": compare["gewerk_name"].tolist()[::-1]},
            color_discrete_map={"DC30": DC30_COLOR, "Peer median": PEER_COLOR},
            title="Chart of gewerk share: DC30 vs peer median (% of task days)",
            labels={"share": "% of task days", "gewerk_name": ""},
            height=max(380, 28 * len(compare)),
        ),
        width="stretch",
    )

    st.markdown("###### Table of gewerk share deltas (percentage points)")
    st.dataframe(
        compare.rename(
            columns={
                "gewerk_name": "Gewerk",
                "DC30": "DC30 %",
                "Peer median": "Peer median %",
                "Delta pp": "Δ pp",
            }
        ).round(1),
        hide_index=True,
        width="stretch",
    )

    over = compare.nlargest(3, "Delta pp")
    under = compare.nsmallest(3, "Delta pp")

    def _delta_list(frame: pd.DataFrame) -> str:
        parts = []
        for name, delta in zip(frame["gewerk_name"], frame["Delta pp"]):
            parts.append(f"{name} ({float(delta):+.1f} pp)")
        return ", ".join(parts)

    st.markdown(
        "- **Over-represented on DC30:** "
        + _delta_list(over)
        + "\n- **Under-represented on DC30:** "
        + _delta_list(under)
    )

    # -------------------------------------------------------------- phases
    st.divider()
    st.markdown("##### Phase mix")
    dc_phase_rows = phase[phase["schedule_id"] == "dc30"][
        ["phase_name", "task_days_share"]
    ].copy()
    dc_phase_rows["Series"] = "DC30"
    peer_phase = (
        phase[phase["schedule_id"] != "dc30"]
        .groupby("phase_name")["task_days_share"]
        .median()
        .reset_index()
    )
    peer_phase["Series"] = "Peer median"
    phase_cmp = pd.concat(
        [
            dc_phase_rows.rename(columns={"task_days_share": "share"}),
            peer_phase.rename(columns={"task_days_share": "share"}),
        ],
        ignore_index=True,
    )
    phase_cmp["share"] = phase_cmp["share"] * 100
    st.plotly_chart(
        px.bar(
            phase_cmp,
            x="share",
            y="phase_name",
            color="Series",
            orientation="h",
            barmode="group",
            color_discrete_map={"DC30": DC30_COLOR, "Peer median": PEER_COLOR},
            title="Chart of phase share: DC30 vs peer median",
            labels={"share": "% of task days", "phase_name": ""},
            height=420,
        ),
        width="stretch",
    )
    st.caption(
        "DC30 is almost entirely **TGA-Installation** and **Ausbau**. Peers spread "
        "time across Planung, Beschaffung, Rohbau, Hülle and Inbetriebnahme as well."
    )

    # ------------------------------------------------------------- montage
    st.divider()
    st.markdown("##### Montage stages")
    if not montage.empty:
        dc_m = montage[montage["schedule_id"] == "dc30"][
            ["montage", "montage_name", "task_days"]
        ].copy()
        dc_total = dc_m["task_days"].sum()
        dc_m["share"] = dc_m["task_days"] / dc_total * 100 if dc_total else 0
        dc_m["Series"] = "DC30"

        peer_m = (
            montage[montage["schedule_id"] != "dc30"]
            .groupby(["montage", "montage_name"], as_index=False)["task_days"]
            .sum()
        )
        peer_total = peer_m["task_days"].sum()
        peer_m["share"] = peer_m["task_days"] / peer_total * 100 if peer_total else 0
        peer_m["Series"] = "Other schedules"

        mont_cmp = pd.concat([dc_m, peer_m], ignore_index=True)
        st.plotly_chart(
            px.bar(
                mont_cmp,
                x="montage_name",
                y="share",
                color="Series",
                barmode="group",
                category_orders={"montage_name": MONTAGE_ORDER},
                color_discrete_map={"DC30": DC30_COLOR, "Other schedules": PEER_COLOR},
                title="Chart of montage-stage mix: DC30 vs other schedules",
                labels={"share": "% of staged task days", "montage_name": ""},
                height=380,
            ),
            width="stretch",
        )

    # ----------------------------------------------------------- programme
    st.divider()
    st.markdown("##### When each gewerk sits on the programme")
    st.caption(
        "Normalized start position (0 = schedule start, 1 = finish). "
        "Pick peers to overlay against DC30."
    )
    if not sequence.empty:
        peer_labels = overview.loc[
            overview["schedule_id"] != "dc30", "schedule_label"
        ].tolist()
        default_peers = distances.nsmallest(3, "gewerk_distance")["schedule_label"].tolist()
        chosen = st.multiselect(
            "Peer schedules to overlay",
            options=peer_labels,
            default=[p for p in default_peers if p in peer_labels][:3],
            key="dc30_peer_overlay",
        )
        selected_labels = ["DC30 Rechenzentrum"] + chosen
        seq = sequence[sequence["schedule_label"].isin(selected_labels)]
        st.plotly_chart(
            px.scatter(
                seq,
                x="start_position",
                y="gewerk_name",
                color="schedule_label",
                size="task_days_share",
                size_max=24,
                title="Graph of gewerk start positions: DC30 vs selected peers",
                labels={
                    "start_position": "Normalized start (0–1)",
                    "gewerk_name": "",
                    "schedule_label": "Schedule",
                },
                height=560,
            ).update_xaxes(range=[-0.05, 1.05]),
            width="stretch",
        )

    # -------------------------------------------------------- unique assets
    st.divider()
    st.markdown("##### What only DC30 has")
    st.caption(
        "Fields no other schedule in the set carries. The material breakdown "
        "itself follows further down this tab."
    )
    dc_tasks = tasks[tasks["schedule_id"] == "dc30"]

    c1, c2, c3 = st.columns(3)
    material_tasks = int((dc_tasks["material_count"].fillna(0) > 0).sum())
    c1.metric(
        "Material coverage",
        f"{material_tasks / max(len(dc_tasks), 1) * 100:.0f}%",
    )
    c2.metric(
        "English descriptions",
        f"{dc_tasks['description_en'].notna().mean() * 100:.0f}%",
    )
    c3.metric(
        "Numbered packages",
        int(dc_tasks["package_code"].nunique()) if "package_code" in dc_tasks else 0,
    )

    # -------------------------------------------------------- size profile
    st.divider()
    st.markdown("##### Size and tempo against every peer")
    profile = overview.copy()
    profile["is_dc30"] = profile["schedule_id"] == "dc30"
    profile["label"] = np.where(profile["is_dc30"], "DC30", "Peer")
    st.plotly_chart(
        px.scatter(
            profile,
            x="project_span_months",
            y="task_days",
            size="work_tasks",
            color="label",
            hover_name="schedule_label",
            color_discrete_map={"DC30": DC30_COLOR, "Peer": PEER_COLOR},
            title="Graph of programme length vs total task days",
            labels={
                "project_span_months": "Programme length (months)",
                "task_days": "Task days",
                "label": "",
            },
            height=420,
            size_max=48,
        ),
        width="stretch",
    )
    st.caption(
        f"DC30 packs {dc['task_days']:,.0f} task days into "
        f"{dc['project_span_months']:.0f} months — dense, short, technical. "
        f"Peer median is {peers['task_days'].median():,.0f} task days over "
        f"{peers['project_span_months'].median():.0f} months.".replace(",", "'")
    )

    # -------------------------------------------------------- takeaways
    st.divider()
    st.markdown("##### Takeaways")
    st.markdown(
        f"""
1. **Tempo.** DC30 is a ~{dc['project_span_months']:.0f}-month data-center fit-out; peers are typically multi-year building programmes (median ~{peers['project_span_months'].median():.0f} months).
2. **Trade mix.** Elektro and Kälte dominate DC30; Ausbau and Konstruktion dominate most peers. Closest mix: **{closest['schedule_label']}**; farthest: **{farthest['schedule_label']}**.
3. **Phase.** ~78% of DC30 task days sit in TGA-Installation — peers spread across planning, procurement, shell and commissioning.
4. **Structure.** DC30 has no milestones and no summary rows; every row is an executable leaf.
5. **Evidence.** It is the only schedule that ties duration to inbound material quantities and English descriptions — use it when you need productivity per unit, not just per task.
"""
    )
