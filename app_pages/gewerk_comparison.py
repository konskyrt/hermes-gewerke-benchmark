"""Gewerk comparison dashboard - one card per trade, every schedule side by side.

    streamlit run dashboard/gewerk_dashboard.py

The schedule classifier puts every task of every schedule into a Gewerk. This
page turns that clustering around: each Gewerk gets a card, and inside the card
the schedules are compared directly - when the trade runs and in which phases,
how long each phase takes, how much material it moves (BuildSync material factors for the
tasks without BIM quantities) and how many trucks per week that means, next to
what historical projects actually delivered.

Data comes from ``schedule_clf.cli analytics`` (run automatically if missing).
Bundled mode (Streamlit Cloud): when the pipeline is not importable, the page
reads the exported tables from ``gewerk_data/`` next to ``gewerk_core.py``
(see ``dashboard/export_bundle.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

try:
    from schedule_clf import gewerk_core as ga
    BUNDLE_DIR = None
except ImportError:                      # bundled: gewerk_core.py sits at ROOT
    import gewerk_core as ga
    BUNDLE_DIR = ROOT / "gewerk_data"

# `width="stretch"` replaced `use_container_width` in Streamlit 1.45; the
# deployed app pins an older release, so pick whichever this one understands.
_ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VER >= (1, 45) else {"use_container_width": True}

st.set_page_config(page_title="Gewerk Comparison", page_icon="🏗️", layout="wide")

# Fixed phase -> colour (reference categorical palette, slot order = phase
# order). Colour follows the phase, so filtering never repaints a survivor.
PHASE_COLORS = dict(zip(ga.PHASE_ORDER, [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]))
PHASE_COLORS["—"] = "#898781"
BAND_MED, BAND_HIGH = "#86b6ef", "#1c5cab"   # one sequential hue: medium -> high
OBSERVED = "#eb6834"                          # historical, observed
MUTED = "#898781"
GRID = "rgba(137,135,129,0.2)"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading Gewerk analytics …")
def load_data() -> dict:
    if BUNDLE_DIR is not None:
        out = BUNDLE_DIR
    else:
        from schedule_clf import gewerk_analytics
        from schedule_clf.config import load_config
        cfg = load_config()
        out = cfg.root / cfg.section("analytics").get("out_dir", "data/analytics")
        if not (out / "gewerk_project.parquet").exists():
            gewerk_analytics.run(cfg)
    d = {p.stem: pd.read_parquet(p) for p in out.glob("*.parquet")}
    with open(out / "project_drivers.json", encoding="utf-8") as fh:
        d["meta"] = json.load(fh)
    d["vehicle"] = {g: tuple(v) for g, v in d["meta"].get("vehicles", {}).items()}
    return d


D = load_data()
TASKS, PROJ, GP, GPP = D["tasks"], D["projects"], D["gewerk_project"], D["phase_project"]
FACT, COMP = D["factors"].set_index("gewerk_label"), D["factor_components"]
HG = D.get("hist_gewerk", pd.DataFrame())
DRIVERS = D["meta"]["project_drivers"]
PROJECTS = PROJ.sort_values("start")["project_label"].tolist()
N_PROJ = len(PROJECTS)

CATEGORY = FACT["category"].to_dict()
GEWERKE = (GP.groupby("gewerk_label")
           .agg(n_proj=("project_label", "nunique"), n_tasks=("n_tasks", "sum"))
           .assign(category=lambda x: x.index.map(lambda g: CATEGORY.get(g, "Andere"))))
PHASE_RANK = {p: i for i, p in enumerate(ga.PHASE_ORDER)}


def phase_sorted(phases) -> list:
    return sorted(phases, key=lambda p: PHASE_RANK.get(p, 99))


def fmt(v, nd=0, suffix="") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{nd}f}{suffix}".replace(",", "'")


def stat_row(items: list[tuple[str, str, str]]) -> str:
    """Compact stat tiles (st.metric truncates at this density)."""
    cells = "".join(
        f"<div style='flex:1;min-width:120px;padding:8px 12px;border-radius:8px;"
        f"background:rgba(137,135,129,0.08)'>"
        f"<div style='font-size:0.75rem;color:{MUTED}'>{label}</div>"
        f"<div style='font-size:1.3rem;font-weight:600;line-height:1.35'>{value}</div>"
        f"<div style='font-size:0.72rem;color:{MUTED}'>{sub}</div></div>"
        for label, value, sub in items)
    return f"<div style='display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px'>{cells}</div>"


def base_layout(fig: go.Figure, title: str, n_rows: float, legend: bool = True) -> go.Figure:
    # Legends sit under the plot so a wrapped legend never collides with the
    # title; phase colours are explained once at the top of the page instead.
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)),
        height=int(100 + 30 * max(n_rows, 2) + (45 if legend else 0)),
        margin=dict(l=8, r=8, t=40, b=55 if legend else 8),
        showlegend=legend, legend=dict(orientation="h", yanchor="top", y=-0.12, x=0,
                                       font=dict(size=11), traceorder="normal"),
        hoverlabel=dict(namelength=-1), bargap=0.25)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ---------------------------------------------------------------------------
# Project sizes (no sidebar: known sizes from the analytics config, the rest
# assumed and marked * in the charts)
# ---------------------------------------------------------------------------

# Hide the sidebar and its toggle entirely - the page is the cards only.
st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'],"
            " [data-testid='collapsedControl'] {display: none;}</style>",
            unsafe_allow_html=True)

ASSUMED_BGF, ASSUMED_STOREYS = 15000.0, 6.0
known = D["meta"].get("project_size", {})
SIZES = {}
for p in PROJECTS:
    k = known.get(p, {})
    given = bool(k.get("bgf_m2"))
    SIZES[p] = dict(bgf=float(k["bgf_m2"]) if given else ASSUMED_BGF,
                    storeys=float(k.get("geschosse") or ASSUMED_STOREYS),
                    assumed=not given)


def ylabel(p: str) -> str:
    return f"{p} *" if SIZES[p]["assumed"] else p


# ---------------------------------------------------------------------------
# Per-Gewerk computation
# ---------------------------------------------------------------------------

def gewerk_frame(g: str) -> dict:
    gp = GP[GP.gewerk_label == g].set_index("project_label")
    scheds = [p for p in PROJECTS if p in gp.index]
    gp = gp.loc[scheds]
    f = FACT.loc[g] if g in FACT.index else None
    cg = ga.FINE_TO_CATALOG.get(g)
    veh, load_m3 = D["vehicle"].get(cg, (None, None))
    eff_load = (load_m3 or np.nan) * ga.LKW_UTILIZATION

    vol = []
    for p in scheds:
        s = SIZES[p]
        est = ga.estimate_volumes(COMP, DRIVERS, g, s["bgf"], s["storeys"])
        m = est["medium_m3"].sum() if not est.empty else np.nan
        h = est["high_m3"].sum() if not est.empty else np.nan
        aw = max(gp.loc[p, "active_weeks"], 1)
        vol.append(dict(project=p, medium=m, high=h, active_weeks=aw,
                        trips_wk_med=m / eff_load / aw, trips_wk_high=h / eff_load / aw))
    hist = HG[HG.catalog_gewerk == cg].sort_values("hist_name") if not HG.empty else HG
    return dict(gp=gp, gpp=GPP[GPP.gewerk_label == g], tasks=TASKS[TASKS.gewerk_label == g],
                scheds=scheds, f=f, cg=cg, veh=veh, load_m3=load_m3,
                vol=pd.DataFrame(vol).set_index("project"), hist=hist)


# ---------------------------------------------------------------------------
# Charts - each one compares the schedules of a single Gewerk
# ---------------------------------------------------------------------------

def big_layout(fig: go.Figure, title: str, subtitle: str, xtitle: str, ytitle: str) -> go.Figure:
    """Full-width chart: large title, project legend underneath, axis titles."""
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>",
                   x=0, xanchor="left", font=dict(size=22)),
        height=520, margin=dict(l=10, r=10, t=90, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.16, x=0, title=dict(text="Project  "),
                    font=dict(size=13), traceorder="normal"),
        hoverlabel=dict(namelength=-1))
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=14)), showgrid=False, zeroline=False)
    fig.update_yaxes(title=dict(text=ytitle, font=dict(size=14)), showgrid=True, gridcolor=GRID,
                     zeroline=False, rangemode="tozero")
    return fig


PHASE_MEASURES = ["Duration (active weeks)", "When (% of schedule)", "Tasks"]


def chart_phase_compare(x: dict, measure: str) -> go.Figure:
    """Phases on the x axis, one bar per project inside each phase."""
    gpp = x["gpp"]
    phases = phase_sorted(gpp.phase.unique())
    kind = measure.split()[0]
    fig = go.Figure()
    for p in x["scheds"]:
        r = gpp[gpp.project_label == p].set_index("phase").reindex(phases)
        if r.n_tasks.isna().all():
            continue
        custom = np.stack([r.n_tasks.fillna(0), r.active_weeks, r.rel_start * 100,
                           r.rel_end * 100], axis=1)
        hover = (f"<b>{p}</b> · %{{x}}<br>%{{customdata[1]:.0f}} active weeks · "
                 "%{customdata[0]:.0f} tasks<br>%{customdata[2]:.0f}–%{customdata[3]:.0f}% "
                 "of schedule<extra></extra>")
        common = dict(x=phases, name=p, customdata=custom, hovertemplate=hover,
                      marker=dict(color=PROJECT_COLORS.get(p, MUTED), cornerradius=3))
        if kind == "When":
            fig.add_trace(go.Bar(y=(r.rel_end - r.rel_start) * 100, base=r.rel_start * 100, **common))
        else:
            vals = r.active_weeks if kind == "Duration" else r.n_tasks
            fig.add_trace(go.Bar(y=vals, text=[fmt(v) if pd.notna(v) else "" for v in vals],
                                 textposition="outside", textfont=dict(size=11), **common))
    ytitle = {"Duration": "active weeks", "When": "% of schedule window", "Tasks": "tasks"}[kind]
    sub = {"Duration": "Weeks with at least one task of the phase running, per project",
           "When": "Each bar spans the phase's first start to last finish within the project's schedule",
           "Tasks": "Number of scheduled tasks per phase and project"}[kind]
    big_layout(fig, "Phases by project", sub, "phase (in build order)", ytitle)
    fig.update_layout(barmode="group", bargap=0.18, bargroupgap=0.04)
    if kind == "When":
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig


def chart_volume(x: dict) -> go.Figure:
    """Delivered material per schedule, medium-high band (m³)."""
    v = x["vol"].reindex(x["scheds"][::-1])
    ys = [ylabel(p) for p in v.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ys, x=v.high - v.medium, base=v.medium, orientation="h",
                         marker=dict(color=BAND_MED, cornerradius=3), name="medium–high range",
                         hovertemplate="<b>%{y}</b><br>from %{base:,.0f} m³ (+%{x:,.0f} to high)"
                                       "<extra></extra>"))
    fig.add_trace(go.Scatter(y=ys, x=v.high, mode="markers+text", name="high",
                             marker=dict(color=BAND_HIGH, size=9),
                             text=[f" {fmt(h)}" for h in v.high], textposition="middle right",
                             hovertemplate="<b>%{y}</b> high: %{x:,.0f} m³<extra></extra>"))
    fig.add_trace(go.Scatter(y=ys, x=v.medium, mode="markers", name="medium",
                             marker=dict(color=BAND_MED, size=9, line=dict(color=BAND_HIGH, width=1)),
                             hovertemplate="<b>%{y}</b> medium: %{x:,.0f} m³<extra></extra>"))
    base_layout(fig, "Material - delivered volume per schedule (m³)", len(ys))
    fig.update_xaxes(range=[0, float(np.nanmax(v.high)) * 1.22])
    return fig


def chart_trucks(x: dict) -> go.Figure:
    """Trucks per active week: schedules (factor estimate) vs history (observed)."""
    v = x["vol"].reindex(x["scheds"][::-1])
    h = x["hist"]
    ys = [ylabel(p) for p in v.index]
    hy = [f"⏱ {n}" for n in h.hist_name] if len(h) else []
    fig = go.Figure()
    if len(h):
        fig.add_trace(go.Bar(y=hy, x=h.fahrten_per_week, orientation="h",
                             marker=dict(color=OBSERVED, cornerradius=3),
                             name=f"historical, observed ({x['cg']})",
                             customdata=np.stack([h.fahrten, h.active_weeks], axis=1),
                             hovertemplate="<b>%{y}</b><br>%{x:.1f} trips/wk observed<br>"
                                           "%{customdata[0]:.0f} trips over %{customdata[1]:.0f} wk"
                                           "<extra></extra>"))
    fig.add_trace(go.Bar(y=ys, x=v.trips_wk_high - v.trips_wk_med, base=v.trips_wk_med,
                         orientation="h", marker=dict(color=BAND_MED, cornerradius=3),
                         name="schedules, factor estimate",
                         hovertemplate="<b>%{y}</b><br>%{base:.1f} trips/wk medium "
                                       "(+%{x:.1f} to high)<extra></extra>"))
    fig.add_trace(go.Scatter(y=ys, x=v.trips_wk_high, mode="markers", showlegend=False,
                             marker=dict(color=BAND_HIGH, size=8),
                             hovertemplate="<b>%{y}</b> high: %{x:.1f} trips/wk<extra></extra>"))
    base_layout(fig, "Logistics - trucks per active week", len(ys) + len(hy))
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=hy[::-1] + ys)
    return fig


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def weekly_profile() -> pd.DataFrame:
    """Share of each (Gewerk, schedule)'s work falling in each week.

    A task's weight is its working duration, spread evenly over its calendar
    days; the weights are summed per week from the Gewerk's first start and
    normalised so each schedule's profile sums to 1. Multiplying by a total
    (m³, trucks, BGF) gives its distribution over time.
    """
    t = TASKS.dropna(subset=["start", "finish"]).copy()
    span = t["span_days"].clip(lower=1).astype(int)
    t["w_day"] = t["duration_days"].fillna(span).clip(lower=1) / span
    days = t.loc[t.index.repeat(span), ["gewerk_label", "project_label", "start", "w_day"]]
    days["date"] = days["start"] + pd.to_timedelta(days.groupby(level=0).cumcount(), unit="D")
    g0 = days.groupby(["gewerk_label", "project_label"])["date"].transform("min")
    days["week"] = (days["date"] - g0).dt.days // 7
    days["g0"] = g0
    prof = (days.groupby(["gewerk_label", "project_label", "week"])
            .agg(w=("w_day", "sum"), g0=("g0", "first")).reset_index())
    prof["share"] = prof["w"] / prof.groupby(["gewerk_label", "project_label"])["w"].transform("sum")
    frame = PROJ.set_index("project_label")
    p_start = prof["project_label"].map(frame["start"])
    p_len = (prof["project_label"].map(frame["finish"]) - p_start).dt.days.clip(lower=1)
    mid = prof["g0"] + pd.to_timedelta(prof["week"] * 7 + 3.5, unit="D")
    prof["pct"] = (mid - p_start).dt.days / p_len * 100
    return prof.drop(columns=["w"])


# One fixed colour per schedule (categorical palette, schedule order = start
# date), so a schedule keeps its colour on every card.
PROJECT_COLORS = dict(zip(PROJECTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))


FLOW_METRICS = {"m³ per week": ("m³ per week", "m³/wk"), "LKW per week": ("trucks per week", "LKW/wk"),
                "m² BGF per week": ("m² BGF per week", "m²/wk"), "Cumulative %": ("cumulative % done", "%")}
FLOW_AXES = ["Weeks from Gewerk start", "Calendar date", "% of schedule"]


def chart_over_time(x: dict, g: str, metric: str, axis: str) -> go.Figure:
    """One line per project: how the Gewerk's work flows over time."""
    prof = weekly_profile()
    prof = prof[prof.gewerk_label == g]
    v = x["vol"]
    eff_load = (x["load_m3"] or np.nan) * ga.LKW_UTILIZATION
    ytitle, unit = FLOW_METRICS[metric]
    kind = axis.split()[0]
    fig = go.Figure()
    for p in x["scheds"]:
        r = prof[prof.project_label == p].sort_values("week")
        if r.empty:
            continue
        # Fill gap weeks with zero, then a centred 4-week mean so schedules with
        # many short tasks stay comparable with ones planned in long blocks.
        g0 = r.g0.iloc[0]
        weeks = np.arange(r.week.min(), r.week.max() + 1)
        r = (r.set_index("week").reindex(weeks)
             .assign(share=lambda d: d.share.fillna(0), pct=lambda d: d.pct.interpolate())
             .reset_index().rename(columns={"index": "week"}))
        r["date"] = g0 + pd.to_timedelta(r.week * 7, unit="D")
        if metric != "Cumulative %":
            r["share"] = r.share.rolling(4, center=True, min_periods=1).mean()
        if metric == "m³ per week":
            y = r.share * v.loc[p, "high"]
        elif metric == "LKW per week":
            y = r.share * v.loc[p, "high"] / eff_load
        elif metric == "m² BGF per week":
            y = r.share * SIZES[p]["bgf"]
        else:
            y = r.share.cumsum() * 100
        xs = {"Weeks": r.week, "Calendar": r.date, "%": r.pct}[kind]
        xfmt = {"Weeks": "week %{x:.0f}", "Calendar": "%{x|%d %b %Y}",
                "%": "%{x:.0f}% of schedule"}[kind]
        fig.add_trace(go.Scatter(
            x=xs, y=y, mode="lines", name=ylabel(p),
            line=dict(color=PROJECT_COLORS.get(p, MUTED), width=2.5),
            hovertemplate=f"<b>{p}</b><br>{xfmt}<br>%{{y:,.1f}} {unit}<extra></extra>"))
    sub = "4-week rolling mean" if metric != "Cumulative %" else "share of the Gewerk's work completed"
    if metric in ("m³ per week", "LKW per week"):
        sub += " · high-intensity factors · * = assumed project size"
    big_layout(fig, f"{g} - {ytitle} over time", sub,
               {"Weeks": "weeks from Gewerk start", "Calendar": "date", "%": "% of schedule"}[kind],
               ytitle)
    fig.update_layout(hovermode="x unified" if kind == "Weeks" else "closest")
    if kind == "%":
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
    if metric == "Cumulative %":
        fig.update_yaxes(range=[0, 102], ticksuffix="%")
    return fig


def render_card(g: str) -> None:
    x = gewerk_frame(g)
    gp, f, v, h = x["gp"], x["f"], x["vol"], x["hist"]
    aw = gp.active_weeks
    spread = aw.max() / aw.min() if len(aw) > 1 and aw.min() > 0 else np.nan
    has_factor = f is not None and f["high_total_m3_per_m2"] > 0
    m2 = (f"{f['medium_total_m3_per_m2']:.3f}–{f['high_total_m3_per_m2']:.3f}"
          if has_factor else "–")
    phases = phase_sorted(x["gpp"].phase.unique())

    with st.container(border=True):
        st.markdown(
            f"### {g}\n<span style='color:{MUTED}'>{CATEGORY.get(g, '')} · "
            f"{len(x['scheds'])} of {N_PROJ} schedules · {int(gp.n_tasks.sum()):,} tasks · "
            f"phases: {' → '.join(phases)}</span>", unsafe_allow_html=True)
        st.markdown(stat_row([
            ("Duration spread", f"{fmt(aw.min())}–{fmt(aw.max())} wk",
             f"×{spread:.1f} · longest: {aw.idxmax()}" if not np.isnan(spread) else "single schedule"),
            ("Typical window",
             f"{fmt(gp.rel_start.median() * 100)}–{fmt(gp.rel_end.median() * 100)} %",
             "median start–end within schedule"),
            ("Median task", f"{fmt(x['tasks'].duration_days.median())} d",
             f"up to {int(gp.peak_parallel_tasks.max())} tasks in parallel"),
            ("Material per m² BGF", f"{m2} m³", "delivered, medium–high"),
            ("Volume per schedule", f"{fmt(v.high.median())} m³" if has_factor else "–",
             "median, high case"),
            ("Historical trucks/wk",
             fmt(h.fahrten_per_week.median(), 1) if len(h) else "–",
             f"observed median · {x['cg']}"),
        ]), unsafe_allow_html=True)

        measure = st.radio("Phase chart shows", PHASE_MEASURES, horizontal=True, key=f"pm-{g}")
        st.plotly_chart(chart_phase_compare(x, measure), key=f"pc-{g}", **STRETCH)

        metrics = (["m³ per week", "LKW per week"] if has_factor else []) + \
                  ["m² BGF per week", "Cumulative %"]
        k1, k2 = st.columns([3, 2])
        metric = k1.radio("Flow chart shows", metrics, horizontal=True, key=f"tm-{g}")
        axis = k2.radio("X axis", FLOW_AXES, horizontal=True, key=f"ta-{g}")
        st.plotly_chart(chart_over_time(x, g, metric, axis), key=f"ot-{g}", **STRETCH)

        if has_factor:
            with st.expander("More: total volume and trucks vs historical projects"):
                c3, c4 = st.columns(2)
                c3.plotly_chart(chart_volume(x), key=f"vo-{g}", **STRETCH)
                c4.plotly_chart(chart_trucks(x), key=f"tr-{g}", **STRETCH)
                note = (f"Volume = BuildSync material factors × schedule BGF (* = assumed size). "
                        f"Trucks = volume ÷ ({x['veh']}, {fmt(x['load_m3'])} m³ × "
                        f"{ga.LKW_UTILIZATION:.0%}) ÷ active weeks. ⏱ rows: observed deliveries of "
                        f"the whole {x['cg']} group on historical projects.")
                if bool(f["uses_unit_driver"]):
                    note += " Count-based drivers (lifts, doors, kitchens) scale linearly - indicative only."
                st.caption(note)
        else:
            st.caption("No material factor for this Gewerk in the BuildSync factor sheet "
                       "(deliberately zero there to avoid double counting), so the flow chart "
                       "shows area and progress only.")

        with st.expander("Tasks and material components"):
            t1, t2 = st.tabs(["Tasks", "Material components"])
            t1.dataframe(x["tasks"].sort_values(["project_label", "start"])[
                ["project_label", "phase", "task_name", "start", "finish", "duration_days",
                 "geschoss", "unternehmer"]], hide_index=True, height=320, **STRETCH)
            comp = COMP[COMP.gewerk.isin(ga.FINE_TO_FACTOR.get(g, [g]))]
            t2.dataframe(comp[["component", "flow", "driver", "unit", "medium_effective_factor",
                               "high_effective_factor", "confidence", "notes"]],
                         hide_index=True, **STRETCH)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Gewerke across all schedules")
st.caption(
    f"{N_PROJ} schedules · {len(TASKS):,} classified tasks · {len(GEWERKE)} Gewerke. "
    "Each card compares the schedules that contain that Gewerk: where it sits in the "
    "programme, how long its phases run, how much material it moves and how many trucks "
    "per week that implies, next to observed deliveries on historical projects. "
    f"Factors: {D['meta'].get('status', '')}.")

# All Gewerke, the ones found in the most schedules first.
sel = GEWERKE.sort_values(["n_proj", "n_tasks"], ascending=False)

legend = " &nbsp; ".join(f"<span style='color:{PROJECT_COLORS[p]}'>■</span> {ylabel(p)}"
                         for p in PROJECTS)
st.markdown(f"<div style='font-size:0.9rem'>Project colours (same in every chart): {legend}"
            f"<br><span style='color:{MUTED}'>* project size not known - volumes assume "
            f"{ASSUMED_BGF:,.0f} m² BGF</span></div>".replace(",", "'"),
            unsafe_allow_html=True)

# Five charts per card; paging keeps the browser from drawing ~200 at once.
PER_PAGE = 5
n_pages = max(1, -(-len(sel) // PER_PAGE))
if n_pages > 1:
    names = list(sel.index)
    page = st.radio(f"{len(sel)} Gewerke, {PER_PAGE} per page", list(range(1, n_pages + 1)),
                    horizontal=True, key="page",
                    format_func=lambda i: f"{i}: {names[(i - 1) * PER_PAGE].split('/')[0]} …")
else:
    page = 1
for g in sel.index[(page - 1) * PER_PAGE:page * PER_PAGE]:
    render_card(g)
