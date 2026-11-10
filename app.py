"""Beyond the Algorithm — Streamlit dashboard.

Reads only the small precomputed tables in data/processed/ (built by
scripts/prepare_data.py), so it starts quickly and deploys without raw data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import TIER_LABELS
from src.metrics import ni_sample_size_per_arm

DATA = Path(__file__).parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------
BG, SURFACE, SURFACE_2, BORDER = "#0B0C10", "#14161D", "#1B1E27", "#262A36"
TEXT, MUTED, FAINT = "#ECEDEF", "#A3A9B6", "#5F6677"
ACCENT = "#2DD4BF"
STRATEGY_COLORS = {
    "Engagement-first": "#7AA7FF",
    "Balanced discovery": "#2DD4BF",
    "Diversity-first": "#F2B35B",
    "Custom": "#C4A1FF",
    "Random": "#5F6677",
}
TIER_COLORS = ["#F07A6A", "#F2B35B", "#2DD4BF", "#7AA7FF"]
TIER_SHORT = {0: "Least likely", 1: "Below median", 2: "Above median", 3: "Most likely"}

st.set_page_config(page_title="Beyond the Algorithm", page_icon="🎧", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 500px at 85% -10%, rgba(45,212,191,0.07), transparent 60%), {BG}; }}
.block-container {{ padding: 3.6rem clamp(1rem, 4vw, 3.5rem) 3rem; max-width: 1440px; }}
h1, h2, h3, h4, h5 {{ letter-spacing: -0.015em; }}
.serif {{ font-family: 'Instrument Serif', Georgia, serif; font-weight: 400; letter-spacing: -0.01em; }}

/* masthead */
.mast {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  border-bottom: 1px solid {BORDER}; padding-bottom: 14px; margin-bottom: 6px; }}
.mast-title {{ font-family: 'Instrument Serif', Georgia, serif; font-size: clamp(2rem, 4vw, 2.9rem); line-height: 1;
  color: {TEXT}; margin: 0; }}
.mast-sub {{ color: {MUTED}; font-size: 0.98rem; margin-top: 6px; }}
.mast-meta {{ color: {FAINT}; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; text-align: right; }}
.eq {{ display: inline-flex; align-items: flex-end; gap: 3px; height: 16px; margin-right: 10px; vertical-align: -1px; }}
.eq span {{ width: 3px; background: {ACCENT}; border-radius: 2px; animation: eq 1.2s ease-in-out infinite; }}
.eq span:nth-child(1) {{ height: 40%; animation-delay: -0.2s; }}
.eq span:nth-child(2) {{ height: 90%; animation-delay: -0.5s; }}
.eq span:nth-child(3) {{ height: 60%; animation-delay: -0.9s; }}
.eq span:nth-child(4) {{ height: 75%; animation-delay: -0.3s; }}
@keyframes eq {{ 0%,100% {{ transform: scaleY(0.45); }} 50% {{ transform: scaleY(1); }} }}
.eq span {{ transform-origin: bottom; }}
@media (prefers-reduced-motion: reduce) {{ .eq span {{ animation: none; }} }}

/* executive summary */
.kicker {{ color: {ACCENT}; font-size: 0.76rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; }}
.lede {{ font-family: 'Instrument Serif', Georgia, serif; font-size: clamp(1.9rem, 3.4vw, 2.85rem); line-height: 1.12;
  color: {TEXT}; margin: 6px 0 10px 0; max-width: 980px; }}
.lede em {{ color: {ACCENT}; font-style: normal; }}
.lede-sub {{ color: {MUTED}; font-size: 1.05rem; max-width: 900px; line-height: 1.55; }}
.stat {{ border-top: 2px solid var(--c, {ACCENT}); padding: 14px 2px 4px; height: 100%; }}
.stat-label {{ color: {MUTED}; font-size: 0.82rem; font-weight: 500; }}
.stat-value {{ font-family: 'Instrument Serif', Georgia, serif; font-size: clamp(2.3rem, 4vw, 3.1rem); line-height: 1.05;
  color: {TEXT}; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.stat-note {{ color: {FAINT}; font-size: 0.8rem; margin-top: 4px; line-height: 1.4; }}
.decision {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 3px solid {ACCENT}; border-radius: 10px;
  padding: 18px 22px; }}
.decision h4 {{ margin: 0 0 6px 0; font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: {ACCENT}; }}
.decision p {{ margin: 0 0 8px 0; color: {TEXT}; line-height: 1.55; }}
.essay p {{ color: {MUTED}; line-height: 1.7; font-size: 0.98rem; }}
.essay p:first-of-type::first-letter {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 3.1rem; float: left;
  line-height: 0.9; padding: 4px 8px 0 0; color: {TEXT}; }}

/* cards */
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
.kpi-label {{ color: {MUTED}; font-size: 0.8rem; font-weight: 500; }}
.kpi-label abbr {{ text-decoration: none; cursor: help; color: {FAINT}; margin-left: 4px; }}
.kpi-value {{ color: {TEXT}; font-size: 1.65rem; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.kpi-delta {{ font-size: 0.84rem; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.pos {{ color: #5EE6A1; }} .neg {{ color: #F07A6A; }} .neu {{ color: {MUTED}; }}
.badge {{ display: inline-block; font-size: 0.72rem; padding: 2px 9px; border-radius: 999px; margin-right: 6px;
  border: 1px solid {BORDER}; color: {MUTED}; white-space: nowrap; }}
.badge.obs {{ border-color: #5EE6A1; color: #5EE6A1; }}
.badge.mod {{ border-color: #F2B35B; color: #F2B35B; }}
.badge.ours {{ border-color: #C4A1FF; color: #C4A1FF; }}
.provenance {{ color: {FAINT}; font-size: 0.82rem; line-height: 2; }}
.section-title {{ font-family: 'Instrument Serif', Georgia, serif; font-size: 1.7rem; color: {TEXT}; margin: 8px 0 2px 0; }}
.section-sub {{ color: {MUTED}; font-size: 0.92rem; margin-bottom: 8px; }}
.muted {{ color: {MUTED}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; overflow-x: auto; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 14px; white-space: nowrap; }}
div[data-testid="stExpander"] details {{ border-color: {BORDER}; background: {SURFACE}; }}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 400, legend_below: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        height=height,
        margin=dict(l=8, r=8, t=36, b=120 if legend_below else 8),
        legend=(dict(orientation="h", yanchor="top", y=-0.26, x=0, bgcolor="rgba(0,0,0,0)") if legend_below else
                dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, title_font=dict(color=MUTED))
    return fig


def kpi(label: str, value: str, delta: str | None = None, tone: str = "neu", tip: str | None = None) -> str:
    t = f'<abbr title="{tip}">ⓘ</abbr>' if tip else ""
    d = f'<div class="kpi-delta {tone}">{delta}</div>' if delta else ""
    return f'<div class="kpi"><div class="kpi-label">{label}{t}</div><div class="kpi-value">{value}</div>{d}</div>'


def kpi_row(cards: list[str]) -> None:
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def signed(x: float, digits: int = 1, suffix: str = "%") -> str:
    return f"{x:+.{digits}f}{suffix}".replace("-", "−")


def tone(x: float, good_if_positive: bool = True, eps: float = 0.05) -> str:
    if abs(x) < eps:
        return "neu"
    return "pos" if (x > 0) == good_if_positive else "neg"


TIPS = {
    "engagement": "Modeled probability that a listener streams at least one of the 12 carousel playlists: "
                  "1 − ∏(1 − p) using Deezer's published model. An expected value, not observed streams.",
    "first_view": "Same, using only the first 3 cards, which are visible without scrolling.",
    "placed": "Catalog coverage: share of the 862 playlists assigned to at least one carousel slot. "
              "A placement is not a confirmed view.",
    "first_screen": "Share of the 862 playlists that appear in the first 3 cards of at least one carousel.",
    "variety": "Effective catalog size: exp(Shannon entropy) of placement shares. The number of equally "
               "exposed playlists that would give the same spread.",
    "lambda": "λ in score = log p − λ · log(1 + placements so far / even share). 0 = pure engagement ranking.",
    "segment": "Quartiles of each listener's modeled engagement under engagement-first ranking.",
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load():
    meta = json.loads((DATA / "meta.json").read_text())
    curve = pd.read_csv(DATA / "curve.csv")
    tiers = pd.read_csv(DATA / "tiers.csv")
    clusters = pd.read_csv(DATA / "clusters.csv")
    exposure = pd.read_csv(DATA / "exposure.csv")
    order = pd.read_csv(DATA / "order_sensitivity.csv")

    metrics = ["expected_reward", "initial_view_reward", "slot_stream_rate", "coverage", "effective_catalog",
               "gini", "top10_share", "fv_coverage", "fv_effective_catalog", "reward_change_pct"]
    ea = curve[curve["strategy"] == "Exposure-aware"]
    summary = ea.groupby("lambda")[metrics].agg(["mean", "min", "max"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    base = summary.iloc[0]
    for col in ["initial_view_reward", "effective_catalog", "coverage", "fv_coverage", "fv_effective_catalog"]:
        summary[f"{col}_chg"] = 100 * (summary[f"{col}_mean"] / base[f"{col}_mean"] - 1)
    summary["reward_change_pts"] = 100 * (summary["expected_reward_mean"] - base["expected_reward_mean"])
    rnd = curve[curve["strategy"] == "Random"][metrics].mean()

    tiers = tiers[tiers["strategy"] == "Exposure-aware"].copy()
    tiers["rel"] = 100 * (tiers["expected_reward"] / tiers["baseline_reward"] - 1)
    tiers["pts"] = 100 * (tiers["expected_reward"] - tiers["baseline_reward"])
    tiers["fv_rel"] = 100 * (tiers["initial_view_reward"] / tiers["baseline_initial_view_reward"] - 1)
    tiers["fv_pts"] = 100 * (tiers["initial_view_reward"] - tiers["baseline_initial_view_reward"])
    tier_summary = (tiers.groupby(["lambda", "tier"])
                    .agg(expected_reward=("expected_reward", "mean"), baseline_reward=("baseline_reward", "mean"),
                         rel=("rel", "mean"), rel_min=("rel", "min"), rel_max=("rel", "max"),
                         pts=("pts", "mean"), pts_min=("pts", "min"), pts_max=("pts", "max"),
                         fv_rel=("fv_rel", "mean"), fv_pts=("fv_pts", "mean"),
                         coverage=("coverage", "mean"), effective_catalog=("effective_catalog", "mean"))
                    .reset_index())
    return meta, summary, rnd, tier_summary, clusters, exposure, order


meta, summary, rnd, tier_summary, clusters, exposure, order = load()
PRESETS: dict[str, float] = meta["presets"]
LAMBDAS: list[float] = meta["lambdas"]
N_PLAYLISTS, N_SLOTS, L_INIT = meta["n_playlists"], meta["n_slots"], meta["l_init"]
N_SAMPLE, N_REPS = meta["users_per_replicate"], meta["n_replicates"]
BUDGETS = meta["loss_budgets_pct"]


def row(lam: float) -> pd.Series:
    return summary.loc[np.isclose(summary["lambda"], lam)].iloc[0]


def tier_row(lam: float, tier: int) -> pd.Series:
    t = tier_summary
    return t[np.isclose(t["lambda"], lam) & (t["tier"] == tier)].iloc[0]


BASE, BAL, DIV = row(PRESETS["Engagement-first"]), row(PRESETS["Balanced discovery"]), row(PRESETS["Diversity-first"])
LAM_BAL, LAM_DIV = PRESETS["Balanced discovery"], PRESETS["Diversity-first"]
Q1_BAL, Q4_BAL, Q1_DIV = tier_row(LAM_BAL, 0), tier_row(LAM_BAL, 3), tier_row(LAM_DIV, 0)
SPREAD = max(r.reward_change_pct_max - r.reward_change_pct_min for r in (BAL, DIV))


def placements_for(lam: float, first_view: bool = False) -> pd.DataFrame:
    col = "first_view_placements" if first_view else "placements"
    e = exposure[(exposure["strategy"] == "Exposure-aware") & np.isclose(exposure["lambda"], lam)]
    return e[["playlist", col]].rename(columns={col: "n"}).set_index("playlist")["n"]


# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    f"""<div class="mast">
  <div><div class="mast-title"><span class="eq"><span></span><span></span><span></span><span></span></span>Beyond the Algorithm</div>
  <div class="mast-sub">The engagement–discovery tradeoff in playlist recommendations</div></div>
  <div class="mast-meta">Product analytics case study<br>Deezer public data · modeled outcomes</div>
</div>""",
    unsafe_allow_html=True,
)

tab_sum, tab_explore, tab_tradeoff, tab_reco, tab_ab, tab_method = st.tabs(
    ["Summary", "Explore strategies", "The tradeoff", "Recommendation", "A/B test plan", "Methodology"]
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
with tab_sum:
    st.markdown('<div class="kicker" style="margin-top:18px">The question</div>', unsafe_allow_html=True)
    st.markdown(
        f"""<div class="lede">Can a playlist carousel spread its recommendations across far more of the catalog
without lowering predicted engagement?<br><em>Across all 12 positions, yes. In the 3 initially visible positions, barely.</em></div>
<div class="lede-sub">A diversity-aware ranker doubles the share of the catalog placed in carousels for a
{abs(BAL.reward_change_pct_mean):.1f}% drop in predicted engagement. Almost all of that added variety sits in
cards 4–12, and the cost falls on listeners who were already least likely to stream.</div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    s0, s1, s2, s3 = st.columns(4, gap="large")
    s0.markdown(
        f"""<div class="stat" style="--c:{STRATEGY_COLORS['Balanced discovery']}">
<div class="stat-label">Catalog placed, all 12 cards</div>
<div class="stat-value">{BASE.coverage_mean:.1%} → {BAL.coverage_mean:.1%}</div>
<div class="stat-note">share of {N_PLAYLISTS} playlists in at least one carousel, engagement-first →
balanced</div></div>""", unsafe_allow_html=True)
    s1.markdown(
        f"""<div class="stat" style="--c:{STRATEGY_COLORS['Diversity-first']}">
<div class="stat-label">Catalog placed, first 3 cards</div>
<div class="stat-value">{BASE.fv_coverage_mean:.1%} → {BAL.fv_coverage_mean:.1%}</div>
<div class="stat-note">share of {N_PLAYLISTS} playlists placed at least once in the initially visible
positions</div></div>""",
        unsafe_allow_html=True)
    s2.markdown(
        f"""<div class="stat" style="--c:{STRATEGY_COLORS['Engagement-first']}">
<div class="stat-label">Predicted engagement</div>
<div class="stat-value">{signed(BAL.reward_change_pct_mean)}</div>
<div class="stat-note">{signed(BAL.reward_change_pts, 1, ' pts')}: modeled chance of streaming any of the 12
cards, {BASE.expected_reward_mean:.1%} → {BAL.expected_reward_mean:.1%}</div></div>""", unsafe_allow_html=True)
    s3.markdown(
        f"""<div class="stat" style="--c:{TIER_COLORS[0]}">
<div class="stat-label">Cost to listeners least likely to stream</div>
<div class="stat-value">{signed(Q1_BAL.rel)}</div>
<div class="stat-note">{signed(Q1_BAL.pts, 1, ' pts')} for the bottom quartile, versus
{signed(Q4_BAL.pts, 2, ' pts')} for the top quartile</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f"""<div class="decision" style="border-left-color:{STRATEGY_COLORS['Diversity-first']}">
<h4 style="color:{STRATEGY_COLORS['Diversity-first']}">Why the first three cards matter</h4>
<p>Under balanced discovery, <b>{BAL.fv_coverage_mean:.1%}</b> of the catalog is placed at least once in the first
three positions, the ones visible without scrolling. Across all 12 positions, <b>{BAL.coverage_mean:.1%}</b> is placed
at least once. Both are <i>recommendation-placement</i> metrics: they count what the ranker assigns, not what
anyone viewed, streamed or discovered. The dataset has no impression or scroll data, so actual viewed exposure is
unknown. It has to be measured in an online experiment.</p>
<p>Most of the added variety goes to positions 4–12. The low engagement cost and the small first-screen change have
the same cause: each carousel still leads with the listener's strongest options. Whether the extra placements lead
to discovery depends on whether listeners view and stream them, which only an experiment can show.</p></div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    v1, v2 = st.columns([1.75, 1], gap="large")
    with v1:
        section(f"Every square is one of the {N_PLAYLISTS} playlists",
                "Which playlists make it into anyone's carousel, in one sample of "
                f"{N_SAMPLE:,} listeners")
        view = st.radio("Slots", ["All 12 cards", "First 3 cards (no scrolling)"], horizontal=True,
                        key="grid_view", label_visibility="collapsed")
        fv = view == "First 3 cards (no scrolling)"
        b, nb = placements_for(0.0, fv), placements_for(LAM_BAL, fv)
        status = np.where(b > 0, 2, np.where(nb > 0, 1, 0))
        order_idx = np.lexsort((-nb.values, -b.values, -status))
        cols = 48
        rows_n = math.ceil(N_PLAYLISTS / cols)
        z = np.full(rows_n * cols, np.nan)
        z[:N_PLAYLISTS] = status[order_idx]
        hover = np.full(rows_n * cols, "", dtype=object)
        hover[:N_PLAYLISTS] = [f"Playlist #{p}<br>Engagement-first: {b.iloc[p]:,} placements<br>"
                               f"Balanced: {nb.iloc[p]:,} placements" for p in b.index[order_idx]]
        grid = go.Figure(go.Heatmap(
            z=z.reshape(rows_n, cols), text=hover.reshape(rows_n, cols), hoverinfo="text",
            colorscale=[[0, SURFACE_2], [0.33, SURFACE_2], [0.33, STRATEGY_COLORS["Balanced discovery"]],
                        [0.66, STRATEGY_COLORS["Balanced discovery"]], [0.66, STRATEGY_COLORS["Engagement-first"]],
                        [1, STRATEGY_COLORS["Engagement-first"]]],
            zmin=0, zmax=2, showscale=False, xgap=3, ygap=3,
        ))
        grid.update_xaxes(visible=False)
        grid.update_yaxes(visible=False, autorange="reversed")
        style_fig(grid, 320).update_layout(margin=dict(l=0, r=0, t=4, b=0))
        st.plotly_chart(grid, config={"displayModeBar": False})
        n_b, n_new = int((status == 2).sum()), int((status == 1).sum())
        st.markdown(
            f"""<div class="provenance">
<span style="color:{STRATEGY_COLORS['Engagement-first']}">■</span> {n_b} placed by engagement-first &nbsp;
<span style="color:{STRATEGY_COLORS['Balanced discovery']}">■</span> {n_new} added by balanced discovery &nbsp;
<span style="color:{FAINT}">■</span> {N_PLAYLISTS - n_b - n_new} placed for no one</div>""",
            unsafe_allow_html=True)
    with v2:
        st.write("")
        st.markdown(
            f"""<div class="decision"><h4>Proposed decision</h4>
<p><b>Test balanced discovery (λ = {LAM_BAL:g}) in an online A/B test</b> against today's ranking, with a
{BUDGETS['Balanced discovery']:.0f}% non-inferiority margin on streaming from each user's first carousel.</p>
<p>Log which cards are actually viewed, since the added variety mostly sits in positions 4–12. Pre-register the
bottom-quartile listeners as a guardrail segment.</p>
<p style="margin:0;color:{MUTED}">Don't make the aggressive setting (λ = {LAM_DIV:g}, {signed(DIV.reward_change_pct_mean)})
the home-screen default.</p></div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """<div class="provenance">
<span class="badge obs">Observed</span>anonymized Deezer user & playlist features<br>
<span class="badge mod">Modeled</span>stream probabilities from Deezer's published simulator<br>
<span class="badge ours">Ours</span>ranking strategies, metrics, segments, recommendation
</div>""", unsafe_allow_html=True)

    st.write("")
    e1, e2 = st.columns([1.15, 1], gap="large")
    with e1:
        section("Why discovery is worth measuring")
        st.markdown(
            """<div class="essay">
<p>A recommendation carousel answers a narrow question well: what is this listener most likely to play right now?
Asked of every listener at once, that question has a side effect. The same few playlists win almost everywhere,
and most of the catalog never gets a slot.</p>
<p>That matters for reasons a single session cannot show. A recommender learns only from what it shows, so
playlists it never places stay unknown to it. Listeners whose tastes sit outside the mainstream may find fewer
good matches, and the people who make the music depend on being heard at all. Whether broader exposure helps on
these fronts is an open question. It is the kind of question an experiment, not this offline analysis, has to
answer.</p>
<p>What this analysis can do is put a price on the first step: how much predicted engagement each unit of
broader exposure costs, and who pays.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with e2:
        section("How to read this")
        st.markdown(
            f"""
- **Engagement-first** gives each listener their 12 most likely playlists.
- **Balanced discovery** and **Diversity-first** use our ranker, which discounts playlists that are already
  over-exposed. One setting, λ, controls how strongly.
- All figures are averages over {N_REPS} samples of {N_SAMPLE:,} real (anonymized) listener profiles.
  Engagement is **predicted** by Deezer's model, not observed.
"""
        )
        st.caption("Data: Bendada, Salha & Bontempelli, RecSys 2020 · Zenodo 10.5281/zenodo.4048678 · CC BY 4.0. "
                   "Independent project, not affiliated with Deezer.")

# ---------------------------------------------------------------------------
# Explore strategies
# ---------------------------------------------------------------------------
with tab_explore:
    st.write("")
    c1, c2, c3 = st.columns([1.3, 1.4, 1.1], gap="large")
    with c1:
        choice = st.radio("Strategy", list(PRESETS) + ["Custom"], index=1, horizontal=True)
    with c2:
        lam_default = PRESETS.get(choice, LAM_BAL)
        lam = st.select_slider("Discovery weight (λ)", options=LAMBDAS, value=lam_default,
                               disabled=choice != "Custom", help=TIPS["lambda"])
        if choice != "Custom":
            lam = lam_default
    with c3:
        seg_options = ["All listeners"] + [TIER_LABELS[t] for t in range(4)]
        segment = st.selectbox("Listeners", seg_options, help=TIPS["segment"])

    sel = row(lam)
    color = STRATEGY_COLORS.get(choice, STRATEGY_COLORS["Custom"])
    if segment == "All listeners":
        reward, base_reward = sel.expected_reward_mean, BASE.expected_reward_mean
        fv_rel = sel.initial_view_reward_chg
        cov, base_cov = sel.coverage_mean, BASE.coverage_mean
    else:
        t = seg_options.index(segment) - 1
        tr, tb = tier_row(lam, t), tier_row(0.0, t)
        reward, base_reward = tr.expected_reward, tr.baseline_reward
        fv_rel = tr.fv_rel
        cov, base_cov = tr.coverage, tb.coverage
    rel, pts = 100 * (reward / base_reward - 1), 100 * (reward - base_reward)

    st.write("")
    kpi_row([
        kpi("Predicted engagement", f"{reward:.1%}", f"{signed(rel)} · {signed(pts, 1, ' pts')}", tone(rel),
            TIPS["engagement"]),
        kpi("…from the first 3 cards", signed(fv_rel, 2), "vs engagement-first", tone(fv_rel), TIPS["first_view"]),
        kpi("Catalog placed", f"{cov:.0%}", f"from {base_cov:.0%}", tone(cov - base_cov, eps=0.001), TIPS["placed"]),
        kpi("Catalog on the first screen", f"{sel.fv_coverage_mean:.1%}",
            f"from {BASE.fv_coverage_mean:.1%} · all listeners", tone(sel.fv_coverage_mean - BASE.fv_coverage_mean,
                                                                     eps=0.001), TIPS["first_screen"]),
    ])
    st.write("")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        section("How concentrated are the placements?", "Cumulative share of placements, playlists ranked "
                "from most to least placed")
        fig = go.Figure()
        for name, l, c in [("Engagement-first", 0.0, STRATEGY_COLORS["Engagement-first"]),
                           (f"Selected (λ = {lam:g})", lam, color)]:
            for fvw, dash in [(False, "solid"), (True, "dot")]:
                sh = np.sort(placements_for(l, fvw).values)[::-1]
                fig.add_scatter(x=np.arange(1, N_PLAYLISTS + 1), y=np.cumsum(sh) / sh.sum(), mode="lines",
                                name=f"{name} · {'first 3 cards' if fvw else 'all 12'}",
                                line=dict(width=2.5 if not fvw else 1.8, color=c, dash=dash))
        fig.update_xaxes(title="Playlists (log scale)", type="log", range=[0, math.log10(N_PLAYLISTS)])
        fig.update_yaxes(title="Share of placements", tickformat=".0%", range=[0, 1.02])
        st.plotly_chart(style_fig(fig, 400, legend_below=True))
    with g2:
        section("Who pays?", "Change in predicted engagement by listener group")
        unit = st.radio("Unit", ["Relative %", "Percentage points"], horizontal=True,
                                    key="bar_unit", label_visibility="collapsed")
        k, kmin, kmax, suffix = (("rel", "rel_min", "rel_max", "%") if unit != "Percentage points"
                                 else ("pts", "pts_min", "pts_max", " pts"))
        bars = go.Figure()
        for name, l in [("Balanced discovery", LAM_BAL), ("Diversity-first", LAM_DIV)] + (
                [("Custom", lam)] if choice == "Custom" else []):
            ts = tier_summary[np.isclose(tier_summary["lambda"], l)].sort_values("tier")
            bars.add_bar(
                x=[TIER_SHORT[t] for t in ts["tier"]], y=ts[k], name=f"{name} (λ = {l:g})",
                marker_color=STRATEGY_COLORS[name],
                error_y=dict(type="data", symmetric=False, array=ts[kmax] - ts[k], arrayminus=ts[k] - ts[kmin],
                             color=MUTED, thickness=1),
                hovertemplate="%{x}: %{y:.2f}" + suffix + "<extra>" + name + "</extra>",
            )
        bars.update_layout(barmode="group")
        bars.update_xaxes(title="Listeners grouped by likelihood to stream (quartiles)")
        bars.update_yaxes(title=f"Change vs engagement-first ({suffix.strip()})", ticksuffix=suffix)
        st.plotly_chart(style_fig(bars, 360, legend_below=True))

    with st.expander("Full comparison table and definitions"):
        table = pd.DataFrame([
            {"Strategy": n, "λ": l, "Predicted engagement": r.expected_reward_mean,
             "Change": r.reward_change_pct_mean / 100, "Change (pts)": r.reward_change_pts,
             "First 3 cards": r.initial_view_reward_mean, "Catalog placed": r.coverage_mean,
             "First-screen catalog": r.fv_coverage_mean, "Effective variety": r.effective_catalog_mean,
             "Top-10% share": r.top10_share_mean, "Gini": r.gini_mean}
            for n, l, r in [("Engagement-first", 0.0, BASE), ("Balanced discovery", LAM_BAL, BAL),
                            ("Diversity-first", LAM_DIV, DIV)]
        ] + [{"Strategy": "Random (reference)", "λ": None, "Predicted engagement": rnd.expected_reward,
              "Change": rnd.reward_change_pct / 100,
              "Change (pts)": 100 * (rnd.expected_reward - BASE.expected_reward_mean),
              "First 3 cards": rnd.initial_view_reward, "Catalog placed": rnd.coverage,
              "First-screen catalog": rnd.fv_coverage, "Effective variety": rnd.effective_catalog,
              "Top-10% share": rnd.top10_share, "Gini": rnd.gini}])
        st.dataframe(
            table.style.format({"Predicted engagement": "{:.1%}", "Change": "{:+.1%}", "Change (pts)": "{:+.1f}",
                                "First 3 cards": "{:.1%}", "Catalog placed": "{:.0%}",
                                "First-screen catalog": "{:.1%}", "Effective variety": "{:.0f}",
                                "Top-10% share": "{:.0%}", "Gini": "{:.2f}",
                                "λ": lambda v: "—" if v is None or pd.isna(v) else f"{v:g}"}),
            hide_index=True, width="stretch",
        )
        st.markdown(
            f"""
- **Predicted engagement:** {TIPS['engagement']}
- **Catalog placed / first-screen catalog:** share of {N_PLAYLISTS} playlists with ≥1 placement in any slot, or
  in slots 1–3, across one {N_SAMPLE:,}-listener sample. For a listener group, it is computed within that group.
- **Effective variety:** {TIPS['variety']}
- Whiskers on the bars show the min–max across {N_REPS} samples. That is sampling stability, not a confidence
  interval for a real effect.
"""
        )

# ---------------------------------------------------------------------------
# The tradeoff
# ---------------------------------------------------------------------------
with tab_tradeoff:
    st.write("")
    t1, t2, t3 = st.columns([1.3, 1.2, 0.8], gap="large")
    with t1:
        x_metric = st.radio("Measure placement variety as", ["Effective variety", "Catalog placed", "First-screen catalog"],
                            horizontal=True, help=f"{TIPS['variety']} | {TIPS['placed']} | {TIPS['first_screen']}")
    with t2:
        budget = st.slider("Engagement loss you'd accept", 0.0, 20.0, BUDGETS["Balanced discovery"], 0.5,
                           format="%.1f%%")
    with t3:
        full_range = st.toggle("Show full curve", value=False)
    pick = summary[summary["reward_change_pct_mean"] >= -budget].iloc[-1]

    xcol, xfmt, xlog = {"Effective variety": ("effective_catalog", ".0f", True),
                        "Catalog placed": ("coverage", ".0%", False),
                        "First-screen catalog": ("fv_coverage", ".0%", False)}[x_metric]
    fig = go.Figure()
    fig.add_hrect(y0=-budget, y1=0.5, fillcolor="rgba(45,212,191,0.07)", line_width=0)
    fig.add_scatter(
        x=summary[f"{xcol}_mean"], y=summary["reward_change_pct_mean"], mode="lines+markers",
        name="Our ranker at each λ", line=dict(color=MUTED, width=2), marker=dict(size=7, color=MUTED),
        error_y=dict(type="data", symmetric=False,
                     array=summary["reward_change_pct_max"] - summary["reward_change_pct_mean"],
                     arrayminus=summary["reward_change_pct_mean"] - summary["reward_change_pct_min"],
                     color=FAINT, thickness=1),
        customdata=summary["lambda"],
        hovertemplate="λ = %{customdata}<br>" + x_metric + ": %{x:" + xfmt + "}<br>Engagement: %{y:.2f}%<extra></extra>",
    )
    for name, l in PRESETS.items():
        r = row(l)
        fig.add_scatter(x=[r[f"{xcol}_mean"]], y=[r.reward_change_pct_mean], mode="markers+text",
                        name=f"{name} (λ = {l:g})", text=[name],
                        textposition="bottom left" if l > 0 else "top right",
                        marker=dict(size=16, color=STRATEGY_COLORS[name], line=dict(color=BG, width=2)),
                        textfont=dict(color=STRATEGY_COLORS[name]))
    fig.add_scatter(x=[pick[f"{xcol}_mean"]], y=[pick.reward_change_pct_mean], mode="markers",
                    name=f"Most variety within {budget:.1f}% (λ = {pick['lambda']:g})",
                    marker=dict(size=24, color="rgba(0,0,0,0)", line=dict(color=TEXT, width=2)))
    fig.update_xaxes(title=x_metric, tickformat=xfmt, type="log" if xlog else "linear")
    fig.update_yaxes(title="Predicted engagement vs engagement-first", ticksuffix="%",
                     range=[-75, 3] if full_range else [-16, 1.5])
    if not full_range:
        zoom = {"effective_catalog": [math.log10(20), math.log10(260)], "coverage": [0.05, 0.75],
                "fv_coverage": [0.03, 0.32]}
        fig.update_xaxes(range=zoom[xcol])
    st.plotly_chart(style_fig(fig, 520, legend_below=True))

    k1, k2, k3 = st.columns(3, gap="large")
    k1.markdown(kpi(f"Best within a {budget:.1f}% budget", f"λ = {pick['lambda']:g}",
                    f"{signed(pick.reward_change_pct_mean, 2)} engagement", tone(pick.reward_change_pct_mean),
                    TIPS["lambda"]), unsafe_allow_html=True)
    k2.markdown(kpi("Catalog placed", f"{pick.coverage_mean:.0%}", f"first screen: {pick.fv_coverage_mean:.1%}",
                    "neu", TIPS["placed"]), unsafe_allow_html=True)
    k3.markdown(kpi("Effective variety", f"{pick.effective_catalog_mean:.0f} playlists",
                    f"{signed(pick.effective_catalog_chg, 0)} vs engagement-first", "pos", TIPS["variety"]),
                unsafe_allow_html=True)

    st.write("")
    flat = summary[summary["reward_change_pct_mean"] >= -0.5].iloc[-1]
    steep = row(1.5)
    r1, r2, r3 = st.columns(3, gap="large")
    r1.markdown(f"**The first step is cheap.** Up to λ = {flat['lambda']:g}, variety grows "
                f"{flat.effective_catalog_chg:+.0f}% for {flat.reward_change_pct_mean:+.2f}% engagement.")
    r2.markdown(f"**The middle needs judgment.** Between λ = {LAM_BAL:g} and {LAM_DIV:g}, each extra unit of "
                "variety costs more engagement. How much variety is worth is a product call.")
    r3.markdown(f"**Then it falls off a cliff.** At λ = 1.5, engagement drops {abs(steep.reward_change_pct_mean):.0f}%, "
                f"and the first 3 cards lose {abs(steep.initial_view_reward_chg):.0f}% too.")

    st.divider()
    section("Who pays as λ rises?", "Change in predicted engagement by listener quartile")
    u1, _ = st.columns([1, 3])
    with u1:
        seg_unit = st.radio("Unit", ["Relative %", "Percentage points"], horizontal=True,
                                        key="seg_unit", label_visibility="collapsed")
    kcol, suffix = ("pts", " pts") if seg_unit == "Percentage points" else ("rel", "%")
    seg_fig = go.Figure()
    for t in range(4):
        ts = tier_summary[(tier_summary["tier"] == t) & (tier_summary["lambda"] <= 2)]
        seg_fig.add_scatter(x=ts["lambda"], y=ts[kcol], mode="lines+markers", name=TIER_LABELS[t],
                            line=dict(color=TIER_COLORS[t], width=2.5), marker=dict(size=6),
                            hovertemplate="λ = %{x}<br>%{y:.2f}" + suffix + "<extra>" + TIER_LABELS[t] + "</extra>")
    for name in ("Balanced discovery", "Diversity-first"):
        seg_fig.add_vline(x=PRESETS[name], line=dict(color=STRATEGY_COLORS[name], dash="dot", width=1))
    seg_fig.update_xaxes(title="Discovery weight (λ)")
    seg_fig.update_yaxes(title=f"Change vs engagement-first ({suffix.strip()})", ticksuffix=suffix)
    s1, s2 = st.columns([1.7, 1], gap="large")
    with s1:
        st.plotly_chart(style_fig(seg_fig, 380))
    with s2:
        st.markdown(
            f"""
At λ = {LAM_BAL:g}, the bottom quartile loses **{abs(Q1_BAL.rel):.1f}%** ({Q1_BAL.pts:+.1f} pts). The top
quartile loses **{abs(Q4_BAL.rel):.2f}%**.

Heavy streamers have many likely matches, so swapping a few changes little. Our likely explanation for the
bottom quartile is that they rely on a handful of popular playlists, which are exactly the ones the ranker
discounts.

A single λ for everyone is probably the wrong design.
"""
        )

    with st.expander("Across Deezer's 100 listener clusters"):
        lam_c = st.select_slider("λ for cluster view", options=[l for l in LAMBDAS if l > 0], value=LAM_BAL,
                                 key="lam_clusters")
        cl = clusters[(clusters["strategy"] == "Exposure-aware") & np.isclose(clusters["lambda"], lam_c)]
        cfig = go.Figure(go.Scatter(
            x=cl["baseline_reward"], y=cl["reward_change_pct"], mode="markers",
            marker=dict(size=np.sqrt(cl["n_users"]) / 3, color=cl["reward_change_pct"],
                        colorscale=[[0, TIER_COLORS[0]], [0.7, TIER_COLORS[1]], [1, TIER_COLORS[2]]],
                        line=dict(color=BG, width=1), showscale=False),
            customdata=np.stack([cl["cluster"], cl["n_users"]], axis=1),
            hovertemplate="Cluster %{customdata[0]}<br>%{customdata[1]:,} sampled listeners<br>"
                          "Baseline: %{x:.1%}<br>Change: %{y:.2f}%<extra></extra>",
        ))
        cfig.update_xaxes(title="Cluster's predicted engagement under engagement-first", tickformat=".0%")
        cfig.update_yaxes(title="Change at selected λ", ticksuffix="%")
        st.plotly_chart(style_fig(cfig, 360))
        st.caption("Deezer's anonymized k-means clusters (k = 100), which have no published labels. Bubble size "
                   f"is the number of sampled listeners, pooled over {N_REPS} samples.")

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
with tab_reco:
    st.markdown('<div class="kicker" style="margin-top:18px">Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="lede" style="font-size:clamp(1.6rem,2.8vw,2.3rem)">Run an A/B test of balanced '
                f'discovery (λ = {LAM_BAL:g}), with a guardrail for listeners least likely to stream.</div>',
                unsafe_allow_html=True)
    st.write("")
    kpi_row([
        kpi("Engagement-first (today)", f"{BASE.coverage_mean:.0%} placed",
            f"first screen {BASE.fv_coverage_mean:.1%}", "neu"),
        kpi(f"Balanced (λ = {LAM_BAL:g})", f"{BAL.coverage_mean:.0%} placed",
            f"{signed(BAL.reward_change_pct_mean)} engagement · first screen {BAL.fv_coverage_mean:.1%}", "neg"),
        kpi(f"Diversity-first (λ = {LAM_DIV:g})", f"{DIV.coverage_mean:.0%} placed",
            f"{signed(DIV.reward_change_pct_mean)} engagement · first screen {DIV.fv_coverage_mean:.1%}", "neg"),
    ])
    st.write("")
    a1, a2 = st.columns(2, gap="large")
    with a1:
        section("What we found")
        st.markdown(
            f"""
1. **Placements are highly concentrated.** Engagement-first places only {BASE.coverage_mean:.0%} of the
   catalog for anyone. Its placements spread as evenly as just {BASE.effective_catalog_mean:.0f} playlists would.
2. **Placement variety is cheap at first.** Balanced discovery doubles the catalog placed for a
   {abs(BAL.reward_change_pct_mean):.1f}% ({BAL.reward_change_pts:+.1f} pts) drop in predicted engagement.
3. **The first screen barely changes.** The catalog in the first 3 cards goes from {BASE.fv_coverage_mean:.1%} to
   {BAL.fv_coverage_mean:.1%}, and first-screen engagement changes {BAL.initial_view_reward_chg:+.2f}%. The
   cost is low largely *because* the variety lands where fewer listeners may look.
4. **The cost is uneven.** Bottom-quartile listeners lose {abs(Q1_BAL.rel):.1f}% ({Q1_BAL.pts:+.1f} pts). The top
   quartile is essentially unaffected.
5. **Aggressive diversity is expensive.** Diversity-first costs {abs(DIV.reward_change_pct_mean):.1f}% overall and
   {abs(Q1_DIV.rel):.0f}% for the bottom quartile.
"""
        )
    with a2:
        section("What to do")
        st.markdown(
            f"""
- **Test, don't launch.** The {abs(BAL.reward_change_pct_mean):.1f}% figure is a model prediction. The real cost
  and any real benefit can only come from an experiment (see *A/B test plan*).
- **Measure what's actually seen.** Log which slots each listener views. If few listeners scroll past card 3,
  balanced discovery changes little in practice.
- **Design a version for light listeners** with a lower λ. We have not evaluated this offline yet.
- **Keep Diversity-first off the home screen.** A dedicated discovery surface is a better place to test that level
  of exploration.
"""
        )

    with st.expander("Measured here vs. hypotheses we have not tested"):
        st.markdown(
            """
| Claim | Status |
|---|---|
| Predicted engagement change for each strategy and listener group | **Measured** (model-based) |
| Catalog placed, first-screen catalog, effective variety | **Measured** (on our simulated carousels) |
| Results are stable across user samples and serving orders | **Measured** (see Methodology) |
| Real listeners would respond as the model predicts | Hypothesis: needs the A/B test |
| Less exposure keeps playlists from being learned about (feedback loop) | Hypothesis: not modeled |
| Broader exposure benefits creators or catalog health | Hypothesis: playlists are anonymized |
| Discovery improves satisfaction or retention | Hypothesis: no long-term outcomes in data |
| Extra exposure has learning value for a future model | Hypothesis: our analysis is a single snapshot |
| "Catalog placed" means genre or artist diversity | **Not supported**: no content metadata |
"""
        )
    with st.expander("Assumptions and limitations"):
        st.markdown(
            f"""
- **Predictions, not outcomes.** Engagement comes from Deezer's published model applied to real anonymized
  features. No streams are observed.
- **Least certain where it matters most.** The model was fit on historical data dominated by popular playlists.
  Its predictions for rarely shown playlists, which diversification promotes, are the least reliable.
- **Placements are not views.** The data has no impression or scroll data, so actual viewed exposure is unknown.
  The first-3 and all-12 figures describe what the ranker assigns. Viewed exposure must be measured online.
- **Static and single-session.** No feedback loops, repeat visits, fatigue, novelty or retention.
- **Independent stream decisions** are assumed across cards (as in Deezer's simulator).
- **Coverage depends on population size.** It is computed per {N_SAMPLE:,}-listener sample, so compare strategies
  with each other, not with an absolute target.
"""
        )

# ---------------------------------------------------------------------------
# A/B test plan
# ---------------------------------------------------------------------------
with tab_ab:
    st.write("")
    st.markdown('<span class="badge ours">Proposal</span> <span class="muted">A design for validating the '
                'recommendation on real listeners. No experiment has been run.</span>', unsafe_allow_html=True)
    st.write("")
    e1, e2 = st.columns([1.2, 1], gap="large")
    with e1:
        section("Design")
        margin_default = BUDGETS["Balanced discovery"]
        st.markdown(
            f"""
| | |
|---|---|
| **Question** | Is balanced discovery *no worse than* today's ranking at getting a stream from the carousel, while increasing discovery? |
| **Control** | Engagement-first ranking (λ = 0) |
| **Treatment** | Balanced discovery (λ = {LAM_BAL:g}) |
| **Randomization (assignment)** | User account, hashed and sticky for the whole test. Assignment alone does not mean the user saw a carousel |
| **Eligible exposure** | The first time during enrollment that the home screen *renders* the playlist carousel for the user. Logged the same way in both arms, before ranking is applied, so the arm cannot affect it |
| **Primary metric** | Whether a randomized user streams at least one playlist from their **first eligible recommendation carousel** within **30 minutes** of its render (illustrative window). One binary outcome per user |
| **Analysis population** | Assigned users with an eligible exposure (see *Handling users without an eligible exposure*) |
| **Test** | One-sided non-inferiority, α = 0.025, margin = −{margin_default:.0f}% relative. The margin mirrors the offline loss budget used to choose λ, but it is a business choice on a different metric |
| **Discovery metric** | Distinct carousel playlists streamed per user; share of carousel streams from playlists outside control's most-placed 10% |
| **Exposure logging** | Viewed slots per carousel render, to measure how many listeners see cards 4–12 |
| **Guardrails** | Listening time, sessions per user, early skips, home-screen bounce, 7-day retention, latency |
| **Pre-registered segments** | Likelihood-to-stream quartile (from pre-period data), new vs. tenured users, platform |
| **Duration** | 14-day enrollment window (illustrative); guardrails measured over the full test; analyze at the pre-set date |
"""
        )
        with st.expander("Handling users without an eligible exposure"):
            st.markdown(
                """
- **Assignment ≠ exposure.** Users are randomized when they enter the test. Some will never open the home screen
  during enrollment and never get a carousel.
- **Why filtering on eligibility is safe here.** Eligibility is defined by the carousel being *rendered*. That event
  is logged identically in both arms and happens before the user sees any arm-specific content, so the treatment
  cannot change who qualifies. Filtering on it therefore does not create post-randomization selection bias.
- **What must not be used as a filter:** whether the user *viewed* particular positions, scrolled or clicked. These
  happen after the arm-specific content is shown and can be affected by treatment. They are diagnostics only.
- **Checks:** sample-ratio mismatch on eligible users (not just assigned users) and balance of pre-period
  covariates among eligible users.
- **Sensitivity analysis:** an intent-to-treat estimate over all assigned users, counting users with no eligible
  exposure as 0. This dilutes the effect but cannot be biased by the eligibility filter.
- **Sample size counts eligible users.** To enroll enough people, divide the per-arm figure by the expected share of
  assigned users who get an eligible carousel.
"""
            )
        with st.expander("Validation checks and decision rules"):
            st.markdown(
                f"""
**Before reading results**
- Sample-ratio mismatch (χ²) on assignment counts
- Pre-period balance on the primary metric (or an A/A test)
- Treatment actually places more distinct playlists, and viewed-slot logs confirm listeners see them
- The primary metric captures each user's first carousel only, so it measures the immediate response. Longer-run
  effects (novelty wearing off, habituation) are read from guardrails over the full test
- Interference: does treatment exposure shift popularity signals that control also uses?

**Decide**
- **Ship** if the lower bound of the 95% CI for the relative change is above −{margin_default:.0f}%, the
  discovery metric (observed streams of new-to-user playlists) improves, and no guardrail is breached, including in the bottom quartile.
- **Iterate** (e.g. lower λ for light listeners) if the average passes but the bottom quartile breaches.
- **Stop** if non-inferiority fails overall.

Secondary metrics that are ratios (e.g. streams per placement) need the delta method or user-level aggregation
for variance. CUPED with pre-period streaming can reduce the required sample.
"""
            )
    with e2:
        section("Sample size", "Eligible users per arm for the non-inferiority test above")
        p0 = st.number_input("Assumed control rate: stream from first eligible carousel within 30 min", 0.01, 0.99,
                             0.30, 0.01,
                             help="An illustrative assumption. Replace with the rate observed in pre-period logs. "
                                  "The offline modeled 74% assumes all 12 cards are seen and is not an estimate "
                                  "of this real-world rate.")
        margin = st.number_input("Non-inferiority margin (% relative)", 0.5, 10.0, margin_default, 0.5)
        effect_opts = {
            "No true difference (0%)": 0.0,
            f"Scenario: offline first-3-position change ({BAL.initial_view_reward_chg:+.2f}%)":
                BAL.initial_view_reward_chg,
            f"Scenario: offline all-12-position change ({BAL.reward_change_pct_mean:+.2f}%)":
                BAL.reward_change_pct_mean,
        }
        effect_label = st.selectbox("Assumed true relative effect", list(effect_opts),
                                    help="The offline changes are planning scenarios. They are modeled "
                                         "per-carousel changes, not predictions of the real primary metric.")
        effect = effect_opts[effect_label]
        power = st.select_slider("Power", options=[0.8, 0.85, 0.9], value=0.8)

        n = ni_sample_size_per_arm(p0, margin / 100, effect / 100, alpha=0.025, power=power)
        if math.isinf(n):
            st.markdown(kpi("Eligible users per arm", "Not achievable",
                            "assumed effect is at or beyond the margin", "neg"), unsafe_allow_html=True)
        else:
            st.markdown(kpi("Eligible users per arm", f"{int(n):,}", f"{2 * int(n):,} total for two arms", "neu"),
                        unsafe_allow_html=True)

        st.markdown("**How close to the margin matters**")
        grid_effects = [0.0, -0.25, -0.5, -0.75, -0.9]
        rows = []
        for frac in grid_effects:
            e = frac * margin
            n_e = ni_sample_size_per_arm(p0, margin / 100, e / 100, alpha=0.025, power=power)
            rows.append({"Assumed true effect": "0.00%" if e == 0 else f"{e:.2f}%".replace("-", "−"),
                         "Share of margin used up": f"{abs(frac):.0%}",
                         "Eligible users per arm": f"{int(n_e):,}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with st.expander("Method"):
            st.markdown(
                r"""
Wald approximation for a non-inferiority test of two independent proportions:

$n = \dfrac{(z_{1-\alpha} + z_{1-\beta})^2\,[p_c(1-p_c) + p_t(1-p_t)]}{(p_t - p_c + \delta)^2}$

- $p_c$: assumed control rate. $p_t = p_c(1 + \text{effect})$.
- $\delta = \text{margin} \times p_c$ (absolute margin). One-sided $\alpha = 0.025$, equal allocation.
- No variance reduction. CUPED would lower $n$.

Why it fits: the primary outcome is one binary value per eligible user, users are randomized independently, and
the comparison is a difference in proportions with a relative margin converted to an absolute $\delta$.
Assumptions: independent users, no interference between arms, and an eligibility filter that does not depend on
the arm (see *Handling users without an eligible exposure*). $n$ counts eligible users per arm.
The unit tests check the formula against a hand calculation and, by simulation, check that it delivers the
stated power and about 2.5% false-positive rate at the margin.
"""
            )
        st.caption(
            "Required users grow with 1 / (distance to margin)². An effect halfway to the margin needs about 4× the "
            "users of no effect. At 90% of the margin it needs about 100×. The offline 12-position figure assumes "
            "all 12 cards are seen and uses a model, so it is not equivalent to the real first-carousel stream rate. "
            "Treat the offline changes as scenarios, not forecasts. If the true cost could be near the margin, choose "
            "the margin first, on business grounds, and expect a large or long test."
        )

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
with tab_method:
    st.write("")
    v = meta["validation"]
    m1, m2 = st.columns(2, gap="large")
    with m1:
        section("Data and model")
        st.markdown(
            f"""
<span class="badge obs">Observed</span> {v['n_users_file']:,} anonymized user embeddings (96 dims) with one of
100 k-means cluster ids, and {N_PLAYLISTS} playlist weight vectors (97 dims, the last acting as an intercept).<br><br>
<span class="badge mod">Modeled</span> Deezer's published ground-truth model: P(stream) = σ(user · playlist), with
a constant 1 appended to user vectors. Slate engagement is 1 − ∏(1 − pᵢ). We use expected values, so there is no
simulation noise.<br><br>
<span class="badge ours">Ours</span> The ranking strategies, the loss-budget presets, all metrics, the listener
quartiles and the recommendation.
""",
            unsafe_allow_html=True,
        )
        st.write("")
        section("Our ranker")
        st.markdown(
            f"""
Users are served in random order, in batches of 250. Each playlist is discounted by how over-exposed it already is:

`score = log p(stream) − λ · log(1 + placements so far / even share)`

The listener's top 12 by score are shown, ordered by their own stream probability. λ = 0 is engagement-first.
Presets are the most diverse λ within a {BUDGETS['Balanced discovery']:.0f}% (balanced) or
{BUDGETS['Diversity-first']:.0f}% (diversity-first) loss in predicted engagement.
"""
        )
    with m2:
        section("Robustness checks")
        o = order[np.isclose(order["lambda"], LAM_BAL)]
        o_div = order[np.isclose(order["lambda"], LAM_DIV)]
        st.markdown(
            f"""
| Check | Result at λ = {LAM_BAL:g} |
|---|---|
| {N_REPS} independent user samples | engagement {signed(BAL.reward_change_pct_min, 3)} to {signed(BAL.reward_change_pct_max, 3)}; catalog placed {BAL.coverage_min:.1%}–{BAL.coverage_max:.1%} |
| {order['order'].nunique()} serving orders, same users | engagement {signed(o.reward_change_pct.min(), 3)} to {signed(o.reward_change_pct.max(), 3)}; catalog placed {o.coverage.min():.1%}–{o.coverage.max():.1%} |
| First vs. last 10% of users served | {signed(o.first_decile_change_pct.mean(), 2)} vs {signed(o.last_decile_change_pct.mean(), 2)} |
| First 3 cards only | engagement {signed(BAL.initial_view_reward_chg, 2)}; first-screen catalog {BASE.fv_coverage_mean:.1%} → {BAL.fv_coverage_mean:.1%} |

Serving order barely affects aggregate results. At λ = {LAM_DIV:g}, though, the first 10% of users served lose
more ({signed(o_div.first_decile_change_pct.mean())} vs {signed(o_div.last_decile_change_pct.mean())}): the even-share
benchmark is noisy early on. A live system would warm-start exposure counts from recent traffic.

These ranges show that the *calculation* is stable. They are not confidence intervals for a real effect.
"""
        )

    with st.expander("Metric definitions and denominators"):
        st.markdown(
            f"""
| Metric | Definition | Denominator |
|---|---|---|
| Predicted engagement | Mean over users of 1 − ∏(1 − pᵢ) over the {N_SLOTS} placed playlists | Users |
| First-screen engagement | Same over slots 1–{L_INIT} | Users |
| Change (%) / (pts) | Relative / absolute difference vs engagement-first on the same users | Engagement-first value |
| Catalog placed | Playlists with ≥ 1 placement in any slot | {N_PLAYLISTS} playlists |
| First-screen catalog | Playlists with ≥ 1 placement in slots 1–{L_INIT} | {N_PLAYLISTS} playlists |
| Effective variety | exp(Shannon entropy of placement shares) | Users × {N_SLOTS} placements |
| Top-10% share, Gini | Concentration of placements | All placements / playlists |

A **placement** is a playlist assigned to a slot. It is not a confirmed view, since the data has no scroll
behavior. Every strategy makes exactly users × {N_SLOTS} placements, so the metrics compare *distribution*, not
volume.
"""
        )
    with st.expander("Sampling, validation and reproducibility"):
        st.markdown(
            f"""
- {N_REPS} disjoint random samples of {N_SAMPLE:,} users (seed {meta['seed']}). The 1.8 GB file is streamed in
  chunks.
- All strategies run on the same users with the same {N_SLOTS} slots.
- Validation: {v['n_users_file']:,} rows, {v['missing_values']} missing values, {v['n_segments']} clusters
  (sizes {v['segment_size_min']:,}–{v['segment_size_max']:,}), playlist intercept mean
  {v['playlist_bias_mean']:.2f}.
- Stream probabilities are skewed: median {v['stream_prob_median']:.2%}, 99.9th percentile
  {v['stream_prob_p99_9']:.0%}.
- Rebuild everything with `python scripts/prepare_data.py` (about 1 minute).
"""
        )
    with st.expander("Offline evaluation vs. an online experiment"):
        st.markdown(
            """
This analysis answers: *if Deezer's model is right, what would each strategy cost and gain?* It makes no causal
claim about real listeners. The model is least reliable for rarely shown playlists, and it has no long-term
dynamics, so both costs and benefits could differ online. The A/B test plan describes how to get causal evidence.
"""
        )
    st.caption("Data © Deezer, CC BY 4.0, Zenodo 10.5281/zenodo.4048678. Bendada, W., Salha, G., & Bontempelli, T. "
               "(2020). Carousel Personalization in Music Streaming Apps with Contextual Bandits. RecSys 2020. "
               "The strategies, metrics and analysis here are independent work.")
