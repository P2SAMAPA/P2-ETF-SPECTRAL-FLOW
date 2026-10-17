import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Spectral-Flow", page_icon="🌊", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#1d4ed8"
POSITIVE = "#16a34a"
NEUTRAL = "#d97706"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
CONF_COLORS = {"high": POSITIVE, "medium": NEUTRAL, "low": SUBTLE}
CLASS_COLORS = {"Fixed Income": "#0e7490", "Real Estate": "#7c3aed", "Commodity": "#b45309",
                "Equity-Broad": "#be123c", "Equity-Sector": "#15803d"}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .universe-heading { font-size: 1.15rem; font-weight: 700; color: __INK__; margin: 0 0 0.15rem 0; }
    .pick-card { background: #ffffff; border: 1px solid __BORDER__; border-left: 4px solid var(--accent);
        border-radius: 12px; padding: 1.0rem 1.2rem; margin: 0.35rem 0; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }
    .pick-ticker { font-size: 1.05rem; font-weight: 700; color: __INK__; letter-spacing: 0.02em; }
    .pick-return { font-size: 1.8rem; font-weight: 800; color: __INK__; margin: 0.2rem 0 0.1rem 0; line-height: 1.1; }
    .pick-sub { font-size: 0.72rem; color: __SUBTLE__; margin-bottom: 0.4rem; }
    .pick-badge { display: inline-block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 0.15rem 0.55rem; border-radius: 999px; color: white; background: var(--accent); }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .banner { background: linear-gradient(90deg, #eff6ff 0%, #f0f9ff 100%); border: 1px solid #bfdbfe;
        border-left: 4px solid __PRIMARY__; border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.88rem;
        color: #1e3a8a; margin: 0.5rem 0 0.9rem 0; }
    .banner-warn { background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid __NEUTRAL__;
        border-radius: 10px; padding: 0.7rem 1rem; font-size: 0.88rem; color: #78350f; margin: 0.5rem 0 0.9rem 0; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE)
       .replace("__BORDER__", CARD_BORDER).replace("__PRIMARY__", PRIMARY).replace("__NEUTRAL__", NEUTRAL))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-spectral-flow-results"

STAT_LABELS = {
    "absorption_ratio": "Absorption ratio (variance in top ~N/5 modes)",
    "rel_gap": "Relative spectral gap (L1 - L2) / L1",
    "bulk_edge": "Bulk edge (L after the top modes)",
    "spacing_ratio": "Level-spacing ratio <r>",
}
STAT_READING = {
    "absorption_ratio": ("variance concentrating in few modes (correlations tightening)", "variance spreading across modes"),
    "rel_gap": ("gap widening: the market mode is pulling away", "gap collapsing: the market mode is losing dominance"),
    "bulk_edge": ("bulk expanding", "bulk contracting"),
    "spacing_ratio": ("levels repelling more (GOE-like)", "levels condensing (Poisson-like)"),
}
TRIGGER_LABELS = {
    "absorption_up": "Absorption-ratio shift up", "gap_collapse": "Spectral gap collapse",
    "bulk_expansion": "Bulk expansion", "condensation": "Eigenvalue condensation",
    "repulsion": "Eigenvalue repulsion",
}


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("spectral_results_") and f.endswith(".json"))
        if not names:
            return None, "No spectral_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("spectral_results_*.json")
    if local:
        latest = sorted(local)[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f), ["Loaded local file: " + latest]
        except Exception as e:
            notes.append("Found local file " + latest + " but couldn't parse it: " + str(e))
    name, err = _find_latest_hf_result(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            url = "https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(resp.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def conf_color(c):
    return CONF_COLORS.get((c or "low").lower(), SUBTLE)


def fmt_bps(x):
    return "n/a" if x is None else ("%+.1f" % (x * 1e4))


def render_pick_cards(picks):
    if not picks:
        st.info("No picks available for this selection.")
        return
    horizon = picks[0].get("horizon_days")
    cols = st.columns(min(len(picks), 3))
    for i, p in enumerate(picks):
        html = ('<div class="pick-card" style="--accent: ' + conf_color(p["confidence"]) + ';">'
                '<div class="pick-ticker">' + p["ticker"] + '</div>'
                '<div class="pick-return">' + ("%+.2f" % p["expected_return"]) + '%</div>'
                '<div class="pick-sub">expected ' + str(horizon) + '-day return vs universe average</div>'
                '<span class="pick-badge">' + p["confidence"] + ' confidence</span></div>')
        with cols[i % len(cols)]:
            st.markdown(html, unsafe_allow_html=True)


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div>'
            '<div class="kpi-label">' + label + '</div></div>')


def render_spectrum(pattern, key):
    now, lag = pattern.get("spectrum_today"), pattern.get("spectrum_lag")
    if not now:
        st.info("No spectrum snapshot available.")
        return
    k = min(12, len(now))
    x = list(range(1, k + 1))
    fig = go.Figure()
    if lag:
        fig.add_trace(go.Bar(x=x, y=lag[:k], name=str(pattern.get("flow_lookback", 10)) + " days ago",
                             marker_color="#cbd5e1"))
    fig.add_trace(go.Bar(x=x, y=now[:k], name="today", marker_color=PRIMARY))
    fig.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", yaxis=dict(title="Eigenvalue (log scale)", type="log", gridcolor="#eef2f7"),
                      xaxis=dict(title="Eigenvalue rank", dtick=1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="spec_" + key)
    st.caption("Top eigenvalues of the rolling correlation matrix, today vs the lookback date. The movement "
               "between the grey and blue bars is the 'flow'. Eigenvalues sum to the number of ETFs.")


def render_structure_state(pattern):
    state = pattern.get("structure_state") or {}
    rows = []
    for stat, v in state.items():
        z = v.get("shift_z")
        if z is None:
            reading = "n/a"
        elif z > 1.0:
            reading = STAT_READING[stat][0]
        elif z < -1.0:
            reading = STAT_READING[stat][1]
        else:
            reading = "near its recent norm"
        rows.append({"Statistic": STAT_LABELS.get(stat, stat),
                     "Value": round(v["value"], 4) if v.get("value") is not None else None,
                     "Shift z": round(z, 2) if z is not None else None, "Reading": reading})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Shift z = (15-day mean - 252-day mean) / 252-day std of the statistic (the absorption-ratio "
                   "'shift' construction of Kritzman et al. 2011, applied to each statistic).")


def render_exposure_map(snapshot, pick_tickers, key):
    if not snapshot:
        return
    df = pd.DataFrame(snapshot)
    df = df[np.isfinite(df["e_mkt"]) & np.isfinite(df["e_idio"])]
    if df.empty:
        return
    fig = go.Figure()
    is_pick = df["ticker"].isin(pick_tickers)
    for cls in sorted(set(c for c in df["asset_class"] if c)):
        sub = df[(df["asset_class"] == cls) & (~is_pick)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(x=sub["e_mkt"], y=sub["e_idio"], mode="markers+text", text=sub["ticker"],
                                 textposition="top center", textfont=dict(size=9, color=SUBTLE),
                                 marker=dict(size=9, color=CLASS_COLORS.get(cls, "#64748b"),
                                            line=dict(width=1, color="#111827")), name=cls,
                                 hovertemplate="%{text}<extra>" + cls + "</extra>"))
    sub = df[is_pick]
    if not sub.empty:
        fig.add_trace(go.Scatter(x=sub["e_mkt"], y=sub["e_idio"], mode="markers+text", text=sub["ticker"],
                                 textposition="top center", textfont=dict(size=11, color=INK, family="Arial Black"),
                                 marker=dict(size=15, color="#4ade80", symbol="diamond", line=dict(width=2, color="#111827")),
                                 name="top picks", hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(title="Market-mode weight (squared loading on mode 1)", gridcolor="#eef2f7"),
                      yaxis=dict(title="Idiosyncratic share (variance outside the top modes)", gridcolor="#eef2f7"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="expo_" + key)
    st.caption("Each ETF's exposure to the spectrum today. ETFs far right load heavily on the market mode "
               "(they move with everything); ETFs high on the vertical axis mostly move on their own. Colors are "
               "hand-labeled asset classes, for reference only.")


def render_curves(curves, key):
    if not curves or not curves.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["spectral_cum"], mode="lines", name="Spectral-flow model top-N",
                             line=dict(color=PRIMARY, width=2.2)))
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["baseline_cum"], mode="lines",
                             name="Baseline (price + static spectral exposure)", line=dict(color="#94a3b8", width=2, dash="dot")))
    si = curves.get("split_index", 0)
    if 0 < si < len(curves["dates"]):
        fig.add_vline(x=curves["dates"][si], line_dash="dash", line_color="#334155")
        fig.add_annotation(x=curves["dates"][si], y=1, yref="paper", text="holdout starts", showarrow=False,
                           xanchor="left", font=dict(size=11, color="#334155"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Cumulative net excess return (%)", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="curves_" + key)


def render_eig_series(pattern, key):
    s = (pattern or {}).get("series")
    if not s or not s.get("dates"):
        return
    fig = go.Figure()
    for nm, lab, col in [("lam1", "lambda 1", PRIMARY), ("lam2", "lambda 2", "#0ea5e9"),
                         ("lam3", "lambda 3", "#14b8a6"), ("bulk_edge", "bulk edge", "#f59e0b")]:
        fig.add_trace(go.Scatter(x=s["dates"], y=s[nm], mode="lines", name=lab, line=dict(width=1.6, color=col)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Eigenvalue (log scale)", type="log", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="eigser_" + key)


def grid_frame(grid, selected):
    rows = []
    for g in grid:
        s, h = g["selection"], g["holdout"]
        rows.append({
            "Corr window (d)": g["corr_window"], "Horizon (d)": g["horizon"], "Window (d)": g["window"],
            "Class-neutral": g.get("neutralize", False),
            "Sel. IC t": s.get("ic_t"), "Sel. net bps": (s.get("net_spread_mean") or 0) * 1e4,
            "Hold n": h.get("n"), "Hold IC": h.get("ic_mean"), "Hold IC t": h.get("ic_t"),
            "Dynamics gain (IC vs static baseline)": h.get("d_ic_mean"),
            "Static gain (IC, baseline vs price-only)": h.get("static_gain_ic"),
            "Hold net bps": (h.get("net_spread_mean") or 0) * 1e4,
            "_sel": (g["corr_window"] == selected["corr_window"] and g["horizon"] == selected["horizon"]
                    and g["window"] == selected["window"]
                    and g.get("neutralize", False) == selected.get("neutralize", False)),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Class-neutral", "Corr window (d)", "Horizon (d)", "Window (d)"]).reset_index(drop=True)


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">🌊 P2 Spectral-Flow</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Where is the correlation spectrum moving? Eigenvalue dynamics as a '
                    'market-structure signal. Experimental: credited only for what the dynamics add beyond static '
                    'spectral exposure.</div>', unsafe_allow_html=True)
    with refresh:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
            st.write("Run `python trainer.py` to create a local `spectral_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return

    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])

    tab1, tab2 = st.tabs(["🔮 Live Signal", "🧪 Diagnostics & Backtest"])

    with tab1:
        st.markdown('<div class="note">Each day the engine eigendecomposes the rolling correlation matrix and tracks '
                    'how the eigenvalues move. Because "the gap collapsed today" is the same for every ETF, the '
                    'movement only helps rank ETFs when combined with each ETF\'s own exposure to the mode that is '
                    'moving. Three nested models are compared on the same dates: price features only; price plus '
                    'static spectral exposure; and that plus the dynamics. Only the last step is the thesis.</div>',
                    unsafe_allow_html=True)
        for uni, picks in data.get("top_picks", {}).items():
            sel = data["selected"][uni]
            h = sel["holdout"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cls = "banner" if sel["confidence"] != "Low" else "banner-warn"
            neut = '<b>asset-class-neutral</b>' if sel.get("neutralize") else '<b>not</b> asset-class-neutral'
            txt = ('Selected: correlation window <b>' + str(sel["corr_window"]) + 'd</b>, <b>' + str(sel["horizon"])
                   + '-day</b> horizon, <b>' + str(sel["window"]) + 'd</b> training window, ' + neut
                   + (' (pinned)' if sel.get("pinned") else '')
                   + '<br/>Holdout (' + str(h.get("n", 0)) + ' non-overlapping periods, never used for selection): rank-IC <b>'
                   + ("%+.3f" % h.get("ic_mean", 0)) + '</b> (t=' + ("%+.2f" % h.get("ic_t", 0))
                   + ') vs static-spectral baseline <b>' + ("%+.3f" % h.get("ic_base_mean", 0))
                   + '</b> and price-only <b>' + ("%+.3f" % h.get("ic_price_mean", 0))
                   + '</b> &nbsp;|&nbsp; net top-N excess <b>' + fmt_bps(h.get("net_spread_mean"))
                   + ' bps</b>/period (baseline ' + fmt_bps(h.get("base_net_spread_mean")) + ')')
            if sel["confidence"] == "Low":
                txt += ('<br/><b>Not validated:</b> on the holdout the dynamics do not clearly add value over static '
                        'spectral exposure, so treat these picks as unproven.')
            if not sel.get("gate_passed"):
                txt += '<br/>No configuration was profitable on the selection segment; this is the best of a weak set.'
            st.markdown('<div class="' + cls + '">' + txt + '</div>', unsafe_allow_html=True)

            render_pick_cards(picks)

            pattern = data["patterns"].get(uni, {}).get(str(sel["corr_window"])) or {}
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("###### The spectrum: today vs the lookback date")
                render_spectrum(pattern, "live_" + uni)
                st.markdown("###### Market-structure state")
                render_structure_state(pattern)
            with c2:
                st.markdown("###### Each ETF's exposure to the spectrum")
                render_exposure_map(pattern.get("snapshot"), [p["ticker"] for p in picks], "live_" + uni)
                flow = pattern.get("flow_today") or {}
                if flow:
                    st.caption("Eigenvalue flow today (time-series z): market mode "
                               + ("%+.2f" % flow["z_mkt"] if flow.get("z_mkt") is not None else "n/a")
                               + ", rotation modes " + ("%+.2f" % flow["z_rot"] if flow.get("z_rot") is not None else "n/a")
                               + ", bulk " + ("%+.2f" % flow["z_bulk"] if flow.get("z_bulk") is not None else "n/a"))
            base = data.get("baseline_picks", {}).get(uni, [])
            if base:
                overlap = len(set(p["ticker"] for p in picks) & set(p["ticker"] for p in base))
                with st.expander("Static-spectral baseline picks for comparison"):
                    st.write("Baseline top picks: " + ", ".join(p["ticker"] for p in base)
                             + " — overlap with the spectral-flow picks: " + str(overlap) + " of " + str(len(picks)))
                    fw = data.get("feature_weights", {}).get(uni)
                    if fw:
                        st.write("Fitted feature weights (standardized), spectral-flow model:")
                        st.dataframe(pd.DataFrame([fw]).T.rename(columns={0: "weight"}), use_container_width=True)
            st.markdown("<hr style='margin: 1.4rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab2:
        for uni, grid in data.get("grid", {}).items():
            sel = data["selected"][uni]
            h = sel["holdout"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            kp = [("Holdout rank-IC", "%+.3f" % h.get("ic_mean", 0)), ("IC t-stat", "%+.2f" % h.get("ic_t", 0)),
                  ("Net excess bps/period", fmt_bps(h.get("net_spread_mean"))),
                  ("Dynamics gain (IC)", "%+.3f" % h.get("d_ic_mean", 0)),
                  ("Static gain (IC)", "%+.3f" % h.get("static_gain_ic", 0))]
            for c, (lab, val) in zip(cols, kp):
                with c:
                    st.markdown(kpi(lab, val), unsafe_allow_html=True)
            st.markdown('<div class="note">Dynamics gain = spectral model minus the static-exposure baseline; static '
                        'gain = static-exposure baseline minus price-only. In class-neutral mode IC and spreads are '
                        'scored against class-neutral returns. Excess is net of a '
                        + str(data.get("settings", {}).get("trading_cost_bps", 15))
                        + ' bps round-trip cost on the fraction of picks replaced; test points are one horizon '
                        'apart, so periods never overlap.</div>', unsafe_allow_html=True)

            st.markdown("###### Cumulative net excess return of the selected configuration")
            render_curves(data.get("curves", {}).get(uni), key=uni)

            df = grid_frame(grid, sel)
            sel_mask = df["_sel"].values
            view = df.drop(columns=["_sel"])
            st.markdown("###### All configurations (selection segment chose; holdout is the honest number)")
            st.caption("Testing " + str(len(view)) + " configurations means the best-looking one on any single column "
                       "can be luck. Only the selected row's holdout is used for the confidence label.")
            styled = view.style.apply(lambda r: ["background-color: #dbeafe" if sel_mask[r.name] else "" for _ in r], axis=1)
            styled = styled.format({"Sel. IC t": "{:.2f}", "Sel. net bps": "{:+.1f}", "Hold IC": "{:+.3f}",
                                    "Hold IC t": "{:+.2f}", "Dynamics gain (IC vs static baseline)": "{:+.3f}",
                                    "Static gain (IC, baseline vs price-only)": "{:+.3f}",
                                    "Hold net bps": "{:+.1f}"}, na_rep="n/a")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            pats = data["patterns"].get(uni, {})
            st.markdown("###### Eigenvalue paths (selected correlation window)")
            render_eig_series(pats.get(str(sel["corr_window"])), key=uni)

            lead_rows = []
            for cw, pp in pats.items():
                for name, rl in (pp.get("leads") or {}).items():
                    if not rl.get("available"):
                        continue
                    lead_rows.append({
                        "Corr window": int(cw), "Trigger": TRIGGER_LABELS.get(name, name),
                        "Onsets": rl.get("n_onsets"), "Large-move events": rl.get("n_large_move_events"),
                        "Onset in prior 20d": rl.get("hit_rate"), "Chance level": rl.get("chance_hit_rate"),
                        "p-value": rl.get("p_value"), "Median lead (d)": rl.get("median_lead_days"),
                        "False alarm rate": rl.get("false_alarm_rate"),
                        "Chance false alarm": rl.get("chance_false_alarm_rate")})
            if lead_rows:
                n_tests = data.get("settings", {}).get("n_lead_tests", len(lead_rows))
                st.markdown("###### Do structure transitions lead a large aggregate move?")
                st.caption("Triggers fixed in advance: each statistic's shift z crosses +/-1.5; large-move event = an "
                           "equal-weight-universe move of at least 2x 21-day realized vol over 10 days. Chance level and "
                           "p-value come from a circular-shift permutation of the onset times. Across all universes "
                           "and windows " + str(n_tests) + " trigger tests are run, so roughly "
                           + ("%.0f" % (0.05 * n_tests)) + " would show p<0.05 by luck alone — a single small p-value "
                           "here is not evidence. Rows with no onsets or no events show n/a.")
                st.dataframe(pd.DataFrame(lead_rows).style.format(
                    {"Onset in prior 20d": "{:.0%}", "Chance level": "{:.0%}", "p-value": "{:.2f}",
                     "Median lead (d)": "{:.0f}", "False alarm rate": "{:.0%}", "Chance false alarm": "{:.0%}"},
                    na_rep="n/a"), use_container_width=True, hide_index=True)

            sp = data.get("cw_picks", {}).get(uni, {})
            if sp:
                st.markdown("###### Best configuration per correlation window")
                stabs = st.tabs([cw + "d" for cw in sp.keys()])
                for tb, (cw, info) in zip(stabs, sp.items()):
                    with tb:
                        c = info["config"]
                        neut = "class-neutral" if c.get("neutralize") else "not class-neutral"
                        st.caption(str(c["horizon"]) + "-day horizon, " + str(c["window"]) + "d window, " + neut
                                   + ", confidence " + info["confidence"] + ", holdout IC "
                                   + ("%+.3f" % info["holdout"].get("ic_mean", 0)))
                        render_pick_cards(info["picks"])
            st.markdown("<hr style='margin: 1.6rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
