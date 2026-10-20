import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Baseline-Book", page_icon="📒", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#0f766e"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
VERDICT_COLORS = {"Supported": "#16a34a", "Suggestive": "#d97706", "Not supported": "#64748b"}
CLASS_COLORS = {"Fixed Income": "#0e7490", "Real Estate": "#7c3aed", "Commodity": "#b45309",
                "Equity-Broad": "#be123c", "Equity-Sector": "#15803d"}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .universe-heading { font-size: 1.15rem; font-weight: 700; color: __INK__; margin: 0 0 0.15rem 0; }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .verdict { background: #ffffff; border: 1px solid __BORDER__; border-left: 6px solid var(--accent);
        border-radius: 10px; padding: 0.8rem 1.1rem; font-size: 0.9rem; color: __INK__; margin: 0.5rem 0 0.9rem 0; }
    .verdict-badge { display: inline-block; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.05em; padding: 0.15rem 0.6rem; border-radius: 999px; color: white; background: var(--accent); margin-right: 0.5rem; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE)
       .replace("__BORDER__", CARD_BORDER))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-baseline-book-results"


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("book_results_") and f.endswith(".json"))
        if not names:
            return None, "No book_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("book_results_*.json")
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


def bps(x):
    return "n/a" if x is None else ("%+.1f" % (x * 1e4))


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div>'
            '<div class="kpi-label">' + label + '</div></div>')


def label_of(data, cons):
    return data.get("settings", {}).get("construction_labels", {}).get(cons, cons)


def render_curves(curves, key):
    if not curves or not curves.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["net_cum"], mode="lines", name="Selected book, net of costs",
                             line=dict(color=PRIMARY, width=2.3)))
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["gross_cum"], mode="lines", name="Selected book, gross",
                             line=dict(color="#94a3b8", width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=curves["dates"], y=curves["ew_cum"], mode="lines", name="Equal-weight universe (for reference)",
                             line=dict(color="#cbd5e1", width=1.4)))
    si = curves.get("split_index", 0)
    if 0 < si < len(curves["dates"]):
        fig.add_vline(x=curves["dates"][si], line_dash="dash", line_color="#334155")
        fig.add_annotation(x=curves["dates"][si], y=1, yref="paper", text="holdout starts", showarrow=False,
                           xanchor="left", font=dict(size=11, color="#334155"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Cumulative return per period summed (%)", gridcolor="#eef2f7"),
                      xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="curves_" + key)


def render_heatmap(data, rob, metric, title, key, fmt):
    cons_list = list(rob.keys())
    horizons = [str(h) for h in data.get("settings", {}).get("horizons", [1, 5, 10, 20])]
    z, text = [], []
    for c in cons_list:
        row, trow = [], []
        for h in horizons:
            v = (rob.get(c) or {}).get(h, {}).get(metric)
            row.append(v)
            trow.append("" if v is None else (fmt % v))
        z.append(row)
        text.append(trow)
    fig = go.Figure(go.Heatmap(z=z, x=[h + "d" for h in horizons], y=[label_of(data, c) for c in cons_list],
                               text=text, texttemplate="%{text}", colorscale="RdYlGn", zmid=0 if metric != "share_positive" else 0.5,
                               showscale=False, hovertemplate="%{y}, %{x}: %{text}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      title=dict(text=title, font=dict(size=13)), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True, key=key)


def grid_frame(data, grid, selected):
    rows = []
    for g in grid:
        h = g["holdout"]
        rows.append({
            "Book": label_of(data, g["construction"]), "Horizon (d)": g["horizon"], "Window (d)": g["window"],
            "Class-neutral score": g["model_neutral"],
            "Sel. net t": g["selection"].get("net_t"), "Hold n": h.get("n"),
            "Hold gross bps": (h.get("gross_mean") or 0) * 1e4, "Hold net bps": (h.get("net_mean") or 0) * 1e4,
            "Hold net t": h.get("net_t"), "Beta": h.get("beta"), "Alpha t": h.get("alpha_t"),
            "Turnover": h.get("turnover_mean"),
            "_sel": (g["construction"] == selected["construction"] and g["horizon"] == selected["horizon"]
                    and g["window"] == selected["window"] and g["model_neutral"] == selected["model_neutral"]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Book", "Horizon (d)", "Window (d)", "Class-neutral score"]).reset_index(drop=True)


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">📒 P2 Baseline-Book</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Is the plain price baseline (momentum, volatility, beta) monetizable '
                    'after costs? A study, not a signal — with a forward record that cannot be fit to the past.</div>',
                    unsafe_allow_html=True)
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
            st.write("Run `python trainer.py` to create a local `book_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return

    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])
    stt = data.get("settings", {})

    tab1, tab2, tab3 = st.tabs(["⚖️ Verdict", "🧪 Robustness & Backtest", "🔭 Forward Record"])

    with tab1:
        st.markdown('<div class="note">For each universe the book is chosen on the first 70% of the periods only, then '
                    'judged once on the untouched last 30%. Costs: ' + str(stt.get("trading_cost_bps", 15))
                    + ' bps round trip on every unit of notional traded, plus ' + str(stt.get("short_borrow_bps_annual", 50))
                    + ' bps/yr borrow on short notional. A verdict of Supported needs the net t-statistic AND the '
                    'market-adjusted alpha t-statistic to clear 1.65.</div>', unsafe_allow_html=True)
        for uni, sel in data.get("selected", {}).items():
            h = sel["holdout"]
            accent = VERDICT_COLORS.get(sel["verdict"], "#64748b")
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            summ = data["summary"][uni]
            txt = ('<span class="verdict-badge">' + sel["verdict"] + '</span>Selected book: <b>'
                   + label_of(data, sel["construction"]) + '</b>, <b>' + str(sel["horizon"]) + '-day</b> holding period, '
                   + str(sel["window"]) + 'd training window, ' + ('class-neutral' if sel["model_neutral"] else 'plain')
                   + ' score.<br/>Holdout (' + str(h.get("n", 0)) + ' non-overlapping periods): gross <b>' + bps(h.get("gross_mean"))
                   + '</b> / net <b>' + bps(h.get("net_mean")) + ' bps</b> per period (t=' + ("%+.2f" % h.get("net_t", 0))
                   + '), market beta <b>' + ("%+.2f" % (h.get("beta") or 0)) + '</b>, alpha t=<b>'
                   + ("%+.2f" % (h.get("alpha_t") or 0)) + '</b>.<br/>Across all ' + str(summ["n_cells_holdout"])
                   + ' configurations, <b>' + ("%.0f" % (100 * (summ.get("share_cells_net_positive") or 0)))
                   + '%</b> made money on the holdout (mean net ' + ("%+.1f" % (summ.get("mean_net_bps") or 0)) + ' bps/period).')
            if not sel.get("gate_passed"):
                txt += '<br/>No configuration was profitable on the selection segment; this is the best of a weak set.'
            st.markdown('<div class="verdict" style="--accent: ' + accent + ';">' + txt + '</div>', unsafe_allow_html=True)

            cols = st.columns(5)
            kp = [("Net bps / period", bps(h.get("net_mean"))), ("Net t-stat", "%+.2f" % h.get("net_t", 0)),
                  ("Net Sharpe (ann.)", "%+.2f" % h.get("net_sharpe", 0)),
                  ("Turnover / period", "%.2f" % (h.get("turnover_mean") or 0)),
                  ("Max drawdown (%)", "%.1f" % ((h.get("max_drawdown") or 0) * 100))]
            for c, (lab, val) in zip(cols, kp):
                with c:
                    st.markdown(kpi(lab, val), unsafe_allow_html=True)

            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("###### What the signal is: each feature's own rank-IC (holdout, selected horizon)")
                fa = (data.get("factor_ic", {}).get(uni, {}) or {}).get(str(sel["horizon"]), {})
                if fa:
                    rows = [{"Feature": nm, "IC (raw)": v["raw"]["ic_holdout"], "t (raw)": v["raw"]["t_holdout"],
                             "IC (class-neutral)": v["neutral"]["ic_holdout"], "t (class-neutral)": v["neutral"]["t_holdout"]}
                            for nm, v in fa.items()]
                    st.dataframe(pd.DataFrame(rows).style.format(
                        {"IC (raw)": "{:+.3f}", "t (raw)": "{:+.2f}", "IC (class-neutral)": "{:+.3f}",
                         "t (class-neutral)": "{:+.2f}"}), use_container_width=True, hide_index=True)
                    st.caption("Raw IC far above class-neutral IC means the feature mostly separates asset classes "
                               "(e.g. bonds from equities) rather than ranking within them.")
            with c2:
                st.markdown("###### Today's target book for this configuration")
                bk = data.get("selected_book", {}).get(uni, [])
                if bk:
                    dfb = pd.DataFrame(bk)
                    dfb["weight %"] = dfb["weight"] * 100
                    st.dataframe(dfb[["ticker", "weight %", "asset_class"]].style.format({"weight %": "{:+.1f}"}),
                                 use_container_width=True, hide_index=True)
                    if sel["verdict"] != "Supported":
                        st.caption("Shown for transparency only: this book is not validated, so it is not a recommendation.")
                else:
                    st.info("No live book available.")
            st.markdown("<hr style='margin: 1.4rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab2:
        for uni, grid in data.get("grid", {}).items():
            sel = data["selected"][uni]
            rob = data["robustness"][uni]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                render_heatmap(data, rob, "mean_net_bps", "Mean holdout net return, bps per period (all models and windows)", "hm_net_" + uni, "%+.0f")
            with c2:
                render_heatmap(data, rob, "mean_alpha_t", "Mean holdout market-adjusted alpha t-statistic", "hm_alpha_" + uni, "%+.1f")
            st.markdown('<div class="note">Each cell averages the holdout results of every score model and training window '
                        'for that book and horizon. Green = makes money after costs. Short horizons lose to turnover; a '
                        'book that is green only at one horizon is a warning sign, not a finding.</div>', unsafe_allow_html=True)

            st.markdown("###### Selected book: cumulative return")
            render_curves(data.get("curves", {}).get(uni), key=uni)

            best = data["summary"][uni].get("best_holdout_cells") or []
            if best:
                st.caption("For reference, the three best-looking holdout cells (best of " + str(data["summary"][uni]["n_cells"])
                           + " — selection by hindsight, so expect these to look good by chance): "
                           + "; ".join(label_of(data, b["construction"]) + " " + str(b["horizon"]) + "d (net t=" + ("%+.2f" % b["net_t"]) + ")" for b in best))

            df = grid_frame(data, grid, sel)
            mask = df["_sel"].values
            view = df.drop(columns=["_sel"])
            st.markdown("###### All configurations (the selection segment chose the highlighted row)")
            styled = view.style.apply(lambda r: ["background-color: #ccfbf1" if mask[r.name] else "" for _ in r], axis=1)
            styled = styled.format({"Sel. net t": "{:+.2f}", "Hold gross bps": "{:+.1f}", "Hold net bps": "{:+.1f}",
                                    "Hold net t": "{:+.2f}", "Beta": "{:+.2f}", "Alpha t": "{:+.2f}", "Turnover": "{:.2f}"},
                                   na_rep="n/a")
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.markdown("<hr style='margin: 1.6rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="note">The only evidence here that is out-of-sample with respect to how this study was '
                    'designed. Every run records the target weights of a small, fixed, pre-registered set of books '
                    '(listed below) with no selection and no later change, and scores each one only after its holding '
                    'period has elapsed, with the same costs as the backtest. It starts empty and accrues one 5-day '
                    'period per week and one 20-day period per month, so expect a long wait before it says anything.</div>',
                    unsafe_allow_html=True)
        fr = data.get("forward_record") or {}
        status = data.get("ledger_status")
        if status == "hf_error":
            st.warning("The ledger could not be downloaded on the last run, so it was not updated or saved (stored history is untouched).")
        if not fr:
            st.info("No forward record yet.")
        else:
            rows = []
            for key, v in fr.items():
                uni, cons, hh = key.split("|")
                rows.append({"Universe": uni, "Book": label_of(data, cons), "Horizon": hh.replace("h", "") + "d",
                             "Recorded": v["n_recorded"], "Scored": v["n_scored"], "Pending": v["n_pending"],
                             "Net bps / period": None if v["net_mean"] is None else v["net_mean"] * 1e4,
                             "Net t": v["net_t"], "Cumulative net %": None if v["cum_net"] is None else v["cum_net"] * 100,
                             "Hit rate": v["hit_rate"], "Since": v["first_date"]})
            dfl = pd.DataFrame(rows).sort_values(["Universe", "Book", "Horizon"]).reset_index(drop=True)
            total_scored = int(dfl["Scored"].sum())
            st.markdown("###### Realized record of the pre-registered books (" + str(total_scored) + " scored periods in total)")
            st.dataframe(dfl.style.format({"Net bps / period": "{:+.1f}", "Net t": "{:+.2f}",
                                           "Cumulative net %": "{:+.2f}", "Hit rate": "{:.0%}"}, na_rep="—"),
                         use_container_width=True, hide_index=True)
            st.caption("A net t-statistic is shown only once a book has at least 3 scored periods, and means little below "
                       "~30. Thirty books are tracked, so a few will look good or bad by luck — judge the whole set, "
                       "not its best member.")

        books_today = data.get("ledger_books_today", {})
        if books_today:
            with st.expander("Today's pre-registered target books"):
                for uni, bks in books_today.items():
                    st.markdown("**" + uni.replace("_", " ").title() + "**")
                    for k, bk in bks.items():
                        cons, hh = k.split("|")
                        longs = ", ".join(r["ticker"] + " " + ("%+.0f" % (r["weight"] * 100)) + "%" for r in bk if r["weight"] > 0)
                        shorts = ", ".join(r["ticker"] + " " + ("%+.0f" % (r["weight"] * 100)) + "%" for r in bk if r["weight"] < 0)
                        st.caption(label_of(data, cons) + " (" + hh.replace("h", "") + "d): long " + (longs or "—")
                                   + ((" | short " + shorts) if shorts else ""))


if __name__ == "__main__":
    main()
