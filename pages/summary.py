import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Summary Valuation", layout="wide")

# ------------------------------------------------------------------------------
# POWERBI DARK THEME (FBC TUNED)
# ------------------------------------------------------------------------------
DARK_BG = "#020617"        # page background
PANEL_BG = "#020c1f"       # panels
CARD_BG = "#071525"        # KPI cards
PRIMARY_TEXT = "#e5f2ff"
MUTED_TEXT = "#9ca3af"
ACCENT_BLUE = "#38bdf8"
ACCENT_CYAN = "#0ea5e9"
ACCENT_GOLD = "#fbbf24"
DANGER = "#f97373"

st.markdown(
    f"""
    <style>

    /* ------------------------------------------------------ */
    /* GLOBAL BACKGROUND & TEXT                               */
    /* ------------------------------------------------------ */
    .main {{
        background: radial-gradient(circle at top left, #0d1424 0, {DARK_BG} 60%, #000 100%);
        color: {PRIMARY_TEXT};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    h1, h2, h3, h4, h5 {{
        color: {PRIMARY_TEXT} !important;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600 !important;
    }}

    /* ------------------------------------------------------ */
    /* TITLE BANNER                                           */
    /* ------------------------------------------------------ */
    .title-banner {{
        background: linear-gradient(90deg, #071426, #0a1b33 50%, #0d243f 100%);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 28px rgba(0,0,0,0.6);
    }}
    .title-main {{
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff !important;
    }}
    .title-sub {{
        font-size: 0.95rem;
        color: #d3e2f5 !important;
        margin-top: 3px;
    }}

    /* ------------------------------------------------------ */
    /* KPI CARDS — HIGH VISIBILITY VERSION                    */
    /* ------------------------------------------------------ */
    .kpi-card {{
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        background: rgba(10, 20, 40, 0.85); /* darker, solid */
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        backdrop-filter: blur(4px); /* gentle */
    }}
    .kpi-title {{
        font-size: 0.85rem;
        color: #c7d4e8; /* brighter */
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .kpi-value {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: {ACCENT_GOLD} !important;
        text-shadow: 0 0 6px rgba(0,0,0,0.6);
    }}
    .kpi-sub {{
        font-size: 0.83rem;
        color: #b4c2d6 !important; /* more visible */
        margin-top: 2px;
    }}

    /* ------------------------------------------------------ */
    /* GLASS PANEL                                            */
    /* ------------------------------------------------------ */
    .glass-panel {{
        background: rgba(14,22,40,0.92);
        border-radius: 16px;
        padding: 1rem 1.3rem;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 10px 28px rgba(0,0,0,0.75);
        backdrop-filter: blur(3px);
    }}

    /* ------------------------------------------------------ */
    /* TABLE VISIBILITY FIX                                   */
    /* ------------------------------------------------------ */
    .stDataFrame, .stTable {{
        color: #f0f6ff !important;  /* TEXT FIX */
    }}

    .stDataFrame tbody td {{
        color: #e8f2ff !important;  /* brighter cell text */
        font-size: 0.95rem !important;
    }}

    .stDataFrame thead th {{
        color: #a6c7ff !important; /* brighter headers */
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    .stDataFrame tbody tr:hover {{
        background-color: rgba(255,255,255,0.08) !important;
    }}

    /* ------------------------------------------------------ */
    /* TABS – HIGH VISIBILITY                                 */
    /* ------------------------------------------------------ */
    button[data-baseweb="tab"] {{
        font-size: 0.9rem;
        color: #c6d8ef !important;
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.4rem 1rem;
        border: 1px solid rgba(255,255,255,0.15);
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: #ffffff !important;
        background: rgba(0,162,255,0.28) !important;
        border: 1px solid {ACCENT_CYAN};
        box-shadow: 0 0 8px rgba(0,162,255,0.55);
    }}

    /* ------------------------------------------------------ */
    /* INPUTS / SELECTS                                       */
    /* ------------------------------------------------------ */
    .stNumberInput input, .stSelectbox select {{
        background: rgba(20,34,60,0.95) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# TITLE BANNER
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="title-banner">
        <div class="title-main">📘 Summary Valuation – Weighted Equity Value</div>
        <div class="title-sub">
            FBC dashboard summarising DCF · DDM · EV/EBITDA · PBV · P/E · Banking valuations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# INITIALISE SESSION STATE (PERSISTENT)
# ------------------------------------------------------------------------------
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E", "BANKING"]

if "model_weights" not in st.session_state:
    st.session_state["model_weights"] = {
        "DCF": 35.0,
        "DDM": 20.0,
        "EV/EBITDA": 15.0,
        "PBV": 10.0,
        "P/E": 10.0,
        "BANKING": 10.0,
    }

if "num_shares" not in st.session_state:
    st.session_state["num_shares"] = 0.0

if "current_price" not in st.session_state:
    st.session_state["current_price"] = 0.0

# ------------------------------------------------------------------------------
# MODEL SELECTION (PERSISTENT)
# ------------------------------------------------------------------------------
st.header("📌 Select Models to Include")

all_models = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E", "BANKING"]

selected_models = st.multiselect(
    "Choose models:",
    options=all_models,
    default=st.session_state["selected_models"],
    key="selected_models_input",
)
st.session_state["selected_models"] = selected_models

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# ------------------------------------------------------------------------------
# RETRIEVE VALUATIONS FROM OTHER PAGES (DCF, DDM, COMPARABLES)
# ------------------------------------------------------------------------------
value_map = {
    "DCF": st.session_state.get("equity_value_dcf"),
    "DDM": st.session_state.get("equity_value_ddm"),
    "EV/EBITDA": st.session_state.get("value_ev_ebitda"),
    "PBV": st.session_state.get("value_pbv"),
    "P/E": st.session_state.get("value_pe"),
    "BANKING": st.session_state.get("equity_value_banking"),
}

# ------------------------------------------------------------------------------
# WEIGHT ASSIGNMENT (PERSISTENT)
# ------------------------------------------------------------------------------
st.header("🧮 Assign Weights (%)")

cols = st.columns(len(all_models))
weights_new = {}

for model, col in zip(all_models, cols):
    if model in selected_models:
        with col:
            new_val = st.number_input(
                f"{model} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["model_weights"].get(model, 0)),
                step=1.0,
                key=f"weight_input_{model}",
            )
        weights_new[model] = new_val
    else:
        weights_new[model] = 0.0

st.session_state["model_weights"] = weights_new

total_w = sum(weights_new[m] for m in selected_models)
if total_w == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

weights_normalized = {m: (weights_new[m] / total_w) * 100 for m in selected_models}

# ------------------------------------------------------------------------------
# SUMMARY DATAFRAME
# ------------------------------------------------------------------------------
rows = []
for model in selected_models:
    val = value_map.get(model)
    w = weights_normalized.get(model, 0)
    weighted_value = val * (w / 100) if val is not None else None
    rows.append([model, val, w, weighted_value])

df_summary = pd.DataFrame(
    rows, columns=["Model", "Value (USD)", "Weight (%)", "Weighted Value"]
)

weighted_equity = df_summary["Weighted Value"].sum()

# Extra stats for dashboard
valid_vals = df_summary["Value (USD)"].dropna()
if not valid_vals.empty:
    max_model = df_summary.loc[df_summary["Value (USD)"].idxmax(), "Model"]
    max_val = float(valid_vals.max())
    min_val = float(valid_vals.min())
    spread = max_val - min_val
    dispersion = float(valid_vals.std()) if len(valid_vals) > 1 else 0.0
else:
    max_model, max_val, min_val, spread, dispersion = "-", 0.0, 0.0, 0.0, 0.0

# ------------------------------------------------------------------------------
# KPI STRIP (POWERBI STYLE)
# ------------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Weighted Equity Value</div>
            <div class="kpi-value">{weighted_equity:,.0f}</div>
            <div class="kpi-sub">Total blended equity output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Models</div>
            <div class="kpi-value">{len(selected_models)}</div>
            <div class="kpi-sub">DCF / DDM / Multiples / Banking in use</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with kpi_col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Highest Model</div>
            <div class="kpi-value">{max_model}</div>
            <div class="kpi-sub">Range: {min_val:,.0f} – {max_val:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# TABS: SUMMARY TABLE | INTERACTIVE DASHBOARD
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Summary Table", "📈 Interactive Dashboard"])

# -------- TAB 1: SUMMARY TABLE --------
with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    st.dataframe(
        df_summary.style.format(
            {
                "Value (USD)": lambda x: f"{x:,.2f}" if pd.notnull(x) else "—",
                "Weight (%)": lambda x: f"{x:.0f}%",
                "Weighted Value": lambda x: f"{x:,.2f}" if pd.notnull(x) else "—",
            }
        ),
        width='stretch',
    )

    st.subheader(f"🟩 Weighted Equity Value: **{weighted_equity:,.2f}**")

    st.markdown("</div>", unsafe_allow_html=True)

# -------- TAB 2: INTERACTIVE DASHBOARD --------
with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Model Comparison Dashboard")

    df_chart = df_summary.copy()

    base = (
        alt.Chart(df_chart)
        .properties(background=DARK_BG, height=260)
        .configure_axis(
            labelColor=PRIMARY_TEXT,
            titleColor=PRIMARY_TEXT,
            gridColor="#1f2937",
        )
        .configure_view(strokeOpacity=0)
        .configure_legend(labelColor=PRIMARY_TEXT, titleColor=PRIMARY_TEXT)
    )

    chart_values = (
        base.mark_bar(color=ACCENT_BLUE)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Value (USD):Q", title="Equity Value (USD)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Value (USD):Q", format=",.2f"),
            ],
        )
        .properties(title="Model Equity Values")
    )

    chart_weights = (
        base.mark_bar(color=ACCENT_GOLD)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Weight (%):Q", title="Weight (%)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Weight (%):Q", format=".1f"),
            ],
        )
        .properties(title="Model Weights")
    )

    chart_weighted = (
        base.mark_bar(color=ACCENT_CYAN)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Weighted Value:Q", title="Weighted Equity (USD)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Weighted Value:Q", format=",.2f"),
            ],
        )
        .properties(title="Weighted Contribution by Model", height=280)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(chart_values, width='stretch')
    with c2:
        st.altair_chart(chart_weights, width='stretch')

    st.altair_chart(chart_weighted, width='stretch')

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# NUMBER OF SHARES & PER SHARE VALUE
# ------------------------------------------------------------------------------
st.header("📘 Number of Shares & Per Share Value")

c_sh1, c_sh2 = st.columns([2, 3])

with c_sh1:
    num_shares = st.number_input(
        "Number of Shares in Issue",
        value=float(st.session_state["num_shares"]),
        step=1000.0,
        format="%.0f",
        key="num_shares_input",
    )
    st.session_state["num_shares"] = num_shares

if num_shares > 0:
    intrinsic_value = weighted_equity / num_shares
else:
    intrinsic_value = None

with c_sh2:
    iv_text = f"{intrinsic_value:,.4f}" if intrinsic_value is not None else "—"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Intrinsic Value per Share</div>
            <div class="kpi-value">{iv_text}</div>
            <div class="kpi-sub">Based on blended equity and shares in issue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# CURRENT PRICE & UPSIDE
# ------------------------------------------------------------------------------
st.header("📘 Current Price & Upside")

c_up1, c_up2 = st.columns([2, 3])

with c_up1:
    current_price = st.number_input(
        "Current Share Price (USD)",
        value=float(st.session_state["current_price"]),
        step=0.01,
        key="current_price_input",
    )
    st.session_state["current_price"] = current_price

if intrinsic_value is not None and current_price > 0:
    upside = (intrinsic_value - current_price) / current_price
    st.session_state["upside"] = upside
    upside_pct = upside * 100
else:
    upside = None
    upside_pct = None

with c_up2:
    if upside_pct is not None:
        colour = ACCENT_CYAN if upside_pct >= 0 else DANGER
        label = "Upside" if upside_pct >= 0 else "Downside"
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left-color:{colour};">
                <div class="kpi-title">{label}</div>
                <div class="kpi-value" style="color:{colour};">
                    {upside_pct:.1f}%
                </div>
                <div class="kpi-sub">
                    Versus current market price of {current_price:,.2f} USD
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upside will appear once intrinsic value and current price are available.")

# (Export section intentionally omitted to avoid xlsxwriter dependency)
