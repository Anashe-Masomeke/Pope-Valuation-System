import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import valuation_ai as vai   # AI ENGINE

# --------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------
st.set_page_config(page_title="Summary Valuation", layout="wide")

# --------------------------------------------------------------
# DARK THEME STYLE
# --------------------------------------------------------------
DARK_BG = "#020617"
ACCENT_BLUE = "#38bdf8"
ACCENT_CYAN = "#0ea5e9"
ACCENT_GOLD = "#fbbf24"
DANGER = "#f97373"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {DARK_BG};
        color: white;
    }}
    .kpi-card {{
        border-radius: 14px;
        padding: 1.15rem;
        background: rgba(10,20,40,0.85);
        border: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 10px;
    }}
    .kpi-title {{
        font-size: 0.85rem;
        color: #c7d4e8;
        text-transform: uppercase;
    }}
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {ACCENT_GOLD};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# TITLE
# --------------------------------------------------------------
st.markdown(
    """
    <div style='background:#0a1b33;padding:18px;border-radius:12px;
    border:1px solid rgba(255,255,255,0.15);'>
        <h2 style='margin:0;color:white;'>📘 Summary Valuation – Blended Model Output</h2>
        <p style='color:#cdd7e6;margin-top:6px;'>Combined results from DCF, DDM & Multiples.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# INITIAL STATE
# --------------------------------------------------------------
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E"]

if "model_weights" not in st.session_state:
    st.session_state["model_weights"] = {
        "DCF": 40.0,
        "DDM": 20.0,
        "EV/EBITDA": 20.0,
        "PBV": 10.0,
        "P/E": 10.0,
    }

if "num_shares" not in st.session_state:
    st.session_state["num_shares"] = 0.0

# --------------------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------------------
st.header("📌 Select Models to Include")

all_models = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E"]

selected_models = st.multiselect(
    "Choose models:",
    options=all_models,
    default=st.session_state["selected_models"],
)
st.session_state["selected_models"] = selected_models

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# --------------------------------------------------------------
# COLLECT VALUATION VALUES
# --------------------------------------------------------------
value_map = {
    "DCF": st.session_state.get("equity_value_dcf"),
    "DDM": st.session_state.get("equity_value_ddm"),
    "EV/EBITDA": st.session_state.get("value_ev_ebitda"),
    "PBV": st.session_state.get("value_pbv"),
    "P/E": st.session_state.get("value_pe"),
}

# --------------------------------------------------------------
# WEIGHTS
# --------------------------------------------------------------
st.header("🧮 Assign Weights (%)")

cols = st.columns(5)
new_weights = {}

for model, col in zip(all_models, cols):
    if model in selected_models:
        with col:
            w = st.number_input(
                f"{model} Weight (%)",
                value=float(st.session_state["model_weights"].get(model, 0)),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
            )
        new_weights[model] = w
    else:
        new_weights[model] = 0.0

st.session_state["model_weights"] = new_weights

total_weight = sum(new_weights[m] for m in selected_models)
if total_weight == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

weights_normalized = {m: (new_weights[m] / total_weight) * 100 for m in selected_models}

# --------------------------------------------------------------
# SUMMARY TABLE + SAFE FORMATTER
# --------------------------------------------------------------
def safe_format_number(x):
    if x is None:
        return "—"
    try:
        if np.isnan(x):
            return "—"
    except:
        pass
    return f"{x:,.2f}"

rows = []
for model in selected_models:
    val = value_map.get(model)
    weight = weights_normalized.get(model, 0)
    weighted_val = None if val is None else val * (weight / 100)
    rows.append([model, val, weight, weighted_val])

df_summary = pd.DataFrame(
    rows,
    columns=["Model", "Value (USD)", "Weight (%)", "Weighted Contribution"],
)

weighted_equity = df_summary["Weighted Contribution"].sum(skipna=True)
st.session_state["weighted_equity"] = weighted_equity

st.subheader("📊 Valuation Breakdown")

st.dataframe(
    df_summary.style.format(
        {
            "Value (USD)": safe_format_number,
            "Weight (%)": lambda x: f"{x:.0f}%" if x is not None else "—",
            "Weighted Contribution": safe_format_number,
        }
    ),
    use_container_width=True,
)
# --------------------------------------------------------------
# KPI STRIP — Weighted Equity + Active Models
# --------------------------------------------------------------
kpi1, kpi2 = st.columns(2)

with kpi1:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Weighted Equity Value</div>
            <div class='kpi-value'>{weighted_equity:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Active Models</div>
            <div class='kpi-value'>{len(selected_models)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------
# INTRINSIC VALUE PER SHARE
# --------------------------------------------------------------
st.header("📘 Intrinsic Value per Share")

num_shares = st.number_input(
    "Number of Shares in Issue",
    value=float(st.session_state["num_shares"]),
    step=1000.0,
    format="%.0f",
)
st.session_state["num_shares"] = num_shares

intrinsic_value = (
    weighted_equity / num_shares if num_shares > 0 else None
)

st.session_state["intrinsic_value"] = intrinsic_value

iv_text = "N/A" if intrinsic_value is None else f"{intrinsic_value:,.4f}"

st.markdown(
    f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Intrinsic Value per Share</div>
        <div class='kpi-value'>{iv_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# UPSIDE / DOWNSIDE
# --------------------------------------------------------------
st.header("📘 Market Price & Upside")

current_price = st.number_input(
    "Current Share Price (USD)",
    value=float(st.session_state.get("current_price", 0)),
    step=0.01,
)
st.session_state["current_price"] = current_price

if intrinsic_value is not None and current_price > 0:
    upside = (intrinsic_value - current_price) / current_price
    st.session_state["upside"] = upside
    up_pct = upside * 100
else:
    up_pct = None
    st.session_state["upside"] = None

color = ACCENT_CYAN if up_pct and up_pct >= 0 else DANGER
label = "Upside" if up_pct and up_pct >= 0 else "Downside"

up_text = "N/A" if up_pct is None else f"{up_pct:.1f}%"

st.markdown(
    f"""
    <div class='kpi-card' style='border-left:4px solid {color};'>
        <div class='kpi-title'>{label}</div>
        <div class='kpi-value' style='color:{color};'>{up_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------
# 🧠 AI INSIGHTS
# --------------------------------------------------------------
st.markdown("---")
st.header("🧠 AI Valuation Intelligence Report")

analysis = vai.generate_commentary()

short = analysis["short"]
long = analysis["long"]
model_scores = analysis["model_scores"]
risk = analysis["risk"]

# KPIs
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Financial Quality Score</div>
            <div class='kpi-value'>{risk['financial_score']:.0f}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Risk Rating</div>
            <div class='kpi-value'>{risk['risk_label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    up = risk["upside_pct"]
    up_text = "N/A" if up is None else f"{up:.0f}%"
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Upside / Downside</div>
            <div class='kpi-value'>{up_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Short summary
st.subheader("🔹 AI Summary (Short)")
st.write(short)

# Detailed commentary
st.subheader("🔹 Full Analyst Commentary")
st.write(long)

# Model reliability table
st.subheader("🔹 Model Reliability Scores")
st.dataframe(
    pd.DataFrame(
        [{"Model": k, "Reliability Score": v} for k, v in model_scores.items()]
    ),
    use_container_width=True,
)
