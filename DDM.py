import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Dividend Discount Model (DDM)", layout="wide")
st.title("📈 Dividend Discount Model (DDM)")

st.markdown(
    """
This module values equity using the Gordon Growth DDM:

### **P₀ = D₁ / (Re − g)**  

Where:  
- **D₁** = Dividend next year  
- **Re** = Cost of Equity  
- **g** = Long-term dividend growth rate  
"""
)


# ---------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------
def init(key, value):
    """Initialize a session_state key once."""
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# STEP 1 — DIVIDEND HISTORY
# ---------------------------------------------------------
st.header("📘 Step 1 — Dividend History")

# Initialise once if missing
init("ddm_start_year", 2021)
init("ddm_end_year", 2025)

col1, col2 = st.columns(2)

with col1:
    start_year_input = st.number_input(
        "Start Year",
        value=int(st.session_state["ddm_start_year"]),
        step=1,
        key="ddm_start_year_input",
    )
    st.session_state["ddm_start_year"] = int(start_year_input)

with col2:
    end_year_input = st.number_input(
        "End Year",
        value=int(st.session_state["ddm_end_year"]),
        step=1,
        key="ddm_end_year_input",
    )
    st.session_state["ddm_end_year"] = int(end_year_input)

start_year = st.session_state["ddm_start_year"]
end_year = st.session_state["ddm_end_year"]

if start_year > end_year:
    st.error("❌ Start year cannot be greater than end year.")
    st.stop()

years = list(range(start_year, end_year + 1))

# Persistent dividend storage per year
for y in years:
    if f"ddm_div_{y}" not in st.session_state:
        st.session_state[f"ddm_div_{y}"] = 0.01  # default once

st.subheader("Enter Dividends")

dividends = []
for y in years:
    div = st.number_input(
        f"Dividend for {y}",
        value=float(st.session_state[f"ddm_div_{y}"]),
        step=0.00001,
        format="%.5f",
        key=f"ddm_div_input_{y}",
    )
    st.session_state[f"ddm_div_{y}"] = div
    dividends.append(div)

# Store full dividend history for AI / summary pages
st.session_state["ddm_dividends"] = {
    str(y): float(d) for y, d in zip(years, dividends)
}

# Display table
df_history = pd.DataFrame({"Year": years, "Dividend": dividends})
st.dataframe(df_history, use_container_width=True)


# ---------------------------------------------------------
# STEP 2 — GROWTH CALCULATION RANGE
# ---------------------------------------------------------
st.header("📘 Step 2 — Growth Calculation Range")

init("ddm_g_start", years[0])
init("ddm_g_end", years[-1])

c1, c2 = st.columns(2)
with c1:
    g_start = st.selectbox("Growth start year:", years, key="ddm_g_start")
with c2:
    g_end = st.selectbox("Growth end year:", years, key="ddm_g_end")

if g_start > g_end:
    st.error("❌ Growth start year must be earlier or equal to end year.")
    st.stop()

D_start = dividends[years.index(g_start)]
D_end = dividends[years.index(g_end)]


# ---------------------------------------------------------
# STEP 3 — DIVIDEND GROWTH RATE (g)
# ---------------------------------------------------------
st.header("📘 Step 3 — Dividend Growth")

if g_start == g_end:
    g = 0.0
elif D_start > 0:
    # CAGR between selected years
    g = (D_end / D_start) ** (1 / (g_end - g_start)) - 1
else:
    # Fallback if starting dividend is zero
    g = 0.02

st.success(f"Growth rate (g): **{g:.2%}**")

D1 = D_end * (1 + g)
st.metric("Next year's dividend (D₁)", f"{D1:,.5f}")


# ---------------------------------------------------------
# STEP 4 — COST OF EQUITY (Re)
# ---------------------------------------------------------
st.header("📘 Step 4 — Cost of Equity Inputs")

# Pull live values from DCF page where possible
rf = st.session_state.get("dcf_rf_pct", st.session_state.get("rf", 0.0)) / 100
mrp = st.session_state.get("dcf_mrp_pct", st.session_state.get("erp", 0.0)) / 100
tax_rate = (
    st.session_state.get("dcf_tax_pct", st.session_state.get("tax_rate", 0.0)) / 100
)
unlevered_beta = st.session_state.get(
    "dcf_unlevered_beta", st.session_state.get("unlevered_beta", 0.0)
)
de_ratio = st.session_state.get("de_ratio", 0.0)

# Store back normalised keys
st.session_state["rf"] = rf
st.session_state["erp"] = mrp
st.session_state["tax_rate"] = tax_rate
st.session_state["unlevered_beta"] = unlevered_beta

st.write("Using parameters loaded from the DCF page (you can override them below).")

use_custom = st.checkbox(
    "Manually override parameters",
    value=st.session_state.get("ddm_use_custom_params", False),
    key="ddm_use_custom_params",
)

if use_custom:
    cA, cB = st.columns(2)

    with cA:
        unlevered_beta = st.number_input(
            "Unlevered Beta",
            value=float(unlevered_beta),
            step=0.001,
            format="%.4f",
            key="ddm_unlevered_beta",
        )

        de_ratio = st.number_input(
            "Debt/Equity Ratio (D/E)",
            value=float(de_ratio),
            step=0.001,
            format="%.4f",
            key="ddm_de_ratio",
        )

    with cB:
        tax_rate = (
            st.number_input(
                "Tax Rate (%)",
                value=float(tax_rate * 100),
                step=0.01,
                format="%.2f",
                key="ddm_tax_rate",
            )
            / 100
        )

        rf = (
            st.number_input(
                "Risk-Free Rate (%)",
                value=float(rf * 100),
                step=0.01,
                format="%.2f",
                key="ddm_rf",
            )
            / 100
        )

        mrp = (
            st.number_input(
                "Equity Risk Premium (%)",
                value=float(mrp * 100),
                step=0.01,
                format="%.2f",
                key="ddm_erp",
            )
            / 100
        )

# Save final values
st.session_state["rf"] = rf
st.session_state["erp"] = mrp
st.session_state["tax_rate"] = tax_rate
st.session_state["unlevered_beta"] = unlevered_beta
st.session_state["de_ratio"] = de_ratio

# CAPM Re
levered_beta = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
Re = rf + levered_beta * mrp

st.metric("Levered Beta", f"{levered_beta:.4f}")
st.metric("Cost of Equity (Re)", f"{Re * 100:.2f}%")


# ---------------------------------------------------------
# STEP 5 — VALUE PER SHARE
# ---------------------------------------------------------
st.header("📘 Step 5 — Equity Value per Share")

if Re <= g:
    st.error("❌ Re must be greater than g for the Gordon Growth DDM to work.")
    P0 = np.nan
else:
    P0 = D1 / (Re - g)
    st.success(f"Equity Value / Share = **{P0:,.4f} USD**")

# Store for AI / summary pages
st.session_state["ddm_g"] = float(g)
st.session_state["ddm_Re"] = float(Re)
st.session_state["ddm_P0"] = None if np.isnan(P0) else float(P0)


# ---------------------------------------------------------
# STEP 6 — TOTAL EQUITY VALUE
# ---------------------------------------------------------
st.header("📘 Step 6 — Total Equity Value")

init("num_shares", 0.0)

num_shares = st.number_input(
    "Number of Shares",
    value=float(st.session_state["num_shares"]),
    step=1000.0,
    format="%.0f",
    key="ddm_num_shares",
)

if num_shares > 0 and not np.isnan(P0):
    equity_value = P0 * num_shares
    st.success(f"Total Equity Value = **{equity_value:,.2f} USD**")

    # Save for use on Summary page and elsewhere
    st.session_state["num_shares"] = float(num_shares)
    st.session_state["equity_value_ddm"] = float(equity_value)
else:
    st.warning("Enter a valid number of shares to compute total equity value.")
