import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# Helper: nicely format numeric columns in a DF
# =========================================================
def format_numeric_columns(df: pd.DataFrame):
    fmt = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = "{:,.2f}"
    return df.style.format(fmt)


# =========================================================
# Helper: filtered average for multiples (outlier-robust)
# =========================================================
def filtered_average(values, band: float = 0.4):
    """
    Take an average after trimming values that are too far from the median.
    """
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr != 0]

    if len(arr) == 0:
        return np.nan

    median = np.median(arr)
    lower = median * (1 - band)
    upper = median * (1 + band)
    keep = arr[(arr >= lower) & (arr <= upper)]
    return float(np.mean(keep if len(keep) > 0 else arr))


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Comparables Valuation (Excel Style)", layout="wide")
st.title("📊 Comparables Valuation – EV/EBITDA, P/B, P/E")

st.markdown(
    """
All values & inputs are **saved in `session_state`**, so switching pages (DCF, DDM, etc.)
does **not** reset anything on this page.
"""
)

# Short alias
S = st.session_state


# =========================================================
# STEP 1 — INPUT COMPARABLE COMPANIES & MULTIPLES
# =========================================================
st.header("Step 1 — Input Comparable Companies & Multiples")

default_num_comps = int(S.get("num_comps", 3))
num_comps = st.number_input(
    "How many comparables?",
    min_value=1,
    max_value=20,
    value=default_num_comps,
    key="num_comps_input",
)
S["num_comps"] = int(num_comps)

# Ensure structure for comps
if "comps" not in S:
    S["comps"] = {}

for i in range(int(num_comps)):
    S["comps"].setdefault(
        i,
        {
            "name": f"Comp {i + 1}",
            "ev": 0.0,
            "pb": 0.0,
            "pe": 0.0,
        },
    )

rows_comps = []
for i in range(int(num_comps)):
    st.subheader(f"Comparable {i + 1}")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

    with c1:
        name_val = st.text_input(
            f"Company {i + 1} name",
            value=S["comps"][i]["name"],
            key=f"comp_name_{i}",
        )
        S["comps"][i]["name"] = name_val

    with c2:
        ev_val = st.number_input(
            f"{name_val} EV/EBITDA",
            value=float(S["comps"][i]["ev"]),
            step=0.01,
            format="%.2f",
            key=f"comp_ev_{i}",
        )
        S["comps"][i]["ev"] = ev_val

    with c3:
        pb_val = st.number_input(
            f"{name_val} P/B",
            value=float(S["comps"][i]["pb"]),
            step=0.01,
            format="%.2f",
            key=f"comp_pb_{i}",
        )
        S["comps"][i]["pb"] = pb_val

    with c4:
        pe_val = st.number_input(
            f"{name_val} P/E",
            value=float(S["comps"][i]["pe"]),
            step=0.01,
            format="%.2f",
            key=f"comp_pe_{i}",
        )
        S["comps"][i]["pe"] = pe_val

    rows_comps.append([name_val, ev_val, pb_val, pe_val])

df_comps = pd.DataFrame(rows_comps, columns=["Company", "EV/EBITDA", "P/B", "P/E"])
st.subheader("Entered Comparables")
st.dataframe(df_comps, use_container_width=True)

# Save for summary page
S["comps_num"] = int(num_comps)
S["comps_ev_list"] = df_comps["EV/EBITDA"].astype(float).tolist()
S["comps_pb_list"] = df_comps["P/B"].astype(float).tolist()
S["comps_pe_list"] = df_comps["P/E"].astype(float).tolist()


# =========================================================
# STEP 2 — AVERAGE & IMPLIED MULTIPLES
# =========================================================
st.header("Step 2 — Average & Implied Multiples")

avg_ev = filtered_average(df_comps["EV/EBITDA"])
avg_pb = filtered_average(df_comps["P/B"])
avg_pe = filtered_average(df_comps["P/E"])

default_discount_pct = float(S.get("discount_pct", 25.0))
discount_pct = st.number_input(
    "Discount factor (%)",
    value=default_discount_pct,
    step=1.0,
    key="discount_pct_input",
)
S["discount_pct"] = discount_pct
discount = discount_pct / 100.0

implied_ev = avg_ev * (1 - discount)
implied_pb = avg_pb * (1 - discount)
implied_pe = avg_pe * (1 - discount)

df_mult = pd.DataFrame(
    {
        "Multiple": ["EV/EBITDA", "P/B", "P/E"],
        "Average": [avg_ev, avg_pb, avg_pe],
        "Discount (%)": [discount_pct] * 3,
        "Implied": [implied_ev, implied_pb, implied_pe],
    }
)
st.dataframe(
    df_mult.style.format({"Average": "{:,.2f}", "Implied": "{:,.2f}"}),
    use_container_width=True,
)

# Persist implied multiples for summary
S["implied_ev"] = float(implied_ev)
S["implied_pb"] = float(implied_pb)
S["implied_pe"] = float(implied_pe)


# =========================================================
# TIMING SOURCE (from DCF) — BASE USED BY BOTH EBITDA & EARNINGS
# =========================================================
st.header("Timing Source (from DCF)")

dcf_timing_list = S.get("dcf_discount_periods_n", [])

# Default base timing from state (if already chosen)
default_base = float(S.get("comp_timing_base", 0.0))

if not dcf_timing_list:
    st.warning(
        "⚠ No timing values detected from DCF. "
        "Either run the DCF model first or set a manual timing base."
    )

    base_timing = st.number_input(
        "Enter starting timing value for comparables (year 1):",
        value=default_base,
        step=0.01,
        format="%.4f",
        key="comp_timing_base_manual_no_dcf",
    )
else:
    # Show full DCF timing table (for info)
    timing_df = pd.DataFrame(
        {
            "Forecast Year Index": list(range(len(dcf_timing_list))),
            "DCF Timing n": dcf_timing_list,
        }
    )
    st.dataframe(timing_df, use_container_width=True)

    dcf_n0 = float(round(dcf_timing_list[0], 4))
    st.info(f"DCF First Timing Value (n₀) = **{dcf_n0} years**")

    timing_choice = st.radio(
        "Choose timing base for Comparables timing effect:",
        [
            f"Use DCF n₀ = {dcf_n0}",
            "Manually override starting timing value",
        ],
        index=0 if default_base == 0.0 or np.isclose(default_base, dcf_n0) else 1,
        key="comp_timing_choice",
    )

    if timing_choice.startswith("Use DCF"):
        base_timing = dcf_n0
    else:
        base_timing = st.number_input(
            "Enter starting timing value for comparables (year 1):",
            value=default_base if default_base != 0.0 else dcf_n0,
            step=0.01,
            format="%.4f",
            key="comp_timing_base_manual",
        )

# Save chosen base timing (used below for EBITDA & Earnings)
S["comp_timing_base"] = float(base_timing)
st.success(f"Timing base for comparables = **{base_timing:.4f}**")


# =========================================================
# STEP 3 — MAINTAINABLE EBITDA (with locked timing)
# =========================================================
st.header("Step 3 — Maintainable EBITDA")

dcf_eb_all = S.get("dcf_ebitda_all", None)
if dcf_eb_all is None:
    dcf_eb_all = S.get("dcf_ebitda_forecast", {})

if not dcf_eb_all:
    st.error("⚠ No EBITDA found in DCF. Please run the DCF page first.")
    st.stop()

# All EBITDA years from DCF (hist + forecast)
eb_years_all = sorted(int(y) for y in dcf_eb_all.keys())
eb_min_year = min(eb_years_all)
eb_max_year = max(eb_years_all)

# Persist controls
S.setdefault("comp_eb_start_year", eb_min_year)
S.setdefault("comp_eb_end_year", eb_max_year)
S.setdefault("comp_eb_weights", {})      # {year: weight}
S.setdefault("comp_use_timing_eb", True)

use_timing_eb = st.checkbox(
    "Apply timing effect from DCF to EBITDA?",
    value=bool(S.get("comp_use_timing_eb", True)),
    key="comp_use_timing_eb_checkbox",
)
S["comp_use_timing_eb"] = use_timing_eb
# New toggle — Apply timing only to first year
use_first_year_only = st.checkbox(
    "Apply timing ONLY to the first year?",
    value=bool(S.get("comp_use_first_year_only_eb", False)),
    key="comp_use_first_year_only_eb_checkbox",
)
S["comp_use_first_year_only_eb"] = use_first_year_only

c_eb1, c_eb2 = st.columns(2)
with c_eb1:
    eb_start_year = st.number_input(
        "EBITDA Start Year",
        value=int(S["comp_eb_start_year"]),
        step=1,
        key="comp_eb_start_year_input",
    )
with c_eb2:
    eb_end_year = st.number_input(
        "EBITDA End Year",
        value=int(S["comp_eb_end_year"]),
        step=1,
        key="comp_eb_end_year_input",
    )

# Clamp to valid range
eb_start_year = int(max(eb_start_year, eb_min_year))
eb_end_year = int(min(eb_end_year, eb_max_year))
if eb_end_year < eb_start_year:
    st.error("❌ EBITDA End Year must be ≥ Start Year.")
    st.stop()

S["comp_eb_start_year"] = eb_start_year
S["comp_eb_end_year"] = eb_end_year

selected_eb_years = list(range(eb_start_year, eb_end_year + 1))

st.subheader("EBITDA Weighting")
rows_eb = []

# Use the global timing base picked above
base_timing = float(S.get("comp_timing_base", 0.0))

for idx, yr in enumerate(selected_eb_years):
    eb_val = float(dcf_eb_all.get(str(yr), 0.0))
    default_w = float(S["comp_eb_weights"].get(str(yr), 0.0))

    # Calculate timing internally (NOT displayed)
    if not use_timing_eb:
        timing_val = 1.0
    elif use_first_year_only:
        timing_val = base_timing if idx == 0 else 1.0
    else:
        timing_val = base_timing + idx

    # UI layout but WITHOUT timing input
    c1, c2, c4 = st.columns([1, 2, 1])

    with c1:
        st.number_input(
            f"EB Year {yr}",
            value=int(yr),
            disabled=True,
            key=f"comp_eb_year_display_{yr}",
        )

    with c2:
        st.number_input(
            f"EBITDA {yr}",
            value=eb_val,
            disabled=True,
            format="%.2f",
            key=f"comp_eb_value_display_{yr}",
        )

    with c4:
        weight_val = st.number_input(
            f"EB Weight {yr} (%)",
            value=float(default_w),
            step=0.1,
            format="%.2f",
            key=f"comp_eb_weight_{yr}",
        )

    S["comp_eb_weights"][str(yr)] = float(weight_val)

    adj_eb = eb_val * timing_val
    weighted_eb = adj_eb * weight_val / 100.0

    rows_eb.append(
        {
            "Year": int(yr),
            "EBITDA": eb_val,
            "Timing": timing_val if use_timing_eb else np.nan,
            "Weight (%)": weight_val,
            "Adjusted EBITDA": adj_eb,
            "Weighted EBITDA": weighted_eb,
        }
    )

df_eb = pd.DataFrame(rows_eb)

# Display table (hide timing columns when toggle is off)
if use_timing_eb:
    df_eb_display = df_eb[
        ["Year", "EBITDA", "Timing", "Weight (%)", "Adjusted EBITDA", "Weighted EBITDA"]
    ]
else:
    df_eb_display = df_eb[["Year", "EBITDA", "Weight (%)", "Weighted EBITDA"]]

df_eb_display = df_eb_display.copy()
df_eb_display.index = df_eb_display.index + 1  # start index at 1

st.dataframe(format_numeric_columns(df_eb_display), use_container_width=True)

maintainable_ebitda = float(df_eb["Weighted EBITDA"].sum())
st.success(f"Maintainable EBITDA = {maintainable_ebitda:,.2f}")
S["maintainable_ebitda"] = maintainable_ebitda


# =========================================================
# STEP 4 — MAINTAINABLE EARNINGS (with locked timing)
# =========================================================
st.header("Step 4 — Maintainable Earnings")

dcf_np_all = S.get("dcf_profit_all", None)
if dcf_np_all is None:
    dcf_np_all = S.get("dcf_profit_forecast", {})

if not dcf_np_all:
    st.error("⚠ No Profit-for-the-Year found in DCF.")
    st.stop()

np_years_all = sorted(int(y) for y in dcf_np_all.keys())
np_min_year = min(np_years_all)
np_max_year = max(np_years_all)

# Persist controls
S.setdefault("comp_np_start_year", np_min_year)
S.setdefault("comp_np_end_year", np_max_year)
S.setdefault("comp_np_weights", {})
S.setdefault("comp_use_timing_np", True)

use_timing_np = st.checkbox(
    "Apply timing effect from DCF to Earnings?",
    value=bool(S.get("comp_use_timing_np", True)),
    key="comp_use_timing_np_checkbox",
)
S["comp_use_timing_np"] = use_timing_np
use_first_year_only_np = st.checkbox(
    "Apply timing ONLY to the first year? (Earnings)",
    value=bool(S.get("comp_use_first_year_only_np", False)),
    key="comp_use_first_year_only_np_checkbox",
)
S["comp_use_first_year_only_np"] = use_first_year_only_np

c_np1, c_np2 = st.columns(2)
with c_np1:
    np_start_year = st.number_input(
        "NP Start Year",
        value=int(S["comp_np_start_year"]),
        step=1,
        key="comp_np_start_year_input",
    )
with c_np2:
    np_end_year = st.number_input(
        "NP End Year",
        value=int(S["comp_np_end_year"]),
        step=1,
        key="comp_np_end_year_input",
    )

np_start_year = int(max(np_start_year, np_min_year))
np_end_year = int(min(np_end_year, np_max_year))
if np_end_year < np_start_year:
    st.error("❌ NP End Year cannot be before Start Year.")
    st.stop()

S["comp_np_start_year"] = np_start_year
S["comp_np_end_year"] = np_end_year

selected_np_years = list(range(np_start_year, np_end_year + 1))

st.subheader("Earnings Weighting")
rows_np = []

# Same timing base as EBITDA
base_timing = float(S.get("comp_timing_base", 0.0))

for idx, yr in enumerate(selected_np_years):
    np_val = float(dcf_np_all.get(str(yr), 0.0))
    default_w = float(S["comp_np_weights"].get(str(yr), 0.0))

    # Internal timing calculation
    if not use_timing_np:
        timing_val = 1.0
    elif use_first_year_only_np:
        timing_val = base_timing if idx == 0 else 1.0
    else:
        timing_val = base_timing + idx

    c1, c2, c4 = st.columns([1, 2, 1])

    with c1:
        st.number_input(
            f"Earnings Year {yr}",
            value=int(yr),
            disabled=True,
            key=f"comp_np_year_display_{yr}",
        )

    with c2:
        st.number_input(
            f"Earnings {yr}",
            value=np_val,
            disabled=True,
            format="%.2f",
            key=f"comp_np_value_display_{yr}",
        )

    with c4:
        weight_val = st.number_input(
            f"NP Weight {yr} (%)",
            value=float(default_w),
            step=0.1,
            format="%.2f",
            key=f"comp_np_weight_{yr}",
        )

    S["comp_np_weights"][str(yr)] = float(weight_val)

    adj_np = np_val * timing_val
    weighted_np = adj_np * weight_val / 100.0

    rows_np.append(
        {
            "Year": int(yr),
            "Earnings": np_val,
            "Timing": timing_val if use_timing_np else np.nan,
            "Weight (%)": weight_val,
            "Adjusted Earnings": adj_np,
            "Weighted Earnings": weighted_np,
        }
    )

df_np = pd.DataFrame(rows_np)

if use_timing_np:
    df_np_display = df_np[
        ["Year", "Earnings", "Timing", "Weight (%)", "Adjusted Earnings", "Weighted Earnings"]
    ]
else:
    df_np_display = df_np[["Year", "Earnings", "Weight (%)", "Weighted Earnings"]]

df_np_display = df_np_display.copy()
df_np_display.index = df_np_display.index + 1

st.dataframe(format_numeric_columns(df_np_display), use_container_width=True)

maintainable_earnings = float(df_np["Weighted Earnings"].sum())
st.success(f"Maintainable Earnings = {maintainable_earnings:,.2f}")
S["maintainable_earnings"] = maintainable_earnings


# =========================================================
# STEP 5 — BOOK VALUE & NET DEBT
# =========================================================
st.header("Step 5 — Book Value & Net Debt")

book_equity_default = float(S.get("book_equity", 0.0))
net_debt_default = float(S.get("net_debt", 0.0))

book_equity = st.number_input(
    "Book Equity (USD)",
    value=book_equity_default,
    step=0.01,
    format="%.2f",
    key="book_equity_input",
)
S["book_equity"] = float(book_equity)

net_debt = st.number_input(
    "Net Debt (USD)",
    value=net_debt_default,
    step=0.01,
    format="%.2f",
    key="net_debt_input",
)
S["net_debt"] = float(net_debt)


# =========================================================
# STEP 6 — FINAL EQUITY VALUES
# =========================================================
st.header("Step 6 — Computed Equity Values")

maintainable_ebitda = float(S.get("maintainable_ebitda", 0.0))
maintainable_earnings = float(S.get("maintainable_earnings", 0.0))

equity_ev = implied_ev * maintainable_ebitda - net_debt
equity_pb = implied_pb * book_equity
equity_pe = implied_pe * maintainable_earnings

S["value_ev_ebitda"] = float(equity_ev)
S["value_pbv"] = float(equity_pb)
S["value_pe"] = float(equity_pe)

df_res = pd.DataFrame(
    {
        "Method": ["EV/EBITDA", "P/B", "P/E"],
        "Equity Value (USD)": [equity_ev, equity_pb, equity_pe],
    }
)

st.dataframe(format_numeric_columns(df_res), use_container_width=True)
