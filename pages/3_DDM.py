import streamlit as st
import pandas as pd
import numpy as np
import io
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
import base64
def add_watermark():
    logo_path = Path("assets") / "fbc_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()

        watermark_css = f"""
        <style>

        /* Make watermark very light */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 40;
            left: 50;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{logo_base64}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 1500px;
            opacity: 0.07;   /* 🔥 control watermark visibility here */
            pointer-events: none;
            z-index: 0;
        }}

        .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """
        st.markdown(watermark_css, unsafe_allow_html=True)

add_watermark()

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Dividend Discount Model (DDM)", layout="wide")
# ---------------------------------------------------------
# ✅ FIX SIDEBAR COLLAPSE ARROW (Material Icons)
# ---------------------------------------------------------
st.markdown("""
<style>

/* ✅ Load Material Icons so Streamlit's sidebar collapse icon renders correctly */
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

/* ✅ Make sure ONLY icons use the Material Icons font (prevents 'keyboard_double_arrow_right' text) */
.material-icons, 
span.material-icons,
i.material-icons,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] i {
    font-family: 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ✅ Style the collapse/expand button nicely */
[data-testid="stSidebarCollapseButton"] button {
    background: #003399 !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 999px !important;
    width: 44px !important;
    height: 44px !important;
    box-shadow: 0 6px 18px rgba(0, 51, 153, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    transform: translateY(-1px) !important;
    background: #0047d6 !important;
    box-shadow: 0 10px 22px rgba(0, 71, 214, 0.35) !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    width: 22px !important;
    height: 22px !important;
    fill: white !important;
}
/* ===== SIDEBAR GLASS STYLE ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003399 0%, #001a4d 100%) !important;
    color: white !important;
    border-right: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Remove default sidebar padding spacing */
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small {
  font-family: Georgia, "Times New Roman", serif !important;
}

</style>
""", unsafe_allow_html=True)

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
st.dataframe(df_history, width='stretch')

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

    st.session_state["num_shares"] = float(num_shares)
    st.session_state["equity_value_ddm"] = float(equity_value)
else:
    st.warning("Enter a valid number of shares to compute total equity value.")

# =========================================================
# ✅ FULL DDM EXCEL EXPORT (ALWAYS VISIBLE)
# =========================================================

def _excel_col(n: int) -> str:
    return get_column_letter(n)

def build_full_ddm_excel_model(
    years, dividends,
    g_start, g_end,
    rf, mrp, tax_rate,
    unlevered_beta, de_ratio,
    num_shares,
):
    wb = Workbook()

    BLUE = "003399"
    DARK = "071426"
    LIGHT_BG = "F7FAFF"
    GRID = "D9E2EF"

    thin = Side(style="thin", color=GRID)
    border_all = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_title(ws, title, end_col=6):
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=end_col)
        c = ws.cell(1, 1, title)
        c.font = Font(bold=True, color="FFFFFF", size=14)
        c.fill = PatternFill("solid", fgColor=DARK)
        c.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 26

    def style_header(ws, r, c1, c2):
        for c in range(c1, c2 + 1):
            cell = ws.cell(r, c)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=BLUE)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border_all
        ws.row_dimensions[r].height = 20

    # =========================================================
    # SHEET 1: Dividend History
    # =========================================================
    wsH = wb.active
    wsH.title = "DividendHistory"
    style_title(wsH, "DDM - Step 1: Dividend History", end_col=4)

    wsH["A3"], wsH["B3"] = "Year", "Dividend"
    style_header(wsH, 3, 1, 2)

    r0 = 4
    for i, (y, d) in enumerate(zip(years, dividends)):
        r = r0 + i
        wsH.cell(r, 1, int(y)).border = border_all
        wsH.cell(r, 2, float(d)).border = border_all
        wsH.cell(r, 2).number_format = "0.00000"

    wsH.column_dimensions["A"].width = 10
    wsH.column_dimensions["B"].width = 16
    wsH.freeze_panes = "A4"

    last_row = r0 + len(years) - 1

    # =========================================================
    # SHEET 2: Growth
    # =========================================================
    wsG = wb.create_sheet("Growth")
    style_title(wsG, "DDM - Steps 2 & 3: Growth Range & g", end_col=6)

    wsG["A3"], wsG["B3"] = "Input", "Value"
    style_header(wsG, 3, 1, 2)

    wsG["A4"], wsG["B4"] = "Growth start year", int(g_start)
    wsG["A5"], wsG["B5"] = "Growth end year", int(g_end)

    wsG["A7"], wsG["B7"] = "D_start", (
        '=INDEX(DividendHistory!$B$4:$B$%d, MATCH($B$4, DividendHistory!$A$4:$A$%d, 0))'
        % (last_row, last_row)
    )
    wsG["A8"], wsG["B8"] = "D_end", (
        '=INDEX(DividendHistory!$B$4:$B$%d, MATCH($B$5, DividendHistory!$A$4:$A$%d, 0))'
        % (last_row, last_row)
    )

    wsG["A10"], wsG["B10"] = "Growth rate (g)", (
        '=IF($B$4=$B$5,0,IF($B$7>0,POWER($B$8/$B$7,1/($B$5-$B$4))-1,0.02))'
    )
    wsG["B10"].number_format = "0.00%"

    wsG["A11"], wsG["B11"] = "Next dividend (D1)", "=$B$8*(1+$B$10)"
    wsG["B11"].number_format = "0.00000"

    wsG.column_dimensions["A"].width = 22
    wsG.column_dimensions["B"].width = 28
    wsG.freeze_panes = "A4"

    # =========================================================
    # SHEET 3: Parameters (CAPM)
    # =========================================================
    wsP = wb.create_sheet("Parameters")
    style_title(wsP, "DDM - Step 4: Cost of Equity (CAPM)", end_col=6)

    wsP["A3"], wsP["B3"] = "Parameter", "Value"
    style_header(wsP, 3, 1, 2)

    wsP["A4"], wsP["B4"] = "Risk-free rate (RF)", float(rf)
    wsP["A5"], wsP["B5"] = "Equity risk premium (MRP)", float(mrp)
    wsP["A6"], wsP["B6"] = "Tax rate", float(tax_rate)
    wsP["A7"], wsP["B7"] = "Unlevered beta (βu)", float(unlevered_beta)
    wsP["A8"], wsP["B8"] = "Debt/Equity (D/E)", float(de_ratio)

    wsP["B4"].number_format = "0.00%"
    wsP["B5"].number_format = "0.00%"
    wsP["B6"].number_format = "0.00%"
    wsP["B7"].number_format = "0.0000"
    wsP["B8"].number_format = "0.0000"

    wsP["A10"], wsP["B10"] = "Levered beta (βL)", "=$B$7*(1+(1-$B$6)*$B$8)"
    wsP["B10"].number_format = "0.0000"

    wsP["A11"], wsP["B11"] = "Cost of Equity (Re)", "=$B$4 + $B$10*$B$5"
    wsP["B11"].number_format = "0.00%"

    wsP.column_dimensions["A"].width = 26
    wsP.column_dimensions["B"].width = 18
    wsP.freeze_panes = "A4"

    # =========================================================
    # SHEET 4: Valuation
    # =========================================================
    wsV = wb.create_sheet("Valuation")
    style_title(wsV, "DDM - Steps 5 & 6: Valuation", end_col=6)

    wsV["A3"], wsV["B3"] = "Metric", "Value"
    style_header(wsV, 3, 1, 2)

    wsV["A4"], wsV["B4"] = "g (from Growth sheet)", "=Growth!$B$10"
    wsV["A5"], wsV["B5"] = "D1 (from Growth sheet)", "=Growth!$B$11"
    wsV["A6"], wsV["B6"] = "Re (from Parameters)", "=Parameters!$B$11"
    wsV["B4"].number_format = "0.00%"
    wsV["B5"].number_format = "0.00000"
    wsV["B6"].number_format = "0.00%"

    wsV["A8"], wsV["B8"] = "Equity Value / Share (P0)", "=IF($B$6<=$B$4,NA(),$B$5/($B$6-$B$4))"
    wsV["B8"].number_format = "#,##0.0000"

    wsV["A10"], wsV["B10"] = "Number of shares", float(num_shares)
    wsV["B10"].number_format = "#,##0"

    wsV["A11"], wsV["B11"] = "Total Equity Value", "=IF(ISNUMBER($B$8),$B$8*$B$10,NA())"
    wsV["B11"].number_format = "#,##0.00"

    wsV.column_dimensions["A"].width = 28
    wsV.column_dimensions["B"].width = 22
    wsV.freeze_panes = "A4"

    # =========================================================
    # SHEET 5: Summary
    # =========================================================
    wsS = wb.create_sheet("Summary")
    style_title(wsS, "DDM Summary", end_col=6)

    wsS["A3"], wsS["B3"], wsS["C3"] = "Metric", "Value", "Unit"
    style_header(wsS, 3, 1, 3)

    rows = [
        ("Growth rate (g)", "=Growth!$B$10", "%"),
        ("Next dividend (D1)", "=Growth!$B$11", "USD"),
        ("Cost of Equity (Re)", "=Parameters!$B$11", "%"),
        ("Value per share (P0)", "=Valuation!$B$8", "USD"),
        ("Number of shares", "=Valuation!$B$10", "shares"),
        ("Total equity value", "=Valuation!$B$11", "USD"),
    ]

    r0 = 4
    for i, (m, v, u) in enumerate(rows):
        r = r0 + i
        wsS.cell(r, 1, m).border = border_all
        wsS.cell(r, 2, v).border = border_all
        wsS.cell(r, 3, u).border = border_all
        if u == "USD":
            wsS.cell(r, 2).number_format = "#,##0.00"
        elif u == "%":
            wsS.cell(r, 2).number_format = "0.00%"
        elif u == "shares":
            wsS.cell(r, 2).number_format = "#,##0"

    wsS.column_dimensions["A"].width = 26
    wsS.column_dimensions["B"].width = 18
    wsS.column_dimensions["C"].width = 10
    wsS.freeze_panes = "A4"

    return wb

def workbook_to_bytes(wb: Workbook) -> bytes:
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.read()

st.markdown("---")
st.subheader("⬇️ Download FULL DDM Excel Model (All Steps + Formulas)")

if "ddm_excel_bytes" not in st.session_state:
    st.session_state["ddm_excel_bytes"] = None

if st.button("📥 Generate / Update FULL DDM Excel Model", key="ddm_generate_excel"):
    wb = build_full_ddm_excel_model(
        years=years,
        dividends=dividends,
        g_start=int(g_start),
        g_end=int(g_end),
        rf=float(rf),
        mrp=float(mrp),
        tax_rate=float(tax_rate),
        unlevered_beta=float(unlevered_beta),
        de_ratio=float(de_ratio),
        num_shares=float(num_shares),
    )
    st.session_state["ddm_excel_bytes"] = workbook_to_bytes(wb)

st.download_button(
    "⬇️ Download FULL_DDM_Model.xlsx",
    data=st.session_state["ddm_excel_bytes"] or b"",
    file_name="FULL_DDM_Model.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    disabled=st.session_state["ddm_excel_bytes"] is None,
    key="ddm_download_excel",
)
