import streamlit as st
from pathlib import Path
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
st.set_page_config(page_title="Help & Guide", layout="wide")
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

# ---------------------------------------------------------
# STYLES (GEORGIA FONT + BLUISH THEME)
# ---------------------------------------------------------
st.markdown(
    """
    <style>

      /* ===== GLOBAL FONT + COLOR ===== */
        html, body, .stApp, .block-container,
        p, div, label,
        h1, h2, h3, h4, h5, h6,
        li, ul, ol, a, small {
          font-family: Georgia, "Times New Roman", serif !important;
        }

      /* ===== TITLE ===== */
      .main-title {
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #1e3a8a;
      }

      .subtle {
        color: #3b82f6;   /* lighter blue subtitle */
        margin-top: 0;
      }


      /* ===== CARD STYLE ===== */
      .card {
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 16px 18px;
        background: #f8fbff;
        box-shadow: 0 4px 18px rgba(30, 58, 138, 0.15);
        margin-bottom: 14px;
      }

      .card h3 {
        margin: 0 0 8px 0;
        font-size: 1.1rem;
        color: #1e40af;
      }

      /* ===== PILL LABEL ===== */
      .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.85rem;
        background: #e0ecff;
        border: 1px solid #bfdbfe;
        margin-right: 8px;
        color: #1e3a8a;
      }

      /* ===== CALLOUTS ===== */
      .callout {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 12px 14px;
        border-radius: 10px;
        margin: 10px 0;
        color: #1e3a8a;
      }

      .warn {
        border-left: 5px solid #f59e0b;
        background: #fffbeb;
        color: #92400e;
      }

      .danger {
        border-left: 5px solid #ef4444;
        background: #fef2f2;
        color: #991b1b;
      }

      /* ===== MONO TEXT (still Georgia as requested) ===== */
      .mono {
        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 0.95rem;
        color: #1e3a8a;
      }

      hr {
        margin: 1rem 0;
        border: 1px solid #dbeafe;
      }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧭 Help & User Guide</div>', unsafe_allow_html=True)
st.markdown(
    "<p class='subtle'>Everything you need to use the valuation app smoothly — inputs, formulas, exports, and troubleshooting.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# QUICK NAV / OVERVIEW
# ---------------------------------------------------------
colA, colB, colC = st.columns([1.2, 1, 1])

with colA:
    st.markdown(
        """
        <div class="card">
          <h3>✅ Quick Start</h3>
          <div class="callout">
            1) Open a valuation module (DCF / DDM / etc.)<br>
            2) Fill inputs step-by-step<br>
            3) Check results & sensitivity tables<br>
            4) Export Excel models for documentation
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with colB:
    st.markdown(
        """
        <div class="card">
          <h3>📌 What gets saved?</h3>
          <p class="subtle">
            The app uses <span class="mono">st.session_state</span> to keep your inputs
            across pages in the same session.
          </p>
          <div class="callout warn">
            Refreshing the browser may clear some values depending on your setup.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with colC:
    st.markdown(
        """
        <div class="card">
          <h3>🧾 Exports</h3>
          <p class="subtle">
            Each module can generate an Excel workbook with formulas, formatted sheets,
            and a summary page.
          </p>
          <div class="callout">
            Best practice: Export after finalizing your assumptions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------
# TABS PER MODULE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📉 DCF", "💰 DDM", "📈 COMPARABLES","🏦 BANKING","🧾 SUMMARY", "🛠 Troubleshooting","⚡ Quick Summary"])

# -------------------------
# DCF TAB
# -------------------------
with tab1:
    with st.expander("1) What the DCF page does (big picture)", expanded=True):
        st.markdown("""
    **This DCF page produces an Equity Value using an Unlevered Free Cash Flow (UFCF/FCFF) model.**

    It takes:
    - **Income Statement + Balance Sheet + Cash Flow** (historical),
    - converts currency to USD if needed,
    - lets you **map the correct lines** (Debt, Cash, Revenue, EBITDA, etc.),
    - forecasts the Income Statement and Working Capital,
    - computes **WACC** using CAPM + cost of debt,
    - discounts UFCF + Terminal Value,
    - outputs **Enterprise Value → Equity Value**,
    - and builds a **WACC vs g sensitivity grid** + a downloadable Excel model.

    **Final formula logic**
    - **EV = PV(UFCF forecast years) + PV(Terminal Value)**
    - **Equity Value = EV − Net Debt**
    """)

    with st.expander("2) Step 0 — Start New Valuation (Reset button)", expanded=False):
        st.markdown("""
    At the top of the DCF page you will see:

    ✅ **🗂️ Clear & Upload New File**

    Use this when you want to start a fresh valuation.

    It clears:
    - uploaded statements & FX files
    - all mappings (Revenue, Debt, Cash, etc.)
    - forecasts, WACC inputs, sensitivity settings, and outputs

    **Tip:** If you uploaded the wrong Excel or mapped wrong rows, click reset first.
    """)

    with st.expander("3) Step 1 — Upload Financial Statements (Excel structure required)", expanded=True):
        st.markdown("""
    ### ✅ Required file format
    Upload **ONE Excel file** with **3 sheets in this exact order**:

    1. **Sheet 1:** Income Statement (IS)  
    2. **Sheet 2:** Balance Sheet (BS)  
    3. **Sheet 3:** Cash Flow (CF)

    ### ✅ Required layout inside each sheet
    - The **first column must be line items** (e.g., Revenue, EBITDA, Total Assets)
    - All other columns must be **years** (e.g., 2022, 2023, 2024)
    - Numbers can include commas and brackets (e.g. `(1,250)`), the system cleans them.

    ### Common upload mistakes
    - If your years are typed like `FY2024` instead of `2024`, results may fail.
    - If the first column is not line items, mapping will be confusing.
    """)

    with st.expander("4) Step 2 — Currency & FX Conversion (USD vs ZWG)", expanded=True):
        st.markdown("""
    ### 4.1 Choose the currency of uploaded statements
    You must choose one:
    - ✅ **USD (already converted)** → no FX conversion is done
    - ⚠️ **ZWG (convert using FX Excel)** → you must upload FX data

    ### 4.2 If ZWG: Upload FX Excel
    Your FX file must have:
    - A **Date** column
    - At least one of these rate columns:
      - **Interbank**
      - **Alternative**
      - **Premium**

    You will select which FX column to use.

    ### 4.3 How FX conversion is applied
    The model converts ZWG → USD like this:

    - **Income Statement & Cash Flow**: uses **Yearly Average FX**  
    - **Balance Sheet**: uses **Closing FX per year** based on the date you choose

    That is correct finance logic because:
    - IS/CF flows happen throughout the year → average rate is reasonable  
    - BS is a point-in-time snapshot → use closing rate

    ### 4.4 Balance Sheet closing dates (per year)
    You will enter the closing date for each Balance Sheet year (default is 31 Dec).
    The system will pick the last FX rate available **on or before** that date.

    ### 4.5 Optional: ZWG → ZiG factor adjustment
    If your data spans a period where ZWG was replaced or a factor is needed:
    - enable the factor,
    - select the year(s),
    - select date ranges,
    - the system divides the FX values by your factor inside those ranges.

    **Use this only if you truly have mixed periods.**
    """)

    with st.expander("5) Step 3 — Mapping (MOST IMPORTANT STEP)", expanded=True):
        st.markdown("""
    Mapping means: **you tell the model which statement lines represent the variables it needs.**  
    If mapping is wrong, the valuation will be wrong.

    You will map in 3 places:

    ✅ **A) Balance Sheet Mapping**  
    ✅ **B) Cash Flow Mapping**  
    ✅ **C) Income Statement Core Totals Mapping**
    """)

        st.markdown("### 5A) Balance Sheet Mapping — what to select")
        st.markdown("""
    You will multi-select rows (you can select more than one if statements are split).

    **1) Total Debt / Borrowings**  
    Select all interest-bearing debt rows, e.g.:
    - Loans and borrowings  
    - Bank loans  
    - Notes / bonds  
    - Lease liabilities (if you want)  
    Avoid: trade payables (those are working capital).

    **2) Cash & Cash Equivalents**  
    Select:
    - Cash, bank balances, cash equivalents, short-term deposits.

    **3) Current Assets (CA) for Working Capital**  
    Select operating current assets like:
    - Inventory  
    - Trade receivables / Debtors  
    - Prepayments (optional)

    Avoid: cash if you already mapped cash separately (unless your company treats cash as operating).

    **4) Current Liabilities (CL) for Working Capital**  
    Select operating current liabilities like:
    - Trade payables / Creditors  
    - Accrued expenses  
    - Other payables

    Avoid: interest-bearing short-term debt if you already include it in “Debt”.

    **5) Equity (Book Equity)**
    This is the equity used for **D/E (Debt-to-Equity)** in your WACC calculation.
    Select:
    - Total equity  
    - Shareholders’ equity  
    - Equity attributable to owners  
    If there are multiple equity lines, you can multi-select them.

    ✅ **Important:** This is **book equity from the balance sheet**, not market cap.
    """)

        st.markdown("### 5B) Cash Flow Mapping — what to select")
        st.markdown("""
    **1) Depreciation & Amortisation (CF)**  
    Select depreciation line in Cash Flow if it exists.

    **2) Capex (CF)**  
    Select capex type lines like:
    - Purchase of property, plant and equipment (PPE)
    - Additions to PPE
    - Purchase of intangibles

    Capex is often negative (cash outflow). The model keeps sign.

    **3) Interest paid (optional)**
    Only select if your Income Statement doesn’t clearly contain interest expense.
    This helps compute **cost of debt**.
    """)

        st.markdown("### 5C) Income Statement Core Totals Mapping — what to select")
        st.markdown("""
    This is a step-by-step wizard. Only **Revenue is mandatory**, but better mapping improves accuracy.

    **Revenue (MANDATORY)**  
    Select total revenue / sales.

    Optional but recommended:
    - Cost of Sales / Raw Materials
    - Gross Profit
    - EBITDA
    - Depreciation & Amortisation (IS line)  ✅ (your code supports this)
    - Operating Profit / EBIT
    - Profit Before Tax (PBT)
    - Income Tax (tax expense)
    - Profit for the Year (Net profit)

    ✅ The model checks that totals appear **top-to-bottom** in correct order.
    If you map totals out of order it will stop and show an error.
    """)

    with st.expander("6) Step 4 — Forecast horizon and revenue growth", expanded=True):
        st.markdown("""
    ### 6.1 Forecast horizon
    Choose how many years to forecast (1 to 15).

    ### 6.2 Revenue growth
    The system calculates a historical average revenue growth and shows it.
    You can override it.

    ### 6.3 Growth method choice
    You can choose:
    - **Uniform growth** (same % every forecast year)
    - **Different growth per year** (enter each year separately)

    Use “Different growth” if you want realistic fade down/up patterns.
    """)

    with st.expander("7) Step 5 — Forecast logic (what the model does automatically)", expanded=True):
        st.markdown("""
    After revenue is forecasted, the system forecasts other IS lines.

    ### 7.1 Gross Profit / COS handling
    There are 4 cases:
    - **GP and COS mapped:** forecasts COS using average GP margin  
    - **GP mapped, COS missing:** forecasts GP using average GP margin  
    - **COS mapped, GP missing:** forecasts COS as % of revenue  
    - **Neither mapped:** no special handling; other rows still forecast as % of revenue

    ### 7.2 “Other rows as % of revenue”
    For every non-total line item not protected:
    - It computes an average historical ratio (Row / Revenue)
    - Forecasts the row using that ratio × forecast revenue

    ### 7.3 Totals chain engine
    Totals like GP, EBITDA, EBIT, PBT, NP are re-calculated by summing the block between totals.
    This prevents double-counting and keeps totals consistent.

    ### 7.4 Tax forecasting
    Tax is derived using:
    - average historical **Tax / PBT** ratio (only when PBT is positive)
    Tax stays negative if your statement shows tax as negative.
    """)

    with st.expander("8) Step 6 — Working Capital (Historical → WC% → Forecast → ΔWC)", expanded=True):
        st.markdown("""
    This section only works if you mapped **Current Assets and Current Liabilities**.

    ### 8.1 Historical Working Capital
    - WC = CA − CL (by year)

    ### 8.2 WC % of Sales
    - WC% = WC / Revenue

    You can **exclude outlier years** using the “Include” checkbox table.
    This is important if one year is abnormal.

    ### 8.3 Choose assumption method
    You choose:
    - **Average WC%** (across included years)
    OR
    - **Most recent WC%** (latest included year)

    ### 8.4 Forecast WC
    - Forecast WC = Forecast Revenue × WC%

    ### 8.5 Change in WC (ΔWC)
    Your model defines:
    - **ΔWC = Old WC − New WC**

    Meaning:
    - If WC increases → ΔWC becomes negative (cash outflow)
    - If WC decreases → ΔWC becomes positive (cash inflow)
    """)

    with st.expander("9) Step 7 — Debt, Cash, Net Debt and D/E ratio", expanded=True):
        st.markdown("""
    These come from the **Balance Sheet mapping** at the last common year.

    - **Total Debt** = sum of selected debt rows
    - **Cash** = sum of selected cash rows
    - **Net Debt** = Debt − Cash
    - **Equity** = sum of selected equity rows
    - **D/E ratio** = Debt / Equity

    ✅ D/E is used to compute weights in WACC:
    - wd = D/E / (1 + D/E)
    - we = 1 / (1 + D/E)

    ⚠️ If your equity mapping is wrong, your D/E and WACC will be wrong.
    """)

    with st.expander("10) Step 8 — DCF Parameters (RF, MRP, Beta, Tax, Rd, g)", expanded=True):
        st.markdown("""
    ### 10.1 Risk-free rate (RF) + Market risk premium (MRP)
    You can run in two ways:
    - **Auto (from Excel)** using Country ERP + Default Spread file
    - **Manual override** (type in values)

    If auto is ON and the Excel has values, the model will “snap” RF & MRP to the auto results.

    ### 10.2 Unlevered beta (βu)
    You can:
    - Select one or more industries from the Industry Betas file
    - Choose:
      - Simple average
      - Weighted average
    - Or manually override βu

    ### 10.3 Cost of debt (Rd)
    Auto mode:
    - Rd = |Interest| / |Total Debt|  (from statements)

    Manual mode:
    - You type Rd as a % and it is used directly

    ### 10.4 Levering beta and WACC
    The model computes:
    - **βL = βu × (1 + (1 − Tax) × D/E)**
    - **Re = RF + βL × MRP**
    - **WACC = we×Re + wd×Rd×(1 − Tax)**

    ### 10.5 Terminal growth (g)
    This is the long-run growth rate used in terminal value.
    **Important rule:** the model needs **WACC > g** for terminal value to work.
    If WACC ≤ g, terminal value becomes invalid and sensitivity cells may be blank.
    """)

    with st.expander("11) Step 9 — Valuation timing (date-based discounting + mid-year)", expanded=True):
        st.markdown("""
    This model uses **date-based discounting**, not just “year 1, year 2”.

    You will provide:
    - **Valuation date** (today/deal date)
    - **Financial statement year-end date for first forecast year**
    - optional **mid-year convention** (subtracts 0.5 years)

    The model computes the first discount period **n₀** from dates, then:
    - discount periods = n₀, n₀+1, n₀+2, ...

    Discount factor = 1 / (1 + WACC)ⁿ
    """)

    with st.expander("12) Step 10 — CAPEX averaging (outlier exclusions)", expanded=True):
        st.markdown("""
    CAPEX is taken from the Cash Flow mapping.

    The model:
    - builds CAPEX history (sum of selected capex rows)
    - lets you exclude outlier years (persistently)
    - computes average CAPEX from remaining years
    - forecasts CAPEX as constant average for all forecast years

    ✅ If CAPEX is negative historically, it stays negative (cash outflow).
    """)

    with st.expander("13) Step 11 — UFCF / FCFF calculation (core DCF engine)", expanded=True):
        st.markdown("""
    Your model calculates UFCF using:

    **UFCF = EBITDA×(1 − T) + (−Depreciation×T) + ΔWC + CAPEX**

    Where:
    - EBITDA×(1−T) = after-tax operating earnings proxy
    - Depreciation×Tax adds back the tax shield (your implementation uses `-dep * tax`)
    - ΔWC is old minus new working capital
    - CAPEX is usually negative

    Then it discounts each UFCF by date-based discount factors to get PV of UFCF.
    """)

    with st.expander("14) Step 12 — Terminal value, Enterprise value, Equity value", expanded=True):
        st.markdown("""
    Terminal value uses Gordon Growth:

    **TV = UFCF_last × (1 + g) / (WACC − g)**

    PV of TV = TV × DiscountFactor_last

    **Enterprise Value (EV)**  
    = Sum(PV of UFCF) + PV(TV)

    **Equity Value**  
    = EV − Net Debt

    ✅ Equity Value is saved to session state for use in Comparables and Summary pages.
    """)

    with st.expander("15) Step 13 — Sensitivity table (WACC vs g)", expanded=True):
        st.markdown("""
    This grid shows how Equity Value changes when:
    - WACC changes (rows)
    - Terminal growth g changes (columns)

    You control:
    - number of WACC points
    - number of g points
    - WACC step %
    - g step %

    ⚠️ Blank cells occur when WACC ≤ g (terminal value invalid).

    The base case cell is highlighted (current WACC & g).
    Min and max are also shown.
    """)

    with st.expander("16) Step 14 — Download FULL Excel model (formulas + sensitivity)", expanded=True):
        st.markdown("""
    This button builds a **full Excel model** that includes:
    - Forecast Income Statement (with formulas)
    - Working Capital sheet
    - Inputs sheet (RF, MRP, beta, Rd, etc.)
    - DCF valuation sheet (UFCF and PV)
    - Sensitivity sheet (formula-driven grid)
    - Summary sheet

    ✅ You can edit assumptions in Excel and outputs update automatically.
    """)

    with st.expander("17) Troubleshooting (common errors + how to fix)", expanded=True):
        st.markdown("""
    ### “Revenue must be selected”
    You did not map the Revenue line in Income Statement mapping.

    ### “Mapping order problem”
    Your totals are mapped out of order (e.g., EBITDA appears above GP in your mapping).
    Fix by selecting the true statement line positions.

    ### “Missing FX data for statement years”
    Your FX file does not cover some statement years.
    Add FX rates for those missing years.

    ### “No FX rate found on or before closing date”
    Your chosen BS closing date has no FX rate before it in the FX file.
    Choose a later date or extend FX data.

    ### “WACC labels collided (duplicate labels)”
    Your sensitivity step is too small or decimals too low → labels become same.
    Increase decimals or increase step.

    ### “Terminal value invalid / blank sensitivity cells”
    This happens when **WACC ≤ g**.
    Reduce g or increase WACC range.
    """)
# -------------------------
# DDM TAB  ✅ REPLACE THIS WHOLE SECTION
# -------------------------
with tab2:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">DDM</span> Dividend Discount Model (Gordon Growth)</h3>
          <p class="subtle">
            This module values equity by converting a growing stream of dividends into a single present value today.
            It’s best for stable, dividend-paying firms with predictable payout and long-run growth.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("1) What the DDM page does (big picture)", expanded=True):
        st.markdown(r"""
**This DDM page produces an Equity Value per Share using the Gordon Growth Dividend Discount Model.**

It takes:
- **Dividend history** (you enter dividends per year),
- chooses a **growth range** (years used to compute dividend CAGR),
- calculates **growth rate (g)** and **next dividend (D₁)**,
- computes **Cost of Equity (Rₑ)** using CAPM (and **levered beta** using D/E and tax),
- outputs **Value per share (P₀)** and optionally **Total Equity Value** using shares outstanding,
- generates an **auditable Excel model** with steps + formulas.

### Core Formula
""")
        st.latex(r"P_0 = \frac{D_1}{R_e - g}")
        st.markdown(r"""
Where:
- **D₁** = Dividend expected next year  
- **Rₑ** = Cost of Equity (usually CAPM)  
- **g** = Long-term dividend growth rate  

✅ **Key rule:** The model only works when **Rₑ > g**.
""")

    with st.expander("2) Step 1 — Dividend History (what to enter and why)", expanded=True):
        st.markdown(r"""
### What you do on the page
1) Choose **Start Year** and **End Year**  
2) Enter the **Dividend for each year** (the app stores it so it doesn’t reset within the session)  
3) The page shows a table of Year vs Dividend

### What the model needs from this step
- A **clean dividend series** to estimate growth.
- The dividend in the **final selected year** will later become the base for **D₁**.

### Tips (so your valuation makes sense)
- Use **dividend per share (DPS)** if you want **P₀ per share**.
- Keep the dividends consistent (same units for all years).
- If dividends are irregular, consider choosing a smaller growth range (Step 2) focusing on stable years.
""")

    with st.expander("3) Step 2 — Growth Calculation Range (select stable years)", expanded=True):
        st.markdown(r"""
### What you do
You select:
- **Growth start year**
- **Growth end year**

These years tell the model which part of history to use for growth.

### Why this matters
Dividend growth can be distorted by:
- special dividends,
- payout policy changes,
- one-off shocks.

So you should pick a range that reflects **“normal” long-term dividend behavior**.
""")

    with st.expander("4) Step 3 — Dividend Growth (g) and next dividend (D₁)", expanded=True):
        st.markdown(r"""
### How growth (g) is calculated
- If **start year = end year**, growth is **0%**.
- If dividends are positive, the model uses **CAGR**:

""")
        st.latex(r"g = \left(\frac{D_{end}}{D_{start}}\right)^{\frac{1}{(end-start)}} - 1")
        st.markdown(r"""
- If the starting dividend is 0 (or unusable), the model uses a **fallback** (e.g., 2%) to avoid breaking.

### How next dividend (D₁) is calculated
""")
        st.latex(r"D_1 = D_{end}\times(1+g)")
        st.markdown(r"""
✅ Interpretation:
- **g** is your long-run dividend growth assumption implied by the selected history.
- **D₁** is the dividend the model expects **next year**.

**Best practice:** For mature firms, g should usually be conservative and close to long-run economic growth.
""")

    with st.expander("5) Step 4 — Cost of Equity (Rₑ) via CAPM (and why DCF values appear here)", expanded=True):
        st.markdown(r"""
### Where inputs come from
This DDM page tries to **reuse your DCF assumptions** if they exist in `st.session_state`, such as:
- **Risk-free rate (RF)**
- **Equity risk premium / Market risk premium (MRP)**
- **Tax rate**
- **Unlevered beta (βu)**
- **Debt/Equity (D/E)**

You can also tick **“Manually override parameters”** to enter custom values.

### Levered beta
The model converts **unlevered beta** to **levered beta** using D/E and tax:

""")
        st.latex(r"\beta_L = \beta_u \times \left(1 + (1 - Tax)\times\frac{D}{E}\right)")
        st.markdown(r"""
### CAPM Cost of Equity
""")
        st.latex(r"R_e = RF + \beta_L \times MRP")
        st.markdown(r"""
✅ Interpretation:
- Higher **β** or **MRP** increases **Rₑ** → lowers valuation.
- Higher **D/E** increases βL (financial risk) → increases **Rₑ**.

**Tip:** If your D/E is extreme or equity is near zero, βL may become huge — your valuation will become unrealistic.
""")

    with st.expander("6) Step 5 — Equity Value per Share (P₀) and the critical validity check", expanded=True):
        st.markdown(r"""
### What the model does
- It checks the Gordon Growth rule: **Rₑ must be greater than g**.
- If **Rₑ ≤ g**, the model stops the valuation and shows an error (because the denominator becomes zero or negative).

### If the rule passes, the model computes:
""")
        st.latex(r"P_0 = \frac{D_1}{R_e - g}")
        st.markdown(r"""
✅ Interpretation:
- Higher **D₁** increases P₀.
- Higher **Rₑ** decreases P₀.
- Higher **g** increases P₀ (but too high g can break the model).

<div class="callout warn">
  <b>Common DDM issue:</b> If <span class="mono">Rₑ ≤ g</span>, Gordon Growth breaks.<br>
  Fix by using a more conservative <b>g</b> or revisiting your <b>Rₑ</b> assumptions (beta, MRP, D/E, tax).
</div>
""", unsafe_allow_html=True)

    with st.expander("7) Step 6 — Total Equity Value (P₀ × shares outstanding)", expanded=True):
        st.markdown(r"""
### What you do
Enter **Number of Shares**.

### What the model computes
If shares > 0 and P₀ is valid:
""")
        st.latex(r"\text{Total Equity Value} = P_0 \times \text{Shares Outstanding}")
        st.markdown(r"""
✅ Interpretation:
- If you already have **per-share dividends (DPS)**, then P₀ is **per share**, so multiplying by shares gives total equity value.

**Tip:** Make sure the “dividends” you entered are truly per share (or else your total value will be off by a scale factor).
""")

    with st.expander("8) Excel Export (FULL DDM model: sheets + what to check)", expanded=True):
        st.markdown(r"""
When you click **Generate / Update FULL DDM Excel Model**, the app creates an auditable workbook with:

- **DividendHistory**: Year & Dividend table  
- **Growth**: growth range + formulas for D_start, D_end, g, and D₁  
- **Parameters**: CAPM inputs + βL + Rₑ formulas  
- **Valuation**: Gordon Growth valuation + shares + total equity value  
- **Summary**: key outputs in one view

### Why this export matters
- It preserves formulas (INDEX/MATCH, CAPM, Gordon Growth).
- It is perfect for reporting, audit trail, and sharing assumptions.
""")

    with st.expander("9) Troubleshooting (DDM-specific)", expanded=True):
        st.markdown(r"""
### “Start year cannot be greater than end year”
- Your dividend history year range is invalid. Set Start Year ≤ End Year.

### “Growth start year must be earlier or equal to end year”
- Your growth range selection is reversed. Choose a valid range.

### “Rₑ must be greater than g”
- Gordon Growth is invalid when Rₑ ≤ g.
- Fix by:
  - lowering **g** (use a conservative perpetual growth),
  - checking **beta**, **MRP**, **D/E**, **tax**, **RF**.

### Excel download button disabled
- The download button is disabled until you click **Generate / Update**.
- Click generate first. If it still fails, check terminal logs for openpyxl errors.
""")

# -------------------------
# COMPARABLES TAB (TAB3) ✅ USER MANUAL SECTION (PASTE INTO help.py)
# -------------------------
with tab3:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">COMPS</span> Comparable Companies Valuation</h3>
          <p class="subtle">
            This module values a target company using market trading multiples from comparable (“peer”) companies.
            It supports <b>EV/EBITDA</b>, <b>P/B</b>, and <b>P/E</b> methods, with optional peer-universe auto-fill and
            an Excel export that contains the full audit trail (inputs + formulas).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("1) What this Comparables page does (big picture)", expanded=True):
        st.markdown(
            r"""
**Comparable Company Analysis (CCA)** estimates the target’s equity value by applying **peer trading multiples**
to the target’s own financial base (EBITDA, earnings, book equity).

This page does 6 main things:

1) **Auto-fills peers** from a Peer Universe Excel (optional but recommended)  
2) Lets you **review/edit comparables** and choose which multiples to include  
3) Computes **Average multiples** and applies a **Discount factor** to get “Implied” multiples  
4) Builds **Maintainable EBITDA** (from DCF) with weights (and optional timing effect)  
5) Builds **Maintainable Earnings** (from DCF) with weights (and optional timing effect)  
6) Computes **Equity values** using:
   - **EV/EBITDA** → Equity = (Implied EV/EBITDA × Maintainable EBITDA) − Net Debt  
   - **P/B** → Equity = (Implied P/B × Book Equity)  
   - **P/E** → Equity = (Implied P/E × Maintainable Earnings)  

✅ Outputs are displayed on-screen and can be exported to Excel with formulas.
"""
        )

    with st.expander("2) Step 0 — Auto-Fill Comparables (Peer Universe)", expanded=True):
        st.markdown(
            r"""
### Purpose
Step 0 helps you build a clean peer set quickly from a **Peer Universe file** (peer_universe.xlsx).

### What you do
1) Turn ON: **“Use Peer Universe Excel to auto-fill comparables”**  
2) Choose a **Target Company** (the firm you are valuing)  
3) Set **Max peers** (how many peers to suggest)  
4) Decide if you want auto-fill to happen instantly:
   - ✅ **Auto-fill comparables instantly when I choose a target**
5) Review the auto-selected peers in the **Peer companies** multiselect (you can edit it)

### How peers are suggested (important logic)
- If the target has a **PeerGroup**, the system uses **ONLY that PeerGroup** (strict matching).
  This prevents cross-industry mistakes.
- If PeerGroup is empty, it falls back to:
  **Industry → Sector** (and picks peers with more available multiples first).

### Uploading your own peer universe
- You can upload another Excel file using the uploader.
- The app stores it in memory for the session, so it stays even if you switch tabs.

### “Clear Comparables”
This button resets:
- company names,
- multiples,
- include/exclude flags,
so you can start fresh.
"""
        )

    with st.expander("3) Step 1 — Input Comparable Companies & Multiples", expanded=True):
        st.markdown(
            r"""
### Purpose
This is where you confirm the peer set and enter (or review) their trading multiples.

### What you do
1) Set **How many comparables?**  
2) For each comparable, fill:
   - Company name
   - **EV/EBITDA**
   - **P/B**
   - **P/E**
3) Use the **Analyst filter** checkboxes to control which multiples are included in the averaging:
   - Include EV
   - Include P/B
   - Include P/E

### How the “Include” filters work
- If you uncheck “Include EV” for a company, that company’s EV/EBITDA is excluded from the EV averaging.
- The same idea applies for P/B and P/E.
- This lets you remove outliers or irrelevant peers *without deleting them*.

### Output you get
At the bottom you see a table of:
- Company
- Multiples
- Include flags
"""
        )

    with st.expander("4) Step 2 — Average & Implied Multiples (with discount)", expanded=True):
        st.markdown(
            r"""
### Purpose
Convert peer multiples into:
- a **simple average**, and then
- an **implied multiple** after applying a discount.

### What you do
1) Enter **Discount factor (%)**  
   Example: 25% means you reduce the peer multiple by 25%.

### What the model computes
For each multiple:

**Average multiple**  
- Uses only peers with the Include flag = True
- Uses the mean of the included values

**Implied multiple**
"""
        )
        st.latex(r"\text{Implied Multiple} = \text{Average Multiple} \times (1-\text{Discount})")
        st.markdown(
            r"""
✅ Interpretation:
- Higher discount → lower implied multiple → lower equity value.
- Discount is typically used to reflect:
  - size/illiquidity discount,
  - country risk,
  - control vs minority differences,
  - quality differences vs peers.

You will see a summary table with Average, Discount, and Implied for EV/EBITDA, P/B, and P/E.
"""
        )

    with st.expander("5) Timing Source (from DCF) — why it exists", expanded=True):
        st.markdown(
            r"""
### Purpose
This section pulls **DCF discount timing values (n)** to create a timing base used in:
- Maintainable EBITDA (Step 3)
- Maintainable Earnings (Step 4)

### What you do
- If DCF timing exists:
  - you can choose **Use DCF n₀** (recommended), or
  - manually override the starting timing value.
- If DCF timing does NOT exist:
  - you must enter a **manual timing base**.

### What “timing effect” means here
When timing is ON, each year gets a factor:
- Year 1 uses base_timing
- Year 2 uses base_timing + 1
- Year 3 uses base_timing + 2
…and so on.

This factor is applied before weighting to compute maintainable values.
"""
        )

    with st.expander("6) Step 3 — Maintainable EBITDA (EV/EBITDA base)", expanded=True):
        st.markdown(
            r"""
### Purpose
Build a single “Maintainable EBITDA” value from DCF forecast EBITDA.

### Where EBITDA comes from
This module reads EBITDA from your DCF page (session_state keys like):
- dcf_ebitda_forecast / dcf_ebitda_all

If DCF EBITDA is missing:
- The EV/EBITDA method is skipped.

### What you do
1) Choose whether to apply timing:
   - **Apply timing effect from DCF to EBITDA?**
2) Select the EBITDA year range:
   - EBITDA Start Year
   - EBITDA End Year
3) Provide a **weight (%)** for each year in the selected range.

### What the model computes
For each year:
- **Timing factor** = 1 (if timing OFF) OR base_timing + index (if timing ON)
- **Adjusted EBITDA** = EBITDA × Timing
- **Weighted EBITDA** = Adjusted EBITDA × Weight

And then:
"""
        )
        st.latex(r"\text{Maintainable EBITDA} = \sum(\text{Weighted EBITDA})")
        st.markdown(
            r"""
✅ Tip (very important):
- Your weights are percentages. Make sure they make sense (many analysts aim for ~100% total,
  but the tool will still compute even if totals are above/below 100).

### Output you get
- A table showing EBITDA, timing, weights, adjusted EBITDA, and weighted EBITDA
- A final “Maintainable EBITDA” total
"""
        )

    with st.expander("7) Step 4 — Maintainable Earnings (P/E base)", expanded=True):
        st.markdown(
            r"""
### Purpose
Build a single “Maintainable Earnings” value from DCF forecast earnings.

### Where Earnings comes from
This module reads earnings from your DCF page (session_state keys like):
- dcf_profit_forecast / dcf_profit_all

If DCF Earnings is missing:
- The P/E method is skipped.

### Auto-sync features (important)
This page can keep Earnings weighting consistent with EBITDA weighting:

1) **Auto-use the SAME years & weights as EBITDA (recommended)**  
   - If ON: Earnings uses the same year range and weights as Step 3.  
   - This ensures method consistency.

2) **Timing is locked to EBITDA timing**  
   - If timing is OFF for EBITDA, timing is forced OFF for Earnings.

### What you do
1) Choose whether to sync years & weights to EBITDA  
2) Choose whether to apply timing (if allowed)  
3) If not syncing, you can manually choose the year range and weights.

### What the model computes
For each year:
- **Adjusted Earnings** = Earnings × Timing (or ×1 if timing off)
- **Weighted Earnings** = Adjusted Earnings × Weight

And then:
"""
        )
        st.latex(r"\text{Maintainable Earnings} = \sum(\text{Weighted Earnings})")
        st.markdown(
            r"""
### Output you get
- A table showing Earnings, timing, weights, adjusted and weighted values
- A final “Maintainable Earnings” total
"""
        )

    with st.expander("8) Step 5 — Book Value & Net Debt (P/B and EV bridge)", expanded=True):
        st.markdown(
            r"""
### Purpose
Provide the balance sheet inputs needed for:
- P/B valuation (Book Equity)
- EV/EBITDA bridge (Net Debt)

### Book Equity auto-fill
If you used the Banking page, the tool can auto-fill Beginning Book Equity from:
- bank.outputs.book_equity_0

### What you do
1) Enter or confirm **Book Equity (USD)**  
2) Enter **Net Debt (USD)**  

✅ Interpretation:
- **Net Debt** is used to move from enterprise value to equity value in the EV/EBITDA method:
  Equity = Enterprise Value − Net Debt
"""
        )

    with st.expander("9) Step 6 — Computed Equity Values (final outputs)", expanded=True):
        st.markdown(
            r"""
### Purpose
Compute equity value using each comparable multiple method.

### Methods and formulas

**A) EV/EBITDA**
"""
        )
        st.latex(r"\text{Equity Value}_{EV/EBITDA} = (\text{Implied EV/EBITDA} \times \text{Maintainable EBITDA}) - \text{Net Debt}")
        st.markdown(
            r"""
**B) P/B**
"""
        )
        st.latex(r"\text{Equity Value}_{P/B} = (\text{Implied P/B} \times \text{Book Equity})")
        st.markdown(
            r"""
**C) P/E**
"""
        )
        st.latex(r"\text{Equity Value}_{P/E} = (\text{Implied P/E} \times \text{Maintainable Earnings})")
        st.markdown(
            r"""
### Output you get
A results table with:
- EV/EBITDA equity value
- P/B equity value
- P/E equity value
"""
        )

    with st.expander("10) Excel Export — Comparables workbook (audit trail + formulas)", expanded=True):
        st.markdown(
            r"""
### Purpose
Download an Excel file that reproduces the model with formulas.

### What the export includes
The exported workbook contains these sheets:

1) **Comps_Input**
   - peer names, multiples, include flags

2) **Multiples**
   - AVERAGEIF formulas using include flags
   - Discount input and implied multiples

3) **EBITDA_Maintainable**
   - timing toggle, base timing, EBITDA, weights, adjusted and weighted EBITDA
   - maintainable EBITDA total

4) **Earnings_Maintainable**
   - timing toggle, base timing, earnings, weights, adjusted and weighted earnings
   - maintainable earnings total

5) **Equity_Values**
   - book equity, net debt
   - links to implied multiples and maintainables
   - final equity values for EV/EBITDA, P/B, P/E

✅ Why this matters:
- It creates a clear audit trail for reporting and review.
- You can share it with stakeholders who prefer Excel.
"""
        )

    with st.expander("11) Troubleshooting (common issues)", expanded=True):
        st.markdown(
            r"""
### “Missing peer_universe.xlsx”
- Ensure **peer_universe.xlsx** is inside your project **/data/** folder,
  OR upload it using the uploader.

### “No timing values detected from DCF”
- Run the DCF model first (so timing exists), or manually set a timing base.

### “No EBITDA found from DCF — skipping EV/EBITDA method”
- Your DCF page has not populated EBITDA into session_state.
- Run DCF first or confirm the EBITDA keys are being stored correctly.

### “No Earnings found from DCF — skipping P/E method”
- Same idea: run DCF first or confirm earnings are stored correctly.

### Weird results from averages
- Check the **Include** flags.
- Check for outliers (e.g., one peer with a huge multiple).
- Ensure you didn’t unintentionally leave a peer with 0.00 that you meant to exclude.

### Excel values don’t match the screen
- Regenerate the Excel export after changing inputs (because Excel is created from current session_state).
"""
        )
# -------------------------
# BANKING TAB (TAB4) ✅ USER MANUAL SECTION (PASTE INTO help.py)
# -------------------------
with tab4:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">BANKING</span> Banking Valuation — Residual Income Method</h3>
          <p class="subtle">
            This module values a bank (or financial institution) using the <b>Residual Income (RI)</b> approach
            with <b>actual year columns</b> from uploaded statements. It also supports optional <b>ZWG → USD</b>
            conversion using an FX Excel file (DCF-style).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("1) What this Banking page does (big picture)", expanded=True):
        st.markdown(
            r"""
### Residual Income Model (Banks)
For banks, traditional FCFF DCF can be tricky because debt is part of operating structure.  
Residual Income (RI) focuses on:
- **Book Value of Equity**, and
- **Earnings in excess of the equity charge (Ke × Book Value)**

**Core idea:**
- If a bank earns exactly Ke on book value → residual income = 0 → value ≈ book value
- If it earns more than Ke → positive residual income → value > book value

This page does 7 main things:
1) Upload your statements (IS + BS + SoCE)
2) (Optional) Convert ZWG → USD using FX Excel (yearly averages + BS closing rates)
3) Map the correct equity rows (Balance Sheet) and SoCE totals (for reference)
4) Select the best earnings line from the Income Statement
5) Choose a base year and pull base equity + base earnings
6) Compute Ke using a CAPM block (auto parameters + beta tools, DCF-style)
7) Forecast book value and earnings, compute residual income PVs, add terminal value, and output total equity value
"""
        )

    with st.expander("2) Step 0 — Upload Statements (IS + BS + SoCE)", expanded=True):
        st.markdown(
            r"""
### Purpose
Load your bank’s Excel statements and store them in session so values do not reset when you switch tabs.

### What you do
1) Upload an Excel file (.xlsx)
2) Select the correct sheets:
   - Income Statement sheet
   - Balance Sheet sheet
   - Statement of Changes in Equity (SoCE) sheet

### What the tool does automatically
- Cleans numeric columns (removes commas, handles brackets for negatives, strips spaces)
- Detects year columns safely (even if headers are messy or “Unnamed”)

✅ If year columns cannot be detected, the tool stops and asks you to fix the Excel headers.
"""
        )

    with st.expander("3) Step 1 — Currency & FX Conversion (DCF-style)", expanded=True):
        st.markdown(
            r"""
### Purpose
Handle currency conversion consistently when statements are in **ZWG**.

### What you do
1) Choose the currency of uploaded statements:
   - **USD (already converted)** → no FX conversion is applied
   - **ZWG (convert using FX Excel)** → you must upload FX Excel

### FX Excel requirements
Your FX file must contain:
- A **Date** column
- At least one of these FX columns:
  - **Interbank**
  - **Alternative**
  - **Premium**

You then choose which FX column to use.

### Conversion rules (very important)
This model uses different FX logic for different statements:

**A) Income Statement (IS) + SoCE**
- Converted using **Yearly Average FX** for each year column.

**B) Balance Sheet (BS)**
- Converted using **Closing FX rate per year**, where you choose the closing date for each year.

This mirrors standard financial practice (flows vs stocks).

### Optional: Manual ZWG → ZiG factor (mixed periods)
If your FX history includes mixed regimes, you can enable a manual factor:
- Select year(s)
- Define date ranges within the year
- Apply a factor that divides the FX rate inside those ranges

✅ The app refreshes FX conversion automatically whenever any FX setting changes.
"""
        )

    with st.expander("4) Step 2 — SoCE Mapping (Closing equity total)", expanded=True):
        st.markdown(
            r"""
### Purpose
Tell the model where “Total Equity” is inside the SoCE.

### What you do
1) Select the **Closing Balance** row (Normalised if available)
2) Select the **TOTAL Equity column**

### Output you get
A small table showing the mapped SoCE equity totals.

⚠ Note:
SoCE mapping is mainly for reference / validation.  
The model uses **Balance Sheet equity** as the base (for consistency).
"""
        )

    with st.expander("5) Step 3 — Balance Sheet Mapping (Equity rows)", expanded=True):
        st.markdown(
            r"""
### Purpose
Define which Balance Sheet row(s) represent **Total Equity**.

### What you do
- Multi-select all rows that represent Total Equity.
  Example: if equity is split into components and there is no single “Total Equity” line,
  you can select multiple equity lines and the tool sums them.

### Why this matters
Book value (equity) is the anchor of the Residual Income model, so mapping must be correct.

✅ If you select nothing, the tool stops.
"""
        )

    with st.expander("6) Step 4 — Earnings line selection (Income Statement)", expanded=True):
        st.markdown(
            r"""
### Purpose
Tell the model which Income Statement line represents the “earnings” used in residual income.

### Default logic
The tool tries to default to:
- “Normalised profit / Normalized profit”

If not found, it falls back to typical lines like:
- Profit for the year
- Profit after tax (PAT)
- Net profit

### What you do
Pick the best earnings line from the dropdown.

✅ The selected earnings becomes the base earnings and forecast starting point.
"""
        )

    with st.expander("7) Step 5 — Base Year selection (actual years)", expanded=True):
        st.markdown(
            r"""
### Purpose
Select the year that will act as your “Year 0” starting point.

### How the tool chooses base-year options
- It uses the intersection of year columns available in:
  - Income Statement
  - Balance Sheet

### What you do
Choose a base year from the dropdown.

### Output you get
The tool displays:
- Total Equity (base year)
- Earnings (base year)
- Earnings line name

✅ In this model, base-year equity is always taken from the Balance Sheet.
"""
        )

    with st.expander("8) Step 6 — Cost of Equity (Ke) via CAPM (DCF-style Auto + Override)", expanded=True):
        st.markdown(
            r"""
### Purpose
Compute **Ke** (required return on equity) using CAPM:

"""
        )
        st.latex(r"K_e = R_f + \beta \times MRP")
        st.markdown(
            r"""
### Inputs supported
**A) Country params (ERP + Default Spread)**
- Uses either:
  - default dcf_parameters.xlsx (if present), or
  - uploaded file (optional)
- Then computes:
  - **MRP = ERP**
  - **Rf = Avg Cost of Debt − Default Spread** (Zimbabwe USD assumption)

**B) Beta selection**
You can:
- Blend **unlevered betas (βu)** from selected industries (simple/weighted average), then lever it
OR
- Override directly with **manual levered beta (β)**

**Levering formula**
"""
        )
        st.latex(r"\beta_L = \beta_u \times (1 + (1 - tax)\times D/E)")
        st.markdown(
            r"""
### What you do
1) Choose whether to upload Country Params / Industry Betas (optional)
2) Select country (for ERP + spread)
3) Choose beta mode (βu then lever, or manual β)
4) Enter RF, MRP, tax and D/E (or accept auto values)

### Output you get
Rf, MRP, β, and final Ke displayed as metrics.
"""
        )

    with st.expander("9) Step 7 — Forecast assumptions (Book Value, Discounts, Earnings growth)", expanded=True):
        st.markdown(
            r"""
### Purpose
Define how book value and earnings evolve after the base year.

### Forecast years
You choose how many forecast years (1 to 15).  
Forecast years are: base_year+1 … base_year+n

### A) Book Value growth (YoY) + Discount
You can choose:
- **Uniform** (same rate every year)
- **Different per year**

There is also an option:
- ✅ “Auto-fill Book Value YoY (%) from BS actual YoY”
  - It computes the last actual YoY from the Balance Sheet (previous year → base year)
  - It then pre-fills the YoY input (you can still override)

### B) Earnings growth
Same structure:
- Uniform or Different per year
- Applied to the base-year earnings

### C) Terminal growth (g)
Used to compute the terminal value based on the last residual income.

✅ Tip:
Ke must be greater than terminal g, otherwise terminal value becomes invalid.
"""
        )

    with st.expander("10) Model engine — Residual Income table and formulas", expanded=True):
        st.markdown(
            r"""
### Core building blocks
For each year:

**1) Equity Charge**
"""
        )
        st.latex(r"\text{Equity Charge}_t = -K_e \times BV_t")
        st.markdown(
            r"""
**2) Residual Income**
"""
        )
        st.latex(r"RI_t = Earnings_t + \text{Equity Charge}_t")
        st.markdown(
            r"""
**3) Discount Factor**
The model supports two timing conventions:
- Base year t = 0 (standard)
- Base year t = 1 (shifted)

Standard discount factor:
"""
        )
        st.latex(r"DF_t = \frac{1}{(1+K_e)^t}")
        st.markdown(
            r"""
**4) Present Value of residual income**
"""
        )
        st.latex(r"PV(RI_t) = RI_t \times DF_t")
        st.markdown(
            r"""
### Terminal value
Terminal value is computed from the last forecast residual income:

"""
        )
        st.latex(r"TV = \frac{RI_{last}\times(1+g)}{K_e - g}")
        st.markdown(
            r"""
and present-valued using the last year discount factor.

### Final equity value
"""
        )
        st.latex(r"Equity\ Value = BV_0 + \sum PV(RI_t) + PV(TV)")
        st.markdown(
            r"""
### Output you get
1) A full “Residual Income Valuation Table (Totals)” showing:
- Book value, YoY, discounts, adjusted YoY
- Earnings and growth
- Equity charge, residual income
- Discount factors and PVs
- Terminal value and PV terminal

2) A summary with:
- Beginning Book Value
- Sum PV of residual income
- PV terminal
- Total equity value
"""
        )

    with st.expander("11) Outputs saved for other tabs (integration)", expanded=True):
        st.markdown(
            r"""
This module stores key outputs into session_state so other tabs can use them:

- bank.outputs.book_equity_0  → Beginning Book Value
- bank.outputs.earnings_0     → Base earnings
- bank.outputs.ke            → Cost of equity
- bank.outputs.equity_value_total → Final equity value
- equity_value_banking       → Final equity value (shortcut)

If you return to Comparables, P/B can auto-use Book Equity from here.
"""
        )

    with st.expander("12) Troubleshooting (common issues)", expanded=True):
        st.markdown(
            r"""
### “Could not detect year columns”
- Your Excel headers may be merged or labelled “Unnamed”.
- Ensure each year appears clearly in the column header (e.g., 2022, 2023, 2024).

### FX conversion stops / missing dates
- Make sure FX file has:
  - Date column, and selected FX column is numeric
- If a Balance Sheet closing date is before the first FX date → no “as of” rate exists.

### Equity mapping gives wrong totals
- Confirm you selected the correct BS equity lines.
- If equity is split across multiple lines, select all relevant lines.

### Terminal value becomes blank/invalid
- Check that Ke > terminal g.
- If not, lower g or revise Ke assumptions.
"""
        )
# -------------------------
# SUMMARY TAB (TAB5) ✅ USER MANUAL SECTION (PASTE INTO help.py)
# -------------------------
with tab5:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">SUMMARY</span> Summary Valuation — Weighted Equity Value</h3>
          <p class="subtle">
            This page combines outputs from <b>DCF</b>, <b>DDM</b>, <b>Comparables</b> (EV/EBITDA · PBV · P/E),
            and <b>Banking</b> into one <b>blended equity value</b>. You select models, assign weights, and the app
            calculates intrinsic value per share, upside/downside, and a simple recommendation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("1) What the Summary page does (big picture)", expanded=True):
        st.markdown(
            r"""
### Purpose
The Summary page is your “final dashboard”. It:
- Pulls **equity values** already calculated in other tabs
- Lets you **choose which models to include**
- Lets you **assign weights (%)** to each selected model
- Produces a **Weighted Equity Value (blended valuation)**
- Converts the blended equity into:
  - **Intrinsic value per share** (if shares are provided)
  - **Upside/Downside %** vs current market price (if price is provided)
  - A simple **Buy / Hold / Reduce** label

### Models supported
- **DCF**
- **DDM**
- **EV/EBITDA**
- **PBV**
- **P/E**
- **BANKING**
"""
        )

    with st.expander("2) Step 1 — Select Models to Include", expanded=True):
        st.markdown(
            r"""
### What you do
Use the multi-select box to choose which models are “active”.

### What the app does
- Stores your selection in session_state so it won’t reset when you switch tabs.
- If you select nothing, the page stops (because we need at least one model to compute a blend).

✅ Tip:
If a model value is missing (because you didn’t run that model tab yet), it may show as blank/None in the table.
So always run the model tabs first if you want them included.
"""
        )

    with st.expander("3) Step 2 — Assign Weights (%)", expanded=True):
        st.markdown(
            r"""
### What you do
For each model you selected, enter a weight (%).
- You can enter any numbers (they do not need to sum to 100).

### What the app does (important)
It automatically **normalizes** weights so the selected models sum to **100%**.

Example:
- DCF = 40, DDM = 20, PBV = 20  → total input = 80  
Normalized:
- DCF = 40/80 = 50%
- DDM = 20/80 = 25%
- PBV = 20/80 = 25%

✅ If total weight for selected models is 0, the page stops (division by zero).
"""
        )

    with st.expander("4) Where the Summary values come from (session_state mapping)", expanded=True):
        st.markdown(
            r"""
### The Summary page pulls results from other tabs using these keys:

- **DCF** → `equity_value_dcf`
- **DDM** → `equity_value_ddm`
- **EV/EBITDA** → `value_ev_ebitda`
- **PBV** → `value_pbv`
- **P/E** → `value_pe`
- **BANKING** → `equity_value_banking`

If a key is missing (because you haven’t run that model yet), the value may show as blank.
"""
        )

    with st.expander("5) How the blended (Weighted) Equity Value is calculated", expanded=True):
        st.markdown(
            r"""
### For each selected model
The app computes:

**Weighted Value = Model Value × (Normalized Weight / 100)**

### Final blended equity
The **Weighted Equity Value** is:

**Weighted Equity = SUM of all Weighted Values**

This is displayed as a KPI card at the top and shown again in the table.
"""
        )

    with st.expander("6) Summary Table & Interactive Dashboard", expanded=True):
        st.markdown(
            r"""
### Summary Table tab
Shows a table with:
- Model
- Value (USD)
- Weight (%) (normalized)
- Weighted Value (USD)

### Interactive Dashboard tab
Shows bar charts for:
- Model equity values
- Model weights
- Weighted contributions (how much each model contributes to the final blended equity)

✅ Use the dashboard to quickly spot:
- which model is driving the valuation most,
- which model is the outlier (highest/lowest),
- and how wide the valuation range is.
"""
        )

    with st.expander("7) Valuation Summary (Shares, Price, Intrinsic, Upside/Downside)", expanded=True):
        st.markdown(
            r"""
### What you do
Enter:
1) **Number of Shares in Issue**
2) **Current Share Price (USD)**

### What the app computes
**Intrinsic Value per Share**
- Intrinsic = Weighted Equity / Shares  
(only computed if shares > 0)

**Upside/Downside (%)**
- Upside% = (Intrinsic − Current Price) / Current Price × 100  
(only computed if current price > 0)

### Output you get
A small table containing:
- Weighted Equity Value
- Shares
- Intrinsic value per share
- Current share price
- Upside / Downside (%)
"""
        )

    with st.expander("8) Recommendation logic (Buy / Hold / Reduce)", expanded=True):
        st.markdown(
            r"""
### The recommendation is based on Upside/Downside (%)

If shares and price are entered correctly, the app labels:

- **🟢 BUY / ACCUMULATE** if upside is meaningfully positive (above the buy threshold)
- **🟡 HOLD / FAIRLY VALUED** if upside is near zero (within a fair value band)
- **🔴 REDUCE / SELL** if upside is negative beyond the band

✅ Note:
This recommendation is purely rule-based and depends heavily on:
- model weights,
- your assumptions in each valuation tab,
- Ke/WACC/growth choices,
- and data quality.
"""
        )

    with st.expander("9) Excel Download — Valuation Summary (with formulas)", expanded=True):
        st.markdown(
            r"""
### What you get in the Excel export
When you click download, the file includes:

**Sheet 1: Model_Summary**
- Model
- Value_USD
- Weight_Input_%
- Weight_Normalized_% (formula)
- Weighted_Value_USD (formula)
- Total blended equity (SUM formula)

**Sheet 2: Valuation_Summary**
- Weighted Equity Value (linked from Sheet 1)
- Shares (your input)
- Intrinsic value per share (formula)
- Current price (your input)
- Upside/Downside % (formula)
- Recommendation (formula)

✅ This makes the export easy to audit and share with stakeholders.
"""
        )

    with st.expander("10) Troubleshooting (common issues)", expanded=True):
        st.markdown(
            r"""
### “My model value is blank / —”
- You probably have not run that model tab yet.
- Go to the model tab (DCF / DDM / Comparables / Banking), complete it, then return.

### “Total weight cannot be zero”
- You selected models, but typed 0 for all their weights.
- Give at least one selected model a non-zero weight.

### Intrinsic value shows blank/NaN
- Shares must be > 0.

### Upside/Downside shows blank/NaN
- Current price must be > 0 (and shares must be valid so intrinsic exists).

### Excel export downloads but values look wrong
- Check that your selected models have real values.
- Check you did not accidentally set weights to 0 or exclude a key model.
"""
        )
# -------------------------
# TAB6 🛠 TROUBLESHOOTING (ALL MODELS) ✅ PASTE INTO help.py
# -------------------------
with tab6:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">HELP</span> 🛠 Troubleshooting — All Models</h3>
          <p class="subtle">
            Use this section when something looks wrong, missing, blank, or “not updating”.
            Most issues are caused by missing inputs, missing source files, wrong sheet selection,
            or session state resets after clearing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------
    # QUICK DIAGNOSTIC CHECKLIST
    # -------------------------
    with st.expander("0) Quick checklist (do this first)", expanded=True):
        st.markdown(
            r"""
### Quick checklist (90% of issues)
1) **Check you ran the model tab first**  
   - Summary pulls outputs from each model’s `session_state` keys.  
   - If you never ran DCF/DDM/Comparables/Banking, Summary will show blanks.

2) **Look for red/amber warnings on the page**
   - Those warnings tell you exactly which input or file is missing.

3) **Avoid clearing a tab accidentally**
   - “Clear” buttons wipe that tab’s session values.

4) **Check currency & FX settings**
   - Wrong FX conversion can make values look 10×, 100×, or 1000× too big.

5) **Check year detection**
   - If your Excel has merged headers (“Unnamed: …”), the app may fail to detect years.
"""
        )

    # -------------------------
    # DCF TROUBLESHOOTING
    # -------------------------
    with st.expander("2) DCF — common issues", expanded=False):
        st.markdown(
            r"""
### Problem: “DCF says missing data / UFCF is empty”
**Common causes**
- You didn’t load the financial inputs correctly.
- You didn’t complete required assumptions (WACC, growth, tax, margins, etc.).
- Forecast years are not aligned.

**Fix**
- Confirm the DCF input sheet has required line items and valid year columns.
- Ensure WACC and terminal growth assumptions are valid.
- Check any warnings about missing revenue/EBIT/FCF drivers.

### Problem: “Terminal value is NaN / error”
**Cause**
- Terminal growth `g` is ≥ WACC (or Ke), making denominator invalid.
**Fix**
- Reduce `g` or increase discount rate so that **WACC > g**.
"""
        )

    # -------------------------
    # DDM TROUBLESHOOTING
    # -------------------------
    with st.expander("3) DDM — common issues", expanded=False):
        st.markdown(
            r"""
### Problem: “DDM Equity is NaN / not computed”
**Common causes**
- Dividends are missing or 0.
- `Ke` is missing or <= terminal dividend growth.

**Fix**
- Ensure dividends (D1 / forecast dividends) exist and are numeric.
- Ensure **Ke > g** for terminal dividend growth.

### Problem: “DDM value is extremely small”
**Cause**
- Dividend inputs are tiny relative to shares/equity.
**Fix**
- Validate dividend units and currency (USD vs ZWG conversion).
"""
        )

    # -------------------------
    # COMPARABLES TROUBLESHOOTING
    # -------------------------
    with st.expander("4) Comparables — common issues (EV/EBITDA, PBV, P/E)", expanded=False):
        st.markdown(
            r"""
### Problem: “Suggested peers look wrong”
**Cause**
- PeerGroup / Sector / Industry mapping is wrong in `peer_universe.xlsx`.
**Fix**
- Check your target company row has the right PeerGroup.
- If PeerGroup is present, the app uses STRICT PeerGroup matching.

### Problem: “Averages look wrong”
**Cause**
- You included/excluded the wrong peers using the Analyst Filter (Include EV/PB/PE).
- You left 0s in multiples (0s are treated as real values, not blanks).

**Fix**
- Use the Include checkboxes carefully (turn off outliers or irrelevant comps).
- Replace “unknown” values with blanks (NaN) rather than 0.

### Problem: “Maintainable EBITDA/Earnings is missing”
**Cause**
- Comparables Step 3 & Step 4 depend on DCF outputs:
  - EBITDA: `dcf_ebitda_all` / `dcf_ebitda_forecast`
  - Earnings: `dcf_profit_all` / `dcf_profit_forecast`

**Fix**
- Run DCF first, then come back to Comparables.
- If DCF has no valid year keys (must be 4-digit years like 2024), fix the DCF output structure.

### Problem: “Excel export formulas don’t work”
**Cause**
- Excel formulas rely on ranges (start/end rows) and include flags.
**Fix**
- Ensure you have at least 1 comparable row in Comps_Input.
- Check Include flags are TRUE/FALSE in Excel (not text).
"""
        )

    # -------------------------
    # BANKING TROUBLESHOOTING
    # -------------------------
    with st.expander("5) Banking — common issues (Residual Income)", expanded=False):
        st.markdown(
            r"""
### Problem: “Year columns not detected”
**Cause**
- Your Excel has merged headers or “Unnamed: …” columns.
**Fix**
- Make sure year headers contain a real year (e.g., 2022, 2023) in the column label.
- Avoid merged year cells in the header row.

### Problem: “FX conversion stopped / missing Date column”
**Cause**
- FX Excel must have **Date** plus one of: **Interbank / Alternative / Premium**.
**Fix**
- Add a `Date` column (Excel date format) and at least one allowed FX column.

### Problem: “Equity / Earnings is wrong”
**Cause**
- Wrong sheet selection (IS vs BS vs SoCE).
- Wrong row mapping for Equity or Earnings.
**Fix**
- Re-check sheet selections at upload step.
- Re-select correct Equity rows (multi-select) and Earnings line.

### Problem: “Terminal value is NaN”
**Cause**
- `Ke <= g_term`.
**Fix**
- Ensure **Ke > terminal growth**.
"""
        )

    # -------------------------
    # SUMMARY TROUBLESHOOTING
    # -------------------------
    with st.expander("6) Summary — common issues (weighted equity)", expanded=False):
        st.markdown(
            r"""
### Problem: “A model shows blank/None”
**Cause**
- You did not run that model tab yet (so session_state key is missing).
**Fix**
- Run the model tab first, then return to Summary.

### Problem: “Total weight cannot be zero”
**Fix**
- Give at least one selected model a non-zero weight.

### Problem: “Intrinsic value is blank”
**Fix**
- Shares must be > 0.

### Problem: “Upside/Downside is blank”
**Fix**
- Current price must be > 0 (and shares must be valid).
"""
        )

    # -------------------------
    # DATA QUALITY & UNITS
    # -------------------------
    with st.expander("7) Data quality, units & currency problems (big values / tiny values)", expanded=True):
        st.markdown(
            r"""
### Problem: “Values are 10× / 100× / 1000× too big/small”
This is almost always **units or FX**.

**Check:**
- Are statements in USD already?  
  If yes, do **NOT** apply FX conversion.
- If converting ZWG → USD:
  - Make sure the FX rate column is correct
  - Check if you applied the ZiG factor correctly (only in relevant date ranges)
- Confirm whether your statements are in:
  - dollars, thousands, or millions
  If statements are “in thousands”, your valuation must be scaled consistently.

✅ Practical check:
Compare one known line item (e.g., Total Equity) to an annual report figure to confirm scale.
"""
        )
# -------------------------
# TAB7 ⚡ QUICK SUMMARY (FAST HOW-TO)
# -------------------------
with tab7:
    st.markdown(
        """
        <div class="card">
          <h3><span class="pill">FAST</span> ⚡ Quick Summary — How to Value Using Each Model</h3>
          <p class="subtle">
            A 60-second guide. Follow these steps to get a valuation quickly (then use the other tabs for deeper detail).
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
              <h3>📉 DCF (UFCF / FCFF)</h3>
              <div class="callout">
                <b>Quick steps</b><br>
                1) Upload IS + BS + CF (one Excel, 3 sheets).<br>
                2) Select currency (USD or ZWG + FX file).<br>
                3) Map: Revenue, Debt, Cash, CA, CL, Equity, Capex, Depreciation (if available).<br>
                4) Choose forecast years + revenue growth method.<br>
                5) Confirm WC% method + CAPEX averaging (exclude outliers).<br>
                6) Set WACC inputs (RF, MRP, beta, tax, Rd) + terminal g.<br>
                7) On cost of debt you can manually override or use the auto cost of debt.<br>
                8) Select Valuation timing by first selecting today's date and then the Financial statement year-end date.<br>
                9) Review EV → Equity and the WACC vs g sensitivity grid.<br>
                10) Export Excel for audit trail.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
              <h3>💰 DDM (Gordon Growth)</h3>
              <div class="callout">
                <b>Quick steps</b><br>
                1) Enter dividend history (prefer DPS if you want value per share).<br>
                2) Pick stable years for growth range (avoid special/irregular dividends).<br>
                3) Confirm computed g and D1.<br>
                4) Set CAPM inputs (RF, MRP, beta, D/E, tax) or override manually.<br>
                5) Check: <span class="mono">Ke &gt; g</span> then compute P0.<br>
                6) Enter shares to get total equity value.<br>
                7) Export Excel model.
              </div>
              <div class="callout warn">
                <b>Key rule:</b> If <span class="mono">Ke ≤ g</span>, the model is invalid.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
              <h3>📈 Comparables (EV/EBITDA · P/B · P/E)</h3>

              <div class="callout">
                <b>Quick Setup</b><br>
                1) (Optional) Turn ON Peer Universe auto-fill and choose target.<br>
                2) Confirm peers and enter multiples (EV/EBITDA, P/B, P/E).<br>
                3) Use Include flags to remove outliers (do not delete).<br>
                4) Enter Discount % → system computes implied multiples.
              </div>

              <div style="margin-top:12px; padding:12px; border-radius:10px; background:#eef6ff; border-left:4px solid #2563eb;">
                <b>🔹 Maintainable EBITDA</b><br>
                • Select EBITDA year range<br>
                • Enter weights (%) for each year<br>
                • Choose whether to apply timing (from DCF) or not
              </div>

              <div style="margin-top:10px; padding:12px; border-radius:10px; background:#f0f9ff; border-left:4px solid #1d4ed8;">
                <b>🔹 Maintainable Earnings</b><br>
                • <b>Auto-applied from Maintainable EBITDA ONLY</b><br>
                • Uses the <b>same years</b>, <b>same weights (%)</b>, and <b>same timing choice</b><br>
                • Your job here is to <b>review</b> and confirm the earnings output
              </div>

              <div class="callout warn" style="margin-top:12px;">
                <b>Important:</b><br>
                EBITDA logic can automatically flow into Earnings.<br>
                Always run DCF first if you want EBITDA/Earnings auto-filled.
              </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="card">
              <h3>🏦 Banking (Residual Income)</h3>
              <div class="callout">
                <b>Quick steps</b><br>
                1) Upload IS + BS + SoCE and select correct sheets.<br>
                2) If ZWG: upload FX and confirm average vs closing conversion logic.<br>
                3) Map Total Equity rows on BS (and SoCE closing total if needed).<br>
                4) Choose the best earnings line (Normalized profit / PAT / Net profit).<br>
                5) Choose base year (Year 0) → confirm BV0 and Earnings0.<br>
                6) Set Ke via CAPM (auto or manual) + forecast years.<br>
                7) Enter growth assumptions (BV growth, earnings growth, terminal g).<br>
                8) Check: <span class="mono">Ke &gt; g</span> then compute equity value + export.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
          <h3>🧾 Summary (Blended / Weighted Valuation)</h3>
          <div class="callout">
            <b>Quick steps</b><br>
            1) Run the valuation tabs you want (DCF/DDM/Comps/Banking) first.<br>
            2) In Summary, select models to include.<br>
            3) Input weights (the app normalizes them to 100%).<br>
            4) Enter shares and current share price.<br>
            5) Review intrinsic value, upside/downside, and recommendation.<br>
            6) Export Summary Excel.
          </div>
          <div class="callout warn">
            <b>Common issue:</b> If a model shows blank, it wasn’t run yet.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

