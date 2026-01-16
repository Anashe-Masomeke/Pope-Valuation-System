import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import io

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # <-- go up from /pages to project root
DATA_DIR = PROJECT_ROOT / "data"

DCF_PARAMS_PATH = DATA_DIR / "dcf_parameters.xlsx"
UNLEVERED_BETAS_PATH = DATA_DIR / "unlevered_betas.xlsx"



# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
st.markdown("""
<style>
.fbc-reset-card {
    background: linear-gradient(135deg, #003399 0%, #0055cc 100%);
    padding: 20px 24px;
    border-radius: 14px;
    color: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.fbc-reset-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 6px;
}

.fbc-reset-sub {
    font-size: 14px;
    opacity: 0.9;
    margin-bottom: 14px;
}

.fbc-reset-btn button {
    background-color: #f5b400 !important;   /* FBC gold */
    color: #002266 !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    border: none !important;
    transition: all 0.25s ease-in-out;
}

.fbc-reset-btn button:hover {
    background-color: #ffd24d !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.dcf-card{
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
  margin-top: 10px;
  margin-bottom: 14px;
}
.dcf-card h3{
  margin: 0 0 8px 0;
}
.dcf-subcard{
  background: rgba(0,51,153,0.03);
  border: 1px solid rgba(0,51,153,0.10);
  border-radius: 14px;
  padding: 14px;
  margin-top: 10px;
}
.dcf-kpi{
  background: linear-gradient(135deg, rgba(0,51,153,0.10), rgba(245,180,0,0.10));
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 12px 14px;
  margin: 6px 0;
}
.dcf-kpi-title{
  font-size: 12px;
  opacity: 0.75;
  margin-bottom: 2px;
}
.dcf-kpi-value{
  font-size: 18px;
  font-weight: 800;
}
.small-note{
  font-size: 12px;
  opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# RESET DCF SESSION STATE (SAFE & CONTROLLED)
# ---------------------------------------------------------
def reset_dcf_state():
    keys_to_clear = [
        # file & parsed data
        "dcf_uploaded_file", "dcf_is_df", "dcf_bs_df", "dcf_cf_df",

        # ✅ ADD THESE (so reset truly clears the file)
        "dcf_file_bytes", "dcf_file_name",

        # FX
        "dcf_fx_file", "dcf_fx_raw", "dcf_yearly_fx",
        "dcf_fx_applied", "dcf_apply_fx_bs", "dcf_fx_column",
        "dcf_closing_fx_rate",
        "dcf_conversion_method", "dcf_currency",
        "dcf_fx_bytes", "dcf_fx_name",
        "dcf_factor_enabled", "dcf_zig_factor", "dcf_factor_year_ranges",
        "dcf_bs_fx_dirty",

        # mappings
        "dcf_mapping", "is_core_mapping",
        "bs_map_step", "bs_jump_radio", "bs_widget_reset",
        "cf_map_step", "cf_jump_radio", "cf_widget_reset",

        # forecasts
        "dcf_rev_forecast", "dcf_ebitda_all", "dcf_ebitda_forecast",
        "dcf_profit_all",
        # parameters + widgets
        "dcf_rf_pct", "dcf_mrp_pct", "dcf_tax_pct", "dcf_unlevered_beta", "dcf_terminal_g_pct",
        "dcf_rf_pct_input", "dcf_mrp_pct_input", "dcf_tax_pct_input",
        "dcf_unlevered_beta_input", "dcf_terminal_g_pct_input",
        "dcf_use_auto_params", "dcf_use_auto_params_ui",
        "dcf_country_select", "dcf_zim_avg_cost_debt_pct", "dcf_zim_avg_cost_debt_pct_input",
        "dcf_beta_manual_mode",
        "dcf_beta_manual_value",
        "dcf_beta_auto_last",
        "dcf_beta_mode_radio",

        # working capital
        "dcf_fcff_array", "dcf_pv_fcff_array", "dcf_discount_periods_n",
        "dcf_is_base", "dcf_bs_base", "dcf_cf_base",
        "dcf_fx_signature", "dcf_bs_fx_rates", "dcf_bs_closing_dates",

        # valuation outputs
        "enterprise_value_dcf", "equity_value_dcf", "equity_value",

        # parameters
        "dcf_init", "dcf_timing_init",
    ]

    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]

def clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    first_col = df.columns[0]
    df.rename(columns={first_col: "Item"}, inplace=True)

    for col in df.columns[1:]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "-", regex=False)
            .str.replace(")", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_year_cols(df: pd.DataFrame):
    return [c for c in df.columns if c != "Item"]


def avg_revenue_growth(revenue_row: pd.DataFrame, year_cols) -> float:
    vals = revenue_row[year_cols].values.flatten().astype(float)
    growth = []
    for i in range(1, len(vals)):
        prev_, curr_ = vals[i - 1], vals[i]
        if curr_ != 0:
            g = (curr_ - prev_) / curr_
            if -0.5 < g < 0.5:
                growth.append(g)
    return float(np.mean(growth)) if growth else 0.05


def ratio_to_revenue(row_vals: np.ndarray, rev_vals: np.ndarray) -> float:
    mask = (~np.isnan(row_vals)) & (~np.isnan(rev_vals)) & (rev_vals != 0)
    if not mask.any():
        return 0.0
    ratios = row_vals[mask] / rev_vals[mask]
    ratios = ratios[(ratios > -5) & (ratios < 5)]
    return float(np.mean(ratios)) if len(ratios) else 0.0


def find_row_indices(df: pd.DataFrame, keywords):
    if df.empty:
        return []
    s = df["Item"].astype(str).str.lower()
    mask = False
    for kw in keywords:
        mask = mask | s.str.contains(kw, na=False)
    return list(df[mask].index)


def find_single_row(df: pd.DataFrame, keywords):
    idx_list = find_row_indices(df, keywords)
    return (idx_list[0], df.iloc[idx_list[0]]) if idx_list else (None, None)


def convert_df_yearwise(df: pd.DataFrame, year_rates: dict) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if col == "Item":
            continue
        if str(col) in year_rates and year_rates[str(col)] != 0:
            df2[col] = df2[col] / year_rates[str(col)]
    return df2
def get_fx_asof_date(fx_df, fx_col, closing_date):
    """
    Returns the last available FX rate on or before the closing_date
    """
    fx_before = fx_df[fx_df["Date"] <= pd.Timestamp(closing_date)]
    if fx_before.empty:
        return None
    return float(fx_before.sort_values("Date").iloc[-1][fx_col])


def load_fx_yearly_from_excel(fx_file) -> dict:
    """
    Excel must contain:
    Date | Interbank | Alternative | Premium  (or similar)
    """
    fx = pd.read_excel(fx_file)
    fx.columns = [str(c) for c in fx.columns]

    date_col = fx.columns[0]
    fx[date_col] = pd.to_datetime(fx[date_col], errors="coerce")

    rate_col = st.selectbox(
        "Which FX rate column should be used?",
        fx.columns[1:]
    )

    fx[rate_col] = pd.to_numeric(fx[rate_col], errors="coerce")
    fx = fx.dropna(subset=[date_col, rate_col])

    fx["Year"] = fx[date_col].dt.year
    yearly = fx.groupby("Year")[rate_col].mean()

    return {str(int(y)): float(v) for y, v in yearly.items()}


def option_labels_from_items(items):
    return [f"{i+1}: {name}" for i, name in enumerate(items)]


def indices_from_labels(labels):
    idx = []
    for s in labels:
        try:
            idx.append(int(s.split(":", 1)[0]) - 1)
        except:
            pass
    return idx

@st.cache_data(show_spinner=False)
def _load_unlevered_betas_any(file_or_path, file_mtime: float = 0.0) -> pd.DataFrame:
    """
    Excel required columns (flexible match):
      Industry Name | Unlevered beta
    Also supports: Column1 (industry) + Column6 (beta)

    file_mtime is used ONLY to invalidate Streamlit cache when the Excel file changes.
    """
    df = pd.read_excel(file_or_path)
    df.columns = [str(c).strip() for c in df.columns]

    possible_industry_cols = [c for c in df.columns if c.lower() in ["industry name", "industry", "column1"]]
    possible_beta_cols = [c for c in df.columns if c.lower() in ["unlevered beta", "unlevered_beta", "beta", "column6"]]

    if not possible_industry_cols or not possible_beta_cols:
        raise ValueError("Excel must have 'Industry Name' and 'Unlevered beta' (or Column1 + Column6).")

    ind_col = possible_industry_cols[0]
    beta_col = possible_beta_cols[0]

    out = df[[ind_col, beta_col]].copy()
    out.columns = ["Industry", "UnleveredBeta"]
    out["Industry"] = out["Industry"].astype(str).str.strip()
    out["UnleveredBeta"] = pd.to_numeric(out["UnleveredBeta"], errors="coerce")
    out = out.dropna(subset=["Industry", "UnleveredBeta"]).sort_values("Industry").reset_index(drop=True)
    return out


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Forecast + DCF (IS + BS + CF)",
    layout="wide"
)

st.title("📊 Forecast + DCF Valuation")
# ---------------------------------------------------------
# 🔄 START NEW VALUATION — FBC STYLE
# ---------------------------------------------------------
st.markdown("""
<div class="fbc-reset-card">
    <div class="fbc-reset-title">🔄 Start New Valuation</div>
    <div class="fbc-reset-sub">
        Reset the workspace and upload a new set of financial statements.
    </div>
</div>
""", unsafe_allow_html=True)

col_reset_left, col_reset_right = st.columns([1, 3])

with col_reset_left:
    st.markdown('<div class="fbc-reset-btn">', unsafe_allow_html=True)
    if st.button("🗂️ Clear & Upload New File", width='stretch'):
        reset_dcf_state()

        # ✅ Increment uploader key to force Streamlit to forget previous upload
        st.session_state["dcf_uploader_key"] = st.session_state.get("dcf_uploader_key", 0) + 1

        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------


st.subheader("📂 Upload Financial Statements")

# --- Persist uploaded file bytes so the app doesn't "forget" after inactivity/rerun ---
if "dcf_file_bytes" not in st.session_state:
    st.session_state["dcf_file_bytes"] = None
if "dcf_file_name" not in st.session_state:
    st.session_state["dcf_file_name"] = None

# ✅ Initialize uploader key BEFORE using it
if "dcf_uploader_key" not in st.session_state:
    st.session_state["dcf_uploader_key"] = 0

uploaded_file = st.file_uploader(
    "Upload Excel with IS, BS, CF",
    type=["xlsx"],
    key=f"dcf_main_uploader_{st.session_state['dcf_uploader_key']}"
)


# Save bytes once
if uploaded_file is not None:
    st.session_state["dcf_file_bytes"] = uploaded_file.getvalue()
    st.session_state["dcf_file_name"] = uploaded_file.name

# Rebuild a file-like object from bytes
if st.session_state["dcf_file_bytes"] is None:
    st.info("⬆️ Please upload an Excel file to begin.")
    st.stop()

file_like = io.BytesIO(st.session_state["dcf_file_bytes"])

# Now use this in ExcelFile
xls = pd.ExcelFile(file_like)


# ---------------------------------------------------------
# LOAD & CACHE PARSED STATEMENTS (ONCE)
# ---------------------------------------------------------
if "dcf_is_df" not in st.session_state:
    xls = pd.ExcelFile(io.BytesIO(st.session_state["dcf_file_bytes"]))  # ✅ always valid

    st.session_state["dcf_is_df"] = clean_numeric_cols(xls.parse(xls.sheet_names[0]))
    st.session_state["dcf_bs_df"] = clean_numeric_cols(xls.parse(xls.sheet_names[1]))
    st.session_state["dcf_cf_df"] = clean_numeric_cols(xls.parse(xls.sheet_names[2]))

is_df = st.session_state["dcf_is_df"]
bs_df = st.session_state["dcf_bs_df"]
cf_df = st.session_state["dcf_cf_df"]

# ---------------------------------------------------------
# STORE BASE (ORIGINAL) STATEMENTS ONCE (for re-conversion)
# ---------------------------------------------------------
if "dcf_is_base" not in st.session_state:
    st.session_state["dcf_is_base"] = st.session_state["dcf_is_df"].copy()
if "dcf_bs_base" not in st.session_state:
    st.session_state["dcf_bs_base"] = st.session_state["dcf_bs_df"].copy()
if "dcf_cf_base" not in st.session_state:
    st.session_state["dcf_cf_base"] = st.session_state["dcf_cf_df"].copy()

year_cols_is = get_year_cols(is_df)
year_cols_bs = get_year_cols(bs_df)
year_cols_cf = get_year_cols(cf_df)
# ---------------------------------------------------------
# FX SECTION — EXCEL-BASED (ZWG → USD) [FIXED VERSION]
# ---------------------------------------------------------
# FX SECTION — EXCEL-BASED (ZWG → USD) — FINAL & CORRECT
# ---------------------------------------------------------
st.markdown("### 💱 Currency & Exchange Rates")
# ✅ Persist conversion method across tabs/pages
# conversion_method can be: "NO_FX" or "FX_EXCEL"
if "dcf_conversion_method" not in st.session_state:
    st.session_state["dcf_conversion_method"] = "NO_FX"

if "dcf_currency" not in st.session_state:
    st.session_state["dcf_currency"] = "USD (already converted)"

# -------------------------------------------------
# 1️⃣ Currency selector (persistent)
# -------------------------------------------------
currency = st.selectbox(
    "Currency of uploaded statements:",
    ["USD (already converted)", "ZWG (convert using FX Excel)"],
    index=0 if st.session_state.get("dcf_conversion_method") == "NO_FX" else 1,
    key="dcf_currency_select"
)

# ✅ Store BOTH: the label + a stable method flag
st.session_state["dcf_currency"] = currency
st.session_state["dcf_conversion_method"] = "NO_FX" if currency.startswith("USD") else "FX_EXCEL"


# -------------------------------------------------
# 2️⃣ FX Excel upload — SHOW ONLY IF ZWG
# -------------------------------------------------
# ✅ Persist FX file across reruns/tabs using bytes
if "dcf_fx_bytes" not in st.session_state:
    st.session_state["dcf_fx_bytes"] = None
if "dcf_fx_name" not in st.session_state:
    st.session_state["dcf_fx_name"] = None
if st.session_state["dcf_conversion_method"] == "FX_EXCEL":

    st.markdown("""
    <div style="
        border: 1px dashed #f5b400;
        padding: 18px;
        border-radius: 12px;
        background-color: #fffaf0;
        margin-bottom: 15px;
    ">
        <strong>📥 FX Data Required</strong><br>
        Upload exchange rates to convert ZWG → USD
    </div>
    """, unsafe_allow_html=True)

    fx_file = st.file_uploader(
        "Upload FX Excel (Date + FX columns)",
        type=["xlsx"],
        key="dcf_fx_uploader"
    )

    # ✅ Save FX bytes once
    if fx_file is not None:
        st.session_state["dcf_fx_bytes"] = fx_file.getvalue()
        st.session_state["dcf_fx_name"] = fx_file.name

else:
    # USD selected → clear FX bytes
    st.session_state["dcf_fx_bytes"] = None
    st.session_state["dcf_fx_name"] = None

# -------------------------------------------------
# 3️⃣ If USD → skip FX
# -------------------------------------------------
if currency.startswith("USD"):
    st.success("✅ Data assumed to be in USD. No FX conversion applied.")

else:
    st.warning("ZWG detected. Upload FX Excel with Dates and Interbank Rates to convert to USD.")

    if st.session_state["dcf_fx_bytes"] is None:
        st.stop()

    # -------------------------------------------------
    # 4️⃣ Load FX Excel ONCE
    # -------------------------------------------------
    if "dcf_fx_raw" not in st.session_state:
        fx_raw = pd.read_excel(io.BytesIO(st.session_state["dcf_fx_bytes"]))
        fx_raw.columns = [str(c).strip() for c in fx_raw.columns]
        st.session_state["dcf_fx_raw"] = fx_raw
    else:
        fx_raw = st.session_state["dcf_fx_raw"]

    st.subheader("Raw FX data (preview)")
    st.dataframe(fx_raw.head(), width='stretch')
    st.subheader("📊 Balance Sheet Closing FX Rates Used")
    bs_fx_rates = st.session_state.get("dcf_bs_fx_rates", {})

    # Build BS FX confirmation table safely
    bs_fx_table = pd.DataFrame([
        {
            "Year": y,
            "Closing Date": st.session_state["dcf_bs_closing_dates"][y],
            "FX Rate Used": bs_fx_rates[y],
        }
        for y in bs_fx_rates.keys()
    ])

    st.dataframe(bs_fx_table, width='stretch')

    # -------------------------------------------------
    # 5️⃣ Validate required columns
    # -------------------------------------------------
    if "Date" not in fx_raw.columns:
        st.error("❌ FX Excel must contain a column named 'Date'.")
        st.stop()

    fx_df = fx_raw.copy()

    fx_df["Date"] = pd.to_datetime(
        fx_df["Date"],
        errors="coerce",
        dayfirst=True
    )

    fx_df = fx_df.dropna(subset=["Date"])

    # -------------------------------------------------
    # 6️⃣ FX column selector (restricted + persistent)
    # -------------------------------------------------
    allowed_fx_cols = ["Interbank", "Alternative", "Premium"]
    available_fx_cols = [c for c in allowed_fx_cols if c in fx_df.columns]

    if not available_fx_cols:
        st.error("❌ FX Excel must contain Interbank / Alternative / Premium columns.")
        st.stop()

    if "dcf_fx_column" not in st.session_state:
        st.session_state["dcf_fx_column"] = available_fx_cols[0]

    fx_col = st.selectbox(
        "Which FX rate column should be used?",
        available_fx_cols,
        index=available_fx_cols.index(st.session_state["dcf_fx_column"]),
        key="dcf_fx_column_select"
    )

    st.session_state["dcf_fx_column"] = fx_col

    fx_df[fx_col] = pd.to_numeric(fx_df[fx_col], errors="coerce")
    fx_df = fx_df.dropna(subset=[fx_col])
    # -------------------------------------------------
    # 🪙 Apply conversion factor by selected Year(s) + Date Ranges
    # -------------------------------------------------
    st.markdown("### 🪙 Apply ZWG→ZiG factor by Year + Range")

    # years available from statements
    available_years = sorted({str(int(y)) for y in year_cols_is})

    if "dcf_factor_enabled" not in st.session_state:
        st.session_state["dcf_factor_enabled"] = False

    if "dcf_zig_factor" not in st.session_state:
        st.session_state["dcf_zig_factor"] = 2498.7242

    if "dcf_factor_year_ranges" not in st.session_state:
        # {"2024": [{"start": date(...), "end": date(...)}], ...}
        st.session_state["dcf_factor_year_ranges"] = {}
    # ✅ Persist selected years across page/tab switches
    if "dcf_factor_years_selected_vals" not in st.session_state:
        st.session_state["dcf_factor_years_selected_vals"] = []

    enable_factor = st.checkbox(
        "Enable manual factor (for mixed ZWG/ZiG periods)",
        value=st.session_state["dcf_factor_enabled"],
        key="dcf_factor_enabled_ui"
    )
    st.session_state["dcf_factor_enabled"] = enable_factor

    zig_factor = st.number_input(
        "ZWG → ZiG conversion factor (divide FX by this inside selected ranges)",
        value=float(st.session_state["dcf_zig_factor"]),
        step=0.0001,
        format="%.6f",
        key="dcf_zig_factor_ui2"
    )
    st.session_state["dcf_zig_factor"] = zig_factor

    if enable_factor:
        years_selected = st.multiselect(
            "Select the year(s) where you want to apply the factor",
            available_years,
            default=[y for y in st.session_state["dcf_factor_years_selected_vals"] if y in available_years],
            key="dcf_factor_years_selected_ui"
        )

        # ✅ store it explicitly (this survives page/tab switching better)
        st.session_state["dcf_factor_years_selected_vals"] = years_selected

        for y in years_selected:
            st.session_state["dcf_factor_year_ranges"].setdefault(y, [])

            st.markdown(f"#### Ranges for {y}")

            if st.button(f"➕ Add range for {y}", key=f"add_range_{y}"):
                st.session_state["dcf_factor_year_ranges"][y].append({
                    "start": date(int(y), 1, 1),
                    "end": date(int(y), 12, 31),
                })

            ranges = st.session_state["dcf_factor_year_ranges"][y]
            for i, r in enumerate(ranges):
                c1, c2, c3 = st.columns([2, 2, 1])

                with c1:
                    new_start = st.date_input(
                        f"{y} range {i + 1} start",
                        value=r["start"],
                        key=f"{y}_r{i}_start"
                    )
                with c2:
                    new_end = st.date_input(
                        f"{y} range {i + 1} end",
                        value=r["end"],
                        key=f"{y}_r{i}_end"
                    )

                if new_end < new_start:
                    st.error("❌ End date cannot be before start date.")
                else:
                    st.session_state["dcf_factor_year_ranges"][y][i]["start"] = new_start
                    st.session_state["dcf_factor_year_ranges"][y][i]["end"] = new_end

                with c3:
                    if st.button("🗑️ Delete", key=f"{y}_r{i}_del"):
                        st.session_state["dcf_factor_year_ranges"][y].pop(i)
                        st.rerun()

        # Apply factor to FX rows in selected ranges
        if zig_factor <= 0:
            st.error("❌ Factor must be > 0.")
            st.stop()

        fx_df["_factor_applied"] = False

        for y in years_selected:
            for r in st.session_state["dcf_factor_year_ranges"].get(y, []):
                s = pd.Timestamp(r["start"])
                e = pd.Timestamp(r["end"])
                mask = (fx_df["Date"] >= s) & (fx_df["Date"] <= e)
                if mask.any():
                    fx_df.loc[mask, fx_col] = fx_df.loc[mask, fx_col] / float(zig_factor)
                    fx_df.loc[mask, "_factor_applied"] = True

        st.success(f"✅ Factor applied to {int(fx_df['_factor_applied'].sum()):,} FX rows.")
        st.dataframe(fx_df.loc[fx_df["_factor_applied"], ["Date", fx_col]].head(20), width='stretch')

    # -------------------------------------------------
    # 7️⃣ Compute YEARLY AVERAGE FX (Income Statement)
    # -------------------------------------------------
    fx_df["Year"] = fx_df["Date"].dt.year.astype(int)

    yearly_fx = (
        fx_df
        .groupby("Year")[fx_col]
        .mean()
        .round(6)
        .to_dict()
    )

    yearly_fx = {str(y): float(v) for y, v in yearly_fx.items()}
    st.session_state["dcf_yearly_fx"] = yearly_fx
    bs_fx_rates = st.session_state.get("dcf_bs_fx_rates", {})

    st.subheader("📊 Yearly FX averages (Income Statement and Cash Flow Statement)")
    st.dataframe(
        pd.DataFrame({
            "Year": yearly_fx.keys(),
            "FX Rate": yearly_fx.values()
        }),
        width='stretch'    )

    # -------------------------------------------------
    # 8️⃣ Balance Sheet FX OPTION (closing rate)
    # -------------------------------------------------
    if "dcf_apply_fx_bs" not in st.session_state:
        st.session_state["dcf_apply_fx_bs"] = False

    apply_fx_bs = st.checkbox(
        "Apply FX to Balance Sheet using closing rate?",
        value=st.session_state["dcf_apply_fx_bs"],
        help="Uses ONE FX rate (latest available date)",
        key="dcf_fx_bs_checkbox"
    )

    st.session_state["dcf_apply_fx_bs"] = apply_fx_bs
    # -------------------------------------------------
    # 8️⃣ Balance Sheet FX — PER-YEAR CLOSING DATES (NEW)
    # -------------------------------------------------
    st.markdown("### 📌 Balance Sheet FX — Closing Dates (per year)")

    # ✅ INIT FIRST (CRITICAL)
    if "dcf_bs_closing_dates" not in st.session_state:
        st.session_state["dcf_bs_closing_dates"] = {}

    # ✅ Dirty flag init
    if "dcf_bs_fx_dirty" not in st.session_state:
        st.session_state["dcf_bs_fx_dirty"] = False


    bs_years = [str(y) for y in year_cols_bs]

    for y in bs_years:
        default_date = st.session_state["dcf_bs_closing_dates"].get(
            y, date(int(y), 12, 31)
        )

        chosen_date = st.date_input(
            f"Closing date for Balance Sheet {y}",
            value=default_date,
            key=f"bs_close_date_{y}"
        )

        # ✅ Detect change immediately (fixes double click)
        if st.session_state["dcf_bs_closing_dates"].get(y) != chosen_date:
            st.session_state["dcf_bs_closing_dates"][y] = chosen_date
            st.session_state["dcf_bs_fx_dirty"] = True

    # -------------------------------------------------
    # 9️⃣ COMPUTE BALANCE SHEET FX RATES (PER YEAR)
    # -------------------------------------------------
    bs_fx_rates = {}

    for y in bs_years:
        closing_date = st.session_state["dcf_bs_closing_dates"][y]

        fx_rate = get_fx_asof_date(
            fx_df=fx_df,
            fx_col=fx_col,
            closing_date=closing_date
        )

        if fx_rate is None:
            st.error(f"❌ No FX rate found on or before {closing_date} for year {y}")
            st.stop()

        bs_fx_rates[y] = fx_rate

    # ✅ STORE IN SESSION STATE
    st.session_state["dcf_bs_fx_rates"] = bs_fx_rates

    # -------------------------------------------------
    # 🔟 Validate FX coverage for IS years
    # -------------------------------------------------
    statement_years = set(year_cols_is)
    fx_years = set(yearly_fx.keys())

    missing_years = sorted(statement_years - fx_years)

    if missing_years:
        st.error(
            f"❌ Missing FX data for statement years: {', '.join(missing_years)}"
        )
        st.stop()

    # -------------------------------------------------
    # 1️⃣1️⃣ APPLY FX CONVERSION (RE-RUN IF SETTINGS CHANGE)
    # -------------------------------------------------
    factor_signature = (
        st.session_state.get("dcf_factor_enabled", False),
        st.session_state.get("dcf_zig_factor", None),
        str(st.session_state.get("dcf_factor_year_ranges", {}))
    )

    fx_signature = (
        currency,
        fx_col,
        factor_signature,
        tuple((y, str(st.session_state["dcf_bs_closing_dates"][y])) for y in bs_years)
    )

    # Recompute if signature changed (or first run)
    if (
            st.session_state.get("dcf_fx_signature") != fx_signature
            or st.session_state.get("dcf_bs_fx_dirty")
    ):

        # Always start from BASE statements (pre-conversion)
        is_base = st.session_state["dcf_is_base"].copy()
        bs_base = st.session_state["dcf_bs_base"].copy()
        cf_base = st.session_state["dcf_cf_base"].copy()

        # Income Statement → YEARLY AVERAGE FX
        is_converted = convert_df_yearwise(is_base, yearly_fx)

        # Balance Sheet → PER-YEAR CLOSING FX
        bs_converted = convert_df_yearwise(bs_base, bs_fx_rates)

        # Cash Flow → SAME YEARLY AVERAGE FX AS IS
        cf_converted = convert_df_yearwise(cf_base, yearly_fx)

        # Save converted versions
        st.session_state["dcf_is_df"] = is_converted
        st.session_state["dcf_bs_df"] = bs_converted
        st.session_state["dcf_cf_df"] = cf_converted

        # Save the signature so we don't reconvert unnecessarily
        st.session_state["dcf_fx_signature"] = fx_signature
        st.session_state["dcf_bs_fx_dirty"] = False

        # Optional: for debugging / clarity
        st.info("🔁 FX conversion refreshed (settings changed).")
    else:
         st.success("✅ FX conversion applied correctly (IS = yearly average, BS = per-year closing rates)")

# ---------------------------------------------------------
# SHOW CLEANED STATEMENTS
# ---------------------------------------------------------
st.subheader("Income Statement (cleaned, in USD)")
st.dataframe(is_df, width='stretch')

st.subheader("Balance Sheet (cleaned, in USD)")
st.dataframe(bs_df, width='stretch')

st.subheader("Cash Flow Statement (cleaned, in USD)")
st.dataframe(cf_df, width='stretch')

# Re-detect year columns (as strings)
year_cols_is = get_year_cols(is_df)
year_cols_bs = get_year_cols(bs_df)
year_cols_cf = get_year_cols(cf_df)

if len(year_cols_is) < 2:
    st.error("❌ Need at least 2 historical year columns in Income Statement.")
    st.stop()

# Prepare year ints/labels
last_hist_label = year_cols_is[-1]           # string label e.g. "2025"
last_hist_year = int(str(last_hist_label))   # int 2025
# ---------------------------------------------------------
# FORECAST HORIZON (USER-DEFINED)
# ---------------------------------------------------------
if "dcf_forecast_years" not in st.session_state:
    st.session_state["dcf_forecast_years"] = 5

forecast_horizon = st.number_input(
    "Number of years to forecast",
    min_value=1,
    max_value=15,
    value=int(st.session_state["dcf_forecast_years"]),
    step=1,
    key="dcf_forecast_years_input"
)

st.session_state["dcf_forecast_years"] = forecast_horizon

forecast_years_int = [
    last_hist_year + i
    for i in range(1, forecast_horizon + 1)
]

forecast_cols = [str(y) for y in forecast_years_int]

# --- Persistent dictionary for DCF row mappings ---
if "dcf_mapping" not in st.session_state:
    st.session_state["dcf_mapping"] = {
        "debt": [],
        "cash": [],
        "ca": [],
        "cl": [],
        "dep": [],
        "capex": [],
        "interest": []
    }
def clean_defaults(default_list, options):
    """
    Keep only those default values that still exist in options.
    Prevents Streamlit error: 'default value ... is not part of the options'.
    """
    if not isinstance(default_list, (list, tuple)):
        return []
    return [x for x in default_list if x in options]

# ---------------------------------------------------------
# BALANCE SHEET — OPTION C WIZARD (multi-select + preview)
# ---------------------------------------------------------
BS_LINES = [
    ("debt",   "Total Debt / Borrowings (multi-select)"),
    ("cash",   "Cash & Cash Equivalents (multi-select)"),
    ("ca",     "Current Assets (for Working Capital) (multi-select)"),
    ("cl",     "Current Liabilities (for Working Capital) (multi-select)"),
    ("equity", "Equity (multi-select)"),
]

def map_bs_wizard(bs_df, year_cols_bs):
    st.markdown("### 🟩 Balance Sheet — Mapping")

    bs_items = list(bs_df["Item"].astype(str))
    bs_labels = option_labels_from_items(bs_items)

    if "dcf_mapping" not in st.session_state:
        st.session_state["dcf_mapping"] = {}
    for k, _ in BS_LINES:
        st.session_state["dcf_mapping"].setdefault(k, [])

    if "bs_map_step" not in st.session_state:
        st.session_state["bs_map_step"] = 0

    # --- progress
    mapped = sum(1 for k, _ in BS_LINES if len(st.session_state["dcf_mapping"].get(k, [])) > 0)
    st.progress(mapped / len(BS_LINES))
    st.caption(f"Mapped: {mapped}/{len(BS_LINES)}")

    step_names = [name for _, name in BS_LINES]

    # ✅ make radio fully controlled
    if "bs_jump_radio" not in st.session_state:
        st.session_state["bs_jump_radio"] = step_names[st.session_state["bs_map_step"]]

    def _set_step(i: int):
        i = max(0, min(i, len(BS_LINES) - 1))
        st.session_state["bs_map_step"] = i
        st.session_state["bs_jump_radio"] = step_names[i]   # ✅ move red dot
        st.rerun()

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("⬅️ Back (BS)", disabled=st.session_state["bs_map_step"] == 0):
            _set_step(st.session_state["bs_map_step"] - 1)

    with c2:
        if st.button("➡️ Next unmapped (BS)"):
            target = None
            for i, (k, _) in enumerate(BS_LINES):
                if len(st.session_state["dcf_mapping"].get(k, [])) == 0:
                    target = i
                    break
            if target is None:
                target = len(BS_LINES) - 1
            _set_step(target)

    # ✅ radio drives step too
    chosen_step = st.radio(
        "Jump to BS line:",
        step_names,
        key="bs_jump_radio",
        horizontal=True
    )
    st.session_state["bs_map_step"] = step_names.index(chosen_step)

    k, title = BS_LINES[st.session_state["bs_map_step"]]
    stored = clean_defaults(st.session_state["dcf_mapping"].get(k, []), bs_labels)

    # ✅ widget reset counters (versioned key)
    if "bs_widget_reset" not in st.session_state:
        st.session_state["bs_widget_reset"] = {}
    st.session_state["bs_widget_reset"].setdefault(k, 0)

    widget_key = f"bs_pick_{k}_{st.session_state['bs_widget_reset'][k]}"

    with st.container(border=True):
        st.markdown(f"#### {title}")

        sel = st.multiselect(
            "Select row(s):",
            bs_labels,
            default=stored,
            key=widget_key
        )

        st.session_state["dcf_mapping"][k] = sel

        if st.button("🧹 Clear selection", key=f"bs_clear_{k}"):
            st.session_state["dcf_mapping"][k] = []
            st.session_state["bs_widget_reset"][k] += 1  # ✅ forces a fresh widget
            st.rerun()

        # preview
        if sel:
            idx_list = indices_from_labels(sel)
            preview_vals = bs_df.loc[idx_list, year_cols_bs].sum(axis=0)
            st.caption("Preview (sum of selected rows):")
            st.dataframe(
                pd.DataFrame({"Year": year_cols_bs, "Total": preview_vals.values}),
                hide_index=True,
                width='stretch'
            )

    out = {}
    for kk, _ in BS_LINES:
        out[kk] = indices_from_labels(st.session_state["dcf_mapping"].get(kk, []))
    return out

bs_idx = map_bs_wizard(bs_df, year_cols_bs)

debt_idx_list   = bs_idx["debt"]
cash_idx_list   = bs_idx["cash"]
ca_idx_list     = bs_idx["ca"]
cl_idx_list     = bs_idx["cl"]
equity_idx_list = bs_idx["equity"]
# ---------------------------------------------------------
# CASH FLOW — OPTION C WIZARD (multi-select + preview)
# ---------------------------------------------------------
CF_LINES = [
    ("dep",      "Depreciation & Amortisation (multi-select)"),
    ("capex",    "Capex  (multi-select)"),
    ("interest", "Interest paid (if using CF for interest) (multi-select)"),
]

def map_cf_wizard(cf_df, year_cols_cf):
    st.markdown("### 📄 Cash Flow — Mapping")

    cf_items = list(cf_df["Item"].astype(str))
    cf_labels = option_labels_from_items(cf_items)

    if "dcf_mapping" not in st.session_state:
        st.session_state["dcf_mapping"] = {}
    for k, _ in CF_LINES:
        st.session_state["dcf_mapping"].setdefault(k, [])

    if "cf_map_step" not in st.session_state:
        st.session_state["cf_map_step"] = 0

    mapped = sum(1 for k, _ in CF_LINES if len(st.session_state["dcf_mapping"].get(k, [])) > 0)
    st.progress(mapped / len(CF_LINES))
    st.caption(f"Mapped: {mapped}/{len(CF_LINES)}")

    step_names = [name for _, name in CF_LINES]

    if "cf_jump_radio" not in st.session_state:
        st.session_state["cf_jump_radio"] = step_names[st.session_state["cf_map_step"]]

    def _set_step(i: int):
        i = max(0, min(i, len(CF_LINES) - 1))
        st.session_state["cf_map_step"] = i
        st.session_state["cf_jump_radio"] = step_names[i]   # ✅ move red dot
        st.rerun()

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("⬅️ Back (CF)", disabled=st.session_state["cf_map_step"] == 0):
            _set_step(st.session_state["cf_map_step"] - 1)

    with c2:
        if st.button("➡️ Next unmapped (CF)"):
            target = None
            for i, (k, _) in enumerate(CF_LINES):
                if len(st.session_state["dcf_mapping"].get(k, [])) == 0:
                    target = i
                    break
            if target is None:
                target = len(CF_LINES) - 1
            _set_step(target)

    chosen_step = st.radio(
        "Jump to CF line:",
        step_names,
        key="cf_jump_radio",
        horizontal=True
    )
    st.session_state["cf_map_step"] = step_names.index(chosen_step)

    k, title = CF_LINES[st.session_state["cf_map_step"]]
    stored = clean_defaults(st.session_state["dcf_mapping"].get(k, []), cf_labels)

    if "cf_widget_reset" not in st.session_state:
        st.session_state["cf_widget_reset"] = {}
    st.session_state["cf_widget_reset"].setdefault(k, 0)

    widget_key = f"cf_pick_{k}_{st.session_state['cf_widget_reset'][k]}"

    with st.container(border=True):
        st.markdown(f"#### {title}")

        sel = st.multiselect(
            "Select row(s):",
            cf_labels,
            default=stored,
            key=widget_key
        )

        st.session_state["dcf_mapping"][k] = sel

        if st.button("🧹 Clear selection", key=f"cf_clear_{k}"):
            st.session_state["dcf_mapping"][k] = []
            st.session_state["cf_widget_reset"][k] += 1
            st.rerun()

        if sel:
            idx_list = indices_from_labels(sel)
            preview_vals = cf_df.loc[idx_list, year_cols_cf].sum(axis=0)
            st.caption("Preview (sum of selected rows):")
            st.dataframe(
                pd.DataFrame({"Year": year_cols_cf, "Total": preview_vals.values}),
                hide_index=True,
                width='stretch'
            )

    out = {}
    for kk, _ in CF_LINES:
        out[kk] = indices_from_labels(st.session_state["dcf_mapping"].get(kk, []))
    return out

cf_idx = map_cf_wizard(cf_df, year_cols_cf)

dep_cf_idx_list    = cf_idx["dep"]
capex_cf_idx_list  = cf_idx["capex"]
int_cf_idx_list    = cf_idx["interest"]

# ---------------------------------------------------------
# INCOME STATEMENT FORECASTING
# ---------------------------------------------------------
# ---------------------------------------------------------
# INCOME STATEMENT — OPTION C (WIZARD: steps + progress + next unmapped)
# ---------------------------------------------------------
CORE_LINES = [
    ("rev", "Revenue"),
    ("cos", "Cost of Sales / Raw Materials (optional)"),
    ("gp", "Gross Profit"),
    ("ebitda", "EBITDA"),
    ("op", "Operating Profit / EBIT"),
    ("pbt", "Profit Before Tax"),
    ("tax", "Income Tax (Tax expense)"),
    ("np", "Profit for the Year"),
]

def _labels_from_items(items):
    return ["N/A (not in statement)"] + [f"{i+1}: {str(name)}" for i, name in enumerate(items)]

def map_core_is_totals_wizard(is_df, year_cols_is):
    st.markdown("### 🧾 Income Statement — Core Totals Mapping")

    items = list(is_df["Item"].astype(str))
    options = _labels_from_items(items)

    # init state once
    if "is_core_mapping" not in st.session_state:
        st.session_state["is_core_mapping"] = {k: None for k, _ in CORE_LINES}
    if "is_map_step" not in st.session_state:
        st.session_state["is_map_step"] = 0

    # progress
    mapped = sum(1 for k, _ in CORE_LINES if st.session_state["is_core_mapping"].get(k))
    st.progress(mapped / len(CORE_LINES))
    st.caption(f"Mapped: {mapped}/{len(CORE_LINES)}")

    # quick navigation buttons
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("⬅️ Back", disabled=st.session_state["is_map_step"] == 0):
            st.session_state["is_map_step"] -= 1
            st.rerun()

    with c2:
        # jump to next unmapped
        if st.button("➡️ Next unmapped"):
            for i, (k, _) in enumerate(CORE_LINES):
                if not st.session_state["is_core_mapping"].get(k):
                    st.session_state["is_map_step"] = i
                    st.rerun()
            # if all mapped, stay at end
            st.session_state["is_map_step"] = len(CORE_LINES) - 1
            st.rerun()

    # step selector (feels interactive + reduces page length)
    step_names = [name for _, name in CORE_LINES]
    step = st.radio(
        "Jump to line:",
        step_names,
        index=int(st.session_state["is_map_step"]),
        horizontal=True
    )
    st.session_state["is_map_step"] = step_names.index(step)

    # current step UI
    k, title = CORE_LINES[st.session_state["is_map_step"]]
    stored = st.session_state["is_core_mapping"].get(k)
    default = stored if stored in options else "N/A (not in statement)"
    default_index = options.index(default)

    chosen = st.selectbox(
        f"Select statement line for: **{title}**",
        options,
        index=default_index,
        key=f"is_pick_{k}"
    )

    st.session_state["is_core_mapping"][k] = None if chosen.startswith("N/A") else chosen

    # small preview for selected row (makes it feel alive)
    if not chosen.startswith("N/A"):
        idx = int(chosen.split(":", 1)[0]) - 1
        row_vals = is_df.iloc[idx][year_cols_is]
        st.dataframe(
            pd.DataFrame({"Year": year_cols_is, "Value": row_vals.values}),
            hide_index=True,
            width='stretch'
        )

    # convert to indices
    idx_map = {}
    for kk, _ in CORE_LINES:
        v = st.session_state["is_core_mapping"].get(kk)
        idx_map[kk] = (int(v.split(":", 1)[0]) - 1) if v else None

    # required check
    if idx_map["rev"] is None:
        st.error("❌ Revenue must be selected.")
        st.stop()

    return idx_map


# ✅ use this instead of your old mapping call
core_idx = map_core_is_totals_wizard(is_df, year_cols_is)



rev_idx    = core_idx["rev"]
cos_idx    = core_idx["cos"]
gp_idx     = core_idx["gp"]
ebitda_idx = core_idx["ebitda"]
op_idx     = core_idx["op"]
pbt_idx    = core_idx["pbt"]
tax_idx    = core_idx["tax"]   # ✅ ADD
np_idx     = core_idx["np"]



# ✅ Only Revenue is mandatory
if rev_idx is None:
    st.error("❌ Revenue must be selected.")
    st.stop()

# ✅ Cost of Sales is OPTIONAL
if cos_idx is None:
    st.warning("⚠️ Cost of Sales / Raw Materials not selected. Forecast will run using other lines as % of revenue.")


revenue_row = is_df.iloc[[rev_idx]]

# Calculate historical growth
calculated_g = avg_revenue_growth(revenue_row, year_cols_is)

st.markdown(f"📌 **Calculated Avg Revenue Growth:** {calculated_g:.2%}")

# --- Persistent revenue growth override ---
if "dcf_rev_growth_override" not in st.session_state:
    st.session_state["dcf_rev_growth_override"] = None   # means "not overridden yet"


# Determine what value to display in the input
default_display_value = (
    st.session_state["dcf_rev_growth_override"] * 100
    if st.session_state["dcf_rev_growth_override"] is not None
    else calculated_g * 100
)

# User override input
override_input = st.number_input(
    "Override revenue growth (%) if needed:",
    value=float(default_display_value),
    step=0.1,
    format="%.2f",
)

# Save to session_state as DECIMAL
st.session_state["dcf_rev_growth_override"] = override_input / 100

# Use final revenue growth for forecasting
avg_g = (
    st.session_state["dcf_rev_growth_override"]
    if st.session_state["dcf_rev_growth_override"] is not None
    else calculated_g
)


# ---------------------------------------------------------
# BUILD FORECAST INCOME STATEMENT
# ---------------------------------------------------------
forecast_is = is_df.copy()

# 🔥 ENSURE forecast columns exist
for col in forecast_cols:
    if col not in forecast_is.columns:
        forecast_is[col] = np.nan

# revenue forecast (ALLOW YEAR-BY-YEAR GROWTH)
rev_hist_vals = revenue_row[year_cols_is].values.flatten().astype(float)


# INCOME TAX RATE (Income Tax / Profit Before Tax) → AVERAGE
# ---------------------------------------------------------
avg_tax_ratio = 0.0

if isinstance(tax_idx, int) and isinstance(pbt_idx, int):

    tax_hist_vals = forecast_is.iloc[tax_idx][year_cols_is].values.astype(float)
    pbt_hist_vals = forecast_is.iloc[pbt_idx][year_cols_is].values.astype(float)

    mask = (~np.isnan(tax_hist_vals)) & (~np.isnan(pbt_hist_vals)) & (pbt_hist_vals > 0)

    if mask.any():
        pbt_valid = pbt_hist_vals[mask]
        tax_valid = tax_hist_vals[mask]

        # only profitable years
        profit_mask = pbt_valid > 0

        # KEEP SIGN: tax is negative, so ratio should be negative
        ratios = tax_valid[profit_mask] / pbt_valid[profit_mask]

        # sane cap for negative tax ratios (-60% to 0%)
        ratios = ratios[(ratios <= 0) & (ratios >= -1.50)]

        if len(ratios):
            avg_tax_ratio = float(np.mean(ratios))

# --- UI choice: same growth vs year-by-year growth ---
st.markdown("### 📈 Revenue Growth Method")

if "dcf_rev_growth_mode" not in st.session_state:
    st.session_state["dcf_rev_growth_mode"] = "Uniform (same % each year)"

growth_mode = st.radio(
    "Choose how you want to apply revenue growth:",
    ["Uniform (same % each year)", "Different growth per year"],
    index=0 if st.session_state["dcf_rev_growth_mode"].startswith("Uniform") else 1,
    key="dcf_rev_growth_mode_radio"
)
st.session_state["dcf_rev_growth_mode"] = growth_mode

# ---------------------------------------------------------
# AUTO YEAR-BY-YEAR GROWTH ENGINE (fade to long-run)
# ---------------------------------------------------------
def auto_growth_curve(start_g: float, terminal_g: float, n: int, speed: float = 0.55):
    """
    Returns a list of n growth rates that fade from start_g to terminal_g.
    speed in (0,1): higher = faster fade.
    """
    out = []
    g = start_g
    for _ in range(n):
        # move part-way toward terminal each year
        g = terminal_g + (g - terminal_g) * (1 - speed)
        out.append(g)
    return out

# --- If year-by-year, store a % for each forecast year ---
if "dcf_yearly_growth_pct" not in st.session_state:
    st.session_state["dcf_yearly_growth_pct"] = {}

yearly_g = {}

if growth_mode == "Different growth per year":
    st.markdown("#### Enter growth for each forecast year (%)")
    for y in forecast_years_int:
        default_pct = st.session_state["dcf_yearly_growth_pct"].get(str(y), avg_g * 100)
        pct = st.number_input(
            f"Growth for {y} (%)",
            value=float(default_pct),
            step=0.1,
            format="%.2f",
            key=f"growth_{y}"
        )
        st.session_state["dcf_yearly_growth_pct"][str(y)] = pct
        yearly_g[y] = pct / 100.0
else:
    # Uniform growth uses avg_g from your existing logic
    for y in forecast_years_int:
        yearly_g[y] = avg_g

# --- Now forecast revenue using the selected growth rates ---
rev_forecast = {}
current_rev = float(rev_hist_vals[-1])

for y in forecast_years_int:
    g_y = yearly_g[y]
    current_rev = current_rev * (1 + g_y)
    rev_forecast[y] = current_rev
    forecast_is.iat[rev_idx, forecast_is.columns.get_loc(str(y))] = current_rev


# ---------------------------------------------------------
# COST / GROSS PROFIT HANDLING (COS OPTIONAL)
# ---------------------------------------------------------
has_cos = isinstance(cos_idx, int)
has_gp  = isinstance(gp_idx, int)

# 1) If GP exists, compute historical GP margin
avg_gp_margin = None
if has_gp:
    gp_hist_vals = forecast_is.iloc[gp_idx][year_cols_is].values.astype(float)
    mask = (rev_hist_vals != 0) & (~np.isnan(gp_hist_vals)) & (~np.isnan(rev_hist_vals))
    gp_margins = gp_hist_vals[mask] / rev_hist_vals[mask]
    gp_margins = gp_margins[(gp_margins > -5) & (gp_margins < 5)]
    avg_gp_margin = float(np.mean(gp_margins)) if len(gp_margins) else 0.30

# ✅ CASE A: GP + COS exist → forecast COS using GP margin (your original approach)
if has_gp and has_cos and avg_gp_margin is not None:

    last_cos_hist = float(forecast_is.iloc[cos_idx][last_hist_label])
    cos_sign = -1 if last_cos_hist < 0 else 1

    for y in forecast_years_int:
        forecast_is.iat[cos_idx, forecast_is.columns.get_loc(str(y))] = (
            cos_sign * rev_forecast[y] * (1 - avg_gp_margin)
        )

    st.success(f"✅ COS forecasted using average GP margin = {avg_gp_margin:.2%}")

# ✅ CASE B: GP exists but COS missing → forecast GP directly
elif has_gp and (not has_cos) and avg_gp_margin is not None:

    for y in forecast_years_int:
        forecast_is.iat[gp_idx, forecast_is.columns.get_loc(str(y))] = (
            rev_forecast[y] * avg_gp_margin
        )

    st.info(f"ℹ️ COS not selected. GP forecasted using average GP margin = {avg_gp_margin:.2%}")

# ✅ CASE C: COS exists but GP missing → forecast COS as % of revenue
elif has_cos and (not has_gp):

    cos_hist_vals = forecast_is.iloc[cos_idx][year_cols_is].values.astype(float)
    cos_ratio = ratio_to_revenue(cos_hist_vals, rev_hist_vals)

    for y in forecast_years_int:
        forecast_is.iat[cos_idx, forecast_is.columns.get_loc(str(y))] = (
            rev_forecast[y] * cos_ratio
        )

    st.info(f"ℹ️ GP not selected. COS forecasted as % of revenue (avg ratio = {cos_ratio:.2%})")

# ✅ CASE D: Neither GP nor COS exists → do nothing special (rest of rows will still forecast)
else:
    st.info("ℹ️ GP and COS not selected. Forecast will rely on other rows as % of revenue.")


industry = st.session_state.get("dcf_industry", "General")

treat_cos_as_normal = (
    industry == "Manufacturing"
    or cos_idx is None
)

# forecast other non-total, non-CoS rows as % of revenue
industry = st.session_state.get("dcf_industry", "General")
treat_cos_as_normal = industry == "Manufacturing" or cos_idx is None

total_keywords = [
    "gross profit", "ebitda",
    "operating profit",
    "profit before tax",
    "profit for the year",
]
# ✅ gp_cos_mode should be True only when you actually forecast COS using GP margin
# Put this right AFTER your upgraded COS/GP handling block:
gp_cos_mode = (has_gp and has_cos and (avg_gp_margin is not None))

total_keywords = [
    "gross profit", "ebitda",
    "operating profit",
    "profit before tax",
    "profit for the year",
]

for idx in range(len(forecast_is)):

    # ✅ Build protected list safely (remove None)
    protected = [x for x in [rev_idx, gp_idx, cos_idx, ebitda_idx, op_idx, pbt_idx, tax_idx, np_idx] if isinstance(x, int)]

    # ✅ If COS was forecasted already using GP margin, skip COS so you don't overwrite it
    if gp_cos_mode and has_cos and idx == cos_idx:
        continue

    # ✅ Don't forecast totals / protected rows
    if idx in protected:
        continue

    item = str(forecast_is.at[idx, "Item"]).lower()
    if any(k in item for k in total_keywords):
        continue

    # Forecast everything else as % of revenue
    row_hist = forecast_is.iloc[idx][year_cols_is].values.astype(float)
    ratio = ratio_to_revenue(row_hist, rev_hist_vals)

    for y in forecast_years_int:
        forecast_is.iat[idx, forecast_is.columns.get_loc(str(y))] = rev_forecast[y] * ratio




def sum_rows(df, start_idx, end_idx, col):
    """Sum from start_idx to end_idx-1 inclusive."""
    if start_idx is None or end_idx is None:
        return df.iloc[start_idx][col] if start_idx is not None else np.nan
    if end_idx <= start_idx:
        return df.iloc[start_idx][col]
    return df.loc[start_idx:end_idx - 1, col].sum(skipna=True)

for col in forecast_cols:

    # 🔹 Derive Gross Profit if row exists
    if gp_idx is not None and cos_idx is not None:
        forecast_is.iat[
            gp_idx,
            forecast_is.columns.get_loc(col)
        ] = (
            forecast_is.iloc[rev_idx][col]
            + forecast_is.iloc[cos_idx][col]
        )

    # -------------------------------------------------
    # EBITDA recomputation (robust to missing GP)
    # -------------------------------------------------
    if ebitda_idx is not None:

        col_idx = forecast_is.columns.get_loc(col)

        existing_val = pd.to_numeric(
            forecast_is.iat[ebitda_idx, col_idx],
            errors="coerce"
        )

        # Only recompute if not explicitly forecasted
        if pd.isna(existing_val) or existing_val == 0:

            # 🔹 Decide where to start summing from
            if isinstance(gp_idx, int):
                start_idx = gp_idx
            else:
                # 🔥 GP missing → start from Revenue
                start_idx = rev_idx

            if ebitda_idx > start_idx:
                ebitda_val = forecast_is.loc[
                    start_idx:ebitda_idx - 1,
                    col
                ].sum(skipna=True)

                forecast_is.iat[ebitda_idx, col_idx] = ebitda_val

    if op_idx is not None and ebitda_idx is not None:
        forecast_is.iat[op_idx, forecast_is.columns.get_loc(col)] = \
            forecast_is.loc[ebitda_idx:op_idx-1, col].sum()

    if pbt_idx is not None and op_idx is not None:
        forecast_is.iat[pbt_idx, forecast_is.columns.get_loc(col)] = \
            forecast_is.loc[op_idx:pbt_idx-1, col].sum()

    # ✅ ADD THIS HERE
    if isinstance(tax_idx, int) and isinstance(pbt_idx, int):
        pbt_val = float(forecast_is.iat[pbt_idx, forecast_is.columns.get_loc(col)])
        tax_val = pbt_val * avg_tax_ratio
        forecast_is.iat[tax_idx, forecast_is.columns.get_loc(col)] = tax_val

    if np_idx is not None and pbt_idx is not None:
        forecast_is.iat[np_idx, forecast_is.columns.get_loc(col)] = \
            forecast_is.loc[pbt_idx:np_idx-1, col].sum()

# ---------------------------------------------------------
# STORE FORECASTED NET PROFIT (Profit for the Year) FOR COMPARABLES
# ---------------------------------------------------------
dcf_np_forecast = {}

if np_idx is not None:
    for y in forecast_years_int:
        col = str(y)
        if isinstance(np_idx, int):
            val = forecast_is.iat[np_idx, forecast_is.columns.get_loc(col)]
            val = float(val) if pd.notna(val) else 0.0
        else:
            val = 0.0
        dcf_np_forecast[col] = val
else:
    dcf_np_forecast = {}

# =========================================================
# STORE ALL NET PROFIT VALUES (HISTORICAL + FORECAST)
# =========================================================
dcf_profit_all = {}

# 1. Include historical
if np_idx is not None:
    for col in year_cols_is:  # historical labels (strings)
        val = float(is_df.iloc[np_idx][col])
        dcf_profit_all[col] = val

# 2. Include forecast
for y in forecast_years_int:
    col = str(y)
    if np_idx is not None:
        val = float(forecast_is.iloc[np_idx][col])
        dcf_profit_all[col] = val

# Save to session_state
st.session_state["dcf_profit_all"] = dcf_profit_all

st.subheader(
    f"📘 Forecasted Income Statement ({forecast_horizon} years, USD)"
)

st.dataframe(
    forecast_is.style.format(
        {c: "{:,.0f}".format for c in forecast_is.select_dtypes(include=[np.number]).columns},
        na_rep="",
    ),
    width='stretch',
)


# Extract EBITDA row for forecast years
if isinstance(ebitda_idx, int):
    ebitda_forecast_vals = np.array(
        [float(forecast_is.iat[ebitda_idx, forecast_is.columns.get_loc(str(y))])
         for y in forecast_years_int],
        dtype=float
    )
else:
    ebitda_forecast_vals = np.zeros(len(forecast_years_int))
# ---------------------------------------------------------
# SAVE ALL EBITDA VALUES (HISTORICAL + FORECAST)
# ---------------------------------------------------------

dcf_all_ebitda = {}

# 1️⃣ Save historical EBITDA
if isinstance(ebitda_idx, int):
    for y in year_cols_is:
        col_idx = forecast_is.columns.get_loc(str(y))
        val = forecast_is.iat[ebitda_idx, col_idx]
        dcf_all_ebitda[str(y)] = float(val) if pd.notna(val) else 0.0

# 2️⃣ Save forecast EBITDA
if isinstance(ebitda_idx, int):
    for y in forecast_years_int:
        col_idx = forecast_is.columns.get_loc(str(y))
        val = forecast_is.iat[ebitda_idx, col_idx]
        dcf_all_ebitda[str(y)] = float(val) if pd.notna(val) else 0.0

# 3️⃣ Store into session_state (BOTH KEYS)
st.session_state["dcf_ebitda_all"] = dcf_all_ebitda
st.session_state["dcf_ebitda_forecast"] = dcf_all_ebitda   # <-- backward compatibility

# Save EVERYTHING to session_state
st.session_state["dcf_ebitda_all"] = dcf_all_ebitda


# Depreciation from IS if present
dep_hist_from_is_idx, _ = find_single_row(forecast_is, ["depreciation"])
if dep_hist_from_is_idx is not None:
    dep_forecast_vals = np.array(
        [forecast_is.iloc[dep_hist_from_is_idx][str(y)] for y in forecast_years_int],
        dtype=float
    )
else:
    # fallback to CF-based ratio (rarely used now)
    if dep_cf_idx_list:
        common = [c for c in year_cols_cf if c in year_cols_is]
        dep_ratio = ratio_to_revenue(
            cf_df.loc[dep_cf_idx_list, common].sum(axis=0).values.astype(float),
            revenue_row[common].values.flatten().astype(float)
        )
    else:
        dep_ratio = 0.0
    dep_forecast_vals = np.array(
        [rev_forecast[y] * dep_ratio for y in forecast_years_int],
        dtype=float
    )
# After building rev_forecast dict
st.session_state["dcf_rev_forecast"] = {str(y): float(rev_forecast[y]) for y in forecast_years_int}

# ---------------------------------------------------------
# CAPITAL STRUCTURE FROM BS: Total Debt, Cash, CA, CL
# ---------------------------------------------------------
common_hist_bs = [c for c in year_cols_bs if c in year_cols_is]
bs_year_used_label = common_hist_bs[-1] if common_hist_bs else year_cols_bs[-1]

total_debt = 0.0
if debt_idx_list:
    total_debt = float(bs_df.loc[debt_idx_list, bs_year_used_label].sum(skipna=True))

cash_bal = 0.0
if cash_idx_list:
    cash_bal = float(bs_df.loc[cash_idx_list, bs_year_used_label].sum(skipna=True))

# equity: try some standard keywords
total_equity = 0.0
if equity_idx_list:
    total_equity = float(bs_df.loc[equity_idx_list, bs_year_used_label].sum(skipna=True))


net_debt = total_debt - cash_bal
de_ratio = (total_debt / total_equity) if total_equity != 0 else 0.0
c_cap5 = st.columns(1)[0]


# Save BS capital structure into session_state for other pages
st.session_state["total_debt"] = float(total_debt)
st.session_state["cash_balance"] = float(cash_bal)
st.session_state["net_debt"] = float(net_debt)
st.session_state["book_equity"] = float(total_equity)
st.session_state["de_ratio"] = float(de_ratio)

# ---------------------------------------------------------
# 🟦 WORKING CAPITAL MODULE (HISTORICAL → WC% → FORECAST → ΔWC)
# ---------------------------------------------------------
st.subheader("📘 Working Capital Calculation (Historical & Forecast)")

delta_wc_forecast_vals = np.zeros(len(forecast_years_int))

if ca_idx_list and cl_idx_list:

    # -------- 1️⃣ HISTORICAL WC (CA - CL)
    st.markdown("### **Historical Working Capital (CA - CL)**")

    ca_hist = bs_df.loc[ca_idx_list, year_cols_bs].sum(axis=0)
    cl_hist = bs_df.loc[cl_idx_list, year_cols_bs].sum(axis=0)
    wc_hist = ca_hist - cl_hist

    df_wc_hist = pd.DataFrame({
        "Year": year_cols_bs,
        "Current Assets": ca_hist.values,
        "Current Liabilities": cl_hist.values,
        "Working Capital (CA-CL)": wc_hist.values,
    })

    st.dataframe(
        df_wc_hist.style.format({
            "Current Assets": "{:,.0f}",
            "Current Liabilities": "{:,.0f}",
            "Working Capital (CA-CL)": "{:,.0f}",
        }),
        width='stretch'    )

    # 2️⃣ WC% OF SALES
    st.markdown("### **Historical Working Capital as % of Sales**")

    common_hist = [c for c in year_cols_is if c in wc_hist.index]

    wc_vals_hist = wc_hist[common_hist].astype(float).values
    rev_vals_hist = revenue_row[common_hist].values.flatten().astype(float)

    wc_percent_hist = wc_vals_hist / rev_vals_hist

    df_wc_pct = pd.DataFrame({
        "Year": common_hist,
        "Working Capital": wc_vals_hist,
        "Revenue": rev_vals_hist,
        "WC % of Sales": wc_percent_hist,
    })

    st.dataframe(
        df_wc_pct.style.format({
            "Working Capital": "{:,.0f}".format,
            "Revenue": "{:,.0f}".format,
            "WC % of Sales": "{:.2%}".format,
        }),
        width='stretch'    )

    # 3️⃣ WC% OF SALES — USER CHOICE (Average vs Most Recent) [PERSISTENT]
    wc_percent_array = wc_percent_hist.copy()
    mask_valid = (wc_percent_array > -5) & (wc_percent_array < 5)
    wc_percent_clean = wc_percent_array[mask_valid]

    # Compute BOTH candidates
    wc_percent_mean = float(np.mean(wc_percent_clean)) if len(wc_percent_clean) else 0.0

    last_year = common_hist[-1]
    last_wc = float(wc_hist[last_year])
    last_rev = float(revenue_row[last_year].values[0])
    wc_percent_last = (last_wc / last_rev) if last_rev != 0 else 0.0

    st.markdown("### ✅ Working Capital Assumption (WC % of Sales)")

    # ✅ INITIALIZE SESSION STATE ONCE
    if "dcf_wc_pct_method" not in st.session_state:
        st.session_state["dcf_wc_pct_method"] = "last"  # default = most recent

    wc_choice = st.radio(
        "Which WC% of Sales should be used for forecasting?",
        [
            f"Use average of historical WC% ({wc_percent_mean:.2%})",
            f"Use most recent WC% ({last_year}) = {wc_percent_last:.2%}"
        ],
        index=0 if st.session_state["dcf_wc_pct_method"] == "average" else 1,
        key="dcf_wc_pct_method_radio"
    )

    # ✅ UPDATE SESSION STATE EXPLICITLY
    if "average" in wc_choice.lower():
        st.session_state["dcf_wc_pct_method"] = "average"
        wc_percent_avg = wc_percent_mean
        st.success(f"✅ Using historical average WC% of Sales = {wc_percent_avg:.2%}")
    else:
        st.session_state["dcf_wc_pct_method"] = "last"
        wc_percent_avg = wc_percent_last
        st.info(f"📌 Using most recent WC% of Sales ({last_year}) = {wc_percent_avg:.2%}")

    # 4️⃣ FORECAST WC
    st.markdown("### **Forecast Working Capital**")

    wc_forecast_vals = np.array(
        [rev_forecast[y] * wc_percent_avg for y in forecast_years_int],
        dtype=float
    )

    df_wc_forecast = pd.DataFrame({
        "Year": forecast_years_int,
        "Forecast Revenue": [rev_forecast[y] for y in forecast_years_int],
        "Forecast WC": wc_forecast_vals,
    })

    st.dataframe(
        df_wc_forecast.style.format({
            "Forecast Revenue": "{:,.0f}",
            "Forecast WC": "{:,.0f}",
        }),
        width='stretch'    )

    # 5️⃣ ΔWC = OLD – NEW
    st.markdown("### **Change in Working Capital (ΔWC = Old – New)**")

    last_wc_hist_value = float(wc_hist[common_hist[-1]])

    prev_wc = last_wc_hist_value
    delta_list = []

    for wc_new in wc_forecast_vals:
        delta_list.append(prev_wc - wc_new)  # Old – New
        prev_wc = wc_new

    delta_wc_forecast_vals = np.array(delta_list, dtype=float)

    df_delta_wc = pd.DataFrame({
        "Year": forecast_years_int,
        "Forecast WC": wc_forecast_vals,
        "ΔWC (Old – New)": delta_wc_forecast_vals,
    })

    st.dataframe(
        df_delta_wc.style.format({
            "Forecast WC": "{:,.0f}",
            "ΔWC (Old – New)": "{:,.0f}",
        }),
        width='stretch'    )

else:
    st.warning("⚠️ Please select Current Assets and Current Liabilities rows first.")

# Capital structure summary
st.subheader("Capital Structure & Working Capital (from Balance Sheet)")
c_cap1, c_cap2, c_cap3, c_cap4 = st.columns(4)
with c_cap1:
    st.metric(f"Total Debt ({bs_year_used_label})", f"{total_debt:,.0f}")
with c_cap2:
    st.metric(f"Cash & Equivalents ({bs_year_used_label})", f"{cash_bal:,.0f}")
with c_cap3:
    st.metric("Net Debt", f"{net_debt:,.0f}")
with c_cap4:
    st.metric("D/E Ratio", f"{de_ratio:.2f}x")
with c_cap5:
    st.metric(f"Equity ({bs_year_used_label})", f"{total_equity:,.0f}")
# ---------------------------------------------------------
# CAPEX: use selected CF rows directly, do NOT require IS overlap
# ---------------------------------------------------------
avg_capex = 0.0

if capex_cf_idx_list:
    # Use ANY cashflow years that have numeric data
    capex_hist_vals = cf_df.loc[capex_cf_idx_list, year_cols_cf].sum(axis=0).values.astype(float)

    # Only use real non-zero values
    valid_capex = capex_hist_vals[~np.isnan(capex_hist_vals)]

    if len(valid_capex) > 0:
        avg_capex = float(np.mean(valid_capex))

# Forecast capex = average of historical (negative number preserved)
capex_forecast_vals = np.full(len(forecast_years_int), avg_capex, dtype=float)


# ---------------------------------------------------------
# COST OF DEBT (Interest / Debt)
# ---------------------------------------------------------
int_is_idx_list = find_row_indices(is_df, ["net finance costs", "finance costs", "interest expense", "interest paid"])
if int_is_idx_list:
    interest_last = float(is_df.loc[int_is_idx_list, last_hist_label].sum(skipna=True))
else:
    if int_cf_idx_list:
        interest_last = float(cf_df.loc[int_cf_idx_list, bs_year_used_label].sum(skipna=True))
    else:
        interest_last = 0.0

if total_debt != 0:
    cost_of_debt = abs(interest_last) / abs(total_debt)
else:
    cost_of_debt = 0.0

rd = cost_of_debt       # <-- ⭐⭐ VERY IMPORTANT ⭐⭐

# ---------------------------------------------------------
# DCF PARAMETERS — AUTO + OVERRIDE (WITH 2 OPTIONAL UPLOADS)
# ✅ FIXED: when you upload Country ERP + Default Spread,
#          RF + MRP textboxes IMMEDIATELY take those values
#          (RF = AvgCoD - Spread, MRP = ERP)
#          and the formulas use the textbox values.
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💰 DCF Parameters (Auto + Override)")

# =============== helpers ===============
def _to_decimal(x):
    """Accepts 0.15 or 15; returns decimal 0.15"""
    try:
        x = float(x)
    except Exception:
        return None
    return x / 100.0 if x > 1.5 else x

def _load_country_params_df(file_or_path) -> pd.DataFrame:
    """
    Excel required columns (flexible match):
      Country | ERP | Default Spread
    Returns normalized df with: Country, ERP, DefaultSpread
    """
    df = pd.read_excel(file_or_path)
    df.columns = [str(c).strip() for c in df.columns]

    col_country = [c for c in df.columns if c.lower() == "country"]
    col_erp = [c for c in df.columns if c.lower() in ["erp", "equity risk premium", "equity_risk_premium"]]
    col_spread = [c for c in df.columns if c.lower() in ["default spread", "default_spread", "spread"]]

    if not (col_country and col_erp and col_spread):
        raise ValueError("Excel must contain columns: Country, ERP, Default Spread")

    out = df[[col_country[0], col_erp[0], col_spread[0]]].copy()
    out.columns = ["Country", "ERP", "DefaultSpread"]
    out["Country"] = out["Country"].astype(str).str.strip()
    return out


def init_widget_key(widget_key: str, master_key: str, default_val: float):
    """
    IMPORTANT: only set widget key if it doesn't exist yet
    (prevents StreamlitAPIException).
    """
    if master_key not in st.session_state:
        st.session_state[master_key] = float(default_val)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = float(st.session_state[master_key])

# =============== layout ===============
left, right = st.columns([1.15, 1.0], vertical_alignment="top")

# =========================================================
# LEFT: Country ERP + Default Spread upload toggle
# =========================================================
with left:
    st.markdown("#### 🌍 Country ERP & Default Spread (Auto RF + MRP)")

    # init upload states
    st.session_state.setdefault("dcf_country_upload_enabled", False)
    st.session_state.setdefault("dcf_country_params_bytes", None)
    st.session_state.setdefault("dcf_country_params_name", None)

    # toggle
    country_upload = st.checkbox(
        "📤 Upload Country ERP + Default Spread Excel (optional)",
        value=st.session_state["dcf_country_upload_enabled"],
        key="dcf_country_upload_enabled_ui"
    )
    st.session_state["dcf_country_upload_enabled"] = country_upload

    if country_upload:
        st.caption("Required Excel columns: **Country, ERP, Default Spread**")
        up_country = st.file_uploader(
            "Upload Country params Excel",
            type=["xlsx"],
            key="dcf_country_params_uploader"
        )
        if up_country is not None:
            st.session_state["dcf_country_params_bytes"] = up_country.getvalue()
            st.session_state["dcf_country_params_name"] = up_country.name
    else:
        st.session_state["dcf_country_params_bytes"] = None
        st.session_state["dcf_country_params_name"] = None

    # load params df (uploaded takes precedence)
    df_params = None
    params_source = None

    try:
        if country_upload and st.session_state["dcf_country_params_bytes"] is not None:
            df_params = _load_country_params_df(io.BytesIO(st.session_state["dcf_country_params_bytes"]))
            params_source = f"Uploaded: {st.session_state.get('dcf_country_params_name','(file)')}"
        else:
            if DCF_PARAMS_PATH.exists():
                df_params = _load_country_params_df(DCF_PARAMS_PATH)
                params_source = f"Default file: {DCF_PARAMS_PATH.name}"
            else:
                st.warning(f"⚠️ Missing default file: {DCF_PARAMS_PATH}. Upload a file above.")
    except Exception as e:
        st.error(f"❌ Country params file error: {e}")
        df_params = None

    if params_source:
        st.caption(f"Source: **{params_source}**")

    # choose country and get ERP + spread
    auto_erp_dec = None
    auto_spread_dec = None

    if df_params is not None and not df_params.empty:
        country_list = sorted(df_params["Country"].dropna().astype(str).unique().tolist())
        default_country = "Zimbabwe" if "Zimbabwe" in country_list else (country_list[0] if country_list else None)

        if default_country is not None:
            chosen_country = st.selectbox(
                "Select country (auto ERP + Default Spread):",
                country_list,
                index=country_list.index(default_country),
                key="dcf_country_select"
            )

            row = df_params[df_params["Country"].astype(str) == str(chosen_country)]
            if not row.empty:
                auto_erp_dec = _to_decimal(row.iloc[0]["ERP"])               # ERP -> MRP
                auto_spread_dec = _to_decimal(row.iloc[0]["DefaultSpread"])  # used in RF formula

    # Zimbabwe Avg Cost of Debt (USD)
    st.session_state.setdefault("dcf_zim_avg_cost_debt_pct", 18.0)
    zim_avg_cod_pct = st.number_input(
        "Average cost of debt Zimbabwe (US$) (%)",
        value=float(st.session_state["dcf_zim_avg_cost_debt_pct"]),
        step=0.1,
        key="dcf_zim_avg_cost_debt_pct_input"
    )
    st.session_state["dcf_zim_avg_cost_debt_pct"] = zim_avg_cod_pct
    zim_avg_cod = zim_avg_cod_pct / 100.0

    # Derive Auto RF & Auto MRP (MRP=ERP)
    auto_mrp_pct = (auto_erp_dec * 100) if auto_erp_dec is not None else None
    auto_rf_pct = ((zim_avg_cod - auto_spread_dec) * 100) if (auto_spread_dec is not None) else None

    if auto_mrp_pct is not None and auto_rf_pct is not None:
        st.success(
            f"✅ Auto from Excel: MRP={auto_mrp_pct:.2f}% | "
            f"RF=(Avg CoD ZW USD − Spread)={auto_rf_pct:.2f}%"
        )
    else:
        st.info("ℹ️ Auto values not available yet (check Excel columns/values).")


# =========================================================
# RIGHT: Industry Betas upload toggle
# =========================================================
with right:
    st.markdown("#### 🧩 Industry Unlevered Betas (βu)")

    st.session_state.setdefault("dcf_beta_upload_enabled", False)
    st.session_state.setdefault("dcf_beta_file_bytes", None)
    st.session_state.setdefault("dcf_beta_file_name", None)

    beta_upload = st.checkbox(
        "📤 Upload Industry Betas Excel (optional)",
        value=st.session_state["dcf_beta_upload_enabled"],
        key="dcf_beta_upload_enabled_ui"
    )
    st.session_state["dcf_beta_upload_enabled"] = beta_upload

    if beta_upload:
        st.caption("Required Excel columns: **Industry Name, Unlevered beta**")
        up_beta = st.file_uploader(
            "Upload Industry betas Excel",
            type=["xlsx"],
            key="dcf_beta_uploader"
        )
        if up_beta is not None:
            st.session_state["dcf_beta_file_bytes"] = up_beta.getvalue()
            st.session_state["dcf_beta_file_name"] = up_beta.name
    else:
        st.session_state["dcf_beta_file_bytes"] = None
        st.session_state["dcf_beta_file_name"] = None


# =========================================================
# INIT defaults once (master keys + states)
# =========================================================
if "dcf_init" not in st.session_state:
    st.session_state["dcf_rf_pct"] = float(auto_rf_pct) if auto_rf_pct is not None else 11.61
    st.session_state["dcf_mrp_pct"] = float(auto_mrp_pct) if auto_mrp_pct is not None else 13.82
    st.session_state["dcf_tax_pct"] = 25.0
    st.session_state["dcf_unlevered_beta"] = 1.00
    st.session_state["dcf_terminal_g_pct"] = 5.0

    st.session_state["dcf_use_auto_params"] = True

    # beta states
    st.session_state["dcf_industries_selected"] = []
    st.session_state["dcf_beta_blend_method"] = "Simple average"
    st.session_state["dcf_industry_weights"] = {}
    st.session_state["dcf_beta_manual_mode"] = False
    st.session_state["dcf_beta_manual_value"] = None
    st.session_state["dcf_beta_auto_last"] = None

    st.session_state["dcf_init"] = True

# =========================================================
# Auto vs Override toggle (RF & MRP)  ✅ HARD SYNC FIX
#    When auto is ON and we have auto values:
#      - force RF textbox to auto_rf_pct
#      - force MRP textbox to auto_mrp_pct
# =========================================================
use_auto = st.checkbox(
    "Use Auto (from Excel) for RF & MRP",
    value=bool(st.session_state.get("dcf_use_auto_params", True)),
    key="dcf_use_auto_params_ui"
)
st.session_state["dcf_use_auto_params"] = use_auto

# Build signature of the auto sources
auto_signature = (
    float(auto_rf_pct) if auto_rf_pct is not None else None,
    float(auto_mrp_pct) if auto_mrp_pct is not None else None,
    st.session_state.get("dcf_country_select", None),
    float(st.session_state.get("dcf_zim_avg_cost_debt_pct", 0.0)),
)

# Track previous signature so we only "snap" inputs when auto data changes
st.session_state.setdefault("dcf_auto_signature", None)

should_snap_to_auto = (
    use_auto
    and (auto_rf_pct is not None)
    and (auto_mrp_pct is not None)
    and (auto_signature != st.session_state["dcf_auto_signature"])
)

# If auto is ON and signature changed -> update BOTH master keys AND widget keys
# so the textboxes visually change immediately.
if should_snap_to_auto:
    # master
    st.session_state["dcf_rf_pct"] = float(auto_rf_pct)
    st.session_state["dcf_mrp_pct"] = float(auto_mrp_pct)

    # widget keys (these control the textbox displayed values)
    st.session_state["dcf_rf_pct_input"] = float(auto_rf_pct)
    st.session_state["dcf_mrp_pct_input"] = float(auto_mrp_pct)

    st.session_state["dcf_auto_signature"] = auto_signature
elif not use_auto:
    # If auto OFF, don't overwrite manual values; just reset signature so next ON snaps again
    st.session_state["dcf_auto_signature"] = None

# =========================================================
# Backfill widget keys (safe)
# =========================================================
init_widget_key("dcf_rf_pct_input", "dcf_rf_pct", 11.61)
init_widget_key("dcf_mrp_pct_input", "dcf_mrp_pct", 13.82)
init_widget_key("dcf_tax_pct_input", "dcf_tax_pct", 25.0)
init_widget_key("dcf_unlevered_beta_input", "dcf_unlevered_beta", 1.0)
init_widget_key("dcf_terminal_g_pct_input", "dcf_terminal_g_pct", 5.0)

# =========================================================
# Main input widgets
# =========================================================
col1, col2 = st.columns(2)

with col1:
    rf_input = st.number_input("Risk-free rate (%)", step=0.1, key="dcf_rf_pct_input")
    mrp_input = st.number_input("Market risk premium (%)", step=0.1, key="dcf_mrp_pct_input")
    tax_input = st.number_input("Tax rate (%)", step=0.5, key="dcf_tax_pct_input")

with col2:
    # Load betas df: uploaded takes precedence
    betas_df = None
    beta_source = None
    try:
        if st.session_state.get("dcf_beta_upload_enabled") and st.session_state.get("dcf_beta_file_bytes") is not None:
            betas_df = _load_unlevered_betas_any(io.BytesIO(st.session_state["dcf_beta_file_bytes"]))
            beta_source = f"Uploaded: {st.session_state.get('dcf_beta_file_name','(file)')}"
        else:
            if UNLEVERED_BETAS_PATH.exists():
                mtime = UNLEVERED_BETAS_PATH.stat().st_mtime  # ✅ changes when you save Excel
                betas_df = _load_unlevered_betas_any(UNLEVERED_BETAS_PATH, file_mtime=mtime)
                beta_source = f"Default file: {UNLEVERED_BETAS_PATH.name}"

            else:
                st.warning(f"⚠️ Missing default file: {UNLEVERED_BETAS_PATH}. Upload a file above.")
    except Exception as e:
        st.warning(f"⚠️ Could not load industry betas: {e}")
        betas_df = None

    if beta_source:
        st.caption(f"Source: **{beta_source}**")

    # Multi-industry selector for blended beta
    if betas_df is not None and not betas_df.empty:
        industry_list = betas_df["Industry"].tolist()

        selected = st.multiselect(
            "Select Industry / Industries (for blended βu):",
            industry_list,
            default=[i for i in st.session_state.get("dcf_industries_selected", []) if i in industry_list],
            key="dcf_industries_multiselect"
        )
        st.session_state["dcf_industries_selected"] = selected

        blend_method = st.radio(
            "How should industries be combined?",
            ["Simple average", "Weighted average"],
            index=0 if st.session_state.get("dcf_beta_blend_method", "Simple average") == "Simple average" else 1,
            key="dcf_beta_blend_method_radio",
            horizontal=True
        )
        st.session_state["dcf_beta_blend_method"] = blend_method

        beta_u_auto = None
        if selected:
            sub = betas_df[betas_df["Industry"].isin(selected)].copy()

            if blend_method == "Simple average":
                beta_u_auto = float(sub["UnleveredBeta"].mean())
            else:
                st.markdown("#### Enter weights (they will be normalized to 100%)")
                weights = []
                for ind in selected:
                    default_w = float(st.session_state.get("dcf_industry_weights", {}).get(ind, 1.0))
                    w = st.number_input(
                        f"Weight for {ind}",
                        min_value=0.0,
                        value=default_w,
                        step=1.0,
                        key=f"w_{ind}"
                    )
                    st.session_state.setdefault("dcf_industry_weights", {})
                    st.session_state["dcf_industry_weights"][ind] = w
                    weights.append(w)

                total_w = float(sum(weights))
                if total_w <= 0:
                    st.error("❌ Total weight must be > 0.")
                else:
                    sub = sub.sort_values("Industry").reset_index(drop=True)
                    w_norm = np.array([st.session_state["dcf_industry_weights"][ind] for ind in sub["Industry"]]) / total_w
                    beta_u_auto = float(np.sum(sub["UnleveredBeta"].values * w_norm))

                    dfw = pd.DataFrame({
                        "Industry": sub["Industry"].values,
                        "UnleveredBeta": sub["UnleveredBeta"].values,
                        "Weight (raw)": [st.session_state["dcf_industry_weights"][i] for i in sub["Industry"]],
                        "Weight (norm %)": (w_norm * 100).round(2)
                    })
                    st.dataframe(dfw, width="stretch", hide_index=True)

            if beta_u_auto is not None and np.isfinite(beta_u_auto):
                st.session_state["dcf_beta_auto_last"] = float(beta_u_auto)
                st.caption(f"Blended industry βu (auto): **{beta_u_auto:.2f}**")
        else:
            st.info("Select at least 1 industry to auto-fill βu.")

    # Auto vs manual beta mode
    st.session_state.setdefault("dcf_beta_manual_mode", False)
    beta_mode = st.radio(
        "Unlevered beta mode:",
        ["Use Auto (from industries)", "Manual override (type my own βu)"],
        index=1 if st.session_state["dcf_beta_manual_mode"] else 0,
        key="dcf_beta_mode_radio",
        horizontal=True
    )
    st.session_state["dcf_beta_manual_mode"] = beta_mode.startswith("Manual")

    # Apply auto beta button
    if not st.session_state["dcf_beta_manual_mode"]:
        auto_beta = st.session_state.get("dcf_beta_auto_last")
        if auto_beta is not None and np.isfinite(auto_beta):
            st.caption(f"Auto βu available: {auto_beta:.2f}")
            if st.button("✅ Apply Auto βu to input", key="apply_auto_beta_btn"):
                st.session_state["dcf_unlevered_beta_input"] = float(auto_beta)

    beta_u_input = st.number_input("Unlevered beta (asset beta)", step=0.05, key="dcf_unlevered_beta_input")
    if st.session_state.get("dcf_beta_manual_mode", False):
        st.session_state["dcf_beta_manual_value"] = float(beta_u_input)

    g_input = st.number_input("Terminal growth rate (%)", step=0.1, key="dcf_terminal_g_pct_input")

# =========================================================
# Save user inputs to master keys (USED BY FORMULAS)
# =========================================================
st.session_state["dcf_rf_pct"] = float(rf_input)
st.session_state["dcf_mrp_pct"] = float(mrp_input)
st.session_state["dcf_tax_pct"] = float(tax_input)
st.session_state["dcf_unlevered_beta"] = float(beta_u_input)
st.session_state["dcf_terminal_g_pct"] = float(g_input)

# Decimals
rf = st.session_state["dcf_rf_pct"] / 100
mrp = st.session_state["dcf_mrp_pct"] / 100
tax = st.session_state["dcf_tax_pct"] / 100
g = st.session_state["dcf_terminal_g_pct"] / 100

# CAPM & WACC
beta_levered = st.session_state["dcf_unlevered_beta"] * (1 + (1 - tax) * de_ratio)

if de_ratio <= 0:
    w_e, w_d = 1, 0
else:
    w_d = de_ratio / (1 + de_ratio)
    w_e = 1 / (1 + de_ratio)

re = rf + beta_levered * mrp
wacc = w_e * re + w_d * rd * (1 - tax)

# Save computed
st.session_state["levered_beta"] = float(beta_levered)
st.session_state["wacc"] = float(wacc)

# =========================================================
# OUTPUT HEADER (STOP HERE)
# =========================================================
st.markdown('<div class="dcf-card">', unsafe_allow_html=True)
st.markdown("### 📌 DCF Output")


k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="dcf-kpi">
      <div class="dcf-kpi-title">Cost of Debt (Rd)</div>
      <div class="dcf-kpi-value">{rd*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="dcf-kpi">
      <div class="dcf-kpi-title">Levered Beta</div>
      <div class="dcf-kpi-value">{beta_levered:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="dcf-kpi">
      <div class="dcf-kpi-title">Cost of Equity (Re)</div>
      <div class="dcf-kpi-value">{re*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="dcf-kpi">
      <div class="dcf-kpi-title">WACC</div>
      <div class="dcf-kpi-value">{wacc*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f'<div class="small-note">Terminal growth (g): {g*100:.2f}% • D/E: {de_ratio:.2f}x</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# DATE-BASED DISCOUNTING (FULLY PERSISTENT — NO RESETTING)
# ---------------------------------------------------------
st.markdown("### 📅 Valuation Timing & Mid-point")
# 1️⃣ INITIALIZE DEFAULTS (only ONCE)
if "dcf_timing_init" not in st.session_state:

    st.session_state["dcf_valuation_date"] = date.today()
    st.session_state["dcf_first_fs_date"] = date(last_hist_year + 1, 12, 31)
    st.session_state["dcf_use_midyear"] = False

    st.session_state["dcf_timing_init"] = True
# 2️⃣ WIDGETS (using separate keys so they do NOT overwrite session_state)
valuation_date_input = st.date_input(
    "Valuation date (today / deal date)",
    value=st.session_state["dcf_valuation_date"],
    key="dcf_valuation_date_input"
)

first_fs_date_input = st.date_input(
    "Financial statement year-end date for forecasts (first forecast year)",
    value=st.session_state["dcf_first_fs_date"],
    key="dcf_first_fs_date_input"
)

use_midyear_input = st.checkbox(
    "Use mid-year (0.5 year earlier) convention?",
    value=st.session_state["dcf_use_midyear"],
    key="dcf_use_midyear_input"
)


# 3️⃣ UPDATE session_state values explicitly
st.session_state["dcf_valuation_date"] = valuation_date_input
st.session_state["dcf_first_fs_date"] = first_fs_date_input
st.session_state["dcf_use_midyear"] = use_midyear_input


# 4️⃣ CALCULATE DISCOUNT PERIODS USING STORED VALUES
valuation_date = st.session_state["dcf_valuation_date"]
first_forecast_fs_date = st.session_state["dcf_first_fs_date"]
use_midyear = st.session_state["dcf_use_midyear"]

gap_days = (first_forecast_fs_date - valuation_date).days
gap_years = gap_days / 365.25

n0 = max(gap_years, 0.0)
if use_midyear:
    n0 = max(n0 - 0.5, 0.0)

# discount periods for each forecast year
discount_periods_n = np.array([n0 + i for i in range(len(forecast_years_int))], dtype=float)

# DF0
midpoint_df0 = (1 / (1 + wacc) ** n0) if wacc > 0 else 1.0


# 5️⃣ DISPLAY SUMMARY TABLE
midpoint_table = pd.DataFrame(
    {
        "Valuation date": [valuation_date],
        "FS date (first forecast year)": [first_forecast_fs_date],
        "Gap (days)": [gap_days],
        "Discount period n₀ (years)": [n0],
        "Mid-point DF₀ = 1/(1+WACC)ⁿ⁰": [midpoint_df0],
    }
)

st.dataframe(midpoint_table, width='stretch')

# ---------------------------------------------------------
# FCFF / UFCF
# ---------------------------------------------------------
ebitda_after_tax = ebitda_forecast_vals * (1 - tax)
dep_tax_vals = -dep_forecast_vals * tax

# UFCF = EBITDA(1-T) + Dep×T + ΔWC + Capex
fcff_vals = ebitda_after_tax + dep_tax_vals + delta_wc_forecast_vals + capex_forecast_vals

# Discount factors using date-based n
discount_factors = np.array([(1 / (1 + wacc) ** n) for n in discount_periods_n])
pv_fcff = fcff_vals * discount_factors

st.session_state["dcf_fcff_array"] = fcff_vals.tolist()
st.session_state["dcf_pv_fcff_array"] = pv_fcff.tolist()
st.session_state["dcf_discount_periods_n"] = discount_periods_n.tolist()

# ---------------------------------------------------------
# TERMINAL VALUE
# ---------------------------------------------------------
if wacc <= g:
    terminal_value = np.nan
    pv_terminal = np.nan
else:
    terminal_value = fcff_vals[-1] * (1 + g) / (wacc - g)
    discount_factor_terminal = float(discount_factors[-1])
    pv_terminal = terminal_value * discount_factor_terminal

enterprise_value = np.nansum(pv_fcff) + (0 if np.isnan(pv_terminal) else pv_terminal)
equity_value = enterprise_value - net_debt
st.session_state["dcf_terminal_value"] = float(terminal_value) if not np.isnan(terminal_value) else None
st.session_state["dcf_pv_terminal"] = float(pv_terminal) if not np.isnan(pv_terminal) else None
st.session_state["dcf_pv_fcff_sum"] = float(np.nansum(pv_fcff))

# Save DCF valuation outputs into session_state
st.session_state["enterprise_value_dcf"] = float(enterprise_value)
st.session_state["equity_value"] = float(equity_value)          # generic key used by COMPARABLES
st.session_state["equity_value_dcf"] = float(equity_value)      # explicit DCF key

# ---------------------------------------------------------
# DCF TABLE (UFCF style)
# ---------------------------------------------------------
st.subheader("📉 DCF Cashflows (UFCF) — Date-based Discounting")

df_dcf = pd.DataFrame(
    {
        "Year": [str(y) for y in forecast_years_int],
        "Discount period n (years)": discount_periods_n,
        "EBITDA × (1−T)": ebitda_after_tax,
        "Depreciation × Tax": dep_tax_vals,
        "Δ Working capital": delta_wc_forecast_vals,
        "Capex": capex_forecast_vals,
        "UFCF": fcff_vals,
        "Discount factor": discount_factors,
        "PV of UFCF": pv_fcff,
    }
)

num_cols_dcf = df_dcf.select_dtypes(include=[np.number]).columns
fmt_dict = {c: "{:,.0f}".format for c in num_cols_dcf if c not in ["Discount period n (years)", "Discount factor"]}
fmt_dict["Discount period n (years)"] = "{:.3f}".format
fmt_dict["Discount factor"] = "{:.3f}".format

styled_dcf = df_dcf.style.format(fmt_dict, na_rep="")
st.dataframe(styled_dcf, width='stretch')

# Terminal summary
st.write("**Terminal Value and Present Value:**")

df_term = pd.DataFrame(
    {
        "Terminal Value": [terminal_value],
        "Discount factor (last year)": [discount_factors[-1]],
        "PV of Terminal Value": [pv_terminal],
    }
)

fmt_term = {}
for c in df_term.columns:
    if c == "Discount factor (last year)":
        fmt_term[c] = "{:.3f}".format
    else:
        fmt_term[c] = "{:,.0f}".format

st.dataframe(
    df_term.style.format(fmt_term, na_rep=""),
    width='stretch',
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Valuation Summary")

c_sum1, c_sum2, c_sum3 = st.columns(3)
with c_sum1:
    st.metric("Enterprise Value (EV)", f"{enterprise_value:,.0f}")
    st.metric("Net Debt", f"{net_debt:,.0f}")
with c_sum2:
    st.metric("Equity Value", f"{equity_value:,.0f}")
    st.metric("WACC", f"{wacc*100:.2f}%")
    st.metric("Terminal Growth Rate", f"{g*100:.2f}%")
