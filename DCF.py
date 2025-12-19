import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Forecast + DCF (IS + BS + CF)",
    layout="wide"
)

st.title("📊 Forecast + DCF Valuation")

# ---------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
st.subheader("📂 Upload Financial Statements")

if "dcf_uploaded_file" not in st.session_state:
    st.session_state["dcf_uploaded_file"] = None

uploaded_file = st.file_uploader(
    "Upload Excel with IS, BS, CF",
    type=["xlsx"],
    key="dcf_main_uploader"
)

# Save once
if uploaded_file is not None:
    st.session_state["dcf_uploaded_file"] = uploaded_file

# Use stored file
uploaded_file = st.session_state["dcf_uploaded_file"]

if uploaded_file is None:
    st.info("⬆️ Please upload an Excel file to begin.")
    st.stop()

# ---------------------------------------------------------
# LOAD & CACHE PARSED STATEMENTS (ONCE)
# ---------------------------------------------------------
if "dcf_is_df" not in st.session_state:
    xls = pd.ExcelFile(uploaded_file)

    st.session_state["dcf_is_df"] = clean_numeric_cols(
        xls.parse(xls.sheet_names[0])
    )
    st.session_state["dcf_bs_df"] = clean_numeric_cols(
        xls.parse(xls.sheet_names[1])
    )
    st.session_state["dcf_cf_df"] = clean_numeric_cols(
        xls.parse(xls.sheet_names[2])
    )

is_df = st.session_state["dcf_is_df"]
bs_df = st.session_state["dcf_bs_df"]
cf_df = st.session_state["dcf_cf_df"]


year_cols_is = get_year_cols(is_df)
year_cols_bs = get_year_cols(bs_df)
year_cols_cf = get_year_cols(cf_df)
# ---------------------------------------------------------
# FX SECTION — EXCEL-BASED (ZWG → USD) [FIXED VERSION]
# ---------------------------------------------------------
# FX SECTION — EXCEL-BASED (ZWG → USD) — FINAL & CORRECT
# ---------------------------------------------------------
st.markdown("### 💱 Currency & Exchange Rates")

# -------------------------------------------------
# 1️⃣ Currency selector (persistent)
# -------------------------------------------------
if "dcf_currency" not in st.session_state:
    st.session_state["dcf_currency"] = "USD (already converted)"

currency = st.selectbox(
    "Currency of uploaded statements:",
    ["USD (already converted)", "ZWG (convert using FX Excel)"],
    index=0 if st.session_state["dcf_currency"].startswith("USD") else 1,
    key="dcf_currency_select"
)

st.session_state["dcf_currency"] = currency

# -------------------------------------------------
# 2️⃣ FX Excel upload (persistent)
# -------------------------------------------------
if "dcf_fx_file" not in st.session_state:
    st.session_state["dcf_fx_file"] = None

fx_file = st.file_uploader(
    "Upload FX Excel (Date + FX columns)",
    type=["xlsx"],
    key="dcf_fx_uploader"
)

if fx_file is not None:
    st.session_state["dcf_fx_file"] = fx_file

fx_file = st.session_state["dcf_fx_file"]

# -------------------------------------------------
# 3️⃣ If USD → skip FX
# -------------------------------------------------
if currency.startswith("USD"):
    st.success("✅ Data assumed to be in USD. No FX conversion applied.")

else:
    st.warning("ZWG detected. Upload FX Excel to convert to USD.")

    if fx_file is None:
        st.stop()

    # -------------------------------------------------
    # 4️⃣ Load FX Excel ONCE
    # -------------------------------------------------
    if "dcf_fx_raw" not in st.session_state:
        fx_raw = pd.read_excel(fx_file)
        fx_raw.columns = [str(c).strip() for c in fx_raw.columns]
        st.session_state["dcf_fx_raw"] = fx_raw
    else:
        fx_raw = st.session_state["dcf_fx_raw"]

    st.subheader("Raw FX data (preview)")
    st.dataframe(fx_raw.head(), use_container_width=True)

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

    st.subheader("📊 Yearly FX averages (Income Statement)")
    st.dataframe(
        pd.DataFrame({
            "Year": yearly_fx.keys(),
            "FX Rate": yearly_fx.values()
        }),
        use_container_width=True
    )

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
    # 9️⃣ Determine CLOSING FX RATE (latest date)
    # -------------------------------------------------
    fx_df_sorted = fx_df.sort_values("Date")

    closing_fx_rate = float(fx_df_sorted.iloc[-1][fx_col])
    closing_fx_date = fx_df_sorted.iloc[-1]["Date"].date()

    st.session_state["dcf_closing_fx_rate"] = closing_fx_rate

    st.info(
        f"📌 Balance Sheet closing FX rate: "
        f"{closing_fx_rate:,.4f} (as at {closing_fx_date})"
    )

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
    # 1️⃣1️⃣ APPLY FX CONVERSION (ONCE)
    # -------------------------------------------------
    if not st.session_state.get("dcf_fx_applied", False):

        # Income Statement → YEARLY AVERAGE
        is_df = convert_df_yearwise(is_df, yearly_fx)

        # Balance Sheet → SINGLE CLOSING FX (OPTIONAL)
        if st.session_state["dcf_apply_fx_bs"]:
            rate = st.session_state["dcf_closing_fx_rate"]
            if rate != 0:
                bs_df = bs_df.copy()
                for col in bs_df.columns:
                    if col != "Item":
                        bs_df[col] = bs_df[col] / rate

        # Save converted data
        st.session_state["dcf_is_df"] = is_df
        st.session_state["dcf_bs_df"] = bs_df
        st.session_state["dcf_cf_df"] = cf_df

        st.session_state["dcf_fx_applied"] = True

    st.success("✅ FX conversion applied correctly (IS = average, BS = closing)")


# ---------------------------------------------------------
# SHOW CLEANED STATEMENTS
# ---------------------------------------------------------
st.subheader("Income Statement (cleaned, in USD)")
st.dataframe(is_df, use_container_width=True)

st.subheader("Balance Sheet (cleaned, in USD)")
st.dataframe(bs_df, use_container_width=True)

st.subheader("Cash Flow Statement (cleaned, in USD)")
st.dataframe(cf_df, use_container_width=True)

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
# BALANCE SHEET MAPPING (PERSISTENT)
# ---------------------------------------------------------
st.markdown("### 🟩 Balance Sheet Mapping (multi-select allowed)")

bs_items = list(bs_df["Item"].astype(str))
bs_labels = option_labels_from_items(bs_items)

# --- DEBT ---
sel_debt_labels = st.multiselect(
    "Select ALL rows that form Total Debt / Borrowings:",
    bs_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["debt"], bs_labels),
    key="dcf_map_debt"
)

st.session_state["dcf_mapping"]["debt"] = sel_debt_labels

# --- CASH ---
sel_cash_labels = st.multiselect(
    "Select ALL rows that form Cash & Cash Equivalents:",
    bs_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["cash"], bs_labels),
    key="dcf_map_cash"
)
st.session_state["dcf_mapping"]["cash"] = sel_cash_labels

# --- CURRENT ASSETS ---
sel_ca_labels = st.multiselect(
    "Select ALL rows that are Current Assets (for Working Capital):",
    bs_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["ca"], bs_labels),
    key="dcf_map_ca"
)
st.session_state["dcf_mapping"]["ca"] = sel_ca_labels

# --- CURRENT LIABILITIES ---

sel_cl_labels = st.multiselect(
    "Select ALL rows that are Current Liabilities (for Working Capital):",
    bs_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["cl"], bs_labels),
    key="dcf_map_cl"
)
st.session_state["dcf_mapping"]["cl"] = sel_cl_labels

# --- EQUITY SELECTOR ---


valid_equity_defaults = clean_defaults(
    st.session_state["dcf_mapping"].get("equity", []),
    bs_labels
)

sel_equity_labels = st.multiselect(
    "Select ALL rows that represent Equity:",
    bs_labels,
    default=valid_equity_defaults,
    key="dcf_map_equity"
)

st.session_state["dcf_mapping"]["equity"] = sel_equity_labels
equity_idx_list = indices_from_labels(sel_equity_labels)

# Convert labels → row indices
debt_idx_list = indices_from_labels(sel_debt_labels)
cash_idx_list = indices_from_labels(sel_cash_labels)
ca_idx_list = indices_from_labels(sel_ca_labels)
cl_idx_list = indices_from_labels(sel_cl_labels)

# ---------------------------------------------------------
# CASH FLOW MAPPING (PERSISTENT)
# ---------------------------------------------------------
st.markdown("### 📄 Cash Flow Mapping (multi-select allowed)")

cf_items = list(cf_df["Item"].astype(str))
cf_labels = option_labels_from_items(cf_items)

# --- DEPRECIATION ---
sel_dep_cf = st.multiselect(
    "Select Depreciation & Amortisation rows (from Cash Flow):",
    cf_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["dep"], cf_labels),
    key="dcf_map_dep"
)

st.session_state["dcf_mapping"]["dep"] = sel_dep_cf

# --- CAPEX ---
sel_capex_cf = st.multiselect(
    "Select ALL Capex rows (purchase of PPE / fixed assets):",
    cf_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["capex"], cf_labels),
    key="dcf_map_capex"
)
st.session_state["dcf_mapping"]["capex"] = sel_capex_cf

# --- INTEREST ---
sel_int_cf = st.multiselect(
    "Select Interest paid rows (if using CF for interest):",
    cf_labels,
    default=clean_defaults(st.session_state["dcf_mapping"]["interest"], cf_labels),
    key="dcf_map_interest"
)
st.session_state["dcf_mapping"]["interest"] = sel_int_cf

# Convert labels → row indices
dep_cf_idx_list = indices_from_labels(sel_dep_cf)
capex_cf_idx_list = indices_from_labels(sel_capex_cf)
int_cf_idx_list = indices_from_labels(sel_int_cf)

# ---------------------------------------------------------
# INCOME STATEMENT FORECASTING
# ---------------------------------------------------------
# find main rows
# ---------------------------------------------------------
# CORE INCOME STATEMENT MAPPING (USER-DRIVEN)
# ---------------------------------------------------------
def map_core_is_totals(is_df, key_prefix="is_core"):
    """
    Persistent Income Statement core totals mapping.
    NEVER resets unless user explicitly changes selections.
    """

    st.markdown("### 🧾 Income Statement — Core Totals Mapping")

    items = list(is_df["Item"].astype(str))
    labels = option_labels_from_items(items)

    # Initialize once
    if "is_core_mapping" not in st.session_state:
        st.session_state["is_core_mapping"] = {
            "rev": None,
            "cos": None,
            "gp": None,
            "ebitda": None,
            "op": None,
            "pbt": None,
            "np": None,
        }

    def pick(label, key):
        stored = st.session_state["is_core_mapping"].get(key)

        # Safety: reset if item disappeared
        if stored not in labels:
            stored = None
            st.session_state["is_core_mapping"][key] = None

        selected = st.selectbox(
            label,
            [""] + labels,
            index=0 if stored is None else (labels.index(stored) + 1),
            key=f"{key_prefix}_{key}",
        )

        if selected != "":
            st.session_state["is_core_mapping"][key] = selected

        return selected

    # User selections
    pick("Revenue", "rev")
    pick("Cost of Sales", "cos")
    pick("Gross Profit", "gp")
    pick("EBITDA", "ebitda")
    pick("Operating Profit / EBIT", "op")
    pick("Profit Before Tax", "pbt")
    pick("Profit for the Year", "np")

    # Convert labels → indices
    idx = {}
    for k, v in st.session_state["is_core_mapping"].items():
        if v:
            idx[k] = int(v.split(":", 1)[0]) - 1
        else:
            idx[k] = None

    # Validation
    if idx["rev"] is None:
        st.error("❌ Revenue must be selected.")
        st.stop()

    return idx

core_idx = map_core_is_totals(is_df)

rev_idx    = core_idx["rev"]
cos_idx    = core_idx["cos"]
gp_idx     = core_idx["gp"]
ebitda_idx = core_idx["ebitda"]
op_idx     = core_idx["op"]
pbt_idx    = core_idx["pbt"]
np_idx     = core_idx["np"]


if rev_idx is None or cos_idx is None:
    st.error("❌ Could not find both 'Revenue' and 'Cost of sales' rows.")
    st.stop()

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

# revenue forecast
rev_hist_vals = revenue_row[year_cols_is].values.flatten().astype(float)

rev_forecast = {}
current_rev = rev_hist_vals[-1]

for y in forecast_years_int:
    current_rev = current_rev * (1 + avg_g)
    rev_forecast[y] = current_rev
    forecast_is.iat[
        rev_idx,
        forecast_is.columns.get_loc(str(y))
    ] = current_rev


# ---------------------------------------------------------
# COST HANDLING LOGIC
# ---------------------------------------------------------
use_gp_method = (cos_idx is not None and gp_idx is not None)

if use_gp_method:
    gp_hist_vals = forecast_is.iloc[gp_idx][year_cols_is].values.astype(float)
    mask = rev_hist_vals != 0
    gp_margins = gp_hist_vals[mask] / rev_hist_vals[mask]
    gp_margins = gp_margins[(gp_margins > -5) & (gp_margins < 5)]
    avg_gp_margin = np.mean(gp_margins) if len(gp_margins) else 0.3

    last_cos_hist = float(forecast_is.iloc[cos_idx][last_hist_label])
    cos_sign = -1 if last_cos_hist < 0 else 1

    for y in forecast_years_int:
        forecast_is.iat[
            cos_idx,
            forecast_is.columns.get_loc(str(y))
        ] = cos_sign * rev_forecast[y] * (1 - avg_gp_margin)


else:
    # === MANUFACTURING / FLEXIBLE MODE ===
    st.info(
        "Cost of sales not explicitly modelled. "
        "All cost items will be forecasted as % of revenue."
    )

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
for idx in range(len(forecast_is)):

    # 🔒 PROTECT COST OF SALES WHEN USING GP MARGIN
    if use_gp_method and idx == cos_idx:
        continue

    protected = [rev_idx, gp_idx, ebitda_idx, op_idx, pbt_idx, np_idx]
    if not treat_cos_as_normal and cos_idx is not None:
        protected.append(cos_idx)

    if idx in protected:
        continue

    # ✅ add here
    if use_gp_method and cos_idx is not None and idx == cos_idx:
        continue

    item = str(forecast_is.at[idx, "Item"]).lower()
    if any(k in item for k in total_keywords):
        continue

    row_hist = forecast_is.iloc[idx][year_cols_is].values.astype(float)
    ratio = ratio_to_revenue(row_hist, rev_hist_vals)

    for y in forecast_years_int:
        forecast_is.iat[
            idx,
            forecast_is.columns.get_loc(str(y))
        ] = rev_forecast[y] * ratio




def sum_rows(df, start_idx, end_idx, col):
    """Sum from start_idx to end_idx-1 inclusive."""
    if start_idx is None or end_idx is None:
        return df.iloc[start_idx][col] if start_idx is not None else np.nan
    if end_idx <= start_idx:
        return df.iloc[start_idx][col]
    return df.loc[start_idx:end_idx - 1, col].sum(skipna=True)

for col in forecast_cols:
    if gp_idx is not None:
        forecast_is.iat[gp_idx, forecast_is.columns.get_loc(col)] = (
            forecast_is.iloc[rev_idx][col]
            + forecast_is.iloc[cos_idx][col]
        )
    # EBITDA: recompute ONLY if it is NOT explicitly forecasted as a line item
    if (
            ebitda_idx is not None
            and gp_idx is not None
            and ebitda_idx > gp_idx
    ):
        # Check if EBITDA row already has meaningful values
        # EBITDA: recompute PER YEAR if not explicitly forecasted
        if (
                ebitda_idx is not None
                and gp_idx is not None
                and ebitda_idx > gp_idx
        ):
            col_idx = forecast_is.columns.get_loc(col)

            existing_val = pd.to_numeric(
                forecast_is.iat[ebitda_idx, col_idx],
                errors="coerce"
            )

            if pd.isna(existing_val) or existing_val == 0:
                ebitda_val = forecast_is.loc[
                    gp_idx:ebitda_idx - 1,
                    col
                ].sum(skipna=True)

                forecast_is.iat[ebitda_idx, col_idx] = ebitda_val

    if op_idx is not None and ebitda_idx is not None:
        forecast_is.iat[op_idx, forecast_is.columns.get_loc(col)] = \
            forecast_is.loc[ebitda_idx:op_idx-1, col].sum()

    if pbt_idx is not None and op_idx is not None:
        forecast_is.iat[pbt_idx, forecast_is.columns.get_loc(col)] = \
            forecast_is.loc[op_idx:pbt_idx-1, col].sum()

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
        val = float(forecast_is.iloc[np_idx][col])
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
    use_container_width=True,
)


# Extract EBITDA row for forecast years
ebitda_forecast_vals = np.array(
    [forecast_is.iloc[ebitda_idx][str(y)] for y in forecast_years_int],
    dtype=float
) if ebitda_idx is not None else np.zeros(len(forecast_years_int))
# ---------------------------------------------------------
# SAVE ALL EBITDA VALUES (HISTORICAL + FORECAST)
# ---------------------------------------------------------

dcf_all_ebitda = {}

# 1️⃣ Save historical EBITDA
if ebitda_idx is not None:
    for y in year_cols_is:
        try:
            val = float(forecast_is.iloc[ebitda_idx][str(y)])
        except:
            val = 0.0
        dcf_all_ebitda[str(y)] = val

# 2️⃣ Save forecast EBITDA
for y in forecast_years_int:
    col = str(y)
    val = float(forecast_is.iloc[ebitda_idx][col])
    dcf_all_ebitda[col] = val

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
        use_container_width=True
    )

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
        use_container_width=True
    )

    # 3️⃣ AVERAGE WC% WITH OUTLIER HANDLING
    wc_percent_array = wc_percent_hist.copy()
    mask_valid = (wc_percent_array > -5) & (wc_percent_array < 5)
    wc_percent_clean = wc_percent_array[mask_valid]

    if len(wc_percent_clean) == 0:
        wc_percent_avg = 0.0
        st.warning("No valid WC% of sales ratios found – using 0% by default.")
    else:
        ratio_spread = float(wc_percent_clean.max() - wc_percent_clean.min())
        spread_threshold = 0.05  # 5%

        if ratio_spread > spread_threshold and len(wc_percent_clean) >= 2:
            last_year = common_hist[-1]
            last_wc = float(wc_hist[last_year])
            last_rev = float(revenue_row[last_year].values[0])
            wc_percent_avg = last_wc / last_rev

            st.warning(
                f"WC% of sales ratios differ a lot (spread ≈ {ratio_spread:.2%}). "
                f"Using the **most recent WC% ({wc_percent_avg:.2%})** for forecasting "
                f"instead of the average."
            )
        else:
            wc_percent_avg = float(np.mean(wc_percent_clean))
            st.success(
                f"WC% of sales ratios are in a similar range (spread ≈ {ratio_spread:.2%}). "
                f"Using the **average WC% = {wc_percent_avg:.2%}** for forecasting."
            )

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
        use_container_width=True
    )

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
        use_container_width=True
    )

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
# DCF PARAMETERS — FINAL FIXED VERSION (NO RESETTING)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💰 DCF Parameters")

# 1️⃣ Initialize ONLY once
if "dcf_init" not in st.session_state:
    st.session_state["dcf_rf_pct"] = 11.61
    st.session_state["dcf_mrp_pct"] = 13.82
    st.session_state["dcf_tax_pct"] = 25.0
    st.session_state["dcf_unlevered_beta"] = 1.00
    st.session_state["dcf_terminal_g_pct"] = 5.0
    st.session_state["dcf_init"] = True


# 2️⃣ Widgets use SEPARATE KEYS (critical fix)
col1, col2 = st.columns(2)

with col1:
    rf_input = st.number_input(
        "Risk-free rate (%)",
        value=float(st.session_state["dcf_rf_pct"]),
        step=0.1,
        key="dcf_rf_pct_input"
    )
    mrp_input = st.number_input(
        "Market risk premium (%)",
        value=float(st.session_state["dcf_mrp_pct"]),
        step=0.1,
        key="dcf_mrp_pct_input"
    )
    tax_input = st.number_input(
        "Tax rate (%)",
        value=float(st.session_state["dcf_tax_pct"]),
        step=0.5,
        key="dcf_tax_pct_input"
    )

with col2:
    beta_u_input = st.number_input(
        "Unlevered beta (asset beta)",
        value=float(st.session_state["dcf_unlevered_beta"]),
        step=0.05,
        key="dcf_unlevered_beta_input"
    )
    g_input = st.number_input(
        "Terminal growth rate (%)",
        value=float(st.session_state["dcf_terminal_g_pct"]),
        step=0.1,
        key="dcf_terminal_g_pct_input"
    )

# 3️⃣ UPDATE session_state explicitly (correct order)
st.session_state["dcf_rf_pct"] = rf_input
st.session_state["dcf_mrp_pct"] = mrp_input
st.session_state["dcf_tax_pct"] = tax_input
st.session_state["dcf_unlevered_beta"] = beta_u_input
st.session_state["dcf_terminal_g_pct"] = g_input

# 4️⃣ Convert to decimals
rf = st.session_state["dcf_rf_pct"] / 100
mrp = st.session_state["dcf_mrp_pct"] / 100
tax = st.session_state["dcf_tax_pct"] / 100
g = st.session_state["dcf_terminal_g_pct"] / 100

# 5️⃣ Calculate CAPM & WACC
beta_levered = st.session_state["dcf_unlevered_beta"] * (1 + (1 - tax) * de_ratio)

if de_ratio <= 0:
    w_e, w_d = 1, 0
else:
    w_d = de_ratio / (1 + de_ratio)
    w_e = 1 / (1 + de_ratio)

re = rf + beta_levered * mrp
wacc = w_e * re + w_d * rd * (1 - tax)

# Save computed
st.session_state["levered_beta"] = beta_levered
st.session_state["wacc"] = wacc

# 6️⃣ Display results
st.markdown(
    f"""
### 📌 DCF Output  
**Auto Cost of Debt:** {rd*100:.2f}%  
**Levered Beta:** {beta_levered:.2f}  
**Cost of Equity (Re):** {re*100:.2f}%  
**WACC:** {wacc*100:.2f}%  
"""
)

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

st.dataframe(midpoint_table, use_container_width=True)


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
st.dataframe(styled_dcf, use_container_width=True)

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
    use_container_width=True,
)

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
st.subheader("📌 Valuation Summary")

c_sum1, c_sum2 = st.columns(2)
with c_sum1:
    st.metric("Enterprise Value (EV)", f"{enterprise_value:,.0f}")
    st.metric("Net Debt", f"{net_debt:,.0f}")
with c_sum2:
    st.metric("Equity Value", f"{equity_value:,.0f}")
    st.metric("WACC", f"{wacc*100:.2f}%")
    st.metric("Terminal Growth Rate", f"{g*100:.2f}%")
