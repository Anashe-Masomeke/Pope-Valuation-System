
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
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
# =========================================================
# Styling helpers
# =========================================================
def format_numeric_columns(df: pd.DataFrame):
    fmt = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            fmt[col] = "{:,.2f}"
    return df.style.format(fmt)


# =========================================================
# PEER UNIVERSE LOADER
# =========================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # pages/ -> project root -> data
DEFAULT_PEER_FILE = DATA_DIR / "peer_universe.xlsx"

def normalize_peer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # ✅ include PeerGroup + Active so your logic works reliably
    needed = [
        "Company", "Ticker", "Exchange", "Country",
        "Sector", "Industry", "PeerGroup", "Active",
        "EV/EBITDA", "P/B", "P/E"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # numeric
    for c in ["EV/EBITDA", "P/B", "P/E"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # text cleanup
    for c in ["Company", "Ticker", "Exchange", "Country", "Sector", "Industry", "PeerGroup"]:
        df[c] = df[c].astype(str).replace("nan", "").str.strip()

    # active cleanup (default TRUE if blank)
    def to_bool(x):
        s = str(x).strip().upper()
        if s in ["", "NAN"]:
            return True
        return s in ["TRUE", "1", "YES", "Y"]

    df["Active"] = df["Active"].apply(to_bool)

    df = df[df["Company"].astype(str).str.len() > 0].copy()
    df = df[df["Active"] == True].copy()

    # stable Label
    df["Label"] = df.apply(
        lambda r: f"{r['Company']} ({r['Ticker']})" if str(r.get("Ticker", "")).strip() else f"{r['Company']}",
        axis=1
    )
    return df

# ✅ Cache that auto-refreshes when the Excel file changes (mtime is part of the cache key)
@st.cache_data(show_spinner=False)
def load_peer_universe_cached(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_excel(path_str)
    return normalize_peer_df(df)

def load_peer_universe_from_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(file_bytes))
    return normalize_peer_df(df)

# =========================================================
# ✅ Peer suggestions (STRICT PeerGroup only if present)
#   - If target has PeerGroup: ONLY peers in same PeerGroup (no sector mixing)
#   - If PeerGroup empty: fallback Industry -> Sector
# =========================================================
def suggest_peers(peer_df: pd.DataFrame, target_row: pd.Series, max_peers=10) -> pd.DataFrame:
    target_company = str(target_row.get("Company", "")).strip()
    target_pg = str(target_row.get("PeerGroup", "")).strip()
    target_industry = str(target_row.get("Industry", "")).strip()
    target_sector = str(target_row.get("Sector", "")).strip()

    df = peer_df.copy()
    df = df[df["Company"] != target_company].copy()

    def eq_ci(series: pd.Series, value: str):
        return series.astype(str).str.strip().str.lower() == str(value).strip().lower()

    # STRICT: if PeerGroup exists, use it ONLY (prevents Delta pulling Innscor etc.)
    if target_pg:
        pool = df[eq_ci(df["PeerGroup"], target_pg)].copy()
    else:
        pool_ind = df[eq_ci(df["Industry"], target_industry)].copy() if target_industry else df.iloc[0:0].copy()
        if len(pool_ind) >= 2:
            pool = pool_ind
        elif target_sector:
            pool = df[eq_ci(df["Sector"], target_sector)].copy()
        else:
            pool = df.copy()

    pool["has_mult"] = (
        pool["EV/EBITDA"].notna().astype(int)
        + pool["P/B"].notna().astype(int)
        + pool["P/E"].notna().astype(int)
    )

    pool = (
        pool.sort_values(["has_mult", "Company"], ascending=[False, True])
            .head(int(max_peers))
            .drop(columns="has_mult", errors="ignore")
    )
    return pool

# =========================================================
# CORE SYNC: peer multiselect -> Step 1 comparables
# =========================================================
def sync_selected_peers_to_comparables(peer_df: pd.DataFrame, selected_peer_labels):
    """
    Always keep Step 1 comps equal to selected peers.
    - removing a peer removes it from comps immediately
    - remaining peers keep their typed values AND include flags
    - company names auto-match peers
    """
    S = st.session_state
    peer_lookup = {r["Label"]: r for _, r in peer_df.iterrows()}

    existing_by_label = {}
    old_labels = S.get("comps_peer_labels", [])
    old_comps = S.get("comps", {})

    # Map old label -> comp dict (including include flags if present)
    for i, lab in enumerate(old_labels):
        if i in old_comps:
            existing_by_label[lab] = old_comps[i].copy()

    new_comps = {}
    new_labels = []

    for i, lab in enumerate(selected_peer_labels):
        row = peer_lookup.get(lab, {})

        default_name = row.get("Label") or lab
        default_ev = float(row.get("EV/EBITDA", 0) or 0)
        default_pb = float(row.get("P/B", 0) or 0)
        default_pe = float(row.get("P/E", 0) or 0)

        # include flags defaults
        default_inc_ev = True
        default_inc_pb = True
        default_inc_pe = True

        if lab in existing_by_label:
            comp = existing_by_label[lab]

            # keep include flags if they exist, otherwise default True
            comp["inc_ev"] = bool(comp.get("inc_ev", default_inc_ev))
            comp["inc_pb"] = bool(comp.get("inc_pb", default_inc_pb))
            comp["inc_pe"] = bool(comp.get("inc_pe", default_inc_pe))

            # keep typed values if any
            comp["name"] = default_name
            comp["ev"] = float(comp.get("ev", default_ev) or 0)
            comp["pb"] = float(comp.get("pb", default_pb) or 0)
            comp["pe"] = float(comp.get("pe", default_pe) or 0)
        else:
            comp = {
                "name": default_name,
                "ev": default_ev,
                "pb": default_pb,
                "pe": default_pe,
                "inc_ev": default_inc_ev,
                "inc_pb": default_inc_pb,
                "inc_pe": default_inc_pe,
            }

        new_comps[i] = comp
        new_labels.append(lab)

        # push into widget keys BEFORE widgets exist
        S[f"comp_name_{i}"] = comp["name"]
        S[f"comp_ev_{i}"] = comp["ev"]
        S[f"comp_pb_{i}"] = comp["pb"]
        S[f"comp_pe_{i}"] = comp["pe"]

        S[f"inc_ev_{i}"] = bool(comp["inc_ev"])
        S[f"inc_pb_{i}"] = bool(comp["inc_pb"])
        S[f"inc_pe_{i}"] = bool(comp["inc_pe"])

    old_n = int(S.get("num_comps", 0))
    new_n = len(new_labels)

    # delete old widget keys beyond new_n (including include keys)
    for i in range(new_n, old_n + 20):
        for k in [
            f"comp_name_{i}", f"comp_ev_{i}", f"comp_pb_{i}", f"comp_pe_{i}",
            f"inc_ev_{i}", f"inc_pb_{i}", f"inc_pe_{i}"
        ]:
            if k in S:
                del S[k]

    S["comps"] = new_comps
    S["comps_peer_labels"] = new_labels
    S["num_comps"] = new_n
    S["num_comps_input"] = new_n
    S["comps_num"] = new_n

# =========================================================
# Streamlit Page
# =========================================================
st.set_page_config(page_title="Comparables Valuation (Excel Style)", layout="wide")
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

st.title("📊 Comparables Valuation – EV/EBITDA, P/B, P/E")
st.caption("All values & inputs are saved in session_state (won’t reset when switching tabs).")

S = st.session_state

# =========================================================
# ✅ Namespaced keys (prevents other pages overwriting your values)
# =========================================================
K_TARGET = "cmp_target_company"
K_SELECTED = "cmp_selected_peer_companies"
K_MAX = "cmp_max_peers"
K_AUTOFILL = "cmp_auto_fill_now"
K_PEERMODE = "cmp_peer_mode"
K_PREV_MAX = "cmp_prev_max_peers"
K_UP_BYTES = "cmp_peer_upload_bytes"
K_UP_NAME = "cmp_peer_upload_name"

# =========================================================
# STEP 0 — PEER UNIVERSE AUTO-FILL
# =========================================================
st.header("Step 0 — Auto-Fill Comparables (Peer Universe)")

S.setdefault(K_PEERMODE, True)
peer_mode = st.toggle(
    "Use Peer Universe Excel to auto-fill comparables",
    value=bool(S[K_PEERMODE]),
    key="cmp_peer_mode_toggle",
)
S[K_PEERMODE] = peer_mode

peer_df = None

if peer_mode:
    # 1) If user previously uploaded a file, persist it across tab switches
    if S.get(K_UP_BYTES):
        try:
            peer_df = load_peer_universe_from_bytes(S[K_UP_BYTES])
            st.info(f"Loaded from uploaded file: {S.get(K_UP_NAME, 'peer_universe.xlsx')}")
        except Exception as e:
            st.error(f"Could not load previously uploaded peer file: {e}")
            peer_df = None

    # 2) Otherwise load default /data/peer_universe.xlsx (auto-refresh on file change)
    if peer_df is None:
        if DEFAULT_PEER_FILE.exists():
            mtime = DEFAULT_PEER_FILE.stat().st_mtime
            peer_df = load_peer_universe_cached(str(DEFAULT_PEER_FILE), mtime)
            st.info(f"Loaded from: {DEFAULT_PEER_FILE} (auto-refresh when file changes)")
        else:
            st.warning(f"Missing: {DEFAULT_PEER_FILE}  → put peer_universe.xlsx inside /data/")

    # 3) Optional uploader (and we persist bytes so it stays after tab switching)
    up = st.file_uploader(
        "Upload peer_universe.xlsx (optional)",
        type=["xlsx"],
        key="cmp_peer_universe_uploader",
    )
    if up is not None:
        try:
            file_bytes = up.getvalue()
            peer_df = load_peer_universe_from_bytes(file_bytes)
            S[K_UP_BYTES] = file_bytes
            S[K_UP_NAME] = up.name
            st.success("Peer universe uploaded, loaded, and saved for this session.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    if peer_df is not None and len(peer_df) > 0:
        labels_all = peer_df["Label"].tolist()
        valid_set = set(labels_all)

        S.setdefault(K_MAX, 8)
        S.setdefault(K_AUTOFILL, True)
        S.setdefault(K_TARGET, labels_all[0])
        S.setdefault(K_SELECTED, [])

        # Clean invalid selections after tab switch / refresh
        S[K_SELECTED] = [x for x in S[K_SELECTED] if x in valid_set]
        if S[K_TARGET] not in valid_set:
            S[K_TARGET] = labels_all[0]

        c0a, c0b = st.columns([2, 1])
        with c0a:
            target_label = st.selectbox(
                "Select Target Company (the one you are valuing)",
                options=labels_all,
                index=labels_all.index(S[K_TARGET]),
                key="cmp_target_company_select",
            )
        with c0b:
            max_peers = st.number_input(
                "Max peers",
                min_value=3,
                max_value=20,
                value=int(S[K_MAX]),
                step=1,
                key="cmp_max_peers_input",
            )

        auto_fill_now = st.checkbox(
            "Auto-fill comparables instantly when I choose a target",
            value=bool(S[K_AUTOFILL]),
            key="cmp_auto_fill_now_checkbox",
        )
        S[K_AUTOFILL] = bool(auto_fill_now)
        S[K_MAX] = int(max_peers)

        # Detect changes
        target_changed = (target_label != S.get(K_TARGET))
        max_changed = (int(max_peers) != int(S.get(K_PREV_MAX, -999)))

        if target_changed or max_changed:
            S[K_TARGET] = target_label
            S[K_PREV_MAX] = int(max_peers)

            target_row = peer_df.loc[peer_df["Label"] == target_label].iloc[0]
            suggested_df = suggest_peers(peer_df, target_row, max_peers=int(max_peers))
            suggested_labels = suggested_df["Label"].tolist()

            S[K_SELECTED] = suggested_labels

            if S[K_AUTOFILL]:
                sync_selected_peers_to_comparables(peer_df, suggested_labels)

        # Multiselect persists and remains editable after tab switching
        def on_peers_change():
            S[K_SELECTED] = [x for x in S.get(K_SELECTED, []) if x in valid_set]
            sync_selected_peers_to_comparables(peer_df, S.get(K_SELECTED, []))

        st.multiselect(
            "Peer companies (auto-selected — edit only if needed)",
            options=labels_all,
            key=K_SELECTED,
            on_change=on_peers_change,
        )

        # ✅ ONLY CLEAR BUTTON (as you requested)
        if st.button("🧹 Clear Comparables", width='stretch'):
            old_n = int(S.get("num_comps", 3))

            # wipe widget keys so UI becomes blank/0
            for i in range(0, max(old_n, 20)):
                S[f"comp_name_{i}"] = ""
                S[f"comp_ev_{i}"] = 0.0
                S[f"comp_pb_{i}"] = 0.0
                S[f"comp_pe_{i}"] = 0.0
                S[f"inc_ev_{i}"] = True
                S[f"inc_pb_{i}"] = True
                S[f"inc_pe_{i}"] = True

            # reset internal comps structure (blank)
            S["num_comps"] = 3
            S["num_comps_input"] = 3
            S["comps"] = {
                0: {"name": "", "ev": 0.0, "pb": 0.0, "pe": 0.0, "inc_ev": True, "inc_pb": True, "inc_pe": True},
                1: {"name": "", "ev": 0.0, "pb": 0.0, "pe": 0.0, "inc_ev": True, "inc_pb": True, "inc_pe": True},
                2: {"name": "", "ev": 0.0, "pb": 0.0, "pe": 0.0, "inc_ev": True, "inc_pb": True, "inc_pe": True},
            }

            # clear helper lists used by summary
            for k in ["comps_peer_labels", "comps_num", "comps_ev_list", "comps_pb_list", "comps_pe_list"]:
                if k in S:
                    del S[k]

            st.success("Comparables cleared (company names + multiples).")
            st.rerun()

st.markdown("---")

# =========================================================
# STEP 1 — INPUT COMPARABLE COMPANIES & MULTIPLES
# =========================================================
st.header("Step 1 — Input Comparable Companies & Multiples")

S.setdefault("num_comps", 3)
S.setdefault("comps", {})

num_comps = st.number_input(
    "How many comparables?",
    min_value=1,
    max_value=20,
    value=int(S.get("num_comps", 3)),
    key="num_comps_input",
)
S["num_comps"] = int(num_comps)

for i in range(int(num_comps)):
    S["comps"].setdefault(i, {
        "name": f"Comp {i + 1}",
        "ev": 0.0, "pb": 0.0, "pe": 0.0,
        "inc_ev": True, "inc_pb": True, "inc_pe": True
    })


rows_comps = []

for i in range(int(num_comps)):
    st.subheader(f"Comparable {i + 1}")

    # 4 inputs + 1 include/exclude column
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.3])

    # ---------- Name ----------
    with c1:
        default_name = S.get(f"comp_name_{i}", S["comps"][i]["name"])
        name_val = st.text_input(
            f"Company {i + 1} name",
            value=str(default_name),
            key=f"comp_name_{i}",
        )
        S["comps"][i]["name"] = name_val

    # ---------- Multiples ----------
    with c2:
        default_ev = float(S.get(f"comp_ev_{i}", S["comps"][i]["ev"]))
        ev_val = st.number_input(
            f"{name_val} EV/EBITDA",
            value=float(default_ev),
            step=0.01,
            format="%.2f",
            key=f"comp_ev_{i}",
        )
        S["comps"][i]["ev"] = ev_val

    with c3:
        default_pb = float(S.get(f"comp_pb_{i}", S["comps"][i]["pb"]))
        pb_val = st.number_input(
            f"{name_val} P/B",
            value=float(default_pb),
            step=0.01,
            format="%.2f",
            key=f"comp_pb_{i}",
        )
        S["comps"][i]["pb"] = pb_val

    with c4:
        default_pe = float(S.get(f"comp_pe_{i}", S["comps"][i]["pe"]))
        pe_val = st.number_input(
            f"{name_val} P/E",
            value=float(default_pe),
            step=0.01,
            format="%.2f",
            key=f"comp_pe_{i}",
        )
        S["comps"][i]["pe"] = pe_val

    # ---------- Analyst relevance toggles (persisted) ----------
    # Defaults to True (include) unless user switches off
    # ---------- Analyst relevance toggles (persisted) ----------
    # ---------- Analyst relevance toggles (persisted + stored in comps) ----------
    ev_key = f"inc_ev_{i}"
    pb_key = f"inc_pb_{i}"
    pe_key = f"inc_pe_{i}"

    # make sure comps has flags
    S["comps"][i].setdefault("inc_ev", True)
    S["comps"][i].setdefault("inc_pb", True)
    S["comps"][i].setdefault("inc_pe", True)

    # set widget defaults BEFORE widgets exist (only if keys not already created)
    if ev_key not in S: S[ev_key] = bool(S["comps"][i]["inc_ev"])
    if pb_key not in S: S[pb_key] = bool(S["comps"][i]["inc_pb"])
    if pe_key not in S: S[pe_key] = bool(S["comps"][i]["inc_pe"])

    with c5:
        st.caption("Analyst filter")
        st.checkbox("Include EV", key=ev_key)
        st.checkbox("Include P/B", key=pb_key)
        st.checkbox("Include P/E", key=pe_key)

    # read values AFTER widgets
    inc_ev = bool(S[ev_key])
    inc_pb = bool(S[pb_key])
    inc_pe = bool(S[pe_key])

    # store back to comps structure (safe: not modifying widget keys)
    S["comps"][i]["inc_ev"] = inc_ev
    S["comps"][i]["inc_pb"] = inc_pb
    S["comps"][i]["inc_pe"] = inc_pe

    rows_comps.append([name_val, ev_val, pb_val, pe_val, inc_ev, inc_pb, inc_pe])

df_comps = pd.DataFrame(
    rows_comps,
    columns=["Company", "EV/EBITDA", "P/B", "P/E", "Include_EV", "Include_PB", "Include_PE"]
)

st.subheader("Entered Comparables")
st.dataframe(df_comps, width='stretch')

# keep your existing lists if you want, but now you also have include flags
S["comps_num"] = int(num_comps)
S["comps_ev_list"] = df_comps["EV/EBITDA"].astype(float).tolist()
S["comps_pb_list"] = df_comps["P/B"].astype(float).tolist()
S["comps_pe_list"] = df_comps["P/E"].astype(float).tolist()

# NEW: store include masks too (useful later)
S["comps_inc_ev"] = df_comps["Include_EV"].astype(bool).tolist()
S["comps_inc_pb"] = df_comps["Include_PB"].astype(bool).tolist()
S["comps_inc_pe"] = df_comps["Include_PE"].astype(bool).tolist()

# =========================================================
# STEP 2 — AVERAGE & IMPLIED MULTIPLES
# =========================================================
st.header("Step 2 — Average & Implied Multiples")

# Excel-style averaging:
# - zeros INCLUDED
# - blanks (NaN) ignored

ev_series = df_comps.loc[df_comps["Include_EV"] == True, "EV/EBITDA"]
pb_series = df_comps.loc[df_comps["Include_PB"] == True, "P/B"]
pe_series = df_comps.loc[df_comps["Include_PE"] == True, "P/E"]

avg_ev = float(ev_series.mean()) if ev_series.notna().any() else np.nan
avg_pb = float(pb_series.mean()) if pb_series.notna().any() else np.nan
avg_pe = float(pe_series.mean()) if pe_series.notna().any() else np.nan




S.setdefault("discount_pct", 25.0)
discount_pct = st.number_input(
    "Discount factor (%)",
    value=float(S["discount_pct"]),
    step=1.0,
    key="discount_pct_input",
)
S["discount_pct"] = float(discount_pct)
discount = float(discount_pct) / 100.0

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

st.dataframe(df_mult.style.format({"Average": "{:,.2f}", "Implied": "{:,.2f}"}), width='stretch')

S["implied_ev"] = float(implied_ev) if not pd.isna(implied_ev) else 0.0
S["implied_pb"] = float(implied_pb) if not pd.isna(implied_pb) else 0.0
S["implied_pe"] = float(implied_pe) if not pd.isna(implied_pe) else 0.0

# =========================================================
# TIMING SOURCE (from DCF) — BASE USED BY BOTH EBITDA & EARNINGS
# =========================================================
st.header("Timing Source (from DCF)")

dcf_timing_list = S.get("dcf_discount_periods_n", [])
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
    timing_df = pd.DataFrame(
        {"Forecast Year Index": list(range(len(dcf_timing_list))), "DCF Timing n": dcf_timing_list}
    )
    st.dataframe(timing_df, width='stretch')

    dcf_n0 = float(round(dcf_timing_list[0], 4))
    st.info(f"DCF First Timing Value (n₀) = **{dcf_n0} years**")

    timing_choice = st.radio(
        "Choose timing base for Comparables timing effect:",
        [f"Use DCF n₀ = {dcf_n0}", "Manually override starting timing value"],
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

S["comp_timing_base"] = float(base_timing)
st.success(f"Timing base for comparables = **{base_timing:.4f}**")

# =========================================================
# STEP 3 — MAINTAINABLE EBITDA (with locked timing)
# =========================================================
st.header("Step 3 — Maintainable EBITDA")

dcf_eb_all = S.get("dcf_ebitda_all", None)
if dcf_eb_all is None:
    dcf_eb_all = S.get("dcf_ebitda_forecast", {})

# ✅ If no DCF EBITDA → SKIP Step 3 (no manual inputs)
if not dcf_eb_all:
    st.warning("⚠ No EBITDA found from DCF — skipping EV/EBITDA method.")
    S["maintainable_ebitda"] = np.nan

else:
    # ✅ SAFETY: only accept 4-digit year keys (e.g., 2024)
    eb_years_all = sorted(
        int(y) for y in dcf_eb_all.keys()
        if str(y).strip().isdigit() and len(str(y).strip()) == 4
    )

    if not eb_years_all:
        st.warning("⚠ DCF EBITDA found, but no valid 4-digit year keys — skipping EV/EBITDA method.")
        S["maintainable_ebitda"] = np.nan

    else:
        eb_min_year = min(eb_years_all)
        eb_max_year = max(eb_years_all)

        S.setdefault("comp_eb_start_year", eb_min_year)
        S.setdefault("comp_eb_end_year", eb_max_year)
        S.setdefault("comp_eb_weights", {})
        S.setdefault("comp_use_timing_eb", True)

        use_timing_eb = st.checkbox(
            "Apply timing effect from DCF to EBITDA?",
            value=bool(S.get("comp_use_timing_eb", True)),
            key="comp_use_timing_eb_checkbox",
        )
        S["comp_use_timing_eb"] = use_timing_eb
        # ---------------------------------------------------------
        # ✅ HARD SYNC: Earnings timing ALWAYS follows EBITDA timing when EBITDA changes
        # ---------------------------------------------------------
        prev_eb = S.get("_prev_comp_use_timing_eb", None)

        # if EBITDA timing changed this run, force earnings timing to match
        if prev_eb is None or bool(prev_eb) != bool(use_timing_eb):
            S["comp_use_timing_np"] = bool(use_timing_eb)
            S["comp_use_timing_np_checkbox"] = bool(use_timing_eb)  # this updates the UI checkbox

        S["_prev_comp_use_timing_eb"] = bool(use_timing_eb)

        # If user turned off EBITDA timing, also turn off Earnings timing immediately
        if not use_timing_eb:
            S["comp_use_timing_np"] = False
            S["comp_use_timing_np_checkbox"] = False

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
        base_timing = float(S.get("comp_timing_base", 0.0))

        for idx, yr in enumerate(selected_eb_years):
            eb_val = float(dcf_eb_all.get(str(yr), 0.0))
            default_w = float(S["comp_eb_weights"].get(str(yr), 0.0))

            if not use_timing_eb:
                timing_val = 1.0
            else:
                timing_val = base_timing + idx

            c1, c2, c4 = st.columns([1, 2, 1])
            with c1:
                st.number_input(f"EB Year {yr}", value=int(yr), disabled=True, key=f"comp_eb_year_display_{yr}")
            with c2:
                st.number_input(f"EBITDA {yr}", value=eb_val, disabled=True, format="%.2f", key=f"comp_eb_value_display_{yr}")
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

        if use_timing_eb:
            df_eb_display = df_eb[["Year", "EBITDA", "Timing", "Weight (%)", "Adjusted EBITDA", "Weighted EBITDA"]]
        else:
            df_eb_display = df_eb[["Year", "EBITDA", "Weight (%)", "Weighted EBITDA"]]

        df_eb_display = df_eb_display.copy()
        df_eb_display.index = df_eb_display.index + 1
        st.dataframe(format_numeric_columns(df_eb_display), width='stretch')

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

# ✅ If no DCF Earnings → SKIP Step 4 (no manual inputs)
if not dcf_np_all:
    st.warning("⚠ No Earnings found from DCF — skipping P/E method.")
    S["maintainable_earnings"] = np.nan

else:
    # ✅ SAFETY: only accept 4-digit year keys (e.g., 2024)
    np_years_all = sorted(
        int(y) for y in dcf_np_all.keys()
        if str(y).strip().isdigit() and len(str(y).strip()) == 4
    )

    if not np_years_all:
        st.warning("⚠ DCF Earnings found, but no valid 4-digit year keys — skipping P/E method.")
        S["maintainable_earnings"] = np.nan

    else:
        np_min_year = min(np_years_all)
        np_max_year = max(np_years_all)

        S.setdefault("comp_np_start_year", np_min_year)
        S.setdefault("comp_np_end_year", np_max_year)
        S.setdefault("comp_np_weights", {})
        S.setdefault("comp_use_timing_np", True)
        # ---------------------------------------------------------
        # AUTO-SYNC Earnings weighting from EBITDA weighting
        # (same years + same weights)
        # ---------------------------------------------------------
        S.setdefault("comp_sync_np_to_eb", True)

        sync_np_to_eb = st.checkbox(
            "Auto-use the SAME years & weights as EBITDA (recommended)",
            value=bool(S.get("comp_sync_np_to_eb", True)),
            key="comp_sync_np_to_eb_checkbox",
        )
        S["comp_sync_np_to_eb"] = bool(sync_np_to_eb)

        if sync_np_to_eb:
            # Copy start/end year from EBITDA section
            eb_sy = int(S.get("comp_eb_start_year", np_min_year))
            eb_ey = int(S.get("comp_eb_end_year", np_max_year))

            # Clamp within NP available year range
            eb_sy = max(eb_sy, np_min_year)
            eb_ey = min(eb_ey, np_max_year)

            S["comp_np_start_year"] = eb_sy
            S["comp_np_end_year"] = eb_ey

            # Copy per-year weights from EBITDA section
            eb_w = S.get("comp_eb_weights", {}) or {}
            S["comp_np_weights"] = {str(y): float(eb_w.get(str(y), 0.0)) for y in range(eb_sy, eb_ey + 1)}

            # IMPORTANT: also prefill the Earnings weight widgets (so UI matches)
            for y in range(eb_sy, eb_ey + 1):
                S[f"comp_np_weight_{y}"] = float(S["comp_np_weights"].get(str(y), 0.0))

            st.info("✅ Earnings years & weights copied from EBITDA automatically.")
        # ---------------------------------------------------------
        # ✅ AUTO-SYNC timing toggle: if EBITDA timing is OFF, Earnings timing must also be OFF
        # ---------------------------------------------------------
        use_timing_eb = bool(S.get("comp_use_timing_eb", True))  # from Step 3

        # If EBITDA timing is OFF, force Earnings timing OFF (also forces UI key)
        if not use_timing_eb:
            S["comp_use_timing_np"] = False
            S["comp_use_timing_np_checkbox"] = False

        use_timing_np = st.checkbox(
            "Apply timing effect from DCF to Earnings?",
            value=bool(S.get("comp_use_timing_np", True)),
            key="comp_use_timing_np_checkbox",
            disabled=(not use_timing_eb),  # lock it when EBITDA timing is OFF
        )
        S["comp_use_timing_np"] = bool(use_timing_np)

        # Show locked years when sync is ON, otherwise allow manual selection
        if sync_np_to_eb:
            np_start_year = int(S["comp_np_start_year"])
            np_end_year = int(S["comp_np_end_year"])

            c_np1, c_np2 = st.columns(2)
            with c_np1:
                st.number_input(
                    "NP Start Year (auto from EBITDA)",
                    value=int(np_start_year),
                    disabled=True,
                    key="np_start_locked",
                )
            with c_np2:
                st.number_input(
                    "NP End Year (auto from EBITDA)",
                    value=int(np_end_year),
                    disabled=True,
                    key="np_end_locked",
                )

        else:
            c_np1, c_np2 = st.columns(2)
            with c_np1:
                np_start_year = st.number_input(
                    "NP Start Year",
                    value=int(S.get("comp_np_start_year", np_min_year)),
                    step=1,
                    key="comp_np_start_year_input"
                )
            with c_np2:
                np_end_year = st.number_input(
                    "NP End Year",
                    value=int(S.get("comp_np_end_year", np_max_year)),
                    step=1,
                    key="comp_np_end_year_input"
                )

            # clamp
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
        base_timing = float(S.get("comp_timing_base", 0.0))

        for idx, yr in enumerate(selected_np_years):
            np_val = float(dcf_np_all.get(str(yr), 0.0))
            default_w = float(S.get(f"comp_np_weight_{yr}", S["comp_np_weights"].get(str(yr), 0.0)))

            if not use_timing_np:
                timing_val = 1.0
            else:
                timing_val = base_timing + idx

            c1, c2, c4 = st.columns([1, 2, 1])
            with c1:
                st.number_input(f"Earnings Year {yr}", value=int(yr), disabled=True, key=f"comp_np_year_display_{yr}")
            with c2:
                st.number_input(f"Earnings {yr}", value=np_val, disabled=True, format="%.2f", key=f"comp_np_value_display_{yr}")
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
            df_np_display = df_np[["Year", "Earnings", "Timing", "Weight (%)", "Adjusted Earnings", "Weighted Earnings"]]
        else:
            df_np_display = df_np[["Year", "Earnings", "Weight (%)", "Weighted Earnings"]]
        df_np_display = df_np_display.copy()
        df_np_display.index = df_np_display.index + 1
        st.dataframe(format_numeric_columns(df_np_display), width='stretch')
        maintainable_earnings = float(df_np["Weighted Earnings"].sum())
        st.success(f"Maintainable Earnings = {maintainable_earnings:,.2f}")
        S["maintainable_earnings"] = maintainable_earnings
# =========================================================
# STEP 5 — BOOK VALUE & NET DEBT
# =========================================================
st.header("Step 5 — Book Value & Net Debt")
# ✅ Pull Beginning Book Value from Banking page (Totals / BV)
bank_outputs = (S.get("bank", {}) or {}).get("outputs", {}) or {}
bank_book_equity = bank_outputs.get("book_equity_0", None)  # Beginning Book Value (Total Equity)
# If user hasn't typed anything yet, auto-fill book equity from banking
if bank_book_equity is not None:
    # only auto-set if user hasn't created/edited the input widget yet
    if "book_equity_input" not in S:
        S["book_equity"] = float(bank_book_equity)
        S["book_equity_input"] = float(bank_book_equity)

book_equity_default = float(S.get("book_equity", 0.0))
net_debt_default = float(S.get("net_debt", 0.0))

book_equity = st.number_input(
    "Book Equity (USD)",
    value=book_equity_default,
    step=1000.0,
    format="%.2f",   # ⚠ no commas here
    key="book_equity_input"
)
S["book_equity"] = float(book_equity)

# Pretty display with commas (read-only)
st.caption(f"💰 Book Equity: **{book_equity:,.2f} USD**")

net_debt = st.number_input(
    "Net Debt (USD)",
    value=net_debt_default,
    step=1000.0,
    format="%.2f",   # ⚠ no commas here
    key="net_debt_input"
)
S["net_debt"] = float(net_debt)

# Pretty display with commas (read-only)
st.caption(f"💳 Net Debt: **{net_debt:,.2f} USD**")

# =========================================================
# STEP 6 — FINAL EQUITY VALUES
# =========================================================
st.header("Step 6 — Computed Equity Values")

maintainable_ebitda = S.get("maintainable_ebitda", np.nan)
maintainable_earnings = S.get("maintainable_earnings", np.nan)

equity_ev = np.nan
equity_pb = np.nan
equity_pe = np.nan

# EV/EBITDA only if EBITDA exists
if maintainable_ebitda is not None and np.isfinite(float(maintainable_ebitda)) and not pd.isna(implied_ev):
    equity_ev = implied_ev * float(maintainable_ebitda) - net_debt

# P/B works as long as Book Equity exists
if book_equity is not None and np.isfinite(float(book_equity)) and not pd.isna(implied_pb):
    equity_pb = implied_pb * float(book_equity)

# P/E only if Earnings exists
if maintainable_earnings is not None and np.isfinite(float(maintainable_earnings)) and not pd.isna(implied_pe):
    equity_pe = implied_pe * float(maintainable_earnings)


S["value_ev_ebitda"] = float(equity_ev)
S["value_pbv"] = float(equity_pb)
S["value_pe"] = float(equity_pe)

df_res = pd.DataFrame(
    {"Method": ["EV/EBITDA", "P/B", "P/E"], "Equity Value (USD)": [equity_ev, equity_pb, equity_pe]}
)
st.dataframe(format_numeric_columns(df_res), width='stretch')
# =========================================================
# ✅ DOWNLOAD EXCEL (NEAT + FORMULAS) — COMPARABLES EXPORT
# =========================================================
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

def _style_range(ws, cell_range, bold=False, fill=None, align_center=False, border=True, font_color=None):
    thin = Side(style="thin", color="000000")
    b = Border(left=thin, right=thin, top=thin, bottom=thin) if border else None
    for row in ws[cell_range]:
        for c in row:
            if bold:
                c.font = Font(bold=True, color=font_color or c.font.color)
            if fill is not None:
                c.fill = fill
            if align_center:
                c.alignment = Alignment(horizontal="center", vertical="center")
            if b is not None:
                c.border = b

def build_comps_excel_with_formulas(S, df_comps) -> bytes:
    wb = Workbook()

    # ---------- Styles ----------
    header_fill = PatternFill("solid", fgColor="0A1B33")
    header_font = Font(bold=True, color="FFFFFF")
    title_font = Font(bold=True, size=14)
    bold_font = Font(bold=True)
    money_fmt = '#,##0.00'
    pct_fmt = '0.00%'
    mult_fmt = '0.00'

    # ============================
    # Sheet 1: Comps_Input
    # ============================
    ws1 = wb.active
    ws1.title = "Comps_Input"

    ws1["B1"] = "Comparable Company"
    ws1["B1"].font = title_font

    headers = ["Company", "Country", "EV/EBITDA", "P/B", "P/E", "Include_EV", "Include_PB", "Include_PE"]
    start_row = 3
    start_col = 2  # column B

    for j, h in enumerate(headers, start=start_col):
        c = ws1.cell(row=start_row, column=j, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Write comps rows
    r = start_row + 1
    for _, row in df_comps.iterrows():
        ws1.cell(r, start_col + 0, row["Company"])
        ws1.cell(r, start_col + 1, "")  # Country (optional)
        ws1.cell(r, start_col + 2, float(row["EV/EBITDA"]) if pd.notna(row["EV/EBITDA"]) else None)
        ws1.cell(r, start_col + 3, float(row["P/B"]) if pd.notna(row["P/B"]) else None)
        ws1.cell(r, start_col + 4, float(row["P/E"]) if pd.notna(row["P/E"]) else None)

        # TRUE/FALSE flags (Excel friendly)
        ws1.cell(r, start_col + 5, bool(row["Include_EV"]))
        ws1.cell(r, start_col + 6, bool(row["Include_PB"]))
        ws1.cell(r, start_col + 7, bool(row["Include_PE"]))

        # formats
        ws1.cell(r, start_col + 2).number_format = mult_fmt
        ws1.cell(r, start_col + 3).number_format = mult_fmt
        ws1.cell(r, start_col + 4).number_format = mult_fmt
        r += 1

    end_row = r - 1

    # Borders + widths
    _style_range(ws1, f"B{start_row}:I{end_row}", border=True)
    for col, w in zip(["B","C","D","E","F","G","H","I"], [30,16,12,10,10,12,12,12]):
        ws1.column_dimensions[col].width = w

    # ============================
    # Sheet 2: Multiples
    # ============================
    ws2 = wb.create_sheet("Multiples")
    ws2["B1"] = "Multiples Summary"
    ws2["B1"].font = title_font

    # Discount cell (user input from session)
    ws2["B3"] = "Discount (%)"
    ws2["C3"] = float(S.get("discount_pct", 25.0)) / 100.0
    ws2["C3"].number_format = pct_fmt
    ws2["B3"].font = bold_font

    # Table headers
    ws2_headers = ["Multiple", "Average", "Implied"]
    for j, h in enumerate(ws2_headers, start=2):
        c = ws2.cell(row=5, column=j, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Formulas using AVERAGEIF on Include flags
    # Range in Comps_Input:
    # EV: D, PB: E, PE: F   | flags: G,H,I
    # Data rows: start_row+1 .. end_row
    drow1 = start_row + 1
    drow2 = end_row

    ws2["B6"] = "EV/EBITDA"
    ws2["C6"] = f'=IFERROR(AVERAGEIF(Comps_Input!$G${drow1}:$G${drow2},TRUE,Comps_Input!$D${drow1}:$D${drow2}),"")'
    ws2["D6"] = f'=IF(C6="","",C6*(1-$C$3))'

    ws2["B7"] = "P/B"
    ws2["C7"] = f'=IFERROR(AVERAGEIF(Comps_Input!$H${drow1}:$H${drow2},TRUE,Comps_Input!$E${drow1}:$E${drow2}),"")'
    ws2["D7"] = f'=IF(C7="","",C7*(1-$C$3))'

    ws2["B8"] = "P/E"
    ws2["C8"] = f'=IFERROR(AVERAGEIF(Comps_Input!$I${drow1}:$I${drow2},TRUE,Comps_Input!$F${drow1}:$F${drow2}),"")'
    ws2["D8"] = f'=IF(C8="","",C8*(1-$C$3))'

    for rr in [6,7,8]:
        ws2[f"C{rr}"].number_format = mult_fmt
        ws2[f"D{rr}"].number_format = mult_fmt

    _style_range(ws2, "B5:D8", border=True)
    ws2.column_dimensions["B"].width = 18
    ws2.column_dimensions["C"].width = 14
    ws2.column_dimensions["D"].width = 14

    # ============================
    # Sheet 3: EBITDA_Maintainable
    # ============================
    ws3 = wb.create_sheet("EBITDA_Maintainable")
    ws3["B1"] = "Maintainable EBITDA (with timing + weights)"
    ws3["B1"].font = title_font

    ws3["B3"] = "Use Timing?"
    ws3["C3"] = bool(S.get("comp_use_timing_eb", True))
    ws3["B4"] = "Base Timing"
    ws3["C4"] = float(S.get("comp_timing_base", 1.0))

    ws3["B3"].font = bold_font
    ws3["B4"].font = bold_font

    # Pull years + EBITDA from session
    dcf_eb_all = S.get("dcf_ebitda_all", None) or S.get("dcf_ebitda_forecast", {}) or {}
    eb_sy = int(S.get("comp_eb_start_year", 0) or 0)
    eb_ey = int(S.get("comp_eb_end_year", 0) or 0)
    eb_years = list(range(eb_sy, eb_ey + 1)) if eb_sy and eb_ey and eb_ey >= eb_sy else []

    # Table
    headers3 = ["Year", "EBITDA", "Timing", "Weight (%)", "Adjusted EBITDA", "Weighted EBITDA"]
    for j, h in enumerate(headers3, start=2):
        c = ws3.cell(row=6, column=j, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    r0 = 7
    for idx, yr in enumerate(eb_years):
        ws3.cell(r0+idx, 2, yr)
        ws3.cell(r0+idx, 3, float(dcf_eb_all.get(str(yr), 0.0)))

        # Timing formula:
        # =IF($C$3, $C$4 + (ROW()-7), 1)
        ws3.cell(r0+idx, 4, f'=IF($C$3,$C$4+{idx},1)')

        # Weight from session (store as percent)
        w = float((S.get("comp_eb_weights", {}) or {}).get(str(yr), 0.0))
        ws3.cell(r0+idx, 5, w/100.0)

        # Adjusted EBITDA = EBITDA * Timing
        ws3.cell(r0+idx, 6, f"=C{r0+idx}*D{r0+idx}")
        # Weighted EBITDA = Adjusted * Weight
        ws3.cell(r0+idx, 7, f"=F{r0+idx}*E{r0+idx}")

        ws3.cell(r0+idx, 3).number_format = money_fmt
        ws3.cell(r0+idx, 4).number_format = '0.0000'
        ws3.cell(r0+idx, 5).number_format = pct_fmt
        ws3.cell(r0+idx, 6).number_format = money_fmt
        ws3.cell(r0+idx, 7).number_format = money_fmt

    last = r0 + len(eb_years) - 1 if eb_years else 7

    # Total maintainable EBITDA
    ws3["B" + str(last+2)] = "Maintainable EBITDA"
    ws3["B" + str(last+2)].font = bold_font
    ws3["G" + str(last+2)] = f"=SUM(G{r0}:G{last})"
    ws3["G" + str(last+2)].font = bold_font
    ws3["G" + str(last+2)].number_format = money_fmt

    _style_range(ws3, f"B6:G{last}", border=True)
    ws3.column_dimensions["B"].width = 10
    ws3.column_dimensions["C"].width = 18
    ws3.column_dimensions["D"].width = 12
    ws3.column_dimensions["E"].width = 12
    ws3.column_dimensions["F"].width = 18
    ws3.column_dimensions["G"].width = 18

    # ============================
    # Sheet 4: Earnings_Maintainable
    # ============================
    ws4 = wb.create_sheet("Earnings_Maintainable")
    ws4["B1"] = "Maintainable Earnings (with timing + weights)"
    ws4["B1"].font = title_font

    ws4["B3"] = "Use Timing?"
    ws4["C3"] = bool(S.get("comp_use_timing_np", True))
    ws4["B4"] = "Base Timing"
    ws4["C4"] = float(S.get("comp_timing_base", 1.0))
    ws4["B3"].font = bold_font
    ws4["B4"].font = bold_font

    dcf_np_all = S.get("dcf_profit_all", None) or S.get("dcf_profit_forecast", {}) or {}
    np_sy = int(S.get("comp_np_start_year", 0) or 0)
    np_ey = int(S.get("comp_np_end_year", 0) or 0)
    np_years = list(range(np_sy, np_ey + 1)) if np_sy and np_ey and np_ey >= np_sy else []

    headers4 = ["Year", "Earnings", "Timing", "Weight (%)", "Adjusted Earnings", "Weighted Earnings"]
    for j, h in enumerate(headers4, start=2):
        c = ws4.cell(row=6, column=j, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    r0 = 7
    for idx, yr in enumerate(np_years):
        ws4.cell(r0+idx, 2, yr)
        ws4.cell(r0+idx, 3, float(dcf_np_all.get(str(yr), 0.0)))

        ws4.cell(r0+idx, 4, f'=IF($C$3,$C$4+{idx},1)')

        w = float((S.get("comp_np_weights", {}) or {}).get(str(yr), 0.0))
        ws4.cell(r0+idx, 5, w/100.0)

        ws4.cell(r0+idx, 6, f"=C{r0+idx}*D{r0+idx}")
        ws4.cell(r0+idx, 7, f"=F{r0+idx}*E{r0+idx}")

        ws4.cell(r0+idx, 3).number_format = money_fmt
        ws4.cell(r0+idx, 4).number_format = '0.0000'
        ws4.cell(r0+idx, 5).number_format = pct_fmt
        ws4.cell(r0+idx, 6).number_format = money_fmt
        ws4.cell(r0+idx, 7).number_format = money_fmt

    last = r0 + len(np_years) - 1 if np_years else 7

    ws4["B" + str(last+2)] = "Maintainable Earnings"
    ws4["B" + str(last+2)].font = bold_font
    ws4["G" + str(last+2)] = f"=SUM(G{r0}:G{last})"
    ws4["G" + str(last+2)].font = bold_font
    ws4["G" + str(last+2)].number_format = money_fmt

    _style_range(ws4, f"B6:G{last}", border=True)
    for col, w in zip(["B","C","D","E","F","G"], [10,18,12,12,18,18]):
        ws4.column_dimensions[col].width = w

    # ============================
    # Sheet 5: Equity_Values
    # ============================
    ws5 = wb.create_sheet("Equity_Values")
    ws5["B1"] = "Computed Equity Values"
    ws5["B1"].font = title_font

    # Inputs needed (from your Step 5)
    ws5["B3"] = "Book Equity"
    ws5["C3"] = float(S.get("book_equity", 0.0))
    ws5["B4"] = "Net Debt"
    ws5["C4"] = float(S.get("net_debt", 0.0))
    ws5["B3"].font = bold_font
    ws5["B4"].font = bold_font
    ws5["C3"].number_format = money_fmt
    ws5["C4"].number_format = money_fmt

    # Link maintainables
    ws5["B6"] = "Maintainable EBITDA"
    ws5["C6"] = "=EBITDA_Maintainable!G" + str((ws3.max_row))  # last maintainable cell
    ws5["B7"] = "Maintainable Earnings"
    ws5["C7"] = "=Earnings_Maintainable!G" + str((ws4.max_row))  # last maintainable cell

    ws5["C6"].number_format = money_fmt
    ws5["C7"].number_format = money_fmt
    ws5["B6"].font = bold_font
    ws5["B7"].font = bold_font

    # Equity table
    ws5_headers = ["Method", "Equity Value (USD)"]
    for j, h in enumerate(ws5_headers, start=2):
        c = ws5.cell(row=9, column=j, value=h)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")

    # Implied multiples link from Multiples sheet:
    # EV implied at D6, PB implied at D7, PE implied at D8
    ws5["B10"] = "EV/EBITDA"
    ws5["C10"] = "=IF(Multiples!D6=\"\",\"\",Multiples!D6*$C$6-$C$4)"

    ws5["B11"] = "P/B"
    ws5["C11"] = "=IF(Multiples!D7=\"\",\"\",Multiples!D7*$C$3)"

    ws5["B12"] = "P/E"
    ws5["C12"] = "=IF(Multiples!D8=\"\",\"\",Multiples!D8*$C$7)"

    for rr in [10,11,12]:
        ws5[f"C{rr}"].number_format = money_fmt

    _style_range(ws5, "B9:C12", border=True)
    ws5.column_dimensions["B"].width = 18
    ws5.column_dimensions["C"].width = 22

    # Save
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()


excel_bytes = build_comps_excel_with_formulas(S, df_comps)

st.download_button(
    label="⬇️ Download Comparables (Excel with formulas)",
    data=excel_bytes,
    file_name="comparables_with_formulas.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
