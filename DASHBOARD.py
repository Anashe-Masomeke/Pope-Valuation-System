"""
DASHBOARD.py — FBC Investment Valuation System
Entry point — no login required.
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide",
)

# ── Ensure session is always "authenticated" ─────────────────────
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = True
if "user" not in st.session_state or not st.session_state.get("user"):
    st.session_state["user"] = {"username": "analyst", "role": "analyst", "full_name": "Analyst"}

# ══════════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Material+Icons&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

/* ── 1. GLOBAL TYPOGRAPHY ─────────────────────────────────── */
html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small,
.stDataFrame, .stTable {
  font-family: "EB Garamond", Georgia, "Times New Roman", serif !important;
  color: #1a1a2e;
}
h1, h2, h3, h4, .fbc-heading, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: "Playfair Display", Georgia, serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}

/* ── 2. PAGE BACKGROUND ───────────────────────────────────── */
.stApp { background: #f5f7fb !important; }
.main .block-container { background: #f5f7fb !important; padding-top: 1.5rem !important; }

/* ── 3. SIDEBAR ───────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #001a5c 0%, #003399 45%, #0044cc 100%) !important;
    border-right: 2px solid rgba(245,180,0,0.25) !important;
    box-shadow: 4px 0 24px rgba(0,26,92,0.35) !important;
}
section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f5b400, #ffd040, #f5b400);
}
section[data-testid="stSidebar"] * {
    color: #e8f0ff !important;
    font-family: "EB Garamond", Georgia, serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    font-weight: 700 !important;
    text-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] a {
    color: #b8d0ff !important;
    transition: color 0.15s !important;
}
section[data-testid="stSidebar"] a:hover {
    color: #f5b400 !important;
}
section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(245,180,0,0.25) !important;
    margin: 12px 0 !important;
}

/* ✅ Ensure Material Icons render correctly */
.material-icons,
.material-icons-outlined,
.material-symbols-outlined,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] i {
    font-family: 'Material Icons', 'Material Symbols Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: linear-gradient(135deg, #f5b400, #ffd040) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 44px !important; height: 44px !important;
    box-shadow: 0 4px 14px rgba(245,180,0,0.50) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    transform: translateY(-1px) scale(1.06) !important;
    box-shadow: 0 8px 20px rgba(245,180,0,0.60) !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    width: 22px !important; height: 22px !important;
    fill: #001a5c !important;
}
/* Move collapse arrow to right edge of sidebar */
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 12px !important;
    right: 12px !important;
    left: auto !important;
    z-index: 999999 !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: linear-gradient(135deg, #003399, #0055ee) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    box-shadow: 0 6px 18px rgba(0,51,153,0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s ease !important;
}

/* ── Hide the "keyboa..." app name text ── */
[data-testid="stSidebarHeader"] > *:not([data-testid="stSidebarCollapseButton"]) {
    display: none !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, #003399 0%, #0055ee 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: "EB Garamond", serif !important;
    font-size: 15px !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(0,51,153,0.30) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(0,51,153,0.40) !important;
}

/* ── INPUTS ── */
.stTextInput input, .stSelectbox select {
    border: 1.5px solid rgba(0,51,153,0.20) !important;
    border-radius: 8px !important;
    font-family: Georgia, serif !important;
    font-size: 15px !important;
    transition: border-color 0.15s !important;
}
.stTextInput input:focus {
    border-color: #003399 !important;
    box-shadow: 0 0 0 3px rgba(0,51,153,0.10) !important;
}

/* ── AUTH CARD ── */
.auth-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 38px 44px;
    max-width: 460px;
    width: 100%;
    box-shadow: 0 20px 60px rgba(0,26,92,0.16);
    border-top: 4px solid #f5b400;
    margin: 0 auto;
}
.auth-title {
    font-family: "Playfair Display", serif !important;
    font-size: 26px !important;
    font-weight: 900 !important;
    color: #001a5c !important;
    text-align: center;
    margin-bottom: 4px !important;
}
.auth-subtitle {
    font-size: 14px !important;
    color: #003399 !important;
    text-align: center;
    font-style: italic;
    font-weight: 600 !important;
    margin-bottom: 20px !important;
}
.auth-divider {
    border: none;
    border-top: 1px solid rgba(0,51,153,0.12);
    margin: 18px 0;
}
.auth-link {
    font-size: 13px;
    color: #003399 !important;
    text-align: center;
    cursor: pointer;
    text-decoration: underline;
}

/* ── TOP NAV ── */
.top-nav {
    background: linear-gradient(90deg, #001a5c, #003399);
    border-radius: 16px;
    padding: 14px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 6px 24px rgba(0,26,92,0.30);
    border-bottom: 3px solid #f5b400;
}
.top-title {
    font-size: 22px;
    font-weight: 900;
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    letter-spacing: -0.01em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.top-logo-text {
    font-size: 13px;
    color: rgba(255,255,255,0.65) !important;
    font-style: italic;
    font-family: "EB Garamond", serif !important;
    margin-top: 2px;
}
.user-badge {
    background: rgba(245,180,0,0.22);
    border: 1.5px solid rgba(245,180,0,0.60);
    color: #ffd040 !important;
    font-size: 14px;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 999px;
    font-family: "EB Garamond", serif !important;
    white-space: nowrap;
    text-shadow: 0 1px 3px rgba(0,0,0,0.20);
}

/* ── PAGE HEADER ── */
.fbc-page-header {
    background: linear-gradient(135deg, #001a5c 0%, #003399 50%, #0044cc 100%);
    border-radius: 18px; padding: 26px 32px; margin-bottom: 24px;
    border-bottom: 3px solid #f5b400;
    box-shadow: 0 12px 40px rgba(0,26,92,0.28);
}
.fbc-page-header-title {
    font-family: "Playfair Display", serif !important;
    font-size: 26px !important; font-weight: 900 !important;
    color: #ffffff !important;
}
.fbc-page-header-sub {
    font-size: 14px; color: rgba(255,255,255,0.85) !important;
    margin-top: 6px; font-style: italic;
}

/* ── FEATURE CARDS ── */
.feature-box {
    background: #ffffff; padding: 20px 22px;
    border-radius: 14px; border-left: 6px solid #003399;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    transition: all 0.25s ease; margin-bottom: 14px;
}
.feature-box:hover {
    background: #f4f8ff;
    box-shadow: 0 8px 22px rgba(0,51,153,0.16);
    transform: translateY(-3px);
}
.feature-icon { font-size: 22px; margin-right: 8px; }

/* ── MISC ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6) !important;
    border: 1px solid rgba(0,51,153,0.12) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
}
.stDataFrame thead th { background: #003399 !important; color: white !important; }
.fbc-divider { border: none; border-top: 1px solid rgba(0,51,153,0.12); margin: 18px 0; }
.fbc-footer {
    text-align: center; padding: 22px; margin-top: 40px;
    color: #5a7099 !important; font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10);
}
.fbc-footer b { color: #003399 !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #003399; border-radius: 999px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important; font-weight: 600 !important;
    font-family: Georgia, serif !important; color: #5a7099 !important;
}
.stTabs [aria-selected="true"] { background: #003399 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════
LOGO_PATH = Path("assets") / "fbc log.png"

# ── Top nav bar ────────────────────────────────────────────────────
import base64 as _b64
_logo_tag = ""
if LOGO_PATH.exists():
    with open(str(LOGO_PATH), "rb") as _lf:
        _logo_b64 = _b64.b64encode(_lf.read()).decode()
    _logo_tag = f'<img src="data:image/png;base64,{_logo_b64}" style="height:90px; width:auto; object-fit:contain; border-radius:6px; margin-right:14px; flex-shrink:0;">'

st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #001233 0%, #001a5c 40%, #003399 100%);
        border-radius: 16px;
        padding: 14px 24px;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 6px 24px rgba(0,26,92,0.32);
        border-bottom: 3px solid #f5b400;">
        <div style="display:flex; align-items:center;">
            {_logo_tag}
            <div style="line-height:1.25;">
                <div style="
                    font-family: 'Playfair Display', serif;
                    font-size: 22px; font-weight: 900;
                    color: #ffffff;
                    letter-spacing: -0.01em;
                    text-shadow: 0 2px 8px rgba(0,0,0,0.30);">
                    🏛️ FBC Valuation Dashboard
                </div>
                <div style="
                    font-family: 'EB Garamond', serif;
                    font-size: 13px; font-style: italic;
                    color: rgba(255,255,255,0.65);
                    margin-top: 2px;">
                    Investment Research &amp; Valuation Platform
                </div>
            </div>
        </div>
    </div>
    <hr style="border:none; border-top:2px solid #dde6f5; margin:6px 0 20px 0;">
""", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────
st.markdown("""
    <div class="fbc-page-header">
        <span class="fbc-page-header-title">
            FBC Investment Valuation System 👋
        </span>
        <p class="fbc-page-header-sub">
            Select a valuation model below to get started.
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Quick-access buttons ──────────────────────────────────────────
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
colA, colB, colC, colD, colE, colF = st.columns(6)

if colA.button("📊 DCF Model",               use_container_width=True): st.switch_page("pages/1_DCF.py")
if colB.button("💰 Dividend Discount Model",  use_container_width=True): st.switch_page("pages/3_DDM.py")
if colC.button("📈 Comparables",             use_container_width=True): st.switch_page("pages/2_COMPARABLES.py")
if colD.button("🏦 Banking (RIM)",            use_container_width=True): st.switch_page("pages/4_BANKING.py")
if colE.button("🧾 Summary",                  use_container_width=True): st.switch_page("pages/5_SUMMARY.py")
if colF.button("🧭 User Guide",               use_container_width=True): st.switch_page("pages/6_USER_GUIDE.py")

st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ── Feature cards ─────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="feature-box">
        <span class="feature-icon">📊</span>
        <b>DCF Forecast + Valuation</b><br>
        Multi-year FCFF forecasting, WACC, terminal value and intrinsic equity value.
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <span class="feature-icon">💰</span>
        <b>Dividend Discount Model (DDM)</b><br>
        Gordon Growth + Multi-Stage with required equity returns.
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-box">
        <span class="feature-icon">📈</span>
        <b>Comparables Valuation</b><br>
        EV/EBITDA, P/E, and P/B multiple benchmarking.
    </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <span class="feature-icon">🏦</span>
        <b>Banking Valuation (Residual Income)</b><br>
        BVPS, residual income, terminal value and implied equity value.
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="fbc-footer">
    Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard
</div>
""", unsafe_allow_html=True)
