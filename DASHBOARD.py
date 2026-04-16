import streamlit as st
from pathlib import Path
def section(title: str):
    st.markdown(
        f"""
        <div class="fbc-section">
            <div class="fbc-section-dot"></div>
            <div class="fbc-section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FBC DESIGN SYSTEM ─────────────────────────────────────────
st.markdown("""
<style>
.top-title {
    color: black;
    font-size: 28px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)
st.markdown('''
<style>
/* ================================================================
   FBC INVESTMENT VALUATION SYSTEM — Design System v3.0
   ================================================================ */

/* ── 0. GOOGLE FONTS ──────────────────────────────────────── */
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

/* Headings — Playfair Display */
h1, h2, h3, h4, .fbc-heading, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: "Playfair Display", Georgia, serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}

/* ── 2. PAGE BACKGROUND ───────────────────────────────────── */
.stApp {
  background: #f5f7fb !important;
}
.main .block-container {
  background: #f5f7fb !important;
  padding-top: 1.5rem !important;
}

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

/* sidebar hr divider */
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

/* ── 5. PAGE HEADER BANNER ────────────────────────────────── */
.fbc-page-header {
    background: linear-gradient(135deg, #001a5c 0%, #003399 50%, #0044cc 100%);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 28px;
    border: 1px solid rgba(255,255,255,0.08);
    border-bottom: 3px solid #f5b400;
    box-shadow: 0 12px 40px rgba(0,26,92,0.28);
    position: relative;
    overflow: hidden;
}
.fbc-page-header::before {
    content: "";
    position: absolute;
    left: -40px; top: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,180,0,0.12), transparent 70%);
    pointer-events: none;
}
.fbc-page-header::after {
    content: "";
    position: absolute;
    right: -30px; bottom: -30px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.06), transparent 65%);
    pointer-events: none;
}
.fbc-page-header-icon {
    font-size: 30px;
    margin-right: 12px;
    vertical-align: middle;
}
.fbc-page-header-title {
    font-family: "Playfair Display", serif !important;
    font-size: 28px !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    display: inline !important;
    vertical-align: middle !important;
    letter-spacing: -0.01em !important;
}
.fbc-page-header-sub {
    font-size: 14px;
    color: rgba(255,255,255,0.78) !important;
    margin-top: 8px;
    font-style: italic;
}
.fbc-badge {
    display: inline-block;
    background: rgba(245,180,0,0.20);
    border: 1px solid rgba(245,180,0,0.50);
    color: #f5c842 !important;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    border-radius: 999px;
    margin-left: 10px;
    vertical-align: middle;
    text-transform: uppercase;
    font-family: "EB Garamond", serif !important;
}

/* ── 6. SECTION HEADINGS ──────────────────────────────────── */
.fbc-section-heading {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 32px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, #003399 0%, #f5b400 55%, transparent 90%) 1;
}
.fbc-section-heading-text {
    font-family: "Playfair Display", serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #001a5c !important;
    letter-spacing: 0.01em !important;
}
.fbc-section-heading-step {
    background: linear-gradient(135deg, #003399, #0044cc);
    color: white !important;
    font-size: 11px;
    font-weight: 900;
    min-width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 3px 8px rgba(0,51,153,0.35);
}
.fbc-subsection-heading {
    font-family: "Playfair Display", serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #003399 !important;
    margin: 20px 0 10px 0 !important;
    padding-left: 12px !important;
    border-left: 3px solid #f5b400 !important;
}

/* ── 7. CARD / PANEL ──────────────────────────────────────── */
.fbc-card {
    background: #ffffff;
    border: 1px solid rgba(0,51,153,0.09);
    border-left: 5px solid #003399;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,26,92,0.07);
    transition: box-shadow 0.2s, transform 0.2s;
}
.fbc-card:hover {
    box-shadow: 0 12px 32px rgba(0,51,153,0.14);
    transform: translateY(-2px);
}
.fbc-card h3, .fbc-card h4 {
    font-family: "Playfair Display", serif !important;
    color: #001a5c !important;
    margin: 0 0 10px 0 !important;
}
.fbc-card-gold {
    border-left-color: #f5b400 !important;
}
.fbc-card-green {
    border-left-color: #10b981 !important;
}

/* sub-card (nested) */
.fbc-subcard {
    background: rgba(0,51,153,0.03);
    border: 1px solid rgba(0,51,153,0.10);
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 12px;
}

/* ── 8. KPI METRIC CARDS ──────────────────────────────────── */
.fbc-kpi {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6);
    border: 1px solid rgba(0,51,153,0.12);
    border-radius: 16px;
    padding: 16px 18px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.fbc-kpi:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,51,153,0.12);
}
.fbc-kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #5a7099 !important;
    margin-bottom: 6px;
    font-family: "EB Garamond", serif !important;
}
.fbc-kpi-value {
    font-family: "Playfair Display", serif !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #001a5c !important;
    line-height: 1.2;
}
.fbc-kpi-unit {
    font-size: 11px;
    color: #7a90b8 !important;
    margin-top: 3px;
}

/* DCF card / kpi aliases */
.dcf-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-left: 5px solid #003399;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-top: 12px;
    margin-bottom: 16px;
}
.dcf-card h3 { margin: 0 0 10px 0; font-family: "Playfair Display", serif !important; color: #001a5c !important; }
.dcf-subcard {
    background: rgba(0,51,153,0.03);
    border: 1px solid rgba(0,51,153,0.10);
    border-radius: 14px;
    padding: 14px;
    margin-top: 10px;
}
.dcf-kpi {
    background: linear-gradient(135deg, rgba(0,51,153,0.08), rgba(245,180,0,0.08));
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 16px;
    padding: 14px 16px;
    margin: 6px 0;
    transition: transform 0.2s;
}
.dcf-kpi:hover { transform: translateY(-1px); }
.dcf-kpi-title { font-size: 11px; opacity: 0.7; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.07em; }
.dcf-kpi-value { font-size: 20px; font-weight: 800; font-family: "Playfair Display", serif !important; color: #001a5c !important; }

/* ── 9. RESET / UTILITY CARDS ─────────────────────────────── */
.fbc-reset-card {
    background: linear-gradient(135deg, #001a5c 0%, #003399 55%, #0055cc 100%);
    padding: 22px 28px;
    border-radius: 16px;
    color: white !important;
    box-shadow: 0 8px 24px rgba(0,26,92,0.30);
    margin-bottom: 22px;
    border-bottom: 3px solid #f5b400;
    position: relative;
    overflow: hidden;
}
.fbc-reset-card::after {
    content: "";
    position: absolute;
    right: -20px; top: -20px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,180,0,0.15), transparent 70%);
    pointer-events: none;
}
.fbc-reset-title {
    font-family: "Playfair Display", serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    margin-bottom: 6px;
    color: white !important;
}
.fbc-reset-sub { font-size: 14px; opacity: 0.88; margin-bottom: 14px; color: rgba(255,255,255,0.85) !important; }

/* ── 10. BUTTONS ──────────────────────────────────────────── */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: "EB Garamond", Georgia, serif !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0,51,153,0.28) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0044cc, #0055ee) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(0,51,153,0.38) !important;
}
.fbc-reset-btn .stButton > button {
    background: linear-gradient(135deg, #f5b400, #ffd040) !important;
    color: #001a5c !important;
    box-shadow: 0 4px 12px rgba(245,180,0,0.32) !important;
}
.fbc-reset-btn .stButton > button:hover {
    background: linear-gradient(135deg, #ffd040, #ffe070) !important;
    box-shadow: 0 8px 20px rgba(245,180,0,0.45) !important;
}
.fbc-nav-btn button,
.fbc-nav-btn .stButton > button {
    background: linear-gradient(135deg, #003399, #001a4d) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.25s ease-in-out !important;
}
.fbc-nav-btn button:hover,
.fbc-nav-btn .stButton > button:hover {
    background: linear-gradient(135deg, #0055cc, #003399) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 14px rgba(0,0,0,0.25) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0044cc, #0055ee) !important;
    transform: translateY(-1px) !important;
}

/* ── 11. INPUTS ───────────────────────────────────────────── */
.stNumberInput input, .stTextInput input {
    border: 1.5px solid rgba(0,51,153,0.18) !important;
    border-radius: 10px !important;
    font-family: "EB Garamond", Georgia, serif !important;
    background: #ffffff !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    padding: 8px 12px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #003399 !important;
    box-shadow: 0 0 0 3px rgba(0,51,153,0.10) !important;
}
.stSelectbox > div > div {
    border: 1.5px solid rgba(0,51,153,0.18) !important;
    border-radius: 10px !important;
    font-family: "EB Garamond", Georgia, serif !important;
}

/* ── 12. TABS ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid rgba(0,51,153,0.12);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 700 !important;
    font-family: "EB Garamond", Georgia, serif !important;
    color: #5a7099 !important;
    padding: 10px 20px !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important;
    box-shadow: 0 -2px 0 0 #f5b400 inset !important;
}

/* ── 13. EXPANDERS ────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: #001a5c !important;
    font-family: "Playfair Display", Georgia, serif !important;
    border-radius: 10px !important;
    background: rgba(0,51,153,0.04) !important;
    padding: 12px 16px !important;
    border-left: 3px solid #f5b400 !important;
}
/* Fix: hide raw "keyboard_arrow_right" text Streamlit injects into expander headers */
[data-testid="stExpander"] details > summary > div > p > span[style],
[data-testid="stExpander"] details > summary p span[data-testid="stMarkdownContainer"] > p > span,
details > summary p > span[style*="font-weight"] {
    display: none !important;
}
[data-testid="stExpanderToggleIcon"] svg,
details > summary svg {
    font-family: "Material Icons", "Material Symbols Outlined" !important;
}

/* ── 14. METRICS (st.metric) ──────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6) !important;
    border: 1px solid rgba(0,51,153,0.12) !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0,51,153,0.10) !important;
}
[data-testid="metric-container"] label {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #5a7099 !important;
    font-family: "EB Garamond", serif !important;
}
[data-testid="stMetricValue"] {
    font-family: "Playfair Display", serif !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #001a5c !important;
}
[data-testid="stMetricDelta"] {
    font-family: "EB Garamond", serif !important;
}

/* ── 15. ALERTS & INFO BOXES ──────────────────────────────── */
.stAlert {
    border-radius: 14px !important;
    font-family: "EB Garamond", Georgia, serif !important;
    border-left-width: 4px !important;
}

/* ── 16. PROGRESS BAR ─────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #003399, #0044cc, #f5b400) !important;
    border-radius: 999px !important;
}

/* ── 17. SCROLLBAR ────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f4ff; border-radius: 999px; }
::-webkit-scrollbar-thumb { background: linear-gradient(#003399, #0044cc); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #0055ee; }

/* ── 18. FOOTER ───────────────────────────────────────────── */
.fbc-footer {
    text-align: center;
    padding: 28px;
    margin-top: 48px;
    color: #5a7099 !important;
    font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10);
    font-style: italic;
}
.fbc-footer b { color: #003399 !important; font-style: normal; }

/* ── 19. FEATURE CARDS (dashboard) ───────────────────────── */
.feature-box {
    background: #ffffff;
    padding: 22px 26px;
    border-radius: 16px;
    border-left: 6px solid #003399;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07);
    transition: all 0.25s ease;
    margin-bottom: 16px;
    font-family: "EB Garamond", serif !important;
}
.feature-box:hover {
    background: linear-gradient(135deg, #f4f8ff, #fffdf0);
    box-shadow: 0 10px 28px rgba(0,51,153,0.16);
    transform: translateY(-3px);
    border-left-color: #f5b400;
}
.feature-icon { font-size: 24px; margin-right: 10px; }

/* ── 20. SMALL UTILITIES ──────────────────────────────────── */
.small-note {
    font-size: 12px;
    color: #7a90b8 !important;
    font-style: italic;
}
.gold-accent { color: #b87c00 !important; font-weight: 700; }
.blue-accent  { color: #003399 !important; font-weight: 700; }

/* ── 21. DATAFRAMES ───────────────────────────────────────── */
.stDataFrame thead th {
    background: linear-gradient(135deg, #001a5c, #003399) !important;
    color: white !important;
    font-weight: 700 !important;
    font-family: "Playfair Display", serif !important;
    letter-spacing: 0.02em !important;
}
.stDataFrame tbody tr:hover td {
    background: rgba(0,51,153,0.04) !important;
}
.stDataFrame tbody tr:nth-child(even) td {
    background: rgba(0,51,153,0.02) !important;
}

/* ── 22. TOP NAV (dashboard) ──────────────────────────────── */
.top-nav {
    position: fixed; top: 0; left: 0; width: 100%; height: 68px;
    background: linear-gradient(90deg, #001a5c, #003399, #0044cc);
    color: white; display: flex; align-items: center;
    padding: 0 28px; z-index: 99999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.30);
    border-bottom: 2px solid #f5b400;
}
.top-title {
    font-family: "Playfair Display", serif !important;
    font-size: 26px; font-weight: 900; margin-left: 16px; letter-spacing: -0.01em;
    color: Black;
}

/* ── 23. DIVIDER ──────────────────────────────────────────── */
.fbc-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #f5b400, #003399, transparent);
    border: none;
    margin: 28px 0;
    border-radius: 999px;
}

/* ── 24. RADIO / CHECKBOX ─────────────────────────────────── */
.stRadio > label, .stCheckbox > label {
    font-family: "EB Garamond", Georgia, serif !important;
    color: #1a1a2e !important;
}

/* ── 25. SELECTBOX OPTIONS ────────────────────────────────── */
div[data-baseweb="select"] span {
    font-family: "EB Garamond", Georgia, serif !important;
}

/* ── 26. PEER PICKER (Comparables) ───────────────────────── */
.peer-picker-wrap {
    border: 1px solid rgba(0,51,153,0.14);
    border-radius: 18px;
    padding: 18px 18px 12px 18px;
    background: #f8fbff;
    box-shadow: 0 6px 20px rgba(0,26,92,0.07);
    margin-top: 12px;
    margin-bottom: 14px;
}
.peer-picker-head {
    font-family: "Playfair Display", serif !important;
    font-size: 17px;
    font-weight: 700;
    color: #001a5c;
    margin-bottom: 8px;
}
.peer-picker-sub {
    font-size: 13px;
    color: #475569;
    margin-bottom: 14px;
    font-style: italic;
}

/* ── 27. SENS TABLE OVERRIDES ─────────────────────────────── */
.sens-outer { border: 2px solid rgba(0,51,153,0.15) !important; border-radius: 16px !important; }
.sens-table { font-family: "EB Garamond", Georgia, serif !important; }
.sens-table thead th { background: #001a5c !important; }

/* ── 28. FOOTER BANNER ────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 24px;
    margin-top: 40px;
    color: #5a7099 !important;
    font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10);
    font-style: italic;
}
.footer b { color: #003399 !important; font-style: normal; }

</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="fbc-page-header">
    <span class="fbc-page-header-icon">🏦</span>
    <span class="fbc-page-header-title">FBC Valuation Dashboard</span>
    <span class="fbc-badge">FBC Securities</span>
    <div class="fbc-page-header-sub">Investment Research & Valuation System — FBC Securities</div>
</div>
''', unsafe_allow_html=True)
st.markdown("""
<style>

/* =========================================================
   FBC CLEAN SECTION HEADER (NO SUBTITLE, NO STEPS)
   ========================================================= */

.fbc-section {
    display: flex;
    align-items: center;
    gap: 14px;

    padding: 16px 22px;
    margin: 28px 0 18px 0;

    background: linear-gradient(
        135deg,
        rgba(0, 51, 153, 0.08),
        rgba(245, 180, 0, 0.05)
    );

    border-left: 6px solid #003399;
    border-radius: 14px;

    box-shadow: 0 4px 14px rgba(0, 26, 92, 0.08);
}

/* Left indicator (circle like your UI) */
.fbc-section {
    display: block;
    padding: 16px 0;
    margin: 28px 0 18px 0;
    border-bottom: 2px solid rgba(0,51,153,0.15);
    transition: all 0.25s ease;
}

/* animated underline */
.fbc-section-title::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -6px;
    width: 40px;
    height: 3px;
    background: #003399;
    transition: width 0.3s ease;
}

.fbc-section:hover .fbc-section-title::after {
    width: 100%;
}

/* Title only */
.fbc-section-title {
    font-family: "Playfair Display", serif !important;
    font-size: 21px;
    font-weight: 700;
    font-style: italic;
    color: #001a5c !important;
    letter-spacing: -0.01em;
    position: relative;
}

/* Gold underline */
.fbc-section-title::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -6px;
    width: 45px;
    height: 3px;

    background: linear-gradient(90deg, #f5b400, #ffcc33);
    border-radius: 2px;

    transition: width 0.35s ease;
}

/* Smooth expand on hover */
.fbc-section:hover .fbc-section-title::after {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────


# ------------------------------------------------------------
# STYLES (FIX SIDEBAR ICON + NICE COLLAPSE BUTTON)
# ------------------------------------------------------------
CUSTOM_STYLE = """
<style>
/* ================================================================
   FBC DESIGN SYSTEM  v2.0
   ================================================================ */

/* ── 1. FONTS ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small,
.stDataFrame, .stTable {
  font-family: Georgia, "Times New Roman", serif !important;
}

/* ── 2. SIDEBAR ───────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #002080 0%, #001040 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.12) !important;
    backdrop-filter: blur(10px) !important;
}
section[data-testid="stSidebar"] * { color: #e8f0ff !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    letter-spacing: 0.02em !important;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem !important;
}

/* sidebar nav links */
section[data-testid="stSidebar"] a {
    color: #b8d0ff !important;
    transition: color 0.15s !important;
}
section[data-testid="stSidebar"] a:hover {
    color: #f5b400 !important;
}

/* ── 3. SIDEBAR COLLAPSE BUTTON ───────────────────────── */
.material-icons,
span.material-icons,
i.material-icons,
.material-symbols-outlined,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapseButton"] i {
    font-family: "Material Icons", "Material Symbols Outlined" !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 50% !important;
    width: 44px !important; height: 44px !important;
    box-shadow: 0 4px 14px rgba(0,51,153,0.45) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: linear-gradient(135deg, #0044cc, #0055ee) !important;
    transform: translateY(-1px) scale(1.05) !important;
    box-shadow: 0 8px 20px rgba(0,51,153,0.5) !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    width: 22px !important; height: 22px !important;
    fill: white !important;
}

/* ── 4. PAGE HEADER BANNER ────────────────────────────── */
.fbc-page-header {
    background: linear-gradient(135deg, #002080 0%, #003399 55%, #0044cc 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 30px rgba(0,32,128,0.30);
    position: relative;
    overflow: hidden;
}
.fbc-page-header::after {
    content: "";
    position: absolute;
    right: -30px; top: -30px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,180,0,0.18), transparent 65%);
    pointer-events: none;
}
.fbc-page-header-icon {
    font-size: 28px;
    margin-right: 10px;
    vertical-align: middle;
}
.fbc-page-header-title {
    font-size: 26px;
    font-weight: 800;
    color: #ffffff !important;
    display: inline;
    vertical-align: middle;
}
.fbc-page-header-sub {
    font-size: 14px;
    color: rgba(255,255,255,0.78) !important;
    margin-top: 6px;
}
.fbc-badge {
    display: inline-block;
    background: rgba(245,180,0,0.20);
    border: 1px solid rgba(245,180,0,0.45);
    color: #f5c842 !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    border-radius: 999px;
    margin-left: 10px;
    vertical-align: middle;
    text-transform: uppercase;
}

/* ── 5. SECTION HEADINGS ──────────────────────────────── */
.fbc-section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, #003399, #f5b400 60%, transparent) 1;
}
.fbc-section-heading-text {
    font-size: 17px;
    font-weight: 700;
    color: #002266 !important;
    letter-spacing: 0.01em;
}
.fbc-section-heading-step {
    background: #003399;
    color: white !important;
    font-size: 12px;
    font-weight: 800;
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

/* ── 6. CARD / PANEL ──────────────────────────────────── */
.fbc-card {
    background: #ffffff;
    border: 1px solid rgba(0,51,153,0.10);
    border-left: 5px solid #003399;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s, transform 0.2s;
}
.fbc-card:hover {
    box-shadow: 0 8px 24px rgba(0,51,153,0.14);
    transform: translateY(-2px);
}
.fbc-card h3, .fbc-card h4 {
    color: #002266 !important;
    margin: 0 0 8px 0;
}

/* sub-card (nested) */
.fbc-subcard {
    background: rgba(0,51,153,0.03);
    border: 1px solid rgba(0,51,153,0.10);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 10px;
}

/* ── 7. KPI METRIC CARDS ──────────────────────────────── */
.fbc-kpi {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6);
    border: 1px solid rgba(0,51,153,0.12);
    border-radius: 14px;
    padding: 14px 16px;
    text-align: center;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
.fbc-kpi-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a7099 !important;
    margin-bottom: 4px;
}
.fbc-kpi-value {
    font-size: 22px;
    font-weight: 800;
    color: #002266 !important;
    line-height: 1.2;
}
.fbc-kpi-unit {
    font-size: 12px;
    color: #7a90b8 !important;
    margin-top: 2px;
}

/* ── 8. RESET / UTILITY CARDS ─────────────────────────── */
.fbc-reset-card {
    background: linear-gradient(135deg, #003399 0%, #0055cc 100%);
    padding: 20px 24px;
    border-radius: 14px;
    color: white !important;
    box-shadow: 0 6px 18px rgba(0,51,153,0.25);
    margin-bottom: 20px;
}
.fbc-reset-title  { font-size: 18px; font-weight: 700; margin-bottom: 4px; color: white !important; }
.fbc-reset-sub    { font-size: 13px; opacity: 0.88; margin-bottom: 12px; color: white !important; }

/* ── 9. BUTTONS ───────────────────────────────────────── */
/* Primary action buttons */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: Georgia, serif !important;
    padding: 9px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 10px rgba(0,51,153,0.25) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0044cc, #0055ee) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(0,51,153,0.35) !important;
}

/* Gold reset/generate button override */
.fbc-reset-btn .stButton > button {
    background: linear-gradient(135deg, #f5b400, #ffd040) !important;
    color: #002266 !important;
    box-shadow: 0 3px 10px rgba(245,180,0,0.30) !important;
}
.fbc-reset-btn .stButton > button:hover {
    background: linear-gradient(135deg, #ffd040, #ffe070) !important;
    box-shadow: 0 6px 18px rgba(245,180,0,0.40) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #003399, #0044cc) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0044cc, #0055ee) !important;
    transform: translateY(-1px) !important;
}

/* ── 10. INPUTS ───────────────────────────────────────── */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    border: 1px solid rgba(0,51,153,0.20) !important;
    border-radius: 8px !important;
    font-family: Georgia, serif !important;
    transition: border-color 0.15s !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #003399 !important;
    box-shadow: 0 0 0 2px rgba(0,51,153,0.12) !important;
}

/* ── 11. TABS ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid rgba(0,51,153,0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0 !important;
    font-weight: 600 !important;
    font-family: Georgia, serif !important;
    color: #5a7099 !important;
    padding: 8px 18px !important;
    transition: all 0.15s !important;
}
.stTabs [aria-selected="true"] {
    background: #003399 !important;
    color: white !important;
}

/* ── 12. EXPANDERS ────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: #002266 !important;
    font-family: Georgia, serif !important;
    border-radius: 8px !important;
    background: rgba(0,51,153,0.04) !important;
    padding: 10px 14px !important;
}

/* ── 13. METRICS (st.metric) ──────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #f0f5ff, #fff8e6) !important;
    border: 1px solid rgba(0,51,153,0.12) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05) !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #5a7099 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #002266 !important;
}

/* ── 14. ALERTS & INFO BOXES ──────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    font-family: Georgia, serif !important;
}

/* ── 15. SCROLLBAR ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0f4ff; border-radius: 999px; }
::-webkit-scrollbar-thumb { background: #003399; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #0044cc; }

/* ── 16. FOOTER ───────────────────────────────────────── */
.fbc-footer {
    text-align: center;
    padding: 22px;
    margin-top: 40px;
    color: #5a7099 !important;
    font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10);
}
.fbc-footer b { color: #003399 !important; }

/* ── 17. FEATURE CARDS (dashboard) ───────────────────── */
.feature-box {
    background: #ffffff;
    padding: 20px 22px;
    border-radius: 14px;
    border-left: 6px solid #003399;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    transition: all 0.25s ease;
    margin-bottom: 14px;
}
.feature-box:hover {
    background: #f4f8ff;
    box-shadow: 0 8px 22px rgba(0,51,153,0.16);
    transform: translateY(-3px);
}
.feature-icon {
    font-size: 22px;
    margin-right: 8px;
}

/* ── 18. SMALL UTILITIES ──────────────────────────────── */
.small-note {
    font-size: 12px;
    color: #7a90b8 !important;
    font-style: italic;
}
.gold-accent { color: #c48a00 !important; font-weight: 700; }
.blue-accent  { color: #003399 !important; font-weight: 700; }

/* ── 19. DATAFRAMES ───────────────────────────────────── */
.stDataFrame thead th {
    background: #003399 !important;
    color: white !important;
    font-weight: 700 !important;
}
.stDataFrame tbody tr:hover td {
    background: rgba(0,51,153,0.04) !important;
}

/* ── 20. TOP NAV (dashboard) ──────────────────────────── */
.top-nav {
    position: fixed; top: 0; left: 0; width: 100%; height: 64px;
    background: linear-gradient(90deg, #002080, #003399);
    color: white; display: flex; align-items: center;
    padding: 0 28px; z-index: 99999;
    box-shadow: 0 3px 12px rgba(0,0,0,0.30);
}
.top-title {
    font-size: 24px; 
    font-weight: 800; 
    margin-left: 14px; 
    letter-spacing: 0.01em;
    color: black !important;
    font-family: "Playfair Display", serif !important;
}

</style>
"""

st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

# ------------------------------------------------------------
# TOP NAVIGATION BAR
# ------------------------------------------------------------
LOGO_PATH = Path("assets") / "fbc log.png"

st.markdown("<div class='top-nav'>", unsafe_allow_html=True)

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=200)

st.markdown("<span class='top-title'>FBC Valuation Dashboard</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Navigation")
    st.markdown("""
    - 📊 DCF Model  
    - 💰 DDM  
    - 📈 Comparables  
    - 🏦 Banking (RIM)  
    - 🧾 Summary  
    - 🧭 USER GUIDE
    """)

# ------------------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------------------


section("Welcome to the FBC Investment Valuation System")
section("Explore valuation models using the sidebar or the quick-access buttons below.")
st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
colA, colB, colC, colD, colE, colF = st.columns(6)

if colA.button("📊 DCF Model", use_container_width=True):
    st.switch_page("pages/1_DCF.py")

if colB.button("💰 Dividend Discount Model", use_container_width=True):
    st.switch_page("pages/3_DDM.py")

if colC.button("📈 Comparables", use_container_width=True):
    st.switch_page("pages/2_COMPARABLES.py")

if colD.button("🏦 Banking (RIM)", use_container_width=True):
    st.switch_page("pages/4_BANKING.py")

if colE.button("🧾 Summary", use_container_width=True):
    st.switch_page("pages/5_SUMMARY.py")

if colF.button("🧭 User Guide", use_container_width=True):
    st.switch_page("pages/6_USER GUIDE.py")


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

st.markdown("</div>", unsafe_allow_html=True)
# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
<div class="footer">
    Powered by <b>FBC Securities</b> • Investment Research & Valuation Dashboard
</div>
""", unsafe_allow_html=True)
