import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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
# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Summary Valuation", layout="wide")

# ─── FBC DESIGN SYSTEM ─────────────────────────────────────────
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
p, div, label, span,
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

/* ── 4. SIDEBAR COLLAPSE BUTTON ───────────────────────────── */
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
    color: white;
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
    <span class="fbc-page-header-icon">🧾</span>
    <span class="fbc-page-header-title">Summary Valuation</span>
    <span class="fbc-badge">FBC Securities</span>
    <div class="fbc-page-header-sub">Consolidated view of all valuation outputs across models.</div>
</div>
''', unsafe_allow_html=True)
# ────────────────────────────────────────────────────────────────

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

# ------------------------------------------------------------------------------
# POWERBI DARK THEME (FBC TUNED)
# ------------------------------------------------------------------------------
DARK_BG = "#020617"        # page background
PANEL_BG = "#020c1f"       # panels
CARD_BG = "#071525"        # KPI cards
PRIMARY_TEXT = "#e5f2ff"
MUTED_TEXT = "#9ca3af"
ACCENT_BLUE = "#38bdf8"
ACCENT_CYAN = "#0ea5e9"
ACCENT_GOLD = "#fbbf24"
DANGER = "#f97373"

st.markdown(
    f"""
    <style>

    /* ------------------------------------------------------ */
    /* GLOBAL BACKGROUND & TEXT                               */
    /* ------------------------------------------------------ */
    .main {{
        background: radial-gradient(circle at top left, #0d1424 0, {DARK_BG} 60%, #000 100%);
        color: {PRIMARY_TEXT};
    }}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}
h1, h2, h3, h4, h5 {{
    color: #000000 !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-weight: 600 !important;
}}


    /* ------------------------------------------------------ */
    /* TITLE BANNER                                           */
    /* ------------------------------------------------------ */
    .title-banner {{
        background: linear-gradient(90deg, #071426, #0a1b33 50%, #0d243f 100%);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 50px rgba(0,0,0,0.6);
    }}
    .title-main {{
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff !important;
    }}
    .title-sub {{
        font-size: 0.95rem;
        color: #d3e2f5 !important;
        margin-top: 3px;
    }}

    /* ------------------------------------------------------ */
    /* KPI CARDS — HIGH VISIBILITY VERSION                    */
    /* ------------------------------------------------------ */
    .kpi-card {{
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        background: rgba(10, 20, 40, 0.85); /* darker, solid */
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 12px 30px rgba(0,0,0,0.7);
        backdrop-filter: blur(4px); /* gentle */
    }}
    .kpi-title {{
        font-size: 0.85rem;
        color: #c7d4e8; /* brighter */
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .kpi-value {{
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: {ACCENT_GOLD} !important;
        text-shadow: 0 0 6px rgba(0,0,0,0.6);
    }}
    .kpi-sub {{
        font-size: 0.83rem;
        color: #b4c2d6 !important; /* more visible */
        margin-top: 2px;
    }}

    /* ------------------------------------------------------ */
    /* GLASS PANEL                                            */
    /* ------------------------------------------------------ */
    .glass-panel {{
        background: rgba(14,22,40,0.92);
        border-radius: 16px;
        padding: 1rem 1.3rem;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 10px 28px rgba(0,0,0,0.75);
        backdrop-filter: blur(3px);
    }}

    /* ------------------------------------------------------ */
    /* TABLE VISIBILITY FIX                                   */
    /* ------------------------------------------------------ */
    .stDataFrame, .stTable {{
        color: #f0f6ff !important;  /* TEXT FIX */
    }}

    .stDataFrame tbody td {{
        color: #1f2937 !important;  /* brighter cell text */
        font-size: 0.95rem !important;
    }}

    .stDataFrame thead th {{
    color: #0f172a !important;   
    font-weight: 700 !important;
    }}


    .stDataFrame tbody tr:hover {{
        background-color: rgba(255,255,255,0.08) !important;
    }}

    /* ------------------------------------------------------ */
    /* TABS – HIGH VISIBILITY                                 */
    /* ------------------------------------------------------ */
button[data-baseweb="tab"] {{
    font-size: 0.9rem;
    color: #1e293b !important;   
    background-color: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.4rem 1rem;
    border: 1px solid rgba(255,255,255,0.15);
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: #ffffff !important;
    background: #1e3a8a !important;   
    border: 1px solid #1e40af;
    box-shadow: 0 0 8px rgba(30,58,138,0.6);
}}


    /* ------------------------------------------------------ */
    /* INPUTS / SELECTS                                       */
    /* ------------------------------------------------------ */
    .stNumberInput input, .stSelectbox select {{
        background: rgba(20,34,60,0.95) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# TITLE BANNER
# ------------------------------------------------------------------------------
st.markdown(
    """
    <div class="title-banner">
        <div class="title-main">📘 Summary Valuation – Weighted Equity Value</div>
        <div class="title-sub">
            FBC dashboard summarising DCF · DDM · EV/EBITDA · PBV · P/E · Banking valuations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# INITIALISE SESSION STATE (PERSISTENT)
# ------------------------------------------------------------------------------
if "selected_models" not in st.session_state:
    st.session_state["selected_models"] = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E", "BANKING"]

if "model_weights" not in st.session_state:
    st.session_state["model_weights"] = {
        "DCF": 35.0,
        "DDM": 20.0,
        "EV/EBITDA": 15.0,
        "PBV": 10.0,
        "P/E": 10.0,
        "BANKING": 10.0,
    }

if "num_shares" not in st.session_state:
    st.session_state["num_shares"] = 0.0

if "current_price" not in st.session_state:
    st.session_state["current_price"] = 0.0

# ------------------------------------------------------------------------------
# MODEL SELECTION (PERSISTENT)
# ------------------------------------------------------------------------------
st.header("📌 Select Models to Include")

all_models = ["DCF", "DDM", "EV/EBITDA", "PBV", "P/E", "BANKING"]

selected_models = st.multiselect(
    "Choose models:",
    options=all_models,
    default=st.session_state["selected_models"],
    key="selected_models_input",
)
st.session_state["selected_models"] = selected_models

if not selected_models:
    st.warning("Please select at least one model.")
    st.stop()

# ------------------------------------------------------------------------------
# RETRIEVE VALUATIONS FROM OTHER PAGES (DCF, DDM, COMPARABLES)
# ------------------------------------------------------------------------------
value_map = {
    "DCF": st.session_state.get("equity_value_dcf"),
    "DDM": st.session_state.get("equity_value_ddm"),
    "EV/EBITDA": st.session_state.get("value_ev_ebitda"),
    "PBV": st.session_state.get("value_pbv"),
    "P/E": st.session_state.get("value_pe"),
    "BANKING": st.session_state.get("equity_value_banking"),
}

# ------------------------------------------------------------------------------
# WEIGHT ASSIGNMENT (PERSISTENT)
# ------------------------------------------------------------------------------
st.header("🧮 Assign Weights (%)")

cols = st.columns(len(all_models))
weights_new = {}

for model, col in zip(all_models, cols):
    if model in selected_models:
        with col:
            new_val = st.number_input(
                f"{model} Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state["model_weights"].get(model, 0)),
                step=1.0,
                key=f"weight_input_{model}",
            )
        weights_new[model] = new_val
    else:
        weights_new[model] = 0.0

st.session_state["model_weights"] = weights_new

total_w = sum(weights_new[m] for m in selected_models)
if total_w == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

weights_normalized = {m: (weights_new[m] / total_w) * 100 for m in selected_models}

# ------------------------------------------------------------------------------
# SUMMARY DATAFRAME
# ------------------------------------------------------------------------------
rows = []
for model in selected_models:
    val = value_map.get(model)
    w = weights_normalized.get(model, 0)
    weighted_value = val * (w / 100) if val is not None else None
    rows.append([model, val, w, weighted_value])

df_summary = pd.DataFrame(
    rows, columns=["Model", "Value (USD)", "Weight (%)", "Weighted Value"]
)

weighted_equity = df_summary["Weighted Value"].sum()

# Extra stats for dashboard
valid_vals = df_summary["Value (USD)"].dropna()
if not valid_vals.empty:
    max_model = df_summary.loc[df_summary["Value (USD)"].idxmax(), "Model"]
    max_val = float(valid_vals.max())
    min_val = float(valid_vals.min())
    spread = max_val - min_val
    dispersion = float(valid_vals.std()) if len(valid_vals) > 1 else 0.0
else:
    max_model, max_val, min_val, spread, dispersion = "-", 0.0, 0.0, 0.0, 0.0

# ------------------------------------------------------------------------------
# KPI STRIP (POWERBI STYLE)
# ------------------------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Weighted Equity Value</div>
            <div class="kpi-value">{weighted_equity:,.0f}</div>
            <div class="kpi-sub">Total blended equity output</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Models</div>
            <div class="kpi-value">{len(selected_models)}</div>
            <div class="kpi-sub">DCF / DDM / Multiples / Banking in use</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with kpi_col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Highest Model</div>
            <div class="kpi-value">{max_model}</div>
            <div class="kpi-sub">Range: {min_val:,.0f} – {max_val:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# TABS: SUMMARY TABLE | INTERACTIVE DASHBOARD
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Summary Table", "📈 Interactive Dashboard"])

# -------- TAB 1: SUMMARY TABLE --------
with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)

    st.dataframe(
        df_summary.style.format(
            {
                "Value (USD)": lambda x: f"{x:,.2f}" if pd.notnull(x) else "—",
                "Weight (%)": lambda x: f"{x:.0f}%",
                "Weighted Value": lambda x: f"{x:,.2f}" if pd.notnull(x) else "—",
            }
        ),
        width='stretch',
    )

# -------- TAB 2: INTERACTIVE DASHBOARD --------
with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### Model Comparison Dashboard")

    df_chart = df_summary.copy()

    base = (
        alt.Chart(df_chart)
        .properties(background=DARK_BG, height=260)
        .configure_axis(
            labelColor=PRIMARY_TEXT,
            titleColor=PRIMARY_TEXT,
            gridColor="#1f2937",
        )
        .configure_view(strokeOpacity=0)
        .configure_legend(labelColor=PRIMARY_TEXT, titleColor=PRIMARY_TEXT)
    )

    chart_values = (
        base.mark_bar(color=ACCENT_BLUE)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Value (USD):Q", title="Equity Value (USD)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Value (USD):Q", format=",.2f"),
            ],
        )
        .properties(title="Model Equity Values")
    )

    chart_weights = (
        base.mark_bar(color=ACCENT_GOLD)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Weight (%):Q", title="Weight (%)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Weight (%):Q", format=".1f"),
            ],
        )
        .properties(title="Model Weights")
    )

    chart_weighted = (
        base.mark_bar(color=ACCENT_CYAN)
        .encode(
            x=alt.X("Model:N", title="Model"),
            y=alt.Y("Weighted Value:Q", title="Weighted Equity (USD)"),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("Weighted Value:Q", format=",.2f"),
            ],
        )
        .properties(title="Weighted Contribution by Model", height=280)
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(chart_values, width='stretch')
    with c2:
        st.altair_chart(chart_weights, width='stretch')

    st.altair_chart(chart_weighted, width='stretch')

    st.markdown("</div>", unsafe_allow_html=True)
# ------------------------------------------------------------------------------
# 📌 GENERAL VALUATION SUMMARY TABLE (DOWNLOADABLE)
# ------------------------------------------------------------------------------
st.header("📌 Valuation Summary")

# ---- Inputs (unchanged logic) ----
c1, c2 = st.columns(2)

with c1:
    num_shares = st.number_input(
        "Number of Shares in Issue",
        value=float(st.session_state["num_shares"]),
        step=1000.0,
        format="%.0f",
        key="num_shares_input",
    )
    st.session_state["num_shares"] = num_shares

with c2:
    current_price = st.number_input(
        "Current Share Price (USD)",
        value=float(st.session_state["current_price"]),
        step=0.01,
        format="%.2f",
        key="current_price_input",
    )
    st.session_state["current_price"] = current_price

# ---- Calculations ----
intrinsic_value = (weighted_equity / num_shares) if (num_shares and num_shares > 0) else np.nan

if not np.isnan(intrinsic_value) and current_price > 0:
    upside_pct = (intrinsic_value - current_price) / current_price * 100
else:
    upside_pct = np.nan
# ---- Recommendation Logic (Buy / Hold / Reduce) ----
rec_label = "N/A"
rec_color = "#94a3b8"  # neutral grey
rec_reason = "Enter shares (>0) and current price (>0) to get a recommendation."

if (not np.isnan(intrinsic_value)) and (current_price > 0) and (not np.isnan(upside_pct)):

    # thresholds you can adjust
    BUY_TH = 10.0      # >= +15% upside => Buy/Accumulate
    HOLD_LOW = -10.0   # between -10% and +10% => Hold
    HOLD_HIGH = 10.0

    if upside_pct >= BUY_TH:
        rec_label = "🟢 BUY / ACCUMULATE"
        rec_color = "#22c55e"
        rec_reason = (
            f"Intrinsic value ({intrinsic_value:,.4f}) is above market price "
            f"({current_price:,.2f}), implying upside of +{upside_pct:.1f}%."
        )
    elif HOLD_LOW <= upside_pct <= HOLD_HIGH:
        rec_label = "🟡 HOLD / FAIRLY VALUED"
        rec_color = "#fbbf24"
        rec_reason = (
            f"Intrinsic value ({intrinsic_value:,.4f}) is close to market price "
            f"({current_price:,.2f}), implying limited upside/downside ({upside_pct:+.1f}%)."
        )
    else:
        rec_label = "🔴 REDUCE / SELL"
        rec_color = "#f97373"
        rec_reason = (
            f"Intrinsic value ({intrinsic_value:,.4f}) is below market price "
            f"({current_price:,.2f}), implying downside of {upside_pct:.1f}%."
        )

# ---- Display Recommendation (simple + clean) ----
st.markdown(
    f"""
    <div class="glass-panel" style="border-left: 6px solid {rec_color}; margin-top: 12px;">
        <div style="font-size: 0.85rem; color: #c7d4e8; text-transform: uppercase; letter-spacing: 0.06em;">
            Recommendation
        </div>
        <div style="font-size: 1.25rem; font-weight: 800; color: {rec_color}; margin-top: 4px;">
            {rec_label}
        </div>
        <div style="font-size: 0.9rem; color: #b4c2d6; margin-top: 6px;">
            {rec_reason}
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 8px;">
            Note: Based on blended valuation assumptions; sensitive to WACC, growth, and model weights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Build general summary table ----
summary_rows = [
    ["Weighted Equity Value", weighted_equity, "USD"],
    ["Number of Shares", num_shares, "Shares"],
    ["Intrinsic Value per Share", intrinsic_value, "USD"],
    ["Current Share Price", current_price, "USD"],
    ["Upside / Downside (%)", upside_pct, "%"],
]

df_valuation_summary = pd.DataFrame(
    summary_rows,
    columns=["Metric", "Value", "Unit"]
)

# ---- Formatting for display ----
def format_value(row):
    if pd.isna(row["Value"]):
        return "—"
    if row["Unit"] == "USD" and row["Metric"] == "Intrinsic Value per Share":
        return f"{row['Value']:,.4f}"
    if row["Unit"] == "USD":
        return f"{row['Value']:,.2f}"
    if row["Unit"] == "%":
        sign = "+" if row["Value"] >= 0 else ""
        return f"{sign}{row['Value']:.1f}%"
    if row["Unit"] == "Shares":
        return f"{row['Value']:,.0f}"
    return str(row["Value"])

df_display = df_valuation_summary.copy()
df_display["Value"] = df_display.apply(format_value, axis=1)

# ---- Display table ----
def highlight_upside(row):
    if row["Metric"] == "Upside / Downside (%)":
        try:
            val = float(row["Value"].replace("%", "").replace("+", ""))
            if val < 0:
                return ["", "color: #f97373;", ""]
            else:
                return ["", "color: #22c55e;", ""]
        except:
            return ["", "", ""]
    return ["", "", ""]

styled_table = df_display.style.apply(highlight_upside, axis=1)

st.dataframe(styled_table, width="stretch", hide_index=True)

# ---- Download button (Excel with formulas) ----
import io
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, numbers
from openpyxl.utils import get_column_letter

def build_summary_excel_with_formulas(
    selected_models,
    value_map,
    weights_new,
    num_shares,
    current_price
) -> bytes:
    wb = Workbook()

    # -----------------------------
    # Sheet 1: Model Summary
    # -----------------------------
    ws1 = wb.active
    ws1.title = "Model_Summary"

    headers = ["Model", "Value_USD", "Weight_Input_%", "Weight_Normalized_%", "Weighted_Value_USD"]
    ws1.append(headers)

    header_fill = PatternFill("solid", fgColor="0A1B33")
    header_font = Font(bold=True, color="FFFFFF")
    for col_idx, h in enumerate(headers, start=1):
        c = ws1.cell(row=1, column=col_idx)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center")

    # write model rows
    start_row = 2
    for i, m in enumerate(selected_models):
        r = start_row + i
        ws1.cell(r, 1, m)

        # Value
        v = value_map.get(m)
        ws1.cell(r, 2, float(v) if v is not None else None)

        # Raw weight input
        w_in = float(weights_new.get(m, 0.0))
        ws1.cell(r, 3, w_in)

        # Weight normalized = C / SUM(C range) * 100
        # We'll compute SUM range based on number of models
        # Example: =C2/SUM($C$2:$C$6)*100
        end_row = start_row + len(selected_models) - 1
        ws1.cell(r, 4, f"=C{r}/SUM($C${start_row}:$C${end_row})*100")

        # Weighted value = Value * WeightNormalized / 100
        ws1.cell(r, 5, f"=B{r}*D{r}/100")

    # Totals row
    total_row = start_row + len(selected_models)
    ws1.cell(total_row, 1, "TOTAL / WEIGHTED EQUITY")
    ws1.cell(total_row, 1).font = Font(bold=True)

    # Weighted equity = SUM(weighted values)
    ws1.cell(total_row, 5, f"=SUM($E${start_row}:$E${total_row-1})")
    ws1.cell(total_row, 5).font = Font(bold=True)

    # formatting
    for r in range(start_row, total_row + 1):
        ws1.cell(r, 2).number_format = '#,##0.00'
        ws1.cell(r, 3).number_format = '0.00'
        ws1.cell(r, 4).number_format = '0.00'
        ws1.cell(r, 5).number_format = '#,##0.00'

    # adjust column widths
    for col in range(1, 6):
        ws1.column_dimensions[get_column_letter(col)].width = 22

    # -----------------------------
    # Sheet 2: Valuation Summary
    # -----------------------------
    ws2 = wb.create_sheet("Valuation_Summary")

    ws2_headers = ["Metric", "Value", "Unit"]
    ws2.append(ws2_headers)
    for col_idx, h in enumerate(ws2_headers, start=1):
        c = ws2.cell(row=1, column=col_idx)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center")

    # Put inputs and formulas:
    # Weighted Equity Value comes from Model_Summary!E{total_row}
    # Shares and current price are typed values
    # Intrinsic = WeightedEquity / Shares
    # Upside% = (Intrinsic - CurrentPrice) / CurrentPrice

    r = 2
    ws2.cell(r, 1, "Weighted Equity Value")
    ws2.cell(r, 2, f"=Model_Summary!E{total_row}")
    ws2.cell(r, 3, "USD")

    r += 1
    ws2.cell(r, 1, "Number of Shares")
    ws2.cell(r, 2, float(num_shares) if num_shares is not None else 0.0)
    ws2.cell(r, 3, "Shares")

    r += 1
    ws2.cell(r, 1, "Intrinsic Value per Share")
    # =B2 / B3  (weighted equity / shares)
    ws2.cell(r, 2, f"=IF(B3>0,B2/B3,NA())")
    ws2.cell(r, 3, "USD")

    r += 1
    ws2.cell(r, 1, "Current Share Price")
    ws2.cell(r, 2, float(current_price) if current_price is not None else 0.0)
    ws2.cell(r, 3, "USD")

    r += 1
    ws2.cell(r, 1, "Upside / Downside (%)")
    # =(Intrinsic - Price)/Price
    ws2.cell(r, 2, f"=IF(B5>0,(B4-B5)/B5,NA())")
    ws2.cell(r, 3, "%")

    r += 1
    ws2.cell(r, 1, "Recommendation")
    # Same thresholds as your Streamlit logic:
    # BUY if upside >= 0.15
    # HOLD if -0.10 <= upside <= 0.10
    # else REDUCE
    ws2.cell(r, 2, '=IF(ISNA(B6),"N/A",IF(B6>=0.10,"BUY / ACCUMULATE",IF(AND(B6>=-0.10,B6<=0.10),"HOLD / FAIRLY VALUED","REDUCE / AVOID")))')
    ws2.cell(r, 3, "")

    # format numbers
    ws2.cell(2, 2).number_format = '#,##0.00'
    ws2.cell(3, 2).number_format = '#,##0'
    ws2.cell(4, 2).number_format = '#,##0.0000'
    ws2.cell(5, 2).number_format = '#,##0.00'
    ws2.cell(6, 2).number_format = '0.0%'

    # style recommendation
    ws2.cell(7, 2).font = Font(bold=True)

    # widths
    ws2.column_dimensions["A"].width = 30
    ws2.column_dimensions["B"].width = 28
    ws2.column_dimensions["C"].width = 10

    # Save to bytes
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()


# Build the Excel file bytes
excel_bytes = build_summary_excel_with_formulas(
    selected_models=selected_models,
    value_map=value_map,
    weights_new=weights_new,
    num_shares=num_shares,
    current_price=current_price
)

st.download_button(
    label="⬇️ Download Valuation Summary (Excel with formulas)",
    data=excel_bytes,
    file_name="valuation_summary_with_formulas.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
