import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date
import io
import re
from pathlib import Path
import hashlib
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
st.set_page_config(page_title="Banking Valuation (Residual Income)", layout="wide")

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
    <span class="fbc-page-header-icon">🏛️</span>
    <span class="fbc-page-header-title">Banking Valuation — Residual Income Model</span>
    <span class="fbc-badge">FBC Securities</span>
    <div class="fbc-page-header-sub">BVPS, residual income & terminal value — purpose-built for financial institutions.</div>
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


st.markdown("""
<style>
html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small {
  font-family: Georgia, "Times New Roman", serif !important;
}

/* Dataframes */
.stDataFrame, .stTable {
    font-family: Georgia, "Times New Roman", serif !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FX SESSION STATE (DCF-style)  ✅ (same approach as 1_DCF.py)
# ---------------------------------------------------------
st.session_state.setdefault("bank_conversion_method", "NO_FX")  # "NO_FX" or "FX_EXCEL"
st.session_state.setdefault("bank_currency", "USD (already converted)")

st.session_state.setdefault("bank_fx_bytes", None)
st.session_state.setdefault("bank_fx_name", None)
st.session_state.setdefault("bank_fx_raw", None)

st.session_state.setdefault("bank_fx_column", None)
st.session_state.setdefault("bank_yearly_fx", {})              # IS + SOCE yearly averages
st.session_state.setdefault("bank_bs_closing_dates", {})        # per-year chosen closing date
st.session_state.setdefault("bank_bs_fx_rates", {})             # per-year closing fx used

st.session_state.setdefault("bank_fx_signature", None)
st.session_state.setdefault("bank_bs_fx_dirty", False)

# =========================================================
# SESSION HELPERS (PERSIST EVERYTHING)
# =========================================================
def ss_init(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def clean_defaults(default_list, options):
    if not isinstance(default_list, (list, tuple)):
        return []
    return [x for x in default_list if x in options]

# One namespace dict for this page
ss_init("bank", {
    "file_bytes": None,
    "file_name": None,

    "is_sheet": None,
    "bs_sheet": None,
    "soce_sheet": None,

    "currency": "USD (already converted)",

    # mappings
    "equity_labels": [],            # BS equity rows
    "soce_total_label": None,       # (not used now, kept)
    "use_soce_for_equity": False,  # ALWAYS use Balance Sheet equity for base year
    "earnings_label": None,
    "base_year": None,

    "shares": 0.0,

    "ke_mode": "Auto (if available)",
    "ke_pct": 12.7,

    "n_years": 5,

    "yoy_mode": "Uniform",
    "disc_mode": "Uniform",
    "eps_mode": "Uniform",

    # store per-year dictionaries
    "yoy": {},
    "disc": {},
    "eps_g": {},

    "yoy_uniform_pct": 15.0,
    "disc_uniform_pct": 5.0,
    "eps_uniform_pct": 3.0,

    "g_term_pct": 4.9,

    "outputs": {}
})
BANK = st.session_state["bank"]

# Clear button (YOU control resets)
cclr1, cclr2 = st.columns([1, 6])
with cclr1:
    if st.button("🔄 Clear Banking Inputs", key="bank_clear_btn"):
        st.session_state.pop("bank", None)

        # also clear FX state
        for k in [
            "bank_conversion_method","bank_currency","bank_fx_bytes","bank_fx_name","bank_fx_raw",
            "bank_fx_column","bank_yearly_fx","bank_bs_closing_dates","bank_bs_fx_rates",
            "bank_fx_signature","bank_bs_fx_dirty",
            "bank_is_base","bank_bs_base","bank_soce_base","bank_is_df","bank_bs_df","bank_soce_df",
            "bank_parse_signature"
        ]:
            st.session_state.pop(k, None)

        st.rerun()

# =========================================================
# HELPERS (same style as DCF)
# =========================================================
def clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-Item columns to numeric (remove commas, brackets, spaces)."""
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

def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def option_labels_from_items(items):
    return [f"{i+1}: {name}" for i, name in enumerate(items)]

def indices_from_labels(labels):
    idx = []
    for s in labels or []:
        try:
            i = int(str(s).split(":", 1)[0]) - 1
            idx.append(i)
        except Exception:
            continue
    return idx

def normalize_year_cols(df: pd.DataFrame):
    """
    Returns: (year_cols, colmap)
    year_cols = list of clean year strings like ["2022","2023","2024"]
    colmap = dict {clean_year_str: original_col_name}
    """
    colmap = {}
    out = []
    for c in df.columns:
        if c == "Item":
            continue
        s = str(c).strip()

        # drop junk headers (merged cells -> Unnamed columns)
        if s.lower().startswith("unnamed"):
            continue

        # try extract a 4-digit year anywhere in header
        m = re.search(r"(19|20)\d{2}", s)
        if m:
            y = m.group(0)
            out.append(y)
            colmap[y] = c
            continue

        # datetime-like header
        try:
            ts = pd.to_datetime(s, errors="coerce")
            if not pd.isna(ts):
                y = str(ts.year)
                out.append(y)
                colmap[y] = c
        except Exception:
            pass

    # keep unique in original order
    seen = set()
    year_cols = []
    for y in out:
        if y not in seen:
            seen.add(y)
            year_cols.append(y)

    return year_cols, colmap

def convert_df_yearwise(df: pd.DataFrame, year_rates: dict, colmap: dict | None = None) -> pd.DataFrame:
    """
    Divide each year column by its matching FX rate, ZWG → USD.
    If colmap is provided (clean_year -> original col), conversion uses that mapping.
    """
    df2 = df.copy()

    if colmap is None:
        for col in df2.columns:
            if col == "Item":
                continue
            key = str(col)
            if key in year_rates and year_rates[key] != 0:
                df2[col] = df2[col] / year_rates[key]
        return df2

    for y, orig_col in colmap.items():
        if y in year_rates and year_rates[y] != 0 and orig_col in df2.columns:
            df2[orig_col] = df2[orig_col] / year_rates[y]
    return df2

def get_fx_asof_date(fx_df: pd.DataFrame, fx_col: str, closing_date: date):
    """
    Returns last available fx rate on or before closing_date.
    fx_df requires columns: Date (datetime), fx_col (numeric).
    """
    cd = pd.Timestamp(closing_date)
    sub = fx_df.loc[fx_df["Date"] <= cd, ["Date", fx_col]].dropna()
    if sub.empty:
        return None
    sub = sub.sort_values("Date")
    return float(sub.iloc[-1][fx_col])

def find_best_earnings_row(items_lower):
    """Default to Normalised profit; fallback to profit for the year / PAT style."""
    for i, txt in enumerate(items_lower):
        if "normalised profit" in txt or "normalized profit" in txt:
            return i

    fallback_keys = [
        "profit for the year",
        "profit for the period",
        "net profit",
        "profit after tax",
        "pat",
        "loss/profit for the year",
        "(loss)/profit for the year",
        "profit for the year attributable",
    ]
    for key in fallback_keys:
        for i, txt in enumerate(items_lower):
            if key in txt:
                return i
    return None

def _md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# =========================================================
# STEP 0 — Upload Excel (PERSIST BYTES)
# =========================================================
st.markdown("### 📄 Upload Statements (Income Statement + Balance Sheet + SoCE)")

uploaded = st.file_uploader("Upload Excel file", type=["xlsx"], key="bank_upload_box")

if uploaded is not None:
    BANK["file_bytes"] = uploaded.getvalue()
    BANK["file_name"] = uploaded.name

if BANK["file_bytes"] is None:
    st.info("Please upload your Excel file to continue.")
    st.stop()

# Parse signature (if file or selected sheets change, rebuild bases)
file_sig = (_md5_bytes(BANK["file_bytes"]), BANK["file_name"])

xls = pd.ExcelFile(io.BytesIO(BANK["file_bytes"]))
st.write("Detected sheets:", xls.sheet_names)

# Persist sheet selections
if BANK["is_sheet"] is None:
    BANK["is_sheet"] = xls.sheet_names[0]
if BANK["bs_sheet"] is None:
    BANK["bs_sheet"] = xls.sheet_names[min(1, len(xls.sheet_names) - 1)]
if BANK["soce_sheet"] is None:
    BANK["soce_sheet"] = xls.sheet_names[min(2, len(xls.sheet_names) - 1)]

is_sheet = st.selectbox(
    "Income Statement sheet",
    xls.sheet_names,
    index=xls.sheet_names.index(BANK["is_sheet"]) if BANK["is_sheet"] in xls.sheet_names else 0,
    key="bank_is_sheet_input",
)
bs_sheet = st.selectbox(
    "Balance Sheet sheet",
    xls.sheet_names,
    index=xls.sheet_names.index(BANK["bs_sheet"]) if BANK["bs_sheet"] in xls.sheet_names else min(1, len(xls.sheet_names)-1),
    key="bank_bs_sheet_input",
)
soce_sheet = st.selectbox(
    "Statement of Changes in Equity (SoCE) sheet",
    xls.sheet_names,
    index=xls.sheet_names.index(BANK["soce_sheet"]) if BANK["soce_sheet"] in xls.sheet_names else min(2, len(xls.sheet_names)-1),
    key="bank_soce_sheet_input",
)

BANK["is_sheet"] = is_sheet
BANK["bs_sheet"] = bs_sheet
BANK["soce_sheet"] = soce_sheet

parse_sig = (file_sig, is_sheet, bs_sheet, soce_sheet)
if st.session_state.get("bank_parse_signature") != parse_sig:
    # Load + clean (BASE = before FX conversion)
    is_df_raw = clean_numeric_cols(xls.parse(is_sheet))
    bs_df_raw = clean_numeric_cols(xls.parse(bs_sheet))
    soce_df_raw = clean_numeric_cols(xls.parse(soce_sheet))

    # store bases for FX re-conversion (same as DCF)
    st.session_state["bank_is_base"] = is_df_raw.copy()
    st.session_state["bank_bs_base"] = bs_df_raw.copy()
    st.session_state["bank_soce_base"] = soce_df_raw.copy()

    # set current working dfs (may be converted later)
    st.session_state["bank_is_df"] = is_df_raw.copy()
    st.session_state["bank_bs_df"] = bs_df_raw.copy()
    st.session_state["bank_soce_df"] = soce_df_raw.copy()

    # reset FX signature so conversion refreshes
    st.session_state["bank_fx_signature"] = None
    st.session_state["bank_bs_fx_dirty"] = True

    st.session_state["bank_parse_signature"] = parse_sig

# Work DFs (may be overwritten by FX conversion block)
is_df = st.session_state["bank_is_df"]
bs_df = st.session_state["bank_bs_df"]
soce_df = st.session_state["bank_soce_df"]

# detect year columns safely (avoids 'Unnamed: 2' issues)
is_years, is_colmap = normalize_year_cols(is_df)
bs_years, bs_colmap = normalize_year_cols(bs_df)
soce_years, soce_colmap = normalize_year_cols(soce_df)

if not is_years:
    st.error("❌ Could not detect year columns in Income Statement (headers may be merged).")
    st.stop()
if not bs_years:
    st.error("❌ Could not detect year columns in Balance Sheet (headers may be merged).")
    st.stop()
if not soce_years:
    st.error("❌ Could not detect year columns in SoCE (headers may be merged).")
    st.stop()

# =========================================================
# STEP 1 — Currency + FX Conversion (DCF-style Excel-based)
# =========================================================
st.markdown("### 💱 Currency & Exchange Rates")

# 1️⃣ Currency selector (persistent)
currency = st.selectbox(
    "Currency of uploaded statements:",
    ["USD (already converted)", "ZWG (convert using FX Excel)"],
    index=0 if st.session_state.get("bank_conversion_method") == "NO_FX" else 1,
    key="bank_currency_select"
)

# store BOTH label + stable method
st.session_state["bank_currency"] = currency
st.session_state["bank_conversion_method"] = "NO_FX" if currency.startswith("USD") else "FX_EXCEL"
BANK["currency"] = currency

# 2️⃣ FX upload — only if ZWG
if st.session_state["bank_conversion_method"] == "FX_EXCEL":
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
        key="bank_fx_uploader"
    )
    if fx_file is not None:
        st.session_state["bank_fx_bytes"] = fx_file.getvalue()
        st.session_state["bank_fx_name"] = fx_file.name
else:
    st.session_state["bank_fx_bytes"] = None
    st.session_state["bank_fx_name"] = None
    st.session_state["bank_fx_raw"] = None
    st.session_state["bank_yearly_fx"] = {}
    st.session_state["bank_bs_fx_rates"] = {}
    st.session_state["bank_fx_signature"] = None
    st.session_state["bank_bs_fx_dirty"] = False

# 3️⃣ If USD → skip FX
if currency.startswith("USD"):
    st.success("✅ Data assumed to be in USD. No FX conversion applied.")
else:
    st.warning("ZWG detected. Upload FX Excel with Dates and Interbank Rates to convert to USD.")
    if st.session_state["bank_fx_bytes"] is None:
        st.stop()

    # 4️⃣ Load FX Excel ONCE
    if st.session_state.get("bank_fx_raw") is None:
        fx_raw = pd.read_excel(io.BytesIO(st.session_state["bank_fx_bytes"]))
        fx_raw.columns = [str(c).strip() for c in fx_raw.columns]
        st.session_state["bank_fx_raw"] = fx_raw
    else:
        fx_raw = st.session_state["bank_fx_raw"]

    st.subheader("Raw FX data (preview)")
    st.dataframe(fx_raw.head(), width='stretch')

    # 5️⃣ Validate columns
    if "Date" not in fx_raw.columns:
        st.error("❌ FX Excel must contain a column named 'Date'.")
        st.stop()

    fx_df = fx_raw.copy()
    fx_df["Date"] = pd.to_datetime(fx_df["Date"], errors="coerce", dayfirst=True)
    fx_df = fx_df.dropna(subset=["Date"])

    # 6️⃣ FX column selector (restricted + persistent)
    allowed_fx_cols = ["Interbank", "Alternative", "Premium"]
    available_fx_cols = [c for c in allowed_fx_cols if c in fx_df.columns]
    if not available_fx_cols:
        st.error("❌ FX Excel must contain Interbank / Alternative / Premium columns.")
        st.stop()

    if st.session_state.get("bank_fx_column") is None:
        st.session_state["bank_fx_column"] = available_fx_cols[0]

    fx_col = st.selectbox(
        "Which FX rate column should be used?",
        available_fx_cols,
        index=available_fx_cols.index(st.session_state["bank_fx_column"]),
        key="bank_fx_column_select"
    )
    st.session_state["bank_fx_column"] = fx_col

    fx_df[fx_col] = pd.to_numeric(fx_df[fx_col], errors="coerce")
    fx_df = fx_df.dropna(subset=[fx_col])

    # -------------------------------------------------
    # 🪙 Apply conversion factor by selected Year(s) + Date Ranges (same as DCF)
    # -------------------------------------------------
    st.markdown("### 🪙 Apply ZWG→ZiG factor by Year + Range")

    # years available from statements (use IS years as master like DCF)
    available_years = sorted({str(int(y)) for y in is_years})

    st.session_state.setdefault("bank_factor_enabled", False)
    st.session_state.setdefault("bank_zig_factor", 2498.7242)
    st.session_state.setdefault("bank_factor_year_ranges", {})     # {"2024":[{"start":..,"end":..}],...}
    st.session_state.setdefault("bank_factor_years_selected_vals", [])

    enable_factor = st.checkbox(
        "Enable manual factor (for mixed ZWG/ZiG periods)",
        value=bool(st.session_state["bank_factor_enabled"]),
        key="bank_factor_enabled_ui"
    )
    st.session_state["bank_factor_enabled"] = enable_factor

    zig_factor = st.number_input(
        "ZWG → ZiG conversion factor (divide FX by this inside selected ranges)",
        value=float(st.session_state["bank_zig_factor"]),
        step=0.0001,
        format="%.6f",
        key="bank_zig_factor_ui"
    )
    st.session_state["bank_zig_factor"] = zig_factor

    if enable_factor:
        years_selected = st.multiselect(
            "Select the year(s) where you want to apply the factor",
            available_years,
            default=[y for y in st.session_state["bank_factor_years_selected_vals"] if y in available_years],
            key="bank_factor_years_selected_ui"
        )
        st.session_state["bank_factor_years_selected_vals"] = years_selected

        for y in years_selected:
            st.session_state["bank_factor_year_ranges"].setdefault(y, [])

            st.markdown(f"#### Ranges for {y}")

            if st.button(f"➕ Add range for {y}", key=f"bank_add_range_{y}"):
                st.session_state["bank_factor_year_ranges"][y].append({
                    "start": date(int(y), 1, 1),
                    "end": date(int(y), 12, 31),
                })

            ranges = st.session_state["bank_factor_year_ranges"][y]
            for i, r in enumerate(ranges):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    new_start = st.date_input(
                        f"{y} range {i + 1} start",
                        value=r["start"],
                        key=f"bank_{y}_r{i}_start"
                    )
                with c2:
                    new_end = st.date_input(
                        f"{y} range {i + 1} end",
                        value=r["end"],
                        key=f"bank_{y}_r{i}_end"
                    )

                if new_end < new_start:
                    st.error("❌ End date cannot be before start date.")
                else:
                    st.session_state["bank_factor_year_ranges"][y][i]["start"] = new_start
                    st.session_state["bank_factor_year_ranges"][y][i]["end"] = new_end

                with c3:
                    if st.button("🗑️ Delete", key=f"bank_{y}_r{i}_del"):
                        st.session_state["bank_factor_year_ranges"][y].pop(i)
                        st.rerun()

        if zig_factor <= 0:
            st.error("❌ Factor must be > 0.")
            st.stop()

        fx_df["_factor_applied"] = False
        for y in years_selected:
            for r in st.session_state["bank_factor_year_ranges"].get(y, []):
                s = pd.Timestamp(r["start"])
                e = pd.Timestamp(r["end"])
                mask = (fx_df["Date"] >= s) & (fx_df["Date"] <= e)
                if mask.any():
                    fx_df.loc[mask, fx_col] = fx_df.loc[mask, fx_col] / float(zig_factor)
                    fx_df.loc[mask, "_factor_applied"] = True

        st.success(f"✅ Factor applied to {int(fx_df['_factor_applied'].sum()):,} FX rows.")
        st.dataframe(fx_df.loc[fx_df["_factor_applied"], ["Date", fx_col]].head(20), width='stretch')

    # 7️⃣ Compute YEARLY AVERAGE FX (Income Statement + SoCE)
    fx_df["Year"] = fx_df["Date"].dt.year.astype(int)
    yearly_fx = (
        fx_df.groupby("Year")[fx_col]
        .mean()
        .round(6)
        .to_dict()
    )
    yearly_fx = {str(y): float(v) for y, v in yearly_fx.items()}
    st.session_state["bank_yearly_fx"] = yearly_fx

    st.subheader("📊 Yearly FX averages (Income Statement + SoCE)")
    st.dataframe(
        pd.DataFrame({"Year": list(yearly_fx.keys()), "FX Rate": list(yearly_fx.values())}),
        width='stretch'
    )

    # 8️⃣ Balance Sheet FX — PER-YEAR CLOSING DATES
    st.markdown("### 📌 Balance Sheet FX — Closing Dates (per year)")

    st.session_state.setdefault("bank_bs_closing_dates", {})
    st.session_state.setdefault("bank_bs_fx_dirty", False)

    for y in [str(y) for y in bs_years]:
        default_date = st.session_state["bank_bs_closing_dates"].get(y, date(int(y), 12, 31))
        chosen_date = st.date_input(
            f"Closing date for Balance Sheet {y}",
            value=default_date,
            key=f"bank_bs_close_date_{y}"
        )

        if st.session_state["bank_bs_closing_dates"].get(y) != chosen_date:
            st.session_state["bank_bs_closing_dates"][y] = chosen_date
            st.session_state["bank_bs_fx_dirty"] = True

    # 9️⃣ Compute per-year closing FX rates for BS
    bs_fx_rates = {}
    for y in [str(y) for y in bs_years]:
        closing_date = st.session_state["bank_bs_closing_dates"][y]
        fx_rate = get_fx_asof_date(fx_df=fx_df, fx_col=fx_col, closing_date=closing_date)
        if fx_rate is None:
            st.error(f"❌ No FX rate found on or before {closing_date} for year {y}")
            st.stop()
        bs_fx_rates[y] = float(fx_rate)

    st.session_state["bank_bs_fx_rates"] = bs_fx_rates

    st.subheader("📊 Balance Sheet Closing FX Rates Used")
    bs_fx_table = pd.DataFrame([
        {"Year": y, "Closing Date": st.session_state["bank_bs_closing_dates"][y], "FX Rate Used": bs_fx_rates[y]}
        for y in bs_fx_rates.keys()
    ])
    st.dataframe(bs_fx_table, width='stretch')

    # 🔟 Validate FX coverage for IS + SOCE years (yearly averages)
    stmt_years_need_avg = sorted(set(is_years) | set(soce_years))
    missing_avg_years = sorted(set(stmt_years_need_avg) - set(yearly_fx.keys()))
    if missing_avg_years:
        st.error(f"❌ Missing FX yearly-average data for statement years: {', '.join(missing_avg_years)}")
        st.stop()

    # 1️⃣1️⃣ APPLY FX CONVERSION (RE-RUN IF SETTINGS CHANGE) (same as DCF signature logic)
    factor_signature = (
        st.session_state.get("bank_factor_enabled", False),
        st.session_state.get("bank_zig_factor", None),
        str(st.session_state.get("bank_factor_year_ranges", {}))
    )

    fx_signature = (
        currency,
        fx_col,
        factor_signature,
        tuple((y, str(st.session_state["bank_bs_closing_dates"][y])) for y in [str(y) for y in bs_years])
    )

    if (
        st.session_state.get("bank_fx_signature") != fx_signature
        or st.session_state.get("bank_bs_fx_dirty")
    ):
        # Always start from BASE statements (pre-conversion)
        is_base = st.session_state["bank_is_base"].copy()
        bs_base = st.session_state["bank_bs_base"].copy()
        soce_base = st.session_state["bank_soce_base"].copy()

        # IS → YEARLY AVERAGE FX
        is_converted = convert_df_yearwise(is_base, yearly_fx, is_colmap)

        # BS → PER-YEAR CLOSING FX
        bs_converted = convert_df_yearwise(bs_base, bs_fx_rates, bs_colmap)

        # SoCE → YEARLY AVERAGE FX (same rule as IS)
        soce_converted = convert_df_yearwise(soce_base, yearly_fx, soce_colmap)

        st.session_state["bank_is_df"] = is_converted
        st.session_state["bank_bs_df"] = bs_converted
        st.session_state["bank_soce_df"] = soce_converted

        st.session_state["bank_fx_signature"] = fx_signature
        st.session_state["bank_bs_fx_dirty"] = False

        st.info("🔁 FX conversion refreshed (settings changed).")
    else:
        st.success("✅ FX conversion applied correctly (IS+SoCE = yearly avg, BS = per-year closing rates)")

    # refresh working dfs after conversion
    is_df = st.session_state["bank_is_df"]
    bs_df = st.session_state["bank_bs_df"]
    soce_df = st.session_state["bank_soce_df"]

    # re-detect year columns (same columns, but safe)
    is_years, is_colmap = normalize_year_cols(is_df)
    bs_years, bs_colmap = normalize_year_cols(bs_df)
    soce_years, soce_colmap = normalize_year_cols(soce_df)

with st.expander("🔎 View cleaned Income Statement (USD)", expanded=False):
    st.dataframe(is_df, width='stretch')

with st.expander("🔎 View cleaned Balance Sheet (USD)", expanded=False):
    st.dataframe(bs_df, width='stretch')

with st.expander("🔎 View cleaned SoCE (USD)", expanded=False):
    st.dataframe(soce_df, width='stretch')

# =========================================================
# STEP 2 — SoCE Mapping (USER SELECTS CLOSING BALANCE + TOTAL COLUMN)
# =========================================================
st.markdown("### 🟦 Statement of Changes in Equity (SoCE) Mapping")

soce_items = soce_df["Item"].astype(str).tolist()
soce_labels = [f"{i+1}: {x}" for i, x in enumerate(soce_items)]

# --- Select Closing Balance row ---
if BANK.get("soce_closing_row") not in soce_labels:
    default_idx = next(
        (i for i, x in enumerate(soce_items) if "closing balance" in x.lower()),
        0
    )
    BANK["soce_closing_row"] = soce_labels[default_idx]

closing_row_label = st.selectbox(
    "Select **Closing Balance** row (Normalised if available):",
    soce_labels,
    index=soce_labels.index(BANK["soce_closing_row"]),
    key="bank_soce_closing_row_select"
)
BANK["soce_closing_row"] = closing_row_label
closing_row_idx = int(closing_row_label.split(":")[0]) - 1

# --- Select TOTAL column ---
numeric_cols = [c for c in soce_df.columns if c != "Item"]

if BANK.get("soce_total_col") not in numeric_cols:
    BANK["soce_total_col"] = numeric_cols[-1]  # default = last column

total_col = st.selectbox(
    "Select **TOTAL Equity column**:",
    numeric_cols,
    index=numeric_cols.index(BANK["soce_total_col"]),
    key="bank_soce_total_col_select"
)
BANK["soce_total_col"] = total_col

# --- Extract equity for all detected years ---
soce_year_equity = {}
for y in soce_years:
    # NOTE: equity is taken from the selected TOTAL column in the selected closing row
    val = safe_float(soce_df.iloc[closing_row_idx][total_col], np.nan)
    if not np.isnan(val):
        soce_year_equity[y] = val

if not soce_year_equity:
    st.error("❌ Could not extract equity from SoCE. Check row/column selection.")
    st.stop()

BANK["soce_year_equity"] = soce_year_equity

st.success("✅ SoCE Closing Equity mapped successfully")
st.dataframe(
    pd.DataFrame({
        "Year": list(soce_year_equity.keys()),
        "Total Equity": list(soce_year_equity.values())
    }),
    width='stretch'
)

# =========================================================
# STEP 3 — Balance Sheet Mapping (Equity) (PERSIST)
# =========================================================
st.markdown("### 🟩 Balance Sheet Mapping (Equity)")

bs_items = list(bs_df["Item"].astype(str))
bs_labels = option_labels_from_items(bs_items)

sel_equity = st.multiselect(
    "Select ALL rows that represent Total Equity (multi-select allowed):",
    bs_labels,
    default=clean_defaults(BANK.get("equity_labels", []), bs_labels),
    key="bank_equity_multiselect_input"
)
BANK["equity_labels"] = sel_equity
equity_idx_list = indices_from_labels(sel_equity)

if not equity_idx_list:
    st.warning("Select at least one Equity row (Balance Sheet) to continue.")
    st.stop()

# =========================================================
# STEP 4 — Earnings line (defaults to Normalised profit) (PERSIST)
# =========================================================
st.markdown("### ✅ Income Statement Earnings (EPS)")

items = is_df["Item"].astype(str).tolist()
earn_opt = [f"{i+1}: {items[i]}" for i in range(len(items))]

if BANK["earnings_label"] is None:
    items_lower = [s.lower().strip() for s in items]
    default_earn_idx = find_best_earnings_row(items_lower)
    if default_earn_idx is None:
        default_earn_idx = 0
    BANK["earnings_label"] = earn_opt[int(default_earn_idx)]

chosen_earn = st.selectbox(
    "Earnings line (defaults to Normalised profit):",
    earn_opt,
    index=earn_opt.index(BANK["earnings_label"]) if BANK["earnings_label"] in earn_opt else 0,
    key="bank_earnings_select_input"
)
BANK["earnings_label"] = chosen_earn
earnings_idx = int(chosen_earn.split(":", 1)[0]) - 1

# =========================================================
# STEP 5 — Base year selection (actual years) (PERSIST)
# =========================================================
st.markdown("### 📅 Base Year (Actual Years)")

# intersection of IS/BS/SoCE years (clean years)
# Base year options should be driven by IS + BS (SoCE may not have same columns)
common_years = sorted(set(is_years) & set(bs_years))

if not common_years:
    st.warning("No common year columns between Income Statement and Balance Sheet.")
    st.stop()

# persist base year
if BANK["base_year"] is None or str(BANK["base_year"]) not in common_years:
    BANK["base_year"] = common_years[-1]

base_year = st.selectbox(
    "Choose base year:",
    common_years,
    index=common_years.index(str(BANK["base_year"])),
    key="bank_base_year_input"
)
BANK["base_year"] = str(base_year)
base_year = str(base_year)

# =========================================================
# STEP 6 — Pull base values (SoCE preferred for Equity)
# =========================================================
is_base_col = is_colmap.get(base_year)
bs_base_col = bs_colmap.get(base_year)
soce_base_col = soce_colmap.get(base_year)

earnings_0 = safe_float(is_df.iloc[earnings_idx][is_base_col], 0.0) if is_base_col is not None else 0.0

# ✅ Always use Balance Sheet equity for the selected base year
if bs_base_col is None:
    st.error(f"❌ Could not locate base-year column ({base_year}) in Balance Sheet.")
    st.stop()

book_equity_0 = float(bs_df.loc[equity_idx_list, bs_base_col].sum(skipna=True))


st.markdown("### 📌 Base Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric(f"Total Equity ({base_year})", f"{book_equity_0:,.0f}")
with c2:
    st.metric(f"Earnings ({base_year})", f"{earnings_0:,.0f}")
with c3:
    st.metric("Earnings line", items[earnings_idx])

# =========================================================
# STEP 7 — Assumptions (NO SHARES)
# =========================================================
st.markdown("### ⚙️ Assumptions")

# =========================================================
# Ke MODULE — DCF-style Auto + Override + Uploads
# Ke = Rf + β * MRP
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DCF_PARAMS_PATH = DATA_DIR / "dcf_parameters.xlsx"
UNLEVERED_BETAS_PATH = DATA_DIR / "unlevered_betas.xlsx"

def _to_decimal(x):
    """Accepts 0.15 or 15; returns decimal 0.15"""
    try:
        x = float(x)
    except Exception:
        return None
    return x / 100.0 if x > 1.5 else x

def _load_country_params_df(file_or_path) -> pd.DataFrame:
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

@st.cache_data(show_spinner=False)
def _load_unlevered_betas_any(file_or_path, file_mtime: float = 0.0) -> pd.DataFrame:
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

def _set_widget_value(widget_key: str, value: float):
    st.session_state[widget_key] = float(value)

st.markdown("### 💰 Cost of Equity (Ke) — DCF-style Auto + Override")

left, right = st.columns([1.2, 1.0], vertical_alignment="top")

with left:
    st.markdown("#### 🌍 Country ERP & Default Spread")

    BANK.setdefault("bank_country_upload_enabled", False)
    BANK.setdefault("bank_country_params_bytes", None)
    BANK.setdefault("bank_country_params_name", None)

    c0, c1 = st.columns([1, 1])
    with c0:
        country_upload = st.checkbox(
            "Upload Country Params (optional)",
            value=bool(BANK["bank_country_upload_enabled"]),
            key="bank_country_upload_enabled_ui"
        )
    with c1:
        st.caption("Columns: Country | ERP | Default Spread")

    BANK["bank_country_upload_enabled"] = country_upload

    if country_upload:
        up_country = st.file_uploader(
            "Upload Country ERP + Default Spread (xlsx)",
            type=["xlsx"],
            key="bank_country_params_uploader"
        )
        if up_country is not None:
            BANK["bank_country_params_bytes"] = up_country.getvalue()
            BANK["bank_country_params_name"] = up_country.name
    else:
        BANK["bank_country_params_bytes"] = None
        BANK["bank_country_params_name"] = None

    df_params = None
    params_source = None
    try:
        if country_upload and BANK["bank_country_params_bytes"] is not None:
            df_params = _load_country_params_df(io.BytesIO(BANK["bank_country_params_bytes"]))
            params_source = f"Uploaded: {BANK.get('bank_country_params_name','(file)')}"
        else:
            if DCF_PARAMS_PATH.exists():
                df_params = _load_country_params_df(DCF_PARAMS_PATH)
                params_source = f"Default: {DCF_PARAMS_PATH.name}"
            else:
                st.warning(f"⚠️ Missing default file: {DCF_PARAMS_PATH}. Upload a file above.")
    except Exception as e:
        st.error(f"❌ Country params file error: {e}")
        df_params = None

    if params_source:
        st.caption(f"Source: **{params_source}**")

    auto_erp_dec = None
    auto_spread_dec = None
    auto_mrp_pct = None
    auto_rf_pct = None

    chosen_country = None
    if df_params is not None and not df_params.empty:
        country_list = sorted(df_params["Country"].dropna().astype(str).unique().tolist())
        default_country = "Zimbabwe" if "Zimbabwe" in country_list else country_list[0]

        chosen_country = st.selectbox(
            "Select Country:",
            country_list,
            index=country_list.index(default_country),
            key="bank_country_select"
        )

        row = df_params[df_params["Country"].astype(str) == str(chosen_country)]
        if not row.empty:
            auto_erp_dec = _to_decimal(row.iloc[0]["ERP"])
            auto_spread_dec = _to_decimal(row.iloc[0]["DefaultSpread"])

    BANK.setdefault("bank_zim_avg_cod_pct", 18.0)
    zim_avg_cod_pct = st.number_input(
        "Avg Cost of Debt (Zimbabwe USD) (%)",
        value=float(BANK["bank_zim_avg_cod_pct"]),
        step=0.1,
        key="bank_zim_avg_cod_pct_input"
    )
    BANK["bank_zim_avg_cod_pct"] = float(zim_avg_cod_pct)
    zim_avg_cod = float(zim_avg_cod_pct) / 100.0

    if auto_erp_dec is not None:
        auto_mrp_pct = auto_erp_dec * 100.0
    if auto_spread_dec is not None:
        auto_rf_pct = (zim_avg_cod - auto_spread_dec) * 100.0

    if auto_mrp_pct is not None and auto_rf_pct is not None:
        st.success(f"Auto: **MRP={auto_mrp_pct:.2f}%**, **Rf={auto_rf_pct:.2f}%** (Rf = AvgCoD − Spread)")
    else:
        st.info("Auto RF/MRP not ready (check file + selected country values).")

with right:
    st.markdown("#### 🧩 Industry Unlevered Betas (βu)")

    BANK.setdefault("bank_beta_upload_enabled", False)
    BANK.setdefault("bank_beta_file_bytes", None)
    BANK.setdefault("bank_beta_file_name", None)

    beta_upload = st.checkbox(
        "Upload Industry Betas (optional)",
        value=bool(BANK["bank_beta_upload_enabled"]),
        key="bank_beta_upload_enabled_ui"
    )
    BANK["bank_beta_upload_enabled"] = beta_upload

    if beta_upload:
        up_beta = st.file_uploader(
            "Upload Industry betas (xlsx)",
            type=["xlsx"],
            key="bank_beta_uploader"
        )
        if up_beta is not None:
            BANK["bank_beta_file_bytes"] = up_beta.getvalue()
            BANK["bank_beta_file_name"] = up_beta.name
    else:
        BANK["bank_beta_file_bytes"] = None
        BANK["bank_beta_file_name"] = None

    betas_df = None
    beta_source = None
    try:
        if BANK.get("bank_beta_upload_enabled") and BANK.get("bank_beta_file_bytes") is not None:
            betas_df = _load_unlevered_betas_any(io.BytesIO(BANK["bank_beta_file_bytes"]))
            beta_source = f"Uploaded: {BANK.get('bank_beta_file_name','(file)')}"
        else:
            if UNLEVERED_BETAS_PATH.exists():
                mtime = UNLEVERED_BETAS_PATH.stat().st_mtime
                betas_df = _load_unlevered_betas_any(UNLEVERED_BETAS_PATH, file_mtime=mtime)
                beta_source = f"Default: {UNLEVERED_BETAS_PATH.name}"
            else:
                st.warning(f"⚠️ Missing default file: {UNLEVERED_BETAS_PATH}. Upload a file above.")
    except Exception as e:
        st.warning(f"⚠️ Could not load industry betas: {e}")
        betas_df = None

    if beta_source:
        st.caption(f"Source: **{beta_source}**")

    BANK.setdefault("bank_industries_selected", [])
    BANK.setdefault("bank_beta_blend_method", "Simple average")
    BANK.setdefault("bank_industry_weights", {})
    BANK.setdefault("bank_auto_apply_beta_u", True)

    beta_u_auto = None

    if betas_df is not None and not betas_df.empty:
        industry_list = betas_df["Industry"].tolist()

        selected = st.multiselect(
            "Industries (blend βu):",
            industry_list,
            default=[i for i in BANK.get("bank_industries_selected", []) if i in industry_list],
            key="bank_industries_multiselect"
        )
        BANK["bank_industries_selected"] = selected

        blend_method = st.radio(
            "Blend method:",
            ["Simple average", "Weighted average"],
            index=0 if BANK.get("bank_beta_blend_method", "Simple average") == "Simple average" else 1,
            key="bank_beta_blend_method_radio",
            horizontal=True
        )
        BANK["bank_beta_blend_method"] = blend_method

        if selected:
            sub = betas_df[betas_df["Industry"].isin(selected)].copy()

            if blend_method == "Simple average":
                beta_u_auto = float(sub["UnleveredBeta"].mean())
            else:
                st.caption("Enter weights (auto-normalized to 100%)")
                weights = []
                for ind in selected:
                    default_w = float(BANK.get("bank_industry_weights", {}).get(ind, 1.0))
                    w = st.number_input(
                        f"Weight: {ind}",
                        min_value=0.0,
                        value=default_w,
                        step=1.0,
                        key=f"bank_w_{ind}"
                    )
                    BANK["bank_industry_weights"][ind] = w
                    weights.append(w)

                total_w = float(sum(weights))
                if total_w > 0:
                    sub = sub.sort_values("Industry").reset_index(drop=True)
                    w_norm = np.array([BANK["bank_industry_weights"][ind] for ind in sub["Industry"]]) / total_w
                    beta_u_auto = float(np.sum(sub["UnleveredBeta"].values * w_norm))
                else:
                    st.error("❌ Total weight must be > 0.")

            if beta_u_auto is not None and np.isfinite(beta_u_auto):
                st.info(f"Blended βu (auto) = **{beta_u_auto:.2f}**")

                BANK.setdefault("bank_auto_apply_beta_u", True)
                auto_apply = st.checkbox(
                    "Auto-apply blended βu to βu input",
                    value=bool(BANK["bank_auto_apply_beta_u"]),
                    key="bank_auto_apply_beta_u_ui"
                )
                BANK["bank_auto_apply_beta_u"] = auto_apply

                c_apply1, c_apply2 = st.columns([1, 1])
                with c_apply1:
                    if st.button("✅ Apply blended βu now", key="bank_apply_beta_u_btn"):
                        BANK["bank_beta_u_input"] = float(beta_u_auto)
                        _set_widget_value("bank_beta_u_input_box", float(beta_u_auto))
                        st.success("Applied blended βu to βu input ✅")
                with c_apply2:
                    if st.button("↩ Reset βu to 1.00", key="bank_reset_beta_u_btn"):
                        BANK["bank_beta_u_input"] = 1.0
                        _set_widget_value("bank_beta_u_input_box", 1.0)

                current_sig = (tuple(selected), blend_method, tuple(sorted(BANK.get("bank_industry_weights", {}).items())))
                st.session_state.setdefault("bank_beta_u_sig", None)
                if auto_apply and (st.session_state["bank_beta_u_sig"] != current_sig):
                    BANK["bank_beta_u_input"] = float(beta_u_auto)
                    _set_widget_value("bank_beta_u_input_box", float(beta_u_auto))
                    st.session_state["bank_beta_u_sig"] = current_sig

    BANK.setdefault("bank_beta_manual_mode", False)
    beta_mode = st.radio(
        "Beta input mode:",
        ["Use βu (then lever it)", "Manual β (levered) override"],
        index=1 if bool(BANK.get("bank_beta_manual_mode", False)) else 0,
        key="bank_beta_mode_radio",
        horizontal=True
    )
    BANK["bank_beta_manual_mode"] = beta_mode.startswith("Manual")

st.markdown("#### ⚙️ CAPM Inputs")

BANK.setdefault("bank_use_auto_params", True)
use_auto = st.checkbox(
    "Use Auto (from Country params) for RF & MRP",
    value=bool(BANK.get("bank_use_auto_params", True)),
    key="bank_use_auto_params_ui"
)
BANK["bank_use_auto_params"] = use_auto

if "bank_ke_init" not in st.session_state:
    BANK["bank_rf_pct"] = float(auto_rf_pct) if auto_rf_pct is not None else 11.61
    BANK["bank_mrp_pct"] = float(auto_mrp_pct) if auto_mrp_pct is not None else 13.82
    BANK["bank_tax_pct_for_beta"] = 25.0
    BANK["bank_de_ratio_for_beta"] = 0.0
    BANK["bank_beta_u_input"] = float(BANK.get("bank_beta_u_input", 1.0))
    BANK["bank_beta_levered_manual"] = 0.22
    st.session_state["bank_ke_init"] = True

auto_signature = (auto_rf_pct, auto_mrp_pct, st.session_state.get("bank_country_select", None), float(BANK.get("bank_zim_avg_cod_pct", 0.0)))
st.session_state.setdefault("bank_auto_signature", None)

if use_auto and (auto_rf_pct is not None) and (auto_mrp_pct is not None) and (auto_signature != st.session_state["bank_auto_signature"]):
    BANK["bank_rf_pct"] = float(auto_rf_pct)
    BANK["bank_mrp_pct"] = float(auto_mrp_pct)
    st.session_state["bank_rf_pct_input"] = float(auto_rf_pct)
    st.session_state["bank_mrp_pct_input"] = float(auto_mrp_pct)
    st.session_state["bank_auto_signature"] = auto_signature
elif not use_auto:
    st.session_state["bank_auto_signature"] = None

st.session_state.setdefault("bank_rf_pct_input", float(BANK["bank_rf_pct"]))
st.session_state.setdefault("bank_mrp_pct_input", float(BANK["bank_mrp_pct"]))
st.session_state.setdefault("bank_beta_u_input_box", float(BANK.get("bank_beta_u_input", 1.0)))

capm1, capm2, capm3 = st.columns([1, 1, 1])

with capm1:
    rf_input = st.number_input("Risk-free rate (%)", step=0.1, key="bank_rf_pct_input")
    mrp_input = st.number_input("Market risk premium (%)", step=0.1, key="bank_mrp_pct_input")

with capm2:
    tax_beta = st.number_input(
        "Tax rate for levering beta (%)",
        value=float(BANK.get("bank_tax_pct_for_beta", 25.0)),
        step=0.5,
        key="bank_tax_beta_input"
    )
    de_beta = st.number_input(
        "D/E ratio for levering beta",
        value=float(BANK.get("bank_de_ratio_for_beta", 0.0)),
        step=0.05,
        key="bank_de_beta_input"
    )
    BANK["bank_tax_pct_for_beta"] = float(tax_beta)
    BANK["bank_de_ratio_for_beta"] = float(de_beta)

with capm3:
    if not BANK.get("bank_beta_manual_mode", False):
        beta_u = st.number_input(
            "Unlevered beta (βu)",
            value=float(st.session_state.get("bank_beta_u_input_box", 1.0)),
            step=0.05,
            key="bank_beta_u_input_box"
        )
        BANK["bank_beta_u_input"] = float(beta_u)

        tax_dec = float(tax_beta) / 100.0
        de_ratio = float(de_beta)
        beta_levered = float(beta_u) * (1.0 + (1.0 - tax_dec) * de_ratio)
        st.metric("Levered beta (computed)", f"{beta_levered:.2f}")
    else:
        beta_levered = st.number_input(
            "Levered beta β (manual)",
            value=float(BANK.get("bank_beta_levered_manual", 0.22)),
            step=0.01,
            key="bank_beta_levered_manual_input"
        )
        BANK["bank_beta_levered_manual"] = float(beta_levered)

BANK["bank_rf_pct"] = float(rf_input)
BANK["bank_mrp_pct"] = float(mrp_input)

rf = BANK["bank_rf_pct"] / 100.0
mrp = BANK["bank_mrp_pct"] / 100.0
beta_levered = float(beta_levered)

ke = rf + beta_levered * mrp

st.session_state["bank_rf_pct"] = float(BANK["bank_rf_pct"])
st.session_state["bank_mrp_pct"] = float(BANK["bank_mrp_pct"])
st.session_state["bank_levered_beta"] = float(beta_levered)
st.session_state["bank_ke_pct"] = float(ke * 100)

st.markdown("#### ✅ Ke Output")
kA, kB, kC, kD = st.columns(4)
with kA:
    st.metric("Rf", f"{rf*100:.2f}%")
with kB:
    st.metric("MRP", f"{mrp*100:.2f}%")
with kC:
    st.metric("β (levered)", f"{beta_levered:.2f}")
with kD:
    st.metric("Ke (computed)", f"{ke*100:.2f}%")

# =========================================================
# (Everything below is YOUR ORIGINAL logic, unchanged)
# =========================================================

# Forecast years
n_years = st.number_input(
    "Forecast years",
    min_value=1,
    max_value=15,
    value=int(BANK.get("n_years", 5)),
    step=1,
    key="bank_n_years_input"
)
BANK["n_years"] = int(n_years)

base_year_int = int(base_year)
forecast_years = [str(base_year_int + i) for i in range(1, int(n_years) + 1)]
all_years_cols = [str(base_year)] + forecast_years

bv_0 = float(book_equity_0)
earn_0 = float(earnings_0)

# Build Balance Sheet Total Equity by Year
bs_equity_by_year = {}
for y in bs_years:
    col = bs_colmap.get(y)
    if col is None:
        continue
    val = float(bs_df.loc[equity_idx_list, col].sum(skipna=True))
    if np.isfinite(val):
        bs_equity_by_year[str(y)] = val

prev_actual_year = None
base_year_int = int(base_year)
for y in sorted([int(k) for k in bs_equity_by_year.keys() if k.isdigit()]):
    if y < base_year_int:
        prev_actual_year = str(y)

last_actual_yoy = None
if prev_actual_year is not None:
    prev_val = bs_equity_by_year.get(prev_actual_year, None)
    curr_val = bs_equity_by_year.get(str(base_year), None)
    if prev_val is not None and curr_val is not None and prev_val != 0:
        last_actual_yoy = (curr_val / prev_val) - 1.0

if last_actual_yoy is not None:
    st.caption(
        f"📌 BS Total Equity YoY (Actual): {prev_actual_year}→{base_year} = "
        f"({bs_equity_by_year[str(base_year)]:,.0f}/{bs_equity_by_year[prev_actual_year]:,.0f}) − 1 "
        f"= **{last_actual_yoy*100:.2f}%**"
    )
else:
    st.warning("⚠️ Could not compute last actual YoY from BS (missing previous year). Will fallback to input YoY%.")
# ✅ Auto-fill YoY input from BS actual YoY (but allow override)
BANK.setdefault("yoy_auto_from_bs", True)
BANK.setdefault("yoy_auto_sig", None)

auto_from_bs = st.checkbox(
    "Auto-fill Book Value YoY (%) from BS actual YoY",
    value=bool(BANK["yoy_auto_from_bs"]),
    key="bank_yoy_auto_from_bs_ui"
)
BANK["yoy_auto_from_bs"] = auto_from_bs

if auto_from_bs and (last_actual_yoy is not None):
    auto_pct = float(last_actual_yoy) * 100.0
    sig = (prev_actual_year, str(base_year), round(auto_pct, 6))

    # only refresh the input if the base year pair changed (prevents fighting user override)
    if BANK.get("yoy_auto_sig") != sig:
        BANK["yoy_uniform_pct"] = auto_pct
        st.session_state["bank_yoy_uniform_input"] = auto_pct  # 👈 this is the textbox key
        BANK["yoy_auto_sig"] = sig

st.markdown("### 📈 Book Value Growth + Risk Discount")

yoy_mode = st.radio(
    "Book Value YoY mode",
    ["Uniform", "Different per year"],
    horizontal=True,
    index=0 if BANK.get("yoy_mode", "Uniform") == "Uniform" else 1,
    key="bank_yoy_mode_input"
)
disc_mode = st.radio(
    "Discount mode",
    ["Uniform", "Different per year"],
    horizontal=True,
    index=0 if BANK.get("disc_mode", "Uniform") == "Uniform" else 1,
    key="bank_disc_mode_input"
)
BANK["yoy_mode"] = yoy_mode
BANK["disc_mode"] = disc_mode

yoy = {}
disc = {}

if yoy_mode == "Uniform":
    yoy_uniform_pct = st.number_input(
        "Book Value YoY (%)",
        value=float(BANK.get("yoy_uniform_pct", 15.0)),
        step=0.5,
        format="%.2f",
        key="bank_yoy_uniform_input"
    )
    BANK["yoy_uniform_pct"] = float(yoy_uniform_pct)
    for y in forecast_years:
        yoy[y] = float(yoy_uniform_pct) / 100.0
else:
    prev_dict = dict(BANK.get("yoy", {}))
    for y in forecast_years:
        prev = float(prev_dict.get(y, 0.15)) * 100.0 if prev_dict.get(y) is not None and prev_dict.get(y) <= 1 else float(prev_dict.get(y, 15.0))
        v = st.number_input(
            f"YoY for {y} (%)",
            value=float(prev),
            step=0.5,
            format="%.2f",
            key=f"bank_yoy_year_input_{y}"
        )
        yoy[y] = float(v) / 100.0

if disc_mode == "Uniform":
    disc_uniform_pct = st.number_input(
        "Discount (compliance/macro/transition) (%)",
        value=float(BANK.get("disc_uniform_pct", 5.0)),
        step=0.5,
        format="%.2f",
        key="bank_disc_uniform_input"
    )
    BANK["disc_uniform_pct"] = float(disc_uniform_pct)
    for y in forecast_years:
        disc[y] = float(disc_uniform_pct) / 100.0
else:
    prev_dict = dict(BANK.get("disc", {}))
    for y in forecast_years:
        prev = float(prev_dict.get(y, 0.05)) * 100.0 if prev_dict.get(y) is not None and prev_dict.get(y) <= 1 else float(prev_dict.get(y, 5.0))
        v = st.number_input(
            f"Discount for {y} (%)",
            value=float(prev),
            step=0.5,
            format="%.2f",
            key=f"bank_disc_year_input_{y}"
        )
        disc[y] = float(v) / 100.0

st.markdown("### 📊 Earnings Growth")

eps_mode = st.radio(
    "Earnings growth mode",
    ["Uniform", "Different per year"],
    horizontal=True,
    index=0 if BANK.get("eps_mode", "Uniform") == "Uniform" else 1,
    key="bank_eps_mode_input"
)
BANK["eps_mode"] = eps_mode

eps_g = {}
if eps_mode == "Uniform":
    eps_uniform_pct = st.number_input(
        "Earnings growth (%)",
        value=float(BANK.get("eps_uniform_pct", 3.0)),
        step=0.5,
        format="%.2f",
        key="bank_eps_uniform_input"
    )
    BANK["eps_uniform_pct"] = float(eps_uniform_pct)
    for y in forecast_years:
        eps_g[y] = float(eps_uniform_pct) / 100.0
else:
    prev_dict = dict(BANK.get("eps_g", {}))
    for y in forecast_years:
        prev = float(prev_dict.get(y, 0.03)) * 100.0 if prev_dict.get(y) is not None and prev_dict.get(y) <= 1 else float(prev_dict.get(y, 3.0))
        v = st.number_input(
            f"Earnings growth for {y} (%)",
            value=float(prev),
            step=0.5,
            format="%.2f",
            key=f"bank_eps_year_input_{y}"
        )
        eps_g[y] = float(v) / 100.0

dcf_g_term_pct = st.session_state.get("dcf_terminal_g_pct", None)
default_g_term = float(dcf_g_term_pct) if dcf_g_term_pct is not None else float(BANK.get("g_term_pct", 4.9))

g_term_pct = st.number_input(
    "Terminal growth g (%)",
    value=float(default_g_term),
    step=0.1,
    format="%.2f",
    key="bank_g_term_input"
)
BANK["g_term_pct"] = float(g_term_pct)
g_term = float(g_term_pct) / 100.0

BANK["yoy"] = {y: float(yoy[y]) for y in forecast_years}
BANK["disc"] = {y: float(disc[y]) for y in forecast_years}
BANK["eps_g"] = {y: float(eps_g[y]) for y in forecast_years}

st.markdown("### 🧠 YoY Forecast Rule (after Year 1)")

BANK.setdefault("yoy_rule_after_y1", "Decay (carry Adjusted YoY forward)")

yoy_rule_after_y1 = st.radio(
    "Choose how YoY behaves after the first forecast year:",
    [
        "Decay (carry Adjusted YoY forward)",
        "Hold constant (use Year-1 Adjusted YoY for all future years)"
    ],
    index=0 if BANK["yoy_rule_after_y1"].startswith("Decay") else 1,
    horizontal=True,
    key="bank_yoy_rule_after_y1_radio"
)
BANK["yoy_rule_after_y1"] = yoy_rule_after_y1

st.markdown("### ⏱ Discount Timing Convention")

BANK.setdefault("discount_t_start", "Base year t = 0 (standard)")

discount_t_start = st.radio(
    "Choose discount factor timing:",
    [
        "Base year t = 0 (standard)",
        "Base year t = 1 (shifted)"
    ],
    index=0 if str(BANK["discount_t_start"]).endswith("standard)") else 1,
    horizontal=True,
    key="bank_discount_t_start_radio"
)
BANK["discount_t_start"] = discount_t_start

DF_base = 1.0 if discount_t_start.startswith("Base year t = 0") else 1.0 / (1.0 + ke)

BV = {str(base_year): float(bv_0)}
YoY = {str(base_year): np.nan}
Discount = {str(base_year): np.nan}
AdjYoY = {str(base_year): np.nan}

EARN = {str(base_year): float(earn_0)}
EARNG = {str(base_year): np.nan}

EquityCharge = {str(base_year): -(ke * BV[str(base_year)])}
RI = {str(base_year): EARN[str(base_year)] + EquityCharge[str(base_year)]}
DF = {str(base_year): DF_base}
PV = {str(base_year): RI[str(base_year)] * DF[str(base_year)]}

prev_actual_year = str(int(base_year) - 1)
bs_prev_col = bs_colmap.get(prev_actual_year)
bs_base_col = bs_colmap.get(base_year)

yoy_from_bs_first = None
if (bs_prev_col is not None) and (bs_base_col is not None):
    equity_prev = float(bs_df.loc[equity_idx_list, bs_prev_col].sum(skipna=True))
    equity_base = float(bs_df.loc[equity_idx_list, bs_base_col].sum(skipna=True))
    if equity_prev != 0:
        yoy_from_bs_first = (equity_base / equity_prev) - 1.0

for i, y in enumerate(forecast_years, start=1):
    prev_y = str(int(y) - 1)
    Discount[y] = disc[y]

    if i == 1:
        # ✅ ALWAYS use the textbox YoY (user override) for the table + logic
        YoY[y] = float(yoy[y])
        AdjYoY[y] = YoY[y] * (1.0 - Discount[y])
        adj_yoy_hold = float(AdjYoY[y])

    else:
        rule = BANK.get("yoy_rule_after_y1", "Decay (carry Adjusted YoY forward)")
        if rule.startswith("Hold constant"):
            AdjYoY[y] = float(adj_yoy_hold)
        elif rule.startswith("Decay"):
            AdjYoY[y] = float(AdjYoY[prev_y]) * (1.0 - Discount[y])
        else:
            AdjYoY[y] = float(yoy[y]) * (1.0 - Discount[y])

        YoY[y] = AdjYoY[y] / (1.0 - Discount[y]) if (1.0 - Discount[y]) != 0 else AdjYoY[y]

    BV[y] = BV[prev_y] * (1.0 + AdjYoY[y])

    EARNG[y] = eps_g[y]
    EARN[y] = EARN[prev_y] * (1.0 + EARNG[y])

    EquityCharge[y] = -(ke * BV[y])
    RI[y] = EARN[y] + EquityCharge[y]

    if BANK["discount_t_start"].startswith("Base year t = 0"):
        t = i
    else:
        t = i + 1

    DF[y] = 1.0 / ((1.0 + ke) ** t)
    PV[y] = RI[y] * DF[y]

last_year = forecast_years[-1]

if ke <= g_term:
    terminal_value = np.nan
    pv_terminal = np.nan
else:
    terminal_value = RI[last_year] * (1.0 + g_term) / (ke - g_term)
    pv_terminal = terminal_value * DF[last_year]

ri_pv_years = [str(base_year)] + list(forecast_years)
pv_resid_sum = float(np.nansum([PV.get(y, np.nan) for y in ri_pv_years]))

pv_total = pv_resid_sum + (0.0 if np.isnan(pv_terminal) else float(pv_terminal))
equity_value_total = float(BV[str(base_year)]) + pv_total

st.markdown("### 🧾 Residual Income Valuation Table (Totals)")

rows = [
    ("Beginning Book Value (Total Equity)", BV, "money"),
    ("YoY Increase / (Decrease)", YoY, "pct"),
    ("Discount (compliance/macro/transition)", Discount, "pct"),
    ("Adjusted YoY = YoY × (1 − Discount)", AdjYoY, "pct"),
    ("Earnings (Total)", EARN, "money"),
    ("% Earnings Growth", EARNG, "pct"),
    ("(–) Equity Charge = −Ke × Book Value(beginning)", EquityCharge, "money"),
    ("Residual Income = Earnings + Equity Charge", RI, "money"),
    ("Discount factor = 1/(1+Ke)^t", DF, "df"),
    ("Present Value (PV) = RI × DF", PV, "money"),
]

df_out = pd.DataFrame({"Item": [r[0] for r in rows]})
for y in all_years_cols:
    df_out[y] = [r[1].get(y, np.nan) for r in rows]

df_out["Terminal Value"] = np.nan
df_out["PV Terminal"] = np.nan

terminal_block = pd.DataFrame({
    "Item": ["Terminal Value (from last RI)", "PV of Terminal Value"],
    **{y: [np.nan, np.nan] for y in all_years_cols},
    "Terminal Value": [terminal_value, np.nan],
    "PV Terminal": [np.nan, pv_terminal],
})

df_final = pd.concat([df_out, terminal_block], ignore_index=True)

def fmt_money(x): return "" if pd.isna(x) else f"{x:,.2f}"
def fmt_pct(x): return "" if pd.isna(x) else f"{x*100:,.2f}%"
def fmt_df(x): return "" if pd.isna(x) else f"{x:,.4f}"

styled = df_final.style.format({c: fmt_money for c in df_final.columns if c != "Item"})

pct_row_idxs = [i for i, r in enumerate(rows) if r[2] == "pct"]
df_row_idxs = [i for i, r in enumerate(rows) if r[2] == "df"]

for i in pct_row_idxs:
    styled = styled.format({c: fmt_pct for c in df_final.columns if c != "Item"}, subset=pd.IndexSlice[i, :])

for i in df_row_idxs:
    styled = styled.format({c: fmt_df for c in df_final.columns if c != "Item"}, subset=pd.IndexSlice[i, :])

styled = styled.format({c: fmt_money for c in df_final.columns if c != "Item"}, subset=pd.IndexSlice[len(rows), :])
styled = styled.format({c: fmt_money for c in df_final.columns if c != "Item"}, subset=pd.IndexSlice[len(rows)+1, :])

st.dataframe(styled, width='stretch')

st.markdown("### ✅ Implied Equity Value — Residual Income Method (Totals)")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(f"Beginning Book Value ({base_year}) [FYI]", f"{BV[str(base_year)]:,.2f}")
with k2:
    st.metric("Sum PV of Residual Income", f"{pv_resid_sum:,.2f}")
with k3:
    st.metric("PV of Terminal Value", f"{0.0 if np.isnan(pv_terminal) else pv_terminal:,.2f}")
with k4:
    st.metric("Equity Value (Total)", f"{equity_value_total:,.2f}")

BANK["outputs"] = {
    "base_year": str(base_year),
    "book_equity_0": float(book_equity_0),
    "earnings_0": float(earnings_0),
    "ke": float(ke),
    "forecast_years": list(forecast_years),
    "terminal_value": None if np.isnan(terminal_value) else float(terminal_value),
    "pv_terminal": None if np.isnan(pv_terminal) else float(pv_terminal),
    "pv_resid_sum": float(pv_resid_sum),
    "equity_value_total": float(equity_value_total),
    "currency": str(currency),
    "fx_rates_used_avg": dict(st.session_state.get("bank_yearly_fx", {})) if not currency.startswith("USD") else None,
    "fx_rates_used_bs": dict(st.session_state.get("bank_bs_fx_rates", {})) if not currency.startswith("USD") else None,
    "use_soce_for_equity": bool(BANK.get("use_soce_for_equity", True)),
    "is_sheet": str(BANK.get("is_sheet")),
    "bs_sheet": str(BANK.get("bs_sheet")),
    "soce_sheet": str(BANK.get("soce_sheet")),
}

st.session_state["equity_value_banking"] = float(equity_value_total)

