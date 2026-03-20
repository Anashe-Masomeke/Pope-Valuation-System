import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
st.set_page_config(page_title="Comparables Valuation (Excel Style)", layout="wide")
import yfinance as yf
import os
import pandas as pd
import numpy as np
from pathlib import Path
import base64
import re
import time
import random
import requests
from bs4 import BeautifulSoup

# =========================================================
# Watermark
# =========================================================
def add_watermark():
    logo_path = Path("assets") / "fbc_logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()

        watermark_css = f"""
        <style>
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
            opacity: 0.07;
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

# ---------------------------------------------------------
# Sidebar styling
# ---------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

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

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003399 0%, #001a4d 100%) !important;
    color: white !important;
    border-right: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px);
}

section[data-testid="stSidebar"] * { color: white !important; }

section[data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Font
# ---------------------------------------------------------
st.markdown(
    """
<style>
html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small {
  font-family: Georgia, "Times New Roman", serif !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Comparables Valuation – EV/EBITDA, P/B, P/E")
st.caption("All values & inputs are saved in session_state (won’t reset when switching tabs).")

S = st.session_state

# =========================================================
# API KEYS
# =========================================================
FMP_API_KEY = st.secrets.get("FMP_API_KEY", os.getenv("FMP_API_KEY", ""))
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", os.getenv("FINNHUB_API_KEY", ""))

# api/v3 usually has better coverage for profile/statements endpoints
FMP_BASE = "https://financialmodelingprep.com/api/v3"
FINNHUB_BASE = "https://finnhub.io/api/v1"

# =========================================================
# Robust HTTP session
# =========================================================
SESSION = requests.Session()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

def _safe_get(url, params=None, timeout=25, tries=3):
    last_err = None
    for i in range(int(tries)):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, headers=HEADERS)
            if r.status_code == 429:
                time.sleep(2 + i * 2 + random.random())
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(0.6 + 0.6 * i + random.random())
    raise last_err

def _safe_get_json(url, params=None, timeout=25, tries=3):
    r = _safe_get(url=url, params=params, timeout=timeout, tries=tries)
    return r.json()

def yahoo_warmup():
    try:
        _safe_get("https://finance.yahoo.com/", timeout=15, tries=2)
        _safe_get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": "test", "quotesCount": 1, "newsCount": 0},
            timeout=15,
            tries=2,
        )
    except Exception:
        pass

yahoo_warmup()

# =========================================================
# Yahoo endpoints
# =========================================================
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_QUOTESUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
def make_yahoo_profile_url(symbol: str) -> str:
    sym = normalize_peer_ticker(symbol)
    if not sym:
        return ""
    return f"https://finance.yahoo.com/quote/{sym}/profile/"
BAD_QUOTETYPES = {"FUTURE", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "ETF", "MUTUALFUND", "OPTION"}

AFRICA_SUFFIXES = (".JO", ".NG", ".KE", ".GH", ".MU", ".TZ", ".UG", ".BW", ".ZM", ".ZW")
AFRICA_EXCHANGES = {
    "JNB", "JSE", "JOH", "ZA",
    "EGX", "CAI",
    "NBO",
    "NGM", "NSI", "NGX",
    "MUN", "SEM",
}

BLOCK_SUFFIXES = (".NS", ".BO", ".L", ".TO", ".V", ".HK", ".T", ".AX", ".SA")
BLOCK_EXCHANGES = {"NSE", "BSE", "NYQ", "NMS", "NAS", "LSE", "HKG", "JPX", "TSX", "ASX", "SAO"}

# =========================================================
# Generic helpers
# =========================================================
def _clean_num(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def _first_non_null(d: dict, keys):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return None

def _clean_text(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()

def _norm_text(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower().strip())

def _looks_like_symbol(x: str) -> bool:
    x = str(x or "").strip()
    if not x:
        return False
    if " " in x:
        return False
    if len(x) > 20:
        return False
    return bool(re.fullmatch(r"[A-Z0-9.\-]+", x.upper()))

def normalize_peer_ticker(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    fixes = {
        "MTNN": "MTNN.NG",
        "MTNN.NGSE": "MTNN.NG",
        "SCOM": "SCOM.KE",
        "SAFARICOM": "SCOM.KE",
        "EQTY": "EQTY.KE",
        "KCB": "KCB.KE",
    }
    return fixes.get(sym, sym)

def _num_input_default(x, fallback=0.0):
    try:
        if x is None or pd.isna(x):
            return float(fallback)
        return float(x)
    except Exception:
        return float(fallback)

# =========================================================
# Yahoo search helpers
# =========================================================
def yahoo_lookup_html(query: str) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    try:
        r = _safe_get(
            "https://finance.yahoo.com/lookup",
            params={"s": q},
            timeout=25,
            tries=2,
        )
        tables = pd.read_html(StringIO(r.text))
    except Exception:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    if not tables:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    t = tables[0].copy()
    t.columns = [str(c).strip() for c in t.columns]

    sym_col = next((c for c in t.columns if c.lower() == "symbol"), None)
    name_col = next((c for c in t.columns if c.lower() == "name"), None)

    if not sym_col:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    out = pd.DataFrame({
        "Company": t[name_col] if name_col else "",
        "Ticker": t[sym_col],
        "Exchange": "",
        "Type": "",
    })
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["Company"] = out["Company"].astype(str).str.strip()

    out = out.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return out

def yahoo_search(query: str, quotes_count: int = 25) -> pd.DataFrame:
    q = (query or "").strip()
    if not q:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    quotes_count = int(max(5, min(int(quotes_count), 50)))
    params = {"q": q, "quotesCount": quotes_count, "newsCount": 0}

    try:
        r = _safe_get(YAHOO_SEARCH_URL, params=params, timeout=20, tries=2)
        data = r.json()

        rows = []
        for item in (data.get("quotes", []) or []):
            rows.append({
                "Company": item.get("shortname") or item.get("longname") or "",
                "Ticker": item.get("symbol") or "",
                "Exchange": item.get("exchange") or "",
                "Type": item.get("quoteType") or "",
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            return df
    except Exception:
        pass

    return yahoo_lookup_html(q)

def is_africa_quote_row(ticker: str, exchange: str, qtype: str) -> bool:
    tkr = (ticker or "").strip()
    exch = (exchange or "").strip().upper()
    qt = (qtype or "").strip().upper()

    if not tkr:
        return False
    if qt in BAD_QUOTETYPES:
        return False
    if tkr.endswith(BLOCK_SUFFIXES):
        return False
    if exch in BLOCK_EXCHANGES:
        return False
    if tkr.endswith(AFRICA_SUFFIXES):
        return True
    if exch in AFRICA_EXCHANGES:
        return True
    return False

def filter_africa(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    keep = []
    for _, r in df.iterrows():
        comp = str(r.get("Company", "") or "").strip()
        tkr = str(r.get("Ticker", "") or "").strip()
        ex = str(r.get("Exchange", "") or "").strip()
        tp = str(r.get("Type", "") or "").strip()
        if is_africa_quote_row(tkr, ex, tp):
            keep.append({
                "Company": comp,
                "Ticker": tkr,
                "Exchange": ex,
                "Type": tp,
            })

    out = pd.DataFrame(keep)
    return out.drop_duplicates(subset=["Ticker"]).reset_index(drop=True) if not out.empty else out
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_discover_sector_candidates_from_profile(
    target_symbol: str,
    target_company: str = "",
    manual_sector: str = "",
    max_results_per_query: int = 20,
) -> list:
    sym = normalize_peer_ticker(target_symbol)
    if not sym:
        return []

    prof = yahoo_profile_and_metrics(sym)

    sector_text = _clean_text(prof.get("Sector"))
    industry_text = _clean_text(prof.get("Industry"))
    desc_text = _clean_text(prof.get("Description"))
    country_text = _clean_text(prof.get("Country"))
    company_text = _clean_text(target_company or prof.get("Company"))

    if manual_sector and manual_sector.strip():
        sector_text = manual_sector.strip()
        industry_text = ""

    if not sector_text and not industry_text:
        sector_text = _clean_text(get_local_counter_sector(sym))

    queries = []
    seen_q = set()

    def add_query(q):
        q = _clean_text(q)
        if q and q.lower() not in seen_q:
            seen_q.add(q.lower())
            queries.append(q)

    if sector_text:
        add_query(sector_text)
        add_query(f"{sector_text} africa")
        add_query(f"{sector_text} listed africa")

    if industry_text:
        add_query(industry_text)
        add_query(f"{industry_text} africa")
        add_query(f"{industry_text} listed africa")

    # sector synonyms
    s = sector_text.lower()
    if s in ["telecommunications", "telecommunication", "telecom"]:
        add_query("telecommunications africa")
        add_query("mobile network operators africa")
        add_query("wireless telecom africa")
        add_query("listed telecom companies africa")

    if s == "banking":
        add_query("banks africa listed")
        add_query("commercial banks africa")

    discovered = []
    seen = set()

    for q in queries:
        try:
            df = yahoo_search(q, quotes_count=max_results_per_query)
            if df is None or df.empty:
                continue

            df = filter_africa(df)
            if df is None or df.empty:
                continue

            for _, r in df.iterrows():
                tkr = normalize_peer_ticker(r.get("Ticker", ""))
                if not tkr or tkr == sym or tkr in seen:
                    continue
                if yahoo_quote_exists(tkr):
                    seen.add(tkr)
                    discovered.append(tkr)

        except Exception:
            continue

    return discovered[:40]
@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def yahoo_quote_exists(symbol: str) -> bool:
    sym = normalize_peer_ticker(symbol)
    if not sym:
        return False
    try:
        r = _safe_get(YAHOO_QUOTE_URL, params={"symbols": sym}, timeout=15, tries=2)
        res = (r.json().get("quoteResponse") or {}).get("result") or []
        return bool(res) and str(res[0].get("symbol", "")).strip().upper() == sym.upper()
    except Exception:
        return False

# =========================================================
# ZIM COUNTERS
# =========================================================
def get_sector_file_path() -> str:
    candidates = [
        "SECTORS.xlsx",
        "assets/SECTORS.xlsx",
        "data/SECTORS.xlsx",
        "/mnt/data/SECTORS.xlsx",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""

FALLBACK_ZIM_COUNTERS = [
    {"symbol": "AFDIS", "company": "Afdis", "exchange": "ZSE", "sector": "Beverages"},
    {"symbol": "ARISTON", "company": "Ariston", "exchange": "ZSE", "sector": "Agriculture"},
    {"symbol": "ART", "company": "ART", "exchange": "ZSE", "sector": "Packaging And Paper"},
    {"symbol": "BAT", "company": "BAT", "exchange": "ZSE", "sector": "Tobacco"},
    {"symbol": "CAFCA", "company": "Cafca Limited", "exchange": "ZSE", "sector": "Electricals And Cables"},
    {"symbol": "CBZ", "company": "CBZ", "exchange": "ZSE", "sector": "Banking"},
    {"symbol": "CFI", "company": "CFI", "exchange": "ZSE", "sector": "Agriculture"},
    {"symbol": "DAIRIBORD", "company": "Dairibord", "exchange": "ZSE", "sector": "Food And Dairy"},
    {"symbol": "DELTA", "company": "Delta", "exchange": "ZSE", "sector": "Beverages"},
    {"symbol": "ECOZIM", "company": "Econet Wireless Zimbabwe", "exchange": "ZSE", "sector": "Telecommunications"},
    {"symbol": "FBC", "company": "FBC", "exchange": "ZSE", "sector": "Banking"},
    {"symbol": "FIDELITY", "company": "Fidelity", "exchange": "ZSE", "sector": "Financial Services"},
    {"symbol": "FML", "company": "FML", "exchange": "ZSE", "sector": "Insurance"},
    {"symbol": "FMP", "company": "FMP", "exchange": "ZSE", "sector": "Real Estate"},
    {"symbol": "GB", "company": "General Beltings", "exchange": "ZSE", "sector": "Industrial Products"},
    {"symbol": "HIPPO", "company": "Hippo", "exchange": "ZSE", "sector": "Agriculture"},
    {"symbol": "MASH", "company": "Mash Holdings", "exchange": "ZSE", "sector": "Real Estate"},
    {"symbol": "MASIMBA", "company": "Masimba", "exchange": "ZSE", "sector": "Construction"},
    {"symbol": "MEIKLES", "company": "Meikles", "exchange": "ZSE", "sector": "Consumer Services"},
    {"symbol": "NAMPAK", "company": "Nampak", "exchange": "ZSE", "sector": "Packaging And Paper"},
    {"symbol": "NMBZ", "company": "NMBZ", "exchange": "ZSE", "sector": "Banking"},
    {"symbol": "OKZIM", "company": "OK Zimbabwe", "exchange": "ZSE", "sector": "Retail"},
    {"symbol": "PROPLASTICS", "company": "Proplastics", "exchange": "ZSE", "sector": "Plastics"},
    {"symbol": "RIOZIM", "company": "RioZim", "exchange": "ZSE", "sector": "Mining"},
    {"symbol": "RTG", "company": "RTG", "exchange": "ZSE", "sector": "Hotels And Leisure"},
    {"symbol": "SEEDCO", "company": "Seed Co Limited", "exchange": "ZSE", "sector": "Agriculture"},
    {"symbol": "STARAFRICA", "company": "Starafrica", "exchange": "ZSE", "sector": "Food Processing"},
    {"symbol": "TANGANDA", "company": "Tanganda", "exchange": "ZSE", "sector": "Agriculture"},
    {"symbol": "TN", "company": "TN Cybertech Investments Holdings Limited", "exchange": "ZSE", "sector": "Technology"},
    {"symbol": "TSL", "company": "TSL", "exchange": "ZSE", "sector": "Logistics And Agriculture"},
    {"symbol": "TURNALL", "company": "Turnall", "exchange": "ZSE", "sector": "Building Materials"},
    {"symbol": "UNIFREIGHT", "company": "Unifreight", "exchange": "ZSE", "sector": "Logistics"},
    {"symbol": "WILLDALE", "company": "Willdale", "exchange": "ZSE", "sector": "Building Materials"},
    {"symbol": "ZB", "company": "ZB", "exchange": "ZSE", "sector": "Banking"},
    {"symbol": "ZECO", "company": "ZECO", "exchange": "ZSE", "sector": "Engineering"},
    {"symbol": "ZIMPAPERS", "company": "Zimpapers", "exchange": "ZSE", "sector": "Media"},
    {"symbol": "ZIMRE", "company": "Zimre", "exchange": "ZSE", "sector": "Insurance"},
    {"symbol": "ZSEH", "company": "ZSE Holdings", "exchange": "ZSE", "sector": "Exchange Services"},
    {"symbol": "ASUN", "company": "African Sun", "exchange": "VFEX", "sector": "Hotels And Leisure"},
    {"symbol": "AXIA", "company": "Axia", "exchange": "VFEX", "sector": "Retail"},
    {"symbol": "CMCL", "company": "Caledonia", "exchange": "VFEX", "sector": "Gold Mining"},
    {"symbol": "EDGARS", "company": "Edgars", "exchange": "VFEX", "sector": "Retail"},
    {"symbol": "FCB", "company": "First Capital Bank Limited", "exchange": "VFEX", "sector": "Banking"},
    {"symbol": "INNSCOR", "company": "Innscor Africa Limited", "exchange": "VFEX", "sector": "Food And Consumer"},
    {"symbol": "INVICTUS", "company": "Invictus Energy ZDR", "exchange": "VFEX", "sector": "Energy"},
    {"symbol": "KAVANGO", "company": "Kavango Resources Plc", "exchange": "VFEX", "sector": "Mining Exploration"},
    {"symbol": "NEDBANKZDR", "company": "Nedbank ZDR", "exchange": "VFEX", "sector": "Banking"},
    {"symbol": "PADENGA", "company": "Padenga", "exchange": "VFEX", "sector": "Gold Mining"},
    {"symbol": "SEEDINT", "company": "Seed Co International", "exchange": "VFEX", "sector": "Agriculture"},
    {"symbol": "SIMBISA", "company": "Simbisa", "exchange": "VFEX", "sector": "Quick Service Restaurants"},
    {"symbol": "WESTPROP", "company": "WestProp Holdings Limited", "exchange": "VFEX", "sector": "Real Estate"},
    {"symbol": "ZIMPLOW", "company": "Zimplow Holdings Limited", "exchange": "VFEX", "sector": "Industrial Equipment"},
    {"symbol": "TIGERE", "company": "TIGERE REIT", "exchange": "VFEX", "sector": "Real Estate"},
]

@st.cache_data(show_spinner=False)
def load_zim_counters_from_excel(path: str) -> list:
    if not path or not Path(path).exists():
        return []

    try:
        raw = pd.read_excel(path, header=None)
    except Exception:
        return []

    header_row = None
    for i in range(len(raw)):
        row_vals = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if "counter" in row_vals and "sector" in row_vals:
            header_row = i
            break

    if header_row is None:
        return []

    try:
        df = pd.read_excel(path, header=header_row)
    except Exception:
        return []

    df.columns = [str(c).strip() for c in df.columns]

    counter_col = next((c for c in df.columns if str(c).strip().lower() == "counter"), None)
    sector_col = next((c for c in df.columns if str(c).strip().lower() == "sector"), None)

    if counter_col is None or sector_col is None:
        return []

    df = df[[counter_col, sector_col]].copy()
    df.columns = ["COUNTER", "SECTOR"]

    df["COUNTER"] = df["COUNTER"].fillna("").astype(str).str.strip()
    df["SECTOR"] = df["SECTOR"].fillna("").astype(str).str.strip()

    out = []
    seen = set()
    for _, r in df.iterrows():
        company = str(r["COUNTER"]).strip()
        sector = str(r["SECTOR"]).strip()

        if not company or company.upper() == "COUNTER":
            continue

        key = _norm_text(company)
        if key in seen:
            continue
        seen.add(key)

        fallback_match = next(
            (x for x in FALLBACK_ZIM_COUNTERS if _norm_text(x["company"]) == key or _norm_text(x["symbol"]) == key),
            None
        )

        if fallback_match:
            out.append({
                "symbol": fallback_match["symbol"],
                "company": fallback_match["company"],
                "exchange": fallback_match["exchange"],
                "sector": sector if sector else fallback_match["sector"],
            })
        else:
            out.append({
                "symbol": company.upper().replace(" ", ""),
                "company": company,
                "exchange": "ZSE",
                "sector": sector,
            })

    return out

SECTOR_FILE = get_sector_file_path()
ZIM_COUNTERS = load_zim_counters_from_excel(SECTOR_FILE)
if not ZIM_COUNTERS:
    ZIM_COUNTERS = FALLBACK_ZIM_COUNTERS

# =========================================================
# Local company map
# =========================================================
LOCAL_COMPANY_MAP = {
    "mtn": {"symbol": "MTN.JO", "company": "MTN Group Limited", "exchange": "JSE", "source": "local_map"},
    "vodacom": {"symbol": "VOD.JO", "company": "Vodacom Group Limited", "exchange": "JSE", "source": "local_map"},
    "econet": {"symbol": "ECOZIM", "company": "Econet Wireless Zimbabwe", "exchange": "ZSE", "source": "local_map"},
}

# =========================================================
# Company-specific peer map
# =========================================================
COMPANY_SPECIFIC_PEERS = {
    "PADENGA": [
        {"ticker": "GFI.JO", "company": "Gold Fields", "country": "South Africa"},
        {"ticker": "ANG.JO", "company": "AngloGold Ashanti", "country": "South Africa"},
        {"ticker": "HAR.JO", "company": "Harmony Gold Mining Company Limited", "country": "South Africa"},
    ],
    "CMCL": [
        {"ticker": "GFI.JO", "company": "Gold Fields", "country": "South Africa"},
        {"ticker": "ANG.JO", "company": "AngloGold Ashanti", "country": "South Africa"},
        {"ticker": "HAR.JO", "company": "Harmony Gold Mining Company Limited", "country": "South Africa"},
    ],
    "DELTA": [
        {"ticker": "ANH.JO", "company": "Anheuser-Busch InBev", "country": "South Africa"},
        {"ticker": "HEIA.AS", "company": "Heineken N.V.", "country": "Netherlands"},
        {"ticker": "HEIO.AS", "company": "Heineken Holding", "country": "Netherlands"},
        {"ticker": "ABEV", "company": "Ambev S.A.", "country": "Brazil"},
        {"ticker": "AFDIS", "company": "African Distillers", "country": "Zimbabwe"},
    ],
    "AFDIS": [
        {"ticker": "ANH.JO", "company": "Anheuser-Busch InBev", "country": "South Africa"},
        {"ticker": "HEIA.AS", "company": "Heineken N.V.", "country": "Netherlands"},
        {"ticker": "HEIO.AS", "company": "Heineken Holding", "country": "Netherlands"},
        {"ticker": "ABEV", "company": "Ambev S.A.", "country": "Brazil"},
        {"ticker": "DELTA", "company": "Delta", "country": "Zimbabwe"},
    ],
    "ECOZIM": [
        {"ticker": "MTN.JO", "company": "MTN Group Limited", "country": "South Africa"},
        {"ticker": "VOD.JO", "company": "Vodacom Group Limited", "country": "South Africa"},
        {"ticker": "MTNN.NG", "company": "MTN Nigeria Communications Plc", "country": "Nigeria"},
        {"ticker": "SCOM.KE", "company": "Safaricom Plc", "country": "Kenya"},
    ],
    "CBZ": [
        {"ticker": "FSR.JO", "company": "FirstRand", "country": "South Africa"},
        {"ticker": "SBK.JO", "company": "Standard Bank Group", "country": "South Africa"},
        {"ticker": "NED.JO", "company": "Nedbank Group", "country": "South Africa"},
        {"ticker": "ABG.JO", "company": "Absa Group", "country": "South Africa"},
        {"ticker": "KCB.KE", "company": "KCB Group", "country": "Kenya"},
        {"ticker": "EQTY.KE", "company": "Equity Group", "country": "Kenya"},
    ],
    "FBC": [
        {"ticker": "FSR.JO", "company": "FirstRand", "country": "South Africa"},
        {"ticker": "SBK.JO", "company": "Standard Bank Group", "country": "South Africa"},
        {"ticker": "NED.JO", "company": "Nedbank Group", "country": "South Africa"},
        {"ticker": "ABG.JO", "company": "Absa Group", "country": "South Africa"},
    ],
    "NMBZ": [
        {"ticker": "FSR.JO", "company": "FirstRand", "country": "South Africa"},
        {"ticker": "SBK.JO", "company": "Standard Bank Group", "country": "South Africa"},
        {"ticker": "NED.JO", "company": "Nedbank Group", "country": "South Africa"},
        {"ticker": "ABG.JO", "company": "Absa Group", "country": "South Africa"},
    ],
    "ZB": [
        {"ticker": "FSR.JO", "company": "FirstRand", "country": "South Africa"},
        {"ticker": "SBK.JO", "company": "Standard Bank Group", "country": "South Africa"},
        {"ticker": "NED.JO", "company": "Nedbank Group", "country": "South Africa"},
        {"ticker": "ABG.JO", "company": "Absa Group", "country": "South Africa"},
    ],
    "ZIMRE": [
        {"ticker": "SLM.JO", "company": "Sanlam", "country": "South Africa"},
        {"ticker": "DSY.JO", "company": "Discovery", "country": "South Africa"},
        {"ticker": "OMU.JO", "company": "Old Mutual", "country": "South Africa"},
    ],
    "FML": [
        {"ticker": "SLM.JO", "company": "Sanlam", "country": "South Africa"},
        {"ticker": "DSY.JO", "company": "Discovery", "country": "South Africa"},
        {"ticker": "OMU.JO", "company": "Old Mutual", "country": "South Africa"},
    ],
    "INNSCOR": [
        {"ticker": "TBS.JO", "company": "Tiger Brands", "country": "South Africa"},
        {"ticker": "RCL.JO", "company": "RCL Foods", "country": "South Africa"},
        {"ticker": "FBR.JO", "company": "Famous Brands", "country": "South Africa"},
        {"ticker": "OCE.JO", "company": "Oceana Group", "country": "South Africa"},
    ],
    "WESTPROP": [
        {"ticker": "GRT.JO", "company": "Growthpoint Properties", "country": "South Africa"},
        {"ticker": "RES.JO", "company": "Resilient REIT", "country": "South Africa"},
        {"ticker": "HYP.JO", "company": "Hyprop", "country": "South Africa"},
        {"ticker": "FFB.JO", "company": "Fortress", "country": "South Africa"},
    ],
    "RTG": [
        {"ticker": "SSU.JO", "company": "Southern Sun", "country": "South Africa"},
        {"ticker": "ASUN", "company": "African Sun", "country": "Zimbabwe"},
    ],
    "ASUN": [
        {"ticker": "SSU.JO", "company": "Southern Sun", "country": "South Africa"},
        {"ticker": "RTG", "company": "Rainbow Tourism Group", "country": "Zimbabwe"},
    ],
}

SECTOR_PROXY_MAP = {
    "telecommunications": ["MTN.JO", "VOD.JO", "MTNN.NG", "SCOM.KE"],
    "banking": ["FSR.JO", "SBK.JO", "NED.JO", "ABG.JO", "KCB.KE", "EQTY.KE"],
    "financial services": ["FSR.JO", "SBK.JO", "NED.JO", "ABG.JO", "SLM.JO", "DSY.JO"],
    "insurance": ["SLM.JO", "DSY.JO", "OMU.JO"],
    "beverages": ["ANH.JO", "HEIA.AS", "HEIO.AS", "ABEV", "AFDIS"],
    "food and dairy": ["NESTLE.NG", "ILH.JO", "BVT.JO", "FBR.JO"],
    "food and consumer": ["NESTLE.NG", "ILH.JO", "BVT.JO", "ANH.JO", "FBR.JO"],
    "consumer services": ["WHL.JO", "MRP.JO", "TRU.JO"],
    "retail": ["WHL.JO", "MRP.JO", "TRU.JO", "TFG.JO", "SHP.JO"],
    "quick service restaurants": ["FBR.JO", "WHL.JO", "MRP.JO"],
    "gold mining": ["GFI.JO", "ANG.JO", "HAR.JO"],
    "mining": ["GFI.JO", "ANG.JO", "HAR.JO", "IMP.JO", "AMS.JO"],
    "mining exploration": ["GFI.JO", "ANG.JO", "HAR.JO"],
    "real estate": ["GRT.JO", "RES.JO", "HYP.JO", "FFB.JO"],
    "construction": ["MNP.JO", "AIP.JO", "BAW.JO"],
    "industrial products": ["KAP.JO", "BAW.JO", "MNP.JO"],
    "industrial equipment": ["KAP.JO", "BAW.JO", "MNP.JO"],
    "engineering": ["BAW.JO", "MNP.JO", "AIP.JO"],
    "electricals and cables": ["ARI.JO", "KAP.JO"],
    "building materials": ["PPC.JO", "MNP.JO"],
    "packaging and paper": ["NPK.JO", "KAP.JO"],
    "plastics": ["KAP.JO", "NPK.JO"],
    "agriculture": ["SEEDINT", "NESTLE.NG"],
    "food processing": ["NESTLE.NG", "BVT.JO", "FBR.JO"],
    "logistics": ["IMP.JO", "BAW.JO"],
    "logistics and agriculture": ["IMP.JO", "BAW.JO", "SEEDINT"],
    "tobacco": ["BTI", "OMN.JO"],
    "technology": ["MTN.JO", "VOD.JO"],
    "media": ["NPN.JO"],
    "exchange services": ["JSE.JO"],
    "energy": ["SOL.JO"],
}

# =========================================================
# Local matching helpers
# =========================================================
def find_local_counter(query: str) -> dict:
    q = _norm_text(query)
    if not q:
        return {}

    mapped = LOCAL_COMPANY_MAP.get(str(query).strip().lower())
    if mapped:
        return mapped

    for row in ZIM_COUNTERS:
        if _norm_text(row["symbol"]) == q or _norm_text(row["company"]) == q:
            return {
                "symbol": row["symbol"],
                "company": row["company"],
                "exchange": row["exchange"],
                "source": "zim_local_universe",
            }

    for row in ZIM_COUNTERS:
        if q in _norm_text(row["company"]):
            return {
                "symbol": row["symbol"],
                "company": row["company"],
                "exchange": row["exchange"],
                "source": "zim_local_universe",
            }

    return {}

def get_local_counter_sector(target_symbol: str) -> str:
    sym = str(target_symbol or "").strip().upper()
    if not sym:
        return ""

    for row in ZIM_COUNTERS:
        if str(row.get("symbol", "")).strip().upper() == sym:
            return str(row.get("sector", "")).strip()

    return ""

def get_company_specific_proxy_peers(target_symbol: str, target_company: str = "", max_peers: int = 8) -> list:
    sym = str(target_symbol or "").strip().upper()
    nm = str(target_company or "").strip().upper()

    candidates = []

    if sym in COMPANY_SPECIFIC_PEERS:
        candidates = COMPANY_SPECIFIC_PEERS[sym]
    else:
        for key, vals in COMPANY_SPECIFIC_PEERS.items():
            if nm and key in nm:
                candidates = vals
                break

    out = []
    seen = set()
    for row in candidates:
        tkr = normalize_peer_ticker(row.get("ticker", ""))
        if tkr and tkr not in seen:
            seen.add(tkr)
            out.append(tkr)

    return out[:max_peers]

def get_africa_sector_proxy_peers(sector_keyword: str, max_peers: int = 8) -> list:
    s = str(sector_keyword or "").strip().lower()
    return [normalize_peer_ticker(x) for x in SECTOR_PROXY_MAP.get(s, [])[:max_peers]]

# =========================================================
# Peer name lookup
# =========================================================
def build_known_peer_name_map() -> dict:
    peer_name_map = {}

    for _, peer_list in COMPANY_SPECIFIC_PEERS.items():
        for row in peer_list:
            tkr = normalize_peer_ticker(row.get("ticker", ""))
            nm = str(row.get("company", "")).strip()
            ctry = str(row.get("country", "")).strip()
            if tkr and nm:
                peer_name_map[tkr] = {
                    "company": nm,
                    "country": ctry,
                    "exchange": "",
                }

    for row in ZIM_COUNTERS:
        tkr = str(row.get("symbol", "")).strip().upper()
        nm = str(row.get("company", "")).strip()
        ex = str(row.get("exchange", "")).strip()
        if tkr and nm:
            peer_name_map[tkr] = {
                "company": nm,
                "country": "Zimbabwe" if ex in ["ZSE", "VFEX"] else "",
                "exchange": ex,
            }

    manual_names = {
        "FSR.JO": {"company": "FirstRand Limited", "country": "South Africa", "exchange": "JSE"},
        "SBK.JO": {"company": "Standard Bank Group Limited", "country": "South Africa", "exchange": "JSE"},
        "NED.JO": {"company": "Nedbank Group Limited", "country": "South Africa", "exchange": "JSE"},
        "ABG.JO": {"company": "Absa Group Limited", "country": "South Africa", "exchange": "JSE"},
        "KCB.KE": {"company": "KCB Group Plc", "country": "Kenya", "exchange": "NSE"},
        "EQTY.KE": {"company": "Equity Group Holdings Plc", "country": "Kenya", "exchange": "NSE"},
        "SLM.JO": {"company": "Sanlam Limited", "country": "South Africa", "exchange": "JSE"},
        "DSY.JO": {"company": "Discovery Limited", "country": "South Africa", "exchange": "JSE"},
        "OMU.JO": {"company": "Old Mutual Limited", "country": "South Africa", "exchange": "JSE"},
        "MTN.JO": {"company": "MTN Group Limited", "country": "South Africa", "exchange": "JSE"},
        "VOD.JO": {"company": "Vodacom Group Limited", "country": "South Africa", "exchange": "JSE"},
        "MTNN.NG": {"company": "MTN Nigeria Communications Plc", "country": "Nigeria", "exchange": "NGX"},
        "SCOM.KE": {"company": "Safaricom Plc", "country": "Kenya", "exchange": "NSE"},
        "GFI.JO": {"company": "Gold Fields Limited", "country": "South Africa", "exchange": "JSE"},
        "ANG.JO": {"company": "AngloGold Ashanti Plc", "country": "South Africa", "exchange": "JSE"},
        "HAR.JO": {"company": "Harmony Gold Mining Company Limited", "country": "South Africa", "exchange": "JSE"},
        "GRT.JO": {"company": "Growthpoint Properties Limited", "country": "South Africa", "exchange": "JSE"},
        "RES.JO": {"company": "Resilient REIT Limited", "country": "South Africa", "exchange": "JSE"},
        "HYP.JO": {"company": "Hyprop Investments Limited", "country": "South Africa", "exchange": "JSE"},
        "FFB.JO": {"company": "Fortress Real Estate Investments Limited", "country": "South Africa", "exchange": "JSE"},
        "HEIA.AS": {"company": "Heineken N.V.", "country": "Netherlands", "exchange": "Euronext Amsterdam"},
        "HEIO.AS": {"company": "Heineken Holding N.V.", "country": "Netherlands", "exchange": "Euronext Amsterdam"},
        "ABEV": {"company": "Ambev S.A.", "country": "Brazil", "exchange": "NYSE"},
        "ANH.JO": {"company": "Anheuser-Busch InBev", "country": "South Africa", "exchange": "JSE"},
        "NESTLE.NG": {"company": "Nestlé Nigeria Plc", "country": "Nigeria", "exchange": "NGX"},
        "BVT.JO": {"company": "Brimstone / Consumer Proxy", "country": "South Africa", "exchange": "JSE"},
        "WHL.JO": {"company": "Woolworths Holdings Limited", "country": "South Africa", "exchange": "JSE"},
        "MRP.JO": {"company": "Mr Price Group Limited", "country": "South Africa", "exchange": "JSE"},
        "TRU.JO": {"company": "Truworths International Limited", "country": "South Africa", "exchange": "JSE"},
        "TFG.JO": {"company": "The Foschini Group Limited", "country": "South Africa", "exchange": "JSE"},
        "SHP.JO": {"company": "Shoprite Holdings Limited", "country": "South Africa", "exchange": "JSE"},
        "SSU.JO": {"company": "Southern Sun Limited", "country": "South Africa", "exchange": "JSE"},
        "SOL.JO": {"company": "Sasol Limited", "country": "South Africa", "exchange": "JSE"},
        "BAW.JO": {"company": "Barloworld Limited", "country": "South Africa", "exchange": "JSE"},
        "MNP.JO": {"company": "Mondi Plc", "country": "South Africa", "exchange": "JSE"},
        "AIP.JO": {"company": "Adcock Ingram Holdings Limited", "country": "South Africa", "exchange": "JSE"},
        "KAP.JO": {"company": "KAP Limited", "country": "South Africa", "exchange": "JSE"},
        "PPC.JO": {"company": "PPC Ltd", "country": "South Africa", "exchange": "JSE"},
        "ARI.JO": {"company": "African Rainbow Minerals / Industrial Proxy", "country": "South Africa", "exchange": "JSE"},
        "IMP.JO": {"company": "Impala Platinum Holdings Limited", "country": "South Africa", "exchange": "JSE"},
        "AMS.JO": {"company": "Anglo American Platinum Limited", "country": "South Africa", "exchange": "JSE"},
        "NPN.JO": {"company": "Naspers Limited", "country": "South Africa", "exchange": "JSE"},
        "JSE.JO": {"company": "JSE Limited", "country": "South Africa", "exchange": "JSE"},
        "BTI": {"company": "British American Tobacco p.l.c.", "country": "United Kingdom", "exchange": "NYSE"},
        "FBR.JO": {"company": "Famous Brands Limited", "country": "South Africa", "exchange": "JSE"},
        "RTG": {"company": "Rainbow Tourism Group", "country": "Zimbabwe", "exchange": "ZSE"},
        "NPK.JO": {"company": "Nampak Limited", "country": "South Africa", "exchange": "JSE"},
        "TBS.JO": {"company": "Tiger Brands Limited", "country": "South Africa", "exchange": "JSE"},
        "RCL.JO": {"company": "RCL Foods Limited", "country": "South Africa", "exchange": "JSE"},
        "FBR.JO": {"company": "Famous Brands Limited", "country": "South Africa", "exchange": "JSE"},
        "OCE.JO": {"company": "Oceana Group Limited", "country": "South Africa", "exchange": "JSE"},
    }

    peer_name_map.update(manual_names)
    return peer_name_map

KNOWN_PEER_NAME_MAP = build_known_peer_name_map()

# =========================================================
# Live market data functions
# =========================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fmp_get_peers(symbol: str, limit: int = 10) -> list:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FMP_API_KEY:
        return []

    try:
        data = _safe_get_json(
            url=f"{FMP_BASE}/stock_peers",
            params={"symbol": sym, "apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )

        peers = []
        if isinstance(data, list):
            peers.extend(data)
        elif isinstance(data, dict):
            peers.extend(data.get("peersList", []) or [])
            peers.extend(data.get("peers", []) or [])

        peers = [normalize_peer_ticker(x) for x in peers if str(x).strip()]
        peers = [x for x in peers if x != sym]

        out = []
        seen = set()
        for p in peers:
            if p not in seen:
                seen.add(p)
                out.append(p)

        return out[:limit]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def finnhub_get_peers(symbol: str, limit: int = 10) -> list:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FINNHUB_API_KEY:
        return []

    try:
        data = _safe_get_json(
            url=f"{FINNHUB_BASE}/stock/peers",
            params={"symbol": sym, "token": FINNHUB_API_KEY},
            timeout=25,
            tries=2,
        )
        peers = [normalize_peer_ticker(x) for x in (data or []) if str(x).strip()]
        peers = [x for x in peers if x != sym]
        return peers[:limit]
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fmp_profile(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FMP_API_KEY:
        return {}

    try:
        data = _safe_get_json(
            url=f"{FMP_BASE}/profile/{sym}",
            params={"apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        if not row:
            return {}

        return {
            "Company": _first_non_null(row, ["companyName", "name", "symbol"]) or "",
            "Ticker": sym,
            "Sector": _first_non_null(row, ["sector"]) or "",
            "Industry": _first_non_null(row, ["industry"]) or "",
            "Exchange": _first_non_null(row, ["exchangeShortName", "exchange"]) or "",
            "Country": _first_non_null(row, ["country"]) or "",
        }
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fmp_ratios(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FMP_API_KEY:
        return {}

    out = {"Ticker": sym, "P/E": np.nan, "P/B": np.nan, "EV/EBITDA": np.nan, "ratio_source": ""}

    try:
        ratios = _safe_get_json(
            url=f"{FMP_BASE}/ratios-ttm/{sym}",
            params={"apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = ratios[0] if isinstance(ratios, list) and ratios else (ratios if isinstance(ratios, dict) else {})

        pe = _clean_num(_first_non_null(row, [
            "priceEarningsRatioTTM", "peRatioTTM", "peTTM", "priceToEarningsRatioTTM"
        ]))
        pb = _clean_num(_first_non_null(row, [
            "priceToBookRatioTTM", "pbRatioTTM", "pbTTM", "priceToBookTTM"
        ]))

        out["P/E"] = pe
        out["P/B"] = pb
    except Exception:
        pass

    try:
        km = _safe_get_json(
            url=f"{FMP_BASE}/key-metrics-ttm/{sym}",
            params={"apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = km[0] if isinstance(km, list) and km else (km if isinstance(km, dict) else {})
        ev = _clean_num(_first_non_null(row, [
            "enterpriseValueOverEBITDATTM", "evToEbitdaTTM", "enterpriseValueOverEBITDA", "evToEbitda"
        ]))
        out["EV/EBITDA"] = ev
    except Exception:
        pass

    if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
        out["ratio_source"] = "FMP"

    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fmp_raw_fundamentals(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FMP_API_KEY:
        return {}

    out = {
        "Ticker": sym,
        "market_cap": np.nan,
        "total_debt": np.nan,
        "cash": np.nan,
        "net_income": np.nan,
        "book_value_equity": np.nan,
        "ebitda": np.nan,
        "raw_source": ""
    }

    try:
        prof = _safe_get_json(
            url=f"{FMP_BASE}/profile/{sym}",
            params={"apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = prof[0] if isinstance(prof, list) and prof else (prof if isinstance(prof, dict) else {})
        out["market_cap"] = _clean_num(_first_non_null(row, ["mktCap", "marketCap"]))
    except Exception:
        pass

    try:
        bs = _safe_get_json(
            url=f"{FMP_BASE}/balance-sheet-statement/{sym}",
            params={"limit": 1, "apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = bs[0] if isinstance(bs, list) and bs else (bs if isinstance(bs, dict) else {})
        out["cash"] = _clean_num(_first_non_null(row, ["cashAndCashEquivalents", "cashAndShortTermInvestments"]))
        out["total_debt"] = _clean_num(_first_non_null(row, ["totalDebt", "shortTermDebt", "longTermDebt"]))
        out["book_value_equity"] = _clean_num(_first_non_null(row, ["totalStockholdersEquity", "totalEquity"]))
    except Exception:
        pass

    try:
        is_ = _safe_get_json(
            url=f"{FMP_BASE}/income-statement/{sym}",
            params={"limit": 1, "apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )
        row = is_[0] if isinstance(is_, list) and is_ else (is_ if isinstance(is_, dict) else {})
        out["net_income"] = _clean_num(_first_non_null(row, ["netIncome", "netIncomeCommonStockholders"]))
        out["ebitda"] = _clean_num(_first_non_null(row, ["ebitda"]))
    except Exception:
        pass

    if any(not pd.isna(out[k]) for k in [
        "market_cap", "total_debt", "cash", "net_income", "book_value_equity", "ebitda"
    ]):
        out["raw_source"] = "FMP raw fundamentals"

    return out


def derive_ratios_from_raw(raw: dict) -> dict:
    out = {
        "Ticker": raw.get("Ticker", ""),
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": ""
    }

    market_cap = _clean_num(raw.get("market_cap"))
    total_debt = _clean_num(raw.get("total_debt"))
    cash = _clean_num(raw.get("cash"))
    net_income = _clean_num(raw.get("net_income"))
    book_value_equity = _clean_num(raw.get("book_value_equity"))
    ebitda = _clean_num(raw.get("ebitda"))

    if not pd.isna(market_cap) and not pd.isna(net_income) and net_income != 0:
        out["P/E"] = market_cap / net_income

    if not pd.isna(market_cap) and not pd.isna(book_value_equity) and book_value_equity != 0:
        out["P/B"] = market_cap / book_value_equity

    if not pd.isna(market_cap) and not pd.isna(ebitda) and ebitda != 0:
        debt = 0.0 if pd.isna(total_debt) else total_debt
        csh = 0.0 if pd.isna(cash) else cash
        enterprise_value = market_cap + debt - csh
        out["EV/EBITDA"] = enterprise_value / ebitda

    if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
        out["ratio_source"] = "Derived from raw fundamentals"

    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def finnhub_basic_metrics(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym or not FINNHUB_API_KEY:
        return {}

    out = {"Ticker": sym, "P/E": np.nan, "P/B": np.nan, "EV/EBITDA": np.nan, "ratio_source": ""}

    try:
        data = _safe_get_json(
            url=f"{FINNHUB_BASE}/stock/metric",
            params={"symbol": sym, "metric": "all", "token": FINNHUB_API_KEY},
            timeout=25,
            tries=2,
        )
        m = (data or {}).get("metric", {}) or {}

        pe = _clean_num(_first_non_null(m, ["peTTM", "peAnnual"]))
        pb = _clean_num(_first_non_null(m, ["pbAnnual"]))
        ev = _clean_num(_first_non_null(m, ["evEbitdaTTM"]))

        out["P/E"] = pe
        out["P/B"] = pb
        out["EV/EBITDA"] = ev

        if not (pd.isna(pe) and pd.isna(pb) and pd.isna(ev)):
            out["ratio_source"] = "Finnhub stock/metric"
    except Exception:
        pass

    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_quote_profile(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym:
        return {}

    try:
        r = _safe_get(
            YAHOO_QUOTE_URL,
            params={"symbols": sym},
            timeout=20,
            tries=2,
        )
        results = ((r.json() or {}).get("quoteResponse") or {}).get("result") or []
        if not results:
            return {}

        row = results[0]
        return {
            "Company": row.get("longName") or row.get("shortName") or "",
            "Ticker": sym,
            "Exchange": row.get("fullExchangeName") or row.get("exchange") or "",
            "Country": row.get("region") or "",
            "Sector": "",
            "Industry": "",
        }
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_profile_and_metrics(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym:
        return {}

    def _raw(v):
        if isinstance(v, dict):
            if "raw" in v:
                return v.get("raw")
            if "fmt" in v:
                return v.get("fmt")
        return v

    out = {
        "Ticker": sym,
        "Company": "",
        "Exchange": "",
        "Country": "",
        "Sector": "",
        "Industry": "",
        "Description": "",
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": "",
    }

    try:
        url = YAHOO_QUOTESUMMARY_URL.format(symbol=sym)
        r = _safe_get(
            url,
            params={
                "modules": "price,summaryDetail,defaultKeyStatistics,financialData,assetProfile"
            },
            timeout=25,
            tries=2,
        )
        data = r.json()
        res = (((data.get("quoteSummary") or {}).get("result")) or [])
        if not res:
            return out

        root = res[0]
        price = root.get("price") or {}
        summary = root.get("summaryDetail") or {}
        dks = root.get("defaultKeyStatistics") or {}
        fin = root.get("financialData") or {}
        ap = root.get("assetProfile") or {}

        out["Company"] = (
            _raw(price.get("longName"))
            or _raw(price.get("shortName"))
            or sym
        )
        out["Exchange"] = _raw(price.get("exchangeName")) or _raw(price.get("fullExchangeName")) or ""
        out["Country"] = ap.get("country") or ""
        out["Sector"] = ap.get("sector") or ""
        out["Industry"] = ap.get("industry") or ""
        out["Description"] = ap.get("longBusinessSummary") or ""

        pe = _raw(summary.get("trailingPE")) or _raw(dks.get("trailingPE")) or _raw(fin.get("trailingPE"))
        pb = _raw(dks.get("priceToBook")) or _raw(summary.get("priceToBook")) or _raw(fin.get("priceToBook"))
        evebitda = _raw(fin.get("enterpriseToEbitda")) or _raw(dks.get("enterpriseToEbitda"))

        out["P/E"] = _clean_num(pe)
        out["P/B"] = _clean_num(pb)
        out["EV/EBITDA"] = _clean_num(evebitda)

        if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
            out["ratio_source"] = "Yahoo quoteSummary"
    except Exception:
        pass

    return out
def yahoo_stats_table_fallback(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)

    out = {
        "Ticker": sym,
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": ""
    }

    if not sym:
        return out

    def parse_ratio_value(x):
        s = str(x).strip()
        if s in ["", "N/A", "NaN", "None", "-", "--"]:
            return np.nan

        s = s.replace(",", "").strip()
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return np.nan

        try:
            return float(m.group(0))
        except Exception:
            return np.nan

    url = f"https://finance.yahoo.com/quote/{sym}/key-statistics?p={sym}"

    html_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    try:
        r = SESSION.get(url, headers=html_headers, timeout=25)
        r.raise_for_status()
        html = r.text or ""

        soup = BeautifulSoup(html, "html.parser")

        # ---- 1) Parse actual HTML tables exactly like the Statistics page ----
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = tr.find_all(["th", "td"])
                vals = [c.get_text(" ", strip=True) for c in cells if c.get_text(" ", strip=True)]

                if len(vals) < 2:
                    continue

                label = vals[0].strip().lower()
                value = vals[-1].strip()

                if pd.isna(out["P/E"]) and "trailing p/e" in label:
                    out["P/E"] = parse_ratio_value(value)

                elif pd.isna(out["P/B"]) and ("price/book" in label or "price to book" in label):
                    out["P/B"] = parse_ratio_value(value)

                elif pd.isna(out["EV/EBITDA"]) and (
                    "enterprise value/ebitda" in label or "ev/ebitda" in label
                ):
                    out["EV/EBITDA"] = parse_ratio_value(value)

        # ---- 2) Fallback: try pandas read_html on same page ----
        if pd.isna(out["P/E"]) or pd.isna(out["P/B"]) or pd.isna(out["EV/EBITDA"]):
            try:
                tables = pd.read_html(StringIO(html))
            except Exception:
                tables = []

            for t in tables:
                if t is None or t.empty:
                    continue

                for _, row in t.iterrows():
                    vals = [
                        str(x).strip()
                        for x in row.tolist()
                        if str(x).strip() not in ["", "nan", "None"]
                    ]
                    if len(vals) < 2:
                        continue

                    label = vals[0].lower()
                    value = vals[-1]

                    if pd.isna(out["P/E"]) and "trailing p/e" in label:
                        out["P/E"] = parse_ratio_value(value)

                    elif pd.isna(out["P/B"]) and ("price/book" in label or "price to book" in label):
                        out["P/B"] = parse_ratio_value(value)

                    elif pd.isna(out["EV/EBITDA"]) and (
                        "enterprise value/ebitda" in label or "ev/ebitda" in label
                    ):
                        out["EV/EBITDA"] = parse_ratio_value(value)

        # ---- 3) Last fallback: regex directly from page source ----
        if pd.isna(out["P/E"]):
            m = re.search(r"Trailing P/E.*?(-?\d+(?:\.\d+)?)", html, re.I | re.S)
            if m:
                out["P/E"] = parse_ratio_value(m.group(1))

        if pd.isna(out["P/B"]):
            m = re.search(r"Price/Book.*?(-?\d+(?:\.\d+)?)", html, re.I | re.S)
            if m:
                out["P/B"] = parse_ratio_value(m.group(1))

        if pd.isna(out["EV/EBITDA"]):
            m = re.search(r"Enterprise Value/EBITDA.*?(-?\d+(?:\.\d+)?)", html, re.I | re.S)
            if m:
                out["EV/EBITDA"] = parse_ratio_value(m.group(1))

        if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
            out["ratio_source"] = "Yahoo Statistics"

    except Exception:
        pass

    return out
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_html_ratio_fallback(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    out = {
        "Ticker": sym,
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": ""
    }

    if not sym:
        return out

    try:
        url = f"https://finance.yahoo.com/quote/{sym}"
        r = _safe_get(url, timeout=20, tries=2)
        html = r.text or ""

        patterns = {
            "P/E": [
                r'"trailingPE"\s*:\s*\{"raw"\s*:\s*([\-0-9.]+)',
                r'"trailingPE"\s*:\s*([\-0-9.]+)',
            ],
            "P/B": [
                r'"priceToBook"\s*:\s*\{"raw"\s*:\s*([\-0-9.]+)',
                r'"priceToBook"\s*:\s*([\-0-9.]+)',
            ],
            "EV/EBITDA": [
                r'"enterpriseToEbitda"\s*:\s*\{"raw"\s*:\s*([\-0-9.]+)',
                r'"enterpriseToEbitda"\s*:\s*([\-0-9.]+)',
            ],
        }

        for field, pats in patterns.items():
            for pat in pats:
                m = re.search(pat, html)
                if m:
                    out[field] = _clean_num(m.group(1))
                    break

        if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
            out["ratio_source"] = "Yahoo HTML fallback"
    except Exception:
        pass

    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yfinance_ratio_snapshot(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    out = {
        "Ticker": sym,
        "Company": "",
        "Exchange": "",
        "Country": "",
        "Sector": "",
        "Industry": "",
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": ""
    }

    if not sym:
        return out

    try:
        tk = yf.Ticker(sym)
        info = tk.info or {}

        out["Company"] = _clean_text(info.get("longName") or info.get("shortName") or "")
        out["Sector"] = _clean_text(info.get("sector") or "")
        out["Industry"] = _clean_text(info.get("industry") or "")
        out["Country"] = _clean_text(info.get("country") or "")
        out["Exchange"] = _clean_text(info.get("exchange") or "")

        pe = _clean_num(_first_non_null(info, ["trailingPE", "forwardPE"]))
        pb = _clean_num(_first_non_null(info, ["priceToBook"]))
        ev = _clean_num(_first_non_null(info, ["enterpriseToEbitda"]))

        if pd.isna(pe) or pd.isna(pb) or pd.isna(ev):
            market_cap = _clean_num(_first_non_null(info, ["marketCap"]))
            total_debt = _clean_num(_first_non_null(info, ["totalDebt"]))
            cash = _clean_num(_first_non_null(info, ["totalCash"]))
            ebitda = _clean_num(_first_non_null(info, ["ebitda"]))
            book_value = _clean_num(_first_non_null(info, ["bookValue"]))
            shares = _clean_num(_first_non_null(info, ["sharesOutstanding"]))
            net_income = _clean_num(_first_non_null(info, ["netIncomeToCommon"]))

            if pd.isna(pb) and not pd.isna(book_value) and not pd.isna(shares) and shares != 0:
                book_equity = book_value * shares
                if not pd.isna(market_cap) and book_equity != 0:
                    pb = market_cap / book_equity

            if pd.isna(pe) and not pd.isna(market_cap) and not pd.isna(net_income) and net_income != 0:
                pe = market_cap / net_income

            if pd.isna(ev) and not pd.isna(market_cap) and not pd.isna(ebitda) and ebitda != 0:
                debt = 0.0 if pd.isna(total_debt) else total_debt
                csh = 0.0 if pd.isna(cash) else cash
                ev = (market_cap + debt - csh) / ebitda

        out["P/E"] = pe
        out["P/B"] = pb
        out["EV/EBITDA"] = ev

        if not (pd.isna(pe) and pd.isna(pb) and pd.isna(ev)):
            out["ratio_source"] = "yfinance"
    except Exception:
        pass

    return out

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yfinance_derived_from_statements(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)

    out = {
        "Ticker": sym,
        "Company": "",
        "Exchange": "",
        "Country": "",
        "Sector": "",
        "Industry": "",
        "P/E": np.nan,
        "P/B": np.nan,
        "EV/EBITDA": np.nan,
        "ratio_source": "",
    }

    if not sym:
        return out

    def _pick_first_series_value(df, possible_rows):
        try:
            if df is None or df.empty:
                return np.nan
            for row_name in possible_rows:
                if row_name in df.index:
                    vals = pd.to_numeric(df.loc[row_name], errors="coerce").dropna()
                    if len(vals) > 0:
                        return float(vals.iloc[0])
        except Exception:
            pass
        return np.nan

    try:
        tk = yf.Ticker(sym)

        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        out["Company"] = _clean_text(info.get("longName") or info.get("shortName") or "")
        out["Exchange"] = _clean_text(info.get("exchange") or "")
        out["Country"] = _clean_text(info.get("country") or "")
        out["Sector"] = _clean_text(info.get("sector") or "")
        out["Industry"] = _clean_text(info.get("industry") or "")

        market_cap = np.nan
        try:
            market_cap = _clean_num(info.get("marketCap"))
        except Exception:
            pass

        if pd.isna(market_cap):
            try:
                fi = tk.fast_info or {}
                market_cap = _clean_num(fi.get("market_cap"))
            except Exception:
                pass

        income_stmt = None
        balance_sheet = None

        try:
            income_stmt = tk.income_stmt
        except Exception:
            income_stmt = None

        if income_stmt is None or getattr(income_stmt, "empty", True):
            try:
                income_stmt = tk.financials
            except Exception:
                income_stmt = None

        try:
            balance_sheet = tk.balance_sheet
        except Exception:
            balance_sheet = None

        net_income = _pick_first_series_value(
            income_stmt,
            [
                "Net Income",
                "NetIncome",
                "Net Income Common Stockholders",
                "Net Income Applicable To Common Shares",
            ],
        )

        ebitda = _pick_first_series_value(
            income_stmt,
            [
                "EBITDA",
                "Ebitda",
            ],
        )

        total_equity = _pick_first_series_value(
            balance_sheet,
            [
                "Stockholders Equity",
                "Total Stockholder Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity",
                "Total Equity",
            ],
        )

        total_debt = _pick_first_series_value(
            balance_sheet,
            [
                "Total Debt",
                "Net Debt",
                "Long Term Debt And Capital Lease Obligation",
                "Long Term Debt",
                "Current Debt",
            ],
        )

        cash = _pick_first_series_value(
            balance_sheet,
            [
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "Cash",
            ],
        )

        # fallbacks from info
        if pd.isna(net_income):
            net_income = _clean_num(info.get("netIncomeToCommon"))

        if pd.isna(ebitda):
            ebitda = _clean_num(info.get("ebitda"))

        if pd.isna(total_debt):
            total_debt = _clean_num(info.get("totalDebt"))

        if pd.isna(cash):
            cash = _clean_num(info.get("totalCash"))

        # derive ratios
        pe = np.nan
        pb = np.nan
        ev_ebitda = np.nan

        if not pd.isna(market_cap) and not pd.isna(net_income) and net_income != 0:
            pe = market_cap / net_income

        if not pd.isna(market_cap) and not pd.isna(total_equity) and total_equity != 0:
            pb = market_cap / total_equity

        if not pd.isna(market_cap) and not pd.isna(ebitda) and ebitda != 0:
            debt = 0.0 if pd.isna(total_debt) else total_debt
            csh = 0.0 if pd.isna(cash) else cash
            enterprise_value = market_cap + debt - csh
            ev_ebitda = enterprise_value / ebitda

        out["P/E"] = _clean_num(pe)
        out["P/B"] = _clean_num(pb)
        out["EV/EBITDA"] = _clean_num(ev_ebitda)

        if not (pd.isna(out["P/E"]) and pd.isna(out["P/B"]) and pd.isna(out["EV/EBITDA"])):
            out["ratio_source"] = "yfinance derived statements"

    except Exception:
        pass

    return out
def get_live_peer_row(symbol: str) -> dict:
    sym = normalize_peer_ticker(symbol)
    if not sym:
        return {}

    known = KNOWN_PEER_NAME_MAP.get(sym, {})

    # 1) Yahoo
    yq = yahoo_quote_profile(sym)
    yh = yahoo_profile_and_metrics(sym)
    yhtml = yahoo_html_ratio_fallback(sym)
    ystats = yahoo_stats_table_fallback(sym)

    # 2) yfinance
    yfin = yfinance_ratio_snapshot(sym)
    yfin_derived = yfinance_derived_from_statements(sym)

    # 3) Finnhub
    fh = finnhub_basic_metrics(sym)

    # 4) FMP direct
    fp = fmp_profile(sym)
    fr = fmp_ratios(sym)

    # 5) FMP derived
    rawf = fmp_raw_fundamentals(sym)
    derived = derive_ratios_from_raw(rawf)

    local_name = sym
    local_exchange = ""
    local_sector = ""
    local_industry = ""
    local_country = ""

    for row in ZIM_COUNTERS:
        if str(row.get("symbol", "")).strip().upper() == sym:
            local_name = str(row.get("company", "")).strip() or sym
            local_exchange = str(row.get("exchange", "")).strip()
            local_sector = str(row.get("sector", "")).strip()
            local_industry = local_sector
            local_country = "Zimbabwe"
            break

    company_name = (
        yh.get("Company")
        or yq.get("Company")
        or yfin.get("Company")
        or fp.get("Company")
        or known.get("company", "")
        or local_name
    )

    exchange_name = (
        yh.get("Exchange")
        or yq.get("Exchange")
        or yfin.get("Exchange")
        or fp.get("Exchange")
        or known.get("exchange", "")
        or local_exchange
    )

    country_name = (
        yh.get("Country")
        or yq.get("Country")
        or yfin.get("Country")
        or fp.get("Country")
        or known.get("country", "")
        or local_country
    )

    sector_name = (
        yh.get("Sector")
        or yfin.get("Sector")
        or fp.get("Sector")
        or local_sector
    )

    industry_name = (
        yh.get("Industry")
        or yfin.get("Industry")
        or fp.get("Industry")
        or local_industry
        or sector_name
    )
    description_text = (
            yh.get("Description")
            or ""
    )
    pe = ystats.get("P/E", np.nan)
    if pd.isna(pe):
        pe = yh.get("P/E", np.nan)
    if pd.isna(pe):
        pe = yhtml.get("P/E", np.nan)
    if pd.isna(pe):
        pe = yfin.get("P/E", np.nan)
    if pd.isna(pe):
        pe = yfin_derived.get("P/E", np.nan)
    if pd.isna(pe):
        pe = fh.get("P/E", np.nan)
    if pd.isna(pe):
        pe = fr.get("P/E", np.nan)
    if pd.isna(pe):
        pe = derived.get("P/E", np.nan)

    pb = ystats.get("P/B", np.nan)
    if pd.isna(pb):
        pb = yh.get("P/B", np.nan)
    if pd.isna(pb):
        pb = yhtml.get("P/B", np.nan)
    if pd.isna(pb):
        pb = yfin.get("P/B", np.nan)
    if pd.isna(pb):
        pb = yfin_derived.get("P/B", np.nan)
    if pd.isna(pb):
        pb = fh.get("P/B", np.nan)
    if pd.isna(pb):
        pb = fr.get("P/B", np.nan)
    if pd.isna(pb):
        pb = derived.get("P/B", np.nan)

    ev_ebitda = ystats.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = yh.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = yhtml.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = yfin.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = yfin_derived.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = fh.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = fr.get("EV/EBITDA", np.nan)
    if pd.isna(ev_ebitda):
        ev_ebitda = derived.get("EV/EBITDA", np.nan)
    source = (
            ystats.get("ratio_source")
            or yh.get("ratio_source")
            or yhtml.get("ratio_source")
            or yfin.get("ratio_source")
            or yfin_derived.get("ratio_source")
            or fh.get("ratio_source")
            or fr.get("ratio_source")
            or derived.get("ratio_source")
            or ""
    )
    return {
        "Company": company_name,
        "Ticker": sym,
        "Exchange": exchange_name,
        "Country": country_name,
        "Sector": sector_name,
        "Industry": industry_name,
        "Description": description_text,
        "EV/EBITDA": _clean_num(ev_ebitda),
        "P/B": _clean_num(pb),
        "P/E": _clean_num(pe),
        "Source": source,
        "YahooProfile": make_yahoo_profile_url(sym),
    }

# =========================================================
# Symbol resolution
# =========================================================
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def fmp_search_symbol(query: str) -> dict:
    q = (query or "").strip()
    if not q or not FMP_API_KEY:
        return {}

    try:
        data = _safe_get_json(
            url=f"{FMP_BASE}/search",
            params={"query": q, "limit": 10, "apikey": FMP_API_KEY},
            timeout=25,
            tries=2,
        )

        if isinstance(data, list) and data:
            row = data[0]
            return {
                "symbol": normalize_peer_ticker(row.get("symbol", "")),
                "company": str(row.get("name", "")).strip(),
                "exchange": str(row.get("exchangeShortName", "")).strip(),
                "source": "fmp_search",
            }
    except Exception:
        pass

    return {}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def finnhub_search_symbol(query: str) -> dict:
    q = (query or "").strip()
    if not q or not FINNHUB_API_KEY:
        return {}

    try:
        data = _safe_get_json(
            url=f"{FINNHUB_BASE}/search",
            params={"q": q, "token": FINNHUB_API_KEY},
            timeout=25,
            tries=2,
        )
        results = data.get("result", []) if isinstance(data, dict) else []
        if results:
            row = results[0]
            return {
                "symbol": normalize_peer_ticker(row.get("symbol", "")),
                "company": str(row.get("description", "")).strip(),
                "exchange": "",
                "source": "finnhub_search",
            }
    except Exception:
        pass

    return {}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def resolve_symbol(query: str) -> dict:
    q = (query or "").strip()
    if not q:
        return {}

    local_hit = find_local_counter(q)
    if local_hit:
        return local_hit

    if _looks_like_symbol(q):
        return {
            "symbol": normalize_peer_ticker(q),
            "company": q.strip().upper(),
            "exchange": "",
            "source": "manual",
        }

    try:
        df = yahoo_search(q, quotes_count=15)
        if df is not None and not df.empty:
            df = df.copy()
            df["Type"] = df["Type"].fillna("").astype(str)
            df["Exchange"] = df["Exchange"].fillna("").astype(str)
            df["Ticker"] = df["Ticker"].fillna("").astype(str)
            df["Company"] = df["Company"].fillna("").astype(str)

            africa_df = filter_africa(df)
            if africa_df is not None and not africa_df.empty:
                row = africa_df.iloc[0]
                sym = normalize_peer_ticker(row.get("Ticker", ""))
                if sym:
                    return {
                        "symbol": sym,
                        "company": str(row.get("Company", "")).strip(),
                        "exchange": str(row.get("Exchange", "")).strip(),
                        "source": "yahoo_africa",
                    }

            eq = df[df["Type"].str.upper().isin(["EQUITY", "COMMON STOCK", ""])].copy()
            row = eq.iloc[0] if not eq.empty else df.iloc[0]
            sym = normalize_peer_ticker(row.get("Ticker", ""))
            if sym:
                return {
                    "symbol": sym,
                    "company": str(row.get("Company", "")).strip(),
                    "exchange": str(row.get("Exchange", "")).strip(),
                    "source": "yahoo_search",
                }
    except Exception:
        pass

    hit = fmp_search_symbol(q)
    if hit:
        return hit

    hit = finnhub_search_symbol(q)
    if hit:
        return hit

    return {}
# =========================================================
# Dynamic Yahoo sector / industry peer discovery
# =========================================================
def peer_similarity_score(target_sector: str, target_industry: str, peer_row: dict) -> int:
    score = 0

    ts = _clean_text(target_sector).lower()
    ti = _clean_text(target_industry).lower()
    ps = _clean_text(peer_row.get("Sector", "")).lower()
    pi = _clean_text(peer_row.get("Industry", "")).lower()
    pdsc = _clean_text(peer_row.get("Description", "")).lower()

    # sector match
    if ts and ps:
        if ts == ps:
            score += 10
        elif ts in ps or ps in ts:
            score += 6

    # industry match
    if ti and pi:
        if ti == pi:
            score += 12
        elif ti in pi or pi in ti:
            score += 7

    # description support
    if ts and ts in pdsc:
        score += 4
    if ti and ti in pdsc:
        score += 4

    # telecom-specific support
    if ts in ["telecommunications", "telecommunication", "telecom"]:
        telecom_words = ["telecom", "telecommunication", "communications", "wireless", "mobile", "cellular", "network", "broadband", "fiber"]
        for w in telecom_words:
            if w in pdsc or w in pi or w in ps:
                score += 2

    return score
def text_keyword_score(target_company: str, target_sector: str, peer_profile: dict) -> int:
    score = 0

    company_text = _clean_text(target_company).lower()
    sector_text = _clean_text(target_sector).lower()

    peer_company = _clean_text(peer_profile.get("Company")).lower()
    peer_sector = _clean_text(peer_profile.get("Sector")).lower()
    peer_industry = _clean_text(peer_profile.get("Industry")).lower()
    peer_desc = _clean_text(peer_profile.get("Description")).lower()

    # sector keyword match
    if sector_text:
        if sector_text in peer_sector:
            score += 5
        if sector_text in peer_industry:
            score += 4
        if sector_text in peer_desc:
            score += 4

    # company keyword tokens
    company_tokens = [w for w in re.split(r"[^a-z0-9]+", company_text) if len(w) >= 4]
    for tok in company_tokens:
        if tok in peer_company:
            score += 2
        if tok in peer_desc:
            score += 1

    # telecom-specific synonyms example
    synonyms = {
        "telecommunications": ["telecommunications", "telecom", "mobile network", "wireless", "cellular"],
        "banking": ["bank", "banking", "financial services", "lending"],
        "insurance": ["insurance", "assurance", "life cover"],
        "mining": ["mining", "minerals", "gold", "platinum", "metals"],
    }

    for base_word, words in synonyms.items():
        if sector_text and base_word in sector_text:
            for w in words:
                if w in peer_desc or w in peer_industry or w in peer_sector:
                    score += 2

    return score
def profile_description_match_score(target_company: str, target_sector: str, target_industry: str, peer_prof: dict) -> int:
    score = 0

    tc = _clean_text(target_company).lower()
    ts = _clean_text(target_sector).lower()
    ti = _clean_text(target_industry).lower()

    ps = _clean_text(peer_prof.get("Sector")).lower()
    pi = _clean_text(peer_prof.get("Industry")).lower()
    desc = _clean_text(peer_prof.get("Description")).lower()
    name = _clean_text(peer_prof.get("Company")).lower()

    # sector / industry exact or partial matches
    if ts:
        if ts == ps:
            score += 6
        elif ts in ps or ps in ts:
            score += 4
        elif ts in desc:
            score += 3

    if ti:
        if ti == pi:
            score += 7
        elif ti in pi or pi in ti:
            score += 5
        elif ti in desc:
            score += 3

    # company-related wording
    if tc:
        tc_words = [w for w in re.split(r"[^a-z0-9]+", tc) if len(w) >= 4]
        for w in tc_words:
            if w in name:
                score += 2
            if w in desc:
                score += 1

    # telecom synonyms example
    telecom_words = ["telecom", "telecommunication", "wireless", "mobile", "cellular", "network", "broadband", "fiber", "data services"]
    if ts in ["telecommunications", "telecommunication", "telecom"]:
        for w in telecom_words:
            if w in desc or w in pi or w in ps:
                score += 1

    return score
def strict_sector_gate(target_sector: str, target_industry: str, peer_prof: dict) -> bool:
    ts = _clean_text(target_sector).lower()
    ti = _clean_text(target_industry).lower()

    ps = _clean_text(peer_prof.get("Sector")).lower()
    pi = _clean_text(peer_prof.get("Industry")).lower()
    desc = _clean_text(peer_prof.get("Description")).lower()
    name = _clean_text(peer_prof.get("Company")).lower()

    combined = " | ".join([ps, pi, desc, name])

    telecom_words = [
        "telecom", "telecommunication", "communications",
        "wireless", "mobile", "cellular", "broadband",
        "network", "operator", "data services", "fiber"
    ]

    banking_words = [
        "bank", "banking", "financial services", "lending",
        "commercial bank", "retail bank"
    ]

    insurance_words = [
        "insurance", "assurance", "life insurance", "general insurance"
    ]

    mining_words = [
        "mining", "minerals", "gold", "platinum", "metals", "resources"
    ]

    # telecommunications strict gate
    if ts in ["telecommunications", "telecommunication", "telecom"]:
        telecom_hits = sum(1 for w in telecom_words if w in combined)
        return telecom_hits >= 2 or "mtn" in name or "vodacom" in name or "safaricom" in name

    # banking strict gate
    if ts == "banking":
        return any(w in combined for w in banking_words)

    # insurance strict gate
    if ts == "insurance":
        return any(w in combined for w in insurance_words)

    # mining strict gate
    if ts in ["mining", "gold mining", "mining exploration"]:
        return any(w in combined for w in mining_words)

    # generic gate for other sectors
    if ts:
        if ts in combined:
            return True

    if ti:
        if ti in combined:
            return True

    return False
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_find_sector_peers(
    target_symbol: str,
    target_company: str = "",
    max_peers: int = 8,
    manual_sector: str = ""
) -> list:
    sym = normalize_peer_ticker(target_symbol)
    if not sym:
        return []

    prof = yahoo_profile_and_metrics(sym)

    target_sector = _clean_text(prof.get("Sector"))
    target_industry = _clean_text(prof.get("Industry"))

    if manual_sector and manual_sector.strip():
        target_sector = manual_sector.strip()
        target_industry = ""

    if not target_sector and not target_industry:
        fp = fmp_profile(sym)
        target_sector = _clean_text(fp.get("Sector"))
        target_industry = _clean_text(fp.get("Industry"))

    if not target_sector:
        target_sector = _clean_text(get_local_counter_sector(sym))

    if not target_sector and not target_industry:
        return []

    # -------------------------------------------------
    # STEP 1: build candidate universe with source tags
    # -------------------------------------------------
    candidate_rows = []
    seen = set()

    # A. FIRST: Yahoo profile-driven discovery
    discovered_from_profile = yahoo_discover_sector_candidates_from_profile(
        target_symbol=sym,
        target_company=target_company,
        manual_sector=manual_sector,
        max_results_per_query=20,
    )
    S["debug_profile_discovered"] = discovered_from_profile
    for t in discovered_from_profile:
        t = normalize_peer_ticker(t)
        if t and t != sym and t not in seen:
            seen.add(t)
            candidate_rows.append({"Ticker": t, "SeedSource": "yahoo_profile_discovered"})

    # B. SECOND: company-specific peers only as fallback/top-up
    if len(candidate_rows) < max_peers:
        for t in get_company_specific_proxy_peers(sym, target_company, max_peers=20):
            t = normalize_peer_ticker(t)
            if t and t != sym and t not in seen:
                seen.add(t)
                candidate_rows.append({"Ticker": t, "SeedSource": "company_specific"})

    # C. THIRD: sector proxy only as fallback/top-up
    if len(candidate_rows) < max_peers:
        for t in get_africa_sector_proxy_peers(target_sector, max_peers=20):
            t = normalize_peer_ticker(t)
            if t and t != sym and t not in seen:
                seen.add(t)
                candidate_rows.append({"Ticker": t, "SeedSource": "sector_proxy"})

    # D. LAST: broader known universe
    if len(candidate_rows) < max_peers:
        for t in KNOWN_PEER_NAME_MAP.keys():
            t = normalize_peer_ticker(t)
            if not t or t == sym or t in seen:
                continue
            if any(t.endswith(sfx) for sfx in AFRICA_SUFFIXES):
                seen.add(t)
                candidate_rows.append({"Ticker": t, "SeedSource": "broad_universe"})
    S["debug_candidate_tickers"] = candidate_rows
    if not candidate_rows:
        return []

    S["debug_candidate_tickers"] = candidate_rows

    # -------------------------------------------------
    # STEP 2: inspect profiles and score them
    # -------------------------------------------------
    scored = []

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(candidate_rows)))) as ex:
        futs = {
            ex.submit(yahoo_profile_and_metrics, row["Ticker"]): row
            for row in candidate_rows[:120]
        }

        for fut in as_completed(futs):
            row = futs[fut]
            tkr = row["Ticker"]
            seed_source = row["SeedSource"]

            try:
                peer_prof = fut.result() or {}
            except Exception:
                peer_prof = {}

            known = KNOWN_PEER_NAME_MAP.get(tkr, {})

            merged_peer = {
                "Ticker": tkr,
                "Company": _clean_text(peer_prof.get("Company")) or _clean_text(known.get("company")),
                "Exchange": _clean_text(peer_prof.get("Exchange")) or _clean_text(known.get("exchange")),
                "Country": _clean_text(peer_prof.get("Country")) or _clean_text(known.get("country")),
                "Sector": _clean_text(peer_prof.get("Sector")),
                "Industry": _clean_text(peer_prof.get("Industry")),
                "Description": _clean_text(peer_prof.get("Description")),
            }

            # -------------------------------------------------
            # Only strict-gate the BROAD universe
            # -------------------------------------------------
            if seed_source in {"broad_universe", "yahoo_profile_discovered"}:
                if not strict_sector_gate(target_sector, target_industry, merged_peer):
                    continue
            score = (
                peer_similarity_score(
                    target_sector=target_sector,
                    target_industry=target_industry,
                    peer_row=merged_peer
                )
                + profile_description_match_score(
                    target_company=target_company,
                    target_sector=target_sector,
                    target_industry=target_industry,
                    peer_prof=merged_peer
                )
            )
            # small bonus by seed source
            if seed_source == "yahoo_profile_discovered":
                score += 7
            elif seed_source == "company_specific":
                score += 4
            elif seed_source == "sector_proxy":
                score += 3
            # lower threshold for curated peers, higher for broad universe
            # thresholds by source
            if seed_source == "yahoo_profile_discovered":
                if score >= 5:
                    scored.append((tkr, score))
            elif seed_source in {"company_specific", "sector_proxy"}:
                if score >= 4:
                    scored.append((tkr, score))
            else:
                if score >= 8:
                    scored.append((tkr, score))

    if not scored:
        return []

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    out = []
    used = set()
    for tkr, _ in scored:
        if tkr not in used:
            used.add(tkr)
            out.append(tkr)
        if len(out) >= max(max_peers * 2, 12):
            break

    return out[:max(max_peers * 2, 12)]
def build_live_comps_from_target(target_query: str, max_peers: int = 8):
    target = resolve_symbol(target_query)
    if not target:
        return pd.DataFrame(), {"error": "Could not resolve the target company/ticker."}

    target_symbol = str(target.get("symbol", "")).strip().upper()
    target_company_name = str(target.get("company", "")).strip()

    if not target_symbol:
        return pd.DataFrame(), {"error": "Resolved target did not return a valid ticker."}

    peers = []
    peer_source = ""

    # -----------------------------------------------------
    # 1) Dynamic Yahoo sector / industry search FIRST
    # -----------------------------------------------------
    yahoo_peers = yahoo_find_sector_peers(
        target_symbol=target_symbol,
        target_company=target_company_name,
        max_peers=max_peers * 2,
        manual_sector=S.get("manual_sector_override", "")
    )

    S["debug_yahoo_peers"] = yahoo_peers

    peers = []
    seen = set()

    for p in yahoo_peers:
        p = normalize_peer_ticker(p)
        if not p or p == normalize_peer_ticker(target_symbol):
            continue
        if p in seen:
            continue
        seen.add(p)
        peers.append(p)

    peers = peers[:max(max_peers * 2, 12)]

    if peers:
        peer_source = "Yahoo/sector-ranked peers"
    # -----------------------------------------------------
    # 3) FMP peers
    # -----------------------------------------------------
    if not peers:
        peers = fmp_get_peers(target_symbol, limit=max_peers)
        if peers:
            peer_source = "FMP peers"

    # -----------------------------------------------------
    # 4) Finnhub peers
    # -----------------------------------------------------
    if not peers:
        peers = finnhub_get_peers(target_symbol, limit=max_peers)
        if peers:
            peer_source = "Finnhub peers"

    # -----------------------------------------------------
    # 5) Static sector proxy fallback last
    # -----------------------------------------------------
    if not peers:
        local_sector = get_local_counter_sector(target_symbol)
        peers = get_africa_sector_proxy_peers(local_sector, max_peers=max_peers)
        if peers:
            peer_source = f"Sector proxy peers ({local_sector})"

    peers = [normalize_peer_ticker(x) for x in peers if str(x).strip()]
    peers = [x for x in peers if x != normalize_peer_ticker(target_symbol)]

    deduped = []
    seen = set()
    for p in peers:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    peers = deduped[:max(max_peers * 2, 12)]

    if not peers:
        return pd.DataFrame(), {
            "error": f"No peers found for {target_symbol}.",
            "target": target,
            "peer_source": peer_source,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(peers)))) as ex:
        futs = {ex.submit(get_live_peer_row, sym): sym for sym in peers}
        for fut in as_completed(futs):
            try:
                row = fut.result()
                if row:
                    rows.append(row)
            except Exception:
                pass

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(), {
            "error": f"Peers were found for {target_symbol}, but ratio download failed.",
            "target": target,
            "peer_source": peer_source,
        }

    df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    # -----------------------------------------------------
    # Rank peers by sector / industry closeness
    # -----------------------------------------------------
    target_prof = yahoo_profile_and_metrics(target_symbol)
    target_sector = _clean_text(target_prof.get("Sector"))
    target_industry = _clean_text(target_prof.get("Industry"))

    if not target_sector and not target_industry:
        fp = fmp_profile(target_symbol)
        target_sector = _clean_text(fp.get("Sector"))
        target_industry = _clean_text(fp.get("Industry"))

    manual_sector_used = _clean_text(S.get("manual_sector_override", ""))
    if manual_sector_used:
        target_sector = manual_sector_used
        target_industry = ""

    df["SimilarityScore"] = df.apply(
        lambda r: (
            peer_similarity_score(
                target_sector=target_sector,
                target_industry=target_industry,
                peer_row=r.to_dict()
            )
            + profile_description_match_score(
                target_company=target_company_name,
                target_sector=target_sector,
                target_industry=target_industry,
                peer_prof=r.to_dict()
            )
        ),
        axis=1
    )

    # prefer rows with actual ratios as well
    df["RatioCount"] = (
        df["EV/EBITDA"].notna().astype(int)
        + df["P/B"].notna().astype(int)
        + df["P/E"].notna().astype(int)
    )

    df = df.sort_values(
        by=["SimilarityScore", "RatioCount"],
        ascending=[False, False]
    ).head(max_peers).reset_index(drop=True)

    meta = {
        "target": target,
        "peer_source": peer_source,
        "peer_count": len(df),
        "target_sector": target_sector,
        "target_industry": target_industry,
    }

    if not df.empty:
        all_ratio_empty = (df["EV/EBITDA"].isna() & df["P/B"].isna() & df["P/E"].isna()).all()
        if all_ratio_empty:
            meta["warning"] = (
                f"Peers were found for {target_symbol}, but live ratios were empty. "
                f"You can still use the peers and fill ratios manually."
            )

    return df, meta


def apply_live_comps_to_session(df_live: pd.DataFrame):
    if df_live is None or df_live.empty:
        return

    S.setdefault("comps", {})

    n = len(df_live)
    S["num_comps"] = n


    for i, (_, r) in enumerate(df_live.iterrows()):
        S[f"comp_name_{i}"] = str(r.get("Company", "")).strip() or str(r.get("Ticker", "")).strip()
        S[f"comp_ticker_{i}"] = str(r.get("Ticker", "")).strip()
        S[f"comp_source_{i}"] = str(r.get("Source", "")).strip()
        S[f"comp_profile_{i}"] = str(r.get("YahooProfile", "")).strip()

        S[f"comp_ev_{i}"] = np.nan if pd.isna(r.get("EV/EBITDA", np.nan)) else float(r["EV/EBITDA"])
        S[f"comp_pb_{i}"] = np.nan if pd.isna(r.get("P/B", np.nan)) else float(r["P/B"])
        S[f"comp_pe_{i}"] = np.nan if pd.isna(r.get("P/E", np.nan)) else float(r["P/E"])

        S[f"inc_ev_{i}"] = not pd.isna(r.get("EV/EBITDA", np.nan))
        S[f"inc_pb_{i}"] = not pd.isna(r.get("P/B", np.nan))
        S[f"inc_pe_{i}"] = not pd.isna(r.get("P/E", np.nan))

        S["comps"].setdefault(i, {})
        S["comps"][i]["name"] = S[f"comp_name_{i}"]
        S["comps"][i]["ticker"] = S[f"comp_ticker_{i}"]
        S["comps"][i]["source"] = S[f"comp_source_{i}"]
        S["comps"][i]["profile"] = S[f"comp_profile_{i}"]
        S["comps"][i]["ev"] = S[f"comp_ev_{i}"]
        S["comps"][i]["pb"] = S[f"comp_pb_{i}"]
        S["comps"][i]["pe"] = S[f"comp_pe_{i}"]
        S["comps"][i]["inc_ev"] = S[f"inc_ev_{i}"]
        S["comps"][i]["inc_pb"] = S[f"inc_pb_{i}"]
        S["comps"][i]["inc_pe"] = S[f"inc_pe_{i}"]

# =========================================================
# STEP 1 — INPUT COMPARABLE COMPANIES & MULTIPLES
# =========================================================
st.header("Step 1 — Input Comparable Companies & Multiples")
st.subheader("Auto Peer Suggestions from Africa Sector Mapping")

S.setdefault("target_company", "")
S.setdefault("auto_peer_count", 8)

cA, cB, cC = st.columns([2.2, 1, 1.2])
with cA:
    target_company = st.text_input(
        "Company you are valuing (ANY market: Zimbabwe/JSE/etc)",
        value=S["target_company"],
        key="target_company_input",
        placeholder="e.g., Innscor, Delta, FBC, Econet, MTN, Vodacom, Safaricom ...",
    )
with cB:
    peer_count = st.number_input(
        "Peers to suggest (JSE list / shortlist size)",
        min_value=3,
        max_value=15,
        value=int(S["auto_peer_count"]),
        step=1,
        key="auto_peer_count_input",
    )
with cC:
    st.caption(" ")
    auto_apply = st.checkbox("Auto-fill Step 1 names", value=True, key="auto_apply_peers")

S["target_company"] = target_company
S["auto_peer_count"] = int(peer_count)
S.setdefault("manual_sector_override", "")

manual_sector = st.text_input(
    "Optional manual sector override",
    value=S.get("manual_sector_override", ""),
    key="manual_sector_override_input",
    placeholder="e.g. Telecommunications, Banking, Beverages"
)
S["manual_sector_override"] = manual_sector

# ================= LIVE PEER SEARCH HERE =================

st.subheader("Live peer search and ratio fill")
st.caption("Zimbabwe counters use mapped sector to fetch Africa peers first, then FMP/Finnhub as fallback.")
live_peer_limit = st.slider(
    "Live peers to import",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
    key="live_peer_limit"
)

run_live_comps = st.button("⚡ Auto-search live peers and ratios")

S.setdefault("live_comps_df", pd.DataFrame())
S.setdefault("live_comps_meta", {})

if run_live_comps:
    if not target_company.strip():
        st.warning("Enter the company name or ticker first.")
    else:
        with st.spinner("Searching live peers and ratios..."):
            live_df, meta = build_live_comps_from_target(
                target_query=target_company,
                max_peers=int(live_peer_limit)
            )
            S["live_comps_df"] = live_df
            S["live_comps_meta"] = meta

            if live_df is not None and not live_df.empty:
                apply_live_comps_to_session(live_df)
                st.success(f"Loaded {len(live_df)} live peers into Step 1.")
                if meta.get("warning"):
                    st.warning(meta["warning"])
            else:
                st.error(meta.get("error", "Live peer search failed."))

live_df = S.get("live_comps_df", pd.DataFrame())
live_meta = S.get("live_comps_meta", {})
debug_yahoo_peers = S.get("debug_yahoo_peers", [])
if debug_yahoo_peers:
    st.caption(f"DEBUG yahoo_peers: {debug_yahoo_peers}")

debug_profile = S.get("debug_profile_discovered", [])
if debug_profile:
    st.caption(f"DEBUG profile_discovered: {debug_profile}")
debug_candidates = S.get("debug_candidate_tickers", [])
if debug_candidates:
    st.caption(f"DEBUG candidate_rows: {debug_candidates}")
if live_meta:
    tgt = live_meta.get("target", {})
    if tgt:
        extra_sector = S.get("manual_sector_override", "").strip()
        extra_txt = f" | Manual sector override: {extra_sector}" if extra_sector else ""

        st.caption(
            f"Resolved target: {tgt.get('company', '')} "
            f"({tgt.get('symbol', '')}) via {tgt.get('source', '')} | "
            f"Peer source: {live_meta.get('peer_source', '')}"
            f"{extra_txt}"
        )
if live_df is not None and not live_df.empty:
    df_show = live_df.copy()

    ratio_cols = ["EV/EBITDA", "P/B", "P/E"]
    for c in ratio_cols:
        df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

    if "YahooProfile" not in df_show.columns:
        df_show["YahooProfile"] = df_show["Ticker"].apply(make_yahoo_profile_url)

    display_cols = [
        "YahooProfile",
        "Company",
        "Ticker",
        "Exchange",
        "Country",
        "Sector",
        "Industry",
        "Description",
        "EV/EBITDA",
        "P/B",
        "P/E",
        "Source",
    ]

    if "SimilarityScore" in df_show.columns:
        display_cols.insert(7, "SimilarityScore")

    st.dataframe(
        df_show[display_cols],
        width="stretch",
        column_config={
            "YahooProfile": st.column_config.LinkColumn(
                "Yahoo Profile",
                help="Open peer profile on Yahoo Finance",
                display_text="Open"
            ),
            "EV/EBITDA": st.column_config.NumberColumn("EV/EBITDA", format="%.2f"),
            "P/B": st.column_config.NumberColumn("P/B", format="%.2f"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.2f"),
            "SimilarityScore": st.column_config.NumberColumn("SimilarityScore", format="%d"),
        }
    )

    missing_all = df_show[ratio_cols].isna().all(axis=1)
    if missing_all.any():
        st.warning(
            "Some peers were found, but a few ratio fields are still unavailable from the live providers."
        )

# ================= END LIVE PEER SEARCH =================



# ---- your existing Step 1 comparables inputs (UNCHANGED below)
S.setdefault("num_comps", 3)
S.setdefault("comps", {})

num_comps = st.number_input(
    "How many comparables?",
    min_value=1,
    max_value=15,
    value=int(S.get("num_comps", 3)),
    key="num_comps_input",
)
S["num_comps"] = int(num_comps)

for i in range(int(num_comps)):
    S["comps"].setdefault(i, {
        "name": f"Comp {i + 1}",
        "ev": np.nan, "pb": np.nan, "pe": np.nan,
        "inc_ev": True, "inc_pb": True, "inc_pe": True
    })

rows_comps = []
for i in range(int(num_comps)):
    st.subheader(f"Comparable {i + 1}")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1.3])

    with c1:
        default_name = S.get(f"comp_name_{i}", S["comps"][i]["name"])
        name_val = st.text_input(
            f"Company {i + 1} name",
            value=str(default_name),
            key=f"comp_name_{i}",
        )
        S["comps"][i]["name"] = name_val

    with c2:
        default_ev = _num_input_default(S.get(f"comp_ev_{i}", S["comps"][i]["ev"]), 0.0)
        ev_val = st.number_input(
            f"{name_val} EV/EBITDA",
            value=default_ev,
            step=0.01,
            format="%.2f",
            key=f"comp_ev_{i}",
        )
        S["comps"][i]["ev"] = ev_val

    with c3:
        default_pb = _num_input_default(S.get(f"comp_pb_{i}", S["comps"][i]["pb"]), 0.0)
        pb_val = st.number_input(
            f"{name_val} P/B",
            value=default_pb,
            step=0.01,
            format="%.2f",
            key=f"comp_pb_{i}",
        )
        S["comps"][i]["pb"] = pb_val

    with c4:
        default_pe = _num_input_default(S.get(f"comp_pe_{i}", S["comps"][i]["pe"]), 0.0)
        pe_val = st.number_input(
            f"{name_val} P/E",
            value=default_pe,
            step=0.01,
            format="%.2f",
            key=f"comp_pe_{i}",
        )
        S["comps"][i]["pe"] = pe_val
    ticker_val = S.get(f"comp_ticker_{i}", "")
    source_val = S.get(f"comp_source_{i}", "")

    if ticker_val or source_val:
        st.caption(f"Ticker: {ticker_val} | Ratio source: {source_val}")
        profile_val = S.get(f"comp_profile_{i}", "")
        if profile_val:
            st.markdown(f"[Open Yahoo profile for {name_val}]({profile_val})")
    ev_key = f"inc_ev_{i}"
    pb_key = f"inc_pb_{i}"
    pe_key = f"inc_pe_{i}"

    S["comps"][i].setdefault("inc_ev", True)
    S["comps"][i].setdefault("inc_pb", True)
    S["comps"][i].setdefault("inc_pe", True)

    if ev_key not in S: S[ev_key] = bool(S["comps"][i]["inc_ev"])
    if pb_key not in S: S[pb_key] = bool(S["comps"][i]["inc_pb"])
    if pe_key not in S: S[pe_key] = bool(S["comps"][i]["inc_pe"])

    with c5:
        st.caption("Analyst filter")
        st.checkbox("Include EV/EBITDA", key=ev_key)
        st.checkbox("Include P/B", key=pb_key)
        st.checkbox("Include P/E", key=pe_key)

    inc_ev = bool(S[ev_key])
    inc_pb = bool(S[pb_key])
    inc_pe = bool(S[pe_key])

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

S["comps_num"] = int(num_comps)
S["comps_ev_list"] = df_comps["EV/EBITDA"].astype(float).tolist()
S["comps_pb_list"] = df_comps["P/B"].astype(float).tolist()
S["comps_pe_list"] = df_comps["P/E"].astype(float).tolist()

S["comps_inc_ev"] = df_comps["Include_EV"].astype(bool).tolist()
S["comps_inc_pb"] = df_comps["Include_PB"].astype(bool).tolist()
S["comps_inc_pe"] = df_comps["Include_PE"].astype(bool).tolist()

# =========================================================
# STEP 2 — AVERAGE & IMPLIED MULTIPLES
# =========================================================
st.header("Step 2 — Average & Implied Multiples")

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
