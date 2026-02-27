import streamlit as st
# =========================================================
# Streamlit Page
# =========================================================
st.set_page_config(page_title="Comparables Valuation (Excel Style)", layout="wide")
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import base64
import re
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
# =========================================================
# WIKIDATA + JSE(WIKIPEDIA) CONFIG + HELPERS (PUT ABOVE STEP 1)
# ✅ UPDATED: Global peers now use Yahoo Finance (NOT Wikidata)
# =========================================================
S = st.session_state

# Use ONE headers dict (avoid overriding twice)
HEADERS = {"User-Agent": "Mozilla/5.0"}

def _safe_get(url, params=None, timeout=25):
    r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return r

# =========================================================
# JSE (Wikipedia) peers - stays the same
# =========================================================
JSE_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_companies_traded_on_the_JSE"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_jse_wikipedia_catalog() -> pd.DataFrame:
    """
    Returns a single cleaned DataFrame with columns:
      Symbol | Company | Notes | Link
    (We IGNORE table headings like A/B/C/W because those are not sectors.)
    """
    try:
        html = _safe_get(JSE_WIKI_URL, timeout=30).text
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Company", "Notes", "Link"])

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        return pd.DataFrame(columns=["Symbol", "Company", "Notes", "Link"])

    rows = []
    for tbl in tables:
        try:
            df_list = pd.read_html(str(tbl))
        except Exception:
            continue
        if not df_list:
            continue

        t = df_list[0].copy()
        t.columns = [str(c).strip() for c in t.columns]
        lower_map = {str(c).strip().lower(): str(c).strip() for c in t.columns}

        sym_col = lower_map.get("symbol") or lower_map.get("ticker") or lower_map.get("code")
        comp_col = lower_map.get("company") or lower_map.get("name")
        notes_col = lower_map.get("notes") or lower_map.get("sector") or lower_map.get("industry")

        if not sym_col or not comp_col:
            continue

        # capture wikipedia company link per row where possible
        link_map = {}
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if len(tds) < 2:
                continue
            sym_txt = tds[0].get_text(" ", strip=True)
            comp_td = tds[1]
            a = comp_td.find("a", href=True)
            if sym_txt and a and a["href"].startswith("/wiki/"):
                link_map[sym_txt.strip()] = "https://en.wikipedia.org" + a["href"]

        for _, r in t.iterrows():
            sym = str(r.get(sym_col, "")).strip()
            comp = str(r.get(comp_col, "")).strip()
            notes = str(r.get(notes_col, "")).strip() if notes_col else ""
            link = link_map.get(sym, "")

            if not sym or not comp:
                continue
            if sym.lower() == "nan" or comp.lower() == "nan":
                continue

            rows.append({
                "Symbol": sym,
                "Company": comp,
                "Notes": "" if str(notes).lower() == "nan" else notes,
                "Link": link,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Symbol", "Company", "Notes", "Link"])

    for c in ["Symbol", "Company", "Notes"]:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df = df.drop_duplicates(subset=["Symbol", "Company"], keep="first").reset_index(drop=True)
    return df

def extract_note_tags(notes: str) -> list[str]:
    """
    Turn Notes into tags, e.g.
    "breweries, beverages, soft drinks" -> ["breweries","beverages","soft drinks"]
    """
    n = (notes or "").strip().lower()
    if not n or n == "nan":
        return []

    parts = re.split(r"[;,/|]| and |\(|\)|—|-", n)
    tags = []
    for p in parts:
        p = p.strip()
        if len(p) < 3:
            continue
        if p.isdigit():
            continue
        tags.append(p)

    seen = set()
    out = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def build_notes_tag_index() -> dict:
    """
    Build tag -> count using ALL Notes on JSE list.
    """
    df = load_jse_wikipedia_catalog()
    tag_counts = {}
    for n in df["Notes"].fillna("").astype(str).tolist():
        for tag in extract_note_tags(n):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return tag_counts

def suggest_matching_tags(keyword: str, top_n: int = 25) -> list[str]:
    """
    Show only tags that relate to what user typed.
    """
    kw = (keyword or "").strip().lower()
    if not kw:
        return []

    tag_counts = build_notes_tag_index()
    hits = []
    for tag, cnt in tag_counts.items():
        if kw in tag:
            hits.append((tag, cnt))

    hits.sort(key=lambda x: (-x[1], x[0]))
    return [h[0] for h in hits[:top_n]]

def peers_from_notes_tag(tag_or_keyword: str, k: int = 8) -> list[dict]:
    """
    Pull JSE peers where Notes contains tag/keyword OR tag appears in extracted tags.
    """
    df = load_jse_wikipedia_catalog()
    if df.empty:
        return []

    q = (tag_or_keyword or "").strip().lower()
    if not q:
        return []

    mask = df["Notes"].fillna("").astype(str).str.lower().str.contains(q, na=False)

    if not mask.any():
        tags_list = df["Notes"].fillna("").astype(str).apply(extract_note_tags)
        mask = tags_list.apply(lambda tags: any(q == t for t in tags))

    filt = df[mask].copy()
    if filt.empty:
        return []

    filt = filt.sort_values(["Symbol", "Company"]).head(int(k))
    return filt.to_dict("records")

def format_peer_lines(peer_rows: list[dict]) -> str:
    lines = []
    for r in peer_rows:
        sym = (r.get("Symbol", "") or "").strip()
        nm = (r.get("Company", "") or "").strip()
        notes = (r.get("Notes", "") or "").strip()
        lines.append(f"- **{sym}** — {nm}" + (f" *(Notes: {notes})*" if notes else ""))
    return "\n".join(lines)
# =========================================================
# Yahoo Finance Search (MUST be above yahoo_africa_peers)
# =========================================================
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

def yahoo_search(query: str, quotes_count: int = 50) -> pd.DataFrame:
    params = {"q": query, "quotesCount": int(quotes_count), "newsCount": 0}
    r = requests.get(YAHOO_SEARCH_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for q in data.get("quotes", []) or []:
        rows.append({
            "Company": q.get("shortname") or q.get("longname") or "",
            "Ticker": q.get("symbol") or "",
            "Exchange": q.get("exchange") or "",
            "Type": q.get("quoteType") or "",
        })

    return pd.DataFrame(rows)
# =========================
# Yahoo Africa filter (STRICT)  ✅ UPDATED (Africa-only + bigger peer pool)
# =========================

# ✅ Africa exchanges (tight list; avoid India/US/etc)
AFRICA_EXCHANGES = {
    # South Africa
    "JNB", "JSE", "JOH",

    # Morocco (Casablanca)
    "CAS",

    # Egypt (Yahoo varies)
    "EGX", "CAI",

    # Kenya (Yahoo often uses NBO)
    "NBO",

    # Ghana
    "GSE",

    # Nigeria (Yahoo varies; keep if you see it working for you)
    "NGM", "NSI",

    # East/Southern Africa (Yahoo varies; keep if you see it working)
    "USE", "DSE", "LUSE", "ZSE", "MSE",
}

# ✅ Africa ticker suffixes (most reliable on Yahoo)
AFRICA_SUFFIXES = (
    ".JO",  # South Africa
    ".NG",  # Nigeria (some listings)
    ".KE",  # Kenya (some listings)
    ".GH",  # Ghana (rare)
    ".MU",  # Mauritius
    ".ZM",  # Zambia (rare)
    ".ZW",  # Zimbabwe (rare)
    ".TZ",  # Tanzania (rare)
    ".UG",  # Uganda (rare)
    ".BW",  # Botswana (rare)
)

# 🚫 Block NON-Africa exchanges that were leaking in (India/US/UK etc)
BLOCK_EXCHANGES = {
    "NSE", "BSE",        # India
    "NYQ", "NMS", "NAS", # USA
    "LSE",               # UK
    "HKG",               # Hong Kong
    "JPX",               # Japan
    "TSX", "TOR",        # Canada
}

# 🚫 Block NON-Africa ticker suffixes that were leaking in (India/UK/US etc)
BLOCK_SUFFIXES = (
    ".NS", ".BO",  # India (BIG problem)
    ".L",          # UK
    ".TO", ".V",   # Canada
    ".HK",         # Hong Kong
    ".T",          # Japan
)

BAD_QUOTETYPES = {"FUTURE", "INDEX", "CURRENCY", "CRYPTOCURRENCY", "ETF", "MUTUALFUND", "OPTION"}

def is_africa_quote(row: dict) -> bool:
    # row may be either raw yahoo json-like or our normalized DF row
    tkr = (row.get("symbol") or row.get("Ticker") or "").strip()
    exch = (row.get("exchange") or row.get("Exchange") or "").strip().upper()
    qtype = (row.get("quoteType") or row.get("Type") or "").strip().upper()

    if not tkr:
        return False

    # remove futures/indices etc
    if qtype in BAD_QUOTETYPES:
        return False

    # 🚫 hard block known non-africa tickers
    if tkr.endswith(BLOCK_SUFFIXES):
        return False

    # 🚫 hard block known non-africa exchanges
    if exch in BLOCK_EXCHANGES:
        return False

    # ✅ accept Africa by exchange OR by suffix
    if exch in AFRICA_EXCHANGES:
        return True

    if tkr.endswith(AFRICA_SUFFIXES):
        return True

    return False


def filter_africa(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    # normalize expected column names from yahoo_search()
    if "symbol" in df.columns and "Ticker" not in df.columns:
        df = df.rename(columns={"symbol": "Ticker"})
    if "shortname" in df.columns and "Company" not in df.columns:
        df = df.rename(columns={"shortname": "Company"})
    if "exchange" in df.columns and "Exchange" not in df.columns:
        df = df.rename(columns={"exchange": "Exchange"})
    if "quoteType" in df.columns and "Type" not in df.columns:
        df = df.rename(columns={"quoteType": "Type"})

    rows = []
    for _, r in df.iterrows():
        d = {
            "Company": str(r.get("Company", "") or "").strip(),
            "Ticker": str(r.get("Ticker", "") or "").strip(),
            "Exchange": str(r.get("Exchange", "") or "").strip(),
            "Type": str(r.get("Type", "") or "").strip(),
        }
        # mirror keys for is_africa_quote
        d["symbol"] = d["Ticker"]
        d["exchange"] = d["Exchange"]
        d["quoteType"] = d["Type"]

        if is_africa_quote(d):
            rows.append(d)

    out = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yahoo_africa_peers(sector_keyword: str, limit: int = 200) -> pd.DataFrame:
    """
    Africa peers driven by SECTOR keyword.
    ✅ Build a BIG Africa pool (don’t stop early), then return head(limit).
    """
    sk = (sector_keyword or "").strip()
    if not sk:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    # More Africa-focused queries (broader coverage)
    queries = [
        sk,
        f"{sk} company",
        f"{sk} listed",
        f"{sk} Africa",
        f"{sk} South Africa",
        f"{sk} Nigeria",
        f"{sk} Kenya",
        f"{sk} Egypt",
        f"{sk} Morocco",
        f"{sk} Ghana",
        f"{sk} Uganda",
        f"{sk} Tanzania",
        f"{sk} Zambia",
        f"{sk} Botswana",
        f"{sk} Namibia",
        f"{sk} Mauritius",
    ]

    combined = pd.DataFrame()

    for q in queries:
        try:
            # get more candidates per query
            dfq = yahoo_search(q, quotes_count=400)
            combined = pd.concat([combined, filter_africa(dfq)], ignore_index=True)
        except Exception:
            pass

        combined = combined.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    if combined.empty:
        return pd.DataFrame(columns=["Company", "Ticker", "Exchange", "Type"])

    # double-safe: equities only
    combined["Type"] = combined["Type"].fillna("").astype(str).str.upper()
    combined = combined[~combined["Type"].isin(list(BAD_QUOTETYPES))]

    combined = combined.sort_values(["Exchange", "Company"], ascending=[True, True])
    return combined.head(int(limit)).reset_index(drop=True)


def format_global_peer_lines_yahoo(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return ""
    out = []
    show = df.head(int(max_rows))
    for _, r in show.iterrows():
        nm = str(r.get("Company", "")).strip()
        tk = str(r.get("Ticker", "")).strip()
        ex = str(r.get("Exchange", "")).strip()
        tp = str(r.get("Type", "")).strip()
        meta = " | ".join([x for x in [tk, ex, tp] if x])
        out.append(f"- **{nm}**" + (f" *({meta})*" if meta else ""))
    return "\n".join(out)

# =========================================================
# STEP 1 — INPUT COMPARABLE COMPANIES & MULTIPLES
# =========================================================

st.header("Step 1 — Input Comparable Companies & Multiples")
st.subheader("Auto Peer Suggestions (JSE Wikipedia Notes-based + Yahoo Africa)")

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

S.setdefault("sector_keyword", "")
sector_keyword = st.text_input(
    "Sector keyword (from Notes / industry idea) — drives peers",
    value=S.get("sector_keyword", ""),
    key="sector_keyword_input",
    placeholder="e.g., beverages, banking, insurance, mining, retail, telecoms ...",
)
S["sector_keyword"] = sector_keyword

tag_options = suggest_matching_tags(sector_keyword, top_n=30) if sector_keyword.strip() else []
chosen_tag = None
if tag_options:
    chosen_tag = st.selectbox(
        "Matching Notes tags found on JSE (pick one to be precise)",
        options=["(use my typed keyword)"] + tag_options,
        index=0,
        key="chosen_notes_tag",
    )

sector_used = sector_keyword.strip().lower()
if chosen_tag and chosen_tag != "(use my typed keyword)":
    sector_used = chosen_tag.strip().lower()

if not sector_used and not target_company.strip():
    st.warning("Type a sector keyword (e.g., 'banking') or a target company name first.")
else:
    if sector_used:
        st.info(f"✅ JSE peers will be pulled by Notes tag: **{sector_used}**")
    else:
        st.info("✅ No sector keyword chosen — JSE peers may be empty.")

    jse_peer_rows = peers_from_notes_tag(sector_used, k=int(peer_count)) if sector_used else []

    # ✅ BIG pool slider
    africa_limit = st.slider("How many Africa peers to fetch (Yahoo Finance)", 20, 300, 120, 20)
    africa_df = yahoo_africa_peers(
        sector_keyword=sector_used if sector_used else sector_keyword,
        limit=int(africa_limit),
    )

    peer_universe = st.radio(
        "Which peers do you want to use?",
        ["JSE only", "Africa only (Yahoo Finance)", "Both (JSE + Africa Yahoo)"],
        index=2,
        key="peer_universe_choice",
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🇿🇦 JSE peers (Wikipedia Notes-based)")
        if jse_peer_rows:
            st.success(f"Suggested JSE peers (Notes match: {sector_used})")
            st.markdown(format_peer_lines(jse_peer_rows))

            jse_names = [r["Company"].strip() for r in jse_peer_rows if r.get("Company")]
            chosen_jse = st.multiselect(
                "Select JSE peers to use",
                options=jse_names,
                default=jse_names,
                key="chosen_jse_peers_names",
            )
        else:
            st.warning("No JSE peers found for that Notes keyword (try 'bank', 'insurance', 'telecom').")
            chosen_jse = []

    with c2:
        st.subheader("🌍 Africa peers (Yahoo Finance)")
        st.caption("Africa-only filter (blocks India/US/UK tickers and exchanges).")

        if africa_df is not None and not africa_df.empty:
            st.success(f"Suggested Africa peers (Yahoo Finance) — pool size: {len(africa_df)}")
            st.markdown(format_global_peer_lines_yahoo(africa_df, max_rows=25))

            global_options = []
            for _, r in africa_df.iterrows():
                nm = str(r.get("Company", "")).strip()
                tk = str(r.get("Ticker", "")).strip()
                ex = str(r.get("Exchange", "")).strip()
                tp = str(r.get("Type", "")).strip()
                meta = " | ".join([x for x in [tk, ex, tp] if x])
                label = f"{nm} — {meta}" if meta else nm
                global_options.append(label)

            chosen_global_labels = st.multiselect(
                "Select Africa peers to use",
                options=global_options,
                default=global_options[: min(len(global_options), int(peer_count))],
                key="chosen_africa_peers_labels",
            )
            chosen_global = [x.split(" — ")[0].strip() for x in chosen_global_labels]
        else:
            st.warning("No Africa peers returned. Try a broader keyword (e.g., 'financial', 'insurance', 'bank').")
            chosen_global = []

    if peer_universe == "JSE only":
        selected = chosen_jse
    elif peer_universe == "Africa only (Yahoo Finance)":
        selected = chosen_global
    else:
        selected = chosen_jse + chosen_global

    seen = set()
    selected_final = []
    for nm in selected:
        k = (nm or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        selected_final.append(nm.strip())

    st.info(f"✅ Selected peers to apply: **{len(selected_final)}**")

    if auto_apply and selected_final:
        if "num_comps" not in S or int(S.get("num_comps", 3)) < len(selected_final):
            S["num_comps"] = len(selected_final)
            S["num_comps_input"] = len(selected_final)

        for i, name in enumerate(selected_final):
            S[f"comp_name_{i}"] = name

# ---- your existing Step 1 comparables inputs (unchanged)
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
        "ev": 0.0, "pb": 0.0, "pe": 0.0,
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
        st.checkbox("Include EV", key=ev_key)
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
