import streamlit as st
from pathlib import Path

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# STYLES (FIX SIDEBAR ICON + NICE COLLAPSE BUTTON)
# ------------------------------------------------------------
CUSTOM_STYLE = """
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

/* ------------------------------------------------------------
   TOP NAV BAR
------------------------------------------------------------ */
.top-nav {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 65px;
    background-color: #003399;
    color: white;
    display: flex;
    align-items: center;
    padding-left: 25px;
    padding-right: 25px;
    z-index: 99999;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
}

.top-title {
    font-size: 26px;
    font-weight: bold;
    margin-left: 12px;
}

/* ------------------------------------------------------------
   MAIN CONTENT
------------------------------------------------------------ */
.main-content {
    max-width: 1100px;
    margin: 100px auto 40px auto;
    padding: 20px;
}

.fbc-title {
    font-size: 42px;
    font-weight: bold;
    color: #003399;
    margin-bottom: 10px;
}

/* ------------------------------------------------------------
   FEATURE CARDS
------------------------------------------------------------ */
.feature-box {
    background: #ffffff;
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #003399;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
    transition: all 0.25s ease-in-out;
    margin-bottom: 14px;
}
.feature-box:hover {
    background: #f5f9ff;
    box-shadow: 0 0 18px #00339933;
    transform: translateY(-4px);
}

.feature-icon {
    font-size: 22px;
    color: #003399;
    margin-right: 8px;
}

/* ------------------------------------------------------------
   FOOTER
------------------------------------------------------------ */
.footer {
    text-align: center;
    padding: 25px;
    margin-top: 40px;
    color: #003399;
    font-weight: 600;
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
    st.image(str(LOGO_PATH), width=150)

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
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown("<p class='fbc-title'>Welcome to the FBC Investment Valuation System</p>", unsafe_allow_html=True)
st.write("Explore valuation models using the sidebar or the quick-access buttons below.")

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
    st.switch_page("pages/5_summary.py")

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
