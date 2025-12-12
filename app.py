import streamlit as st
from pathlib import Path
import os

current_page = os.path.basename(__file__).replace(".py", "").lower()

st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide"
)

# =========================================================
# CSS: FBC THEME + PROFILE CARD + DARK MODE + NAV + CARDS
# =========================================================
CUSTOM_STYLE = """
<style>

.stApp {
    background: linear-gradient(135deg, #e8f1ff 0%, #ffffff 70%);
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}

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

/* -------- NAVIGATION BUTTONS -------- */
.nav-btn {
    background-color: #003399;
    color: white;
    padding: 14px 25px;
    border-radius: 10px;
    text-align: center;
    font-size: 17px;
    margin-bottom: 12px;
    font-weight: 600;
    transition: 0.2s;
}
.nav-btn:hover {
    background-color: #0055dd;
    box-shadow: 0 0 10px #0055ddaa;
}

/* -------- FEATURE CARDS -------- */
.feature-box {
    background: #ffffffcc;
    padding: 22px;
    border-radius: 14px;
    border-left: 6px solid #003399;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08);
    transition: all 0.25s ease-in-out;
}
.feature-box:hover {
    background: #e8f0ff;
    box-shadow: 0 0 18px #00339955;
    transform: translateY(-4px);
}
.feature-icon {
    font-size: 22px;
    color: #003399;
    margin-right: 8px;
}

/* -------- SIDEBAR PROFILE CARD -------- */
.profile-card {
    background: linear-gradient(135deg, #ffffffee 0%, #eef3ffdd 100%);
    border: 1px solid #c5d3ee;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 4px 12px rgba(0,45,120,0.15);
    transition: all 0.22s ease-in-out;
}
.profile-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(0,60,160,0.25);
}
.profile-avatar {
    width: 78px;
    height: 78px;
    border-radius: 50%;
    border: 3px solid #003399;
    box-shadow: 0 0 8px rgba(0,0,0,0.15);
    margin-bottom: 10px;
}
.profile-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #003399;
}
.profile-role {
    font-size: 0.83rem;
    color: #667fb0;
    margin-bottom: 4px;
}
.profile-company {
    font-size: 0.78rem;
    color: #3b4b6c;
    font-weight: 600;
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    padding: 25px;
    margin-top: 40px;
    color: #003399;
    font-weight: 600;
}

/* -------- DARK MODE -------- */
.darkmode-btn {
    position: fixed;
    right: 25px;
    top: 75px;
    padding: 8px 16px;
    background: #003399;
    color: white;
    border-radius: 6px;
    cursor: pointer;
    z-index: 100000;
}

</style>
"""

st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

# =========================================================
# TOP NAVIGATION BAR
# =========================================================
LOGO_PATH = Path("assets") / "fbc_logo.png"
st.markdown("<div class='top-nav'>", unsafe_allow_html=True)

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=45)

st.markdown("<span class='top-title'>FBC Valuation Dashboard</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SIDEBAR — USER PROFILE CARD
# =========================================================
with st.sidebar:
    st.markdown("## ")

    # -------------------- USER PROFILE --------------------
    avatar_path = "assets/profile.png"

    if not Path(avatar_path).exists():
        avatar_path = "https://ui-avatars.com/api/?name=A+M&background=003399&color=fff&size=128"

    st.markdown(
        f"""
        <div class="profile-card">
            <img src="{avatar_path}" class="profile-avatar">
            <div class="profile-name">ANASHE MASOMEKE</div>
            <div class="profile-role">ENGINEER</div>
            <div class="profile-company">FBC SECURITIES</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Navigation")

    st.markdown("""
    - 📊 DCF Model (multi-industry)  
    - 💰 DDM  
    - 📈 Comparables  
    - 🧾 Summary  
    """)

# =========================================================
# DARK MODE TOGGLE
# =========================================================
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

toggle = st.checkbox("Dark Mode", value=st.session_state["dark_mode"])
st.session_state["dark_mode"] = toggle

if toggle:
    st.markdown("""
    <style>
    .stApp {
        background: #1a1a1a !important;
        color: white !important;
    }
    .feature-box {
        background: #2a2a2aaa !important;
        border-left: 6px solid #66aaff !important;
    }
    .profile-card {
        background: #2f2f2f !important;
        border-color: #444 !important;
    }
    .profile-name { color: #79c0ff !important; }
    .profile-role, .profile-company { color: #cccccc !important; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN CONTENT
# =========================================================
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown("<p class='fbc-title'>Welcome to the FBC Investment Valuation System</p>", unsafe_allow_html=True)

st.write("Explore valuation models using the sidebar or the quick-access buttons below.")

# =========================================================
# NAVIGATION BUTTONS
# =========================================================
colA, colB, colC, colD = st.columns(4)

if colA.button("📊 DCF Model", use_container_width=True):
    st.switch_page("pages/DCF.py")

if colB.button("💰 DDM Model", use_container_width=True):
    st.switch_page("pages/DDM.py")

if colC.button("📈 Comparables", use_container_width=True):
    st.switch_page("pages/COMPARABLES.py")

if colD.button("🧾 Summary", use_container_width=True):
    st.switch_page("pages/summary.py")

# =========================================================
# FEATURE CARDS
# =========================================================
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
        <span class="feature-icon">🧾</span>
        <b>Summary Valuation</b><br>
        Weighted blended model combining all approaches.
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Powered by <b>FBC Securities</b> • Investment Research & Valuation Dashboard
</div>
""", unsafe_allow_html=True)
