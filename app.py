import streamlit as st
from pathlib import Path
import os

current_page = os.path.basename(__file__).replace(".py", "").lower()

st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide"
)

CUSTOM_STYLE = """
<style>
.stApp { background: linear-gradient(135deg, #e8f1ff 0%, #ffffff 70%); animation: fadeIn 1.2s ease-in-out; }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

.top-nav {
    position: fixed; top: 0; left: 0; width: 100%; height: 65px;
    background-color: #003399; color: white; display: flex; align-items: center;
    padding-left: 25px; padding-right: 25px; z-index: 99999;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
}
.top-title { font-size: 26px; font-weight: bold; margin-left: 12px; }

.main-content { max-width: 1100px; margin: 100px auto 40px auto; padding: 20px; }
.fbc-title { font-size: 42px; font-weight: bold; color: #003399; margin-bottom: 10px; }

.nav-btn { background-color: #003399; color: white; padding: 14px 25px; border-radius: 10px;
    text-align: center; font-size: 17px; margin-bottom: 12px; font-weight: 600; transition: 0.2s; }
.nav-btn:hover { background-color: #0055dd; box-shadow: 0 0 10px #0055ddaa; }

.feature-box {
    background: #ffffffcc; padding: 22px; border-radius: 14px; border-left: 6px solid #003399;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08); transition: all 0.25s ease-in-out;
}
.feature-box:hover { background: #e8f0ff; box-shadow: 0 0 18px #00339955; transform: translateY(-4px); }
.feature-icon { font-size: 22px; color: #003399; margin-right: 8px; }

.footer { text-align: center; padding: 25px; margin-top: 40px; color: #003399; font-weight: 600; }
</style>
"""

st.markdown(CUSTOM_STYLE, unsafe_allow_html=True)

LOGO_PATH = Path("assets") / "fbc_logo.png"
st.markdown("<div class='top-nav'>", unsafe_allow_html=True)
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=45)
st.markdown("<span class='top-title'>FBC Valuation Dashboard</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:

    st.markdown("### 📂 Navigation")
    st.markdown("""
    - 📊 DCF Model  
    - 💰 DDM  
    - 📈 Comparables  
    - 🏦 Banking (RIM)  
    - 🧾 Summary  
    """)

st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.markdown("<p class='fbc-title'>Welcome to the FBC Investment Valuation System</p>", unsafe_allow_html=True)
st.write("Explore valuation models using the sidebar or the quick-access buttons below.")

colA, colB, colC, colD, colE = st.columns(5)

if colA.button("📊 DCF Model", use_container_width=True):
    st.switch_page("pages/DCF.py")

if colB.button("💰 DDM Model", use_container_width=True):
    st.switch_page("pages/DDM.py")

if colC.button("📈 Comparables", use_container_width=True):
    st.switch_page("pages/COMPARABLES.py")

if colD.button("🏦 Banking (RIM)", use_container_width=True):
    st.switch_page("pages/BANKING.py")

if colE.button("🧾 Summary", use_container_width=True):
    st.switch_page("pages/summary.py")

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

st.markdown("""
<div class="footer">
    Powered by <b>FBC Securities</b> • Investment Research & Valuation Dashboard
</div>
""", unsafe_allow_html=True)
