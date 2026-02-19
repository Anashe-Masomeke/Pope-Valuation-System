import streamlit as st
from pathlib import Path
import os
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
    st.markdown("""
    <style>

    /* Apply Georgia to everything */
    html, body, [class*="css"]  {
        font-family: Georgia, "Times New Roman", serif !important;
    }

    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        font-family: Georgia, "Times New Roman", serif !important;
    }

    /* Streamlit widgets */
    div, p, span, label {
        font-family: Georgia, "Times New Roman", serif !important;
    }

    /* Dataframes */
    .stDataFrame, .stTable {
        font-family: Georgia, "Times New Roman", serif !important;
    }

    </style>
    """, unsafe_allow_html=True)

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
    - 🧭 USER GUIDE
    """)

st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.markdown("<p class='fbc-title'>Welcome to the FBC Investment Valuation System</p>", unsafe_allow_html=True)
st.write("Explore valuation models using the sidebar or the quick-access buttons below.")

colA, colB, colC, colD, colE, colF = st.columns(6)

if colA.button("📊 DCF Model", width='stretch'):
    st.switch_page("pages/1_DCF.py")

if colB.button("💰 Dividend Discount Model",width='stretch'):
    st.switch_page("pages/3_DDM.py")

if colC.button("📈 Comparables", width='stretch'):
    st.switch_page("pages/2_COMPARABLES.py")

if colD.button("🏦 Banking (RIM)", width='stretch'):
    st.switch_page("pages/4_BANKING.py")

if colE.button("🧾 Summary",width='stretch'):
    st.switch_page("pages/5_summary.py")

if colF.button("🧭 User Guide", width='stretch'):
    st.switch_page("pages/6_user guide.py")

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

