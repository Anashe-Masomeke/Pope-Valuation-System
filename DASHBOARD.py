"""
app.py  —  FBC Investment Valuation System
Entry point: Login / Register / Forgot-Password  →  Dashboard
"""

import streamlit as st
from pathlib import Path
from auth import (
    authenticate,
    register_user,
    get_security_question,
    verify_security_answer,
    reset_password,
)

# ── Must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="FBC Valuation Dashboard",
    layout="wide",
)

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
    font-size: 24px !important;
    font-weight: 900 !important;
    color: #001a5c !important;
    text-align: center;
    margin-bottom: 4px !important;
}
.auth-subtitle {
    font-size: 13px;
    color: #7a90b8 !important;
    text-align: center;
    font-style: italic;
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
    position: fixed; top: 0; left: 0; width: 100%; height: 60px;
    background: linear-gradient(90deg, #002080, #003399);
    display: flex; align-items: center;
    padding: 0 28px; z-index: 99999;
    box-shadow: 0 3px 12px rgba(0,0,0,0.30);
}
.top-title {
    font-size: 20px; font-weight: 800; margin-left: 14px;
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
}
.user-badge {
    margin-left: auto;
    background: rgba(245,180,0,0.20);
    border: 1px solid rgba(245,180,0,0.50);
    color: #f5c842 !important;
    font-size: 13px; font-weight: 700;
    padding: 5px 14px; border-radius: 999px;
    font-family: "EB Garamond", serif !important;
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
    font-size: 14px; color: rgba(255,255,255,0.78) !important;
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
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for key, default in [
    ("authenticated", False),
    ("user", None),
    ("auth_mode", "login"),          # 'login' | 'register' | 'forgot'
    ("reset_step", 1),               # 1=enter username  2=answer  3=new password
    ("reset_username", ""),
    ("active_project_id",   None),   # ID of the currently open project
    ("active_project_name", ""),     # Display name for the active project
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════
# HIDE SIDEBAR ON AUTH PAGES
# ══════════════════════════════════════════════════════════════════
def hide_sidebar():
    st.markdown("""
        <style>
            section[data-testid="stSidebar"]  { display: none !important; }
            [data-testid="collapsedControl"]   { display: none !important; }
        </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SECURITY QUESTIONS LIST
# ══════════════════════════════════════════════════════════════════
SECURITY_QUESTIONS = [
    "Select a security question…",
    "What is the name of your primary school?",
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your favourite book?",
    "What was the make of your first car?",
    "What is the company name?",
]


# ══════════════════════════════════════════════════════════════════
# AUTH PAGE — LOGIN
# ══════════════════════════════════════════════════════════════════
def show_login():
    hide_sidebar()
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        LOGO_PATH = Path("assets") / "fbc log.png"
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=150)

        st.markdown("""
            <p class="auth-title">FBC Valuation System</p>
            <p class="auth-subtitle">Investment Research & Valuation Dashboard</p>
            <hr class="auth-divider">
        """, unsafe_allow_html=True)

        username = st.text_input("👤  Username", key="li_user", placeholder="Enter your username")
        password = st.text_input("🔒  Password", type="password", key="li_pass", placeholder="Enter your password")

        err = st.empty()

        if st.button("Sign In  →", use_container_width=True, key="btn_login"):
            if not username or not password:
                err.error("Please enter both username and password.")
            else:
                user = authenticate(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    err.error("❌  Invalid username or password.")

        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📝 Create Account", use_container_width=True, key="go_register"):
                st.session_state.auth_mode = "register"
                st.rerun()
        with col_b:
            if st.button("🔑 Forgot Password", use_container_width=True, key="go_forgot"):
                st.session_state.auth_mode = "forgot"
                st.session_state.reset_step = 1
                st.session_state.reset_username = ""
                st.rerun()

        st.markdown("""
            <p style="text-align:center; font-size:12px; color:#9aabcc;
                      font-style:italic; margin-top:16px;">
               Authorised personnel only &nbsp;·&nbsp; FBC Securities
            </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# AUTH PAGE — REGISTER
# ══════════════════════════════════════════════════════════════════
def show_register():
    hide_sidebar()
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
            <p class="auth-title">Create Account</p>
            <p class="auth-subtitle">Fill in the details below to register</p>
            <hr class="auth-divider">
        """, unsafe_allow_html=True)

        full_name = st.text_input("🪪  Full Name",    key="reg_name",  placeholder="e.g. Tafara Moyo")
        username  = st.text_input("👤  Username",     key="reg_user",  placeholder="Choose a username (min 3 chars)")
        email     = st.text_input("✉️  Email",        key="reg_email", placeholder="your@email.com  (optional)")
        password  = st.text_input("🔒  Password",     type="password", key="reg_pass",  placeholder="Min 6 characters")
        confirm   = st.text_input("🔒  Confirm Password", type="password", key="reg_conf", placeholder="Repeat password")

        st.markdown("**🛡️ Security Question** *(used to reset your password)*")
        sec_q = st.selectbox("", SECURITY_QUESTIONS, key="reg_sq")
        sec_a = st.text_input("Your Answer", key="reg_sa", placeholder="Answer (not case-sensitive)")

        err = st.empty()

        if st.button("Create Account  →", use_container_width=True, key="btn_register"):
            if password != confirm:
                err.error("Passwords do not match.")
            elif sec_q == SECURITY_QUESTIONS[0]:
                err.error("Please select a security question.")
            else:
                ok, msg = register_user(
                    username=username, password=password,
                    full_name=full_name, email=email,
                    security_question=sec_q, security_answer=sec_a,
                )
                if ok:
                    err.success(msg)
                    import time; time.sleep(1.5)
                    st.session_state.auth_mode = "login"
                    st.rerun()
                else:
                    err.error(msg)

        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="back_login_r"):
            st.session_state.auth_mode = "login"
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# AUTH PAGE — FORGOT PASSWORD  (3-step)
# ══════════════════════════════════════════════════════════════════
def show_forgot():
    hide_sidebar()
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("""
            <p class="auth-title">Reset Password</p>
            <p class="auth-subtitle">Answer your security question to reset</p>
            <hr class="auth-divider">
        """, unsafe_allow_html=True)

        step = st.session_state.reset_step
        err = st.empty()

        # ── Step 1: Enter username ─────────────────────────────────
        if step == 1:
            uname = st.text_input("👤  Username", key="fp_user", placeholder="Enter your username")
            if st.button("Continue  →", use_container_width=True, key="fp_next1"):
                if not uname:
                    err.error("Please enter your username.")
                else:
                    q = get_security_question(uname)
                    if q:
                        st.session_state.reset_username = uname
                        st.session_state.reset_step = 2
                        st.rerun()
                    else:
                        err.error("Username not found or account is inactive.")

        # ── Step 2: Answer security question ──────────────────────
        elif step == 2:
            uname = st.session_state.reset_username
            q = get_security_question(uname)
            st.info(f"🛡️  Security question for **{uname}**:\n\n*{q}*")
            answer = st.text_input("Your Answer", key="fp_ans", placeholder="Enter your answer")
            if st.button("Verify Answer  →", use_container_width=True, key="fp_next2"):
                if not answer:
                    err.error("Please enter your answer.")
                elif verify_security_answer(uname, answer):
                    st.session_state.reset_step = 3
                    st.rerun()
                else:
                    err.error("❌  Incorrect answer. Please try again.")

        # ── Step 3: Set new password ───────────────────────────────
        elif step == 3:
            st.success("✅  Identity verified! Set your new password below.")
            new_pass = st.text_input("🔒  New Password",     type="password", key="fp_new",  placeholder="Min 6 characters")
            confirm  = st.text_input("🔒  Confirm Password", type="password", key="fp_conf", placeholder="Repeat new password")
            if st.button("Reset Password  →", use_container_width=True, key="fp_reset"):
                if not new_pass:
                    err.error("Please enter a new password.")
                elif new_pass != confirm:
                    err.error("Passwords do not match.")
                else:
                    ok, msg = reset_password(st.session_state.reset_username, new_pass)
                    if ok:
                        err.success(msg)
                        import time; time.sleep(1.5)
                        st.session_state.reset_step = 1
                        st.session_state.reset_username = ""
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        err.error(msg)

        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        if st.button("← Back to Sign In", use_container_width=True, key="back_login_f"):
            st.session_state.auth_mode = "login"
            st.session_state.reset_step = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════
def show_dashboard():
    user = st.session_state.user
    LOGO_PATH = Path("assets") / "fbc log.png"

    # ── Autosave active project ───────────────────────────────────
    from auth import autosave_project as _autosave
    _autosave(st.session_state)

    # ── Top nav ───────────────────────────────────────────────────
    st.markdown("<div class='top-nav'>", unsafe_allow_html=True)
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=160)
    st.markdown(
        f"<span class='top-title'>FBC Valuation Dashboard</span>"
        f"<span class='user-badge'>👤 {user['full_name'] or user['username']}  "
        f"[{user['role'].upper()}]</span>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🧑‍💼 Analyst Profile")

        st.markdown("---")
        st.markdown(f"**Signed in as:** {user['username']}")
        st.markdown(f"*Role: {user['role']}*")
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True, key="signout"):
            from auth import save_project_session as _save_proj
            _pid = st.session_state.get("active_project_id")
            if _pid:
                _save_proj(_pid, dict(st.session_state))
            for _k in list(st.session_state.keys()):
                del st.session_state[_k]
            st.rerun()

    # ── Page header ───────────────────────────────────────────────
    st.markdown(f"""
        <div class="fbc-page-header">
            <span class="fbc-page-header-title">
                🏛️ Welcome to FBC Investment Valuation System
            </span>
            <p class="fbc-page-header-sub">
                Good day, {user['full_name'] or user['username']} —
                select a valuation model to get started.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Active project banner ─────────────────────────────────────
    active_id   = st.session_state.get("active_project_id")
    active_name = st.session_state.get("active_project_name", "")
    if active_id:
        st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #001a5c, #003399);
                border-radius: 12px; padding: 14px 20px; margin-bottom: 18px;
                border-left: 5px solid #f5b400;">
                <div style="color:#ffffff !important;
                            font-family:'Playfair Display',serif;
                            font-size:16px; font-weight:700;">
                    🏢 Active Project: {active_name}
                </div>
                <div style="color:rgba(255,255,255,0.72) !important;
                            font-size:12px; font-style:italic;">
                    All model inputs are being saved to this project.
                    Visit <b>My Projects</b> to save or switch.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info(
            "💡 No project is active. Go to **📁 My Projects** to create or resume "
            "a company valuation — your work will be saved automatically."
        )

    # ── Quick-access buttons  ← FIXED PAGE PATHS ─────────────────
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    colP, colA, colB, colC, colD, colE, colF = st.columns(7)

    if colP.button("📁 My Projects",             use_container_width=True): st.switch_page("pages/projects.py")
    if colA.button("📊 DCF Model",               use_container_width=True): st.switch_page("pages/dcf.py")
    if colB.button("💰 Dividend Discount Model",  use_container_width=True): st.switch_page("pages/DDM.py")
    if colC.button("📈 Comparables",             use_container_width=True): st.switch_page("pages/comparables.py")
    if colD.button("🏦 Banking (RIM)",            use_container_width=True): st.switch_page("pages/BANKING.py")
    if colE.button("🧾 Summary",                  use_container_width=True): st.switch_page("pages/summary.py")
    if colF.button("🧭 User Guide",               use_container_width=True): st.switch_page("pages/user_guide.py")

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    # ── Feature cards ─────────────────────────────────────────────
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

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="fbc-footer">
        Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════
if st.session_state.authenticated:
    show_dashboard()
else:
    mode = st.session_state.auth_mode
    if mode == "register":
        show_register()
    elif mode == "forgot":
        show_forgot()
    else:
        show_login()
