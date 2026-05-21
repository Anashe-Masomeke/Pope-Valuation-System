"""
projects.py  —  FBC Valuation System · Company Project Manager
"""

import streamlit as st
import time
from auth import (
    create_project, list_projects, update_project_meta, delete_project,
    save_project_session, switch_project,
    get_project_data_summary, clear_project_data,
)

# ── Auth guard ────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.warning("🔒 Please sign in first.")
    st.stop()

user     = st.session_state.user
username = user["username"]

# ══════════════════════════════════════════════════════════════════
# STYLES  (same FBC design system)
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

/* BUTTONS */
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

/* PAGE HEADER */
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

/* PROJECT CARDS */
.project-card {
    background: #ffffff;
    border-radius: 16px;
    border-left: 6px solid #003399;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 14px rgba(0,26,92,0.08);
    transition: all 0.22s ease;
}
.project-card:hover {
    box-shadow: 0 10px 28px rgba(0,51,153,0.16);
    transform: translateY(-3px);
}
.project-card-gold { border-left-color: #f5b400 !important; }
.project-card-green { border-left-color: #10b981 !important; }
.project-name {
    font-family: "Playfair Display", serif !important;
    font-size: 20px; font-weight: 800; color: #001a5c !important;
    margin-bottom: 4px;
}
.project-meta {
    font-size: 13px; color: #5a7099 !important;
    font-style: italic; margin-bottom: 10px;
}
.project-badge {
    display: inline-block;
    background: rgba(0,51,153,0.10);
    border: 1px solid rgba(0,51,153,0.20);
    color: #003399 !important;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.10em;
    padding: 3px 10px; border-radius: 999px;
    text-transform: uppercase;
    margin-right: 6px;
}
.project-badge-gold {
    background: rgba(245,180,0,0.15) !important;
    border-color: rgba(245,180,0,0.40) !important;
    color: #b38200 !important;
}

.fbc-divider { border: none; border-top: 1px solid rgba(0,51,153,0.12); margin: 20px 0; }

/* ACTIVE PROJECT BANNER */
.active-project-banner {
    background: linear-gradient(90deg, #001a5c, #003399);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 18px;
    border-left: 5px solid #f5b400;
    display: flex; align-items: center;
}
.active-project-banner-text {
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    font-size: 16px; font-weight: 700;
}
.active-project-banner-sub {
    color: rgba(255,255,255,0.70) !important;
    font-size: 12px; font-style: italic;
}

.fbc-footer {
    text-align: center; padding: 22px; margin-top: 40px;
    color: #5a7099 !important; font-size: 13px;
    border-top: 1px solid rgba(0,51,153,0.10);
}
.fbc-footer b { color: #003399 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧑‍💼 Analyst Profile")
    st.markdown("---")
    st.markdown(f"**Signed in as:** {username}")
    st.markdown(f"*Role: {user['role']}*")
    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True, key="proj_signout"):
        _pid = st.session_state.get("active_project_id")
        if _pid:
            save_project_session(_pid, dict(st.session_state))
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.switch_page("DASHBOARD.py")

# ══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
    <div class="fbc-page-header">
        <span class="fbc-page-header-title">📁 My Valuation Projects</span>
        <p class="fbc-page-header-sub">
            Create, resume and manage company valuations — all inputs and uploaded
            files are saved per-project. Switching projects never mixes data.
        </p>
    </div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ACTIVE PROJECT BANNER
# ══════════════════════════════════════════════════════════════════
active_id   = st.session_state.get("active_project_id")
active_name = st.session_state.get("active_project_name", "")

if active_id:
    _last_ts = st.session_state.get("_autosave_last_ts", 0)
    _ago     = int(time.time() - _last_ts) if _last_ts else None
    _ago_str = f"{_ago}s ago" if _ago is not None else "pending…"
    st.markdown(f"""
        <div class="active-banner">
            <div class="active-banner-title">🏢 Active Project: {active_name}</div>
            <div class="active-banner-sub">
                ✅ <b>Autosave ON</b> — inputs and files saved automatically every 30 s.
                &nbsp;|&nbsp; Last save: <b>{_ago_str}</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_list, tab_new = st.tabs(["📋 My Projects", "➕ New Project"])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — PROJECT LIST
# ──────────────────────────────────────────────────────────────────
with tab_list:
    projects = list_projects(username)

    if not projects:
        st.info("No projects yet. Go to **➕ New Project** to create one.")
    else:
        st.markdown(f"**{len(projects)} project(s) — most recently updated first.**")
        st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

        for proj in projects:
            pid       = proj["id"]
            pname     = proj["company_name"]
            ticker    = proj["ticker"] or "—"
            sector    = proj["sector"] or "—"
            status    = proj["status"]
            created   = proj["created_at"][:10]
            updated   = proj["updated_at"][:16].replace("T", " ")
            is_active = (pid == active_id)

            summary  = get_project_data_summary(pid)
            n_saved  = summary["count"]
            n_files  = summary.get("files", 0)
            last_sv  = (summary["last_saved"] or "Never")[:16]

            card_cls    = "project-card-gold" if is_active else ""
            badge_html  = ('<span class="project-badge project-badge-gold">● Active</span>'
                           if is_active else "")

            st.markdown(f"""
                <div class="project-card {card_cls}">
                    <div class="project-name">🏢 {pname} {badge_html}</div>
                    <div class="project-meta">
                        Ticker: <b>{ticker}</b> &nbsp;|&nbsp;
                        Sector: <b>{sector}</b> &nbsp;|&nbsp;
                        Status: <b>{status}</b>
                    </div>
                    <div class="project-meta">
                        Created: {created} &nbsp;|&nbsp;
                        Last updated: {updated} &nbsp;|&nbsp;
                        <b>{n_saved}</b> inputs &nbsp;|&nbsp;
                        <b>{n_files}</b> file(s) saved &nbsp;|&nbsp;
                        Last save: {last_sv}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            cols = st.columns([2, 2, 2, 2, 1])

            # ── Open & Resume ─────────────────────────────────────
            with cols[0]:
                lbl = "✅ Currently Open" if is_active else "📂 Open & Resume"
                if st.button(lbl, key=f"open_{pid}",
                             use_container_width=True, disabled=is_active):
                    data_r, files_r = switch_project(
                        new_project_id   = pid,
                        new_project_name = pname,
                        session_state    = st.session_state,
                        current_project_id = active_id,
                    )
                    st.success(
                        f"✅ Opened **{pname}** — "
                        f"{data_r} inputs and {files_r} file(s) restored. "
                        "Go to any model page to continue."
                    )
                    st.rerun()

            # ── Save Now ─────────────────────────────────────────
            with cols[1]:
                if st.button("💾 Save Now", key=f"save_{pid}",
                             use_container_width=True, disabled=not is_active,
                             help="Autosave runs every 30 s. Click for immediate save."):
                    ok, msg = save_project_session(pid, dict(st.session_state))
                    st.session_state["_autosave_last_ts"] = time.time()
                    st.success(f"✅ {msg}") if ok else st.error(msg)
                    st.rerun()

            # ── Edit metadata ────────────────────────────────────
            with cols[2]:
                if st.button("✏️ Edit Details", key=f"edit_{pid}",
                             use_container_width=True):
                    st.session_state[f"editing_{pid}"] = True

            # ── Clear data ───────────────────────────────────────
            with cols[3]:
                if st.button("🗑️ Clear Data", key=f"clear_{pid}",
                             use_container_width=True):
                    st.session_state[f"confirm_clear_{pid}"] = True

            # ── Delete ───────────────────────────────────────────
            with cols[4]:
                if st.button("❌", key=f"del_{pid}",
                             use_container_width=True,
                             help="Delete this project"):
                    st.session_state[f"confirm_delete_{pid}"] = True

            # Confirm clear
            if st.session_state.get(f"confirm_clear_{pid}"):
                st.warning(f"⚠️ Clear all saved data and files for **{pname}**? Cannot be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, clear", key=f"yes_clear_{pid}", use_container_width=True):
                        clear_project_data(pid)
                        if is_active:
                            st.session_state.active_project_id   = None
                            st.session_state.active_project_name = ""
                        del st.session_state[f"confirm_clear_{pid}"]
                        st.success(f"Data and files for **{pname}** cleared.")
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"no_clear_{pid}", use_container_width=True):
                        del st.session_state[f"confirm_clear_{pid}"]
                        st.rerun()

            # Confirm delete
            if st.session_state.get(f"confirm_delete_{pid}"):
                st.error(f"⛔ Permanently delete **{pname}** and ALL its data? CANNOT be undone.")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes, delete", key=f"yes_del_{pid}", use_container_width=True):
                        delete_project(pid)
                        if is_active:
                            st.session_state.active_project_id   = None
                            st.session_state.active_project_name = ""
                        del st.session_state[f"confirm_delete_{pid}"]
                        st.success(f"Project **{pname}** deleted.")
                        st.rerun()
                with c2:
                    if st.button("Cancel", key=f"no_del_{pid}", use_container_width=True):
                        del st.session_state[f"confirm_delete_{pid}"]
                        st.rerun()

            # Edit form
            if st.session_state.get(f"editing_{pid}"):
                with st.expander(f"✏️ Edit: {pname}", expanded=True):
                    new_cn  = st.text_input("Company Name",       value=pname,           key=f"ecn_{pid}")
                    new_tk  = st.text_input("Ticker Symbol",      value=proj["ticker"],  key=f"etk_{pid}")
                    new_sec = st.text_input("Sector",             value=proj["sector"],  key=f"esec_{pid}")
                    new_des = st.text_area("Description / Notes", value=proj["description"], key=f"edes_{pid}", height=80)
                    new_st  = st.selectbox("Status",
                        ["In Progress", "Under Review", "Completed", "On Hold"],
                        index=["In Progress","Under Review","Completed","On Hold"].index(
                            proj["status"] if proj["status"] in
                            ["In Progress","Under Review","Completed","On Hold"]
                            else "In Progress"),
                        key=f"est_{pid}")
                    s1, s2 = st.columns(2)
                    with s1:
                        if st.button("💾 Save Changes", key=f"save_edit_{pid}", use_container_width=True):
                            ok, msg = update_project_meta(
                                pid, company_name=new_cn, ticker=new_tk,
                                sector=new_sec, description=new_des, status=new_st)
                            if ok:
                                if is_active:
                                    st.session_state.active_project_name = new_cn
                                del st.session_state[f"editing_{pid}"]
                                st.success(msg); st.rerun()
                            else:
                                st.error(msg)
                    with s2:
                        if st.button("Cancel", key=f"cancel_edit_{pid}", use_container_width=True):
                            del st.session_state[f"editing_{pid}"]
                            st.rerun()

            st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — NEW PROJECT
# ──────────────────────────────────────────────────────────────────
with tab_new:
    st.markdown("### 🏢 Start a New Valuation Project")
    st.markdown(
        "Create a project for a company. All your DCF, DDM, Comparables and Banking "
        "inputs — and every Excel you upload — are saved to this project only. "
        "Opening a different project will never touch them."
    )
    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

    SECTORS = [
        "Select sector…",
        "Banking & Financial Services", "Mining & Resources", "Agriculture",
        "Manufacturing", "Retail & Consumer", "Telecommunications",
        "Energy & Utilities", "Real Estate", "Healthcare & Pharma",
        "Technology", "Transport & Logistics", "Other",
    ]

    new_company = st.text_input("🏢 Company Name *",
                                placeholder="e.g. Econet Wireless Zimbabwe",
                                key="nc_company")
    new_ticker  = st.text_input("📌 Ticker Symbol (optional)",
                                placeholder="e.g. ECO.ZW", key="nc_ticker")
    new_sector  = st.selectbox("🏭 Sector", SECTORS, key="nc_sector")
    new_desc    = st.text_area("📝 Notes / Description (optional)",
                               placeholder="e.g. FY2024 valuation — DCF + DDM",
                               height=90, key="nc_desc")

    err_box = st.empty()
    if st.button("➕ Create Project", key="btn_create_proj"):
        if not new_company.strip():
            err_box.error("Please enter a company name.")
        elif new_sector == SECTORS[0]:
            err_box.error("Please select a sector.")
        else:
            ok, msg, new_pid = create_project(
                username=username,
                company_name=new_company,
                ticker=new_ticker,
                sector=new_sector,
                description=new_desc,
            )
            if ok:
                # Clear stale data so the new project starts fresh
                from auth import _clear_all_file_bytes, _STALE_PARSE_KEYS
                _clear_all_file_bytes(st.session_state)
                for k in _STALE_PARSE_KEYS:
                    st.session_state.pop(k, None)
                st.session_state.active_project_id   = new_pid
                st.session_state.active_project_name = new_company.strip()
                err_box.success(
                    f"✅ {msg} This project is now active. "
                    "Go to any model page to start."
                )
                time.sleep(1.2)
                st.rerun()
            else:
                err_box.error(f"❌ {msg}")

    st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
    st.markdown("""
    **How it works:**
    - Every number, setting and uploaded Excel in DCF / DDM / Banking / Comparables
      is saved to the active project only — completely separate from all other projects.
    - **Autosave is always on** — data and files are saved automatically every 30 seconds.
    - Opening a different project first saves your current work, then clears the session
      and loads the other project's own data and files — nothing ever mixes.
    - Click **Save Now** on any card for an immediate manual save.
    """)

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fbc-footer">
    Powered by <b>FBC Securities</b> · Investment Research &amp; Valuation Dashboard
</div>
""", unsafe_allow_html=True)
