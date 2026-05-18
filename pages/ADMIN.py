"""
8_ADMIN.py  —  FBC Valuation System · Admin Dashboard
Only accessible to users with role = 'admin'.
Gives full visibility and control over users, projects, and system activity.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from auth import (
    admin_list_all_projects,
    admin_get_stats,
    admin_get_user,
    admin_update_user,
    admin_reset_password,
    admin_delete_user,
    admin_get_full_login_history,
    admin_delete_project,
    list_users,
    register_user,
    get_project_data_summary,
)

# ── Auth + Role Guard ─────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.error("🔒 Please sign in first.")
    st.stop()

if st.session_state.user.get("role") != "admin":
    st.error("⛔ Access denied. This page is for administrators only.")
    st.stop()

admin_user = st.session_state.user["username"]
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Material+Icons&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

/* ═══════════════════════════════════════════════════════════
   GLOBAL TYPOGRAPHY
═══════════════════════════════════════════════════════════ */
html, body, .stApp, .block-container,
p, div, label,
h1, h2, h3, h4, h5, h6,
li, ul, ol, a, small {
    font-family: "EB Garamond", Georgia, "Times New Roman", serif !important;
    color: #1a1a2e;
}

h1, h2, h3, h4 {
    font-family: "Playfair Display", Georgia, serif !important;
    font-weight: 700 !important;
}

/* FIX MATERIAL ICONS */
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

/* ═══════════════════════════════════════════════════════════
   APP BACKGROUND
═══════════════════════════════════════════════════════════ */
.stApp {
    background: #f0f2f8 !important;
}

.main .block-container {
    background: #f0f2f8 !important;
    padding-top: 1.5rem !important;
}

/* ═══════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg,#1a0033 0%,#3d0066 45%,#5c0099 100%) !important;
    border-right: 2px solid rgba(245,180,0,0.3) !important;
    box-shadow: 4px 0 24px rgba(26,0,51,0.4) !important;
}

section[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg,#f5b400,#ffd040,#f5b400);
}

section[data-testid="stSidebar"] * {
    color: #f0e6ff !important;
    font-family: "EB Garamond", serif !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: "Playfair Display", serif !important;
    font-weight: 700 !important;
}

section[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(245,180,0,0.25) !important;
    margin: 12px 0 !important;
}

/* ═══════════════════════════════════════════════════════════
   REMOVE keybo... TEXT COMPLETELY
═══════════════════════════════════════════════════════════ */

/* Sidebar header layout */
[data-testid="stSidebarHeader"] {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    padding-right: 10px !important;
}

/* Hide ALL sidebar title/app text */
[data-testid="stSidebarHeader"] p,
[data-testid="stSidebarHeader"] span,
[data-testid="stSidebarHeader"] h1,
[data-testid="stSidebarHeader"] h2,
[data-testid="stSidebarHeader"] h3,
[data-testid="stSidebarHeader"] svg:first-child,
[data-testid="stSidebarHeader"] div:not([data-testid="stSidebarCollapseButton"]) {
    display: none !important;
}

/* ═══════════════════════════════════════════════════════════
   COLLAPSE BUTTON
═══════════════════════════════════════════════════════════ */
[data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 10px !important;
    right: 10px !important;
    left: auto !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapseButton"] button {
    background: linear-gradient(135deg,#3d0066,#7700cc) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 44px !important;
    height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 14px rgba(61,0,102,0.40) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebarCollapseButton"] button:hover {
    transform: translateY(-2px) scale(1.05) !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    width: 22px !important;
    height: 22px !important;
    fill: #ffffff !important;
}

/* ═══════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg,#3d0066 0%,#7700cc 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: "EB Garamond", serif !important;
    font-size: 15px !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 14px rgba(61,0,102,0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
}

/* ═══════════════════════════════════════════════════════════
   STAT CARDS
═══════════════════════════════════════════════════════════ */
.stat-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(61,0,102,0.10);
    border-top: 4px solid #7700cc;
}

.stat-number {
    font-family: "Playfair Display", serif !important;
    font-size: 38px !important;
    font-weight: 900 !important;
    color: #3d0066 !important;
    line-height: 1;
}

.stat-label {
    font-size: 13px !important;
    color: #7a5c99 !important;
    margin-top: 6px;
    font-style: italic;
}

.stat-card-gold  { border-top-color:#f5b400 !important; }
.stat-card-green { border-top-color:#10b981 !important; }
.stat-card-red   { border-top-color:#ef4444 !important; }

/* ═══════════════════════════════════════════════════════════
   PAGE HEADER
═══════════════════════════════════════════════════════════ */
.admin-header {
    background: linear-gradient(135deg,#1a0033 0%,#3d0066 50%,#7700cc 100%);
    border-radius: 18px;
    padding: 26px 32px;
    margin-bottom: 28px;
    border-bottom: 3px solid #f5b400;
    box-shadow: 0 12px 40px rgba(26,0,51,0.35);
}

.admin-header-title {
    font-family: "Playfair Display", serif !important;
    font-size: 28px !important;
    font-weight: 900 !important;
    color: #ffffff !important;
}

.admin-header-sub {
    font-size: 14px;
    color: rgba(255,255,255,0.75) !important;
    margin-top: 6px;
    font-style: italic;
}

/* ═══════════════════════════════════════════════════════════
   USER CARDS
═══════════════════════════════════════════════════════════ */
.user-card {
    background: #ffffff;
    border-radius: 14px;
    border-left: 6px solid #7700cc;
    padding: 18px 22px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(61,0,102,0.08);
}

.user-card-inactive {
    border-left-color: #9ca3af !important;
    opacity: 0.75;
}

.user-card-admin {
    border-left-color: #f5b400 !important;
}

.user-name {
    font-family: "Playfair Display", serif !important;
    font-size: 18px;
    font-weight: 800;
    color: #1a0033 !important;
    margin-bottom: 3px;
}

.user-meta {
    font-size: 13px;
    color: #6b46a0 !important;
    font-style: italic;
}

/* ═══════════════════════════════════════════════════════════
   BADGES
═══════════════════════════════════════════════════════════ */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 5px;
}

.badge-admin {
    background: rgba(245,180,0,0.15);
    color: #b38200 !important;
    border: 1px solid rgba(245,180,0,0.4);
}

.badge-analyst {
    background: rgba(119,0,204,0.10);
    color: #5c0099 !important;
    border: 1px solid rgba(119,0,204,0.25);
}

.badge-active {
    background: rgba(16,185,129,0.12);
    color: #065f46 !important;
    border: 1px solid rgba(16,185,129,0.3);
}

.badge-inactive {
    background: rgba(156,163,175,0.15);
    color: #4b5563 !important;
    border: 1px solid #d1d5db;
}

/* DIVIDER */
.fbc-divider {
    border: none;
    border-top: 1px solid rgba(119,0,204,0.12);
    margin: 20px 0;
}

/* SECTION TITLE */
.section-title {
    font-family: "Playfair Display", serif !important;
    font-size: 20px;
    font-weight: 800;
    color: #3d0066 !important;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(119,0,204,0.15);
}

/* FOOTER */
.fbc-footer {
    text-align: center;
    padding: 22px;
    margin-top: 40px;
    color: #7a5c99 !important;
    font-size: 13px;
    border-top: 1px solid rgba(119,0,204,0.12);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Admin Panel")
    st.markdown("---")
    st.markdown(f"**Signed in as:** {admin_user}")
    st.markdown("*Role: Administrator*")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
    - 🏠 [Dashboard](/)
    - 📁 My Projects
    - 🛡️ **Admin Panel** ← here
    """)
    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True, key="admin_signout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.switch_page("app.py")


# ══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
    <div class="admin-header">
        <div class="admin-header-title">🛡️ System Administration</div>
        <div class="admin-header-sub">
            Full control over users, projects, and system activity.
            Changes take effect immediately.
        </div>
    </div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LIVE STATS
# ══════════════════════════════════════════════════════════════════
stats = admin_get_stats()

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, stats["total_users"],    "Total Users",        ""),
    (c2, stats["active_users"],   "Active Users",       "stat-card-green"),
    (c3, stats["total_projects"], "Total Projects",     ""),
    (c4, stats["total_files"],    "Files Stored",       ""),
    (c5, stats["logins_today"],   "Logins Today",       "stat-card-green"),
    (c6, stats["failed_today"],   "Failed Logins Today","stat-card-red"),
]
for col, num, label, extra in cards:
    with col:
        st.markdown(f"""
            <div class="stat-card {extra}">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_users, tab_new_user, tab_projects, tab_activity = st.tabs([
    "👥 Manage Users",
    "➕ Create User",
    "📁 All Projects",
    "📋 Login Activity",
])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — MANAGE USERS
# ──────────────────────────────────────────────────────────────────
with tab_users:
    st.markdown('<div class="section-title">👥 All Registered Users</div>', unsafe_allow_html=True)

    users = list_users()
    if not users:
        st.info("No users found.")
    else:
        # Search bar
        search = st.text_input("🔍 Search by username or name", key="admin_user_search",
                               placeholder="Type to filter...")

        filtered = [u for u in users if
                    search.lower() in u["username"].lower() or
                    search.lower() in (u["full_name"] or "").lower()
                    ] if search else users

        st.markdown(f"**{len(filtered)} user(s) shown**")
        st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

        for u in filtered:
            uname    = u["username"]
            fname    = u["full_name"] or "—"
            email    = u["email"] or "—"
            role     = u["role"]
            active   = u["is_active"]
            created  = u["created_at"][:10]

            # Count their projects
            all_proj = admin_list_all_projects()
            u_proj   = [p for p in all_proj if p["username"] == uname]

            role_badge   = f'<span class="badge badge-{role}">{role}</span>'
            status_badge = (f'<span class="badge badge-active">● Active</span>'
                            if active else
                            f'<span class="badge badge-inactive">○ Inactive</span>')
            card_cls     = ("user-card-admin" if role == "admin" else
                            "" if active else "user-card-inactive")

            st.markdown(f"""
                <div class="user-card {card_cls}">
                    <div class="user-name">{fname} ({uname}) {role_badge} {status_badge}</div>
                    <div class="user-meta">
                        Email: <b>{email}</b> &nbsp;|&nbsp;
                        Registered: <b>{created}</b> &nbsp;|&nbsp;
                        Projects: <b>{len(u_proj)}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])

            # ── Edit profile ──────────────────────────────────────
            with col1:
                if st.button("✏️ Edit Profile", key=f"adm_edit_{uname}",
                             use_container_width=True):
                    st.session_state[f"adm_editing_{uname}"] = True

            # ── Reset password ────────────────────────────────────
            with col2:
                if st.button("🔑 Reset Password", key=f"adm_pw_{uname}",
                             use_container_width=True):
                    st.session_state[f"adm_pw_{uname}"] = True

            # ── Toggle active/inactive ────────────────────────────
            with col3:
                toggle_label = "🔴 Deactivate" if active else "🟢 Activate"
                if st.button(toggle_label, key=f"adm_tog_{uname}",
                             use_container_width=True,
                             disabled=(uname == admin_user)):
                    new_status = 0 if active else 1
                    admin_update_user(uname, is_active=new_status)
                    st.success(f"{'Deactivated' if new_status==0 else 'Activated'} '{uname}'.")
                    st.rerun()

            # ── Change role ───────────────────────────────────────
            with col4:
                new_role = "analyst" if role == "admin" else "admin"
                role_label = f"⬆️ Make Admin" if new_role == "admin" else "⬇️ Make Analyst"
                if st.button(role_label, key=f"adm_role_{uname}",
                             use_container_width=True,
                             disabled=(uname == admin_user)):
                    admin_update_user(uname, role=new_role)
                    st.success(f"'{uname}' is now {new_role}.")
                    st.rerun()

            # ── Delete user ───────────────────────────────────────
            with col5:
                if st.button("🗑️", key=f"adm_del_{uname}",
                             use_container_width=True,
                             disabled=(uname == admin_user),
                             help="Delete this user and ALL their data"):
                    st.session_state[f"adm_confirm_del_{uname}"] = True

            # ── Edit profile form ─────────────────────────────────
            if st.session_state.get(f"adm_editing_{uname}"):
                with st.expander(f"✏️ Edit Profile: {uname}", expanded=True):
                    new_fn  = st.text_input("Full Name", value=u["full_name"] or "",
                                            key=f"adm_fn_{uname}")
                    new_em  = st.text_input("Email",     value=u["email"] or "",
                                            key=f"adm_em_{uname}")
                    s1, s2 = st.columns(2)
                    with s1:
                        if st.button("💾 Save", key=f"adm_save_edit_{uname}",
                                     use_container_width=True):
                            ok, msg = admin_update_user(uname,
                                                        full_name=new_fn,
                                                        email=new_em)
                            if ok:
                                del st.session_state[f"adm_editing_{uname}"]
                                st.success(msg); st.rerun()
                            else:
                                st.error(msg)
                    with s2:
                        if st.button("Cancel", key=f"adm_cancel_edit_{uname}",
                                     use_container_width=True):
                            del st.session_state[f"adm_editing_{uname}"]
                            st.rerun()

            # ── Reset password form ───────────────────────────────
            if st.session_state.get(f"adm_pw_{uname}"):
                with st.expander(f"🔑 Reset Password: {uname}", expanded=True):
                    new_pw = st.text_input("New Password", type="password",
                                           key=f"adm_newpw_{uname}")
                    conf_pw = st.text_input("Confirm Password", type="password",
                                            key=f"adm_confpw_{uname}")
                    p1, p2 = st.columns(2)
                    with p1:
                        if st.button("🔑 Reset", key=f"adm_do_pw_{uname}",
                                     use_container_width=True):
                            if new_pw != conf_pw:
                                st.error("Passwords don't match.")
                            else:
                                ok, msg = admin_reset_password(uname, new_pw)
                                if ok:
                                    del st.session_state[f"adm_pw_{uname}"]
                                    st.success(msg); st.rerun()
                                else:
                                    st.error(msg)
                    with p2:
                        if st.button("Cancel", key=f"adm_cancel_pw_{uname}",
                                     use_container_width=True):
                            del st.session_state[f"adm_pw_{uname}"]
                            st.rerun()

            # ── Confirm delete ────────────────────────────────────
            if st.session_state.get(f"adm_confirm_del_{uname}"):
                st.error(
                    f"⛔ Delete **{uname}** and ALL their projects/data? "
                    "This CANNOT be undone."
                )
                d1, d2 = st.columns(2)
                with d1:
                    if st.button("Yes, delete user", key=f"adm_yes_del_{uname}",
                                 use_container_width=True):
                        ok, msg = admin_delete_user(uname)
                        del st.session_state[f"adm_confirm_del_{uname}"]
                        st.success(msg); st.rerun()
                with d2:
                    if st.button("Cancel", key=f"adm_no_del_{uname}",
                                 use_container_width=True):
                        del st.session_state[f"adm_confirm_del_{uname}"]
                        st.rerun()

            st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TAB 2 — CREATE NEW USER
# ──────────────────────────────────────────────────────────────────
with tab_new_user:
    st.markdown('<div class="section-title">➕ Create New User Account</div>',
                unsafe_allow_html=True)
    st.markdown("Create accounts for new analysts. They can change their password after first login.")

    col_a, col_b = st.columns(2)
    with col_a:
        nu_username = st.text_input("Username *",        key="nu_username",
                                    placeholder="e.g. jsmith")
        nu_fullname = st.text_input("Full Name *",       key="nu_fullname",
                                    placeholder="e.g. John Smith")
        nu_email    = st.text_input("Email",             key="nu_email",
                                    placeholder="e.g. jsmith@fbc.co.zw")
    with col_b:
        nu_password  = st.text_input("Password *",       key="nu_password",
                                     type="password",
                                     placeholder="Minimum 6 characters")
        nu_confirm   = st.text_input("Confirm Password *", key="nu_confirm",
                                     type="password")
        nu_role      = st.selectbox("Role", ["analyst", "admin"], key="nu_role")

    st.markdown("**Security Question** (for password recovery)")
    col_c, col_d = st.columns(2)
    with col_c:
        nu_sq = st.text_input("Security Question *", key="nu_sq",
                              placeholder="e.g. What city were you born in?")
    with col_d:
        nu_sa = st.text_input("Security Answer *",   key="nu_sa",
                              type="password",
                              placeholder="Answer (case-insensitive)")

    err_nu = st.empty()
    if st.button("➕ Create Account", key="nu_create_btn", use_container_width=False):
        if not nu_username.strip():
            err_nu.error("Username is required.")
        elif not nu_fullname.strip():
            err_nu.error("Full name is required.")
        elif not nu_password:
            err_nu.error("Password is required.")
        elif nu_password != nu_confirm:
            err_nu.error("Passwords do not match.")
        elif not nu_sq.strip() or not nu_sa.strip():
            err_nu.error("Security question and answer are required.")
        else:
            ok, msg = register_user(
                username=nu_username,
                password=nu_password,
                full_name=nu_fullname,
                email=nu_email,
                security_question=nu_sq,
                security_answer=nu_sa,
                role=nu_role,
            )
            if ok:
                err_nu.success(f"✅ {msg}")
                st.rerun()
            else:
                err_nu.error(f"❌ {msg}")


# ──────────────────────────────────────────────────────────────────
# TAB 3 — ALL PROJECTS
# ──────────────────────────────────────────────────────────────────
with tab_projects:
    st.markdown('<div class="section-title">📁 All Valuation Projects</div>',
                unsafe_allow_html=True)

    all_projects = admin_list_all_projects()

    if not all_projects:
        st.info("No projects exist yet.")
    else:
        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            proj_search = st.text_input("🔍 Search company or analyst",
                                        key="adm_proj_search",
                                        placeholder="Type to filter…")
        with fc2:
            status_filter = st.selectbox("Filter by status",
                                         ["All", "In Progress", "Completed",
                                          "Under Review", "On Hold"],
                                         key="adm_status_filter")

        filtered_proj = all_projects
        if proj_search:
            filtered_proj = [p for p in filtered_proj if
                             proj_search.lower() in p["company_name"].lower() or
                             proj_search.lower() in p["username"].lower()]
        if status_filter != "All":
            filtered_proj = [p for p in filtered_proj if p["status"] == status_filter]

        st.markdown(f"**{len(filtered_proj)} project(s) shown**")

        # Summary table
        if filtered_proj:
            df = pd.DataFrame([{
                "Analyst":       p["username"],
                "Company":       p["company_name"],
                "Ticker":        p["ticker"] or "—",
                "Sector":        p["sector"] or "—",
                "Status":        p["status"],
                "Inputs Saved":  p["data_count"],
                "Files Saved":   p["file_count"],
                "Last Updated":  p["updated_at"][:16],
                "Created":       p["created_at"][:10],
            } for p in filtered_proj])
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)
        st.markdown("**Delete a project:**")

        # Delete controls
        for p in filtered_proj:
            pid   = p["id"]
            pname = p["company_name"]
            puser = p["username"]

            col_i, col_ii = st.columns([5, 1])
            with col_i:
                st.markdown(f"**{pname}** — *{puser}* — {p['status']}")
            with col_ii:
                if st.button("🗑️ Delete", key=f"adm_del_proj_{pid}",
                             use_container_width=True):
                    st.session_state[f"adm_confirm_proj_{pid}"] = True

            if st.session_state.get(f"adm_confirm_proj_{pid}"):
                st.error(f"⛔ Delete project **{pname}** by {puser}? Cannot be undone.")
                dp1, dp2 = st.columns(2)
                with dp1:
                    if st.button("Yes, delete", key=f"adm_yes_proj_{pid}",
                                 use_container_width=True):
                        ok, msg = admin_delete_project(pid)
                        del st.session_state[f"adm_confirm_proj_{pid}"]
                        st.success(msg); st.rerun()
                with dp2:
                    if st.button("Cancel", key=f"adm_no_proj_{pid}",
                                 use_container_width=True):
                        del st.session_state[f"adm_confirm_proj_{pid}"]
                        st.rerun()


# ──────────────────────────────────────────────────────────────────
# TAB 4 — LOGIN ACTIVITY
# ──────────────────────────────────────────────────────────────────
with tab_activity:
    st.markdown('<div class="section-title">📋 Login Audit Log</div>',
                unsafe_allow_html=True)

    logs = admin_get_full_login_history(limit=300)

    if not logs:
        st.info("No login history yet.")
    else:
        # Summary metrics
        m1, m2, m3 = st.columns(3)
        total_logins  = len(logs)
        successful    = sum(1 for l in logs if l["success"] == 1)
        failed        = sum(1 for l in logs if l["success"] == 0)

        with m1:
            st.metric("Total Login Attempts", total_logins)
        with m2:
            st.metric("Successful", successful)
        with m3:
            st.metric("Failed", failed)

        st.markdown('<hr class="fbc-divider">', unsafe_allow_html=True)

        # Filter
        log_search = st.text_input("🔍 Filter by username",
                                   key="adm_log_search",
                                   placeholder="Type username…")
        show_only  = st.radio("Show", ["All", "Successful only", "Failed only"],
                              horizontal=True, key="adm_log_filter")

        filtered_logs = logs
        if log_search:
            filtered_logs = [l for l in filtered_logs
                             if log_search.lower() in l["username"].lower()]
        if show_only == "Successful only":
            filtered_logs = [l for l in filtered_logs if l["success"] == 1]
        elif show_only == "Failed only":
            filtered_logs = [l for l in filtered_logs if l["success"] == 0]

        st.markdown(f"**Showing {len(filtered_logs)} records**")

        df_log = pd.DataFrame([{
            "#":         l["id"],
            "Username":  l["username"],
            "Result":    "✅ Success" if l["success"] == 1 else "❌ Failed",
            "Timestamp": l["timestamp"],
        } for l in filtered_logs])

        st.dataframe(df_log, use_container_width=True, hide_index=True,
                     column_config={
                         "Result": st.column_config.TextColumn(width="small"),
                         "Timestamp": st.column_config.TextColumn(width="medium"),
                     })

        # Export
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download as CSV",
            data=csv,
            file_name=f"fbc_login_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="adm_download_log",
        )


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fbc-footer">
    <b>FBC Securities</b> · System Administration Panel ·
    Changes take effect immediately and are permanent.
</div>
""", unsafe_allow_html=True)
