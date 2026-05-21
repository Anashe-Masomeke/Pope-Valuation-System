"""
auth.py  —  FBC Valuation System  ·  Authentication & Database Module
SQLite backend — zero server setup required.

Database file : fbc_users.db  (auto-created on first run)
Tables
  users           — credentials + profile
  login_log       — audit trail
  projects        — one row per valuation project per user
  project_data    — key-value store for all non-file session_state per project
  project_files   — raw file bytes per project (Excel, FX, beta, country params)
                    SEPARATE table so uploaded files NEVER bleed across projects
"""

import sqlite3
import hashlib
import os
import json
import base64
import re as _re
import datetime as _dt
from datetime import datetime

DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "fbc_users.db")
)


# ══════════════════════════════════════════════════════════════════
# 1. DATABASE INIT
# ══════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            username            TEXT    UNIQUE NOT NULL,
            password_hash       TEXT    NOT NULL,
            full_name           TEXT    DEFAULT '',
            email               TEXT    DEFAULT '',
            role                TEXT    DEFAULT 'analyst',
            security_question   TEXT    DEFAULT '',
            security_answer     TEXT    DEFAULT '',
            is_active           INTEGER DEFAULT 1,
            created_at          TEXT    DEFAULT (datetime('now'))
        )
    """)
    for col, defn in [
        ("email",             "TEXT DEFAULT ''"),
        ("security_question", "TEXT DEFAULT ''"),
        ("security_answer",   "TEXT DEFAULT ''"),
    ]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS login_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT,
            success   INTEGER,
            timestamp TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    NOT NULL,
            company_name TEXT    NOT NULL,
            ticker       TEXT    DEFAULT '',
            sector       TEXT    DEFAULT '',
            description  TEXT    DEFAULT '',
            status       TEXT    DEFAULT 'In Progress',
            created_at   TEXT    DEFAULT (datetime('now')),
            updated_at   TEXT    DEFAULT (datetime('now')),
            UNIQUE(username, company_name)
        )
    """)

    # Numeric / string / list data — everything except raw file bytes
    c.execute("""
        CREATE TABLE IF NOT EXISTS project_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            data_key    TEXT    NOT NULL,
            value_json  TEXT    NOT NULL,
            updated_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(project_id, data_key)
        )
    """)

    # Raw file bytes — one row per (project, file_key).
    # Keeping this separate ensures switching projects atomically swaps
    # ALL file bytes without touching any numeric/string data.
    c.execute("""
        CREATE TABLE IF NOT EXISTS project_files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            file_key    TEXT    NOT NULL,
            file_name   TEXT    NOT NULL DEFAULT '',
            file_data   BLOB    NOT NULL,
            updated_at  TEXT    DEFAULT (datetime('now')),
            UNIQUE(project_id, file_key)
        )
    """)

    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO users
                (username, password_hash, full_name, email, role,
                 security_question, security_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("admin", _hash("admin123"), "System Administrator",
              "admin@fbc.co.zw", "admin",
              "What is the company name?", _hash("fbc")))

    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════
def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════
# 3. AUTHENTICATION
# ══════════════════════════════════════════════════════════════════
def authenticate(username: str, password: str):
    conn = _connect()
    row = conn.execute("""
        SELECT id, username, full_name, email, role, is_active, security_question
        FROM   users WHERE username = ? AND password_hash = ? AND is_active = 1
    """, (username.strip(), _hash(password))).fetchone()
    success = row is not None
    conn.execute("INSERT INTO login_log (username, success) VALUES (?, ?)",
                 (username.strip(), 1 if success else 0))
    conn.commit()
    conn.close()
    return dict(row) if success else None


# ══════════════════════════════════════════════════════════════════
# 4. REGISTRATION
# ══════════════════════════════════════════════════════════════════
def register_user(username: str, password: str, full_name: str = "",
                  email: str = "", security_question: str = "",
                  security_answer: str = "", role: str = "analyst"):
    if len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not security_question or not security_answer:
        return False, "Please set a security question and answer for password recovery."
    try:
        conn = _connect()
        conn.execute("""
            INSERT INTO users
                (username, password_hash, full_name, email, role,
                 security_question, security_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.strip(), _hash(password), full_name.strip(),
              email.strip(), role, security_question.strip(), _hash(security_answer)))
        conn.commit()
        conn.close()
        return True, f"Account created! You can now sign in as '{username.strip()}'."
    except sqlite3.IntegrityError:
        return False, f"Username '{username.strip()}' is already taken."


# ══════════════════════════════════════════════════════════════════
# 5. PASSWORD RESET
# ══════════════════════════════════════════════════════════════════
def get_security_question(username: str):
    conn = _connect()
    row = conn.execute(
        "SELECT security_question FROM users WHERE username = ? AND is_active = 1",
        (username.strip(),)).fetchone()
    conn.close()
    return row["security_question"] if row else None

def verify_security_answer(username: str, answer: str) -> bool:
    conn = _connect()
    row = conn.execute(
        "SELECT security_answer FROM users WHERE username = ? AND is_active = 1",
        (username.strip(),)).fetchone()
    conn.close()
    return bool(row and row["security_answer"] == _hash(answer))

def reset_password(username: str, new_password: str):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    conn = _connect()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                 (_hash(new_password), username.strip()))
    conn.commit()
    conn.close()
    return True, "Password reset successfully. You can now sign in."


# ══════════════════════════════════════════════════════════════════
# 6. USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════
def change_password(username: str, new_password: str):
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    conn = _connect()
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                 (_hash(new_password), username))
    conn.commit()
    conn.close()
    return True, "Password updated."

def list_users():
    conn = _connect()
    rows = conn.execute(
        "SELECT id, username, full_name, email, role, is_active, created_at "
        "FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def deactivate_user(username: str):
    conn = _connect()
    conn.execute("UPDATE users SET is_active = 0 WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return True, f"User '{username}' deactivated."

def get_login_history(limit: int = 50):
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM login_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# 7. PROJECT MANAGEMENT
# ══════════════════════════════════════════════════════════════════
def create_project(username: str, company_name: str, ticker: str = "",
                   sector: str = "", description: str = ""):
    if not company_name.strip():
        return False, "Company name cannot be empty.", None
    try:
        conn = _connect()
        cur = conn.execute("""
            INSERT INTO projects (username, company_name, ticker, sector, description)
            VALUES (?, ?, ?, ?, ?)
        """, (username, company_name.strip(), ticker.strip(),
              sector.strip(), description.strip()))
        pid = cur.lastrowid
        conn.commit()
        conn.close()
        return True, f"Project '{company_name.strip()}' created.", pid
    except sqlite3.IntegrityError:
        conn.close()
        return False, f"You already have a project called '{company_name.strip()}'.", None

def list_projects(username: str):
    conn = _connect()
    rows = conn.execute("""
        SELECT id, company_name, ticker, sector, description, status,
               created_at, updated_at
        FROM   projects WHERE username = ? ORDER BY updated_at DESC
    """, (username,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_project(project_id: int):
    conn = _connect()
    row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def update_project_meta(project_id: int, company_name=None, ticker=None,
                        sector=None, description=None, status=None):
    conn = _connect()
    proj = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if not proj:
        conn.close()
        return False, "Project not found."
    cn  = company_name.strip() if company_name is not None else proj["company_name"]
    tk  = ticker.strip()       if ticker       is not None else proj["ticker"]
    sec = sector.strip()       if sector       is not None else proj["sector"]
    des = description.strip()  if description  is not None else proj["description"]
    st  = status               if status       is not None else proj["status"]
    conn.execute("""
        UPDATE projects SET company_name=?, ticker=?, sector=?,
               description=?, status=?, updated_at=? WHERE id=?
    """, (cn, tk, sec, des, st, _now(), project_id))
    conn.commit()
    conn.close()
    return True, "Project updated."

def delete_project(project_id: int):
    conn = _connect()
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()
    return True, "Project deleted."


# ══════════════════════════════════════════════════════════════════
# 8. FILE MANAGEMENT  (project_files table)
# ══════════════════════════════════════════════════════════════════
#
# Every uploaded Excel / FX / beta / country-params file is stored here,
# keyed by (project_id, file_key).  This physical separation guarantees
# that opening Project B can never expose Project A's files.
#
FILE_KEY_DCF_MAIN     = "dcf_file"
FILE_KEY_DCF_FX       = "dcf_fx"
FILE_KEY_DCF_COUNTRY  = "dcf_country_params"
FILE_KEY_DCF_BETA     = "dcf_beta"
FILE_KEY_BANK_MAIN    = "bank_file"
FILE_KEY_BANK_FX      = "bank_fx"
FILE_KEY_BANK_COUNTRY = "bank_country_params"
FILE_KEY_BANK_BETA    = "bank_beta"

# Logical file_key → (session bytes key,  session name key)
FILE_KEY_TO_SESSION = {
    FILE_KEY_DCF_MAIN:     ("dcf_file_bytes",           "dcf_file_name"),
    FILE_KEY_DCF_FX:       ("dcf_fx_bytes",              "dcf_fx_name"),
    FILE_KEY_DCF_COUNTRY:  ("dcf_country_params_bytes",  "dcf_country_params_name"),
    FILE_KEY_DCF_BETA:     ("dcf_beta_file_bytes",       "dcf_beta_file_name"),
    FILE_KEY_BANK_MAIN:    ("bank_file_bytes",           "bank_file_name"),
    FILE_KEY_BANK_FX:      ("bank_fx_bytes",             "bank_fx_name"),
    FILE_KEY_BANK_COUNTRY: ("bank_country_params_bytes", "bank_country_params_name"),
    FILE_KEY_BANK_BETA:    ("bank_beta_file_bytes",      "bank_beta_file_name"),
}

_ALL_FILE_BYTES_KEYS = {bk for bk, _ in FILE_KEY_TO_SESSION.values()}
_ALL_FILE_NAME_KEYS  = {nk for _, nk in FILE_KEY_TO_SESSION.values()}


def save_project_file(project_id: int, file_key: str,
                      file_bytes: bytes, file_name: str = "") -> bool:
    """Upsert raw file bytes for a project under a logical file_key."""
    if not file_bytes:
        return False
    conn = _connect()
    conn.execute("""
        INSERT INTO project_files (project_id, file_key, file_name, file_data, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(project_id, file_key)
        DO UPDATE SET file_name=excluded.file_name,
                      file_data=excluded.file_data,
                      updated_at=excluded.updated_at
    """, (project_id, file_key, file_name or "", file_bytes, _now()))
    conn.commit()
    conn.close()
    return True


def load_project_files(project_id: int) -> dict:
    """Load all files for a project → {file_key: {"bytes": …, "name": …}}"""
    conn = _connect()
    rows = conn.execute(
        "SELECT file_key, file_name, file_data FROM project_files WHERE project_id = ?",
        (project_id,)).fetchall()
    conn.close()
    return {r["file_key"]: {"bytes": bytes(r["file_data"]), "name": r["file_name"]}
            for r in rows}


def _save_all_files_from_session(project_id: int, session_state: dict) -> int:
    """
    Scan session_state for every known file-bytes key and save to project_files.
    Banking is special: its file bytes live inside session_state["bank"]["file_bytes"].
    """
    saved = 0
    bytes_to_file_key = {bk: fk for fk, (bk, _) in FILE_KEY_TO_SESSION.items()}
    name_map          = {bk: nk for _,  (bk, nk) in FILE_KEY_TO_SESSION.items()}

    # 1. BANKING nested dict
    bank_dict = session_state.get("bank") or {}
    if isinstance(bank_dict, dict):
        fb = bank_dict.get("file_bytes")
        fn = bank_dict.get("file_name") or ""
        if fb and isinstance(fb, (bytes, bytearray)):
            save_project_file(project_id, FILE_KEY_BANK_MAIN, bytes(fb), fn)
            saved += 1

    # 2. All flat bytes keys
    for bytes_key, file_key in bytes_to_file_key.items():
        if file_key == FILE_KEY_BANK_MAIN:
            continue  # handled above
        val = session_state.get(bytes_key)
        if val and isinstance(val, (bytes, bytearray)):
            fname = session_state.get(name_map[bytes_key]) or ""
            save_project_file(project_id, file_key, bytes(val), fname)
            saved += 1
    return saved


def _restore_files_to_session(project_id: int, session_state: dict) -> int:
    """
    Load project_files from DB → write bytes + name into session_state.
    BANKING bytes go into both session_state["bank"]["file_bytes"] AND
    the flat key so any page that reads either location works correctly.
    """
    files = load_project_files(project_id)
    restored = 0
    for file_key, info in files.items():
        if file_key not in FILE_KEY_TO_SESSION:
            continue
        bytes_key, name_key = FILE_KEY_TO_SESSION[file_key]
        fb, fn = info["bytes"], info["name"]

        if file_key == FILE_KEY_BANK_MAIN:
            if "bank" not in session_state or not isinstance(session_state.get("bank"), dict):
                session_state["bank"] = {}
            try:
                session_state["bank"]["file_bytes"] = fb
                session_state["bank"]["file_name"]  = fn
            except Exception:
                pass
            try:
                session_state[bytes_key] = fb
                session_state[name_key]  = fn
            except Exception:
                pass
        else:
            try:
                session_state[bytes_key] = fb
                session_state[name_key]  = fn
            except Exception:
                pass
        restored += 1
    return restored


def _clear_all_file_bytes(session_state: dict):
    """
    Zero out ALL file bytes/name keys in session_state.
    Must be called before restoring a different project.
    """
    for _, (bytes_key, name_key) in FILE_KEY_TO_SESSION.items():
        try:
            session_state[bytes_key] = None
        except Exception:
            pass
        try:
            session_state[name_key] = None
        except Exception:
            pass
    bank = session_state.get("bank")
    if isinstance(bank, dict):
        bank["file_bytes"] = None
        bank["file_name"]  = None


# ══════════════════════════════════════════════════════════════════
# 9. SERIALISATION
# ══════════════════════════════════════════════════════════════════
def _serialise(value):
    """Return (json_safe_value, True) or (None, False)."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return {"__type__": "date", "iso": value.isoformat()}, True
    if isinstance(value, (bytes, bytearray)):
        # Should be rare — bytes go to project_files.  Encode as b64 fallback.
        return {"__type__": "bytes", "b64": base64.b64encode(value).decode()}, True
    if isinstance(value, dict) and value and all(isinstance(k, int) for k in value):
        inner = {}
        for k, v in value.items():
            sv, ok = _serialise(v)
            if not ok:
                return None, False
            inner[str(k)] = sv
        return {"__type__": "int_keyed_dict", "data": inner}, True
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            sv, ok = _serialise(v)
            if not ok:
                return None, False
            out[k] = sv
        return out, True
    if isinstance(value, list):
        out = []
        for v in value:
            sv, ok = _serialise(v)
            if not ok:
                return None, False
            out.append(sv)
        return out, True
    try:
        json.dumps(value)
        return value, True
    except (TypeError, ValueError):
        return None, False


def _deserialise(value):
    """Reverse _serialise tags."""
    if isinstance(value, dict):
        t = value.get("__type__")
        if t == "date":
            try:
                return _dt.date.fromisoformat(value["iso"])
            except Exception:
                return value
        if t == "bytes":
            try:
                return base64.b64decode(value["b64"])
            except Exception:
                return value
        if t == "int_keyed_dict":
            try:
                return {int(k): _deserialise(v) for k, v in value["data"].items()}
            except Exception:
                return value
        return {k: _deserialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deserialise(v) for v in value]
    return value


# ══════════════════════════════════════════════════════════════════
# 10. WHAT GETS SAVED vs SKIPPED
# ══════════════════════════════════════════════════════════════════
_EXCLUDED_KEYS = {
    # Auth / identity
    "authenticated", "user", "auth_mode", "reset_step", "reset_username",
    "active_project_id", "active_project_name", "_autosave_last_ts",
    # Projects page UI
    "proj_signout", "btn_create_proj",
    "nc_company", "nc_ticker", "nc_sector", "nc_desc",
    # DataFrames — rebuilt from file bytes; too large / not serialisable
    "dcf_is_df", "dcf_bs_df", "dcf_cf_df",
    "dcf_is_base", "dcf_bs_base", "dcf_cf_base",
    "forecast_is_df", "df_dcf_export",
    "bank_is_df", "bank_bs_df", "bank_soce_df",
    "bank_is_base", "bank_bs_base", "bank_soce_base",
}

_WIDGET_SUFFIXES = (
    "_input", "_ui", "_radio", "_select", "_uploader",
    "_widget", "_checkbox", "_editor", "_picker", "_multiselect",
)

_EXACT_WIDGET_KEYS = {
    "ddm_de_ratio", "ddm_erp", "ddm_g_end", "ddm_g_start", "ddm_rf",
    "ddm_tax_rate", "ddm_unlevered_beta", "ddm_use_custom_params",
    "ddm_num_shares", "ddm_start_year_input", "ddm_end_year_input",
    "ddm_download_excel", "ddm_generate_excel",
    "bank_clear_btn", "bank_g_term_input", "bank_n_years_input",
    "bank_base_year_input", "bank_de_beta_input", "bank_tax_beta_input",
    "bank_beta_u_input_box", "bank_disc_uniform_input",
    "bank_eps_uniform_input", "bank_yoy_uniform_input",
    "bank_mrp_pct_input", "bank_rf_pct_input", "bank_zim_avg_cod_pct_input",
    "net_debt_input", "book_equity_input", "target_company_input",
    "live_peer_limit", "num_comps_input", "auto_peer_count_input",
    "bs_jump_radio", "cf_jump_radio", "np_end_locked", "np_start_locked",
    "discount_factor_widget",
    "comp_timing_base_manual", "comp_timing_base_manual_no_dcf", "comp_timing_choice",
    "dl_forecast_is_xlsx", "dl_full_dcf_model_btn", "gen_full_dcf_excel_btn",
    "use_selected_peers_btn", "toggle_capex_expander_btn",
    "reset_capex_exclusions_btn", "apply_auto_beta_btn",
    "auto_apply_peers", "bank_apply_beta_u_btn", "bank_reset_beta_u_btn",
    "btn_comp_exp_0_debug_peer_search",
}

_WIDGET_PREFIX_PAT = _re.compile(
    r"^(FormSubmitter|uploaded_|_qsave_btn_|"
    r"open_|edit_|del_|clear_|confirm_|yes_|no_|"
    r"cancel_|ecn_|etk_|esec_|edes_|est_|li_|reg_|fp_|go_|back_)"
)


def _should_skip(key: str) -> bool:
    if key in _EXCLUDED_KEYS:
        return True
    if key in _ALL_FILE_BYTES_KEYS or key in _ALL_FILE_NAME_KEYS:
        return True  # goes to project_files instead
    if key in _EXACT_WIDGET_KEYS:
        return True
    if key.endswith(_WIDGET_SUFFIXES):
        return True
    if _WIDGET_PREFIX_PAT.match(key):
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# 11. SAVE PROJECT SESSION
# ══════════════════════════════════════════════════════════════════
def save_project_session(project_id: int, session_state: dict):
    """
    Save everything for a project:
    - Numeric / string / list / dict data  → project_data
    - Uploaded file bytes                  → project_files (per-project isolation)
    Returns (True, summary_message).
    """
    conn = _connect()
    now  = _now()
    saved = skipped = 0

    for key, value in session_state.items():
        if _should_skip(key):
            skipped += 1
            continue

        # "bank" dict: save it but strip out file_bytes/file_name first
        # (those go to project_files, not here)
        if key == "bank" and isinstance(value, dict):
            clean = {k: v for k, v in value.items()
                     if k not in ("file_bytes", "file_name")}
            serial_val, ok = _serialise(clean)
        else:
            serial_val, ok = _serialise(value)

        if not ok:
            skipped += 1
            continue

        try:
            json_val = json.dumps(serial_val)
        except Exception:
            skipped += 1
            continue

        conn.execute("""
            INSERT INTO project_data (project_id, data_key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, data_key)
            DO UPDATE SET value_json=excluded.value_json,
                          updated_at=excluded.updated_at
        """, (project_id, key, json_val, now))
        saved += 1

    conn.execute(
        "UPDATE projects SET updated_at=?, status='In Progress' WHERE id=?",
        (now, project_id))
    conn.commit()
    conn.close()

    # Save file bytes to project_files (separate connection / table)
    files_saved = _save_all_files_from_session(project_id, session_state)

    return True, (f"Saved {saved} inputs + {files_saved} file(s). "
                  f"({skipped} items skipped.)")


# ══════════════════════════════════════════════════════════════════
# 12. LOAD PROJECT SESSION  (data only — no file bytes)
# ══════════════════════════════════════════════════════════════════
_MASTER_TO_INPUT = {
    "dcf_rf_pct":                "dcf_rf_pct_input",
    "dcf_mrp_pct":               "dcf_mrp_pct_input",
    "dcf_tax_pct":               "dcf_tax_pct_input",
    "dcf_unlevered_beta":        "dcf_unlevered_beta_input",
    "dcf_terminal_g_pct":        "dcf_terminal_g_pct_input",
    "dcf_zim_avg_cost_debt_pct": "dcf_zim_avg_cost_debt_pct_input",
    "dcf_rd_manual_value":       "dcf_rd_manual_input",
    "dcf_forecast_years":        "dcf_forecast_years_input",
    "bank_rf_pct":               "bank_rf_pct_input",
    "bank_mrp_pct":              "bank_mrp_pct_input",
    "num_shares":                "num_shares_input",
    # Summary page uses its own dedicated keys (not shared with DDM)
    "summary_num_shares":        "summary_num_shares",
    "summary_current_price":     "summary_current_price",
}
_INT_MASTER_KEYS = {"dcf_forecast_years"}


def load_project_session(project_id: int) -> dict:
    """
    Load all saved non-file data for a project.
    Call switch_project() instead for a full project open — it also handles
    clearing stale files and loading the new project's files.
    """
    conn = _connect()
    rows = conn.execute(
        "SELECT data_key, value_json FROM project_data WHERE project_id = ?",
        (project_id,)).fetchall()
    conn.close()

    result = {}
    for row in rows:
        try:
            raw = json.loads(row["value_json"])
            result[row["data_key"]] = _deserialise(raw)
        except Exception:
            pass

    # Mirror master numeric keys → widget _input keys so widgets
    # show the saved value instead of their default on first render.
    for master, input_key in _MASTER_TO_INPUT.items():
        if master in result:
            val = result[master]
            if isinstance(val, (int, float)):
                casted = int(val) if master in _INT_MASTER_KEYS else float(val)
                result[master]    = casted
                result[input_key] = casted

    return result


# ══════════════════════════════════════════════════════════════════
# 13. SWITCH PROJECT  (the correct way to open a different project)
# ══════════════════════════════════════════════════════════════════
_PROTECTED_KEYS = {
    "authenticated", "user", "auth_mode",
    "reset_step", "reset_username",
    "active_project_id", "active_project_name",
}
_THIS_PAGE_SKIP = _re.compile(
    r"^(_qsave_btn_|open_|save_|edit_|del_|clear_|confirm_|yes_|no_|"
    r"cancel_|ecn_|etk_|esec_|edes_|est_|"
    r"FormSubmitter|uploaded_|proj_signout|btn_create_proj|nc_)"
)
# Keys that hold parsed state and must be cleared so the page re-parses
# the newly loaded file bytes from scratch
_STALE_PARSE_KEYS = {
    "dcf_is_df", "dcf_bs_df", "dcf_cf_df",
    "dcf_is_base", "dcf_bs_base", "dcf_cf_base",
    "dcf_mapping", "is_core_mapping",
    "dcf_init", "dcf_timing_init",
    "bank_is_df", "bank_bs_df", "bank_soce_df",
    "bank_is_base", "bank_bs_base", "bank_soce_base",
    "bank_init",
    "dcf_uploader_key",   # force the uploader widget to reset
}


def switch_project(new_project_id: int, new_project_name: str,
                   session_state: dict,
                   current_project_id: int = None):
    """
    Safely open a different project:
      1. Save the CURRENT project (data + files) so nothing is lost.
      2. Clear stale file bytes and parsed DataFrames from session.
      3. Load the NEW project's data into session.
      4. Load the NEW project's file bytes into session.
      5. Set active_project_id / active_project_name.

    Returns (data_keys_restored, files_restored).
    """
    # 1. Save current project first
    if current_project_id and current_project_id != new_project_id:
        save_project_session(current_project_id, dict(session_state))

    # 2. Clear stale files and parsed state
    _clear_all_file_bytes(session_state)
    for k in _STALE_PARSE_KEYS:
        session_state.pop(k, None)

    # 3. Load data (no files)
    saved_data = load_project_session(new_project_id)
    data_restored = 0
    for k, v in saved_data.items():
        if k in _PROTECTED_KEYS or _THIS_PAGE_SKIP.match(k):
            continue
        try:
            session_state[k] = v
            data_restored += 1
        except Exception:
            pass

    # 4. Load file bytes (isolated to this project)
    files_restored = _restore_files_to_session(new_project_id, session_state)

    # 5. Update active project markers
    session_state["active_project_id"]   = new_project_id
    session_state["active_project_name"] = new_project_name

    return data_restored, files_restored


# ══════════════════════════════════════════════════════════════════
# 14. MISC HELPERS
# ══════════════════════════════════════════════════════════════════
def get_project_data_summary(project_id: int) -> dict:
    conn = _connect()
    row = conn.execute("""
        SELECT COUNT(*) as cnt, MAX(updated_at) as last_saved
        FROM project_data WHERE project_id = ?
    """, (project_id,)).fetchone()
    frow = conn.execute(
        "SELECT COUNT(*) as fcnt FROM project_files WHERE project_id = ?",
        (project_id,)).fetchone()
    conn.close()
    return {
        "count":      row["cnt"],
        "files":      frow["fcnt"],
        "last_saved": row["last_saved"],
    }


def clear_project_data(project_id: int):
    conn = _connect()
    conn.execute("DELETE FROM project_data  WHERE project_id = ?", (project_id,))
    conn.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))
    conn.commit()
    conn.close()
    return True, "Project data and files cleared."


# ══════════════════════════════════════════════════════════════════
# 15. AUTOSAVE
# ══════════════════════════════════════════════════════════════════
def autosave_project(session_state, interval_seconds: int = 30) -> bool:
    """
    Call at the top of every model page to silently save the active project.
    Returns True if a save was performed.
    """
    import time as _time
    project_id = session_state.get("active_project_id")
    if not project_id:
        return False
    now_ts  = _time.time()
    last_ts = session_state.get("_autosave_last_ts", 0)
    if (now_ts - last_ts) < interval_seconds:
        return False
    try:
        save_project_session(project_id, dict(session_state))
        session_state["_autosave_last_ts"] = now_ts
        return True
    except Exception:
        return False



# ══════════════════════════════════════════════════════════════════
# ADMIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def admin_list_all_projects() -> list:
    """List ALL projects across ALL users — admin only."""
    conn = _connect()
    rows = conn.execute("""
        SELECT p.id, p.username, p.company_name, p.ticker, p.sector,
               p.status, p.created_at, p.updated_at,
               (SELECT COUNT(*) FROM project_data  pd WHERE pd.project_id = p.id) as data_count,
               (SELECT COUNT(*) FROM project_files pf WHERE pf.project_id = p.id) as file_count
        FROM   projects p
        ORDER  BY p.updated_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def admin_get_stats() -> dict:
    """Return high-level system statistics."""
    conn = _connect()
    total_users    = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    active_users   = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    total_projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    total_files    = conn.execute("SELECT COUNT(*) FROM project_files").fetchone()[0]
    logins_today   = conn.execute(
        "SELECT COUNT(*) FROM login_log WHERE date(timestamp)=date('now') AND success=1"
    ).fetchone()[0]
    failed_today   = conn.execute(
        "SELECT COUNT(*) FROM login_log WHERE date(timestamp)=date('now') AND success=0"
    ).fetchone()[0]
    conn.close()
    return {
        "total_users":    total_users,
        "active_users":   active_users,
        "total_projects": total_projects,
        "total_files":    total_files,
        "logins_today":   logins_today,
        "failed_today":   failed_today,
    }


def admin_get_user(username: str) -> dict | None:
    """Get full user profile for admin editing."""
    conn = _connect()
    row = conn.execute(
        "SELECT id, username, full_name, email, role, is_active, created_at "
        "FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def admin_update_user(username: str, full_name: str = None,
                      email: str = None, role: str = None,
                      is_active: int = None) -> tuple:
    """Update any user field — admin only."""
    conn = _connect()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."
    fn  = full_name.strip() if full_name  is not None else user["full_name"]
    em  = email.strip()     if email      is not None else user["email"]
    rl  = role              if role       is not None else user["role"]
    act = is_active         if is_active  is not None else user["is_active"]
    conn.execute(
        "UPDATE users SET full_name=?, email=?, role=?, is_active=? WHERE username=?",
        (fn, em, rl, act, username)
    )
    conn.commit()
    conn.close()
    return True, f"User '{username}' updated."


def admin_reset_password(username: str, new_password: str) -> tuple:
    """Admin force-reset any user's password."""
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    conn = _connect()
    conn.execute(
        "UPDATE users SET password_hash=? WHERE username=?",
        (_hash(new_password), username)
    )
    conn.commit()
    conn.close()
    return True, f"Password for '{username}' reset successfully."


def admin_delete_user(username: str) -> tuple:
    """Permanently delete a user and ALL their projects/data."""
    conn = _connect()
    # Delete all projects (cascade deletes project_data and project_files)
    conn.execute("DELETE FROM projects WHERE username=?", (username,))
    conn.execute("DELETE FROM users    WHERE username=?", (username,))
    conn.commit()
    conn.close()
    return True, f"User '{username}' and all their data deleted."


def admin_get_full_login_history(limit: int = 200) -> list:
    """Full login audit log — admin only."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM login_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def admin_delete_project(project_id: int) -> tuple:
    """Admin delete any project regardless of owner."""
    conn = _connect()
    proj = conn.execute("SELECT company_name, username FROM projects WHERE id=?",
                        (project_id,)).fetchone()
    if not proj:
        conn.close()
        return False, "Project not found."
    name = proj["company_name"]
    conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
    conn.commit()
    conn.close()
    return True, f"Project '{name}' deleted."


# ── Auto-init on import ───────────────────────────────────────────
init_db()
