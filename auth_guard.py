"""
auth_guard.py — Authentication guard (login-free version).
All pages pass through automatically; no login required.
"""
import streamlit as st


def require_auth(signin_key: str = "goto_signin") -> None:
    """No-op auth guard — always passes through."""
    st.session_state["authenticated"] = True
    if "user" not in st.session_state or not st.session_state.get("user"):
        st.session_state["user"] = {"username": "analyst", "role": "analyst", "full_name": "Analyst"}
