import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _safe_dict_to_series(d):
    if not isinstance(d, dict) or len(d) == 0:
        return pd.Series(dtype=float)
    items = sorted(d.items(), key=lambda x: x[0])
    years = [i[0] for i in items]
    vals = [float(i[1]) for i in items]
    return pd.Series(vals, index=years, dtype=float)

def _volatility(series):
    if len(series) < 2:
        return 0.0
    return float(series.pct_change().dropna().std())

def _bounded_score(value, low_good, high_bad):
    if value <= low_good:
        return 100.0
    if value >= high_bad:
        return 0.0
    return float(100.0 * (high_bad - value) / (high_bad - low_good))


# ---------------------------------------------------------
# 1. FINANCIAL QUALITY ANALYSIS
# ---------------------------------------------------------
def compute_financial_diagnostics():

    rev_hist = st.session_state.get("dcf_rev_forecast", {})
    profit_hist = st.session_state.get("dcf_profit_all", {})
    ebitda_hist = st.session_state.get("dcf_ebitda_all", {})

    rev = _safe_dict_to_series(rev_hist)
    profit = _safe_dict_to_series(profit_hist)
    ebitda = _safe_dict_to_series(ebitda_hist)

    # Revenue stability
    rev_vol = _volatility(rev)
    score_rev_stability = _bounded_score(rev_vol, 0.02, 0.25)

    # EBITDA margin
    if len(rev) > 0 and len(ebitda) > 0:
        idx = rev.index.intersection(ebitda.index)
        m = (ebitda[idx] / rev[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        m = pd.Series(dtype=float)

    if len(m) == 0:
        avg_margin = None
        margin_vol = 0
        score_margin = 50
    else:
        avg_margin = float(m.mean())
        margin_vol = float(m.std())
        score_margin = (
            min(max(avg_margin * 400, 0), 100) * 0.6
            + _bounded_score(margin_vol, 0.01, 0.20) * 0.4
        )

    # Leverage
    net_debt = float(st.session_state.get("net_debt", 0.0))
    equity = float(st.session_state.get("book_equity", 0.0))

    if equity <= 0:
        leverage_ratio = 0
        score_leverage = 40
    else:
        leverage_ratio = abs(net_debt) / (abs(net_debt) + equity)
        score_leverage = _bounded_score(leverage_ratio, 0.15, 0.75)

    # FCFF stability
    fcff_arr = st.session_state.get("dcf_fcff_array", [])
    fcff = pd.Series([float(x) for x in fcff_arr]) if fcff_arr else pd.Series(dtype=float)

    if len(fcff) == 0:
        score_fcff = 50
        fcff_comment = "FCFF data unavailable."
    else:
        neg_frac = float((fcff < 0).sum()) / len(fcff)
        score_fcff = _bounded_score(neg_frac, 0.0, 0.6)
        fcff_comment = (
            "Several negative FCFF years."
            if neg_frac > 0.3
            else "Mostly positive FCFF."
        )

    overall_score = float(
        0.30 * score_rev_stability
        + 0.25 * score_margin
        + 0.25 * score_leverage
        + 0.20 * score_fcff
    )

    return {
        "overall_score": overall_score,
        "rev_vol": rev_vol,
        "avg_margin": avg_margin,
        "margin_vol": margin_vol,
        "leverage_ratio": leverage_ratio,
        "score_rev_stability": score_rev_stability,
        "score_margin": score_margin,
        "score_leverage": score_leverage,
        "score_fcff": score_fcff,
        "fcff_comment": fcff_comment,
    }


# ---------------------------------------------------------
# 2. MODEL RELIABILITY SCORING
# ---------------------------------------------------------
def compute_model_reliability():

    out = {}

    # -------- DCF score --------
    wacc = float(st.session_state.get("wacc", 0.0))
    g_term = float(st.session_state.get("dcf_terminal_g_pct", 0.0)) / 100
    pv_term = st.session_state.get("dcf_pv_terminal", None)
    pv_fcff = st.session_state.get("dcf_pv_fcff_sum", None)
    fcff = st.session_state.get("dcf_fcff_array", [])

    dcf_score = 60

    if wacc <= g_term:
        dcf_score -= 20
    else:
        dcf_score += 5

    if pv_term and pv_fcff:
        share = pv_term / (pv_term + pv_fcff)
        if share > 0.85:
            dcf_score -= 15
        elif share < 0.50:
            dcf_score += 5

    if fcff:
        fc = np.array(fcff)
        if (fc < 0).sum() > len(fc) / 2:
            dcf_score -= 20
        elif (fc < 0).sum() == 0:
            dcf_score += 5

    out["DCF"] = float(np.clip(dcf_score, 0, 100))

    # -------- DDM score --------
    divs = st.session_state.get("ddm_dividends", {})
    divs_series = _safe_dict_to_series(divs)
    Re = st.session_state.get("ddm_Re", 0)
    g = st.session_state.get("ddm_g", 0)
    eq_ddm = st.session_state.get("equity_value_ddm", None)

    ddm_score = 40
    if len(divs_series) > 0:
        zeros = float((divs_series <= 0).sum()) / len(divs_series)
        if zeros > 0.5:
            ddm_score -= 20
        else:
            ddm_score += 10

        g_vol = _volatility(divs_series)
        ddm_score += 10 if g_vol < 0.10 else -5

    if Re <= g:
        ddm_score -= 20
    if eq_ddm is None:
        ddm_score -= 5

    out["DDM"] = float(np.clip(ddm_score, 0, 100))

    # -------- Multiples score --------
    comps = {
        "EV/EBITDA": st.session_state.get("comps_ev_list", []),
        "PBV": st.session_state.get("comps_pb_list", []),
        "P/E": st.session_state.get("comps_pe_list", []),
    }

    for k, arr in comps.items():
        arr = np.array(arr, dtype=float)
        arr = arr[(arr > 0) & (~np.isnan(arr))]
        if len(arr) < 2:
            out[k] = 40
            continue

        vol = float(arr.std() / arr.mean())
        score = 50
        if len(arr) >= 4:
            score += 10
        score += 15 if vol < 0.25 else -10
        out[k] = float(np.clip(score, 0, 100))

    return out


# ---------------------------------------------------------
# 3. RISK & IDEA LABEL
# ---------------------------------------------------------
def compute_risk_and_upside():
    diag = compute_financial_diagnostics()
    fin_score = diag["overall_score"]

    upside = st.session_state.get("upside", None)
    up_pct = None if upside is None else float(upside * 100)

    if fin_score >= 75:
        risk_label = "Low"
    elif fin_score >= 50:
        risk_label = "Moderate"
    else:
        risk_label = "High"

    if up_pct is None:
        idea = "Inconclusive"
    elif up_pct > 40 and fin_score >= 65:
        idea = "High-Conviction Value Idea"
    elif up_pct > 15 and fin_score >= 50:
        idea = "Attractive Opportunity"
    elif up_pct > 0:
        idea = "Modest Upside"
    else:
        idea = "Fully Priced or Overvalued"

    return {
        "financial_score": fin_score,
        "risk_label": risk_label,
        "upside_pct": up_pct,
        "idea_label": idea,
        "details": diag,
    }


# ---------------------------------------------------------
# 4. COMMENTARY (SHORT + LONG)
# ---------------------------------------------------------
def generate_commentary():
    risk = compute_risk_and_upside()
    model_scores = compute_model_reliability()

    fs = risk["financial_score"]
    risk_label = risk["risk_label"]
    up = risk["upside_pct"]
    idea = risk["idea_label"]

    intr = st.session_state.get("intrinsic_value", None)
    price = st.session_state.get("current_price", None)
    w_equity = st.session_state.get("weighted_equity", None)

    # ---------------- SHORT SUMMARY ----------------
    short = f"{idea}. Risk is **{risk_label.lower()}**, financial score **{fs:.0f}/100**."

    if intr and price:
        short += f" Intrinsic value **{intr:.3f}**, market price **{price:.3f}**, implying **{up:.0f}%**."

    # ---------------- LONG FORM COMMENTARY ----------------
    parts = []

    parts.append(
        f"The company exhibits a **{risk_label.lower()} risk profile**, with a "
        f"financial quality score of **{fs:.0f}/100**, reflecting revenue behaviour, margins, "
        f"leverage and FCFF consistency."
    )

    if intr and price:
        parts.append(
            f"The blended intrinsic value per share is estimated at **{intr:.3f}**, "
            f"versus a market price of **{price:.3f}**, indicating approximately "
            f"**{up:.0f}%** {'upside' if up>=0 else 'downside'}."
        )

    if w_equity:
        parts.append(
            f"Total weighted equity value from all models is approximately "
            f"**{w_equity:,.0f} USD**."
        )

    # Model reliability commentary
    best = max(model_scores, key=model_scores.get)
    parts.append(
        "Model reliability scores (0–100): "
        + ", ".join([f"{k}: {v:.0f}" for k, v in model_scores.items()])
        + f". The **{best}** valuation appears most reliable at present."
    )

    # Financial diagnostics details
    d = risk["details"]
    if d["avg_margin"] is not None:
        parts.append(
            f"EBITDA margins average **{d['avg_margin']*100:.1f}%** with volatility "
            f"**{d['margin_vol']*100:.1f}%**."
        )
    parts.append(d["fcff_comment"])

    long = " ".join(parts)

    return {
        "short": short,
        "long": long,
        "model_scores": model_scores,
        "risk": risk,
    }
