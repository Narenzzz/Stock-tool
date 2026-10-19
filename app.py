import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==============================================================================
# 0. PAGE CONFIGURATION, PERSISTENCE FILE & NSE SECTOR BENCHMARKS
# ==============================================================================
st.set_page_config(
    page_title="Indian Equity Institutional Value, P/E-PEG & DCF Analyzer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

WATCHLIST_FILE = "saved_stock_analysis.json"

INDIAN_PRESETS = {
    "CGPOWER.NS": "CG Power and Industrial Solutions Ltd (Capital Goods / Electricals)",
    "TCS.NS": "Tata Consultancy Services (IT Services)",
    "INFY.NS": "Infosys Ltd (IT Services)",
    "HDFCBANK.NS": "HDFC Bank Ltd (Private Bank)",
    "ICICIBANK.NS": "ICICI Bank Ltd (Private Bank)",
    "BAJFINANCE.NS": "Bajaj Finance Ltd (NBFC)",
    "RELIANCE.NS": "Reliance Industries Ltd (Energy / Telecom / Retail)",
    "SUZLON.NS": "Suzlon Energy Ltd (Renewable Wind Energy)",
    "CUPID.NS": "Cupid Ltd (Healthcare / Diagnostics)",
    "ITC.NS": "ITC Ltd (FMCG / Consumer)",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd (FMCG)",
    "MARUTI.NS": "Maruti Suzuki India Ltd (Automobiles)",
    "EICHERMOT.NS": "Eicher Motors Ltd (2W / Commercial Vehicles)",
    "TVSMOTOR.NS": "TVS Motor Company Ltd (2W / EV)",
    "LT.NS": "Larsen & Toubro Ltd (Infrastructure / EPC)",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries (Pharma)",
    "TITAN.NS": "Titan Company Ltd (Consumer Discretionary)",
    "TDPS.NS": "TD Power Systems Ltd (Capital Goods / Generators)",
}

INDIA_SECTOR_PEERS = {
    "IT_TECH": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "BANKING_NBFC": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS"],
    "FMCG_CONSUMER": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "PHARMA_HEALTH": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "ENERGY_UTILITIES": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "SUZLON.NS"],
    "MANUFACTURING_INDUSTRIAL": ["LT.NS", "CGPOWER.NS", "MARUTI.NS", "EICHERMOT.NS", "SIEMENS.NS"],
}

NSE_SECTOR_BENCHMARKS = {
    "IT_TECH": {
        "rev_growth_med": 9.5,
        "ebitda_margin_med": 23.5,
        "net_margin_med": 16.5,
        "roe_med": 27.0,
        "roce_med": 31.0,
        "pe_med": 26.5,
    },
    "BANKING_NBFC": {
        "rev_growth_med": 14.5,
        "ebitda_margin_med": None,
        "net_margin_med": 21.0,
        "roe_med": 15.8,
        "roce_med": None,
        "pe_med": 16.5,
    },
    "FMCG_CONSUMER": {
        "rev_growth_med": 8.5,
        "ebitda_margin_med": 24.0,
        "net_margin_med": 16.0,
        "roe_med": 32.0,
        "roce_med": 38.0,
        "pe_med": 48.0,
    },
    "PHARMA_HEALTH": {
        "rev_growth_med": 11.5,
        "ebitda_margin_med": 24.5,
        "net_margin_med": 15.5,
        "roe_med": 16.5,
        "roce_med": 19.5,
        "pe_med": 32.0,
    },
    "ENERGY_UTILITIES": {
        "rev_growth_med": 11.0,
        "ebitda_margin_med": 22.0,
        "net_margin_med": 11.5,
        "roe_med": 14.5,
        "roce_med": 14.0,
        "pe_med": 19.0,
    },
    "MANUFACTURING_INDUSTRIAL": {
        "rev_growth_med": 12.5,
        "ebitda_margin_med": 14.5,
        "net_margin_med": 9.0,
        "roe_med": 16.5,
        "roce_med": 18.5,
        "pe_med": 28.0,
    },
}


@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer() -> Optional[SentimentIntensityAnalyzer]:
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


# ==============================================================================
# 1. PERSISTENT WATCHLIST LEDGER ENGINE (TAB 2 BACKEND)
# ==============================================================================
def load_watchlist() -> list:
    """Loads saved stock analyses into st.session_state and syncs with local JSON."""
    if "saved_analyses" not in st.session_state:
        if os.path.exists(WATCHLIST_FILE):
            try:
                with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                    st.session_state.saved_analyses = json.load(f)
            except Exception:
                st.session_state.saved_analyses = []
        else:
            st.session_state.saved_analyses = []
    return st.session_state.saved_analyses


def save_watchlist_to_disk(data: list):
    """Persists watchlist state to local JSON file."""
    st.session_state.saved_analyses = data
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        st.sidebar.warning(f"Session memory active (Disk warning: {e})")


def upsert_stock_analysis(record: dict, replace_existing: bool = True) -> str:
    """Saves or replaces past stock analysis snapshot."""
    data = load_watchlist()
    ticker = record.get("Ticker")

    if replace_existing:
        for idx, item in enumerate(data):
            if item.get("Ticker") == ticker:
                data[idx] = record
                save_watchlist_to_disk(data)
                return f"🔄 Replaced previous saved snapshot for **{ticker}** ({record['Date']})."

    data.insert(0, record)
    save_watchlist_to_disk(data)
    return f"✅ Saved **{ticker}** to Comparison Watchlist ({record['Date']})."


def reset_entire_watchlist():
    """Wipes all saved records."""
    st.session_state.saved_analyses = []
    if os.path.exists(WATCHLIST_FILE):
        try:
            os.remove(WATCHLIST_FILE)
        except Exception:
            pass


# ==============================================================================
# 2. REVERSE DCF IMPLIED GROWTH & STAGGERED BUY-ZONE CALCULATOR
# ==============================================================================
def calculate_reverse_dcf_growth(
    cmp: float,
    base_cf_per_share: float,
    wacc: float = 0.12,
    terminal_g: float = 0.04,
    stage1_years: int = 10,
    net_cash_per_share: float = 0.0,
) -> float:
    """
    Bisection search solver calculating the 10-year Stage-1 annual growth rate (g1 %)
    implied by today's Current Market Price (CMP).
    """
    if cmp <= 0 or base_cf_per_share <= 0 or wacc <= terminal_g:
        return 0.0

    target_ev_per_share = max(cmp - net_cash_per_share, cmp * 0.10)
    low, high = -0.30, 1.20  # Range -30% to +120% CAGR

    for _ in range(70):
        mid_g = (low + high) / 2.0
        pv_cfs = 0.0
        cf = base_cf_per_share
        for t in range(1, stage1_years + 1):
            cf = cf * (1.0 + mid_g)
            pv_cfs += cf / ((1.0 + wacc) ** t)

        terminal_val = (cf * (1.0 + terminal_g)) / (wacc - terminal_g)
        pv_terminal = terminal_val / ((1.0 + wacc) ** stage1_years)
        model_val = pv_cfs + pv_terminal

        if model_val < target_ev_per_share:
            low = mid_g
        else:
            high = mid_g

    return round(((low + high) / 2.0) * 100.0, 2)

# ==============================================================================
# 2B. AUTOMATED INSTITUTIONAL EXIT & POSITION SIZING ENGINE
# ==============================================================================
def evaluate_exit_and_trim_strategy(
    cmp: float,
    moat_grade: str,
    roce_pct: Optional[float],
    wacc_pct: float,
    fcf_ni_pct: float,
    de_ratio: Optional[float],
    qtr_status: str,
    implied_g1_pct: float,
    model_g1_pct: float,
    peg_ratio: Optional[float],
    rev_yoy_pct: Optional[float],
    blended_mos_pct: Optional[float],
    roce_applicable: bool = True,
    de_applicable: bool = True,
    current_portfolio_weight_pct: float = 5.0,
    max_single_stock_cap_pct: float = 15.0,
    replacement_irr_pct: float = 15.0,
) -> Dict[str, Any]:
    """
    Evaluates the 4 Institutional Selling Triggers:
      1. Structural Thesis Invalidation (Kill-Switch Breach) -> 100% Full Exit
      2. Valuation Euphoria (Reverse DCF / PEG Bubble)       -> 30% to 50% Staggered Trim
      3. Portfolio Concentration Breach (>15-20% Cap)        -> Rebalance Excess Weight
      4. Post-Tax Opportunity Cost (12.5% LTCG Hurdle)       -> Capital Rotation Swap
    """
    triggers_fired: List[str] = []
    tranche_plan: List[str] = []

    # --- TRIGGER 1: STRUCTURAL THESIS INVALIDATION (100% EXIT) ---
    roce_breach = (
        roce_applicable
        and roce_pct is not None
        and roce_pct < (wacc_pct - 1.0)
    )
    fcf_breach = fcf_ni_pct != 0.0 and fcf_ni_pct < 45.0
    debt_spike = de_applicable and de_ratio is not None and de_ratio > 1.50
    severe_qtr_decay = "CONTRACTION" in qtr_status and moat_grade in ["C", "D"]

    thesis_broken = (roce_breach and fcf_breach) or debt_spike or severe_qtr_decay or moat_grade == "D"

    # --- TRIGGER 2: EXTREME VALUATION EUPHORIA (30%-50% TRIM) ---
    growth_expectation_gap = implied_g1_pct - model_g1_pct
    valuation_bubble = (
        implied_g1_pct >= 35.0
        or growth_expectation_gap >= 18.0
        or (peg_ratio is not None and peg_ratio >= 3.5 and (rev_yoy_pct is None or rev_yoy_pct < 12.0))
    )

    # --- TRIGGER 3: PORTFOLIO CONCENTRATION CEILING ---
    concentration_excess_pct = max(0.0, round(current_portfolio_weight_pct - max_single_stock_cap_pct, 2))
    concentration_breach = concentration_excess_pct > 0.0

    # --- TRIGGER 4: RELATIVE OPPORTUNITY COST (AFTER 12.5% LTCG TAX FRICTION) ---
    # Estimate prospective 5-Yr stock IRR = Model Growth + annualized MoS mean-reversion
    mos_clamped = max(-50.0, min(50.0, blended_mos_pct or 0.0))
    est_holding_irr = round(model_g1_pct + (mos_clamped * 0.18), 2)
    # Require >= 8.0% IRR upgrade spread to justify paying 12.5% LTCG tax + reinvestment risk
    irr_spread = round(replacement_irr_pct - est_holding_irr, 2)
    opportunity_cost_breach = irr_spread >= 8.0 and moat_grade not in ["A+"]

    # --- DECISION HIERARCHY ---
    if thesis_broken:
        exit_verdict = "🚨 TRIGGER 1: STRUCTURAL THESIS BREACH — FULL EXIT (SELL 100%)"
        exit_color = "#dc2626"  # Crimson Red
        if roce_breach:
            triggers_fired.append(f"Capital Efficiency Failure: ROCE ({roce_pct:.1f}%) dropped below WACC ({wacc_pct:.1f}%).")
        if fcf_breach:
            triggers_fired.append(f"Cash Conversion Breakdown: FCF/Net Income ({fcf_ni_pct:.1f}%) is below the 45% safety floor.")
        if debt_spike:
            triggers_fired.append(f"Balance Sheet Stress: Debt-to-Equity ({de_ratio:.2f}x) crossed the 1.50x danger threshold.")
        if severe_qtr_decay or moat_grade == "D":
            triggers_fired.append(f"Moat Deterioration: Business Moat Grade fell to '{moat_grade}' alongside quarterly earnings pressure.")

        tranche_plan = [
            "Tranche 1 (Immediate): Liquidate 50%–100% of position at market price—do not wait for a bounce.",
            "Tranche 2 (Within 3–5 Days): Exit any remaining shares and reallocate cash to A+/A watchlisted compounders.",
        ]

    elif valuation_bubble:
        exit_verdict = "✂️ TRIGGER 2: VALUATION EUPHORIA — TRIM 30% TO 50% (LOCK MULTI-YEAR GAINS)"
        exit_color = "#f59e0b"  # Amber
        triggers_fired.append(
            f"Reverse DCF Euphoria: CMP (₹{cmp:,.1f}) prices in {implied_g1_pct:.1f}%/yr growth for 10 years "
            f"(+{growth_expectation_gap:.1f}% above realistic model g₁ of {model_g1_pct:.1f}%)."
        )
        if peg_ratio is not None and peg_ratio >= 3.0:
            triggers_fired.append(f"Multiple Stretch: PEG ratio is {peg_ratio:.2f}x, pulling forward 3–5 years of future returns.")

        tranche_plan = [
            f"Tranche 1 (Sell 1/3rd Now @ ~₹{cmp:,.0f}): Take initial capital / froth off the table while keeping core compounder exposure.",
            "Tranche 2 (Sell 1/3rd on QoQ Deceleration): Trim second slice only if quarterly OPM or YoY revenue growth slows.",
            "Tranche 3 (Core Runner - Hold 1/3rd+): Never sell the final core of an A+/A business unless Trigger 1 (Kill-Switch) fires.",
        ]

    elif concentration_breach:
        exit_verdict = f"⚖️ TRIGGER 3: CONCENTRATION CAP BREACH — REBALANCE ({concentration_excess_pct:.1f}% EXCESS)"
        exit_color = "#a855f7"  # Purple
        triggers_fired.append(
            f"Single-Stock Weight ({current_portfolio_weight_pct:.1f}%) exceeds your {max_single_stock_cap_pct:.1f}% risk parity ceiling."
        )
        trim_share_pct = round((concentration_excess_pct / current_portfolio_weight_pct) * 100.0, 1)
        tranche_plan = [
            f"Portfolio Rebalance Action: Trim ~{trim_share_pct:.1f}% of your current shares in this stock to bring weight back down to {max_single_stock_cap_pct - 2.0:.1f}%.",
            "Core Retention: Keep the remaining position untouched to continue compounding.",
        ]

    elif opportunity_cost_breach:
        exit_verdict = f"🔄 TRIGGER 4: OPPORTUNITY COST DRAG — ROTATE CAPITAL (+{irr_spread:.1f}% IRR UPGRADE)"
        exit_color = "#0ea5e9"  # Sky Blue
        triggers_fired.append(
            f"Prospective 5-Yr IRR ({est_holding_irr:.1f}%/yr) trails your best replacement idea ({replacement_irr_pct:.1f}%/yr) "
            f"by {irr_spread:+.1f}%/yr—clearing the 8% hurdle required to absorb 12.5% LTCG tax."
        )
        tranche_plan = [
            "Tranche 1 (Swap 50%): Liquidate half of this stagnant holding to fund Tranche 1 & 2 of your higher-IRR target.",
            "Tranche 2 (Swap Remaining 50%): Rotate the second half once the new stock confirms quarterly execution.",
        ]

    else:
        exit_verdict = "🛡️ NO SELL TRIGGERS ACTIVE — HOLD CORE & LET IT COMPOUND"
        exit_color = "#10b981"  # Emerald Green
        triggers_fired.append(
            f"Thesis Intact: Moat Grade ({moat_grade}), ROCE ({roce_pct or 0:.1f}%), and Implied Growth ({implied_g1_pct:.1f}%) are within healthy bounds."
        )
        triggers_fired.append(
            f"Prospective Holding IRR (~{est_holding_irr:.1f}%/yr) does not warrant paying 12.5% LTCG tax to switch."
        )
        tranche_plan = [
            "Action: Do nothing. Sit tight and let compounding work without unnecessary tax friction.",
            "Add on Pullbacks: Use the Staggered Buy Zones above if the stock dips during market corrections.",
        ]

    return {
        "exit_verdict": exit_verdict,
        "exit_color": exit_color,
        "triggers_fired": triggers_fired,
        "tranche_plan": tranche_plan,
        "est_holding_irr": est_holding_irr,
        "irr_spread": irr_spread,
        "thesis_broken": thesis_broken,
    }

def calculate_staggered_buy_zones(
    cmp: float,
    dcf_intrinsic: Optional[float],
    graham_val: Optional[float],
    eps_ttm: Optional[float],
    expected_g1_pct: float,
    moat_grade: str,
) -> Dict[str, Any]:
    """
    Calculates institutional Staggered Entry Price Zones:
      1. Aggressive SIP Zone (Quality Compounder Fair Anchor)
      2. GARP Fair-Multiple Buy Zone (Core accumulation zone)
      3. Deep Value Graham Zone (High Margin-of-Safety Floor)
    """
    valid_dcf = dcf_intrinsic if (dcf_intrinsic is not None and dcf_intrinsic > 0) else (cmp * 0.75)
    valid_eps = eps_ttm if (eps_ttm is not None and eps_ttm > 0) else (cmp / 35.0)
    safe_g = max(min(expected_g1_pct, 35.0), 8.0)

    garp_pe = safe_g * 1.25 if moat_grade in ["A+", "A"] else safe_g * 1.00
    compounder_pe = safe_g * 1.70 if moat_grade == "A+" else safe_g * 1.35

    garp_multiple_price = valid_eps * garp_pe
    compounder_multiple_price = valid_eps * compounder_pe

    # Zone 1: Aggressive SIP Zone (Tranche 1: Compounder pullback accumulation)
    sip_upper = round(max(valid_dcf * 1.10, compounder_multiple_price), 2)
    sip_lower = round(max(valid_dcf, (sip_upper + garp_multiple_price) / 2.0), 2)
    if sip_lower > sip_upper:
        sip_lower, sip_upper = sip_upper * 0.94, sip_upper

    # Zone 2: GARP Fair-Multiple Buy Zone (Tranche 2: Core sizing)
    garp_upper = round(min(sip_lower * 0.98, max(valid_dcf, garp_multiple_price)), 2)
    garp_lower = round(garp_upper * 0.90, 2)

    # Zone 3: Deep Value Graham Zone (Tranche 3: Deep value floor)
    valid_graham = graham_val if (graham_val is not None and graham_val > 0) else (valid_dcf * 0.70)
    deep_upper = round(min(garp_lower * 0.95, max(valid_graham, valid_dcf * 0.75)), 2)
    deep_lower = round(deep_upper * 0.88, 2)

    valuation_anchor = round((sip_lower + garp_upper) / 2.0, 2)

    return {
        "sip_zone": (sip_lower, sip_upper),
        "garp_zone": (garp_lower, garp_upper),
        "deep_value_zone": (deep_lower, deep_upper),
        "valuation_anchor": valuation_anchor,
    }


# ==============================================================================
# 3. INDIAN CURRENCY & SAFE FORMATTING UTILITIES
# ==============================================================================
def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
            if not cleaned or cleaned in ("-", "."):
                return default
            val = float(cleaned)
        else:
            val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def format_inr_price(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.2f}"


def format_inr_crores(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e7:
        return f"{sign}₹{abs_val / 1e7:,.2f} Cr"
    if abs_val >= 1e5:
        return f"{sign}₹{abs_val / 1e5:,.2f} L"
    return f"{sign}₹{abs_val:,.0f}"


def format_shares_cr(shares_val: Optional[float]) -> str:
    if shares_val is None or np.isnan(shares_val) or shares_val <= 0:
        return "N/A"
    return f"{shares_val / 1e7:,.2f} Cr"


def get_statement_row(df: pd.DataFrame, possible_labels: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    normalized_index = {str(idx).strip().lower(): idx for idx in df.index}
    for label in possible_labels:
        key = label.strip().lower()
        if key in normalized_index:
            return df.loc[normalized_index[key]]
    return None


def classify_industry_profile(sector: str, industry: str, ticker: str = "") -> Dict[str, Any]:
    sec_lower = (sector or "").lower()
    ind_lower = (industry or "").lower()
    t_upper = (ticker or "").upper()

    financial_keywords = ["bank", "financial", "insurance", "credit", "capital markets", "nbfc", "lending", "housing finance"]
    is_financial = any(k in sec_lower or k in ind_lower for k in financial_keywords) or any(
        k in t_upper for k in ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "AXISBANK", "KOTAKBANK", "PFC", "RECLTD", "CHOLAFIN"]
    )

    if is_financial:
        archetype = "BANKING_NBFC"
    elif any(k in sec_lower or k in ind_lower for k in ["technology", "software", "it services", "computer"]) or any(
        k in t_upper for k in ["TCS.", "INFY.", "WIPRO.", "HCLTECH.", "TECHM.", "LTIM.", "PERSISTENT."]
    ):
        archetype = "IT_TECH"
    elif any(k in sec_lower or k in ind_lower for k in ["consumer defensive", "fmcg", "packaged foods", "household", "beverages", "tobacco"]) or any(
        k in t_upper for k in ["ITC.", "HINDUNILVR.", "NESTLEIND.", "BRITANNIA.", "DABUR.", "MARICO.", "GODREJCP."]
    ):
        archetype = "FMCG_CONSUMER"
    elif any(k in sec_lower or k in ind_lower for k in ["healthcare", "pharma", "biotech", "medical", "diagnostics"]) or any(
        k in t_upper for k in ["SUNPHARMA.", "DRREDDY.", "CIPLA.", "DIVISLAB.", "LUPIN.", "CUPID."]
    ):
        archetype = "PHARMA_HEALTH"
    elif any(k in sec_lower or k in ind_lower for k in ["energy", "oil", "gas", "utilities", "power", "renewable", "wind", "solar"]) or any(
        k in t_upper for k in ["ONGC.", "NTPC.", "POWERGRID.", "SUZLON.", "TATAPOWER.", "COALINDIA.", "ADANIGREEN."]
    ):
        archetype = "ENERGY_UTILITIES"
    else:
        archetype = "MANUFACTURING_INDUSTRIAL"

    if is_financial:
        return {
            "is_financial": True,
            "archetype": archetype,
            "profile_name": "Banking / NBFC / Financial Institution",
            "roce_applicable": False,
            "ebitda_applicable": False,
            "de_applicable": False,
            "applicable_note": (
                "Banking / NBFC Detected: Deposits & borrowings are operational raw material for lenders. "
                "EBITDA, ROCE, and Debt-to-Equity are marked N/A; analysis prioritizes ROE (>=14%), ROA (>=1.2%), "
                "P/E & P/B Historical Valuation Bands, and PEG."
            ),
        }
    return {
        "is_financial": False,
        "archetype": archetype,
        "profile_name": f"Indian Operating Business ({sector or 'Manufacturing / Capital Goods'})",
        "roce_applicable": True,
        "ebitda_applicable": True,
        "de_applicable": True,
        "applicable_note": (
            "Operating Company Detected: Full analysis active across ROCE, ROE, Trailing vs Historical P/E Band, "
            "PEG Ratio, Quarterly QoQ/YoY Turnaround Scanner, 2-Stage DCF, and Elite Compounder Moat grading."
        ),
    }


# ==============================================================================
# 4. MULTI-SOURCE DATA FETCHER (SCREENER.IN + YFINANCE)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_india_data(clean_symbol: str) -> Dict[str, Any]:
    base_sym = clean_symbol.split(".")[0].strip().upper()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }

    result: Dict[str, Any] = {
        "available": False,
        "ratios": {},
        "quarters_df": pd.DataFrame(),
        "pnl_df": pd.DataFrame(),
        "cf_df": pd.DataFrame(),
        "bs_df": pd.DataFrame(),
    }

    urls = [
        f"https://www.screener.in/company/{base_sym}/consolidated/",
        f"https://www.screener.in/company/{base_sym}/",
    ]

    soup = None
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code == 200 and "top-ratios" in resp.text:
                cand_soup = BeautifulSoup(resp.text, "html.parser")
                pnl_sec = cand_soup.find("section", id="profit-loss")
                qtr_sec = cand_soup.find("section", id="quarters")
                if (pnl_sec and len(pnl_sec.find_all("tr")) >= 3) or (qtr_sec and len(qtr_sec.find_all("tr")) >= 3):
                    soup = cand_soup
                    break
        except Exception:
            continue

    if soup is None:
        return result

    ratios: Dict[str, Optional[float]] = {}
    top_ul = soup.find("ul", id="top-ratios")
    if top_ul:
        for li in top_ul.find_all("li"):
            name_el = li.find("span", class_="name")
            num_el = li.find("span", class_="number")
            if name_el and num_el:
                r_name = name_el.get_text(strip=True).lower()
                r_val = safe_float(num_el.get_text(strip=True))
                if "market cap" in r_name and r_val is not None:
                    ratios["market_cap_cr"] = r_val
                elif "current price" in r_name and r_val is not None:
                    ratios["current_price"] = r_val
                elif "stock p/e" in r_name and r_val is not None:
                    ratios["pe"] = r_val
                elif "book value" in r_name and r_val is not None:
                    ratios["book_value"] = r_val
                elif "roce" in r_name and r_val is not None:
                    ratios["roce_pct"] = r_val
                elif "roe" in r_name and r_val is not None:
                    ratios["roe_pct"] = r_val

    def parse_screener_table(section_id: str) -> pd.DataFrame:
        sec = soup.find("section", id=section_id)
        if not sec:
            return pd.DataFrame()
        table = sec.find("table", class_="data-table")
        if not table:
            return pd.DataFrame()

        headers_list = []
        thead = table.find("thead")
        if thead:
            headers_list = [th.get_text(strip=True) for th in thead.find_all("th")]

        rows_dict = {}
        tbody = table.find("tbody")
        if tbody and len(headers_list) > 1:
            for tr in tbody.find_all("tr"):
                tds = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if len(tds) == len(headers_list):
                    row_label = re.sub(r"\+$", "", tds[0]).strip()
                    vals = [safe_float(v) for v in tds[1:]]
                    rows_dict[row_label] = vals

        if not rows_dict:
            return pd.DataFrame()
        cols_clean = []
        seen_c: Dict[str, int] = {}
        for c in headers_list[1:]:
            c_str = c if c else "Period"
            if c_str in seen_c:
                seen_c[c_str] += 1
                cols_clean.append(f"{c_str}_{seen_c[c_str]}")
            else:
                seen_c[c_str] = 0
                cols_clean.append(c_str)

        df_out = pd.DataFrame.from_dict(rows_dict, orient="index", columns=cols_clean)
        return df_out

    quarters_df = parse_screener_table("quarters")
    pnl_df = parse_screener_table("profit-loss")
    bs_df = parse_screener_table("balance-sheet")
    cf_df = parse_screener_table("cash-flow")

    result["available"] = bool(ratios or not pnl_df.empty or not quarters_df.empty)
    result["ratios"] = ratios
    result["quarters_df"] = quarters_df
    result["pnl_df"] = pnl_df
    result["bs_df"] = bs_df
    result["cf_df"] = cf_df
    return result


@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_bundle(ticker_symbol: str) -> Dict[str, Any]:
    ticker_clean = ticker_symbol.strip().upper()
    if "." not in ticker_clean:
        ticker_clean = f"{ticker_clean}.NS"

    ticker_obj = yf.Ticker(ticker_clean)

    info: Dict[str, Any] = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    fast_price, fast_mcap, fast_shares = None, None, None
    try:
        fi = ticker_obj.fast_info
        if fi is not None:
            fast_price = safe_float(getattr(fi, "last_price", None)) or safe_float(getattr(fi, "previous_close", None))
            fast_mcap = safe_float(getattr(fi, "market_cap", None))
            fast_shares = safe_float(getattr(fi, "shares", None))
    except Exception:
        pass

    screener_data = fetch_screener_india_data(ticker_clean)
    scr_ratios = screener_data.get("ratios", {})

    hist_5y = pd.DataFrame()
    try:
        hist_5y = ticker_obj.history(period="5y")
    except Exception:
        hist_5y = pd.DataFrame()

    current_price = (
        safe_float(info.get("currentPrice"))
        or safe_float(info.get("regularMarketPrice"))
        or safe_float(info.get("previousClose"))
        or fast_price
        or scr_ratios.get("current_price")
    )

    if current_price is None and hist_5y is not None and not hist_5y.empty:
        current_price = safe_float(hist_5y["Close"].iloc[-1])

    if current_price is None:
        alt_ticker = ticker_clean.replace(".NS", ".BO") if ticker_clean.endswith(".NS") else ticker_clean.replace(".BO", ".NS")
        try:
            alt_obj = yf.Ticker(alt_ticker)
            alt_hist = alt_obj.history(period="5y")
            if alt_hist is not None and not alt_hist.empty:
                current_price = safe_float(alt_hist["Close"].iloc[-1])
                hist_5y = alt_hist
                ticker_clean = alt_ticker
                ticker_obj = alt_obj
                info = alt_obj.info or {}
        except Exception:
            pass

    if current_price is None:
        raise ValueError(f"Could not retrieve live market pricing for '{ticker_clean}'.")

    yearly_close_map: Dict[int, float] = {}
    if hist_5y is not None and not hist_5y.empty and "Close" in hist_5y.columns:
        try:
            close_s = hist_5y["Close"].dropna()
            for yr, grp in close_s.groupby(close_s.index.year):
                yearly_close_map[int(yr)] = float(grp.median())
        except Exception:
            pass

    try:
        income_stmt = ticker_obj.financials
        if income_stmt is None or income_stmt.empty:
            income_stmt = ticker_obj.income_stmt
    except Exception:
        income_stmt = pd.DataFrame()

    try:
        quarterly_stmt = ticker_obj.quarterly_financials
        if quarterly_stmt is None or quarterly_stmt.empty:
            quarterly_stmt = ticker_obj.quarterly_income_stmt
    except Exception:
        quarterly_stmt = pd.DataFrame()

    try:
        balance_sheet = ticker_obj.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cash_flow = ticker_obj.cashflow
        if cash_flow is None or cash_flow.empty:
            cash_flow = ticker_obj.cash_flow
    except Exception:
        cash_flow = pd.DataFrame()

    try:
        raw_news = ticker_obj.news or []
    except Exception:
        raw_news = []

    sector = info.get("sector") or "Industrials"
    industry = info.get("industry") or "General"
    profile = classify_industry_profile(sector, industry, ticker_clean)

    return {
        "ticker": ticker_clean,
        "currency": "INR",
        "info": info,
        "fast_mcap": fast_mcap or (scr_ratios.get("market_cap_cr") * 1e7 if scr_ratios.get("market_cap_cr") else None),
        "fast_shares": fast_shares,
        "screener": screener_data,
        "yearly_close_map": yearly_close_map,
        "sector": sector,
        "industry": industry,
        "profile": profile,
        "current_price": current_price,
        "income_stmt": income_stmt,
        "quarterly_stmt": quarterly_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "raw_news": raw_news,
    }


# ==============================================================================
# 5. QUARTERLY RESULTS & METRICS EXTRACTION
# ==============================================================================
def extract_quarterly_trend_analysis(bundle: Dict[str, Any]) -> Dict[str, Any]:
    scr_qtr = bundle.get("screener", {}).get("quarters_df", pd.DataFrame())
    yf_qtr = bundle.get("quarterly_stmt", pd.DataFrame())
    profile = bundle["profile"]

    q_cols: List[str] = []
    sales_list: List[Optional[float]] = []
    op_profit_list: List[Optional[float]] = []
    opm_list: List[Optional[float]] = []
    net_profit_list: List[Optional[float]] = []
    eps_list: List[Optional[float]] = []
    source_used = "Screener.in Quarterly Filings (₹ Cr)"

    if scr_qtr is not None and not scr_qtr.empty and len(scr_qtr.columns) >= 4:
        num_q = min(5, len(scr_qtr.columns))
        recent_cols = list(scr_qtr.columns[-num_q:])
        q_cols = recent_cols

        sales_row = get_statement_row(scr_qtr, ["Sales", "Revenue", "Interest", "Financing Profit"])
        op_row = get_statement_row(scr_qtr, ["Operating Profit", "Financing Profit", "EBIT"])
        opm_row = get_statement_row(scr_qtr, ["OPM %", "OPM", "Financing Margin %"])
        np_row = get_statement_row(scr_qtr, ["Net Profit"])
        eps_row = get_statement_row(scr_qtr, ["EPS in Rs", "EPS"])

        for c in recent_cols:
            s_v = safe_float(sales_row.get(c)) if sales_row is not None else None
            op_v = safe_float(op_row.get(c)) if op_row is not None else None
            opm_v = safe_float(opm_row.get(c)) if opm_row is not None else None
            if opm_v is None and s_v and s_v > 0 and op_v is not None:
                opm_v = (op_v / s_v) * 100.0
            np_v = safe_float(np_row.get(c)) if np_row is not None else None
            ep_v = safe_float(eps_row.get(c)) if eps_row is not None else None

            sales_list.append(s_v)
            op_profit_list.append(op_v)
            opm_list.append(opm_v)
            net_profit_list.append(np_v)
            eps_list.append(ep_v)

    elif yf_qtr is not None and not yf_qtr.empty and len(yf_qtr.columns) >= 2:
        source_used = "Yahoo Finance Quarterly Filings (₹ Cr)"
        num_q = min(5, len(yf_qtr.columns))
        raw_cols = list(yf_qtr.columns[:num_q])[::-1]
        q_cols = [c.strftime("%b %Y") if hasattr(c, "strftime") else str(c)[:7] for c in raw_cols]

        rev_s = get_statement_row(yf_qtr, ["Total Revenue", "Operating Revenue", "Revenue", "Interest Income"])
        ebitda_s = get_statement_row(yf_qtr, ["EBITDA", "Normalized EBITDA", "Operating Income", "EBIT"])
        ni_s = get_statement_row(yf_qtr, ["Net Income", "Net Income Common Stockholders"])
        eps_s = get_statement_row(yf_qtr, ["Diluted EPS", "Basic EPS"])

        for c in raw_cols:
            r_raw = safe_float(rev_s.get(c)) if rev_s is not None else None
            eb_raw = safe_float(ebitda_s.get(c)) if ebitda_s is not None else None
            ni_raw = safe_float(ni_s.get(c)) if ni_s is not None else None
            ep_v = safe_float(eps_s.get(c)) if eps_s is not None else None

            s_cr = (r_raw / 1e7) if r_raw is not None else None
            op_cr = (eb_raw / 1e7) if eb_raw is not None else None
            np_cr = (ni_raw / 1e7) if ni_raw is not None else None
            opm_v = (op_cr / s_cr * 100.0) if (op_cr is not None and s_cr and s_cr > 0) else None

            sales_list.append(s_cr)
            op_profit_list.append(op_cr)
            opm_list.append(opm_v)
            net_profit_list.append(np_cr)
            eps_list.append(ep_v)

    if len(q_cols) < 2:
        return {
            "available": False,
            "df_quarterly": pd.DataFrame(),
            "turnaround_status": "DATA UNAVAILABLE",
            "turnaround_badge_color": "#64748b",
            "turnaround_summary": "Quarterly breakdown was not returned by the data feed for this symbol.",
            "score_bonus": 0.0,
            "qoq_sales_pct": None,
            "yoy_sales_pct": None,
            "qoq_np_pct": None,
            "yoy_np_pct": None,
        }

    def calc_pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
        if curr is None or prev is None or prev == 0:
            return None
        return ((curr - prev) / abs(prev)) * 100.0

    has_yoy = len(q_cols) >= 5
    qoq_sales = calc_pct_change(sales_list[-1], sales_list[-2])
    yoy_sales = calc_pct_change(sales_list[-1], sales_list[0]) if has_yoy else None
    qoq_op = calc_pct_change(op_profit_list[-1], op_profit_list[-2])
    yoy_op = calc_pct_change(op_profit_list[-1], op_profit_list[0]) if has_yoy else None
    qoq_opm_bps = (opm_list[-1] - opm_list[-2]) if (opm_list[-1] is not None and opm_list[-2] is not None) else None
    yoy_opm_bps = (opm_list[-1] - opm_list[0]) if (has_yoy and opm_list[-1] is not None and opm_list[0] is not None) else None
    qoq_np = calc_pct_change(net_profit_list[-1], net_profit_list[-2])
    yoy_np = calc_pct_change(net_profit_list[-1], net_profit_list[0]) if has_yoy else None
    qoq_eps = calc_pct_change(eps_list[-1], eps_list[-2])
    yoy_eps = calc_pct_change(eps_list[-1], eps_list[0]) if has_yoy else None

    disp_cols = q_cols[-4:]
    idx_offset = len(q_cols) - len(disp_cols)

    def fmt_cr_cell(v: Optional[float]) -> str:
        return f"₹{v:,.2f} Cr" if v is not None else "N/A"

    def fmt_pct_cell(v: Optional[float]) -> str:
        return f"{v:+.2f}%" if v is not None else "N/A"

    table_rows = []
    r_sales: Dict[str, Any] = {"Quarterly Metric": "Revenue / Sales (₹ Cr)"}
    for i, col_name in enumerate(disp_cols):
        r_sales[col_name] = fmt_cr_cell(sales_list[idx_offset + i])
    r_sales["QoQ Change (%)"] = fmt_pct_cell(qoq_sales)
    r_sales["YoY Change (%)"] = fmt_pct_cell(yoy_sales) if has_yoy else "Requires 5th Qtr"
    table_rows.append(r_sales)

    op_label = "Financing Profit (₹ Cr)" if profile["is_financial"] else "Operating Profit / EBITDA (₹ Cr)"
    r_op: Dict[str, Any] = {"Quarterly Metric": op_label}
    for i, col_name in enumerate(disp_cols):
        r_op[col_name] = fmt_cr_cell(op_profit_list[idx_offset + i])
    r_op["QoQ Change (%)"] = fmt_pct_cell(qoq_op)
    r_op["YoY Change (%)"] = fmt_pct_cell(yoy_op) if has_yoy else "—"
    table_rows.append(r_op)

    m_label = "Financing Margin (%)" if profile["is_financial"] else "Operating Profit Margin (OPM %)"
    r_opm: Dict[str, Any] = {"Quarterly Metric": m_label}
    for i, col_name in enumerate(disp_cols):
        v_m = opm_list[idx_offset + i]
        r_opm[col_name] = f"{v_m:.2f}%" if v_m is not None else "N/A"
    r_opm["QoQ Change (%)"] = f"{qoq_opm_bps:+.2f}% pts" if qoq_opm_bps is not None else "N/A"
    r_opm["YoY Change (%)"] = f"{yoy_opm_bps:+.2f}% pts" if yoy_opm_bps is not None else "—"
    table_rows.append(r_opm)

    r_np: Dict[str, Any] = {"Quarterly Metric": "Net Profit (₹ Cr)"}
    for i, col_name in enumerate(disp_cols):
        r_np[col_name] = fmt_cr_cell(net_profit_list[idx_offset + i])
    r_np["QoQ Change (%)"] = fmt_pct_cell(qoq_np)
    r_np["YoY Change (%)"] = fmt_pct_cell(yoy_np) if has_yoy else "—"
    table_rows.append(r_np)

    r_eps: Dict[str, Any] = {"Quarterly Metric": "Quarterly EPS (₹)"}
    for i, col_name in enumerate(disp_cols):
        v_e = eps_list[idx_offset + i]
        r_eps[col_name] = format_inr_price(v_e)
    r_eps["QoQ Change (%)"] = fmt_pct_cell(qoq_eps)
    r_eps["YoY Change (%)"] = fmt_pct_cell(yoy_eps) if has_yoy else "—"
    table_rows.append(r_eps)

    df_qtr_display = pd.DataFrame(table_rows).set_index("Quarterly Metric")

    np_latest = net_profit_list[-1]
    np_prev = net_profit_list[-2]
    np_yoy_base = net_profit_list[0] if has_yoy else None

    loss_to_profit = (
        np_latest is not None
        and np_latest > 0
        and ((np_prev is not None and np_prev <= 0) or (np_yoy_base is not None and np_yoy_base <= 0))
    )
    strong_expansion = (
        (qoq_sales is not None and qoq_sales > 3.0)
        and (qoq_np is not None and qoq_np > 8.0)
        and (qoq_opm_bps is None or qoq_opm_bps >= -0.5)
    )
    yoy_breakout = (
        (yoy_sales is not None and yoy_sales >= 12.0)
        and (yoy_np is not None and yoy_np >= 15.0)
    )

    if loss_to_profit:
        t_status = "🚀 QUARTERLY LOSS-TO-PROFIT TURNAROUND DETECTED"
        t_color = "#16a34a"
        t_summary = (
            f"Latest quarter swung to a positive Net Profit of ₹{np_latest:,.2f} Cr "
            f"from a prior loss, signaling an early operational inflection."
        )
        score_bonus = 2.0
    elif strong_expansion or yoy_breakout:
        t_status = "🟢 STRONG QUARTERLY EARNINGS & MARGIN MOMENTUM"
        t_color = "#16a34a"
        s_qoq_str = f"{qoq_sales:+.1f}% QoQ" if qoq_sales is not None else ""
        s_yoy_str = f" / {yoy_sales:+.1f}% YoY" if yoy_sales is not None else ""
        np_qoq_str = f"{qoq_np:+.1f}% QoQ" if qoq_np is not None else ""
        np_yoy_str = f" / {yoy_np:+.1f}% YoY" if yoy_np is not None else ""
        t_summary = (
            f"Revenue grew {s_qoq_str}{s_yoy_str} while Net Profit expanded {np_qoq_str}{np_yoy_str}, "
            f"confirming strong execution."
        )
        score_bonus = 1.5
    elif (qoq_np is not None and qoq_np < -12.0) or (yoy_np is not None and yoy_np < -15.0):
        t_status = "🔴 QUARTERLY EARNINGS CONTRACTION / MARGIN PRESSURE"
        t_color = "#dc2626"
        np_qoq_str = f"{qoq_np:+.1f}% QoQ" if qoq_np is not None else "N/A"
        np_yoy_str = f"{yoy_np:+.1f}% YoY" if yoy_np is not None else "N/A"
        t_summary = f"Latest quarter faced earnings pressure (Net Profit: {np_qoq_str}, {np_yoy_str})."
        score_bonus = -1.5
    else:
        t_status = "⚖️ STEADY / MIXED QUARTERLY RUN-RATE"
        t_color = "#2563eb"
        t_summary = "Steady operational run-rate without sharp quarterly contraction."
        score_bonus = 0.5

    return {
        "available": True,
        "source_used": source_used,
        "latest_quarter_label": disp_cols[-1],
        "df_quarterly": df_qtr_display,
        "turnaround_status": t_status,
        "turnaround_badge_color": t_color,
        "turnaround_summary": t_summary,
        "score_bonus": score_bonus,
        "qoq_sales_pct": qoq_sales,
        "yoy_sales_pct": yoy_sales,
        "qoq_np_pct": qoq_np,
        "yoy_np_pct": yoy_np,
    }


def compute_cagr(recent_val: Optional[float], oldest_val: Optional[float], periods: int) -> Optional[float]:
    if recent_val is None or oldest_val is None or periods <= 0 or recent_val <= 0 or oldest_val <= 0:
        return None
    return float(((recent_val / oldest_val) ** (1.0 / periods) - 1.0) * 100.0)


def get_screener_series_recent_first(df: pd.DataFrame, row_candidates: List[str], scale_mult: float = 1.0) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    norm_idx = {str(idx).strip().lower(): idx for idx in df.index}
    matched_row = None
    for cand in row_candidates:
        c_low = cand.strip().lower()
        if c_low in norm_idx:
            matched_row = df.loc[norm_idx[c_low]]
            break
        for k, orig_k in norm_idx.items():
            if c_low in k:
                matched_row = df.loc[orig_k]
                break
        if matched_row is not None:
            break

    if matched_row is None:
        return None

    s_clean = matched_row.dropna()
    if s_clean.empty:
        return None
    return s_clean.iloc[::-1] * scale_mult


def extract_comprehensive_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    info = bundle["info"]
    inc = bundle["income_stmt"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    profile = bundle["profile"]
    scr = bundle.get("screener", {})
    scr_ratios = scr.get("ratios", {})
    scr_pnl = scr.get("pnl_df", pd.DataFrame())
    scr_bs = scr.get("bs_df", pd.DataFrame())
    scr_cf = scr.get("cf_df", pd.DataFrame())
    yearly_close_map = bundle.get("yearly_close_map", {})

    rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue", "Interest Income"])
    if rev_s is None or len(rev_s.dropna()) < 2:
        rev_s = get_screener_series_recent_first(scr_pnl, ["Sales", "Revenue", "Financing Profit"], scale_mult=1e7)

    gp_s = get_statement_row(inc, ["Gross Profit"])
    ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA"])
    if (ebitda_s is None or len(ebitda_s.dropna()) < 2) and profile["ebitda_applicable"]:
        ebitda_s = get_screener_series_recent_first(scr_pnl, ["Operating Profit"], scale_mult=1e7)

    ebit_s = get_statement_row(inc, ["EBIT", "Operating Income"])
    if ebit_s is None or len(ebit_s.dropna()) < 2:
        ebit_s = ebitda_s

    ni_s = get_statement_row(inc, ["Net Income", "Net Income Common Stockholders"])
    if ni_s is None or len(ni_s.dropna()) < 2:
        ni_s = get_screener_series_recent_first(scr_pnl, ["Net Profit"], scale_mult=1e7)

    eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
    if eps_s is None or len(eps_s.dropna()) < 2:
        eps_s = get_screener_series_recent_first(scr_pnl, ["EPS in Rs", "EPS"], scale_mult=1.0)

    shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])
    if shares_s is None:
        shares_s = get_statement_row(bs, ["Ordinary Shares Number", "Share Issued"])

    assets_s = get_statement_row(bs, ["Total Assets"])
    if assets_s is None or assets_s.dropna().empty:
        assets_s = get_screener_series_recent_first(scr_bs, ["Total Assets"], scale_mult=1e7)

    curr_liab_s = get_statement_row(bs, ["Current Liabilities", "Total Current Liabilities"])
    if curr_liab_s is None or curr_liab_s.dropna().empty:
        curr_liab_s = get_screener_series_recent_first(scr_bs, ["Other Liabilities"], scale_mult=1e7)

    curr_assets_s = get_statement_row(bs, ["Current Assets", "Total Current Assets"])
    equity_s = get_statement_row(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt_s = get_statement_row(bs, ["Total Debt", "Long Term Debt"])
    if debt_s is None or debt_s.dropna().empty:
        debt_s = get_screener_series_recent_first(scr_bs, ["Borrowings"], scale_mult=1e7)

    fcf_s = get_statement_row(cf, ["Free Cash Flow"])
    ocf_s = get_statement_row(cf, ["Operating Cash Flow"])
    if ocf_s is None or ocf_s.dropna().empty:
        ocf_s = get_screener_series_recent_first(scr_cf, ["Cash from Operating Activity"], scale_mult=1e7)

    capex_s = get_statement_row(cf, ["Capital Expenditure"])
    buyback_cf_s = get_statement_row(cf, ["Repurchase Of Capital Stock", "Common Stock Payments"])

    roe_pct = scr_ratios.get("roe_pct")
    if roe_pct is None:
        roe_yf = safe_float(info.get("returnOnEquity"))
        if roe_yf is not None:
            roe_pct = roe_yf * 100.0
    if roe_pct is None and ni_s is not None and equity_s is not None:
        ni_0 = safe_float(ni_s.dropna().iloc[0]) if not ni_s.dropna().empty else None
        eq_0 = safe_float(equity_s.dropna().iloc[0]) if not equity_s.dropna().empty else None
        if ni_0 is not None and eq_0 and eq_0 > 0:
            roe_pct = (ni_0 / eq_0) * 100.0

    roce_pct = scr_ratios.get("roce_pct") if profile["roce_applicable"] else None
    roce_prev_pct = None
    if profile["roce_applicable"] and ebit_s is not None and assets_s is not None and curr_liab_s is not None:
        common_cols = [c for c in ebit_s.index if c in assets_s.index and c in curr_liab_s.index]
        roce_vals = []
        for col in common_cols[:4]:
            ebit_v = safe_float(ebit_s.get(col))
            ass_v = safe_float(assets_s.get(col))
            cl_v = safe_float(curr_liab_s.get(col))
            if ebit_v is not None and ass_v is not None and cl_v is not None:
                cap_emp = ass_v - cl_v
                if cap_emp > 0:
                    roce_vals.append((ebit_v / cap_emp) * 100.0)
        if roce_vals:
            if roce_pct is None:
                roce_pct = roce_vals[0]
            if len(roce_vals) > 1:
                roce_prev_pct = float(np.mean(roce_vals[1:]))

    if roce_prev_pct is None and roce_pct is not None:
        roce_prev_pct = roce_pct * 0.94

    roa_yf = safe_float(info.get("returnOnAssets"))
    roa_pct = roa_yf * 100.0 if roa_yf is not None else None
    if roa_pct is None and ni_s is not None and assets_s is not None:
        ni_0 = safe_float(ni_s.dropna().iloc[0]) if not ni_s.dropna().empty else None
        as_0 = safe_float(assets_s.dropna().iloc[0]) if not assets_s.dropna().empty else None
        if ni_0 is not None and as_0 and as_0 > 0:
            roa_pct = (ni_0 / as_0) * 100.0

    raw_de = safe_float(info.get("debtToEquity"))
    de_ratio = (raw_de / 100.0) if (raw_de is not None and profile["de_applicable"]) else None
    if de_ratio is None and profile["de_applicable"] and debt_s is not None and equity_s is not None:
        d_0 = safe_float(debt_s.dropna().iloc[0]) if not debt_s.dropna().empty else None
        eq_0 = safe_float(equity_s.dropna().iloc[0]) if not equity_s.dropna().empty else None
        if d_0 is not None and eq_0 and eq_0 > 0:
            de_ratio = d_0 / eq_0

    current_ratio = safe_float(info.get("currentRatio"))
    if current_ratio is None and curr_assets_s is not None and curr_liab_s is not None:
        ca_0 = safe_float(curr_assets_s.dropna().iloc[0]) if not curr_assets_s.dropna().empty else None
        cl_0 = safe_float(curr_liab_s.dropna().iloc[0]) if not curr_liab_s.dropna().empty else None
        if ca_0 is not None and cl_0 and cl_0 > 0:
            current_ratio = ca_0 / cl_0

    pb_ratio = safe_float(info.get("priceToBook"))
    if pb_ratio is None and scr_ratios.get("book_value") and scr_ratios["book_value"] > 0:
        pb_ratio = bundle["current_price"] / scr_ratios["book_value"]

    latest_stmt_eps = None
    if eps_s is not None and not eps_s.dropna().empty:
        latest_stmt_eps = safe_float(eps_s.dropna().iloc[0])

    pe_ratio = safe_float(info.get("trailingPE")) or scr_ratios.get("pe") or safe_float(info.get("forwardPE"))
    if pe_ratio is None and latest_stmt_eps and latest_stmt_eps > 0:
        pe_ratio = bundle["current_price"] / latest_stmt_eps
    forward_pe = safe_float(info.get("forwardPE"))

    hist_pe_list: List[float] = []
    if eps_s is not None and not eps_s.dropna().empty and yearly_close_map:
        for idx_label, eps_val in eps_s.dropna().items():
            e_float = safe_float(eps_val)
            if e_float is None or e_float <= 0:
                continue
            yr_match = re.search(r"(20\d{2})", str(idx_label))
            if yr_match:
                yr_int = int(yr_match.group(1))
                if yr_int in yearly_close_map:
                    implied_pe = yearly_close_map[yr_int] / e_float
                    if 2.0 <= implied_pe <= 200.0:
                        hist_pe_list.append(implied_pe)

    historical_median_pe = float(np.median(hist_pe_list)) if hist_pe_list else None
    if historical_median_pe is None and pe_ratio is not None and forward_pe is not None and forward_pe > 0:
        historical_median_pe = float(np.mean([pe_ratio, forward_pe]))

    latest_ni = safe_float(ni_s.dropna().iloc[0]) if (ni_s is not None and not ni_s.dropna().empty) else safe_float(info.get("netIncomeToCommon"))
    latest_ocf = safe_float(ocf_s.dropna().iloc[0]) if (ocf_s is not None and not ocf_s.dropna().empty) else safe_float(info.get("operatingCashflow"))
    latest_fcf = safe_float(fcf_s.dropna().iloc[0]) if (fcf_s is not None and not fcf_s.dropna().empty) else safe_float(info.get("freeCashflow"))
    if latest_fcf is None and latest_ocf is not None and capex_s is not None and not capex_s.dropna().empty:
        cx_0 = safe_float(capex_s.dropna().iloc[0], 0.0) or 0.0
        latest_fcf = latest_ocf + cx_0 if cx_0 < 0 else latest_ocf - cx_0

    ocf_to_ni_ratio = (latest_ocf / latest_ni) if (latest_ocf is not None and latest_ni and latest_ni > 0) else None
    fcf_to_ni_ratio = (latest_fcf / latest_ni) if (latest_fcf is not None and latest_ni and latest_ni > 0) else None

    def extract_growth_profile(series: Optional[pd.Series], info_yoy_key: Optional[str] = None) -> Dict[str, Optional[float]]:
        recent_yoy, past_cagr, latest_val, prev_val = None, None, None, None
        if series is not None and not series.dropna().empty:
            clean_s = series.dropna()
            latest_val = safe_float(clean_s.iloc[0])
            if len(clean_s) >= 2:
                prev_val = safe_float(clean_s.iloc[1])
                if latest_val is not None and prev_val is not None and prev_val != 0:
                    recent_yoy = ((latest_val - prev_val) / abs(prev_val)) * 100.0
            if len(clean_s) >= 3:
                idx_old = min(3, len(clean_s) - 1)
                oldest_val = safe_float(clean_s.iloc[idx_old])
                past_cagr = compute_cagr(latest_val, oldest_val, idx_old)
            elif len(clean_s) == 2:
                past_cagr = recent_yoy

        if recent_yoy is None and info_yoy_key:
            info_v = safe_float(info.get(info_yoy_key))
            if info_v is not None:
                recent_yoy = info_v * 100.0
        if past_cagr is None and recent_yoy is not None:
            past_cagr = recent_yoy * 0.90

        return {"latest": latest_val, "prev": prev_val, "recent_yoy_pct": recent_yoy, "past_cagr_pct": past_cagr}

    rev_growth = extract_growth_profile(rev_s, "revenueGrowth")
    ebitda_growth = extract_growth_profile(ebitda_s) if profile["ebitda_applicable"] else {"latest": None, "prev": None, "recent_yoy_pct": None, "past_cagr_pct": None}
    ni_growth = extract_growth_profile(ni_s, "earningsGrowth")
    eps_growth = extract_growth_profile(eps_s, "earningsGrowth")

    ebitda_margin_curr, ebitda_margin_past = None, None
    net_margin_curr, net_margin_past = None, None
    gross_margin_curr, gross_margin_past = None, None

    if rev_s is not None and not rev_s.dropna().empty:
        cols = list(rev_s.dropna().index[:4])
        gm_list, em_list, nm_list = [], [], []
        for c in cols:
            r_v = safe_float(rev_s.get(c))
            if r_v and r_v > 0:
                if gp_s is not None and c in gp_s.index and safe_float(gp_s.get(c)) is not None:
                    gm_list.append((safe_float(gp_s.get(c)) / r_v) * 100.0)
                if profile["ebitda_applicable"] and ebitda_s is not None and c in ebitda_s.index and safe_float(ebitda_s.get(c)) is not None:
                    em_list.append((safe_float(ebitda_s.get(c)) / r_v) * 100.0)
                if ni_s is not None and c in ni_s.index and safe_float(ni_s.get(c)) is not None:
                    nm_list.append((safe_float(ni_s.get(c)) / r_v) * 100.0)

        if gm_list:
            gross_margin_curr = gm_list[0]
            gross_margin_past = float(np.mean(gm_list[1:])) if len(gm_list) > 1 else gm_list[0]
        if em_list:
            ebitda_margin_curr = em_list[0]
            ebitda_margin_past = float(np.mean(em_list[1:])) if len(em_list) > 1 else em_list[0]
        if nm_list:
            net_margin_curr = nm_list[0]
            net_margin_past = float(np.mean(nm_list[1:])) if len(nm_list) > 1 else nm_list[0]

    if ebitda_margin_curr is None and profile["ebitda_applicable"]:
        opm_s = get_screener_series_recent_first(scr_pnl, ["OPM %", "OPM"], scale_mult=1.0)
        if opm_s is not None and not opm_s.empty:
            ebitda_margin_curr = safe_float(opm_s.iloc[0])
            ebitda_margin_past = float(np.mean(opm_s.iloc[1:4])) if len(opm_s) > 1 else ebitda_margin_curr

    share_change_pct = None
    latest_stmt_shares = None
    if shares_s is not None and not shares_s.dropna().empty:
        clean_sh = shares_s.dropna()
        latest_stmt_shares = safe_float(clean_sh.iloc[0])
        if len(clean_sh) >= 2:
            oldest_shares = safe_float(clean_sh.iloc[min(2, len(clean_sh) - 1)])
            if latest_stmt_shares and oldest_shares and oldest_shares > 0:
                share_change_pct = ((latest_stmt_shares - oldest_shares) / oldest_shares) * 100.0

    total_3yr_buyback_cash = 0.0
    if buyback_cf_s is not None and not buyback_cf_s.dropna().empty:
        clean_bb = buyback_cf_s.dropna()
        bb_vals = [abs(safe_float(v, 0.0) or 0.0) for v in clean_bb.iloc[:3] if (safe_float(v, 0.0) or 0.0) < 0]
        if bb_vals:
            total_3yr_buyback_cash = float(np.sum(bb_vals))

    eps_cagr = eps_growth["past_cagr_pct"] or eps_growth["recent_yoy_pct"]
    ni_cagr = ni_growth["past_cagr_pct"] or ni_growth["recent_yoy_pct"]
    eps_buyback_boost_pct = (eps_cagr - ni_cagr) if (eps_cagr is not None and ni_cagr is not None) else None

    buyback_detected = (
        (share_change_pct is not None and share_change_pct <= -1.0)
        or (total_3yr_buyback_cash > 0)
        or (eps_buyback_boost_pct is not None and eps_buyback_boost_pct >= 1.5)
    )

    if buyback_detected and eps_buyback_boost_pct is not None and eps_buyback_boost_pct >= 1.5:
        buyback_verdict = "BUYBACK-BOOSTED EPS"
        sh_str = f"{share_change_pct:+.2f}%" if share_change_pct is not None else "reduced"
        bb_cash_str = format_inr_crores(total_3yr_buyback_cash)
        buyback_explanation = (
            f"Share count changed by {sh_str} over the multi-year window (3-Yr Buybacks: {bb_cash_str}). "
            f"EPS grew at {eps_cagr:+.2f}%/yr vs Net Income growth of {ni_cagr:+.2f}%/yr."
        )
    elif share_change_pct is not None and share_change_pct >= 2.0:
        buyback_verdict = "EQUITY DILUTION DETECTED"
        buyback_explanation = (
            f"Diluted share count increased by {share_change_pct:+.2f}% over the multi-year window, "
            f"causing per-share EPS growth to lag Net Profit growth."
        )
    else:
        buyback_verdict = "100% ORGANIC EPS GROWTH"
        buyback_explanation = "No material share-count distortion detected. Reported EPS closely tracks Net Profit growth."

    return {
        "roe_pct": roe_pct,
        "roce_pct": roce_pct,
        "roce_prev_pct": roce_prev_pct,
        "roa_pct": roa_pct,
        "de_ratio": de_ratio,
        "current_ratio": current_ratio,
        "pb_ratio": pb_ratio,
        "pe_ratio": pe_ratio,
        "forward_pe": forward_pe,
        "historical_median_pe": historical_median_pe,
        "hist_pe_count": len(hist_pe_list),
        "latest_fcf": latest_fcf,
        "latest_ocf": latest_ocf,
        "latest_ni": latest_ni,
        "latest_stmt_shares": latest_stmt_shares,
        "latest_stmt_eps": latest_stmt_eps,
        "ocf_to_ni_ratio": ocf_to_ni_ratio,
        "fcf_to_ni_ratio": fcf_to_ni_ratio,
        "rev_growth": rev_growth,
        "ebitda_growth": ebitda_growth,
        "ni_growth": ni_growth,
        "eps_growth": eps_growth,
        "gross_margin_curr": gross_margin_curr,
        "gross_margin_past": gross_margin_past,
        "ebitda_margin_curr": ebitda_margin_curr,
        "ebitda_margin_past": ebitda_margin_past,
        "net_margin_curr": net_margin_curr,
        "net_margin_past": net_margin_past,
        "share_change_pct": share_change_pct,
        "total_3yr_buyback_cash": total_3yr_buyback_cash,
        "eps_buyback_boost_pct": eps_buyback_boost_pct,
        "buyback_detected": buyback_detected,
        "buyback_verdict": buyback_verdict,
        "buyback_explanation": buyback_explanation,
    }


def build_three_year_history_table(bundle: Dict[str, Any]) -> pd.DataFrame:
    inc = bundle["income_stmt"]
    cf = bundle["cash_flow"]
    profile = bundle["profile"]
    scr_pnl = bundle.get("screener", {}).get("pnl_df", pd.DataFrame())

    if inc is not None and not inc.empty and len(inc.columns) >= 2:
        cols = list(inc.columns[:3])
        rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue", "Interest Income"])
        ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA", "Operating Income"])
        ni_s = get_statement_row(inc, ["Net Income", "Net Income Common Stockholders"])
        eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
        shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])
        fcf_s = get_statement_row(cf, ["Free Cash Flow"])
        ocf_s = get_statement_row(cf, ["Operating Cash Flow"])
        capex_s = get_statement_row(cf, ["Capital Expenditure"])

        records = []
        for col in cols:
            year_label = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
            rev = safe_float(rev_s.get(col)) if rev_s is not None else None
            ebitda = safe_float(ebitda_s.get(col)) if (ebitda_s is not None and profile["ebitda_applicable"]) else None
            ni = safe_float(ni_s.get(col)) if ni_s is not None else None
            eps_v = safe_float(eps_s.get(col)) if eps_s is not None else None
            sh_v = safe_float(shares_s.get(col)) if shares_s is not None else None

            fcf = safe_float(fcf_s.get(col)) if fcf_s is not None else None
            if fcf is None and ocf_s is not None and capex_s is not None:
                ocf = safe_float(ocf_s.get(col))
                capex = safe_float(capex_s.get(col))
                if ocf is not None and capex is not None:
                    fcf = ocf + capex if capex < 0 else ocf - capex

            ebitda_margin = (ebitda / rev * 100.0) if (ebitda is not None and rev and rev > 0) else None
            net_margin = (ni / rev * 100.0) if (ni is not None and rev and rev > 0) else None

            records.append(
                {
                    "Fiscal Year": year_label,
                    "Revenue": format_inr_crores(rev),
                    "EBITDA": format_inr_crores(ebitda) if profile["ebitda_applicable"] else "N/A (Bank/NBFC)",
                    "EBITDA Margin": f"{ebitda_margin:.2f}%" if ebitda_margin is not None else "N/A",
                    "Net Profit": format_inr_crores(ni),
                    "Net Profit Margin": f"{net_margin:.2f}%" if net_margin is not None else "N/A",
                    "Diluted EPS": format_inr_price(eps_v),
                    "Diluted Shares": format_shares_cr(sh_v),
                    "Free Cash Flow": format_inr_crores(fcf),
                }
            )
        return pd.DataFrame(records).set_index("Fiscal Year")

    if not scr_pnl.empty:
        rev_s = get_screener_series_recent_first(scr_pnl, ["Sales", "Revenue"], scale_mult=1e7)
        ebitda_s = get_screener_series_recent_first(scr_pnl, ["Operating Profit"], scale_mult=1e7)
        ni_s = get_screener_series_recent_first(scr_pnl, ["Net Profit"], scale_mult=1e7)
        eps_s = get_screener_series_recent_first(scr_pnl, ["EPS in Rs", "EPS"], scale_mult=1.0)
        if rev_s is not None:
            cols = list(rev_s.index[:3])
            records = []
            for col in cols:
                r_v = safe_float(rev_s.get(col))
                eb_v = safe_float(ebitda_s.get(col)) if (ebitda_s is not None and profile["ebitda_applicable"]) else None
                ni_v = safe_float(ni_s.get(col)) if ni_s is not None else None
                ep_v = safe_float(eps_s.get(col)) if eps_s is not None else None
                em_v = (eb_v / r_v * 100.0) if (eb_v is not None and r_v and r_v > 0) else None
                nm_v = (ni_v / r_v * 100.0) if (ni_v is not None and r_v and r_v > 0) else None
                records.append(
                    {
                        "Fiscal Year": str(col),
                        "Revenue": format_inr_crores(r_v),
                        "EBITDA": format_inr_crores(eb_v) if profile["ebitda_applicable"] else "N/A",
                        "EBITDA Margin": f"{em_v:.2f}%" if em_v is not None else "N/A",
                        "Net Profit": format_inr_crores(ni_v),
                        "Net Profit Margin": f"{nm_v:.2f}%" if nm_v is not None else "N/A",
                        "Diluted EPS": format_inr_price(ep_v),
                    }
                )
            return pd.DataFrame(records).set_index("Fiscal Year")

    return pd.DataFrame()


# ==============================================================================
# 6. SECTOR-SPECIFIC VALUE INVESTING CHECKLIST ENGINE
# ==============================================================================
def evaluate_sector_checklist(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    archetype = bundle["profile"]["archetype"]
    roe = metrics.get("roe_pct")
    roce = metrics.get("roce_pct")
    roa = metrics.get("roa_pct")
    de = metrics.get("de_ratio")
    cr = metrics.get("current_ratio")
    pb = metrics.get("pb_ratio")
    gm = metrics.get("gross_margin_curr")
    em = metrics.get("ebitda_margin_curr")
    em_past = metrics.get("ebitda_margin_past")
    nm = metrics.get("net_margin_curr")
    fcf = metrics.get("latest_fcf")
    ocf_ni = metrics.get("ocf_to_ni_ratio")

    def status_badge(passed: Optional[bool], borderline: bool = False) -> str:
        if passed is None:
            return "⚪ DATA N/A"
        if passed:
            return "✅ PASS"
        if borderline:
            return "⚠️ BORDERLINE"
        return "❌ FAIL"

    if archetype == "MANUFACTURING_INDUSTRIAL":
        sector_title = "🏭 Manufacturing, Auto & Heavy Industrials Checklist"
        moat_focus = "Capital efficiency on physical plants (ROCE >= 15%), debt discipline across cycles, and positive FCF after CapEx."
        em_str = f"{em:.2f}% (3Y Avg: {em_past:.1f}%)" if (em is not None and em_past is not None) else (f"{em:.2f}%" if em is not None else "N/A")
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 15.0% (Acceptable >= 12%)",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 15.0 if roce is not None else None, borderline=(roce is not None and 11.5 <= roce < 15.0)),
                "Why It Matters in This Sector": "Proves factories and machinery generate returns above the cost of capital.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 0.80x (Safe < 1.2x)",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 0.80, borderline=(de is not None and 0.80 < de <= 1.20)),
                "Why It Matters in This Sector": "Prevents interest payments from wiping out profits during industrial downturns.",
            },
            {
                "Parameter": "Free Cash Flow After CapEx",
                "Target Benchmark": "Positive (> ₹0 Cr)",
                "Actual Value": format_inr_crores(fcf),
                "Status": status_badge(fcf > 0 if fcf is not None else None),
                "Why It Matters in This Sector": "Checks if plant CapEx leaves real cash for shareholders.",
            },
            {
                "Parameter": "Operating EBITDA Margin Stability",
                "Target Benchmark": ">= 12.0% & Stable vs 3-Yr Avg",
                "Actual Value": em_str,
                "Status": status_badge(em >= 12.0 if em is not None else None, borderline=(em is not None and 8.0 <= em < 12.0)),
                "Why It Matters in This Sector": "Demonstrates pricing power against raw material cost swings.",
            },
        ]
        qual_questions = [
            "CapEx Reinvestment Check: Is recent plant/machinery CapEx expanding profitable capacity?",
            "Working Capital Check: Are receivables and inventory growing in line with revenue?",
            "Cyclicality Buffer: Can the company survive an industrial slowdown without excessive debt?",
        ]

    elif archetype == "BANKING_NBFC":
        sector_title = "🏦 Banking, NBFCs & Financial Services Checklist"
        moat_focus = "Asset quality (low NPAs), Return on Assets (ROA >= 1.2%), compounding book value (ROE >= 14%), and P/B valuation."
        quant_checks = [
            {
                "Parameter": "Return on Assets (ROA)",
                "Target Benchmark": ">= 1.20% (Elite >= 1.5%)",
                "Actual Value": f"{roa:.2f}%" if roa is not None else "N/A",
                "Status": status_badge(roa >= 1.20 if roa is not None else None, borderline=(roa is not None and 0.90 <= roa < 1.20)),
                "Why It Matters in This Sector": "The primary measure of a lender's core asset productivity.",
            },
            {
                "Parameter": "Return on Equity (ROE)",
                "Target Benchmark": ">= 14.0%",
                "Actual Value": f"{roe:.2f}%" if roe is not None else "N/A",
                "Status": status_badge(roe >= 14.0 if roe is not None else None, borderline=(roe is not None and 11.5 <= roe < 14.0)),
                "Why It Matters in This Sector": "Measures how fast book value compounds.",
            },
            {
                "Parameter": "Price-to-Book Ratio (P/B)",
                "Target Benchmark": "<= 3.0x (Value < 2.2x)",
                "Actual Value": f"{pb:.2f}x" if pb is not None else "N/A",
                "Status": status_badge(pb <= 3.0 if pb is not None else None, borderline=(pb is not None and 3.0 < pb <= 4.2)),
                "Why It Matters in This Sector": "Valuing banks against tangible net worth protects downside.",
            },
            {
                "Parameter": "Net Profit Margin",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{nm:.2f}%" if nm is not None else "N/A",
                "Status": status_badge(nm >= 15.0 if nm is not None else None, borderline=(nm is not None and 10.0 <= nm < 15.0)),
                "Why It Matters in This Sector": "Reflects Net Interest Margin health after credit costs.",
            },
        ]
        qual_questions = [
            "Asset Quality (GNPA / NNPA): Are NPAs stable or declining over the last 4 quarters?",
            "Liability Franchise (CASA): Does the lender possess sticky, low-cost deposits?",
            "Capital Adequacy (CRAR): Is Tier-1 Capital well above regulatory thresholds?",
        ]

    elif archetype == "IT_TECH":
        sector_title = "💻 Information Technology & Software Services Checklist"
        moat_focus = "Asset-light scalability, high switching costs, strong cash conversion, and low leverage."
        roe_roce_str = f"ROE: {roe:.1f}% | ROCE: {roce:.1f}%" if (roe is not None and roce is not None) else (f"ROE: {roe:.1f}%" if roe is not None else "N/A")
        quant_checks = [
            {
                "Parameter": "Return on Equity & ROCE",
                "Target Benchmark": "ROE >= 20% & ROCE >= 20%",
                "Actual Value": roe_roce_str,
                "Status": status_badge((roe or 0) >= 20.0 and (roce or 20.0) >= 20.0 if (roe is not None or roce is not None) else None, borderline=((roe or 0) >= 15.0)),
                "Why It Matters in This Sector": "Asset-light tech leaders should generate superior capital returns.",
            },
            {
                "Parameter": "Cash Conversion (OCF / Net Income)",
                "Target Benchmark": ">= 0.85x (85%+ conversion)",
                "Actual Value": f"{ocf_ni:.2f}x" if ocf_ni is not None else "N/A",
                "Status": status_badge(ocf_ni >= 0.85 if ocf_ni is not None else None, borderline=(ocf_ni is not None and 0.70 <= ocf_ni < 0.85)),
                "Why It Matters in This Sector": "Verifies that software billings convert directly into bank balances.",
            },
            {
                "Parameter": "Balance Sheet Leverage (D/E)",
                "Target Benchmark": "<= 0.40x (Near Zero Debt)",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x (Debt-Free)",
                "Status": status_badge((de or 0.0) <= 0.40, borderline=(de is not None and 0.40 < de <= 0.80)),
                "Why It Matters in This Sector": "Software leaders self-fund R&D from cash flow.",
            },
            {
                "Parameter": "Operating / EBITDA Margin",
                "Target Benchmark": ">= 18.0%",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 18.0 if em is not None else None, borderline=(em is not None and 14.0 <= em < 18.0)),
                "Why It Matters in This Sector": "Protects against wage hikes and subcontractor inflation.",
            },
        ]
        qual_questions = [
            "Order Book & Deal Pipeline: Is recurring client revenue / TCV expanding?",
            "Tech Moat: Are client switching costs strong enough to resist fee compression?",
            "Share Count Discipline: Are ESOP dilutions minimal?",
        ]

    elif archetype == "FMCG_CONSUMER":
        sector_title = "🛒 FMCG, Consumer Goods & Retail Brand Checklist"
        moat_focus = "Brand pricing power, distribution dominance, negative/low working capital, and ROCE."
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 22.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 22.0 if roce is not None else None, borderline=(roce is not None and 15.0 <= roce < 22.0)),
                "Why It Matters in This Sector": "Hallmark of enduring consumer brand moats.",
            },
            {
                "Parameter": "Operating EBITDA Margin",
                "Target Benchmark": ">= 18.0% (Strong Brand Moat)",
                "Actual Value": f"{em:.2f}%" if em is not None else (f"Gross: {gm:.1f}%" if gm is not None else "N/A"),
                "Status": status_badge((em or 0) >= 18.0 if em is not None else None, borderline=(em is not None and 13.0 <= em < 18.0)),
                "Why It Matters in This Sector": "Confirms consumers willingly pay a brand premium.",
            },
            {
                "Parameter": "Free Cash Flow Conversion (OCF / Net Income)",
                "Target Benchmark": ">= 0.85x",
                "Actual Value": f"{ocf_ni:.2f}x" if ocf_ni is not None else "N/A",
                "Status": status_badge(ocf_ni >= 0.85 if ocf_ni is not None else None, borderline=(ocf_ni is not None and 0.70 <= ocf_ni < 0.85)),
                "Why It Matters in This Sector": "Prevents inventory stagnation across distribution channels.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 0.50x",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 0.50, borderline=(de is not None and 0.50 < de <= 0.90)),
                "Why It Matters in This Sector": "Surplus cash generators should require minimal debt.",
            },
        ]
        qual_questions = [
            "Volume vs Price Growth: Is revenue driven by unit volume rather than purely price hikes?",
            "Market Share Defense: Is brand market share holding steady against new entrants?",
            "Valuation Discipline: Is the multiple reasonable relative to underlying growth?",
        ]

    elif archetype == "PHARMA_HEALTH":
        sector_title = "💊 Pharmaceuticals, Healthcare & Diagnostics Checklist"
        moat_focus = "R&D pipeline productivity, operating margins, USFDA/regulatory compliance, and working capital."
        quant_checks = [
            {
                "Parameter": "Operating EBITDA Margin",
                "Target Benchmark": ">= 20.0% (Specialty/Formulations)",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 20.0 if em is not None else None, borderline=(em is not None and 15.0 <= em < 20.0)),
                "Why It Matters in This Sector": "Funds continuous R&D and regulatory filings.",
            },
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 15.0 if roce is not None else None, borderline=(roce is not None and 11.5 <= roce < 15.0)),
                "Why It Matters in This Sector": "Validates manufacturing plant capital allocation.",
            },
            {
                "Parameter": "Current Ratio (Liquidity)",
                "Target Benchmark": ">= 1.50x",
                "Actual Value": f"{cr:.2f}x" if cr is not None else "N/A",
                "Status": status_badge(cr >= 1.50 if cr is not None else None, borderline=(cr is not None and 1.15 <= cr < 1.50)),
                "Why It Matters in This Sector": "Safeguards against long pharma working capital cycles.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 0.60x",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 0.60, borderline=(de is not None and 0.60 < de <= 1.0)),
                "Why It Matters in This Sector": "Protects solvency through unexpected regulatory audits.",
            },
        ]
        qual_questions = [
            "Regulatory Compliance: Are key facilities free from warning letters?",
            "Therapy Diversification: Is revenue balanced across diverse portfolios?",
            "Receivables Cycle: Are institutional hospital payments collected on schedule?",
        ]

    else:
        sector_title = "⚡ Energy, Power & Infrastructure Checklist"
        moat_focus = "Capital efficiency (ROCE >= 12%), order-book-to-cash conversion, manageable leverage."
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 12.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 12.0 if roce is not None else None, borderline=(roce is not None and 9.0 <= roce < 12.0)),
                "Why It Matters in This Sector": "Heavy infrastructure capital must clear the borrowing cost hurdle.",
            },
            {
                "Parameter": "EBITDA Margin",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 15.0 if em is not None else None, borderline=(em is not None and 11.0 <= em < 15.0)),
                "Why It Matters in This Sector": "Provides margin cushion against EPC execution swings.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 1.20x",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 1.20, borderline=(de is not None and 1.20 < de <= 1.80)),
                "Why It Matters in This Sector": "Keeps project debt from overwhelming solvency.",
            },
            {
                "Parameter": "Operating Cash Flow Generation",
                "Target Benchmark": "Positive (> ₹0 Cr)",
                "Actual Value": format_inr_crores(metrics.get("latest_ocf")),
                "Status": status_badge((metrics.get("latest_ocf") or 0) > 0 if metrics.get("latest_ocf") is not None else None),
                "Why It Matters in This Sector": "Proves order execution is generating hard cash.",
            },
        ]
        qual_questions = [
            "Order Execution: Is the order book converting into cash without receivables accumulation?",
            "Margin Reliability: Are margins sustainable without one-off subsidies?",
            "Expansion Funding: Can CapEx be met from internal cash flows?",
        ]

    pass_count = sum(1 for item in quant_checks if item["Status"] == "✅ PASS")
    borderline_count = sum(1 for item in quant_checks if item["Status"] == "⚠️ BORDERLINE")
    total_checks = len(quant_checks)

    return {
        "sector_title": sector_title,
        "moat_focus": moat_focus,
        "quant_checks": quant_checks,
        "qual_questions": qual_questions,
        "pass_count": pass_count,
        "borderline_count": borderline_count,
        "total_checks": total_checks,
    }


# ==============================================================================
# 7. PEER BENCHMARKING & VALUATION MODELS
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_industry_peer_benchmarks(
    target_ticker: str, archetype: str, custom_peers_str: str = ""
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    fallback_meds = NSE_SECTOR_BENCHMARKS.get(archetype, NSE_SECTOR_BENCHMARKS["MANUFACTURING_INDUSTRIAL"]).copy()

    default_peers = INDIA_SECTOR_PEERS.get(archetype, ["LT.NS", "TCS.NS", "ITC.NS", "MARUTI.NS"])
    if custom_peers_str.strip():
        raw_list = [p.strip().upper() for p in custom_peers_str.split(",") if p.strip()]
        peer_tickers = [f"{p}.NS" if "." not in p else p for p in raw_list]
    else:
        peer_tickers = [p for p in default_peers if p != target_ticker][:3]

    rows = []
    for pt in peer_tickers:
        try:
            t_obj = yf.Ticker(pt)
            p_info = t_obj.info or {}
            scr_p = fetch_screener_india_data(pt)
            scr_r = scr_p.get("ratios", {})

            rev_g = safe_float(p_info.get("revenueGrowth"))
            ebitda_m = safe_float(p_info.get("ebitdaMargins"))
            net_m = safe_float(p_info.get("profitMargins"))
            roe_v = scr_r.get("roe_pct") or (safe_float(p_info.get("returnOnEquity")) * 100.0 if safe_float(p_info.get("returnOnEquity")) is not None else None)
            roce_v = scr_r.get("roce_pct")
            pe_v = scr_r.get("pe") or safe_float(p_info.get("trailingPE"))

            if any(x is not None for x in [rev_g, ebitda_m, net_m, roe_v, roce_v, pe_v]):
                rows.append(
                    {
                        "Ticker": pt,
                        "Company": (p_info.get("shortName") or pt)[:24],
                        "Rev Growth (%)": rev_g * 100.0 if rev_g is not None else fallback_meds["rev_growth_med"],
                        "EBITDA Margin (%)": ebitda_m * 100.0 if ebitda_m is not None else fallback_meds["ebitda_margin_med"],
                        "Net Margin (%)": net_m * 100.0 if net_m is not None else fallback_meds["net_margin_med"],
                        "ROE (%)": roe_v if roe_v is not None else fallback_meds["roe_med"],
                        "ROCE (%)": roce_v if roce_v is not None else fallback_meds["roce_med"],
                        "Trailing P/E": pe_v if pe_v is not None else fallback_meds["pe_med"],
                    }
                )
        except Exception:
            continue

    df_peers = pd.DataFrame(rows)
    medians = fallback_meds.copy()

    if not df_peers.empty:
        for col, key in [
            ("Rev Growth (%)", "rev_growth_med"),
            ("EBITDA Margin (%)", "ebitda_margin_med"),
            ("Net Margin (%)", "net_margin_med"),
            ("ROE (%)", "roe_med"),
            ("ROCE (%)", "roce_med"),
            ("Trailing P/E", "pe_med"),
        ]:
            val = safe_float(df_peers[col].median(skipna=True))
            if val is not None:
                medians[key] = val

    return df_peers, medians


def extract_dcf_inputs(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    info = bundle["info"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    current_price = bundle["current_price"]

    shares = (
        safe_float(info.get("sharesOutstanding"))
        or safe_float(info.get("impliedSharesOutstanding"))
        or bundle.get("fast_shares")
        or metrics.get("latest_stmt_shares")
    )
    if shares is None or shares <= 0:
        mcap = safe_float(info.get("marketCap")) or bundle.get("fast_mcap")
        if mcap and current_price > 0:
            shares = mcap / current_price

    if (shares is None or shares <= 0) and metrics.get("latest_ni") and metrics.get("latest_stmt_eps"):
        if metrics["latest_stmt_eps"] > 0 and metrics["latest_ni"] > 0:
            shares = metrics["latest_ni"] / metrics["latest_stmt_eps"]

    fcf_0 = safe_float(info.get("freeCashflow")) or metrics.get("latest_fcf")
    fcf_source = "Reported TTM / Latest Annual Free Cash Flow"

    if fcf_0 is None or fcf_0 <= 0:
        ocf_val = metrics.get("latest_ocf")
        capex_s = get_statement_row(cf, ["Capital Expenditure"])
        capex_val = safe_float(capex_s.dropna().iloc[0]) if (capex_s is not None and not capex_s.dropna().empty) else 0.0
        if ocf_val is not None and ocf_val > 0:
            fcf_0 = ocf_val + capex_val if (capex_val or 0.0) < 0 else ocf_val - (capex_val or 0.0)
            fcf_source = "Operating Cash Flow minus CapEx"

    if fcf_0 is None or fcf_0 <= 0:
        ni_val = metrics.get("latest_ni") or safe_float(info.get("netIncomeToCommon"))
        if ni_val is not None and ni_val > 0:
            fcf_0 = ni_val * 0.85
            fcf_source = "Normalized Owner Earnings (85% Net Profit Proxy)"

    eps_ttm = safe_float(info.get("trailingEps")) or metrics.get("latest_stmt_eps") or safe_float(info.get("forwardEps"))
    if (eps_ttm is None or eps_ttm <= 0) and metrics.get("latest_ni") and shares and shares > 0:
        if metrics["latest_ni"] > 0:
            eps_ttm = metrics["latest_ni"] / shares
    if (eps_ttm is None or eps_ttm <= 0) and fcf_0 and shares and shares > 0:
        if fcf_0 > 0:
            eps_ttm = fcf_0 / shares

    if (fcf_0 is None or fcf_0 <= 0) and eps_ttm and eps_ttm > 0 and shares and shares > 0:
        fcf_0 = eps_ttm * shares * 0.85
        fcf_source = "Implied Owner Earnings (85% of EPS × Shares)"

    total_cash = safe_float(info.get("totalCash"))
    if total_cash is None:
        cash_s = get_statement_row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
        total_cash = safe_float(cash_s.dropna().iloc[0]) if (cash_s is not None and not cash_s.dropna().empty) else 0.0

    total_debt = safe_float(info.get("totalDebt"))
    if total_debt is None:
        debt_s = get_statement_row(bs, ["Total Debt", "Long Term Debt"])
        total_debt = safe_float(debt_s.dropna().iloc[0]) if (debt_s is not None and not debt_s.dropna().empty) else 0.0

    if bundle["profile"]["is_financial"]:
        total_cash = 0.0
        total_debt = 0.0

    eg_info = safe_float(info.get("earningsGrowth"))
    rg_info = safe_float(info.get("revenueGrowth"))

    candidates = [
        v
        for v in [
            eg_info * 100.0 if eg_info is not None else None,
            rg_info * 100.0 if rg_info is not None else None,
            metrics["rev_growth"]["recent_yoy_pct"],
            metrics["rev_growth"]["past_cagr_pct"],
            metrics["ni_growth"]["recent_yoy_pct"],
            metrics["ni_growth"]["past_cagr_pct"],
        ]
        if v is not None and v > 0
    ]
    est_g1 = float(np.median(candidates)) if candidates else 11.0
    base_g1_pct = float(np.clip(est_g1, 4.0, 20.0))

    beta = safe_float(info.get("beta"), default=1.0) or 1.0
    beta_clamped = float(np.clip(beta, 0.65, 1.65))

    rf_rate, erp, base_terminal_g_pct = 6.8, 5.0, 4.0
    base_discount_pct = float(np.clip(rf_rate + beta_clamped * erp, 9.5, 15.0))

    return {
        "shares": shares,
        "fcf_0": fcf_0,
        "fcf_source": fcf_source,
        "total_cash": total_cash or 0.0,
        "total_debt": total_debt or 0.0,
        "eps_ttm": eps_ttm,
        "beta": beta_clamped,
        "base_g1_pct": base_g1_pct,
        "base_discount_pct": base_discount_pct,
        "base_terminal_g_pct": base_terminal_g_pct,
    }


def extract_pe_peg_analysis(
    current_price: float,
    eps_ttm: Optional[float],
    g1_pct: float,
    metrics: Dict[str, Any],
    peer_medians: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    trailing_pe = metrics.get("pe_ratio")
    if trailing_pe is None and eps_ttm and eps_ttm > 0:
        trailing_pe = current_price / eps_ttm

    sector_pe = peer_medians.get("pe_med") or 24.0
    hist_pe = metrics.get("historical_median_pe") or sector_pe

    peg_ratio = None
    if trailing_pe is not None and trailing_pe > 0 and g1_pct > 0:
        peg_ratio = trailing_pe / g1_pct

    garp_pe = float(np.clip(g1_pct * 1.35, 10.0, 35.0))
    anchor_pes = [v for v in [hist_pe, sector_pe, garp_pe] if v is not None and v > 0]
    fair_base_pe = float(np.clip(np.median(anchor_pes), 10.0, 38.0)) if anchor_pes else 20.0

    pe_fair_value = (eps_ttm * fair_base_pe) if (eps_ttm is not None and eps_ttm > 0) else None
    pe_fair_mos_pct = (
        ((pe_fair_value - current_price) / pe_fair_value) * 100.0
        if (pe_fair_value is not None and pe_fair_value > 0)
        else None
    )

    vs_hist_pct = (((trailing_pe - hist_pe) / hist_pe) * 100.0) if (trailing_pe is not None and hist_pe and hist_pe > 0) else None
    vs_sector_pct = (((trailing_pe - sector_pe) / sector_pe) * 100.0) if (trailing_pe is not None and sector_pe and sector_pe > 0) else None

    if peg_ratio is None:
        peg_pts = 2.0
        peg_status = "⚪ PEG N/A"
    elif peg_ratio <= 1.0:
        peg_pts = 5.0
        peg_status = "🟢 Undervalued Growth (PEG <= 1.0x)"
    elif peg_ratio <= 1.5:
        peg_pts = 3.75
        peg_status = "🟢 Fair GARP Multiple (1.0x - 1.5x)"
    elif peg_ratio <= 2.2:
        peg_pts = 2.0
        peg_status = "🟡 Fully Priced for Growth (1.5x - 2.2x)"
    else:
        peg_pts = 0.0
        peg_status = "🔴 Expensive / Stretched PEG (> 2.2x)"

    rel_pe_pts = 0.0
    if trailing_pe is not None and trailing_pe > 0:
        if hist_pe and trailing_pe <= hist_pe * 0.90:
            rel_pe_pts += 2.5
        elif hist_pe and trailing_pe <= hist_pe * 1.08:
            rel_pe_pts += 1.5
        elif hist_pe and trailing_pe <= hist_pe * 1.25:
            rel_pe_pts += 0.5

        if sector_pe and trailing_pe <= sector_pe * 0.90:
            rel_pe_pts += 2.5
        elif sector_pe and trailing_pe <= sector_pe * 1.08:
            rel_pe_pts += 1.5
        elif sector_pe and trailing_pe <= sector_pe * 1.25:
            rel_pe_pts += 0.5
    else:
        rel_pe_pts = 1.5

    total_pe_peg_pts = float(np.clip(peg_pts + rel_pe_pts, 0.0, 10.0))

    if total_pe_peg_pts >= 7.5:
        overall_pe_verdict = "🟢 ATTRACTIVE MULTIPLE (Reasonable PEG & Multiples)"
        pe_badge_color = "#16a34a"
    elif total_pe_peg_pts >= 4.5:
        overall_pe_verdict = "⚖️ FAIR VALUATION MULTIPLE (In Line with Historical/Sector Norms)"
        pe_badge_color = "#2563eb"
    else:
        overall_pe_verdict = "🔴 STRETCHED P/E & PEG MULTIPLE (Trading at a Valuation Premium)"
        pe_badge_color = "#dc2626"

    return {
        "trailing_pe": trailing_pe,
        "hist_pe": hist_pe,
        "sector_pe": sector_pe,
        "peg_ratio": peg_ratio,
        "peg_status": peg_status,
        "fair_base_pe": fair_base_pe,
        "pe_fair_value": pe_fair_value,
        "pe_fair_mos_pct": pe_fair_mos_pct,
        "vs_hist_pct": vs_hist_pct,
        "vs_sector_pct": vs_sector_pct,
        "peg_pts": peg_pts,
        "rel_pe_pts": rel_pe_pts,
        "total_pe_peg_pts": total_pe_peg_pts,
        "overall_pe_verdict": overall_pe_verdict,
        "pe_badge_color": pe_badge_color,
    }


def estimate_years_to_target(
    current_price: float,
    target_iv: Optional[float],
    annual_growth_pct: float,
    discount_rate_pct: float,
) -> Tuple[Optional[float], str]:
    if target_iv is None or target_iv <= 0 or current_price <= 0:
        return None, "N/A (Insufficient valuation data)"

    ratio = target_iv / current_price
    if 0.95 <= ratio <= 1.05:
        return 0.5, "Trading within ±5% of Fair Value (~0 to 6 months)"

    if target_iv > current_price:
        comp_rate = max(0.11, (annual_growth_pct * 0.65 + discount_rate_pct * 0.50) / 100.0)
        years = math.log(target_iv / current_price) / math.log(1.0 + comp_rate)
        years_clamped = float(np.clip(years, 0.5, 12.0))
        low_y = max(0.5, round(years_clamped * 0.8, 1))
        high_y = round(years_clamped * 1.25, 1)
        return (
            round(years_clamped, 1),
            f"~{low_y} to {high_y} Years (Target Upside Re-Rating at ~{comp_rate * 100:.1f}%/yr)",
        )
    else:
        g_rate = max(0.08, annual_growth_pct / 100.0)
        years_catchup = math.log(current_price / target_iv) / math.log(1.0 + g_rate)
        years_clamped = float(np.clip(years_catchup, 0.5, 15.0))
        return (
            round(years_clamped, 1),
            f"Priced Above Fair Value — Needs ~{years_clamped:.1f} Years of earnings growth at {g_rate * 100:.1f}%/yr to justify today's price",
        )


def compute_single_dcf_graham_and_pe(
    fcf_0: Optional[float],
    shares: Optional[float],
    total_cash: float,
    total_debt: float,
    eps_ttm: Optional[float],
    g1_pct: float,
    discount_pct: float,
    terminal_g_pct: float,
    fair_pe_mult: float,
    current_price: float,
) -> Dict[str, Any]:
    safe_term_g_pct = min(terminal_g_pct, discount_pct - 1.5)
    g1 = g1_pct / 100.0
    r = discount_pct / 100.0
    g_term = safe_term_g_pct / 100.0

    dcf_per_share, dcf_mos_pct, pv_stage1, pv_terminal = None, None, None, None

    if fcf_0 is not None and fcf_0 > 0 and shares is not None and shares > 0 and r > g_term:
        pv_fcfs = []
        projected_fcf = fcf_0
        for year in range(1, 6):
            projected_fcf = projected_fcf * (1.0 + g1)
            pv_fcfs.append(projected_fcf / ((1.0 + r) ** year))

        pv_stage1 = float(np.sum(pv_fcfs))
        fcf_year_5 = fcf_0 * ((1.0 + g1) ** 5)
        terminal_value_year_5 = (fcf_year_5 * (1.0 + g_term)) / (r - g_term)
        pv_terminal = terminal_value_year_5 / ((1.0 + r) ** 5)

        enterprise_value = pv_stage1 + pv_terminal
        equity_value = enterprise_value + total_cash - total_debt
        if equity_value > 0:
            dcf_per_share = equity_value / shares
            dcf_mos_pct = ((dcf_per_share - current_price) / dcf_per_share) * 100.0

    graham_per_share, graham_mos_pct = None, None
    if eps_ttm is not None and eps_ttm > 0:
        graham_per_share = eps_ttm * (8.5 + 2.0 * g1_pct)
        if graham_per_share > 0:
            graham_mos_pct = ((graham_per_share - current_price) / graham_per_share) * 100.0

    pe_target_per_share, pe_target_mos_pct = None, None
    if eps_ttm is not None and eps_ttm > 0 and fair_pe_mult > 0:
        pe_target_per_share = eps_ttm * fair_pe_mult
        if pe_target_per_share > 0:
            pe_target_mos_pct = ((pe_target_per_share - current_price) / pe_target_per_share) * 100.0

    valid_vals = [v for v in [dcf_per_share, graham_per_share, pe_target_per_share] if v is not None and v > 0]
    blended_iv = float(np.mean(valid_vals)) if valid_vals else None
    blended_mos_pct = (((blended_iv - current_price) / blended_iv) * 100.0) if (blended_iv is not None and blended_iv > 0) else None

    est_years, est_years_label = estimate_years_to_target(
        current_price=current_price,
        target_iv=blended_iv,
        annual_growth_pct=g1_pct,
        discount_rate_pct=discount_pct,
    )

    return {
        "g1_pct": g1_pct,
        "discount_pct": discount_pct,
        "terminal_g_pct": safe_term_g_pct,
        "fair_pe_mult": fair_pe_mult,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "dcf_intrinsic": dcf_per_share,
        "dcf_mos_pct": dcf_mos_pct,
        "graham_intrinsic": graham_per_share,
        "graham_mos_pct": graham_mos_pct,
        "pe_intrinsic": pe_target_per_share,
        "pe_mos_pct": pe_target_mos_pct,
        "blended_intrinsic": blended_iv,
        "blended_mos_pct": blended_mos_pct,
        "est_years": est_years,
        "est_years_label": est_years_label,
    }


def build_three_scenarios(
    dcf_inputs: Dict[str, Any],
    base_g1_pct: float,
    base_discount_pct: float,
    base_terminal_g_pct: float,
    fair_base_pe: float,
    current_price: float,
) -> Dict[str, Dict[str, Any]]:
    cons_g1 = max(2.0, base_g1_pct * 0.70)
    cons_r = min(18.0, base_discount_pct + 1.5)
    cons_term = max(2.0, base_terminal_g_pct - 0.5)
    cons_pe = max(8.0, fair_base_pe * 0.80)

    opt_g1 = min(25.0, base_g1_pct * 1.25)
    opt_r = max(8.0, base_discount_pct - 1.0)
    opt_term = min(opt_r - 2.0, base_terminal_g_pct + 0.5)
    opt_pe = min(45.0, fair_base_pe * 1.20)

    return {
        "Conservative": compute_single_dcf_graham_and_pe(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], cons_g1, cons_r, cons_term, cons_pe, current_price
        ),
        "Base Case": compute_single_dcf_graham_and_pe(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], base_g1_pct, base_discount_pct, base_terminal_g_pct, fair_base_pe, current_price
        ),
        "Optimistic": compute_single_dcf_graham_and_pe(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], opt_g1, opt_r, opt_term, opt_pe, current_price
        ),
    }


# ==============================================================================
# 8. NEWS SENTIMENT ANALYSIS
# ==============================================================================
NOISE_PATTERNS = [
    r"\bmidday movers\b", r"\bstock market today\b", r"\btop gainers\b",
    r"\btop losers\b", r"\boptions alert\b", r"\bunusual options\b",
    r"\bchart watch\b", r"\btechnical analysis\b", r"\bintraday\b",
]


def is_market_noise(headline: str) -> bool:
    text = headline.lower()
    return any(re.search(pattern, text) for pattern in NOISE_PATTERNS)


def analyze_corporate_news(raw_news: List[Dict[str, Any]], max_items: int = 7) -> Dict[str, Any]:
    sia = load_sentiment_analyzer()
    parsed_articles = []

    for item in raw_news:
        if not isinstance(item, dict):
            continue
        content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
        title = item.get("title") or content.get("title")
        publisher = (
            item.get("publisher")
            or (content.get("provider", {}) or {}).get("displayName")
            or "Financial Press"
        )
        link = (
            item.get("link")
            or (content.get("canonicalUrl", {}) or {}).get("url")
            or (content.get("clickThroughUrl", {}) or {}).get("url")
            or "#"
        )
        if not title or not isinstance(title, str):
            continue
        parsed_articles.append(
            {"title": title.strip(), "publisher": publisher, "link": link, "is_noise": is_market_noise(title)}
        )

    seen_titles = set()
    unique_articles = []
    for art in parsed_articles:
        key = art["title"].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_articles.append(art)

    structural = [a for a in unique_articles if not a["is_noise"]]
    noise_fallback = [a for a in unique_articles if a["is_noise"]]
    selected = (structural + noise_fallback)[:max_items]

    grouped: Dict[str, List[Dict[str, Any]]] = {"Positive": [], "Neutral": [], "Negative": []}
    compound_scores: List[float] = []

    for art in selected:
        score = 0.0
        if sia is not None:
            score = float(sia.polarity_scores(art["title"])["compound"])
        else:
            pos_words = {"beat", "growth", "record", "acquisition", "dividend", "upgrade", "profit", "launch", "expands", "order"}
            neg_words = {"miss", "lawsuit", "probe", "decline", "downgrade", "loss", "cut", "recall", "investigation"}
            words = set(re.findall(r"\w+", art["title"].lower()))
            score = 0.25 * len(words & pos_words) - 0.25 * len(words & neg_words)

        compound_scores.append(score)
        label = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
        art["compound"] = score
        art["sentiment"] = label
        grouped[label].append(art)

    avg_compound = float(np.mean(compound_scores)) if compound_scores else 0.0
    return {"selected_count": len(selected), "grouped": grouped, "avg_compound": avg_compound}


# ==============================================================================
# 9. UPGRADED DUAL-AXIS INSTITUTIONAL DECISION ENGINE
# ==============================================================================
def compute_decision_verdict(
    ticker: str,
    company_name: str,
    current_price: float,
    profile: Dict[str, Any],
    metrics: Dict[str, Any],
    qtr_analysis: Dict[str, Any],
    pe_peg_analysis: Dict[str, Any],
    peer_medians: Dict[str, Optional[float]],
    scenarios: Dict[str, Dict[str, Any]],
    news_summary: Dict[str, Any],
    checklist_result: Dict[str, Any],
    user_confirmed_count: int,
) -> Dict[str, Any]:
    base_case = scenarios["Base Case"]
    cons_case = scenarios["Conservative"]
    opt_case = scenarios["Optimistic"]

    mos = base_case.get("blended_mos_pct")
    dcf_iv = base_case.get("dcf_intrinsic")
    graham_iv = base_case.get("graham_intrinsic")
    pe_iv = base_case.get("pe_intrinsic")
    blended_iv = base_case.get("blended_intrinsic")

    roe = metrics.get("roe_pct")
    roce = metrics.get("roce_pct")
    roa = metrics.get("roa_pct")
    de = metrics.get("de_ratio")

    # 1. Blended Margin of Safety Sub-Score (0 - 20 pts)
    if mos is None:
        mos_pts = 4.0
    elif mos >= 30.0:
        mos_pts = 20.0
    elif mos >= 0.0:
        mos_pts = 10.0 + (mos / 30.0) * 10.0
    elif mos >= -30.0:
        mos_pts = max(0.0, 10.0 + (mos / 30.0) * 10.0)
    else:
        mos_pts = 0.0

    # 2. P/E & PEG Valuation Multiple Sub-Score (0 - 10 pts)
    pe_peg_pts = pe_peg_analysis["total_pe_peg_pts"]

    # 3. Capital Efficiency Sub-Score (0 - 18 pts)
    roe_pts = min(9.0, max(0.0, ((roe or 0.0) / 20.0) * 9.0)) if roe is not None else 4.5
    if profile["roce_applicable"]:
        roce_pts = min(9.0, max(0.0, ((roce or 0.0) / 20.0) * 9.0)) if roce is not None else 4.5
    else:
        roce_pts = min(9.0, max(0.0, ((roa or 0.0) / 1.5) * 9.0)) if roa is not None else 4.5
    efficiency_pts = roe_pts + roce_pts

    # 4. Growth & Turnaround Quality (0 - 12 pts)
    rev_yoy = metrics["rev_growth"]["recent_yoy_pct"]
    rev_cagr = metrics["rev_growth"]["past_cagr_pct"]
    growth_pts = 2.5
    if rev_yoy is not None and rev_yoy > 8.0:
        growth_pts += 1.5
    if rev_yoy is not None and rev_cagr is not None and rev_yoy >= rev_cagr:
        growth_pts += 1.5

    nm_curr = metrics.get("net_margin_curr")
    nm_past = metrics.get("net_margin_past")
    if nm_curr is not None and nm_past is not None and nm_curr >= nm_past:
        growth_pts += 1.5

    if metrics["buyback_verdict"] == "100% ORGANIC EPS GROWTH":
        growth_pts += 1.5
    elif metrics["buyback_verdict"] == "BUYBACK-BOOSTED EPS":
        growth_pts += 0.8

    growth_pts += float(qtr_analysis.get("score_bonus", 0.0))
    growth_pts += min(2.0, user_confirmed_count * (2.0 / 3.0))
    growth_pts = float(np.clip(growth_pts, 0.0, 12.0))

    # 5. Sector Checklist Pass Rate & Peer Outperformance (0 - 10 pts)
    quant_pass_ratio = checklist_result["pass_count"] / max(1, checklist_result["total_checks"])
    bs_peer_pts = quant_pass_ratio * 6.0

    roe_med = peer_medians.get("roe_med")
    nm_med = peer_medians.get("net_margin_med")
    if roe is not None and roe_med is not None and roe >= roe_med:
        bs_peer_pts += 2.0
    else:
        bs_peer_pts += 0.5
    if nm_curr is not None and nm_med is not None and nm_curr >= nm_med:
        bs_peer_pts += 2.0
    else:
        bs_peer_pts += 0.5
    bs_peer_pts = min(10.0, bs_peer_pts)

    fundamental_score_70 = float(np.clip(mos_pts + pe_peg_pts + efficiency_pts + growth_pts + bs_peer_pts, 0.0, 70.0))

    # 6. News Sentiment (0 - 30 pts)
    avg_comp = news_summary.get("avg_compound", 0.0)
    clamped_comp = float(np.clip(avg_comp, -0.5, 0.5))
    sentiment_score_30 = ((clamped_comp + 0.5) / 1.0) * 30.0
    total_score = float(np.clip(fundamental_score_70 + sentiment_score_30, 0.0, 100.0))

    # --- UPGRADE: SEPARATE BUSINESS MOAT GRADE (A+ TO D) ---
    is_a_plus = (
        (roce is not None and roce >= 20.0 if profile["roce_applicable"] else (roa is not None and roa >= 1.5))
        and (roe is not None and roe >= 18.0)
        and ((de or 0.0) <= 0.30 if profile["de_applicable"] else True)
        and (checklist_result["pass_count"] >= checklist_result["total_checks"])
    )

    if is_a_plus:
        moat_grade = "A+"
        moat_badge = "🏆 A+ (Elite Institutional Moat)"
    elif (
        (roce is not None and roce >= 15.0 if profile["roce_applicable"] else (roa is not None and roa >= 1.2))
        and (roe is not None and roe >= 14.0)
        and ((de or 0.0) <= 0.60 if profile["de_applicable"] else True)
        and (checklist_result["pass_count"] >= 3)
    ):
        moat_grade = "A"
        moat_badge = "✅ A (High-Quality Compounder)"
    elif (roe is not None and roe >= 11.0) and ((de or 0.0) <= 1.10 if profile["de_applicable"] else True):
        moat_grade = "B"
        moat_badge = "⚖️ B (Cyclical / Moderate Moat)"
    elif (roe is not None and roe >= 6.0) and ((de or 0.0) <= 1.60 if profile["de_applicable"] else True):
        moat_grade = "C"
        moat_badge = "⚠️ C (Weak Capital Efficiency)"
    else:
        moat_grade = "D"
        moat_badge = "🚨 D (Capital Destroyer / Structural Risk)"

    # --- UPGRADE: SEPARATE VALUATION PRICE GRADE ---
    peg_val = pe_peg_analysis.get("peg_ratio")
    if mos is not None and mos >= 20.0 and (peg_val is not None and peg_val <= 1.25):
        val_grade = "DEEP VALUE / BARGAIN"
    elif mos is not None and mos >= 0.0 and (peg_val is not None and peg_val <= 2.00):
        val_grade = "FAIR / GARP PRICED"
    elif mos is not None and mos > -25.0 and (peg_val is not None and peg_val <= 3.00):
        val_grade = "MODERATE QUALITY PREMIUM"
    else:
        val_grade = "EXPENSIVE / PRICED FOR PERFECTION"

    # --- UPGRADE: INSTITUTIONAL VERDICT (REPLACES LINE 1147 GUARDRAIL) ---
    severely_expensive_multiple = (peg_val is not None and peg_val > 3.0) and (mos is not None and mos < 0.0)

    if severely_expensive_multiple:
        if moat_grade == "A+":
            verdict = "💎 ELITE COMPOUNDER — PRICED FOR PERFECTION (HOLD EXISTING / BUY ON PULLBACK)"
            badge_color = "#00b4d8"
            verdict_summary = (
                f"Passes 100% of sector moat checks with exceptional efficiency "
                f"(ROCE: {roce or 0:.1f}%, ROE: {roe or 0:.1f}%, D/E: {de or 0:.2f}x). "
                f"Do NOT lump with weak businesses. Hold core positions and accumulate via Staggered Buy Zones below."
            )
        elif moat_grade == "A":
            verdict = "🌟 QUALITY COMPOUNDER — RICH VALUATION (HOLD / TRACK PULLBACK ZONES)"
            badge_color = "#f59e0b"
            verdict_summary = "Strong operational moat, but current market multiples leave zero margin of safety. Wait for pullbacks."
        else:
            verdict = "🚫 OVERVALUED / MULTIPLE TRAP (AVOID)"
            badge_color = "#dc2626"
            verdict_summary = (
                f"Negative Margin of Safety ({mos:.1f}%) combined with an extreme PEG ({peg_val:.2f}x) "
                f"and a {moat_grade}-grade business moat creates severe de-rating risk."
            )
    elif total_score >= 75.0 and (mos is not None and mos >= 15.0) and checklist_result["pass_count"] >= 3:
        verdict = "🚀 STRONG BUY (HIGH-CONVICTION GARP)"
        badge_color = "#16a34a"
        verdict_summary = "Rare institutional combination: high-quality moat trading at an attractive Margin of Safety."
    elif total_score >= 60.0 and (mos is not None and mos >= -5.0):
        verdict = "✅ ACCUMULATE / FAIRLY VALUED GARP"
        badge_color = "#2563eb"
        verdict_summary = "Solid fundamentals trading within fair valuation bounds."
    elif total_score >= 45.0:
        verdict = "👀 HOLD / WATCHLIST"
        badge_color = "#d97706"
        verdict_summary = "Moderate valuation and fundamentals; track quarterly execution."
    else:
        verdict = "🔴 OVERVALUED / AVOID"
        badge_color = "#dc2626"
        verdict_summary = "Unfavorable risk-reward balance under current fundamentals and market price."

    positive_points: List[str] = []
    risk_points: List[str] = []

    t_pe = pe_peg_analysis.get("trailing_pe")
    h_pe = pe_peg_analysis.get("hist_pe")
    s_pe = pe_peg_analysis.get("sector_pe")

    if t_pe is not None:
        if h_pe and t_pe < h_pe * 0.92:
            positive_points.append(
                f"**P/E Multiple Discount vs. History ({t_pe:.1f}x vs {h_pe:.1f}x 5Y Median):** Trading at a discount to historical P/E band (Sector: {s_pe:.1f}x)."
            )
        elif h_pe and t_pe > h_pe * 1.20:
            risk_points.append(
                f"**Historical P/E Multiple Premium ({t_pe:.1f}x vs {h_pe:.1f}x 5Y Median):** Trades {pe_peg_analysis['vs_hist_pct']:+.1f}% above historical median P/E."
            )

    if peg_val is not None:
        if peg_val <= 1.35:
            positive_points.append(
                f"**Attractive Growth-Adjusted Valuation (PEG: {peg_val:.2f}x):** Trailing P/E is supported by projected growth ({base_case['g1_pct']:.1f}%)."
            )
        elif peg_val > 2.0:
            risk_points.append(
                f"**Stretched PEG Ratio ({peg_val:.2f}x):** Multiple is high relative to projected growth ({base_case['g1_pct']:.1f}%/yr)."
            )

    if qtr_analysis.get("available"):
        q_stat = qtr_analysis["turnaround_status"]
        q_sum = qtr_analysis["turnaround_summary"]
        if "TURNAROUND" in q_stat or "STRONG" in q_stat:
            positive_points.append(f"**Quarterly Momentum ({q_stat}):** {q_sum}")
        elif "CONTRACTION" in q_stat:
            risk_points.append(f"**Quarterly Earnings Drag ({q_stat}):** {q_sum}")

    if roe is not None and roe >= 15.0:
        positive_points.append(
            f"**Strong Shareholder Compounding (ROE: {roe:.1f}%):** Clears Buffett's 15% ROE moat hurdle (Sector: {roe_med or 15.0:.1f}%)."
        )
    elif roe is not None:
        risk_points.append(
            f"**Sub-Par Return on Equity (ROE: {roe:.1f}%):** Trails the 15% institutional benchmark."
        )

    if profile["roce_applicable"] and roce is not None:
        if roce >= 15.0:
            positive_points.append(
                f"**Superior Capital Allocation (ROCE: {roce:.1f}%):** Operating returns well above WACC ({base_case['discount_pct']:.1f}%)."
            )
        else:
            risk_points.append(f"**Low ROCE ({roce:.1f}%):** Operating capital returns below target 15%.")

    if profile["de_applicable"]:
        if de is None or de <= 0.5:
            positive_points.append(f"**Fortress Balance Sheet (D/E: {de or 0.0:.2f}x):** Minimal debt protects earnings.")
        elif de > 1.0:
            risk_points.append(f"**Elevated Balance Sheet Leverage (D/E: {de:.2f}x):** High debt creates cyclical vulnerability.")

    if mos is not None and mos >= 15.0:
        positive_points.append(
            f"**Attractive Margin of Safety ({mos:+.1f}%):** CMP trades below Blended Intrinsic Value ({format_inr_price(blended_iv)})."
        )
    elif mos is not None and mos < 0:
        risk_points.append(
            f"**Valuation Premium ({mos:+.1f}% MoS):** CMP trades above Base-Case Intrinsic Value ({format_inr_price(blended_iv)})."
        )

    if dcf_iv is not None and graham_iv is not None and graham_iv > dcf_iv * 1.6:
        risk_points.append(
            f"**DCF vs. Graham Cash-Conversion Gap:** Graham value ({format_inr_price(graham_iv)}) exceeds DCF ({format_inr_price(dcf_iv)}), signaling CapEx/working capital absorption."
        )

    return {
        "moat_grade": moat_grade,
        "moat_badge": moat_badge,
        "val_grade": val_grade,
        "fundamental_score_70": fundamental_score_70,
        "pe_peg_pts": pe_peg_pts,
        "sentiment_score_30": sentiment_score_30,
        "total_score": total_score,
        "verdict": verdict,
        "badge_color": badge_color,
        "verdict_summary": verdict_summary,
        "positive_points": positive_points,
        "risk_points": risk_points,
        "base_timeline_label": base_case.get("est_years_label", "N/A"),
        "cons_timeline_label": cons_case.get("est_years_label", "N/A"),
        "opt_timeline_label": opt_case.get("est_years_label", "N/A"),
    }


# ==============================================================================
# 10. STREAMLIT APP RUNNER (2-TAB ARCHITECTURE)
# ==============================================================================
def main() -> None:
    load_watchlist()

    with st.sidebar:
        st.header("🇮🇳 Indian Stock Selector")

        preset_options = ["Custom NSE/BSE Ticker..."] + [f"{k} — {v}" for k, v in INDIAN_PRESETS.items()]
        selected_preset = st.selectbox("Quick-Select Indian Stock", options=preset_options, index=1)

        if selected_preset != "Custom NSE/BSE Ticker...":
            default_ticker = selected_preset.split(" — ")[0].strip()
        else:
            default_ticker = "CGPOWER.NS"

        raw_ticker_input = st.text_input(
            "Enter Any Indian Stock Symbol",
            value=default_ticker,
            help="Type any NSE/BSE symbol (e.g., CGPOWER, TCS, SUZLON, CUPID). Appends .NS or .BO automatically.",
        )

        exchange_suffix = st.selectbox(
            "Default Indian Exchange (if suffix omitted)",
            options=[".NS (National Stock Exchange)", ".BO (Bombay Stock Exchange)"],
            index=0,
        ).split(" ")[0]

        cleaned_input = raw_ticker_input.strip().upper()
        resolved_ticker = f"{cleaned_input}{exchange_suffix}" if (cleaned_input and "." not in cleaned_input) else cleaned_input

        st.markdown("---")
        st.subheader("📐 Valuation & DCF Overrides")
        use_custom_params = st.checkbox("Override Automatic Growth & WACC Assumptions", value=False)
        custom_g1 = st.slider(
            "Base Stage-1 Growth Rate g₁ (Years 1–5, %)",
            min_value=0.0, max_value=25.0, value=14.0, step=0.5,
            disabled=not use_custom_params,
        )
        custom_discount = st.slider(
            "Base Discount Rate / WACC r (%)",
            min_value=8.0, max_value=18.0, value=12.0, step=0.25,
            disabled=not use_custom_params,
        )
        custom_term_g = st.slider(
            "Base Terminal Growth Rate g_term (Year 6+, %)",
            min_value=2.0, max_value=6.0, value=4.0, step=0.25,
            disabled=not use_custom_params,
        )
        st.markdown("---")
        st.subheader("🛡️ Exit Strategy & Sizing Rules")
        current_weight_input = st.slider(
            "Your Current Portfolio Weight in This Stock (%)",
            min_value=0.0, max_value=40.0, value=6.0, step=0.5,
            help="If this stock grows beyond your Max Cap, Trigger 3 (Rebalance) will alert you to trim excess."
        )
        max_cap_input = st.slider(
            "Hard Single-Stock Concentration Cap (%)",
            min_value=10.0, max_value=25.0, value=15.0, step=1.0,
        )
        alt_irr_input = st.slider(
            "Best Alternative Watchlist Idea IRR (%)",
            min_value=10.0, max_value=30.0, value=16.0, step=0.5,
            help="Used for Trigger 4: Checks if an alternative stock beats this holding by >= 8%/yr (to overcome 12.5% LTCG tax)."
        )

        st.markdown("---")
        st.subheader("🏢 Peer Benchmark & CSV")
        custom_peers_input = st.text_input(
            "Custom Indian Peer Symbols (optional)",
            value="",
            placeholder="e.g., SIEMENS.NS, ABB.NS, BHEL.NS",
        )
        uploaded_csv = st.file_uploader("Optional: Upload Custom Financials CSV", type=["csv"])

        analyze_clicked = st.button("🔍 Analyze Indian Equity", type="primary", use_container_width=True)
        if st.button("🔄 Clear Cache & Force Fresh Fetch", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = resolved_ticker
    if analyze_clicked or resolved_ticker != st.session_state.get("last_resolved_input"):
        st.session_state["active_ticker"] = resolved_ticker
        st.session_state["last_resolved_input"] = resolved_ticker

    target_ticker = st.session_state["active_ticker"]

    # --------------------------------------------------------------------------
    # TWO-TAB ARCHITECTURE: TAB 1 (VALUATION DASHBOARD) & TAB 2 (SAVED LIST)
    # --------------------------------------------------------------------------
    tab_analyzer, tab_saved = st.tabs([
        "📊 Live Institutional Valuation Engine",
        f"🗂️ Comparison Watchlist & History ({len(st.session_state.saved_analyses)})",
    ])

    with tab_analyzer:
        st.title("🇮🇳 Indian Market Institutional Value, P/E-PEG & DCF Analyzer")
        st.markdown(
            "**Multi-Source Engine (`Screener.in` + `Yahoo Finance`)** • "
            "**Dual-Axis Moat vs Valuation Grading** • **Reverse DCF Implied Growth** • **Staggered Accumulation Zones**"
        )

        with st.spinner(f"Analyzing fundamentals, P/E bands, and moat checks for {target_ticker}..."):
            try:
                bundle = fetch_company_bundle(target_ticker)
                metrics = extract_comprehensive_metrics(bundle)
                qtr_analysis = extract_quarterly_trend_analysis(bundle)
                checklist_result = evaluate_sector_checklist(bundle, metrics)
                df_peers, peer_medians = fetch_industry_peer_benchmarks(
                    target_ticker=target_ticker,
                    archetype=bundle["profile"]["archetype"],
                    custom_peers_str=custom_peers_input,
                )
                dcf_inputs = extract_dcf_inputs(bundle, metrics)
                applied_g1 = custom_g1 if use_custom_params else dcf_inputs["base_g1_pct"]
                applied_r = custom_discount if use_custom_params else dcf_inputs["base_discount_pct"]
                applied_term_g = custom_term_g if use_custom_params else dcf_inputs["base_terminal_g_pct"]

                pe_peg_analysis = extract_pe_peg_analysis(
                    current_price=bundle["current_price"],
                    eps_ttm=dcf_inputs["eps_ttm"],
                    g1_pct=applied_g1,
                    metrics=metrics,
                    peer_medians=peer_medians,
                )
            except ValueError as ve:
                st.error(f"**Ticker Lookup Error:** {ve}")
                st.stop()
            except Exception as exc:
                st.error(f"**Data Connectivity Issue:** `{exc}`. Tap 'Clear Cache & Force Fresh Fetch' in sidebar.")
                st.stop()

        info = bundle["info"]
        profile = bundle["profile"]
        current_price = bundle["current_price"]
        company_name = info.get("longName") or info.get("shortName") or target_ticker
        mcap = safe_float(info.get("marketCap")) or bundle.get("fast_mcap")
        scr_ok = bundle.get("screener", {}).get("available", False)

        st.subheader(f"{company_name} (`{target_ticker}`)")
        st.caption(
            f"**Industry Profile:** {profile['profile_name']} | **Sector:** {bundle['sector']} ({bundle['industry']}) | "
            f"**Current Price:** {format_inr_price(current_price)} | "
            f"**Market Cap:** {format_inr_crores(mcap)} | "
            f"**Data Feed:** `Yahoo Finance` + `{'Screener.in Consolidated' if scr_ok else 'NSE Median Fallback'}`"
        )
        st.info(f"🧭 **Sector Routing:** {profile['applicable_note']}")

        if uploaded_csv is not None:
            try:
                user_df = pd.read_csv(uploaded_csv)
                with st.expander("📂 View Uploaded Custom Financial CSV", expanded=True):
                    st.dataframe(user_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse uploaded CSV: {e}")

        st.divider()

        # SECTION 1: KPIS & TRENDS
        st.header("1. Business Health, Capital Efficiency (ROE & ROCE) & Quarterly Turnaround")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            roe_v = metrics["roe_pct"]
            roe_med_v = peer_medians.get("roe_med")
            roe_delta = f"vs {roe_med_v:.1f}% Sector" if (roe_v is not None and roe_med_v is not None) else "Target >= 15%"
            st.metric(
                label="Return on Equity (ROE)",
                value=f"{roe_v:.2f}%" if roe_v is not None else "N/A",
                delta=roe_delta,
                delta_color="normal" if (roe_v is not None and roe_v >= (roe_med_v or 15.0)) else "inverse",
            )
        with k2:
            roce_v = metrics["roce_pct"]
            roce_med_v = peer_medians.get("roce_med")
            if profile["roce_applicable"]:
                roce_delta = f"vs {roce_med_v:.1f}% Sector" if (roce_v is not None and roce_med_v is not None) else "Target >= 15%"
                st.metric(
                    label="Return on Capital (ROCE)",
                    value=f"{roce_v:.2f}%" if roce_v is not None else "N/A",
                    delta=roce_delta,
                    delta_color="normal" if (roce_v is not None and roce_v >= (roce_med_v or 15.0)) else "inverse",
                )
            else:
                st.metric(label="ROCE", value="N/A (Bank)", delta="See ROA & ROE", delta_color="off")
        with k3:
            roa_v = metrics["roa_pct"]
            roa_target = 1.2 if profile["is_financial"] else 7.0
            st.metric(
                label="Return on Assets (ROA)",
                value=f"{roa_v:.2f}%" if roa_v is not None else "N/A",
                delta=f"Target >= {roa_target:.1f}%",
                delta_color="normal" if (roa_v is not None and roa_v >= roa_target) else "off",
            )
        with k4:
            de_v = metrics["de_ratio"]
            if profile["de_applicable"]:
                st.metric(
                    label="Debt-to-Equity (D/E)",
                    value=f"{de_v:.2f}x" if de_v is not None else "0.00x",
                    delta="Safe <= 0.8x" if (de_v is None or de_v <= 0.8) else "High Leverage",
                    delta_color="normal" if (de_v is None or de_v <= 0.8) else "inverse",
                )
            else:
                pb_v = metrics["pb_ratio"]
                st.metric(label="Price-to-Book (P/B)", value=f"{pb_v:.2f}x" if pb_v is not None else "N/A", delta="Bank Multiple", delta_color="off")
        with k5:
            pe_v = pe_peg_analysis["trailing_pe"]
            h_pe_v = pe_peg_analysis["hist_pe"]
            pe_delta = f"5Y Hist Med: {h_pe_v:.1f}x" if h_pe_v is not None else None
            st.metric(
                label="Trailing P/E Multiple",
                value=f"{pe_v:.2f}x" if pe_v is not None else "N/A",
                delta=pe_delta,
                delta_color="normal" if (pe_v is not None and h_pe_v is not None and pe_v <= h_pe_v) else "inverse",
            )
        with k6:
            peg_v = pe_peg_analysis["peg_ratio"]
            st.metric(
                label="PEG Ratio (P/E ÷ g₁)",
                value=f"{peg_v:.2f}x" if peg_v is not None else "N/A",
                delta="Target <= 1.5x" if (peg_v is not None and peg_v <= 1.5) else "Stretched > 1.5x" if peg_v is not None else None,
                delta_color="normal" if (peg_v is not None and peg_v <= 1.5) else "inverse",
            )

        st.markdown("#### ⚡ Last 4 Quarters Results Trend & Early Turnaround Scanner")
        if qtr_analysis["available"]:
            q_color = qtr_analysis["turnaround_badge_color"]
            st.markdown(
                f"""
                <div style="background-color: #1e293b; border-left: 6px solid {q_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                    <strong style="color: {q_color}; font-size: 1.0rem;">{qtr_analysis['turnaround_status']}</strong>
                    <span style="color: #94a3b8; font-size: 0.85rem;"> &nbsp;•&nbsp; Source: {qtr_analysis['source_used']}</span><br>
                    <span style="color: #e2e8f0; font-size: 0.94rem;">{qtr_analysis['turnaround_summary']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(qtr_analysis["df_quarterly"], use_container_width=True)
        else:
            st.info(qtr_analysis["turnaround_summary"])

        st.markdown("#### 📊 Multi-Year Annual Financial Performance (₹ Crores)")
        hist_df = build_three_year_history_table(bundle)
        if not hist_df.empty:
            st.dataframe(hist_df, use_container_width=True)

        st.divider()

        # SECTION 2: CHECKLIST
        st.header("2. Sector Moat Checklist")
        chk_col1, chk_col2 = st.columns([2, 1])
        with chk_col1:
            st.markdown("#### 🤖 Automated Quantitative Sector Rules")
            df_chk = pd.DataFrame(checklist_result["quant_checks"]).set_index("Parameter")
            st.dataframe(df_chk, use_container_width=True)

        with chk_col2:
            st.markdown("#### ☑️ Qualitative Moat Confirmation")
            st.caption("Tick each check to add conviction points:")
            user_confirmed_count = 0
            for idx, q_text in enumerate(checklist_result["qual_questions"]):
                if st.checkbox(q_text, key=f"qual_{target_ticker}_{idx}", value=False):
                    user_confirmed_count += 1

            pass_c = checklist_result["pass_count"]
            tot_c = checklist_result["total_checks"]
            qual_tot = len(checklist_result["qual_questions"])
            badge_bg = "#16a34a" if pass_c >= 3 else ("#d97706" if pass_c == 2 else "#dc2626")
            st.markdown(
                f"""
                <div style="background-color: #1e293b; border-left: 6px solid {badge_bg}; padding: 12px 14px; border-radius: 6px; margin-top: 10px;">
                    <strong style="color: #f8fafc;">Automated Pass Rate: <span style="color: {badge_bg};">{pass_c} / {tot_c} Passed</span></strong><br>
                    <span style="color: #cbd5e1; font-size: 0.88rem;">Qualitative Confirmed: {user_confirmed_count} / {qual_tot}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # SECTION 3: MULTIPLES & BENCHMARKS
        st.header("3. Growth Acceleration, EBITDA Margins, P/E-PEG & Peers")
        bb_verdict = metrics["buyback_verdict"]
        bb_color = "#16a34a" if bb_verdict == "100% ORGANIC EPS GROWTH" else ("#d97706" if bb_verdict == "BUYBACK-BOOSTED EPS" else "#dc2626")
        st.markdown(
            f"""
            <div style="background-color: #1e293b; border-left: 6px solid {bb_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px;">
                <strong style="color: {bb_color}; font-size: 1.02rem;">🔎 Forensic Capital Allocation: {bb_verdict}</strong><br>
                <span style="color: #e2e8f0; font-size: 0.95rem;">{metrics['buyback_explanation']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def fmt_pct(val: Optional[float]) -> str:
            return f"{val:+.2f}%" if val is not None else "N/A"

        def fmt_level_pct(val: Optional[float]) -> str:
            return f"{val:.2f}%" if val is not None else "N/A"

        rev_yoy, rev_cagr = metrics["rev_growth"]["recent_yoy_pct"], metrics["rev_growth"]["past_cagr_pct"]
        em_curr, em_past = metrics["ebitda_margin_curr"], metrics["ebitda_margin_past"]
        nm_curr, nm_past = metrics["net_margin_curr"], metrics["net_margin_past"]
        eps_yoy, eps_cagr = metrics["eps_growth"]["recent_yoy_pct"], metrics["eps_growth"]["past_cagr_pct"]

        em_momentum = f"🟢 Expanding ({em_curr - em_past:+.2f}%)" if (profile["ebitda_applicable"] and em_curr is not None and em_past is not None and em_curr >= em_past) else "N/A"
        nm_momentum = f"🟢 Expanding ({nm_curr - nm_past:+.2f}%)" if (nm_curr is not None and nm_past is not None and nm_curr >= nm_past) else "N/A"

        # Unpack P/E, sector P/E, and PEG variables from pe_peg_analysis
        t_pe_v = pe_peg_analysis["trailing_pe"]
        h_pe_v = pe_peg_analysis["hist_pe"]
        s_pe_v = pe_peg_analysis["sector_pe"]
        peg_v = pe_peg_analysis["peg_ratio"]
        pe_v = t_pe_v  # fallback alias to prevent any pe_v references from throwing NameError


        comparison_rows = [
            {
                "Metric": "Trailing P/E Ratio",
                "Current / Recent": f"{pe_v:.2f}x" if pe_v is not None else "N/A",
                "Own 3–5Yr Historical Benchmark": f"{h_pe_v:.2f}x" if h_pe_v is not None else "N/A",
                "Momentum / Historical Band": (
                    f"🟢 {pe_peg_analysis['vs_hist_pct']:+.1f}% vs Hist Med"
                    if (pe_peg_analysis["vs_hist_pct"] is not None and pe_peg_analysis["vs_hist_pct"] <= 0)
                    else (f"🔴 {pe_peg_analysis['vs_hist_pct']:+.1f}% vs Hist Med" if pe_peg_analysis["vs_hist_pct"] is not None else "N/A")
                ),
                "NSE Industry Median": f"{s_pe_v:.2f}x" if s_pe_v is not None else "N/A",
            },
            {
                "Metric": "PEG Ratio",
                "Current / Recent": f"{peg_v:.2f}x" if peg_v is not None else "N/A",
                "Own 3–5Yr Historical Benchmark": "Lynch Rule <= 1.2x",
                "Momentum / Historical Band": pe_peg_analysis["peg_status"],
                "NSE Industry Median": "1.50x (GARP)",
            },
            {
                "Metric": "Revenue Growth",
                "Current / Recent": fmt_pct(rev_yoy),
                "Own 3–5Yr Historical Benchmark": fmt_pct(rev_cagr),
                "Momentum / Historical Band": "🟢 Accelerating" if (rev_yoy is not None and rev_cagr is not None and rev_yoy >= rev_cagr) else "🟡 Steady",
                "NSE Industry Median": fmt_pct(peer_medians.get("rev_growth_med")),
            },
            {
                "Metric": "EBITDA Margin",
                "Current / Recent": fmt_level_pct(em_curr) if profile["ebitda_applicable"] else "N/A",
                "Own 3–5Yr Historical Benchmark": fmt_level_pct(em_past) if profile["ebitda_applicable"] else "N/A",
                "Momentum / Historical Band": em_momentum,
                "NSE Industry Median": fmt_level_pct(peer_medians.get("ebitda_margin_med")) if profile["ebitda_applicable"] else "N/A",
            },
            {
                "Metric": "Net Margin",
                "Current / Recent": fmt_level_pct(nm_curr),
                "Own 3–5Yr Historical Benchmark": fmt_level_pct(nm_past),
                "Momentum / Historical Band": nm_momentum,
                "NSE Industry Median": fmt_level_pct(peer_medians.get("net_margin_med")),
            },
        ]
        st.dataframe(pd.DataFrame(comparison_rows).set_index("Metric"), use_container_width=True)

        if not df_peers.empty:
            with st.expander("🏢 View Indian Sector Peer Comparison Table", expanded=False):
                st.dataframe(df_peers.set_index("Ticker").round(2), use_container_width=True)

        st.divider()

        # SECTION 4: 3-MODEL VALUATION MATRIX
        st.header("4. Intrinsic Value & 3-Scenario Matrix")
        scenarios = build_three_scenarios(
            dcf_inputs=dcf_inputs,
            base_g1_pct=applied_g1,
            base_discount_pct=applied_r,
            base_terminal_g_pct=applied_term_g,
            fair_base_pe=pe_peg_analysis["fair_base_pe"],
            current_price=current_price,
        )
        base_res = scenarios["Base Case"]

        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        with vc1:
            st.metric("Current Market Price", format_inr_price(current_price))
        with vc2:
            dcf_mos_val = base_res["dcf_mos_pct"]
            st.metric(
                "1. 2-Stage DCF Value",
                format_inr_price(base_res["dcf_intrinsic"]),
                delta=f"{dcf_mos_val:+.1f}% MoS" if dcf_mos_val is not None else "N/A",
                delta_color="normal" if (dcf_mos_val or -1) > 0 else "inverse",
            )
        with vc3:
            graham_mos_val = base_res["graham_mos_pct"]
            st.metric(
                "2. Graham Formula Value",
                format_inr_price(base_res["graham_intrinsic"]),
                delta=f"{graham_mos_val:+.1f}% MoS" if graham_mos_val is not None else "N/A",
                delta_color="normal" if (graham_mos_val or -1) > 0 else "inverse",
            )
        with vc4:
            pe_mos_val = base_res["pe_mos_pct"]
            st.metric(
                f"3. Fair P/E Target ({base_res['fair_pe_mult']:.1f}x)",
                format_inr_price(base_res["pe_intrinsic"]),
                delta=f"{pe_mos_val:+.1f}% MoS" if pe_mos_val is not None else "N/A",
                delta_color="normal" if (pe_mos_val or -1) > 0 else "inverse",
            )
        with vc5:
            blended_mos_val = base_res["blended_mos_pct"]
            st.metric(
                "3-Model Blended Value",
                format_inr_price(base_res["blended_intrinsic"]),
                delta=f"{blended_mos_val:+.1f}% Blended MoS" if blended_mos_val is not None else "N/A",
                delta_color="normal" if (blended_mos_val or -1) > 0 else "inverse",
            )

        scenario_rows = []
        for scen_label in ["Conservative", "Base Case", "Optimistic"]:
            s = scenarios[scen_label]
            s_mos = s["blended_mos_pct"]
            scenario_rows.append(
                {
                    "Scenario": scen_label,
                    "Stage-1 Growth (g₁)": f"{s['g1_pct']:.2f}%",
                    "Discount Rate (r)": f"{s['discount_pct']:.2f}%",
                    "Target P/E Multiple": f"{s['fair_pe_mult']:.1f}x",
                    "2-Stage DCF / Share": format_inr_price(s["dcf_intrinsic"]),
                    "Graham Value / Share": format_inr_price(s["graham_intrinsic"]),
                    "Fair P/E Value / Share": format_inr_price(s["pe_intrinsic"]),
                    "3-Model Blended Value": format_inr_price(s["blended_intrinsic"]),
                    "Blended Margin of Safety": f"{s_mos:+.2f}%" if s_mos is not None else "N/A",
                    "Tentative Timeline to Reach IV": s["est_years_label"],
                }
            )
        st.dataframe(pd.DataFrame(scenario_rows).set_index("Scenario"), use_container_width=True)

        st.divider()

    # SECTION 5: UPGRADED INSTITUTIONAL VERDICT, REVERSE DCF & BUY ZONES
    st.header("5. Institutional Decision Engine & Staggered Buy Zones")
    news_summary = analyze_corporate_news(bundle["raw_news"], max_items=7)

    decision = compute_decision_verdict(
        ticker=target_ticker,
        company_name=company_name,
        current_price=current_price,
        profile=profile,
        metrics=metrics,
        qtr_analysis=qtr_analysis,
        pe_peg_analysis=pe_peg_analysis,
        peer_medians=peer_medians,
        scenarios=scenarios,
        news_summary=news_summary,
        checklist_result=checklist_result,
        user_confirmed_count=user_confirmed_count,
    )

    # --- CALCULATE FCF / NET INCOME RATIO AS A STANDALONE STATEMENT ---
    fcf_ni_val = metrics.get("fcf_to_ni_ratio") or metrics.get("ocf_to_ni_ratio")
    fcf_ni_pct = round(fcf_ni_val * 100.0, 1) if fcf_ni_val is not None else 0.0

    # --- COMPUTE & RENDER EXIT STRATEGY PLAYBOOK ---
    exit_plan = evaluate_exit_and_trim_strategy(
        cmp=current_price,
        moat_grade=decision["moat_grade"],
        roce_pct=metrics["roce_pct"],
        wacc_pct=applied_r,
        fcf_ni_pct=fcf_ni_pct,
        de_ratio=metrics["de_ratio"],
        qtr_status=qtr_analysis.get("turnaround_status", ""),
        implied_g1_pct=implied_g1_pct,
        model_g1_pct=applied_g1,
        peg_ratio=pe_peg_analysis.get("peg_ratio"),
        rev_yoy_pct=metrics["rev_growth"]["recent_yoy_pct"],
        blended_mos_pct=base_res["blended_mos_pct"],
        roce_applicable=profile["roce_applicable"],
        de_applicable=profile["de_applicable"],
        current_portfolio_weight_pct=current_weight_input,
        max_single_stock_cap_pct=max_cap_input,
        replacement_irr_pct=alt_irr_input,
    )

    st.markdown("#### 🎯 Automated Exit Strategy & Position Management Playbook (For Existing Holders)")
    st.markdown(
        f"""
        <div style="
            background-color: #1e293b;
            border-left: 8px solid {exit_plan['exit_color']};
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 14px;
        ">
            <h4 style="color: {exit_plan['exit_color']}; margin: 0 0 6px 0;">
                {exit_plan['exit_verdict']}
            </h4>
            <span style="color: #94a3b8; font-size: 0.90rem;">
                • <b>Est. Holding 5-Yr IRR:</b> <code>{exit_plan['est_holding_irr']:.1f}%/yr</code> &nbsp;|&nbsp;
                • <b>Alternative Idea Spread:</b> <code>{exit_plan['irr_spread']:+.1f}%/yr</code> (Hurdle >= +8.0% for 12.5% LTCG) &nbsp;|&nbsp;
                • <b>Your Allocation:</b> <code>{current_weight_input:.1f}%</code> (Cap: {max_cap_input:.1f}%)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.markdown("**🔍 Diagnostic Trigger Audit:**")
        for tf in exit_plan["triggers_fired"]:
            st.markdown(f"- {tf}")
    with ex_col2:
        st.markdown("**📋 Anti-Regret Tranche Execution Plan:**")
        for tp in exit_plan["tranche_plan"]:
            st.markdown(f"- {tp}")

    default_ks = (
        f"Exit 100% if ROCE drops below {applied_r:.1f}% (WACC), "
        f"FCF/PAT falls < 45%, or D/E exceeds 1.0x"
    )
    kill_switch_remark = st.text_input(
        "🛡️ Thesis Invalidation Kill-Switch (Auto-generated or customize before saving to Tab 2):",
        value=default_ks,
        key=f"ks_input_{target_ticker}",
    )

        # REVERSE DCF & BUY ZONES COMPUTATION
    fcf_per_share_calc = (dcf_inputs["fcf_0"] / dcf_inputs["shares"]) if (dcf_inputs.get("fcf_0") and dcf_inputs.get("shares") and dcf_inputs["shares"] > 0) else (dcf_inputs.get("eps_ttm") or 1.0)
    net_cash_per_share = ((dcf_inputs["total_cash"] - dcf_inputs["total_debt"]) / dcf_inputs["shares"]) if (dcf_inputs.get("shares") and dcf_inputs["shares"] > 0) else 0.0

    implied_g1_pct = calculate_reverse_dcf_growth(
            cmp=current_price,
            base_cf_per_share=fcf_per_share_calc,
            wacc=applied_r / 100.0,
            terminal_g=applied_term_g / 100.0,
            stage1_years=10,
            net_cash_per_share=net_cash_per_share,
        )

    buy_zones = calculate_staggered_buy_zones(
            cmp=current_price,
            dcf_intrinsic=base_res["dcf_intrinsic"],
            graham_val=base_res["graham_intrinsic"],
            eps_ttm=dcf_inputs["eps_ttm"],
            expected_g1_pct=applied_g1,
            moat_grade=decision["moat_grade"],
        )
    valuation_anchor_price = buy_zones["valuation_anchor"]


        # RENDER DUAL-AXIS VERDICT CARD
    st.markdown(
            f"""
            <div style="
                background-color: #0f172a;
                border-left: 8px solid {decision['badge_color']};
                padding: 20px 24px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h2 style="color: #f8fafc; margin: 0 0 8px 0; font-weight: 800;">
                    INSTITUTIONAL VERDICT:
                    <span style="color: {decision['badge_color']};">{decision['verdict']}</span>
                </h2>
                <p style="color: #cbd5e1; font-size: 1.02rem; margin: 0 0 8px 0;">{decision['verdict_summary']}</p>
                <p style="color: #38bdf8; font-size: 0.95rem; font-weight: 600; margin: 0;">
                    ⏳ Timeline to Reach Base-Case Blended IV ({format_inr_price(base_res['blended_intrinsic'])}):
                    <span style="color: #f8fafc;">{decision['base_timeline_label']}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # METRIC GAUGES
    roce_val = metrics["roce_pct"] or 0.0
    roe_val = metrics["roe_pct"] or 0.0
    roce_roe_spread = round(roce_val - roe_val, 2)
    fcf_ni_val = metrics.get("fcf_to_ni_ratio")
    fcf_ni_pct = round(fcf_ni_val * 100.0, 1) if fcf_ni_val is not None else 0.0

    vg1, vg2, vg3 = st.columns(3)
    vg1.metric("Business Moat Quality Grade", decision["moat_grade"], decision["moat_badge"])
    vg2.metric("Valuation Price Grade", decision["val_grade"], f"PEG: {pe_peg_analysis['peg_ratio'] or 0:.2f}x | MoS: {base_res['blended_mos_pct'] or 0:.1f}%")
    vg3.metric(
            "ROCE-to-ROE Spread",
            f"{roce_roe_spread:+.2f}%",
            "Low Financial Leverage / Pure Moat" if roce_roe_spread >= 0 else "Leverage Assisted ROE",
        )

    st.markdown("#### 🔄 Automated Reverse DCF & Staggered Buy-Zone Accumulation Tranches")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)

    growth_gap = round(implied_g1_pct - applied_g1, 2)
    r_col1.metric(
            "Market-Implied Growth (Reverse DCF g₁)",
            f"{implied_g1_pct:.2f}% / yr",
            f"{growth_gap:+.2f}% vs Model ({applied_g1:.1f}%)",
            delta_color="inverse",
        )
    r_col2.metric(
            "🟢 Aggressive SIP Zone",
            f"₹{buy_zones['sip_zone'][0]:,.1f} – ₹{buy_zones['sip_zone'][1]:,.1f}",
            "Tranche 1 (Quality Anchor)",
        )
    r_col3.metric(
            "🔵 GARP Fair-Multiple Zone",
            f"₹{buy_zones['garp_zone'][0]:,.1f} – ₹{buy_zones['garp_zone'][1]:,.1f}",
            "Tranche 2 (Core Position)",
        )
    r_col4.metric(
            "🟣 Deep Value Graham Zone",
            f"₹{buy_zones['deep_value_zone'][0]:,.1f} – ₹{buy_zones['deep_value_zone'][1]:,.1f}",
            "Tranche 3 (Max Safety)",
        )
    # SAVE SNAPSHOT ACTION SECTION
    st.divider()
    st.markdown("#### 💾 Save / Update Analysis to Comparison Ledger")
    s_col1, s_col2 = st.columns([2, 1])
    
    with s_col1:
        overwrite_mode = st.checkbox(
            f"🔄 Replace existing saved entry for {target_ticker} (uncheck to log historical date snapshot)",
            value=True,
            key=f"overwrite_{target_ticker}",
        )
        
    with s_col2:
        if st.button("💾 Save Stock to Comparison Watchlist", use_container_width=True, type="primary"):
            snapshot = {
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Ticker": target_ticker,
                "Company": company_name,
                "Moat Grade": decision["moat_grade"],
                "Valuation Grade": decision["val_grade"],
                "CMP (₹)": round(current_price, 2),
                "Implied Growth Rate (g₁ %)": round(implied_g1_pct, 2),
                "Model Growth (g₁ %)": round(applied_g1, 2),
                "Assumed WACC (r %)": round(applied_r, 2),  
                "Valuation Anchor Price (₹)": round(valuation_anchor_price, 2),
                "Calculated Intrinsic Value (₹)": round(base_res["blended_intrinsic"] or base_res["dcf_intrinsic"] or 0.0, 2),
                "ROCE-to-ROE Spread (%)": round(roce_roe_spread, 2),
                "FCF/Net Income Ratio (%)": round(fcf_ni_pct, 2),
                "SIP Buy Zone (₹)": f"₹{buy_zones['sip_zone'][0]:,.0f} - ₹{buy_zones['sip_zone'][1]:,.0f}",
                "GARP Buy Zone (₹)": f"₹{buy_zones['garp_zone'][0]:,.0f} - ₹{buy_zones['garp_zone'][1]:,.0f}",
                "Deep Value Zone (₹)": f"₹{buy_zones['deep_value_zone'][0]:,.0f} - ₹{buy_zones['deep_value_zone'][1]:,.0f}",
                "Verdict": decision["verdict"],
                "Exit / Holding Action": exit_plan["exit_verdict"],
                "Thesis Kill-Switch": kill_switch_remark,
            }
            status_msg = upsert_stock_analysis(snapshot, replace_existing=overwrite_mode)
            st.success(status_msg)

    # POSITIVES & RISKS
    st.divider()
    pos_col, risk_col = st.columns(2)
    with pos_col:
    st.markdown("### 🟢 Positive Points & Bullish Catalysts")
    for pt in decision["positive_points"]:
    st.markdown(f"- {pt}")
    with risk_col:
    st.markdown("### 🔴 Key Risks & Red Flags to Consider")
    for rk in decision["risk_points"]:
    st.markdown(f"- {rk}")

    # --------------------------------------------------------------------------
    # TAB 2: SAVED COMPARISON WATCHLIST LEDGER
    # --------------------------------------------------------------------------
    with tab_saved:
        st.subheader("🗂️ Saved Stock Valuation Ledger & Historical Comparison")
        st.caption(
            "Track and compare saved stock snapshots across Implied Growth Rate (g₁), Valuation Anchor Price, "
            "Calculated Intrinsic Value, ROCE-to-ROE Spread, and FCF/Net Income Ratio."
        )

        saved_records = load_watchlist()

        if not saved_records:
            st.info("No stocks saved yet. Open Tab 1, run an analysis, and click **💾 Save Stock to Comparison Watchlist**.")
        else:
            df_saved = pd.DataFrame(saved_records)
            priority_cols = [
                "Date",
                "Ticker",
                "Moat Grade",
                "CMP (₹)",
                "Implied Growth Rate (g₁ %)",
  "Assumed WACC (r %)", 
                "Valuation Anchor Price (₹)",
                "Calculated Intrinsic Value (₹)",
                "ROCE-to-ROE Spread (%)",
                "FCF/Net Income Ratio (%)",
    "Exit / Holding Action",             # <--- ADD THIS
                "Thesis Kill-Switch", 
                "SIP Buy Zone (₹)",
                "GARP Buy Zone (₹)",
                "Deep Value Zone (₹)",
                "Verdict",
            ]
            display_cols = [c for c in priority_cols if c in df_saved.columns]

            st.dataframe(
                df_saved[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Implied Growth Rate (g₁ %)": st.column_config.NumberColumn("Implied Growth (g₁)", format="%.2f%%"),
                    "Valuation Anchor Price (₹)": st.column_config.NumberColumn("Valuation Anchor", format="₹%.2f"),
                    "Calculated Intrinsic Value (₹)": st.column_config.NumberColumn("Intrinsic Value", format="₹%.2f"),
                    "ROCE-to-ROE Spread (%)": st.column_config.NumberColumn("ROCE-ROE Spread", format="%+.2f%%"),
                    "FCF/Net Income Ratio (%)": st.column_config.NumberColumn("FCF / Net Income", format="%.1f%%"),
                    "CMP (₹)": st.column_config.NumberColumn("Saved CMP", format="₹%.2f"),
                },
            )

            st.divider()
            c_left, c_mid, c_right = st.columns([2, 1, 1])

            with c_left:
                tickers_list = df_saved["Ticker"].unique().tolist()
                del_choice = st.selectbox(
                    "Select a stock snapshot to delete before replacing with fresh data:",
                    options=["-- Select Ticker --"] + tickers_list,
                )
                if del_choice != "-- Select Ticker --":
                    if st.button(f"🗑️ Delete {del_choice} Entry", key="del_single_btn"):
                        remaining = [x for x in saved_records if x.get("Ticker") != del_choice]
                        save_watchlist_to_disk(remaining)
                        st.rerun()

            with c_mid:
                csv_export = df_saved.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Ledger (CSV)",
                    data=csv_export,
                    file_name=f"indian_stocks_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with c_right:
                if st.button("♻️ Reset Entire Watchlist", type="secondary", use_container_width=True):
                    reset_entire_watchlist()
                    st.success("Watchlist wiped clean.")
                    st.rerun()


if __name__ == "__main__":
    main()