import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ==============================================================================
# 0. PAGE CONFIGURATION, PRESETS & SECTOR PEER MAPS
# ==============================================================================
st.set_page_config(
    page_title="Buffett Institutional Value & Forensic Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDIAN_PRESETS = {
    "TCS.NS": "Tata Consultancy Services (IT)",
    "INFY.NS": "Infosys Ltd (IT)",
    "HDFCBANK.NS": "HDFC Bank Ltd (Banking)",
    "BAJFINANCE.NS": "Bajaj Finance Ltd (NBFC)",
    "RELIANCE.NS": "Reliance Industries Ltd (Energy/Retail)",
    "ITC.NS": "ITC Ltd (FMCG)",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd (FMCG)",
    "MARUTI.NS": "Maruti Suzuki India Ltd (Auto)",
    "EICHERMOT.NS": "Eicher Motors Ltd (Auto)",
    "LT.NS": "Larsen & Toubro Ltd (Industrials)",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries (Pharma)",
    "TITAN.NS": "Titan Company Ltd (Consumer)",
}

US_PRESETS = {
    "AAPL": "Apple Inc. (Technology)",
    "GOOG": "Alphabet Inc. (Communication/Tech)",
    "MSFT": "Microsoft Corporation (Technology)",
    "BRK-B": "Berkshire Hathaway Inc. (Financials)",
    "JPM": "JPMorgan Chase & Co. (Banking)",
    "NVDA": "NVIDIA Corporation (Semiconductors)",
    "JNJ": "Johnson & Johnson (Healthcare)",
    "PG": "Procter & Gamble Co. (Consumer Defensive)",
    "KO": "The Coca-Cola Company (Beverages)",
    "XOM": "Exxon Mobil Corporation (Energy)",
}

# Sector-to-Peer mapping for automated Industry Benchmarking
INDIA_SECTOR_PEERS = {
    "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "Financial Services": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS"],
    "Consumer Defensive": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Consumer Cyclical": ["MARUTI.NS", "EICHERMOT.NS", "TITAN.NS", "M&M.NS"],
    "Industrials": ["LT.NS", "SIEMENS.NS", "ABB.NS", "BEL.NS"],
    "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
    "Basic Materials": ["TATASTEEL.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "HINDALCO.NS"],
    "Communication Services": ["BHARTIARTL.NS", "TCS.NS", "INFY.NS"],
}

US_SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "ORCL"],
    "Communication Services": ["GOOG", "META", "NFLX", "DIS"],
    "Financial Services": ["BRK-B", "JPM", "BAC", "V"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT"],
    "Consumer Cyclical": ["AMZN", "HD", "MCD", "NKE"],
    "Healthcare": ["JNJ", "LLY", "UNH", "MRK"],
    "Industrials": ["CAT", "GE", "HON", "UNP"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
}


@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer() -> Optional[SentimentIntensityAnalyzer]:
    """Initializes and caches the NLTK VADER sentiment engine."""
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


# ==============================================================================
# 1. CURRENCY, FORMATTING & SAFE EXTRACTION UTILITIES
# ==============================================================================
def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except (ValueError, TypeError):
        return default


def get_currency_symbol(currency_code: str) -> str:
    mapping = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return mapping.get((currency_code or "USD").upper(), "$")


def format_price(value: Optional[float], currency_code: str = "USD") -> str:
    if value is None or np.isnan(value):
        return "N/A"
    sym = get_currency_symbol(currency_code)
    sign = "-" if value < 0 else ""
    return f"{sign}{sym}{abs(value):,.2f}"


def format_large_currency(value: Optional[float], currency_code: str = "USD") -> str:
    if value is None or np.isnan(value):
        return "N/A"
    sym = get_currency_symbol(currency_code)
    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if (currency_code or "").upper() == "INR":
        if abs_val >= 1e7:
            return f"{sign}{sym}{abs_val / 1e7:,.2f} Cr"
        if abs_val >= 1e5:
            return f"{sign}{sym}{abs_val / 1e5:,.2f} L"
        return f"{sign}{sym}{abs_val:,.0f}"

    if abs_val >= 1e12:
        return f"{sign}{sym}{abs_val / 1e12:,.2f}T"
    if abs_val >= 1e9:
        return f"{sign}{sym}{abs_val / 1e9:,.2f}B"
    if abs_val >= 1e6:
        return f"{sign}{sym}{abs_val / 1e6:,.2f}M"
    return f"{sign}{sym}{abs_val:,.0f}"


def get_statement_row(df: pd.DataFrame, possible_labels: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    normalized_index = {str(idx).strip().lower(): idx for idx in df.index}
    for label in possible_labels:
        key = label.strip().lower()
        if key in normalized_index:
            return df.loc[normalized_index[key]]
    return None


def classify_industry_profile(sector: str, industry: str) -> Dict[str, Any]:
    """
    Determines which metrics apply to the company's specific sector/industry.
    Banks, NBFCs, and Insurance firms do not use EBITDA, ROCE, or standard D/E.
    """
    sec_lower = (sector or "").lower()
    ind_lower = (industry or "").lower()
    financial_keywords = ["bank", "financial", "insurance", "credit", "capital markets", "nbfc", "lending"]
    is_financial = any(k in sec_lower or k in ind_lower for k in financial_keywords)

    if is_financial:
        return {
            "is_financial": True,
            "profile_name": "Banking / NBFC / Financial Institution",
            "roce_applicable": False,
            "ebitda_applicable": False,
            "de_applicable": False,
            "applicable_note": (
                "Financial Institution Detected: Debt/Deposits are raw operating capital for lenders. "
                "EBITDA, ROCE, and Debt-to-Equity are automatically marked N/A; evaluation focuses on "
                "ROE (>=14%), ROA (>=1.2%), Net Profit Margin, and Price-to-Book."
            ),
        }
    return {
        "is_financial": False,
        "profile_name": f"Operating Business ({sector or 'General Industry'})",
        "roce_applicable": True,
        "ebitda_applicable": True,
        "de_applicable": True,
        "applicable_note": (
            "Standard Operating Company: Full analysis active across ROE, ROCE (EBIT / Capital Employed), "
            "EBITDA Margin, Debt-to-Equity, Free Cash Flow, and Share Buyback EPS Quality."
        ),
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_company_bundle(ticker_symbol: str) -> Dict[str, Any]:
    ticker_clean = ticker_symbol.strip().upper()
    ticker_obj = yf.Ticker(ticker_clean)

    info: Dict[str, Any] = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        info = {}

    current_price = (
        safe_float(info.get("currentPrice"))
        or safe_float(info.get("regularMarketPrice"))
        or safe_float(info.get("previousClose"))
    )

    if current_price is None:
        try:
            hist = ticker_obj.history(period="5d")
            if hist is not None and not hist.empty:
                current_price = safe_float(hist["Close"].iloc[-1])
        except Exception:
            current_price = None

    if current_price is None:
        raise ValueError(
            f"Could not retrieve live market pricing for '{ticker_clean}'. "
            "For Indian stocks, ensure the suffix is .NS (NSE) or .BO (BSE), e.g., TCS.NS or RELIANCE.NS."
        )

    currency = info.get("currency")
    if not currency:
        currency = "INR" if ticker_clean.endswith((".NS", ".BO")) else "USD"

    try:
        income_stmt = ticker_obj.financials
    except Exception:
        income_stmt = pd.DataFrame()

    try:
        balance_sheet = ticker_obj.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cash_flow = ticker_obj.cashflow
    except Exception:
        cash_flow = pd.DataFrame()

    try:
        raw_news = ticker_obj.news or []
    except Exception:
        raw_news = []

    sector = info.get("sector") or "Technology"
    industry = info.get("industry") or "General"
    profile = classify_industry_profile(sector, industry)

    return {
        "ticker": ticker_clean,
        "currency": currency.upper(),
        "info": info,
        "sector": sector,
        "industry": industry,
        "profile": profile,
        "current_price": current_price,
        "income_stmt": income_stmt,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
        "raw_news": raw_news,
    }


# ==============================================================================
# 2. ROE, ROCE, EBITDA, GROWTH MOMENTUM & BUYBACK FORENSIC ENGINE
# ==============================================================================
def compute_cagr(recent_val: Optional[float], oldest_val: Optional[float], periods: int) -> Optional[float]:
    """Calculates Compound Annual Growth Rate (%) when both endpoints are positive."""
    if (
        recent_val is None
        or oldest_val is None
        or periods <= 0
        or recent_val <= 0
        or oldest_val <= 0
    ):
        return None
    return float(((recent_val / oldest_val) ** (1.0 / periods) - 1.0) * 100.0)


def extract_comprehensive_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes ROE, ROCE, ROA, D/E, Current Ratio, EBITDA & Margins,
    Recent vs Past Growth Comparison, and Share Buyback EPS Distortion.
    """
    info = bundle["info"]
    inc = bundle["income_stmt"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    profile = bundle["profile"]

    # Statement series extraction
    rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    gp_s = get_statement_row(inc, ["Gross Profit"])
    ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA"])
    ebit_s = get_statement_row(inc, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
    ni_s = get_statement_row(
        inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
    )
    eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
    shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])

    assets_s = get_statement_row(bs, ["Total Assets"])
    curr_liab_s = get_statement_row(bs, ["Current Liabilities", "Total Current Liabilities"])
    curr_assets_s = get_statement_row(bs, ["Current Assets", "Total Current Assets"])
    equity_s = get_statement_row(
        bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]
    )
    debt_s = get_statement_row(
        bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]
    )

    buyback_cf_s = get_statement_row(
        cf, ["Repurchase Of Capital Stock", "Common Stock Payments", "Net Common Stock Issuance"]
    )

    # 1. ROE (%)
    roe = safe_float(info.get("returnOnEquity"))
    if roe is None and ni_s is not None and equity_s is not None:
        ni_0 = safe_float(ni_s.dropna().iloc[0]) if not ni_s.dropna().empty else None
        eq_0 = safe_float(equity_s.dropna().iloc[0]) if not equity_s.dropna().empty else None
        if ni_0 is not None and eq_0 and eq_0 > 0:
            roe = ni_0 / eq_0
    roe_pct = roe * 100.0 if roe is not None else None

    # 2. ROCE (%) = EBIT / (Total Assets - Current Liabilities)
    roce_pct = None
    roce_prev_pct = None
    if profile["roce_applicable"] and ebit_s is not None and assets_s is not None and curr_liab_s is not None:
        common_cols = [c for c in ebit_s.index if c in assets_s.index and c in curr_liab_s.index]
        roce_vals = []
        for col in common_cols[:3]:
            ebit_v = safe_float(ebit_s.get(col))
            ass_v = safe_float(assets_s.get(col))
            cl_v = safe_float(curr_liab_s.get(col))
            if ebit_v is not None and ass_v is not None and cl_v is not None:
                cap_employed = ass_v - cl_v
                if cap_employed > 0:
                    roce_vals.append((ebit_v / cap_employed) * 100.0)
        if roce_vals:
            roce_pct = roce_vals[0]
            if len(roce_vals) > 1:
                roce_prev_pct = float(np.mean(roce_vals[1:]))

    # 3. ROA (%)
    roa = safe_float(info.get("returnOnAssets"))
    if roa is None and ni_s is not None and assets_s is not None:
        ni_0 = safe_float(ni_s.dropna().iloc[0]) if not ni_s.dropna().empty else None
        as_0 = safe_float(assets_s.dropna().iloc[0]) if not assets_s.dropna().empty else None
        if ni_0 is not None and as_0 and as_0 > 0:
            roa = ni_0 / as_0
    roa_pct = roa * 100.0 if roa is not None else None

    # 4. Debt-to-Equity & Current Ratio
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

    # 5. Multi-Year Growth & Margin Comparison (Recent YoY vs Past 3-Year CAGR)
    def extract_growth_profile(series: Optional[pd.Series], info_yoy_key: Optional[str] = None) -> Dict[str, Optional[float]]:
        recent_yoy = None
        past_cagr = None
        latest_val = None
        prev_val = None
        oldest_val = None

        if series is not None and not series.dropna().empty:
            clean_s = series.dropna()
            latest_val = safe_float(clean_s.iloc[0])
            if len(clean_s) >= 2:
                prev_val = safe_float(clean_s.iloc[1])
                if latest_val is not None and prev_val is not None and prev_val > 0:
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

        return {
            "latest": latest_val,
            "prev": prev_val,
            "recent_yoy_pct": recent_yoy,
            "past_cagr_pct": past_cagr,
        }

    rev_growth = extract_growth_profile(rev_s, "revenueGrowth")
    ebitda_growth = extract_growth_profile(ebitda_s) if profile["ebitda_applicable"] else {
        "latest": None, "prev": None, "recent_yoy_pct": None, "past_cagr_pct": None
    }
    ni_growth = extract_growth_profile(ni_s, "earningsGrowth")
    eps_growth = extract_growth_profile(eps_s, "earningsGrowth")

    # Margins: Current vs Past Historical Average
    ebitda_margin_curr = None
    ebitda_margin_past = None
    net_margin_curr = None
    net_margin_past = None
    gross_margin_curr = None
    gross_margin_past = None

    if rev_s is not None and not rev_s.dropna().empty:
        cols = list(rev_s.dropna().index[:4])
        gm_list, em_list, nm_list = [], [], []
        for c in cols:
            r_v = safe_float(rev_s.get(c))
            if r_v and r_v > 0:
                if gp_s is not None and safe_float(gp_s.get(c)) is not None:
                    gm_list.append((safe_float(gp_s.get(c)) / r_v) * 100.0)
                if profile["ebitda_applicable"] and ebitda_s is not None and safe_float(ebitda_s.get(c)) is not None:
                    em_list.append((safe_float(ebitda_s.get(c)) / r_v) * 100.0)
                if ni_s is not None and safe_float(ni_s.get(c)) is not None:
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

    # 6. Share Buyback & EPS Quality Detector
    share_change_pct = None
    latest_shares = None
    oldest_shares = None
    if shares_s is not None and len(shares_s.dropna()) >= 2:
        clean_sh = shares_s.dropna()
        latest_shares = safe_float(clean_sh.iloc[0])
        oldest_shares = safe_float(clean_sh.iloc[min(2, len(clean_sh) - 1)])
        if latest_shares and oldest_shares and oldest_shares > 0:
            share_change_pct = ((latest_shares - oldest_shares) / oldest_shares) * 100.0

    latest_buyback_cash = 0.0
    total_3yr_buyback_cash = 0.0
    if buyback_cf_s is not None and not buyback_cf_s.dropna().empty:
        clean_bb = buyback_cf_s.dropna()
        # In yfinance cashflow, stock repurchases are negative cash outflows
        bb_vals = [abs(safe_float(v, 0.0) or 0.0) for v in clean_bb.iloc[:3] if (safe_float(v, 0.0) or 0.0) < 0]
        if bb_vals:
            latest_buyback_cash = bb_vals[0]
            total_3yr_buyback_cash = float(np.sum(bb_vals))

    eps_cagr = eps_growth["past_cagr_pct"] or eps_growth["recent_yoy_pct"]
    ni_cagr = ni_growth["past_cagr_pct"] or ni_growth["recent_yoy_pct"]
    eps_buyback_boost_pct = None
    if eps_cagr is not None and ni_cagr is not None:
        eps_buyback_boost_pct = eps_cagr - ni_cagr

    buyback_detected = (
        (share_change_pct is not None and share_change_pct <= -1.0)
        or (total_3yr_buyback_cash > 0)
        or (eps_buyback_boost_pct is not None and eps_buyback_boost_pct >= 1.5)
    )

    if buyback_detected and eps_buyback_boost_pct is not None and eps_buyback_boost_pct >= 1.5:
        buyback_verdict = "BUYBACK-BOOSTED EPS"
        buyback_explanation = (
            f"Share count changed by {share_change_pct:+.2f}% over the multi-year window "
            f"(3-Yr Buybacks: {format_large_currency(total_3yr_buyback_cash, bundle['currency'])}). "
            f"EPS grew at {eps_cagr:+.2f}%/yr vs Net Income growth of {ni_cagr:+.2f}%/yr — meaning "
            f"+{eps_buyback_boost_pct:.2f}%/yr of EPS growth came from share buybacks shrinking the share base."
        )
    elif share_change_pct is not None and share_change_pct >= 2.0:
        buyback_verdict = "EQUITY DILUTION DETECTED"
        buyback_explanation = (
            f"Diluted share count increased by {share_change_pct:+.2f}% over the multi-year window, "
            f"causing per-share EPS growth to lag total Net Income growth."
        )
    else:
        buyback_verdict = "100% ORGANIC EPS GROWTH"
        buyback_explanation = (
            "No material share-count distortion detected. Reported EPS growth closely tracks "
            "genuine underlying Net Profit growth."
        )

    return {
        "roe_pct": roe_pct,
        "roce_pct": roce_pct,
        "roce_prev_pct": roce_prev_pct,
        "roa_pct": roa_pct,
        "de_ratio": de_ratio,
        "current_ratio": current_ratio,
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
        "latest_buyback_cash": latest_buyback_cash,
        "total_3yr_buyback_cash": total_3yr_buyback_cash,
        "eps_buyback_boost_pct": eps_buyback_boost_pct,
        "buyback_detected": buyback_detected,
        "buyback_verdict": buyback_verdict,
        "buyback_explanation": buyback_explanation,
    }


def build_three_year_history_table(bundle: Dict[str, Any]) -> pd.DataFrame:
    """Builds a 3-year historical table with Revenue, EBITDA, Margins, EPS, Shares, and FCF."""
    inc = bundle["income_stmt"]
    cf = bundle["cash_flow"]
    currency = bundle["currency"]
    profile = bundle["profile"]

    if inc is None or inc.empty:
        return pd.DataFrame()

    cols = list(inc.columns[:3])
    rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA"])
    ni_s = get_statement_row(
        inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
    )
    eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
    shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])
    fcf_s = get_statement_row(cf, ["Free Cash Flow"])
    ocf_s = get_statement_row(
        cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
    )
    capex_s = get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])

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
                "Revenue": format_large_currency(rev, currency),
                "EBITDA": format_large_currency(ebitda, currency) if profile["ebitda_applicable"] else "N/A (Financial)",
                "EBITDA Margin": f"{ebitda_margin:.2f}%" if ebitda_margin is not None else "N/A (Financial)",
                "Net Profit Margin": f"{net_margin:.2f}%" if net_margin is not None else "N/A",
                "Diluted EPS": format_price(eps_v, currency),
                "Diluted Shares": f"{sh_v / 1e7:,.2f} Cr" if (sh_v and currency == "INR") else (f"{sh_v / 1e9:,.2f}B" if sh_v else "N/A"),
                "Free Cash Flow": format_large_currency(fcf, currency),
            }
        )

    df_hist = pd.DataFrame(records)
    if not df_hist.empty:
        df_hist = df_hist.set_index("Fiscal Year")
    return df_hist


# ==============================================================================
# 3. LIVE INDUSTRY & PEER BENCHMARKING ENGINE
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_industry_peer_benchmarks(
    target_ticker: str, sector: str, currency: str, custom_peers_str: str = ""
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Fetches live metrics for 3-4 industry peers in the same market and computes
    the Industry Peer Median across Revenue Growth, EBITDA Margin, Net Margin, ROE, ROCE, and P/E.
    """
    is_india = currency == "INR" or target_ticker.endswith((".NS", ".BO"))
    peer_map = INDIA_SECTOR_PEERS if is_india else US_SECTOR_PEERS
    default_peers = peer_map.get(sector, ["TCS.NS", "INFY.NS", "ITC.NS", "LT.NS"] if is_india else ["AAPL", "MSFT", "GOOG", "JNJ"])

    if custom_peers_str.strip():
        raw_list = [p.strip().upper() for p in custom_peers_str.split(",") if p.strip()]
        peer_tickers = []
        for p in raw_list:
            if is_india and "." not in p:
                peer_tickers.append(f"{p}.NS")
            else:
                peer_tickers.append(p)
    else:
        peer_tickers = [p for p in default_peers if p != target_ticker][:4]

    rows = []
    for pt in peer_tickers:
        try:
            t_obj = yf.Ticker(pt)
            p_info = t_obj.info or {}
            p_inc = t_obj.financials
            p_bs = t_obj.balance_sheet

            rev_g = safe_float(p_info.get("revenueGrowth"))
            ebitda_m = safe_float(p_info.get("ebitdaMargins"))
            net_m = safe_float(p_info.get("profitMargins"))
            roe_v = safe_float(p_info.get("returnOnEquity"))
            pe_v = safe_float(p_info.get("trailingPE")) or safe_float(p_info.get("forwardPE"))

            # Calculate peer ROCE from statements
            roce_v = None
            ebit_s = get_statement_row(p_inc, ["EBIT", "Operating Income"])
            ass_s = get_statement_row(p_bs, ["Total Assets"])
            cl_s = get_statement_row(p_bs, ["Current Liabilities", "Total Current Liabilities"])
            if ebit_s is not None and ass_s is not None and cl_s is not None:
                eb_0 = safe_float(ebit_s.dropna().iloc[0]) if not ebit_s.dropna().empty else None
                as_0 = safe_float(ass_s.dropna().iloc[0]) if not ass_s.dropna().empty else None
                cl_0 = safe_float(cl_s.dropna().iloc[0]) if not cl_s.dropna().empty else None
                if eb_0 is not None and as_0 is not None and cl_0 is not None and (as_0 - cl_0) > 0:
                    roce_v = (eb_0 / (as_0 - cl_0)) * 100.0

            rows.append(
                {
                    "Ticker": pt,
                    "Company": (p_info.get("shortName") or pt)[:24],
                    "Rev Growth (%)": rev_g * 100.0 if rev_g is not None else np.nan,
                    "EBITDA Margin (%)": ebitda_m * 100.0 if ebitda_m is not None else np.nan,
                    "Net Margin (%)": net_m * 100.0 if net_m is not None else np.nan,
                    "ROE (%)": roe_v * 100.0 if roe_v is not None else np.nan,
                    "ROCE (%)": roce_v if roce_v is not None else np.nan,
                    "Trailing P/E": pe_v if pe_v is not None else np.nan,
                }
            )
        except Exception:
            continue

    df_peers = pd.DataFrame(rows)
    medians: Dict[str, Optional[float]] = {
        "rev_growth_med": None,
        "ebitda_margin_med": None,
        "net_margin_med": None,
        "roe_med": None,
        "roce_med": None,
        "pe_med": None,
    }

    if not df_peers.empty:
        medians["rev_growth_med"] = safe_float(df_peers["Rev Growth (%)"].median(skipna=True))
        medians["ebitda_margin_med"] = safe_float(df_peers["EBITDA Margin (%)"].median(skipna=True))
        medians["net_margin_med"] = safe_float(df_peers["Net Margin (%)"].median(skipna=True))
        medians["roe_med"] = safe_float(df_peers["ROE (%)"].median(skipna=True))
        medians["roce_med"] = safe_float(df_peers["ROCE (%)"].median(skipna=True))
        medians["pe_med"] = safe_float(df_peers["Trailing P/E"].median(skipna=True))

    return df_peers, medians


# ==============================================================================
# 4. AUTOMATED 2-STAGE DCF & BENJAMIN GRAHAM SCENARIO ENGINE
# ==============================================================================
def extract_dcf_inputs(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    info = bundle["info"]
    inc = bundle["income_stmt"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    currency = bundle["currency"]
    current_price = bundle["current_price"]

    shares = safe_float(info.get("sharesOutstanding")) or safe_float(info.get("impliedSharesOutstanding"))
    if shares is None or shares <= 0:
        mcap = safe_float(info.get("marketCap"))
        if mcap and current_price > 0:
            shares = mcap / current_price

    fcf_0 = safe_float(info.get("freeCashflow"))
    fcf_source = "Reported TTM Free Cash Flow"

    if fcf_0 is None or fcf_0 <= 0:
        fcf_s = get_statement_row(cf, ["Free Cash Flow"])
        if fcf_s is not None and not fcf_s.dropna().empty:
            fcf_0 = safe_float(fcf_s.dropna().iloc[0])
            fcf_source = "Latest Annual Cash Flow Statement FCF"

    if fcf_0 is None or fcf_0 <= 0:
        ocf_s = get_statement_row(
            cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
        )
        capex_s = get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])
        if ocf_s is not None and not ocf_s.dropna().empty:
            ocf_val = safe_float(ocf_s.dropna().iloc[0])
            capex_val = safe_float(capex_s.dropna().iloc[0]) if (capex_s is not None and not capex_s.dropna().empty) else 0.0
            if ocf_val is not None:
                fcf_0 = ocf_val + capex_val if (capex_val or 0.0) < 0 else ocf_val - (capex_val or 0.0)
                fcf_source = "Operating Cash Flow minus CapEx"

    if fcf_0 is None or fcf_0 <= 0:
        ni_s = get_statement_row(
            inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
        )
        ni_val = safe_float(info.get("netIncomeToCommon"))
        if (ni_val is None or ni_val <= 0) and ni_s is not None and not ni_s.dropna().empty:
            ni_val = safe_float(ni_s.dropna().iloc[0])
        if ni_val is not None and ni_val > 0:
            fcf_0 = ni_val * 0.85
            fcf_source = "Normalized Owner Earnings (85% of Net Income Proxy)"

    total_cash = safe_float(info.get("totalCash"))
    if total_cash is None:
        cash_s = get_statement_row(
            bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]
        )
        total_cash = safe_float(cash_s.dropna().iloc[0]) if (cash_s is not None and not cash_s.dropna().empty) else 0.0

    total_debt = safe_float(info.get("totalDebt"))
    if total_debt is None:
        debt_s = get_statement_row(bs, ["Total Debt", "Long Term Debt"])
        total_debt = safe_float(debt_s.dropna().iloc[0]) if (debt_s is not None and not debt_s.dropna().empty) else 0.0

    # For financial institutions, debt is operational funding; do not subtract raw bank debt in equity bridge
    if bundle["profile"]["is_financial"]:
        total_cash = 0.0
        total_debt = 0.0

    candidates = [
        v
        for v in [
            metrics["rev_growth"]["recent_yoy_pct"],
            metrics["rev_growth"]["past_cagr_pct"],
            metrics["ni_growth"]["recent_yoy_pct"],
            metrics["ni_growth"]["past_cagr_pct"],
        ]
        if v is not None
    ]
    default_g = 10.0 if currency == "INR" else 7.5
    est_g1 = float(np.median(candidates)) if candidates else default_g
    base_g1_pct = float(np.clip(est_g1, 2.0, 20.0))

    beta = safe_float(info.get("beta"), default=1.0) or 1.0
    beta_clamped = float(np.clip(beta, 0.6, 1.8))

    if currency == "INR":
        rf_rate, erp, base_terminal_g_pct = 6.8, 5.2, 4.0
    else:
        rf_rate, erp, base_terminal_g_pct = 4.2, 4.8, 2.5

    base_discount_pct = float(np.clip(rf_rate + beta_clamped * erp, 8.0, 15.0))

    eps_ttm = safe_float(info.get("trailingEps")) or safe_float(info.get("forwardEps"))
    if (eps_ttm is None or eps_ttm <= 0) and fcf_0 and shares and shares > 0:
        eps_ttm = fcf_0 / shares

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


def compute_single_dcf_and_graham(
    fcf_0: Optional[float],
    shares: Optional[float],
    total_cash: float,
    total_debt: float,
    eps_ttm: Optional[float],
    g1_pct: float,
    discount_pct: float,
    terminal_g_pct: float,
    current_price: float,
) -> Dict[str, Optional[float]]:
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

    valid_vals = [v for v in [dcf_per_share, graham_per_share] if v is not None and v > 0]
    blended_iv = float(np.mean(valid_vals)) if valid_vals else None
    blended_mos_pct = (
        ((blended_iv - current_price) / blended_iv) * 100.0
        if (blended_iv is not None and blended_iv > 0)
        else None
    )

    return {
        "g1_pct": g1_pct,
        "discount_pct": discount_pct,
        "terminal_g_pct": safe_term_g_pct,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "dcf_intrinsic": dcf_per_share,
        "dcf_mos_pct": dcf_mos_pct,
        "graham_intrinsic": graham_per_share,
        "graham_mos_pct": graham_mos_pct,
        "blended_intrinsic": blended_iv,
        "blended_mos_pct": blended_mos_pct,
    }


def build_three_scenarios(
    dcf_inputs: Dict[str, Any],
    base_g1_pct: float,
    base_discount_pct: float,
    base_terminal_g_pct: float,
    current_price: float,
) -> Dict[str, Dict[str, Optional[float]]]:
    cons_g1 = max(0.0, base_g1_pct * 0.70)
    cons_r = min(18.0, base_discount_pct + 1.5)
    cons_term = max(1.5, base_terminal_g_pct - 0.5)

    opt_g1 = min(25.0, base_g1_pct * 1.25)
    opt_r = max(7.0, base_discount_pct - 1.0)
    opt_term = min(opt_r - 2.0, base_terminal_g_pct + 0.5)

    return {
        "Conservative": compute_single_dcf_and_graham(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], cons_g1, cons_r, cons_term, current_price
        ),
        "Base Case": compute_single_dcf_and_graham(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], base_g1_pct, base_discount_pct, base_terminal_g_pct, current_price
        ),
        "Optimistic": compute_single_dcf_and_graham(
            dcf_inputs["fcf_0"], dcf_inputs["shares"], dcf_inputs["total_cash"],
            dcf_inputs["total_debt"], dcf_inputs["eps_ttm"], opt_g1, opt_r, opt_term, current_price
        ),
    }


# ==============================================================================
# 5. CORPORATE NEWS FILTERING & VADER SENTIMENT ENGINE
# ==============================================================================
NOISE_PATTERNS = [
    r"\bmidday movers\b", r"\bstock market today\b", r"\btop gainers\b",
    r"\btop losers\b", r"\boptions alert\b", r"\bunusual options\b",
    r"\bchart watch\b", r"\btechnical analysis\b", r"\bintraday\b",
    r"\bpre-market\b", r"\bafter-hours movers\b", r"\bwhy .* stock moved\b",
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
# 6. INDUSTRY-ADAPTIVE DECISION ENGINE (70% FUNDAMENTALS / 30% SENTIMENT)
# ==============================================================================
def compute_decision_verdict(
    ticker: str,
    company_name: str,
    currency: str,
    profile: Dict[str, Any],
    metrics: Dict[str, Any],
    peer_medians: Dict[str, Optional[float]],
    scenarios: Dict[str, Dict[str, Optional[float]]],
    news_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    70% Weight on Financial Health, Industry Comparison, Growth Quality & Valuation (70 pts):
      - Blended Margin of Safety (DCF + Graham): up to 25 pts
      - Capital Efficiency (ROE + ROCE if non-financial, or ROE + ROA if financial): up to 20 pts
      - Growth & Margin Momentum + EPS Organic Quality: up to 15 pts
      - Balance Sheet Safety & Industry Peer Outperformance: up to 10 pts
    30% Weight on Long-Term Corporate News Sentiment (30 pts).
    """
    base_case = scenarios["Base Case"]
    cons_case = scenarios["Conservative"]
    mos = base_case.get("blended_mos_pct")
    cons_mos = cons_case.get("blended_mos_pct")

    roe = metrics.get("roe_pct")
    roce = metrics.get("roce_pct")
    roa = metrics.get("roa_pct")
    de = metrics.get("de_ratio")

    # 1. Margin of Safety Sub-Score (0 - 25 pts)
    if mos is None:
        mos_pts = 5.0
    elif mos >= 30.0:
        mos_pts = 25.0
    elif mos >= 0.0:
        mos_pts = 12.5 + (mos / 30.0) * 12.5
    elif mos >= -30.0:
        mos_pts = max(0.0, 12.5 + (mos / 30.0) * 12.5)
    else:
        mos_pts = 0.0

    # 2. ROE & ROCE / ROA Capital Efficiency Sub-Score (0 - 20 pts)
    roe_pts = min(10.0, max(0.0, ((roe or 0.0) / 20.0) * 10.0)) if roe is not None else 5.0
    if profile["roce_applicable"]:
        roce_pts = min(10.0, max(0.0, ((roce or 0.0) / 20.0) * 10.0)) if roce is not None else 5.0
    else:
        # For Banks/NBFCs, reward ROA >= 1.5% for full 10 pts
        roce_pts = min(10.0, max(0.0, ((roa or 0.0) / 1.5) * 10.0)) if roa is not None else 5.0
    efficiency_pts = roe_pts + roce_pts

    # 3. Growth Acceleration, Margin Expansion & Buyback Quality (0 - 15 pts)
    rev_yoy = metrics["rev_growth"]["recent_yoy_pct"]
    rev_cagr = metrics["rev_growth"]["past_cagr_pct"]
    growth_pts = 5.0
    if rev_yoy is not None and rev_yoy > 8.0:
        growth_pts += 3.0
    if rev_yoy is not None and rev_cagr is not None and rev_yoy >= rev_cagr:
        growth_pts += 2.0  # Accelerating revenue growth

    nm_curr = metrics.get("net_margin_curr")
    nm_past = metrics.get("net_margin_past")
    if nm_curr is not None and nm_past is not None and nm_curr >= nm_past:
        growth_pts += 2.5  # Expanding profit margins

    # Reward organic EPS growth; slightly discount heavy buyback distortion or dilution
    if metrics["buyback_verdict"] == "100% ORGANIC EPS GROWTH":
        growth_pts += 2.5
    elif metrics["buyback_verdict"] == "BUYBACK-BOOSTED EPS":
        growth_pts += 1.5
    growth_pts = min(15.0, growth_pts)

    # 4. Balance Sheet & Industry Peer Comparison (0 - 10 pts)
    bs_peer_pts = 0.0
    if profile["de_applicable"]:
        if de is None:
            bs_peer_pts += 2.5
        elif de <= 0.5:
            bs_peer_pts += 5.0
        elif de <= 1.5:
            bs_peer_pts += 2.5
    else:
        bs_peer_pts += 4.0  # Financial baseline

    # Industry outperformance check (ROE & Margin vs Peer Median)
    roe_med = peer_medians.get("roe_med")
    nm_med = peer_medians.get("net_margin_med")
    if roe is not None and roe_med is not None and roe >= roe_med:
        bs_peer_pts += 2.5
    else:
        bs_peer_pts += 1.0
    if nm_curr is not None and nm_med is not None and nm_curr >= nm_med:
        bs_peer_pts += 2.5
    else:
        bs_peer_pts += 1.0
    bs_peer_pts = min(10.0, bs_peer_pts)

    fundamental_score_70 = mos_pts + efficiency_pts + growth_pts + bs_peer_pts

    # 5. Sentiment Sub-Score (0 - 30 pts)
    avg_comp = news_summary.get("avg_compound", 0.0)
    clamped_comp = float(np.clip(avg_comp, -0.5, 0.5))
    sentiment_score_30 = ((clamped_comp + 0.5) / 1.0) * 30.0

    total_score = float(np.clip(fundamental_score_70 + sentiment_score_30, 0.0, 100.0))

    if total_score >= 75.0 and (mos is not None and mos >= 15.0):
        verdict, badge_color = "STRONG BUY", "#16a34a"
    elif total_score >= 60.0:
        verdict, badge_color = "ACCUMULATE", "#2563eb"
    elif total_score >= 45.0:
        verdict, badge_color = "HOLD", "#d97706"
    else:
        verdict, badge_color = "OVERVALUED", "#dc2626"

    roe_str = f"{roe:.1f}%" if roe is not None else "N/A"
    roce_str = f", ROCE of {roce:.1f}%," if (profile["roce_applicable"] and roce is not None) else ""
    base_iv_str = format_price(base_case.get("blended_intrinsic"), currency)
    cons_iv_str = format_price(cons_case.get("blended_intrinsic"), currency)
    mos_str = f"{mos:+.1f}%" if mos is not None else "indeterminate"

    sentence_1 = (
        f"{company_name} ({ticker}) operates with a trailing ROE of {roe_str}{roce_str} and "
        f"{metrics['buyback_verdict'].lower()} across the recent 3-year cycle."
    )
    sentence_2 = (
        f"Blending the 2-Stage DCF and Benjamin Graham models yields a Base-Case Intrinsic Value of {base_iv_str} "
        f"({mos_str} Margin of Safety) and a Conservative Bear-Case floor of {cons_iv_str}, alongside a "
        f"corporate news sentiment score of {sentiment_score_30:.1f}/30 pts."
    )

    risk_factors = []
    if rev_yoy is not None and rev_cagr is not None and rev_yoy < rev_cagr - 3.0:
        risk_factors.append("recent top-line revenue deceleration relative to its 3-year historical CAGR")
    if metrics["buyback_verdict"] == "BUYBACK-BOOSTED EPS":
        risk_factors.append("EPS growth partially flattered by share buybacks rather than pure net profit expansion")
    if cons_mos is not None and cons_mos < -15.0:
        risk_factors.append("downside sensitivity if growth slows toward the Conservative scenario")
    if mos is not None and mos < 0:
        risk_factors.append("multiple compression from trading above Base-Case intrinsic value")
    if not risk_factors:
        risk_factors.append("macroeconomic demand cyclicality and sector multiple normalization")

    sentence_3 = f"Core long-term risk factors to monitor include {', '.join(risk_factors)}."

    return {
        "fundamental_score_70": fundamental_score_70,
        "sentiment_score_30": sentiment_score_30,
        "total_score": total_score,
        "verdict": verdict,
        "badge_color": badge_color,
        "thesis": f"{sentence_1} {sentence_2} {sentence_3}",
    }


# ==============================================================================
# 7. STREAMLIT USER INTERFACE
# ==============================================================================
def main() -> None:
    with st.sidebar:
        st.header("⚙️ Market & Ticker Selector")

        market_choice = st.radio(
            "Select Equity Market",
            options=["🇮🇳 India (NSE / BSE - ₹)", "🇺🇸 US Market (NYSE / NASDAQ - $)"],
            index=0,
        )

        is_india = market_choice.startswith("🇮🇳")
        presets = INDIAN_PRESETS if is_india else US_PRESETS
        preset_options = ["Custom Ticker..."] + [f"{k} — {v}" for k, v in presets.items()]

        selected_preset = st.selectbox("Quick-Select Blue-Chip Stock", options=preset_options, index=1)

        if selected_preset != "Custom Ticker...":
            default_ticker = selected_preset.split(" — ")[0].strip()
        else:
            default_ticker = "TCS.NS" if is_india else "AAPL"

        raw_ticker_input = st.text_input(
            "Or Enter Any Stock Ticker",
            value=default_ticker,
            help="Type any ticker (e.g., TCS, HDFCBANK, MARUTI, AAPL, GOOG). In India mode, .NS is appended automatically if omitted.",
        )

        exchange_suffix = ".NS"
        if is_india:
            exchange_suffix = st.selectbox(
                "Default Indian Exchange",
                options=[".NS (NSE)", ".BO (BSE)"],
                index=0,
            ).split(" ")[0]

        cleaned_input = raw_ticker_input.strip().upper()
        if is_india and cleaned_input and "." not in cleaned_input:
            resolved_ticker = f"{cleaned_input}{exchange_suffix}"
        else:
            resolved_ticker = cleaned_input

        st.markdown("---")
        st.subheader("🏢 Custom Industry Peers (Optional)")
        custom_peers_input = st.text_input(
            "Override Sector Peer Tickers (comma-separated)",
            value="",
            placeholder="e.g., INFY.NS, HCLTECH.NS, WIPRO.NS",
            help="Leave blank to automatically benchmark against top sector peers.",
        )

        st.markdown("---")
        st.subheader("📐 Valuation & DCF Overrides")
        use_custom_params = st.checkbox(
            "Override Automatic Growth & WACC Assumptions",
            value=False,
        )
        custom_g1 = st.slider(
            "Base Stage-1 Growth Rate g₁ (Years 1–5, %)",
            min_value=0.0, max_value=25.0, value=11.0 if is_india else 8.5, step=0.5,
            disabled=not use_custom_params,
        )
        custom_discount = st.slider(
            "Base Discount Rate / WACC r (%)",
            min_value=7.0, max_value=18.0, value=12.0 if is_india else 9.5, step=0.25,
            disabled=not use_custom_params,
        )
        custom_term_g = st.slider(
            "Base Terminal Growth Rate g_term (Year 6+, %)",
            min_value=1.0, max_value=6.0, value=4.0 if is_india else 2.5, step=0.25,
            disabled=not use_custom_params,
        )

        analyze_clicked = st.button("🔍 Analyze Fundamental Value", type="primary", use_container_width=True)

    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = resolved_ticker
    if analyze_clicked or resolved_ticker != st.session_state.get("last_resolved_input"):
        st.session_state["active_ticker"] = resolved_ticker
        st.session_state["last_resolved_input"] = resolved_ticker

    target_ticker = st.session_state["active_ticker"]

    st.title("🏛️ Buffett Institutional Value, ROCE & Forensic Analyzer")
    st.markdown(
        "**Industry-Adaptive Screening** • **Current vs. Past Growth & Margin Momentum** • "
        "**Buyback EPS Forensics** • **Live Sector Peer Benchmarks** • **3-Scenario 2-Stage DCF**"
    )

    with st.spinner(f"Running institutional statements, buyback forensics, and peer benchmarks for {target_ticker}..."):
        try:
            bundle = fetch_company_bundle(target_ticker)
            metrics = extract_comprehensive_metrics(bundle)
            df_peers, peer_medians = fetch_industry_peer_benchmarks(
                target_ticker=target_ticker,
                sector=bundle["sector"],
                currency=bundle["currency"],
                custom_peers_str=custom_peers_input,
            )
        except ValueError as ve:
            st.error(f"**Ticker Lookup Error:** {ve}")
            st.stop()
        except Exception as exc:
            st.error(f"**API Connectivity Issue:** Unable to fetch data (`{exc}`). Please try again.")
            st.stop()

    info = bundle["info"]
    currency = bundle["currency"]
    profile = bundle["profile"]
    current_price = bundle["current_price"]
    company_name = info.get("longName") or info.get("shortName") or target_ticker
    mcap = safe_float(info.get("marketCap"))

    st.subheader(f"{company_name} (`{target_ticker}`)")
    st.caption(
        f"**Industry Profile:** {profile['profile_name']} | **Sector:** {bundle['sector']} | "
        f"**Market Price:** {format_price(current_price, currency)} ({currency}) | "
        f"**Market Cap:** {format_large_currency(mcap, currency)}"
    )
    st.info(f"🧭 **Industry-Adaptive Routing:** {profile['applicable_note']}")

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 1: ROE, ROCE, MOAT KPIs & 3-YEAR HISTORICAL TABLE
    # --------------------------------------------------------------------------
    st.header("1. Business Health, Capital Efficiency (ROE & ROCE) & Moat")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        roe_v = metrics["roe_pct"]
        st.metric(
            label="Return on Equity (ROE)",
            value=f"{roe_v:.2f}%" if roe_v is not None else "N/A",
            delta=f"vs {peer_medians['roe_med']:.1f}% Peer Med" if (roe_v is not None and peer_medians["roe_med"] is not None) else "Target >= 15%",
            delta_color="normal" if (roe_v is not None and roe_v >= (peer_medians["roe_med"] or 15.0)) else "inverse",
        )
    with k2:
        roce_v = metrics["roce_pct"]
        if profile["roce_applicable"]:
            st.metric(
                label="Return on Capital (ROCE)",
                value=f"{roce_v:.2f}%" if roce_v is not None else "N/A",
                delta=f"vs {peer_medians['roce_med']:.1f}% Peer Med" if (roce_v is not None and peer_medians["roce_med"] is not None) else "Target >= 15%",
                delta_color="normal" if (roce_v is not None and roce_v >= (peer_medians["roce_med"] or 15.0)) else "inverse",
            )
        else:
            st.metric(label="ROCE (EBIT / Cap Emp)", value="N/A (Bank/NBFC)", delta="See ROA & ROE", delta_color="off")
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
                value=f"{de_v:.2f}x" if de_v is not None else "N/A",
                delta="Conservative <= 0.8x" if (de_v is not None and de_v <= 0.8) else "Elevated Leverage",
                delta_color="normal" if (de_v is not None and de_v <= 0.8) else "inverse",
            )
        else:
            pb_v = safe_float(info.get("priceToBook"))
            st.metric(
                label="Price-to-Book (P/B)",
                value=f"{pb_v:.2f}x" if pb_v is not None else "N/A",
                delta="Bank Valuation Multiple",
                delta_color="off",
            )
    with k5:
        cr_v = metrics["current_ratio"]
        st.metric(
            label="Current Ratio",
            value=f"{cr_v:.2f}x" if cr_v is not None else "N/A (Bank)",
            delta="Liquid >= 1.2x" if (cr_v is not None and cr_v >= 1.2) else None,
            delta_color="normal" if (cr_v is not None and cr_v >= 1.2) else "off",
        )

    st.markdown(f"#### 📊 3-Year Historical Financial Statements ({currency})")
    hist_df = build_three_year_history_table(bundle)
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 2: CURRENT VS. PAST GROWTH, MARGINS, BUYBACK CHECK & INDUSTRY PEERS
    # --------------------------------------------------------------------------
    st.header("2. Growth Acceleration, EBITDA/Profit Margins, Buyback Forensics & Industry Comparison")

    # 2A. Buyback & EPS Quality Alert Box
    bb_verdict = metrics["buyback_verdict"]
    bb_color = "#16a34a" if bb_verdict == "100% ORGANIC EPS GROWTH" else ("#d97706" if bb_verdict == "BUYBACK-BOOSTED EPS" else "#dc2626")
    st.markdown(
        f"""
        <div style="background-color: #1e293b; border-left: 6px solid {bb_color}; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px;">
            <strong style="color: {bb_color}; font-size: 1.02rem;">🔎 EPS & Share Buyback Forensic Check: {bb_verdict}</strong><br>
            <span style="color: #e2e8f0; font-size: 0.95rem;">{metrics['buyback_explanation']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2B. Current vs Past Growth & Margin Comparison Table (with Industry Benchmark)
    def fmt_pct(val: Optional[float]) -> str:
        return f"{val:+.2f}%" if val is not None else "N/A"

    def fmt_level_pct(val: Optional[float]) -> str:
        return f"{val:.2f}%" if val is not None else "N/A"

    rev_yoy, rev_cagr = metrics["rev_growth"]["recent_yoy_pct"], metrics["rev_growth"]["past_cagr_pct"]
    eb_yoy, eb_cagr = metrics["ebitda_growth"]["recent_yoy_pct"], metrics["ebitda_growth"]["past_cagr_pct"]
    ni_yoy, ni_cagr = metrics["ni_growth"]["recent_yoy_pct"], metrics["ni_growth"]["past_cagr_pct"]
    eps_yoy, eps_cagr = metrics["eps_growth"]["recent_yoy_pct"], metrics["eps_growth"]["past_cagr_pct"]

    comparison_rows = [
        {
            "Metric": "Revenue Growth Rate",
            "Current / Recent (YoY)": fmt_pct(rev_yoy),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_pct(rev_cagr),
            "Momentum (Current vs Past)": (
                "🟢 Accelerating" if (rev_yoy is not None and rev_cagr is not None and rev_yoy >= rev_cagr)
                else ("🟡 Decelerating" if (rev_yoy is not None and rev_cagr is not None) else "N/A")
            ),
            "Industry Peer Median": fmt_pct(peer_medians.get("rev_growth_med")),
            "vs. Industry": (
                "🏆 Outperforming" if (rev_yoy is not None and peer_medians.get("rev_growth_med") is not None and rev_yoy >= peer_medians["rev_growth_med"])
                else ("📉 Lagging Peers" if (rev_yoy is not None and peer_medians.get("rev_growth_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "EBITDA Growth Rate",
            "Current / Recent (YoY)": fmt_pct(eb_yoy) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_pct(eb_cagr) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": (
                "🟢 Accelerating" if (profile["ebitda_applicable"] and eb_yoy is not None and eb_cagr is not None and eb_yoy >= eb_cagr)
                else ("🟡 Decelerating" if (profile["ebitda_applicable"] and eb_yoy is not None and eb_cagr is not None) else "N/A")
            ),
            "Industry Peer Median": "Tracked via Margin",
            "vs. Industry": "—",
        },
        {
            "Metric": "EBITDA Margin (%)",
            "Current / Recent (YoY)": fmt_level_pct(metrics["ebitda_margin_curr"]) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(metrics["ebitda_margin_past"]) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": (
                f"🟢 Expanding ({metrics['ebitda_margin_curr'] - metrics['ebitda_margin_past']:+.2f}%)"
                if (profile["ebitda_applicable"] and metrics["ebitda_margin_curr"] is not None and metrics["ebitda_margin_past"] is not None and metrics["ebitda_margin_curr"] >= metrics["ebitda_margin_past"])
                else (
                    f"🔴 Compressing ({metrics['ebitda_margin_curr'] - metrics['ebitda_margin_past']:+.2f}%)"
                    if (profile["ebitda_applicable"] and metrics["ebitda_margin_curr"] is not None and metrics["ebitda_margin_past"] is not None)
                    else "N/A"
                )
            ),
            "Industry Peer Median": fmt_level_pct(peer_medians.get("ebitda_margin_med")) if profile["ebitda_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Higher Margin" if (profile["ebitda_applicable"] and metrics["ebitda_margin_curr"] is not None and peer_medians.get("ebitda_margin_med") is not None and metrics["ebitda_margin_curr"] >= peer_medians["ebitda_margin_med"])
                else ("📉 Below Peer Margin" if (profile["ebitda_applicable"] and metrics["ebitda_margin_curr"] is not None and peer_medians.get("ebitda_margin_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "Net Profit Margin (%)",
            "Current / Recent (YoY)": fmt_level_pct(metrics["net_margin_curr"]),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(metrics["net_margin_past"]),
            "Momentum (Current vs Past)": (
                f"🟢 Expanding ({metrics['net_margin_curr'] - metrics['net_margin_past']:+.2f}%)"
                if (metrics["net_margin_curr"] is not None and metrics["net_margin_past"] is not None and metrics["net_margin_curr"] >= metrics["net_margin_past"])
                else (
                    f"🔴 Compressing ({metrics['net_margin_curr'] - metrics['net_margin_past']:+.2f}%)"
                    if (metrics["net_margin_curr"] is not None and metrics["net_margin_past"] is not None)
                    else "N/A"
                )
            ),
            "Industry Peer Median": fmt_level_pct(peer_medians.get("net_margin_med")),
            "vs. Industry": (
                "🏆 Higher Margin" if (metrics["net_margin_curr"] is not None and peer_medians.get("net_margin_med") is not None and metrics["net_margin_curr"] >= peer_medians["net_margin_med"])
                else ("📉 Below Peer Margin" if (metrics["net_margin_curr"] is not None and peer_medians.get("net_margin_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "Diluted EPS Growth Rate",
            "Current / Recent (YoY)": fmt_pct(eps_yoy),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_pct(eps_cagr),
            "Momentum (Current vs Past)": (
                "🟢 Accelerating" if (eps_yoy is not None and eps_cagr is not None and eps_yoy >= eps_cagr)
                else ("🟡 Decelerating" if (eps_yoy is not None and eps_cagr is not None) else "N/A")
            ),
            "Industry Peer Median": f"Net Income YoY: {fmt_pct(ni_yoy)}",
            "vs. Industry": bb_verdict,
        },
        {
            "Metric": "Return on Capital Employed (ROCE)",
            "Current / Recent (YoY)": fmt_level_pct(metrics["roce_pct"]) if profile["roce_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(metrics["roce_prev_pct"]) if profile["roce_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": (
                "🟢 Improving Efficiency" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and metrics["roce_prev_pct"] is not None and metrics["roce_pct"] >= metrics["roce_prev_pct"])
                else ("🟡 Lower vs Past Avg" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and metrics["roce_prev_pct"] is not None) else "N/A")
            ),
            "Industry Peer Median": fmt_level_pct(peer_medians.get("roce_med")) if profile["roce_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Superior ROCE" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and peer_medians.get("roce_med") is not None and metrics["roce_pct"] >= peer_medians["roce_med"])
                else ("📉 Below Peer ROCE" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and peer_medians.get("roce_med") is not None) else "N/A")
            ),
        },
    ]

    st.markdown("#### 📈 Current vs. Past Growth & Margin Trajectory (Benchmarked vs. Industry)")
    st.dataframe(pd.DataFrame(comparison_rows).set_index("Metric"), use_container_width=True)

    if not df_peers.empty:
        with st.expander("🏢 View Live Sector Peer Comparison Table", expanded=False):
            st.dataframe(df_peers.set_index("Ticker").round(2), use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 3: 2-STAGE DCF & BENJAMIN GRAHAM 3-SCENARIO VALUATION
    # --------------------------------------------------------------------------
    st.header("3. Intrinsic Value & 3-Scenario DCF / Graham Calculator")

    dcf_inputs = extract_dcf_inputs(bundle, metrics)
    applied_g1 = custom_g1 if use_custom_params else dcf_inputs["base_g1_pct"]
    applied_r = custom_discount if use_custom_params else dcf_inputs["base_discount_pct"]
    applied_term_g = custom_term_g if use_custom_params else dcf_inputs["base_terminal_g_pct"]

    scenarios = build_three_scenarios(
        dcf_inputs=dcf_inputs,
        base_g1_pct=applied_g1,
        base_discount_pct=applied_r,
        base_terminal_g_pct=applied_term_g,
        current_price=current_price,
    )
    base_res = scenarios["Base Case"]

    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.metric("Current Market Price", format_price(current_price, currency))
    with vc2:
        st.metric(
            "2-Stage DCF Value (Base)",
            format_price(base_res["dcf_intrinsic"], currency),
            delta=f"{base_res['dcf_mos_pct']:+.1f}% MoS" if base_res["dcf_mos_pct"] is not None else "N/A",
            delta_color="normal" if (base_res["dcf_mos_pct"] or -1) > 0 else "inverse",
        )
    with vc3:
        st.metric(
            "Graham Value (Base)",
            format_price(base_res["graham_intrinsic"], currency),
            delta=f"{base_res['graham_mos_pct']:+.1f}% MoS" if base_res["graham_mos_pct"] is not None else "N/A",
            delta_color="normal" if (base_res["graham_mos_pct"] or -1) > 0 else "inverse",
        )
    with vc4:
        st.metric(
            "Blended Intrinsic Value (Base)",
            format_price(base_res["blended_intrinsic"], currency),
            delta=f"{base_res['blended_mos_pct']:+.1f}% Blended MoS" if base_res["blended_mos_pct"] is not None else "N/A",
            delta_color="normal" if (base_res["blended_mos_pct"] or -1) > 0 else "inverse",
        )

    scenario_rows = []
    for scen_label in ["Conservative", "Base Case", "Optimistic"]:
        s = scenarios[scen_label]
        scenario_rows.append(
            {
                "Scenario": scen_label,
                "Stage-1 Growth (g₁)": f"{s['g1_pct']:.2f}%",
                "Discount Rate (r)": f"{s['discount_pct']:.2f}%",
                "Terminal Growth": f"{s['terminal_g_pct']:.2f}%",
                "2-Stage DCF / Share": format_price(s["dcf_intrinsic"], currency),
                "Graham Value / Share": format_price(s["graham_intrinsic"], currency),
                "Blended Value / Share": format_price(s["blended_intrinsic"], currency),
                "Blended Margin of Safety": f"{s['blended_mos_pct']:+.2f}%" if s["blended_mos_pct"] is not None else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows).set_index("Scenario"), use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 4: CORPORATE NEWS SENTIMENT & FINAL VERDICT
    # --------------------------------------------------------------------------
    st.header("4. Long-Term News Sentiment & Final Investment Verdict")
    news_summary = analyze_corporate_news(bundle["raw_news"], max_items=7)

    if news_summary["selected_count"] > 0:
        nc1, nc2, nc3 = st.columns(3)
        for col, label, icon in zip([nc1, nc2, nc3], ["Positive", "Neutral", "Negative"], ["🟢", "⚪", "🔴"]):
            with col:
                items = news_summary["grouped"][label]
                st.markdown(f"#### {icon} {label} ({len(items)})")
                for item in items:
                    st.markdown(f"- **[{item['title']}]({item['link']})** (*{item['publisher']} • `{item['compound']:+.2f}`*)")

    decision = compute_decision_verdict(
        ticker=target_ticker,
        company_name=company_name,
        currency=currency,
        profile=profile,
        metrics=metrics,
        peer_medians=peer_medians,
        scenarios=scenarios,
        news_summary=news_summary,
    )

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("Fundamentals, ROCE, Peers & Valuation (70%)", f"{decision['fundamental_score_70']:.1f} / 70.0")
    with dc2:
        st.metric("Corporate Catalyst Sentiment (30%)", f"{decision['sentiment_score_30']:.1f} / 30.0")
    with dc3:
        st.metric("Composite Fundamental Value Score", f"{decision['total_score']:.1f} / 100")

    st.progress(int(round(decision["total_score"])))

    st.markdown(
        f"""
        <div style="
            background-color: #0f172a;
            border-left: 8px solid {decision['badge_color']};
            padding: 24px;
            border-radius: 8px;
            margin-top: 16px;
        ">
            <h2 style="color: #f8fafc; margin: 0 0 12px 0; font-weight: 800;">
                LONG-TERM INVESTMENT VERDICT:
                <span style="color: {decision['badge_color']};">{decision['verdict']}</span>
            </h2>
            <p style="color: #e2e8f0; font-size: 1.06rem; line-height: 1.6; margin: 0;">
                {decision['thesis']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
