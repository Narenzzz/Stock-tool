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


def format_shares(shares_val: Optional[float], currency_code: str = "USD") -> str:
    if shares_val is None or np.isnan(shares_val) or shares_val <= 0:
        return "N/A"
    if (currency_code or "").upper() == "INR":
        return f"{shares_val / 1e7:,.2f} Cr"
    return f"{shares_val / 1e9:,.2f}B"


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

    financial_keywords = ["bank", "financial", "insurance", "credit", "capital markets", "nbfc", "lending"]
    is_financial = any(k in sec_lower or k in ind_lower for k in financial_keywords) or any(
        k in t_upper for k in ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "AXISBANK", "KOTAKBANK", "JPM", "BAC"]
    )

    if is_financial:
        archetype = "BANKING_NBFC"
    elif any(k in sec_lower or k in ind_lower for k in ["technology", "software", "it services", "semiconductor", "internet"]) or any(
        k in t_upper for k in ["TCS.", "INFY.", "WIPRO.", "HCLTECH.", "TECHM.", "AAPL", "MSFT", "NVDA", "GOOG"]
    ):
        archetype = "IT_TECH"
    elif any(k in sec_lower or k in ind_lower for k in ["consumer defensive", "fmcg", "packaged foods", "household", "beverages", "tobacco"]) or any(
        k in t_upper for k in ["ITC.", "HINDUNILVR.", "NESTLEIND.", "BRITANNIA.", "PG", "KO"]
    ):
        archetype = "FMCG_CONSUMER"
    elif any(k in sec_lower or k in ind_lower for k in ["healthcare", "pharma", "biotech", "medical", "diagnostics"]) or any(
        k in t_upper for k in ["SUNPHARMA.", "DRREDDY.", "CIPLA.", "DIVISLAB.", "JNJ"]
    ):
        archetype = "PHARMA_HEALTH"
    elif any(k in sec_lower or k in ind_lower for k in ["energy", "oil", "gas", "utilities", "power", "renewable"]) or any(
        k in t_upper for k in ["ONGC.", "NTPC.", "POWERGRID.", "XOM", "CVX"]
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
                "Financial Institution Detected: Deposits & borrowings are operational raw material for lenders. "
                "EBITDA, ROCE, and Debt-to-Equity are marked N/A; analysis prioritizes ROE (>=14%), ROA (>=1.2%), "
                "Net Profit Margin, and Price-to-Book (P/B)."
            ),
        }
    return {
        "is_financial": False,
        "archetype": archetype,
        "profile_name": f"Operating Business ({sector or 'Manufacturing / Services'})",
        "roce_applicable": True,
        "ebitda_applicable": True,
        "de_applicable": True,
        "applicable_note": (
            "Standard Operating Company: Full analysis active across ROE, ROCE (EBIT / Capital Employed), "
            "EBITDA Margin, Debt-to-Equity, Free Cash Flow conversion, and Share Buyback EPS Quality."
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

    fast_price = None
    fast_mcap = None
    fast_shares = None
    fast_currency = None
    try:
        fi = ticker_obj.fast_info
        if fi is not None:
            fast_price = safe_float(getattr(fi, "last_price", None)) or safe_float(getattr(fi, "previous_close", None))
            fast_mcap = safe_float(getattr(fi, "market_cap", None))
            fast_shares = safe_float(getattr(fi, "shares", None))
            fast_currency = getattr(fi, "currency", None)
    except Exception:
        pass

    current_price = (
        safe_float(info.get("currentPrice"))
        or safe_float(info.get("regularMarketPrice"))
        or safe_float(info.get("previousClose"))
        or fast_price
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

    currency = info.get("currency") or fast_currency
    if not currency:
        currency = "INR" if ticker_clean.endswith((".NS", ".BO")) else "USD"

    try:
        income_stmt = ticker_obj.financials
        if income_stmt is None or income_stmt.empty:
            income_stmt = ticker_obj.income_stmt
    except Exception:
        income_stmt = pd.DataFrame()

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

    sector = info.get("sector") or ("Technology" if ("TCS" in ticker_clean or "INFY" in ticker_clean) else "Industrials")
    industry = info.get("industry") or "General"
    profile = classify_industry_profile(sector, industry, ticker_clean)

    return {
        "ticker": ticker_clean,
        "currency": str(currency).upper(),
        "info": info,
        "fast_mcap": fast_mcap,
        "fast_shares": fast_shares,
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
    info = bundle["info"]
    inc = bundle["income_stmt"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    profile = bundle["profile"]

    rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    gp_s = get_statement_row(inc, ["Gross Profit"])
    ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA"])
    ebit_s = get_statement_row(inc, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
    ni_s = get_statement_row(
        inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
    )
    eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
    shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])
    if shares_s is None:
        shares_s = get_statement_row(bs, ["Ordinary Shares Number", "Share Issued", "Common Stock"])

    assets_s = get_statement_row(bs, ["Total Assets"])
    curr_liab_s = get_statement_row(bs, ["Current Liabilities", "Total Current Liabilities"])
    curr_assets_s = get_statement_row(bs, ["Current Assets", "Total Current Assets"])
    equity_s = get_statement_row(
        bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]
    )
    debt_s = get_statement_row(
        bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]
    )

    fcf_s = get_statement_row(cf, ["Free Cash Flow"])
    ocf_s = get_statement_row(
        cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
    )
    capex_s = get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])
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

    # 4. Debt-to-Equity, Current Ratio, Price-to-Book
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
    pe_ratio = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))

    # Latest OCF, FCF & Net Income
    latest_ni = safe_float(ni_s.dropna().iloc[0]) if (ni_s is not None and not ni_s.dropna().empty) else safe_float(info.get("netIncomeToCommon"))
    latest_ocf = safe_float(ocf_s.dropna().iloc[0]) if (ocf_s is not None and not ocf_s.dropna().empty) else safe_float(info.get("operatingCashflow"))
    latest_fcf = safe_float(fcf_s.dropna().iloc[0]) if (fcf_s is not None and not fcf_s.dropna().empty) else safe_float(info.get("freeCashflow"))
    if latest_fcf is None and latest_ocf is not None and capex_s is not None and not capex_s.dropna().empty:
        cx_0 = safe_float(capex_s.dropna().iloc[0], 0.0) or 0.0
        latest_fcf = latest_ocf + cx_0 if cx_0 < 0 else latest_ocf - cx_0

    ocf_to_ni_ratio = (latest_ocf / latest_ni) if (latest_ocf is not None and latest_ni and latest_ni > 0) else None

    # 5. Multi-Year Growth & Margin Comparison
    def extract_growth_profile(series: Optional[pd.Series], info_yoy_key: Optional[str] = None) -> Dict[str, Optional[float]]:
        recent_yoy, past_cagr, latest_val, prev_val = None, None, None, None
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

        return {"latest": latest_val, "prev": prev_val, "recent_yoy_pct": recent_yoy, "past_cagr_pct": past_cagr}

    rev_growth = extract_growth_profile(rev_s, "revenueGrowth")
    ebitda_growth = extract_growth_profile(ebitda_s) if profile["ebitda_applicable"] else {
        "latest": None, "prev": None, "recent_yoy_pct": None, "past_cagr_pct": None
    }
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

    if gross_margin_curr is None:
        gm_info = safe_float(info.get("grossMargins"))
        if gm_info is not None:
            gross_margin_curr = gm_info * 100.0
    if ebitda_margin_curr is None and profile["ebitda_applicable"]:
        em_info = safe_float(info.get("ebitdaMargins"))
        if em_info is not None:
            ebitda_margin_curr = em_info * 100.0
    if net_margin_curr is None:
        nm_info = safe_float(info.get("profitMargins"))
        if nm_info is not None:
            net_margin_curr = nm_info * 100.0

    # 6. Share Buyback & EPS Quality Detector
    share_change_pct = None
    latest_stmt_shares = None
    if shares_s is not None and not shares_s.dropna().empty:
        clean_sh = shares_s.dropna()
        latest_stmt_shares = safe_float(clean_sh.iloc[0])
        if len(clean_sh) >= 2:
            oldest_shares = safe_float(clean_sh.iloc[min(2, len(clean_sh) - 1)])
            if latest_stmt_shares and oldest_shares and oldest_shares > 0:
                share_change_pct = ((latest_stmt_shares - oldest_shares) / oldest_shares) * 100.0

    latest_stmt_eps = None
    if eps_s is not None and not eps_s.dropna().empty:
        latest_stmt_eps = safe_float(eps_s.dropna().iloc[0])

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
        bb_cash_str = format_large_currency(total_3yr_buyback_cash, bundle["currency"])
        buyback_explanation = (
            f"Share count changed by {sh_str} over the multi-year window "
            f"(3-Yr Buybacks: {bb_cash_str}). "
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
        "pb_ratio": pb_ratio,
        "pe_ratio": pe_ratio,
        "latest_fcf": latest_fcf,
        "latest_ocf": latest_ocf,
        "latest_ni": latest_ni,
        "latest_stmt_shares": latest_stmt_shares,
        "latest_stmt_eps": latest_stmt_eps,
        "ocf_to_ni_ratio": ocf_to_ni_ratio,
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
                "Diluted Shares": format_shares(sh_v, currency),
                "Free Cash Flow": format_large_currency(fcf, currency),
            }
        )

    df_hist = pd.DataFrame(records)
    if not df_hist.empty:
        df_hist = df_hist.set_index("Fiscal Year")
    return df_hist


# ==============================================================================
# 3. SECTOR-SPECIFIC VALUE INVESTING CHECKLIST ENGINE
# ==============================================================================
def evaluate_sector_checklist(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    archetype = bundle["profile"]["archetype"]
    currency = bundle["currency"]

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

    quant_checks = []
    qual_questions = []
    sector_title = ""
    moat_focus = ""

    if archetype == "MANUFACTURING_INDUSTRIAL":
        sector_title = "🏭 Manufacturing, Auto & Heavy Industrials Checklist"
        moat_focus = "Capital efficiency on physical plants (ROCE), debt discipline across cycles, and positive Free Cash Flow after CapEx."
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
                "Actual Value": f"{de:.2f}x" if de is not None else "N/A",
                "Status": status_badge(de <= 0.80 if de is not None else None, borderline=(de is not None and 0.80 < de <= 1.20)),
                "Why It Matters in This Sector": "Prevents interest payments from wiping out profits during industrial downturns.",
            },
            {
                "Parameter": "Free Cash Flow After CapEx",
                "Target Benchmark": "Positive (> 0)",
                "Actual Value": format_large_currency(fcf, currency),
                "Status": status_badge(fcf > 0 if fcf is not None else None),
                "Why It Matters in This Sector": "Checks if heavy plant CapEx leaves real cash for shareholders (explains low DCF vs Graham).",
            },
            {
                "Parameter": "Operating EBITDA Margin Stability",
                "Target Benchmark": ">= 12.0% & Stable vs 3-Yr Avg",
                "Actual Value": em_str,
                "Status": status_badge(em >= 12.0 if em is not None else None, borderline=(em is not None and 8.0 <= em < 12.0)),
                "Why It Matters in This Sector": "Demonstrates ability to pass raw material cost spikes on to customers.",
            },
        ]
        qual_questions = [
            "CapEx Reinvestment Check: Is recent plant/machinery CapEx expanding profitable capacity rather than just maintaining old equipment?",
            "Working Capital Check: Are receivables and inventory growing slower than or in line with revenue growth?",
            "Cyclicality Buffer: Can the company survive a 20% commodity/demand slowdown without raising new debt?",
        ]

    elif archetype == "BANKING_NBFC":
        sector_title = "🏦 Banking, NBFCs & Financial Services Checklist"
        moat_focus = "Asset quality (low NPAs), Return on Assets (ROA), compounding book value (ROE), and reasonable Price-to-Book multiple."
        quant_checks = [
            {
                "Parameter": "Return on Assets (ROA)",
                "Target Benchmark": ">= 1.20% (Elite >= 1.5%)",
                "Actual Value": f"{roa:.2f}%" if roa is not None else "N/A",
                "Status": status_badge(roa >= 1.20 if roa is not None else None, borderline=(roa is not None and 0.90 <= roa < 1.20)),
                "Why It Matters in This Sector": "The single most important gauge of a lender's underwriting profitability.",
            },
            {
                "Parameter": "Return on Equity (ROE)",
                "Target Benchmark": ">= 14.0%",
                "Actual Value": f"{roe:.2f}%" if roe is not None else "N/A",
                "Status": status_badge(roe >= 14.0 if roe is not None else None, borderline=(roe is not None and 11.5 <= roe < 14.0)),
                "Why It Matters in This Sector": "Measures how fast the bank compounds shareholder book value per share.",
            },
            {
                "Parameter": "Price-to-Book Ratio (P/B)",
                "Target Benchmark": "<= 3.0x (Value < 2.2x)",
                "Actual Value": f"{pb:.2f}x" if pb is not None else "N/A",
                "Status": status_badge(pb <= 3.0 if pb is not None else None, borderline=(pb is not None and 3.0 < pb <= 4.2)),
                "Why It Matters in This Sector": "Banks are valued on tangible book value; overpaying for book erodes Margin of Safety.",
            },
            {
                "Parameter": "Net Profit Margin",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{nm:.2f}%" if nm is not None else "N/A",
                "Status": status_badge(nm >= 15.0 if nm is not None else None, borderline=(nm is not None and 10.0 <= nm < 15.0)),
                "Why It Matters in This Sector": "Reflects Net Interest Margin (NIM) strength after credit provisions.",
            },
        ]
        qual_questions = [
            "Asset Quality (GNPA / NNPA): Are Gross and Net Non-Performing Assets stable or declining over the last 4 quarters?",
            "Liability Franchise (CASA): Does the lender have low-cost sticky deposits (or AAA credit rating for NBFCs)?",
            "Capital Adequacy (CRAR): Is Tier-1 Capital well above regulatory minimums so frequent equity dilution isn't needed?",
        ]

    elif archetype == "IT_TECH":
        sector_title = "💻 Information Technology, Software & Platform Checklist"
        moat_focus = "Asset-light scalability, high client switching costs, strong cash conversion (OCF/Net Income), and organic EPS growth."
        roe_roce_str = f"ROE: {roe:.1f}% | ROCE: {roce:.1f}%" if (roe is not None and roce is not None) else (f"ROE: {roe:.1f}%" if roe is not None else "N/A")
        quant_checks = [
            {
                "Parameter": "Return on Equity & ROCE",
                "Target Benchmark": "ROE >= 20% & ROCE >= 20%",
                "Actual Value": roe_roce_str,
                "Status": status_badge((roe or 0) >= 20.0 and (roce or 20.0) >= 20.0 if (roe is not None or roce is not None) else None, borderline=((roe or 0) >= 15.0)),
                "Why It Matters in This Sector": "Asset-light tech leaders should generate exceptional returns on minimal physical capital.",
            },
            {
                "Parameter": "Cash Conversion (Operating Cash Flow / Net Income)",
                "Target Benchmark": ">= 0.85x (85%+ conversion)",
                "Actual Value": f"{ocf_ni:.2f}x" if ocf_ni is not None else "N/A",
                "Status": status_badge(ocf_ni >= 0.85 if ocf_ni is not None else None, borderline=(ocf_ni is not None and 0.70 <= ocf_ni < 0.85)),
                "Why It Matters in This Sector": "Confirms software/IT accounting profits are turning into real collectible cash.",
            },
            {
                "Parameter": "Balance Sheet Leverage (D/E)",
                "Target Benchmark": "<= 0.40x (Near Zero Debt)",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x (Debt-Free)",
                "Status": status_badge((de or 0.0) <= 0.40, borderline=(de is not None and 0.40 < de <= 0.80)),
                "Why It Matters in This Sector": "Quality tech compounders self-fund R&D from internal cash flows without needing debt.",
            },
            {
                "Parameter": "Operating / EBITDA Margin",
                "Target Benchmark": ">= 18.0%",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 18.0 if em is not None else None, borderline=(em is not None and 14.0 <= em < 18.0)),
                "Why It Matters in This Sector": "Protects profitability against wage hikes and subcontractor inflation.",
            },
        ]
        qual_questions = [
            "Order Book & Deal Pipeline: Is Total Contract Value (TCV) or recurring subscription revenue growing steadily?",
            "AI & Tech Obsolescence Moat: Are client switching costs high enough to prevent pricing compression?",
            "Share Count Discipline: Are employee stock options (ESOPs) reasonable rather than diluting shareholders?",
        ]

    elif archetype == "FMCG_CONSUMER":
        sector_title = "🛒 FMCG, Consumer Goods & Retail Brand Checklist"
        moat_focus = "Brand pricing power (Gross Margin), distribution reach, negative/low working capital, and high ROCE."
        quant_checks = [
            {
                "Parameter": "Gross Profit Margin (Brand Pricing Power)",
                "Target Benchmark": ">= 40.0% (Strong Brand Moat)",
                "Actual Value": f"{gm:.2f}%" if gm is not None else "N/A",
                "Status": status_badge(gm >= 40.0 if gm is not None else None, borderline=(gm is not None and 30.0 <= gm < 40.0)),
                "Why It Matters in This Sector": "High gross margins prove customers pay a premium for the brand over generic rivals.",
            },
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 22.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 22.0 if roce is not None else None, borderline=(roce is not None and 15.0 <= roce < 22.0)),
                "Why It Matters in This Sector": "Hallmark of an enduring Buffett-style consumer monopoly.",
            },
            {
                "Parameter": "Free Cash Flow Conversion (OCF / Net Income)",
                "Target Benchmark": ">= 0.85x",
                "Actual Value": f"{ocf_ni:.2f}x" if ocf_ni is not None else "N/A",
                "Status": status_badge(ocf_ni >= 0.85 if ocf_ni is not None else None, borderline=(ocf_ni is not None and 0.70 <= ocf_ni < 0.85)),
                "Why It Matters in This Sector": "Ensures inventory isn't piling up unsold at distributors.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 0.50x",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 0.50, borderline=(de is not None and 0.50 < de <= 0.90)),
                "Why It Matters in This Sector": "Consumer staples should generate surplus cash and require minimal debt.",
            },
        ]
        qual_questions = [
            "Volume vs. Price Growth: Is revenue growing from real unit volume expansion, not just price hikes?",
            "Market Share Defense: Is the flagship brand holding or gaining market share against D2C/private-label entrants?",
            "Valuation Sanity Check: Even if the brand is elite, is the P/E multiple reasonable relative to its 10–12% growth rate?",
        ]

    elif archetype == "PHARMA_HEALTH":
        sector_title = "💊 Pharmaceuticals, Healthcare & Diagnostics Checklist"
        moat_focus = "R&D pipeline productivity, high gross margins, regulatory compliance (USFDA/domestic), and healthy working capital."
        quant_checks = [
            {
                "Parameter": "Gross Profit Margin",
                "Target Benchmark": ">= 50.0% (Specialty/Formulations)",
                "Actual Value": f"{gm:.2f}%" if gm is not None else "N/A",
                "Status": status_badge(gm >= 50.0 if gm is not None else None, borderline=(gm is not None and 38.0 <= gm < 50.0)),
                "Why It Matters in This Sector": "Provides the buffer needed to fund continuous R&D and clinical/ANDA filings.",
            },
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 15.0 if roce is not None else None, borderline=(roce is not None and 11.5 <= roce < 15.0)),
                "Why It Matters in This Sector": "Validates that manufacturing plants and R&D spend earn strong returns.",
            },
            {
                "Parameter": "Current Ratio (Working Capital Liquidity)",
                "Target Benchmark": ">= 1.50x",
                "Actual Value": f"{cr:.2f}x" if cr is not None else "N/A",
                "Status": status_badge(cr >= 1.50 if cr is not None else None, borderline=(cr is not None and 1.15 <= cr < 1.50)),
                "Why It Matters in This Sector": "Pharma companies carry longer inventory & receivable cycles.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 0.60x",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 0.60, borderline=(de is not None and 0.60 < de <= 1.0)),
                "Why It Matters in This Sector": "Avoids debt distress if a plant faces regulatory/FDA import delays.",
            },
        ]
        qual_questions = [
            "Regulatory / FDA Cleanliness: Are manufacturing facilities free from unresolved FDA Warning Letters or import alerts?",
            "Product Concentration Check: Is revenue diversified across therapies rather than reliant on a single windfall product/tender?",
            "R&D & Patent Cliff: Is the complex generics / domestic branded pipeline strong enough to offset pricing erosion?",
        ]

    else:
        sector_title = "⚡ Energy, Oil & Gas, Power & Utilities Checklist"
        moat_focus = "Low production cost, regulated/contracted cash flows, manageable debt-to-capital, and sustainable dividend/FCF yield."
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 12.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 12.0 if roce is not None else None, borderline=(roce is not None and 9.0 <= roce < 12.0)),
                "Why It Matters in This Sector": "Capital-heavy utilities and energy assets must clear their cost of capital.",
            },
            {
                "Parameter": "EBITDA Margin",
                "Target Benchmark": ">= 18.0%",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 18.0 if em is not None else None, borderline=(em is not None and 12.0 <= em < 18.0)),
                "Why It Matters in This Sector": "Indicates operating cushion against fuel and commodity price swings.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 1.20x (Infrastructure Standard)",
                "Actual Value": f"{de:.2f}x" if de is not None else "N/A",
                "Status": status_badge(de <= 1.20 if de is not None else None, borderline=(de is not None and 1.20 < de <= 1.80)),
                "Why It Matters in This Sector": "Utilities use project debt, but excessive leverage threatens dividend safety.",
            },
            {
                "Parameter": "Operating Cash Flow Generation",
                "Target Benchmark": "Positive (> 0)",
                "Actual Value": format_large_currency(metrics.get("latest_ocf"), currency),
                "Status": status_badge((metrics.get("latest_ocf") or 0) > 0 if metrics.get("latest_ocf") is not None else None),
                "Why It Matters in This Sector": "Funds ongoing grid/refinery CapEx and shareholder dividends.",
            },
        ]
        qual_questions = [
            "Cycle Peak Check: Are current earnings reflecting normal mid-cycle refining/energy margins rather than a temporary commodity spike?",
            "Receivables from Discoms/Offtakers: Are payments from state utilities or industrial buyers collected on time?",
            "Energy Transition CapEx: Is green/new-energy CapEx earning clear visibility on returns?",
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
# 4. LIGHTWEIGHT INDUSTRY PEER BENCHMARKING ENGINE
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_industry_peer_benchmarks(
    target_ticker: str, sector: str, currency: str, custom_peers_str: str = ""
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    is_india = currency == "INR" or target_ticker.endswith((".NS", ".BO"))
    peer_map = INDIA_SECTOR_PEERS if is_india else US_SECTOR_PEERS
    default_peers = peer_map.get(
        sector,
        ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"] if is_india else ["AAPL", "MSFT", "GOOG", "JNJ"],
    )

    if custom_peers_str.strip():
        raw_list = [p.strip().upper() for p in custom_peers_str.split(",") if p.strip()]
        peer_tickers = [f"{p}.NS" if (is_india and "." not in p) else p for p in raw_list]
    else:
        peer_tickers = [p for p in default_peers if p != target_ticker][:3]

    rows = []
    for pt in peer_tickers:
        try:
            t_obj = yf.Ticker(pt)
            p_info = t_obj.info or {}
            if not p_info:
                continue

            rev_g = safe_float(p_info.get("revenueGrowth"))
            ebitda_m = safe_float(p_info.get("ebitdaMargins"))
            net_m = safe_float(p_info.get("profitMargins"))
            roe_v = safe_float(p_info.get("returnOnEquity"))
            roa_v = safe_float(p_info.get("returnOnAssets"))
            pe_v = safe_float(p_info.get("trailingPE")) or safe_float(p_info.get("forwardPE"))

            roce_approx = (roe_v * 0.88 * 100.0) if roe_v is not None else (roa_v * 1.4 * 100.0 if roa_v is not None else np.nan)

            rows.append(
                {
                    "Ticker": pt,
                    "Company": (p_info.get("shortName") or pt)[:24],
                    "Rev Growth (%)": rev_g * 100.0 if rev_g is not None else np.nan,
                    "EBITDA Margin (%)": ebitda_m * 100.0 if ebitda_m is not None else np.nan,
                    "Net Margin (%)": net_m * 100.0 if net_m is not None else np.nan,
                    "ROE (%)": roe_v * 100.0 if roe_v is not None else np.nan,
                    "ROCE Est (%)": roce_approx,
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
        medians["roce_med"] = safe_float(df_peers["ROCE Est (%)"].median(skipna=True))
        medians["pe_med"] = safe_float(df_peers["Trailing P/E"].median(skipna=True))

    return df_peers, medians


# ==============================================================================
# 5. BULLETPROOF 2-STAGE DCF & BENJAMIN GRAHAM SCENARIO ENGINE
# ==============================================================================
def extract_dcf_inputs(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    info = bundle["info"]
    inc = bundle["income_stmt"]
    bs = bundle["balance_sheet"]
    cf = bundle["cash_flow"]
    currency = bundle["currency"]
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

    eps_ttm = (
        safe_float(info.get("trailingEps"))
        or safe_float(info.get("forwardEps"))
        or metrics.get("latest_stmt_eps")
    )
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
        cash_s = get_statement_row(
            bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]
        )
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
    eqg_info = safe_float(info.get("earningsQuarterlyGrowth"))

    candidates = [
        v
        for v in [
            eg_info * 100.0 if eg_info is not None else None,
            rg_info * 100.0 if rg_info is not None else None,
            eqg_info * 100.0 if eqg_info is not None else None,
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
# 6. CORPORATE NEWS FILTERING & VADER SENTIMENT ENGINE
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
# 7. INDUSTRY-ADAPTIVE DECISION ENGINE
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
    checklist_result: Dict[str, Any],
    user_confirmed_count: int,
) -> Dict[str, Any]:
    base_case = scenarios["Base Case"]
    cons_case = scenarios["Conservative"]
    mos = base_case.get("blended_mos_pct")
    cons_mos = cons_case.get("blended_mos_pct")

    roe = metrics.get("roe_pct")
    roce = metrics.get("roce_pct")
    roa = metrics.get("roa_pct")

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

    roe_pts = min(10.0, max(0.0, ((roe or 0.0) / 20.0) * 10.0)) if roe is not None else 5.0
    if profile["roce_applicable"]:
        roce_pts = min(10.0, max(0.0, ((roce or 0.0) / 20.0) * 10.0)) if roce is not None else 5.0
    else:
        roce_pts = min(10.0, max(0.0, ((roa or 0.0) / 1.5) * 10.0)) if roa is not None else 5.0
    efficiency_pts = roe_pts + roce_pts

    rev_yoy = metrics["rev_growth"]["recent_yoy_pct"]
    rev_cagr = metrics["rev_growth"]["past_cagr_pct"]
    growth_pts = 4.0
    if rev_yoy is not None and rev_yoy > 8.0:
        growth_pts += 2.5
    if rev_yoy is not None and rev_cagr is not None and rev_yoy >= rev_cagr:
        growth_pts += 2.0

    nm_curr = metrics.get("net_margin_curr")
    nm_past = metrics.get("net_margin_past")
    if nm_curr is not None and nm_past is not None and nm_curr >= nm_past:
        growth_pts += 2.0

    if metrics["buyback_verdict"] == "100% ORGANIC EPS GROWTH":
        growth_pts += 2.0
    elif metrics["buyback_verdict"] == "BUYBACK-BOOSTED EPS":
        growth_pts += 1.0

    growth_pts += min(2.5, user_confirmed_count * (2.5 / 3.0))
    growth_pts = min(15.0, growth_pts)

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

    fundamental_score_70 = mos_pts + efficiency_pts + growth_pts + bs_peer_pts

    avg_comp = news_summary.get("avg_compound", 0.0)
    clamped_comp = float(np.clip(avg_comp, -0.5, 0.5))
    sentiment_score_30 = ((clamped_comp + 0.5) / 1.0) * 30.0

    total_score = float(np.clip(fundamental_score_70 + sentiment_score_30, 0.0, 100.0))

    if total_score >= 75.0 and (mos is not None and mos >= 15.0) and checklist_result["pass_count"] >= 3:
        verdict, badge_color = "STRONG BUY", "#16a34a"
    elif total_score >= 60.0 and (mos is not None and mos >= 0.0):
        verdict, badge_color = "ACCUMULATE", "#2563eb"
    elif total_score >= 45.0:
        verdict, badge_color = "HOLD", "#d97706"
    else:
        verdict, badge_color = "OVERVALUED / AVOID", "#dc2626"

    roe_str = f"{roe:.1f}%" if roe is not None else "N/A"
    roce_str = f", ROCE of {roce:.1f}%," if (profile["roce_applicable"] and roce is not None) else ""
    base_iv_str = format_price(base_case.get("blended_intrinsic"), currency)
    cons_iv_str = format_price(cons_case.get("blended_intrinsic"), currency)
    mos_str = f"{mos:+.1f}%" if mos is not None else "indeterminate"

    sentence_1 = (
        f"{company_name} ({ticker}) passes {checklist_result['pass_count']}/{checklist_result['total_checks']} "
        f"automated sector checklist benchmarks with a trailing ROE of {roe_str}{roce_str} and "
        f"{metrics['buyback_verdict'].lower()}."
    )
    sentence_2 = (
        f"Blending the 2-Stage DCF and Benjamin Graham models yields a Base-Case Intrinsic Value of {base_iv_str} "
        f"({mos_str} Margin of Safety) and a Conservative Bear-Case floor of {cons_iv_str}, alongside a "
        f"corporate news sentiment score of {sentiment_score_30:.1f}/30 pts."
    )

    risk_factors = []
    if checklist_result["pass_count"] <= 2:
        failed_num = checklist_result["total_checks"] - checklist_result["pass_count"]
        risk_factors.append(f"failing {failed_num} mandatory sector checklist thresholds")
    if rev_yoy is not None and rev_cagr is not None and rev_yoy < rev_cagr - 3.0:
        risk_factors.append("recent top-line revenue deceleration relative to its 3-year CAGR")
    if metrics["buyback_verdict"] == "BUYBACK-BOOSTED EPS":
        risk_factors.append("EPS growth partially flattered by share buybacks")
    if cons_mos is not None and cons_mos < -15.0:
        risk_factors.append("downside vulnerability if growth slows toward the Conservative scenario")
    if mos is not None and mos < 0:
        risk_factors.append("multiple compression from trading above Base-Case intrinsic value")
    if not risk_factors:
        risk_factors.append("macroeconomic demand cyclicality and terminal multiple normalization")

    sentence_3 = f"Core long-term risk factors to monitor before allocating capital include {', '.join(risk_factors)}."

    return {
        "fundamental_score_70": fundamental_score_70,
        "sentiment_score_30": sentiment_score_30,
        "total_score": total_score,
        "verdict": verdict,
        "badge_color": badge_color,
        "thesis": f"{sentence_1} {sentence_2} {sentence_3}",
    }


# ==============================================================================
# 8. STREAMLIT USER INTERFACE
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
            help="Type any ticker (e.g., TCS, HDFCBANK, BMW, CUPID, AAPL). In India mode, .NS or .BO is appended automatically if omitted.",
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
        st.subheader("📐 Valuation & DCF Overrides")
        use_custom_params = st.checkbox("Override Automatic Growth & WACC Assumptions", value=False)
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

        st.markdown("---")
        st.subheader("🏢 Industry Peer Benchmark")
        enable_peers = st.checkbox("Fetch Live Sector Peer Medians", value=True)
        custom_peers_input = st.text_input(
            "Custom Peer Tickers (optional)",
            value="",
            placeholder="e.g., INFY.NS, HCLTECH.NS",
        )

        analyze_clicked = st.button("🔍 Analyze Fundamental Value", type="primary", use_container_width=True)
        if st.button("🔄 Clear Cache & Force Fresh Fetch", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = resolved_ticker
    if analyze_clicked or resolved_ticker != st.session_state.get("last_resolved_input"):
        st.session_state["active_ticker"] = resolved_ticker
        st.session_state["last_resolved_input"] = resolved_ticker

    target_ticker = st.session_state["active_ticker"]

    st.title("🏛️ Buffett Institutional Value, ROCE & Sector Checklist Analyzer")
    st.markdown(
        "**Dynamic Sector Checklist** • **ROE & ROCE Moat** • **Current vs. Past Growth Momentum** • "
        "**Buyback Forensics** • **3-Scenario 2-Stage DCF**"
    )

    with st.spinner(f"Running statements, sector checklist, and DCF valuation for {target_ticker}..."):
        try:
            bundle = fetch_company_bundle(target_ticker)
            metrics = extract_comprehensive_metrics(bundle)
            checklist_result = evaluate_sector_checklist(bundle, metrics)
            if enable_peers:
                df_peers, peer_medians = fetch_industry_peer_benchmarks(
                    target_ticker=target_ticker,
                    sector=bundle["sector"],
                    currency=bundle["currency"],
                    custom_peers_str=custom_peers_input,
                )
            else:
                df_peers = pd.DataFrame()
                peer_medians = {
                    "rev_growth_med": None, "ebitda_margin_med": None,
                    "net_margin_med": None, "roe_med": None, "roce_med": None, "pe_med": None,
                }
        except ValueError as ve:
            st.error(f"**Ticker Lookup Error:** {ve}")
            st.stop()
        except Exception as exc:
            st.error(f"**API Connectivity Issue:** Unable to fetch data (`{exc}`). Tap 'Clear Cache & Force Fresh Fetch' in the sidebar.")
            st.stop()

    info = bundle["info"]
    currency = bundle["currency"]
    profile = bundle["profile"]
    current_price = bundle["current_price"]
    company_name = info.get("longName") or info.get("shortName") or target_ticker
    mcap = safe_float(info.get("marketCap")) or bundle.get("fast_mcap")

    st.subheader(f"{company_name} (`{target_ticker}`)")
    st.caption(
        f"**Industry Profile:** {profile['profile_name']} | **Sector:** {bundle['sector']} ({bundle['industry']}) | "
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
        roe_med_v = peer_medians.get("roe_med")
        roe_delta = f"vs {roe_med_v:.1f}% Peer Med" if (roe_v is not None and roe_med_v is not None) else "Target >= 15%"
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
            roce_delta = f"vs {roce_med_v:.1f}% Peer Med" if (roce_v is not None and roce_med_v is not None) else "Target >= 15%"
            st.metric(
                label="Return on Capital (ROCE)",
                value=f"{roce_v:.2f}%" if roce_v is not None else "N/A",
                delta=roce_delta,
                delta_color="normal" if (roce_v is not None and roce_v >= (roce_med_v or 15.0)) else "inverse",
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
                value=f"{de_v:.2f}x" if de_v is not None else "0.00x",
                delta="Conservative <= 0.8x" if (de_v is None or de_v <= 0.8) else "Elevated Leverage",
                delta_color="normal" if (de_v is None or de_v <= 0.8) else "inverse",
            )
        else:
            pb_v = metrics["pb_ratio"]
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
    # SECTION 2: DYNAMIC SECTOR-SPECIFIC VALUE INVESTING CHECKLIST
    # --------------------------------------------------------------------------
    st.header("2. Dynamic Sector-Specific Value Investing Checklist")
    st.markdown(
        f"**Detected Sector Framework:** `{checklist_result['sector_title']}`  \n"
        f"**Primary Sector Moat Drivers:** *{checklist_result['moat_focus']}*"
    )

    chk_col1, chk_col2 = st.columns([2, 1])
    with chk_col1:
        st.markdown("#### 🤖 Automated Quantitative Sector Rules")
        df_chk = pd.DataFrame(checklist_result["quant_checks"]).set_index("Parameter")
        st.dataframe(df_chk, use_container_width=True)

    with chk_col2:
        st.markdown("#### ☑️ Qualitative Moat Reconfirmation")
        st.caption(
            "Check off each sector-specific qualitative rule below after verifying it to add up to +2.5 bonus conviction points:"
        )
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
                <strong style="color: #f8fafc;">Automated Sector Pass Rate: <span style="color: {badge_bg};">{pass_c} / {tot_c} Passed</span></strong><br>
                <span style="color: #cbd5e1; font-size: 0.88rem;">Qualitative Investor Checks Confirmed: {user_confirmed_count} / {qual_tot}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 3: CURRENT VS. PAST GROWTH, MARGINS, BUYBACK CHECK & INDUSTRY PEERS
    # --------------------------------------------------------------------------
    st.header("3. Growth Acceleration, EBITDA/Profit Margins, Buyback Forensics & Industry Comparison")

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

    def fmt_pct(val: Optional[float]) -> str:
        return f"{val:+.2f}%" if val is not None else "N/A"

    def fmt_level_pct(val: Optional[float]) -> str:
        return f"{val:.2f}%" if val is not None else "N/A"

    rev_yoy, rev_cagr = metrics["rev_growth"]["recent_yoy_pct"], metrics["rev_growth"]["past_cagr_pct"]
    eb_yoy, eb_cagr = metrics["ebitda_growth"]["recent_yoy_pct"], metrics["ebitda_growth"]["past_cagr_pct"]
    ni_yoy, ni_cagr = metrics["ni_growth"]["recent_yoy_pct"], metrics["ni_growth"]["past_cagr_pct"]
    eps_yoy, eps_cagr = metrics["eps_growth"]["recent_yoy_pct"], metrics["eps_growth"]["past_cagr_pct"]

    em_curr = metrics["ebitda_margin_curr"]
    em_past = metrics["ebitda_margin_past"]
    if profile["ebitda_applicable"] and em_curr is not None and em_past is not None:
        em_diff = em_curr - em_past
        em_momentum = f"🟢 Expanding ({em_diff:+.2f}%)" if em_diff >= 0 else f"🔴 Compressing ({em_diff:+.2f}%)"
    else:
        em_momentum = "N/A"

    nm_curr = metrics["net_margin_curr"]
    nm_past = metrics["net_margin_past"]
    if nm_curr is not None and nm_past is not None:
        nm_diff = nm_curr - nm_past
        nm_momentum = f"🟢 Expanding ({nm_diff:+.2f}%)" if nm_diff >= 0 else f"🔴 Compressing ({nm_diff:+.2f}%)"
    else:
        nm_momentum = "N/A"

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
            "Current / Recent (YoY)": fmt_level_pct(em_curr) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(em_past) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": em_momentum,
            "Industry Peer Median": fmt_level_pct(peer_medians.get("ebitda_margin_med")) if profile["ebitda_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Higher Margin" if (profile["ebitda_applicable"] and em_curr is not None and peer_medians.get("ebitda_margin_med") is not None and em_curr >= peer_medians["ebitda_margin_med"])
                else ("📉 Below Peer Margin" if (profile["ebitda_applicable"] and em_curr is not None and peer_medians.get("ebitda_margin_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "Net Profit Margin (%)",
            "Current / Recent (YoY)": fmt_level_pct(nm_curr),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(nm_past),
            "Momentum (Current vs Past)": nm_momentum,
            "Industry Peer Median": fmt_level_pct(peer_medians.get("net_margin_med")),
            "vs. Industry": (
                "🏆 Higher Margin" if (nm_curr is not None and peer_medians.get("net_margin_med") is not None and nm_curr >= peer_medians["net_margin_med"])
                else ("📉 Below Peer Margin" if (nm_curr is not None and peer_medians.get("net_margin_med") is not None) else "N/A")
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
    # SECTION 4: 2-STAGE DCF & BENJAMIN GRAHAM 3-SCENARIO VALUATION
    # --------------------------------------------------------------------------
    st.header("4. Intrinsic Value & 3-Scenario DCF / Graham Calculator")

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

    fcf_basis_str = format_large_currency(dcf_inputs["fcf_0"], currency)
    eps_used_str = format_price(dcf_inputs["eps_ttm"], currency)
    shares_used_str = format_shares(dcf_inputs["shares"], currency)

    st.caption(
        f"**Cash Flow Basis:** {dcf_inputs['fcf_source']} = **{fcf_basis_str}** | "
        f"**Trailing EPS Used:** {eps_used_str} | "
        f"**Diluted Shares:** {shares_used_str} | "
        f"**Base Growth (g₁):** `{applied_g1:.2f}%` | **WACC (r):** `{applied_r:.2f}%`"
    )

    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.metric("Current Market Price", format_price(current_price, currency))
    with vc2:
        dcf_mos_val = base_res["dcf_mos_pct"]
        st.metric(
            "2-Stage DCF Value (Base)",
            format_price(base_res["dcf_intrinsic"], currency),
            delta=f"{dcf_mos_val:+.1f}% MoS" if dcf_mos_val is not None else "N/A",
            delta_color="normal" if (dcf_mos_val or -1) > 0 else "inverse",
        )
    with vc3:
        graham_mos_val = base_res["graham_mos_pct"]
        st.metric(
            "Graham Value (Base)",
            format_price(base_res["graham_intrinsic"], currency),
            delta=f"{graham_mos_val:+.1f}% MoS" if graham_mos_val is not None else "N/A",
            delta_color="normal" if (graham_mos_val or -1) > 0 else "inverse",
        )
    with vc4:
        blended_mos_val = base_res["blended_mos_pct"]
        st.metric(
            "Blended Intrinsic Value (Base)",
            format_price(base_res["blended_intrinsic"], currency),
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
                "Terminal Growth": f"{s['terminal_g_pct']:.2f}%",
                "2-Stage DCF / Share": format_price(s["dcf_intrinsic"], currency),
                "Graham Value / Share": format_price(s["graham_intrinsic"], currency),
                "Blended Value / Share": format_price(s["blended_intrinsic"], currency),
                "Blended Margin of Safety": f"{s_mos:+.2f}%" if s_mos is not None else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows).set_index("Scenario"), use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 5: CORPORATE NEWS SENTIMENT & FINAL VERDICT
    # --------------------------------------------------------------------------
    st.header("5. Long-Term News Sentiment & Final Investment Verdict")
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
        checklist_result=checklist_result,
        user_confirmed_count=user_confirmed_count,
    )

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("Fundamentals, Sector Checklist & Valuation (70%)", f"{decision['fundamental_score_70']:.1f} / 70.0")
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
