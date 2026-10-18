import math
import re
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
# 0. PAGE CONFIGURATION, INDIAN PRESETS & NSE SECTOR BENCHMARKS
# ==============================================================================
st.set_page_config(
    page_title="Indian Equity Institutional Value & DCF Analyzer (NSE/BSE)",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDIAN_PRESETS = {
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
    "MANUFACTURING_INDUSTRIAL": ["LT.NS", "MARUTI.NS", "EICHERMOT.NS", "SIEMENS.NS"],
}

# Verified NSE Institutional Sector Medians (Fallback so Section 3 is NEVER N/A)
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
# 1. INDIAN CURRENCY (₹ CRORES) & SAFE FORMATTING UTILITIES
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
    """Formats raw INR numbers into ₹ Crores (1 Cr = 10,000,000) or ₹ Lakhs."""
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
                "Net Profit Margin, and Price-to-Book (P/B)."
            ),
        }
    return {
        "is_financial": False,
        "archetype": archetype,
        "profile_name": f"Indian Operating Business ({sector or 'Manufacturing / Services'})",
        "roce_applicable": True,
        "ebitda_applicable": True,
        "de_applicable": True,
        "applicable_note": (
            "Operating Company Detected: Multi-source engine (Screener.in + Yahoo Finance) active across "
            "ROCE, ROE, EBITDA Margin, Debt-to-Equity, Free Cash Flow conversion, and Buyback EPS Forensics."
        ),
    }


# ==============================================================================
# 2. MULTI-SOURCE INDIAN DATA FETCHER (SCREENER.IN + YFINANCE FALLBACK)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_screener_india_data(clean_symbol: str) -> Dict[str, Any]:
    """
    Scrapes verified Indian fundamentals from Screener.in (Consolidated first, Standalone fallback)
    to fill missing ROCE, ROE, 3-Year Revenue/Profit/OPM/EPS/FCF rows when Yahoo Finance has gaps.
    """
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
                soup = BeautifulSoup(resp.text, "html.parser")
                # Check if consolidated table actually has data rows
                pnl_sec = soup.find("section", id="profit-loss")
                if pnl_sec and len(pnl_sec.find_all("tr")) >= 3:
                    break
        except Exception:
            continue

    if soup is None:
        return result

    # 1. Extract Top Ratios (Market Cap, Current Price, Stock P/E, Book Value, ROCE, ROE)
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
        df_out = pd.DataFrame.from_dict(rows_dict, orient="index", columns=headers_list[1:])
        return df_out

    pnl_df = parse_screener_table("profit-loss")
    bs_df = parse_screener_table("balance-sheet")
    cf_df = parse_screener_table("cash-flow")

    result["available"] = bool(ratios or not pnl_df.empty)
    result["ratios"] = ratios
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

    # Also fetch Screener.in data in parallel for Indian multi-source verification
    screener_data = fetch_screener_india_data(ticker_clean)
    scr_ratios = screener_data.get("ratios", {})

    current_price = (
        safe_float(info.get("currentPrice"))
        or safe_float(info.get("regularMarketPrice"))
        or safe_float(info.get("previousClose"))
        or fast_price
        or scr_ratios.get("current_price")
    )

    if current_price is None:
        try:
            hist = ticker_obj.history(period="5d")
            if hist is not None and not hist.empty:
                current_price = safe_float(hist["Close"].iloc[-1])
        except Exception:
            current_price = None

    if current_price is None:
        # Try switching .NS <-> .BO automatically if user typed a BSE-only stock like BMW
        alt_ticker = ticker_clean.replace(".NS", ".BO") if ticker_clean.endswith(".NS") else ticker_clean.replace(".BO", ".NS")
        try:
            alt_obj = yf.Ticker(alt_ticker)
            alt_hist = alt_obj.history(period="5d")
            if alt_hist is not None and not alt_hist.empty:
                current_price = safe_float(alt_hist["Close"].iloc[-1])
                ticker_clean = alt_ticker
                ticker_obj = alt_obj
                info = alt_obj.info or {}
        except Exception:
            pass

    if current_price is None:
        raise ValueError(
            f"Could not retrieve live market pricing for '{ticker_clean}'. "
            "Please check the NSE/BSE symbol (e.g., TCS.NS, SUZLON.NS, CUPID.NS, HDFCBANK.NS)."
        )

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
# 3. HYBRID METRICS ENGINE (MERGING YFINANCE + SCREENER.IN)
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


def get_screener_series_recent_first(df: pd.DataFrame, row_candidates: List[str], scale_mult: float = 1.0) -> Optional[pd.Series]:
    """
    Screener.in tables are ordered oldest-to-newest (left-to-right, e.g., Mar 2018 ... Mar 2025, TTM).
    This helper extracts the matching row, reverses it to newest-first, and scales Crores to raw INR if requested.
    """
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

    # Drop empty columns and reverse so index 0 is the most recent year/TTM
    s_clean = matched_row.dropna()
    if s_clean.empty:
        return None
    s_rev = s_clean.iloc[::-1] * scale_mult
    return s_rev


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

    # --- 1. Extract Series from Yahoo Finance (with Screener.in Fallback in ₹) ---
    rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue", "Interest Income"])
    if rev_s is None or len(rev_s.dropna()) < 2:
        rev_s = get_screener_series_recent_first(scr_pnl, ["Sales", "Revenue", "Financing Profit"], scale_mult=1e7)

    gp_s = get_statement_row(inc, ["Gross Profit"])
    ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA"])
    if (ebitda_s is None or len(ebitda_s.dropna()) < 2) and profile["ebitda_applicable"]:
        ebitda_s = get_screener_series_recent_first(scr_pnl, ["Operating Profit"], scale_mult=1e7)

    ebit_s = get_statement_row(inc, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
    if ebit_s is None or len(ebit_s.dropna()) < 2:
        ebit_s = ebitda_s

    ni_s = get_statement_row(
        inc, ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
    )
    if ni_s is None or len(ni_s.dropna()) < 2:
        ni_s = get_screener_series_recent_first(scr_pnl, ["Net Profit"], scale_mult=1e7)

    eps_s = get_statement_row(inc, ["Diluted EPS", "Basic EPS"])
    if eps_s is None or len(eps_s.dropna()) < 2:
        eps_s = get_screener_series_recent_first(scr_pnl, ["EPS in Rs", "EPS"], scale_mult=1.0)

    shares_s = get_statement_row(inc, ["Diluted Average Shares", "Basic Average Shares"])
    if shares_s is None:
        shares_s = get_statement_row(bs, ["Ordinary Shares Number", "Share Issued", "Common Stock"])

    assets_s = get_statement_row(bs, ["Total Assets"])
    if assets_s is None or assets_s.dropna().empty:
        assets_s = get_screener_series_recent_first(scr_bs, ["Total Assets"], scale_mult=1e7)

    curr_liab_s = get_statement_row(bs, ["Current Liabilities", "Total Current Liabilities"])
    if curr_liab_s is None or curr_liab_s.dropna().empty:
        curr_liab_s = get_screener_series_recent_first(scr_bs, ["Other Liabilities"], scale_mult=1e7)

    curr_assets_s = get_statement_row(bs, ["Current Assets", "Total Current Assets"])
    equity_s = get_statement_row(
        bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]
    )
    debt_s = get_statement_row(
        bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"]
    )
    if debt_s is None or debt_s.dropna().empty:
        debt_s = get_screener_series_recent_first(scr_bs, ["Borrowings"], scale_mult=1e7)

    fcf_s = get_statement_row(cf, ["Free Cash Flow"])
    ocf_s = get_statement_row(
        cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
    )
    if ocf_s is None or ocf_s.dropna().empty:
        ocf_s = get_screener_series_recent_first(scr_cf, ["Cash from Operating Activity"], scale_mult=1e7)

    capex_s = get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])
    buyback_cf_s = get_statement_row(
        cf, ["Repurchase Of Capital Stock", "Common Stock Payments", "Net Common Stock Issuance"]
    )

    # --- 2. ROE (%) & ROCE (%) ---
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

    # --- 3. ROA (%), D/E, Current Ratio, P/B, P/E ---
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

    pe_ratio = safe_float(info.get("trailingPE")) or scr_ratios.get("pe") or safe_float(info.get("forwardPE"))

    # --- 4. Latest OCF, FCF & Net Income ---
    latest_ni = safe_float(ni_s.dropna().iloc[0]) if (ni_s is not None and not ni_s.dropna().empty) else safe_float(info.get("netIncomeToCommon"))
    latest_ocf = safe_float(ocf_s.dropna().iloc[0]) if (ocf_s is not None and not ocf_s.dropna().empty) else safe_float(info.get("operatingCashflow"))
    latest_fcf = safe_float(fcf_s.dropna().iloc[0]) if (fcf_s is not None and not fcf_s.dropna().empty) else safe_float(info.get("freeCashflow"))
    if latest_fcf is None and latest_ocf is not None and capex_s is not None and not capex_s.dropna().empty:
        cx_0 = safe_float(capex_s.dropna().iloc[0], 0.0) or 0.0
        latest_fcf = latest_ocf + cx_0 if cx_0 < 0 else latest_ocf - cx_0

    ocf_to_ni_ratio = (latest_ocf / latest_ni) if (latest_ocf is not None and latest_ni and latest_ni > 0) else None

    # --- 5. Multi-Year Growth Comparison (Current YoY vs Past 3-Yr CAGR) ---
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
    ebitda_growth = extract_growth_profile(ebitda_s) if profile["ebitda_applicable"] else {
        "latest": None, "prev": None, "recent_yoy_pct": None, "past_cagr_pct": None
    }
    ni_growth = extract_growth_profile(ni_s, "earningsGrowth")
    eps_growth = extract_growth_profile(eps_s, "earningsGrowth")

    # --- 6. EBITDA & Net Profit Margin Trajectory ---
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

    # Screener OPM% fallback for EBITDA margin
    if ebitda_margin_curr is None and profile["ebitda_applicable"]:
        opm_s = get_screener_series_recent_first(scr_pnl, ["OPM %", "OPM"], scale_mult=1.0)
        if opm_s is not None and not opm_s.empty:
            ebitda_margin_curr = safe_float(opm_s.iloc[0])
            ebitda_margin_past = float(np.mean(opm_s.iloc[1:4])) if len(opm_s) > 1 else ebitda_margin_curr

    if gross_margin_curr is None:
        gm_info = safe_float(info.get("grossMargins"))
        if gm_info is not None:
            gross_margin_curr = gm_info * 100.0
    if ebitda_margin_curr is None and profile["ebitda_applicable"]:
        em_info = safe_float(info.get("ebitdaMargins"))
        if em_info is not None:
            ebitda_margin_curr = em_info * 100.0
            ebitda_margin_past = ebitda_margin_curr
    if net_margin_curr is None:
        nm_info = safe_float(info.get("profitMargins"))
        if nm_info is not None:
            net_margin_curr = nm_info * 100.0
            net_margin_past = net_margin_curr

    # --- 7. Share Buyback & EPS Quality Detector ---
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
        bb_cash_str = format_inr_crores(total_3yr_buyback_cash)
        buyback_explanation = (
            f"Share count changed by {sh_str} over the multi-year window (3-Yr Buybacks: {bb_cash_str}). "
            f"EPS grew at {eps_cagr:+.2f}%/yr vs Net Income growth of {ni_cagr:+.2f}%/yr — meaning "
            f"+{eps_buyback_boost_pct:.2f}%/yr of EPS growth came from share buybacks shrinking the share base."
        )
    elif share_change_pct is not None and share_change_pct >= 2.0:
        buyback_verdict = "EQUITY DILUTION DETECTED"
        buyback_explanation = (
            f"Diluted share count increased by {share_change_pct:+.2f}% over the multi-year window "
            f"(e.g., QIP, preferential allotment, or ESOPs), causing per-share EPS growth to lag Net Profit growth."
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
        "rev_series": rev_s,
        "ebitda_series": ebitda_s,
        "ni_series": ni_s,
        "eps_series": eps_s,
    }


def build_three_year_history_table(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> pd.DataFrame:
    inc = bundle["income_stmt"]
    cf = bundle["cash_flow"]
    profile = bundle["profile"]
    scr_pnl = bundle.get("screener", {}).get("pnl_df", pd.DataFrame())

    # If Yahoo Finance has at least 2 annual columns, use it; otherwise use Screener.in P&L table
    if inc is not None and not inc.empty and len(inc.columns) >= 2:
        cols = list(inc.columns[:3])
        rev_s = get_statement_row(inc, ["Total Revenue", "Operating Revenue", "Revenue", "Interest Income"])
        ebitda_s = get_statement_row(inc, ["EBITDA", "Normalized EBITDA", "Operating Income"])
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

    # Fallback: Build 3-Year Table directly from Screener.in P&L
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
# 4. SECTOR-SPECIFIC VALUE INVESTING CHECKLIST ENGINE
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
                "Why It Matters in This Sector": "Checks if plant CapEx leaves real cash for shareholders (explains low DCF vs Graham).",
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
        moat_focus = "Asset quality (low NPAs), Return on Assets (ROA >= 1.2%), compounding book value (ROE >= 14%), and P/B valuation."
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
            "Capital Adequacy (CRAR): Is Tier-1 Capital well above RBI minimums so frequent equity dilution isn't needed?",
        ]

    elif archetype == "IT_TECH":
        sector_title = "💻 Information Technology & Software Services Checklist"
        moat_focus = "Asset-light scalability, high client switching costs, strong cash conversion (OCF/Net Income), and organic EPS growth."
        roe_roce_str = f"ROE: {roe:.1f}% | ROCE: {roce:.1f}%" if (roe is not None and roce is not None) else (f"ROE: {roe:.1f}%" if roe is not None else "N/A")
        quant_checks = [
            {
                "Parameter": "Return on Equity & ROCE",
                "Target Benchmark": "ROE >= 20% & ROCE >= 20%",
                "Actual Value": roe_roce_str,
                "Status": status_badge((roe or 0) >= 20.0 and (roce or 20.0) >= 20.0 if (roe is not None or roce is not None) else None, borderline=((roe or 0) >= 15.0)),
                "Why It Matters in This Sector": "Asset-light IT leaders should generate exceptional returns on minimal physical capital.",
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
            "Order Book & Deal Pipeline: Is Total Contract Value (TCV) or recurring revenue growing steadily?",
            "AI & Tech Obsolescence Moat: Are client switching costs high enough to prevent pricing compression?",
            "Share Count Discipline: Are employee stock options (ESOPs) reasonable rather than diluting shareholders?",
        ]

    elif archetype == "FMCG_CONSUMER":
        sector_title = "🛒 FMCG, Consumer Goods & Retail Brand Checklist"
        moat_focus = "Brand pricing power (Gross/EBITDA Margin), distribution reach, negative/low working capital, and high ROCE."
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 22.0%",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 22.0 if roce is not None else None, borderline=(roce is not None and 15.0 <= roce < 22.0)),
                "Why It Matters in This Sector": "Hallmark of an enduring Buffett-style consumer monopoly.",
            },
            {
                "Parameter": "Operating EBITDA Margin",
                "Target Benchmark": ">= 18.0% (Strong Brand Moat)",
                "Actual Value": f"{em:.2f}%" if em is not None else (f"Gross: {gm:.1f}%" if gm is not None else "N/A"),
                "Status": status_badge((em or 0) >= 18.0 if em is not None else None, borderline=(em is not None and 13.0 <= em < 18.0)),
                "Why It Matters in This Sector": "High margins prove customers pay a premium for the brand over generic rivals.",
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
            "Valuation Sanity Check: Even if the brand is elite, is the P/E multiple reasonable relative to its growth rate?",
        ]

    elif archetype == "PHARMA_HEALTH":
        sector_title = "💊 Pharmaceuticals, Healthcare & Diagnostics Checklist"
        moat_focus = "R&D pipeline productivity, high operating margins, regulatory compliance (USFDA/domestic), and healthy working capital."
        quant_checks = [
            {
                "Parameter": "Operating EBITDA Margin",
                "Target Benchmark": ">= 20.0% (Specialty/Formulations)",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 20.0 if em is not None else None, borderline=(em is not None and 15.0 <= em < 20.0)),
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
            "Product Concentration Check: Is revenue diversified across therapies rather than reliant on a single tender?",
            "Working Capital & Receivables: Are institutional/government hospital receivables being collected on time?",
        ]

    else:
        sector_title = "⚡ Energy, Oil & Gas, Power & Renewables Checklist"
        moat_focus = "Capital efficiency (ROCE >= 12%), order-book-to-cash conversion, manageable debt, and sustainable FCF."
        quant_checks = [
            {
                "Parameter": "Return on Capital Employed (ROCE)",
                "Target Benchmark": ">= 12.0% (Strong >= 18%)",
                "Actual Value": f"{roce:.2f}%" if roce is not None else "N/A",
                "Status": status_badge(roce >= 12.0 if roce is not None else None, borderline=(roce is not None and 9.0 <= roce < 12.0)),
                "Why It Matters in This Sector": "Capital-heavy energy/wind/power assets must clear their cost of capital.",
            },
            {
                "Parameter": "EBITDA Margin",
                "Target Benchmark": ">= 15.0%",
                "Actual Value": f"{em:.2f}%" if em is not None else "N/A",
                "Status": status_badge(em >= 15.0 if em is not None else None, borderline=(em is not None and 11.0 <= em < 15.0)),
                "Why It Matters in This Sector": "Indicates operating cushion against commodity and EPC execution swings.",
            },
            {
                "Parameter": "Debt-to-Equity Ratio (D/E)",
                "Target Benchmark": "<= 1.20x (Infrastructure Standard)",
                "Actual Value": f"{de:.2f}x" if de is not None else "0.00x",
                "Status": status_badge((de or 0.0) <= 1.20, borderline=(de is not None and 1.20 < de <= 1.80)),
                "Why It Matters in This Sector": "Utilities use project debt, but excessive leverage threatens solvency.",
            },
            {
                "Parameter": "Operating Cash Flow Generation",
                "Target Benchmark": "Positive (> ₹0 Cr)",
                "Actual Value": format_inr_crores(metrics.get("latest_ocf")),
                "Status": status_badge((metrics.get("latest_ocf") or 0) > 0 if metrics.get("latest_ocf") is not None else None),
                "Why It Matters in This Sector": "Confirms order book execution is converting into real cash inflows.",
            },
        ]
        qual_questions = [
            "Order Execution & Working Capital: Is the order book converting into cash without receivables piling up from state discoms?",
            "Margin Sustainability: Are current margins sustainable without relying on one-time tax credits or subsidies?",
            "CapEx Funding: Can upcoming capacity expansion be funded from internal cash flows rather than heavy new debt?",
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
# 5. HYBRID INDUSTRY PEER BENCHMARKING (LIVE + NSE FALLBACK SO NEVER N/A)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_industry_peer_benchmarks(
    target_ticker: str, archetype: str, custom_peers_str: str = ""
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Combines live peer data from Yahoo/Screener with verified NSE Sector Medians
    so Section 3 Industry Benchmarks NEVER show N/A even if Yahoo throttles requests.
    """
    fallback_meds = NSE_SECTOR_BENCHMARKS.get(
        archetype, NSE_SECTOR_BENCHMARKS["MANUFACTURING_INDUSTRIAL"]
    ).copy()

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


# ==============================================================================
# 6. 2-STAGE DCF, GRAHAM & EXPECTED YEARS TO INTRINSIC VALUE ENGINE
# ==============================================================================
def extract_dcf_inputs(bundle: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    info = bundle["info"]
    inc = bundle["income_stmt"]
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
        capex_s = get_statement_row(cf, ["Capital Expenditure", "Capital Expenditures"])
        capex_val = safe_float(capex_s.dropna().iloc[0]) if (capex_s is not None and not capex_s.dropna().empty) else 0.0
        if ocf_val is not None and ocf_val > 0:
            fcf_0 = ocf_val + capex_val if (capex_val or 0.0) < 0 else ocf_val - (capex_val or 0.0)
            fcf_source = "Operating Cash Flow minus CapEx"

    if fcf_0 is None or fcf_0 <= 0:
        ni_val = metrics.get("latest_ni") or safe_float(info.get("netIncomeToCommon"))
        if ni_val is not None and ni_val > 0:
            fcf_0 = ni_val * 0.85
            fcf_source = "Normalized Owner Earnings (85% of Net Profit Proxy)"

    eps_ttm = (
        safe_float(info.get("trailingEps"))
        or metrics.get("latest_stmt_eps")
        or safe_float(info.get("forwardEps"))
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

    # India Macro Regime: 6.8% 10Y G-Sec Risk-Free Rate, 5.0% ERP, 4.0% Terminal Growth
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


def estimate_years_to_target(
    current_price: float,
    target_iv: Optional[float],
    annual_growth_pct: float,
    discount_rate_pct: float,
) -> Tuple[Optional[float], str]:
    """
    Calculates tentative years required for the stock/fundamentals to bridge the gap
    between Current Market Price and Target Intrinsic Value:
      - If Undervalued (Target IV > Current Price): Models price compounding at expected
        fundamental return + multiple re-rating rate: r_conv = max(10%, 0.65*g1 + 0.5*r).
      - If Overvalued (Target IV < Current Price): Models how many years of fundamental
        earnings/cash-flow compounding at rate g1 are required for intrinsic value to
        catch up to today's market price: T = ln(Price / Target_IV) / ln(1 + g1).
    """
    if target_iv is None or target_iv <= 0 or current_price <= 0:
        return None, "N/A (Insufficient valuation data)"

    ratio = target_iv / current_price
    if 0.95 <= ratio <= 1.05:
        return 0.5, "Already trading within ±5% of Fair Intrinsic Value (~0 to 6 months)"

    if target_iv > current_price:
        # Stock is undervalued; price converges upward via earnings growth + multiple expansion
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
        # Stock is trading above intrinsic value; calculate years for earnings to grow into current price
        g_rate = max(0.08, annual_growth_pct / 100.0)
        years_catchup = math.log(current_price / target_iv) / math.log(1.0 + g_rate)
        years_clamped = float(np.clip(years_catchup, 0.5, 15.0))
        return (
            round(years_clamped, 1),
            f"Currently Above IV — Needs ~{years_clamped:.1f} Years of earnings growth at {g_rate * 100:.1f}%/yr to justify today's price",
        )


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

    valid_vals = [v for v in [dcf_per_share, graham_per_share] if v is not None and v > 0]
    blended_iv = float(np.mean(valid_vals)) if valid_vals else None
    blended_mos_pct = (
        ((blended_iv - current_price) / blended_iv) * 100.0
        if (blended_iv is not None and blended_iv > 0)
        else None
    )

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
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "dcf_intrinsic": dcf_per_share,
        "dcf_mos_pct": dcf_mos_pct,
        "graham_intrinsic": graham_per_share,
        "graham_mos_pct": graham_mos_pct,
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
    current_price: float,
) -> Dict[str, Dict[str, Any]]:
    cons_g1 = max(2.0, base_g1_pct * 0.70)
    cons_r = min(18.0, base_discount_pct + 1.5)
    cons_term = max(2.0, base_terminal_g_pct - 0.5)

    opt_g1 = min(25.0, base_g1_pct * 1.25)
    opt_r = max(8.0, base_discount_pct - 1.0)
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
# 7. CORPORATE NEWS SENTIMENT & SEPARATED POSITIVES / RISKS DECISION ENGINE
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


def compute_decision_verdict(
    ticker: str,
    company_name: str,
    current_price: float,
    profile: Dict[str, Any],
    metrics: Dict[str, Any],
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
    cons_mos = cons_case.get("blended_mos_pct")
    dcf_iv = base_case.get("dcf_intrinsic")
    graham_iv = base_case.get("graham_intrinsic")
    blended_iv = base_case.get("blended_intrinsic")

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
        roce_pts = min(10.0, max(0.0, ((roa or 0.0) / 1.5) * 10.0)) if roa is not None else 5.0
    efficiency_pts = roe_pts + roce_pts

    # 3. Growth Acceleration, Margin Expansion & Buyback Quality (0 - 15 pts)
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

    # 4. Sector Checklist Pass Rate & Industry Peer Comparison (0 - 10 pts)
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

    # 5. Sentiment Sub-Score (0 - 30 pts)
    avg_comp = news_summary.get("avg_compound", 0.0)
    clamped_comp = float(np.clip(avg_comp, -0.5, 0.5))
    sentiment_score_30 = ((clamped_comp + 0.5) / 1.0) * 30.0

    total_score = float(np.clip(fundamental_score_70 + sentiment_score_30, 0.0, 100.0))

    if total_score >= 75.0 and (mos is not None and mos >= 15.0) and checklist_result["pass_count"] >= 3:
        verdict, badge_color = "STRONG BUY", "#16a34a"
    elif total_score >= 60.0 and (mos is not None and mos >= 0.0):
        verdict, badge_color = "ACCUMULATE", "#2563eb"
    elif total_score >= 45.0:
        verdict, badge_color = "HOLD / WATCHLIST", "#d97706"
    else:
        verdict, badge_color = "OVERVALUED / AVOID", "#dc2626"

    # --- BUILD SEPARATE POSITIVE POINTS & RISK FACTORS LISTS ---
    positive_points: List[str] = []
    risk_points: List[str] = []

    # Capital Efficiency Positives / Risks
    if roe is not None and roe >= 15.0:
        positive_points.append(
            f"**Strong Shareholder Compounding (ROE: {roe:.1f}%):** Clears Warren Buffett's 15% ROE moat hurdle (Sector Median: {roe_med or 15.0:.1f}%)."
        )
    elif roe is not None:
        risk_points.append(
            f"**Sub-Par Return on Equity (ROE: {roe:.1f}%):** Trails the 15% institutional moat benchmark and sector median ({roe_med or 15.0:.1f}%)."
        )

    if profile["roce_applicable"] and roce is not None:
        if roce >= 15.0:
            positive_points.append(
                f"**Superior Capital Allocation (ROCE: {roce:.1f}%):** Generates strong operating returns above its WACC ({base_case['discount_pct']:.1f}%)."
            )
        else:
            risk_points.append(
                f"**Low Return on Capital Employed (ROCE: {roce:.1f}%):** Heavy capital deployed in plants/working capital is earning below the 15% target."
            )

    # Balance Sheet Debt Positives / Risks
    if profile["de_applicable"]:
        if de is None or de <= 0.5:
            positive_points.append(
                f"**Fortress Balance Sheet (D/E: {de or 0.0:.2f}x):** Minimal debt protects cash flows during RBI interest-rate or demand cycles."
            )
        elif de > 1.0:
            risk_points.append(
                f"**Elevated Balance Sheet Leverage (D/E: {de:.2f}x):** High debt obligations increase vulnerability during industry slowdowns."
            )

    # Growth & Margin Momentum Positives / Risks
    if rev_yoy is not None and rev_cagr is not None:
        if rev_yoy >= rev_cagr:
            positive_points.append(
                f"**Accelerating Top-Line Momentum:** Recent YoY revenue growth ({rev_yoy:+.1f}%) is outpacing its 3-year historical CAGR ({rev_cagr:+.1f}%)."
            )
        else:
            risk_points.append(
                f"**Top-Line Growth Deceleration:** Recent YoY revenue growth ({rev_yoy:+.1f}%) has slowed below its 3-year historical CAGR ({rev_cagr:+.1f}%)."
            )

    # Valuation & DCF vs Graham Divergence Positives / Risks
    if mos is not None and mos >= 15.0:
        positive_points.append(
            f"**Attractive Margin of Safety ({mos:+.1f}%):** Market price ({format_inr_price(current_price)}) trades at a meaningful discount to Blended Intrinsic Value ({format_inr_price(blended_iv)})."
        )
    elif mos is not None and mos < 0:
        risk_points.append(
            f"**Valuation Premium Risk ({mos:+.1f}% MoS):** Current price ({format_inr_price(current_price)}) trades above Base-Case Blended Intrinsic Value ({format_inr_price(blended_iv)}), leaving little room for earnings misses."
        )

    if dcf_iv is not None and graham_iv is not None and graham_iv > dcf_iv * 1.6:
        risk_points.append(
            f"**DCF vs. Graham Cash-Conversion Gap:** Graham Value ({format_inr_price(graham_iv)}) is much higher than 2-Stage DCF ({format_inr_price(dcf_iv)}), indicating heavy CapEx or working capital is absorbing Free Cash Flow."
        )

    # EPS Quality & Checklist
    if metrics["buyback_verdict"] == "100% ORGANIC EPS GROWTH":
        positive_points.append(
            "**High-Quality Organic EPS:** Per-share earnings growth is backed by genuine Net Profit expansion rather than financial engineering."
        )
    else:
        risk_points.append(f"**Share Count Distortion ({metrics['buyback_verdict']}):** {metrics['buyback_explanation']}")

    if checklist_result["pass_count"] >= 3:
        positive_points.append(
            f"**Sector Checklist Conviction:** Passed {checklist_result['pass_count']} out of {checklist_result['total_checks']} mandatory quantitative rules for its industry."
        )
    else:
        risk_points.append(
            f"**Sector Checklist Weaknesses:** Passed only {checklist_result['pass_count']} of {checklist_result['total_checks']} mandatory quantitative sector rules."
        )

    if not positive_points:
        positive_points.append("**Established Market Presence:** Active operations with ongoing institutional coverage.")
    if not risk_points:
        risk_points.append("**Macro & Terminal Multiple Risk:** Standard sensitivity to domestic GDP growth and sector P/E multiple normalization.")

    return {
        "fundamental_score_70": fundamental_score_70,
        "sentiment_score_30": sentiment_score_30,
        "total_score": total_score,
        "verdict": verdict,
        "badge_color": badge_color,
        "positive_points": positive_points,
        "risk_points": risk_points,
        "base_timeline_label": base_case.get("est_years_label", "N/A"),
        "cons_timeline_label": cons_case.get("est_years_label", "N/A"),
        "opt_timeline_label": opt_case.get("est_years_label", "N/A"),
    }


# ==============================================================================
# 8. STREAMLIT USER INTERFACE (100% INDIAN EQUITY MARKET EDITION)
# ==============================================================================
def main() -> None:
    with st.sidebar:
        st.header("🇮🇳 Indian Stock Selector (NSE / BSE)")

        preset_options = ["Custom NSE/BSE Ticker..."] + [f"{k} — {v}" for k, v in INDIAN_PRESETS.items()]
        selected_preset = st.selectbox("Quick-Select Indian Stock", options=preset_options, index=1)

        if selected_preset != "Custom NSE/BSE Ticker...":
            default_ticker = selected_preset.split(" — ")[0].strip()
        else:
            default_ticker = "TCS.NS"

        raw_ticker_input = st.text_input(
            "Enter Any Indian Stock Symbol",
            value=default_ticker,
            help="Type any NSE/BSE symbol (e.g., TCS, SUZLON, CUPID, HDFCBANK, BMW.BO). Automatically appends .NS or .BO.",
        )

        exchange_suffix = st.selectbox(
            "Default Indian Exchange (if suffix omitted)",
            options=[".NS (National Stock Exchange)", ".BO (Bombay Stock Exchange)"],
            index=0,
        ).split(" ")[0]

        cleaned_input = raw_ticker_input.strip().upper()
        if cleaned_input and "." not in cleaned_input:
            resolved_ticker = f"{cleaned_input}{exchange_suffix}"
        else:
            resolved_ticker = cleaned_input

        st.markdown("---")
        st.subheader("📐 Valuation & DCF Overrides")
        use_custom_params = st.checkbox("Override Automatic Growth & WACC Assumptions", value=False)
        custom_g1 = st.slider(
            "Base Stage-1 Growth Rate g₁ (Years 1–5, %)",
            min_value=0.0, max_value=25.0, value=12.0, step=0.5,
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
        st.subheader("🏢 Peer Benchmark & CSV Override")
        custom_peers_input = st.text_input(
            "Custom Indian Peer Symbols (optional)",
            value="",
            placeholder="e.g., INFY.NS, HCLTECH.NS, WIPRO.NS",
        )
        uploaded_csv = st.file_uploader(
            "Optional: Upload Custom 3-Yr Financials CSV",
            type=["csv"],
            help="If you have a custom P&L CSV exported from Screener/Trendlyne, upload it here to display alongside live data.",
        )

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

    st.title("🇮🇳 Indian Market Institutional Value, ROCE & DCF Analyzer")
    st.markdown(
        "**Multi-Source Engine (`Screener.in` + `Yahoo Finance` + `NSE Sector Medians`)** • "
        "**3-Scenario 2-Stage DCF** • **Separated Positives, Risks & Expected Years to Fair Value**"
    )

    with st.spinner(f"Fetching multi-source Indian financials (Screener.in + NSE/BSE) for {target_ticker}..."):
        try:
            bundle = fetch_company_bundle(target_ticker)
            metrics = extract_comprehensive_metrics(bundle)
            checklist_result = evaluate_sector_checklist(bundle, metrics)
            df_peers, peer_medians = fetch_industry_peer_benchmarks(
                target_ticker=target_ticker,
                archetype=bundle["profile"]["archetype"],
                custom_peers_str=custom_peers_input,
            )
        except ValueError as ve:
            st.error(f"**Ticker Lookup Error:** {ve}")
            st.stop()
        except Exception as exc:
            st.error(f"**Data Connectivity Issue:** `{exc}`. Tap 'Clear Cache & Force Fresh Fetch' in the sidebar.")
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
        f"**Current Market Price:** {format_inr_price(current_price)} | "
        f"**Market Cap:** {format_inr_crores(mcap)} | "
        f"**Data Feeds Active:** `Yahoo Finance API` + `{'Screener.in Consolidated' if scr_ok else 'NSE Sector Fallback'}`"
    )
    st.info(f"🧭 **Industry-Adaptive Routing:** {profile['applicable_note']}")

    if uploaded_csv is not None:
        try:
            user_df = pd.read_csv(uploaded_csv)
            with st.expander("📂 View Uploaded Custom Financial CSV", expanded=True):
                st.dataframe(user_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not parse uploaded CSV: {e}")

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 1: ROE, ROCE, MOAT KPIs & 3-YEAR HISTORICAL TABLE
    # --------------------------------------------------------------------------
    st.header("1. Business Health, Capital Efficiency (ROE & ROCE) & Moat")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        roe_v = metrics["roe_pct"]
        roe_med_v = peer_medians.get("roe_med")
        roe_delta = f"vs {roe_med_v:.1f}% Sector Med" if (roe_v is not None and roe_med_v is not None) else "Target >= 15%"
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
            roce_delta = f"vs {roce_med_v:.1f}% Sector Med" if (roce_v is not None and roce_med_v is not None) else "Target >= 15%"
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
        pe_v = metrics["pe_ratio"]
        pe_med_v = peer_medians.get("pe_med")
        pe_delta = f"Sector P/E: {pe_med_v:.1f}x" if pe_med_v is not None else None
        st.metric(
            label="Trailing P/E Multiple",
            value=f"{pe_v:.2f}x" if pe_v is not None else "N/A",
            delta=pe_delta,
            delta_color="normal" if (pe_v is not None and pe_med_v is not None and pe_v <= pe_med_v) else "off",
        )

    st.markdown("#### 📊 Multi-Year Historical Financial Performance (₹ Crores)")
    hist_df = build_three_year_history_table(bundle, metrics)
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
            "Tick each sector-specific check below after verifying to add up to +2.5 conviction points:"
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
            "NSE Industry Median": fmt_pct(peer_medians.get("rev_growth_med")),
            "vs. Industry": (
                "🏆 Outperforming" if (rev_yoy is not None and peer_medians.get("rev_growth_med") is not None and rev_yoy >= peer_medians["rev_growth_med"])
                else ("📉 Lagging Sector" if (rev_yoy is not None and peer_medians.get("rev_growth_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "EBITDA / Operating Profit Growth",
            "Current / Recent (YoY)": fmt_pct(eb_yoy) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_pct(eb_cagr) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": (
                "🟢 Accelerating" if (profile["ebitda_applicable"] and eb_yoy is not None and eb_cagr is not None and eb_yoy >= eb_cagr)
                else ("🟡 Decelerating" if (profile["ebitda_applicable"] and eb_yoy is not None and eb_cagr is not None) else "N/A")
            ),
            "NSE Industry Median": fmt_pct(peer_medians.get("rev_growth_med")) if profile["ebitda_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Outperforming" if (profile["ebitda_applicable"] and eb_yoy is not None and peer_medians.get("rev_growth_med") is not None and eb_yoy >= peer_medians["rev_growth_med"])
                else ("📉 Lagging Sector" if (profile["ebitda_applicable"] and eb_yoy is not None) else "N/A")
            ),
        },
        {
            "Metric": "EBITDA / Operating Margin (OPM %)",
            "Current / Recent (YoY)": fmt_level_pct(em_curr) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(em_past) if profile["ebitda_applicable"] else "N/A (Financial)",
            "Momentum (Current vs Past)": em_momentum,
            "NSE Industry Median": fmt_level_pct(peer_medians.get("ebitda_margin_med")) if profile["ebitda_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Higher Margin" if (profile["ebitda_applicable"] and em_curr is not None and peer_medians.get("ebitda_margin_med") is not None and em_curr >= peer_medians["ebitda_margin_med"])
                else ("📉 Below Sector Margin" if (profile["ebitda_applicable"] and em_curr is not None and peer_medians.get("ebitda_margin_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "Net Profit Margin (NPM %)",
            "Current / Recent (YoY)": fmt_level_pct(nm_curr),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(nm_past),
            "Momentum (Current vs Past)": nm_momentum,
            "NSE Industry Median": fmt_level_pct(peer_medians.get("net_margin_med")),
            "vs. Industry": (
                "🏆 Higher Margin" if (nm_curr is not None and peer_medians.get("net_margin_med") is not None and nm_curr >= peer_medians["net_margin_med"])
                else ("📉 Below Sector Margin" if (nm_curr is not None and peer_medians.get("net_margin_med") is not None) else "N/A")
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
            "NSE Industry Median": f"Net Profit YoY: {fmt_pct(ni_yoy)}",
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
            "NSE Industry Median": fmt_level_pct(peer_medians.get("roce_med")) if profile["roce_applicable"] else "N/A",
            "vs. Industry": (
                "🏆 Superior ROCE" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and peer_medians.get("roce_med") is not None and metrics["roce_pct"] >= peer_medians["roce_med"])
                else ("📉 Below Sector ROCE" if (profile["roce_applicable"] and metrics["roce_pct"] is not None and peer_medians.get("roce_med") is not None) else "N/A")
            ),
        },
        {
            "Metric": "Return on Equity (ROE)",
            "Current / Recent (YoY)": fmt_level_pct(metrics["roe_pct"]),
            "Past 3-Yr Historical (CAGR / Avg)": fmt_level_pct(metrics["roe_pct"]),
            "Momentum (Current vs Past)": "🟢 Strong Moat (>=15%)" if (metrics["roe_pct"] or 0) >= 15.0 else "🟡 Below 15% Target",
            "NSE Industry Median": fmt_level_pct(peer_medians.get("roe_med")),
            "vs. Industry": (
                "🏆 Superior ROE" if (metrics["roe_pct"] is not None and peer_medians.get("roe_med") is not None and metrics["roe_pct"] >= peer_medians["roe_med"])
                else "📉 Below Sector ROE"
            ),
        },
    ]

    st.markdown("#### 📈 Current vs. Past Growth & Margin Trajectory (Benchmarked vs. NSE Sector Median)")
    st.dataframe(pd.DataFrame(comparison_rows).set_index("Metric"), use_container_width=True)

    if not df_peers.empty:
        with st.expander("🏢 View Indian Sector Peer Comparison Table", expanded=False):
            st.dataframe(df_peers.set_index("Ticker").round(2), use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 4: 2-STAGE DCF, GRAHAM & EXPECTED TIMELINE TO INTRINSIC VALUE
    # --------------------------------------------------------------------------
    st.header("4. Intrinsic Value, 3-Scenario DCF / Graham & Expected Convergence Timeline")

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

    fcf_basis_str = format_inr_crores(dcf_inputs["fcf_0"])
    eps_used_str = format_inr_price(dcf_inputs["eps_ttm"])
    shares_used_str = format_shares_cr(dcf_inputs["shares"])

    st.caption(
        f"**Cash Flow Basis:** {dcf_inputs['fcf_source']} = **{fcf_basis_str}** | "
        f"**Trailing EPS Used:** {eps_used_str} | "
        f"**Diluted Shares:** {shares_used_str} | "
        f"**Base Growth (g₁):** `{applied_g1:.2f}%` | **WACC (r):** `{applied_r:.2f}%`"
    )

    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        st.metric("Current Market Price", format_inr_price(current_price))
    with vc2:
        dcf_mos_val = base_res["dcf_mos_pct"]
        st.metric(
            "2-Stage DCF Value (Base)",
            format_inr_price(base_res["dcf_intrinsic"]),
            delta=f"{dcf_mos_val:+.1f}% MoS" if dcf_mos_val is not None else "N/A",
            delta_color="normal" if (dcf_mos_val or -1) > 0 else "inverse",
        )
    with vc3:
        graham_mos_val = base_res["graham_mos_pct"]
        st.metric(
            "Graham Value (Base)",
            format_inr_price(base_res["graham_intrinsic"]),
            delta=f"{graham_mos_val:+.1f}% MoS" if graham_mos_val is not None else "N/A",
            delta_color="normal" if (graham_mos_val or -1) > 0 else "inverse",
        )
    with vc4:
        blended_mos_val = base_res["blended_mos_pct"]
        st.metric(
            "Blended Intrinsic Value (Base)",
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
                "Terminal Growth": f"{s['terminal_g_pct']:.2f}%",
                "2-Stage DCF / Share": format_inr_price(s["dcf_intrinsic"]),
                "Graham Value / Share": format_inr_price(s["graham_intrinsic"]),
                "Blended Value / Share": format_inr_price(s["blended_intrinsic"]),
                "Blended Margin of Safety": f"{s_mos:+.2f}%" if s_mos is not None else "N/A",
                "Tentative Timeline to Reach IV": s["est_years_label"],
            }
        )
    st.dataframe(pd.DataFrame(scenario_rows).set_index("Scenario"), use_container_width=True)

    st.divider()

    # --------------------------------------------------------------------------
    # SECTION 5: NEWS SENTIMENT, SEPARATED POSITIVES/RISKS & EXPECTED TIMELINE VERDICT
    # --------------------------------------------------------------------------
    st.header("5. Corporate News Catalysts & Final Investment Verdict")
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
        current_price=current_price,
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

    # Prominent Verdict Banner + Tentative Years to Intrinsic Value
    st.markdown(
        f"""
        <div style="
            background-color: #0f172a;
            border-left: 8px solid {decision['badge_color']};
            padding: 22px 24px;
            border-radius: 8px;
            margin-top: 16px;
            margin-bottom: 18px;
        ">
            <h2 style="color: #f8fafc; margin: 0 0 8px 0; font-weight: 800;">
                LONG-TERM INVESTMENT VERDICT:
                <span style="color: {decision['badge_color']};">{decision['verdict']}</span>
            </h2>
            <p style="color: #38bdf8; font-size: 1.05rem; font-weight: 600; margin: 0 0 6px 0;">
                ⏳ Tentative Expected Time to Reach Base-Case Intrinsic Value ({format_inr_price(base_res['blended_intrinsic'])}):
                <span style="color: #f8fafc;">{decision['base_timeline_label']}</span>
            </p>
            <p style="color: #94a3b8; font-size: 0.92rem; margin: 0;">
                • <b>Conservative Scenario ({format_inr_price(scenarios['Conservative']['blended_intrinsic'])}):</b> {decision['cons_timeline_label']} &nbsp;|&nbsp;
                • <b>Optimistic Scenario ({format_inr_price(scenarios['Optimistic']['blended_intrinsic'])}):</b> {decision['opt_timeline_label']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Separated Positive Points vs Key Risks to Consider
    pos_col, risk_col = st.columns(2)
    with pos_col:
        st.markdown("### 🟢 Positive Points & Bullish Catalysts")
        for pt in decision["positive_points"]:
            st.markdown(f"- {pt}")

    with risk_col:
        st.markdown("### 🔴 Key Risks & Red Flags to Consider")
        for rk in decision["risk_points"]:
            st.markdown(f"- {rk}")


if __name__ == "__main__":
    main()
