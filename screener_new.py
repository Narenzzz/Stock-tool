"""
News + Momentum Swing Screener (companion to the Value Analyser app)
====================================================================
Funnel:  headlines (RSS)  ->  match to NSE stocks  ->  catalyst scoring  ->  price confirmation
         ->  ranked shortlist + sector/theme heat  ->  open the stock in your Value Analyser.

This produces *setup candidates*, NOT price forecasts or probabilities. News is often priced in by the
time it is published; the price/volume confirmation and the track-record tab exist to keep you honest.
Educational tool - not investment advice.
"""
import io
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SwingNewsScreener/1.0)"}
GNEWS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
BENCH = "^NSEI"

# ------------------------------------------------------------------------------
# 1. CONFIG: FEEDS, EVENTS, THEMES
# ------------------------------------------------------------------------------
DIRECT_FEEDS = {
    "Economic Times - Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Economic Times - Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "Moneycontrol - Business": "https://www.moneycontrol.com/rss/business.xml",
    "Moneycontrol - Market Reports": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Livemint - Markets": "https://www.livemint.com/rss/markets",
    "Business Standard - Markets": "https://www.business-standard.com/rss/markets-106.rss",
    "Hindu BusinessLine - Markets": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
}
# NDTV Profit / CNBC-TV18 are pulled through Google News RSS (site: filter) because it is the most stable public route.
SITE_FEEDS = {"NDTV Profit": "ndtvprofit.com", "CNBC-TV18": "cnbctv18.com"}
SITE_TOPICS = ["", "order win", "results profit", "target price", "promoter stake", "capex expansion"]
GENERIC_QUERIES = {
    "GNews - order wins": "NSE listed company bags order",
    "GNews - brokerage calls": "India brokerage upgrade stock target price",
    "GNews - sector policy": "India sector PLI scheme cabinet approves",
}

# (label, weight, phrases).  Weight = bullish(+)/bearish(-) strength of the event for the named company.
EVENTS: List[Tuple[str, float, List[str]]] = [
    ("Order win", 3.0, ["bags order", "wins order", "secures order", "order win", "order inflow", "bags contract", "wins contract",
                        "awarded contract", "letter of award", "lowest bidder", "bags orders", "deal win", "wins deal", "bags deal", "large deal"]),
    ("Earnings beat", 2.5, ["profit jumps", "profit surges", "profit rises", "profit up", "net profit rises", "beats estimates",
                            "record profit", "record revenue", "margin expansion", "ebitda rises", "profit soars"]),
    ("Earnings miss", -2.5, ["profit falls", "profit drops", "profit declines", "profit slumps", "misses estimates", "net loss",
                             "loss widens", "margin contraction", "revenue declines", "profit plunges"]),
    ("Brokerage upgrade", 2.0, ["upgrade", "initiates coverage with buy", "target price raised", "raises target", "buy rating",
                                "outperform", "overweight", "top pick"]),
    ("Brokerage downgrade", -2.0, ["downgrade", "cuts target", "target price cut", "sell rating", "underperform", "underweight"]),
    ("Capex / expansion", 2.0, ["capacity expansion", "capex plan", "new plant", "commissioning", "commissions", "expansion plan",
                                "to invest rs", "greenfield", "capex of rs"]),
    ("Policy tailwind", 2.0, ["pli scheme", "government approves", "cabinet approves", "subsidy", "gst cut", "rate cut",
                              "duty cut", "safeguard duty", "anti-dumping", "antidumping"]),
    ("Policy headwind", -2.0, ["ban on", "windfall tax", "price cap", "duty hike", "export curbs", "export ban", "tariff hike on"]),
    ("Insider / institutional buying", 2.0, ["promoter buys", "raises stake", "acquires stake", "bulk deal", "block deal",
                                             "fii buying", "stake increase"]),
    ("Insider / institutional selling", -2.0, ["promoter sells", "stake sale", "pledge", "offloads stake", "fii selling"]),
    ("Capital return", 1.5, ["buyback", "dividend", "bonus issue", "stock split"]),
    ("Regulatory / legal risk", -3.0, ["sebi order", "sebi probe", "fraud", "raid", "ed probe", "penalty", "show cause", "nclt",
                                       "insolvency", "default", "auditor resigns", "licence cancelled", "recall", "resigns"]),
    ("Approval / product", 2.0, ["usfda nod", "usfda approval", "gets nod", "gets approval", "clinical trial success"]),
    ("Deal / M&A", 1.5, ["acquisition", "acquires", "merger", "joint venture", "strategic partnership", "tie-up"]),
    ("Fund raise (dilution)", -1.0, ["qip", "rights issue", "preferential allotment"]),
    ("Demand strength", 1.5, ["price hike", "demand surge", "strong demand", "volume growth", "sales rise", "sales jump"]),
    ("Demand weakness", -1.5, ["demand slowdown", "sales fall", "volume decline", "price war"]),
]
_EVENT_RX = [(lab, w, re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ph) + ")", re.I)) for lab, w, ph in EVENTS]
UNCERTAIN = re.compile(r"\b(may|might|likely|could|rumou?r|reportedly|sources say|eyes|plans? to)\b", re.I)
LISTICLE = re.compile(r"(stocks to (watch|buy)|buzzing stocks|stocks in news|top gainers|top losers|market wrap|closing bell|"
                      r"opening bell|stock market today|stocks to focus)", re.I)
LONG_HORIZON = {"Capex / expansion", "Policy tailwind"}
SHORT_HORIZON = {"Order win", "Earnings beat", "Insider / institutional buying", "Brokerage upgrade", "Capital return", "Approval / product"}

THEMES: Dict[str, List[str]] = {
    "Defence": ["defence", "defense", "missile", "ammunition", "drdo", "warship"],
    "Railways": ["railway", "vande bharat", "wagon", "rvnl", "locomotive"],
    "Power, Renewables & Grid": ["solar", "renewable", "wind energy", "transmission", "green hydrogen", "battery storage", "power demand", "discom"],
    "Banks & NBFCs": ["repo rate", "rbi", "credit growth", "npa", "nbfc", "net interest margin"],
    "IT Services": ["it services", "deal win", "generative ai", "tcv", "deal pipeline"],
    "Pharma & Healthcare": ["usfda", "pharma", "drug approval", "hospital"],
    "Autos & EV": ["electric vehicle", " ev ", "auto sales", "two-wheeler", "passenger vehicle"],
    "Metals & Mining": ["steel", "aluminium", "copper", "iron ore", "coal", "anti-dumping"],
    "Cement & Infra": ["cement", "infra spending", "road construction", "nhai", "capex"],
    "Real Estate": ["housing sales", "real estate", "realty", "residential launches"],
    "FMCG & Consumption": ["fmcg", "rural demand", "festive demand", "consumption"],
    "Electronics & Capital Goods": ["pli", "electronics manufacturing", "semiconductor", "data centre", "capital goods"],
    "Oil & Gas": ["crude", "refining", "lng", "gas price", "omc"],
    "Chemicals & Agri": ["agrochemical", "fertiliser", "specialty chemical", "monsoon", "sugar", "ethanol"],
    "Aviation & Travel": ["airline", "aviation", "hotel occupancy", "air traffic"],
    "Telecom": ["tariff hike", "spectrum", "arpu", "telecom"],
}

THEME_RX = {t: [re.compile(r"\b" + re.escape(k.strip()) + r"\b", re.I) for k in kws] for t, kws in THEMES.items()}

# Offline fallback universe (used only if the NSE Nifty-500 file cannot be downloaded).
FALLBACK_UNIVERSE = [
    ("RELIANCE", "Reliance Industries", "Oil Gas & Consumable Fuels"), ("TCS", "Tata Consultancy Services", "Information Technology"),
    ("INFY", "Infosys", "Information Technology"), ("HDFCBANK", "HDFC Bank", "Financial Services"),
    ("ICICIBANK", "ICICI Bank", "Financial Services"), ("SBIN", "State Bank of India", "Financial Services"),
    ("BHARTIARTL", "Bharti Airtel", "Telecommunication"), ("ITC", "ITC", "Fast Moving Consumer Goods"),
    ("LT", "Larsen & Toubro", "Construction"), ("HINDUNILVR", "Hindustan Unilever", "Fast Moving Consumer Goods"),
    ("AXISBANK", "Axis Bank", "Financial Services"), ("KOTAKBANK", "Kotak Mahindra Bank", "Financial Services"),
    ("BAJFINANCE", "Bajaj Finance", "Financial Services"), ("MARUTI", "Maruti Suzuki", "Automobile and Auto Components"),
    ("TATAMOTORS", "Tata Motors", "Automobile and Auto Components"), ("M&M", "Mahindra & Mahindra", "Automobile and Auto Components"),
    ("SUNPHARMA", "Sun Pharmaceutical", "Healthcare"), ("DRREDDY", "Dr. Reddy's Laboratories", "Healthcare"),
    ("CIPLA", "Cipla", "Healthcare"), ("TITAN", "Titan Company", "Consumer Durables"), ("ADANIENT", "Adani Enterprises", "Metals & Mining"),
    ("ADANIPORTS", "Adani Ports", "Services"), ("ADANIGREEN", "Adani Green Energy", "Power"),
    ("ADANIENSOL", "Adani Energy Solutions", "Power"), ("ADANIPOWER", "Adani Power", "Power"), ("NTPC", "NTPC", "Power"),
    ("POWERGRID", "Power Grid Corporation", "Power"), ("TATAPOWER", "Tata Power", "Power"), ("ONGC", "ONGC", "Oil Gas & Consumable Fuels"),
    ("COALINDIA", "Coal India", "Oil Gas & Consumable Fuels"), ("TATASTEEL", "Tata Steel", "Metals & Mining"),
    ("JSWSTEEL", "JSW Steel", "Metals & Mining"), ("HINDALCO", "Hindalco Industries", "Metals & Mining"),
    ("VEDL", "Vedanta", "Metals & Mining"), ("ULTRACEMCO", "UltraTech Cement", "Construction Materials"),
    ("GRASIM", "Grasim Industries", "Construction Materials"), ("BEL", "Bharat Electronics", "Capital Goods"),
    ("HAL", "Hindustan Aeronautics", "Capital Goods"), ("BHEL", "Bharat Heavy Electricals", "Capital Goods"),
    ("SIEMENS", "Siemens", "Capital Goods"), ("ABB", "ABB India", "Capital Goods"), ("CGPOWER", "CG Power and Industrial Solutions", "Capital Goods"),
    ("SUZLON", "Suzlon Energy", "Capital Goods"), ("RVNL", "Rail Vikas Nigam", "Construction"), ("IRFC", "Indian Railway Finance Corporation", "Financial Services"),
    ("JIOFIN", "Jio Financial Services", "Financial Services"), ("ZOMATO", "Zomato", "Consumer Services"), ("TRENT", "Trent", "Consumer Services"),
    ("DMART", "Avenue Supermarts", "Consumer Services"), ("HAVELLS", "Havells India", "Consumer Durables"), ("POLYCAB", "Polycab India", "Capital Goods"),
    ("DIXON", "Dixon Technologies", "Consumer Durables"), ("BPCL", "Bharat Petroleum", "Oil Gas & Consumable Fuels"),
    ("IOC", "Indian Oil Corporation", "Oil Gas & Consumable Fuels"), ("GAIL", "GAIL", "Oil Gas & Consumable Fuels"),
    ("HEROMOTOCO", "Hero MotoCorp", "Automobile and Auto Components"), ("BAJAJ-AUTO", "Bajaj Auto", "Automobile and Auto Components"),
    ("EICHERMOT", "Eicher Motors", "Automobile and Auto Components"), ("TVSMOTOR", "TVS Motor", "Automobile and Auto Components"),
    ("NESTLEIND", "Nestle India", "Fast Moving Consumer Goods"), ("BRITANNIA", "Britannia Industries", "Fast Moving Consumer Goods"),
    ("VBL", "Varun Beverages", "Fast Moving Consumer Goods"), ("APOLLOHOSP", "Apollo Hospitals", "Healthcare"),
    ("LUPIN", "Lupin", "Healthcare"), ("DIVISLAB", "Divi's Laboratories", "Healthcare"), ("WIPRO", "Wipro", "Information Technology"),
    ("HCLTECH", "HCL Technologies", "Information Technology"), ("TECHM", "Tech Mahindra", "Information Technology"),
    ("LTIM", "LTIMindtree", "Information Technology"),
]
ALIASES = {
    "RELIANCE": ["RIL"], "SBIN": ["SBI"], "TCS": ["TCS"], "LT": ["L&T"], "M&M": ["M&M"], "HAL": ["HAL"], "BEL": ["BEL"],
    "HINDUNILVR": ["HUL"], "ITC": ["ITC"], "ONGC": ["ONGC"], "NTPC": ["NTPC"], "IOC": ["IOC", "IndianOil"], "BPCL": ["BPCL"],
    "ZOMATO": ["Eternal"], "HDFCBANK": ["HDFC Bank"], "ICICIBANK": ["ICICI Bank"], "CGPOWER": ["CG Power"],
    "ADANIENSOL": ["Adani Energy Solutions", "AESL"], "VEDL": ["Vedanta"], "JIOFIN": ["Jio Financial"],
}
STOPWORDS = {"india", "indian", "bharat", "global", "power", "steel", "bank", "royal", "prime", "star", "gold", "vision", "action",
             "empire", "united", "universal", "sun", "delta", "north", "east", "west", "national", "general", "standard", "capital", "reliance"}
SUFFIX_RX = re.compile(r"\b(limited|ltd\.?|corporation|corp\.?|company|co\.?)\b\.?", re.I)


# ------------------------------------------------------------------------------
# 2. UNIVERSE + HEADLINE-TO-STOCK MATCHER
# ------------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def load_universe() -> Tuple[pd.DataFrame, str]:
    urls = ["https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
            "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"]
    for u in urls:
        try:
            r = requests.get(u, headers=HEADERS, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            df = df.rename(columns={"Company Name": "name", "Industry": "industry", "Symbol": "symbol"})
            if {"name", "industry", "symbol"} <= set(df.columns) and len(df) > 100:
                return df[["symbol", "name", "industry"]].dropna().reset_index(drop=True), "NSE Nifty 500 list"
        except Exception:
            continue
    return pd.DataFrame(FALLBACK_UNIVERSE, columns=["symbol", "name", "industry"]), "built-in fallback list (NSE file unreachable)"


class Matcher:
    """Finds NSE companies mentioned in a headline. Short tickers (<=4 chars) are matched case-sensitively."""

    def __init__(self, universe: pd.DataFrame):
        long_map: Dict[str, str] = {}
        short_map: Dict[str, str] = {}
        for sym, name, _ in universe.itertuples(index=False):
            base = re.sub(r"\s+", " ", SUFFIX_RX.sub("", str(name))).strip(" .,&-")
            cands = {base, str(name).strip()} | set(ALIASES.get(sym, []))
            if base.lower().endswith(" india") and not base.lower().endswith(" of india"):
                cands.add(base[:-6].strip())          # "ABB India" -> also "ABB"
            for c in cands:
                c = c.strip()
                if not c:
                    continue
                if len(c) <= 4:
                    short_map.setdefault(c, sym)
                elif len(c) >= 5 and c.lower() not in STOPWORDS:
                    long_map.setdefault(c.lower(), sym)
        self.long_map, self.short_map = long_map, short_map
        guard_l, guard_r = r"(?<![A-Za-z0-9&])", r"(?![A-Za-z0-9&])"
        self.long_rx = (re.compile(guard_l + "(" + "|".join(re.escape(k) for k in sorted(long_map, key=len, reverse=True)) + ")" + guard_r, re.I)
                        if long_map else None)
        self.short_rx = (re.compile(guard_l + "(" + "|".join(re.escape(k) for k in sorted(short_map, key=len, reverse=True)) + ")" + guard_r)
                         if short_map else None)

    def find(self, text: str) -> List[str]:
        found: List[str] = []
        if self.long_rx:
            for m in self.long_rx.finditer(text):
                s = self.long_map.get(m.group(1).lower())
                if s and s not in found:
                    found.append(s)
        if self.short_rx:
            for m in self.short_rx.finditer(text):
                s = self.short_map.get(m.group(1))
                if s and s not in found:
                    found.append(s)
        return found


# ------------------------------------------------------------------------------
# 3. NEWS COLLECTION
# ------------------------------------------------------------------------------
def build_feeds(lookback_days: int, custom_urls: List[str]) -> Dict[str, str]:
    feeds = dict(DIRECT_FEEDS)
    for site, domain in SITE_FEEDS.items():
        for topic in SITE_TOPICS:
            q = f"site:{domain} {topic} when:{lookback_days}d".replace("  ", " ")
            feeds[f"{site} (via Google News)" + (f" - {topic}" if topic else "")] = GNEWS.format(q=quote_plus(q))
    for name, q in GENERIC_QUERIES.items():
        feeds[name] = GNEWS.format(q=quote_plus(f"{q} when:{lookback_days}d"))
    for i, u in enumerate(custom_urls, 1):
        feeds[f"Custom feed {i}"] = u
    return feeds


def _clean(txt: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", txt or "")).strip()


def fetch_one(name: str, url: str) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    try:
        if feedparser is None:
            return name, [], "feedparser not installed"
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        fp = feedparser.parse(r.content)
        out = []
        for e in fp.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            pub = datetime(*ts[:6], tzinfo=timezone.utc) if ts else None
            title = _clean(e.get("title", ""))
            title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title) if "news.google.com" in url else title   # strip " - Source"
            out.append({"source": name, "title": title, "link": e.get("link", ""), "published": pub,
                        "summary": _clean(e.get("summary", ""))[:400]})
        return name, out, (None if out else "no entries returned")
    except Exception as exc:  # network / parse problems must never crash the app
        return name, [], str(exc)[:140]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all(feeds: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda kv: fetch_one(kv[0], kv[1]), feeds.items()))
    items, status = [], []
    for name, ents, err in results:
        items.extend(ents)
        status.append({"Feed": name, "Items": len(ents), "Status": "OK" if not err else f"FAILED: {err}"})
    return items, status


# ------------------------------------------------------------------------------
# 4. CATALYST SCORING
# ------------------------------------------------------------------------------
def lexicon_score(text: str) -> Tuple[float, List[str]]:
    hits, total = [], 0.0
    for lab, w, rx in _EVENT_RX:
        if rx.search(text):
            hits.append(lab)
            total += w
    if hits and UNCERTAIN.search(text):
        total *= 0.7
    return max(-4.0, min(4.0, total)), hits


def claude_rescore(df: pd.DataFrame, api_key: str, model: str, max_items: int = 80) -> Tuple[pd.DataFrame, Optional[str]]:
    """Optional: let Claude judge direction/event for the most relevant headlines. Falls back silently on any error."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        sub = df.sort_values("score", key=lambda s: s.abs(), ascending=False).head(max_items)
        lines = "\n".join(f"{i}. {r.title}" for i, r in zip(sub.index, sub.itertuples()))
        prompt = ("You are an equity analyst. For each numbered Indian-market headline, judge the effect on the named company's share price "
                  "over the next 3-6 months. Reply with ONLY a JSON list of objects: "
                  '{"id": <number>, "direction": <integer -2..2>, "event": "<2-4 word label>", "horizon_months": <1, 3, 6 or 0 if unclear>}. '
                  "Be conservative: routine news, opinion pieces and list articles are 0.\n\n" + lines)
        resp = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        arr = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        df = df.copy()
        df["claude_event"] = None
        for o in arr:
            idx = int(o["id"])
            if idx in df.index:
                df.at[idx, "score"] = float(o["direction"]) * 1.5
                df.at[idx, "claude_event"] = str(o.get("event", ""))[:40]
        return df, None
    except Exception as exc:
        return df, f"Claude re-scoring skipped: {str(exc)[:120]}"


def score_items(items: List[Dict[str, Any]], matcher: Matcher, half_life: float, lookback_days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    seen, rows = set(), []
    for it in items:
        key = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        pub = it["published"] or now
        age = max((now - pub).total_seconds() / 86400.0, 0.0)
        if age > lookback_days:
            continue
        text = f"{it['title']}. {it['summary']}"
        score, events = lexicon_score(text)
        syms = matcher.find(it["title"]) or matcher.find(it["summary"][:200])
        low = it["title"].lower()
        themes = [t for t, rxs in THEME_RX.items() if any(rx.search(low) for rx in rxs)]
        rows.append({"source": it["source"], "title": it["title"], "link": it["link"], "published": pub, "age_days": age,
                     "recency": 0.5 ** (age / half_life), "score": score, "events": events, "symbols": syms, "themes": themes,
                     "listicle": bool(LISTICLE.search(it["title"])) or len(syms) >= 4})
    return pd.DataFrame(rows)


def explode_stock_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        if not r.symbols:
            continue
        share = 1.0 if len(r.symbols) == 1 else 0.7
        pen = 0.35 if r.listicle else 1.0
        for s in r.symbols:
            rows.append({"symbol": s, "title": r.title, "link": r.link, "source": r.source, "age_days": r.age_days,
                         "events": r.events, "raw": r.score, "eff": r.score * r.recency * share * pen})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 5. PRICE CONFIRMATION
# ------------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    v = (100 - 100 / (1 + rs)).iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def momentum_metrics(df: pd.DataFrame, bench: pd.Series) -> Optional[Dict[str, float]]:
    if df is None or df.empty or "Close" not in df:
        return None
    c = df["Close"].dropna()
    if len(c) < 120:
        return None
    v = df["Volume"].reindex(c.index).fillna(0)
    ma50, ma200 = c.rolling(50).mean().iloc[-1], (c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan)
    px = float(c.iloc[-1])
    r3 = px / float(c.iloc[-64]) - 1
    b = bench.dropna()
    b3 = float(b.iloc[-1] / b.iloc[-64] - 1) if len(b) >= 64 else 0.0
    rs3 = (r3 - b3) * 100
    v20, v60 = v.iloc[-20:].mean(), v.iloc[-60:].mean()
    vol_ratio = float(v20 / v60) if v60 > 0 else float("nan")
    hi52 = float(c.iloc[-252:].max())
    off_high = (px / hi52 - 1) * 100
    liq_cr = float((c.iloc[-20:] * v.iloc[-20:]).mean() / 1e7)
    rsi = _rsi(c)
    ext50 = (px / ma50 - 1) * 100 if ma50 else float("nan")

    pts = 0.0
    pts += 15 if px > ma50 else 0
    pts += 15 if (pd.notna(ma200) and ma50 > ma200) else 0
    pts += 10 if (pd.notna(ma200) and px > ma200) else 0
    pts += 15 + 15 * max(-1.0, min(1.0, rs3 / 15.0))
    pts += 15 * max(0.0, min(1.0, ((vol_ratio if pd.notna(vol_ratio) else 0.8) - 0.8) / 0.8))
    pts += 15 if off_high >= -5 else 10 if off_high >= -15 else 5 if off_high >= -25 else 0
    if pd.notna(rsi) and rsi > 78:
        pts -= 10
    if pd.notna(ext50) and ext50 > 35:
        pts -= 10
    return {"mom_pts": float(max(0, min(100, pts))), "price": px, "rs3m": rs3, "r3m": r3 * 100, "vs50": ext50,
            "vs200": (px / ma200 - 1) * 100 if pd.notna(ma200) else float("nan"), "rsi": rsi, "vol_ratio": vol_ratio,
            "off_high": off_high, "liq_cr": liq_cr}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(symbols: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
    tick = [f"{s}.NS" for s in symbols] + [BENCH]
    out: Dict[str, pd.DataFrame] = {}
    try:
        data = yf.download(" ".join(tick), period="1y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True)
    except Exception:
        return out
    for t in tick:
        try:
            d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            d = d.dropna(how="all")
            if not d.empty:
                out[t] = d
        except Exception:
            continue
    return out


# ------------------------------------------------------------------------------
# 6. PIPELINE
# ------------------------------------------------------------------------------
def label_setup(news_pts: float, net: float, mom: Optional[float], rsi: float) -> str:
    if net <= -2:
        return "AVOID - negative news flow"
    if mom is None or pd.isna(mom):
        return "News only - no price data"
    if news_pts >= 62 and mom >= 55 and (pd.isna(rsi) or rsi <= 75):
        return "Catalyst + price confirmed"
    if news_pts >= 62 and mom >= 70:
        return "Strong but extended - wait for pullback"
    if news_pts >= 62:
        return "Catalyst, awaiting price confirmation"
    return "Watch"


def horizon_of(events: List[str]) -> str:
    s = set(events)
    if s & LONG_HORIZON:
        return "3-6 months (structural catalyst)"
    if s & SHORT_HORIZON:
        return "1-3 months (event-driven)"
    return "Unclear"


def run_screen(cfg: Dict[str, Any]) -> Dict[str, Any]:
    universe, uni_src = load_universe()
    matcher = Matcher(universe)
    feeds = {k: v for k, v in build_feeds(cfg["lookback"], cfg["custom"]).items() if k in cfg["enabled"] or k.startswith("Custom")}
    raw_items, feed_status = fetch_all(feeds)
    notes: List[str] = []
    items = score_items(raw_items, matcher, cfg["half_life"], cfg["lookback"])
    if items.empty:
        return {"items": items, "stocks": pd.DataFrame(), "themes": pd.DataFrame(), "industries": pd.DataFrame(),
                "feeds": pd.DataFrame(feed_status), "universe_src": uni_src, "notes": ["No headlines were collected."]}
    if cfg.get("claude_key"):
        items, err = claude_rescore(items, cfg["claude_key"], cfg["claude_model"])
        if err:
            notes.append(err)
    srows = explode_stock_rows(items)
    themes = []
    for r in items.itertuples():
        for t in r.themes:
            themes.append({"theme": t, "eff": r.score * r.recency * (0.35 if r.listicle else 1.0), "rec": r.recency})
    theme_df = (pd.DataFrame(themes).groupby("theme").agg(Headlines=("eff", "size"), Net_catalyst=("eff", "sum"), Buzz=("rec", "sum"))
                .sort_values("Net_catalyst", ascending=False).round(2).reset_index() if themes else pd.DataFrame())
    if srows.empty:
        return {"items": items, "stocks": pd.DataFrame(), "themes": theme_df, "industries": pd.DataFrame(),
                "feeds": pd.DataFrame(feed_status), "universe_src": uni_src, "notes": notes + ["No NSE stocks were recognised in the headlines."]}

    agg = srows.groupby("symbol").agg(net=("eff", "sum"), headlines=("title", "nunique"),
                                      positives=("eff", lambda s: int((s > 0).sum())), latest_age=("age_days", "min")).reset_index()
    ev = srows.explode("events").dropna(subset=["events"]).groupby(["symbol", "events"]).size().reset_index(name="n")
    top_ev = ev.sort_values("n", ascending=False).groupby("symbol")["events"].apply(lambda s: list(s)[:3]).to_dict()
    agg["events"] = agg["symbol"].map(top_ev).apply(lambda x: x if isinstance(x, list) else [])
    agg["news_pts"] = 50 + 50 * np.tanh(agg["net"] / 6.0)
    agg = agg[agg["headlines"] >= cfg["min_headlines"]].merge(universe, on="symbol", how="left")

    cand = agg.sort_values("net", ascending=False).head(cfg["max_candidates"])
    prices = fetch_prices(tuple(cand["symbol"]))
    bench = prices.get(BENCH, pd.DataFrame()).get("Close", pd.Series(dtype=float))
    mrows = []
    for s in cand["symbol"]:
        m = momentum_metrics(prices.get(f"{s}.NS"), bench) if len(bench) else None
        mrows.append({"symbol": s, **(m or {})})
    if not prices:
        notes.append("Price data could not be downloaded (Yahoo unreachable) - scores below are news-only.")
    stocks = cand.merge(pd.DataFrame(mrows), on="symbol", how="left")
    for col in ["mom_pts", "price", "rs3m", "r3m", "vs50", "vs200", "rsi", "vol_ratio", "off_high", "liq_cr"]:
        if col not in stocks:
            stocks[col] = np.nan
    stocks["swing_score"] = np.where(stocks["mom_pts"].notna(), 0.5 * stocks["news_pts"] + 0.5 * stocks["mom_pts"], 0.5 * stocks["news_pts"])
    stocks["setup"] = [label_setup(r.news_pts, r.net, r.mom_pts, r.rsi) for r in stocks.itertuples()]
    stocks["horizon"] = stocks["events"].apply(horizon_of)
    if cfg["min_liq"] > 0:
        stocks = stocks[stocks["liq_cr"].isna() | (stocks["liq_cr"] >= cfg["min_liq"])]
    avoid = stocks[stocks["net"] <= -2].sort_values("net").reset_index(drop=True)
    stocks = stocks[stocks["net"] > 0].sort_values("swing_score", ascending=False).reset_index(drop=True)

    ind = (stocks.groupby("industry").agg(Stocks_in_news=("symbol", "size"), Net_news=("net", "sum"), Avg_momentum=("mom_pts", "mean"),
                                          Confirmed=("setup", lambda s: int((s == "Catalyst + price confirmed").sum())))
           .round(1).sort_values(["Confirmed", "Net_news"], ascending=False).reset_index())
    return {"items": items, "stocks": stocks, "avoid": avoid, "stock_rows": srows, "themes": theme_df, "industries": ind,
            "feeds": pd.DataFrame(feed_status), "universe_src": uni_src, "notes": notes}


# ------------------------------------------------------------------------------
# 7. TRACK RECORD
# ------------------------------------------------------------------------------
LOG_COLS = ["date", "symbol", "entry_price", "swing_score", "setup"]


def evaluate_log(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    syms = tuple(sorted(set(log["symbol"])))
    px = fetch_prices(syms)
    bench = px.get(BENCH, pd.DataFrame()).get("Close", pd.Series(dtype=float)).dropna()
    rows = []
    for r in log.itertuples():
        d = px.get(f"{r.symbol}.NS")
        if d is None or d.empty or bench.empty:
            continue
        last = float(d["Close"].dropna().iloc[-1])
        pdate = pd.to_datetime(r.date)
        bsub = bench[bench.index.tz_localize(None) >= pdate] if getattr(bench.index, "tz", None) is not None else bench[bench.index >= pdate]
        if bsub.empty:
            continue
        b0, b1 = float(bsub.iloc[0]), float(bench.iloc[-1])
        ret, bret = (last / float(r.entry_price) - 1) * 100, (b1 / b0 - 1) * 100
        rows.append({"date": r.date, "symbol": r.symbol, "setup": r.setup, "days_held": (pd.Timestamp.today().normalize() - pdate).days,
                     "return_%": round(ret, 1), "nifty_%": round(bret, 1), "excess_%": round(ret - bret, 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 8. UI
# ------------------------------------------------------------------------------
def fmt_events(x: Any) -> str:
    return ", ".join(x) if isinstance(x, list) else ""


def main() -> None:
    st.set_page_config(page_title="News + Momentum Swing Screener", layout="wide")
    st.title("News + Momentum Swing Screener")
    st.caption("Finds stocks with a fresh catalyst AND price confirmation, then hands them to your Value Analyser. "
               "Setup candidates only - not forecasts, not advice.")

    with st.sidebar:
        st.header("Settings")
        lookback = st.slider("News look-back (days)", 2, 14, 7)
        half_life = st.slider("Recency half-life (days)", 1.0, 7.0, 3.0, 0.5, help="How quickly older headlines lose weight.")
        max_cand = st.slider("Stocks to price-check", 20, 100, 50, 5)
        min_head = st.slider("Min headlines per stock", 1, 5, 1)
        min_liq = st.number_input("Min avg daily turnover (Rs Cr)", 0.0, 500.0, 5.0, 1.0, help="Filters out illiquid, operator-prone names.")
        all_feeds = list(build_feeds(lookback, []).keys())
        enabled = st.multiselect("Feeds", all_feeds, default=all_feeds)
        custom_raw = st.text_area("Extra RSS URLs (one per line)", "")
        custom = [u.strip() for u in custom_raw.splitlines() if u.strip().startswith("http")]
        st.markdown("---")
        analyser_url = st.text_input("Your Value Analyser URL", "", placeholder="https://your-analyser.streamlit.app",
                                     help="If set, each pick gets an 'Open in Analyser' link.")
        st.markdown("---")
        st.subheader("Optional: Claude re-scoring")
        key = ""
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            key = ""
        key = st.text_input("Anthropic API key", value=key, type="password", help="Optional. Better headline judgement than keyword rules.")
        model = st.text_input("Model", "claude-haiku-5-5")
        run = st.button("Run screener", type="primary", use_container_width=True)

    cfg = {"lookback": lookback, "half_life": half_life, "max_candidates": max_cand, "min_headlines": min_head, "min_liq": min_liq,
           "enabled": enabled, "custom": custom, "claude_key": key or None, "claude_model": model}
    if run:
        with st.spinner("Collecting headlines, scoring catalysts and checking prices..."):
            st.session_state["screen"] = run_screen(cfg)
            st.session_state["screen_time"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    res = st.session_state.get("screen")
    if "log" not in st.session_state:
        st.session_state["log"] = pd.DataFrame(columns=LOG_COLS)

    tabs = st.tabs(["Shortlist", "Sector & Theme Heat", "Headlines (audit)", "Feed Health", "Track Record"])
    if not res:
        with tabs[0]:
            st.info("Press **Run screener** in the sidebar.")
        _render_log_tab(tabs[4])
        return
    st.caption(f"Last run: {st.session_state['screen_time']}  |  Universe: {res['universe_src']}")
    for n in res["notes"]:
        st.warning(n)
    stocks = res["stocks"]

    with tabs[0]:
        if stocks.empty:
            st.info("No stocks passed the filters. Try a longer look-back or lower the liquidity filter.")
        else:
            view = stocks.copy()
            view["Catalysts"] = view["events"].apply(fmt_events)
            if analyser_url:
                base = analyser_url.rstrip("/")
                view["Analyse"] = view["symbol"].apply(lambda s: f"{base}/?ticker={s}")
            cols = ["symbol", "name", "industry", "setup", "swing_score", "news_pts", "mom_pts", "net", "headlines", "Catalysts", "horizon",
                    "rs3m", "vs50", "rsi", "vol_ratio", "off_high", "liq_cr"] + (["Analyse"] if analyser_url else [])
            show = view[cols].rename(columns={"symbol": "Symbol", "name": "Company", "industry": "Industry", "setup": "Setup", "swing_score": "Swing score",
                                              "news_pts": "News pts", "mom_pts": "Momentum pts", "net": "Net news", "headlines": "Headlines",
                                              "horizon": "Catalyst horizon", "rs3m": "3M vs Nifty (pp)", "vs50": "% vs 50DMA", "rsi": "RSI",
                                              "vol_ratio": "Vol ratio (20d/60d)", "off_high": "% from 52w high", "liq_cr": "Turnover (Rs Cr)"})
            cfgc = {"Analyse": st.column_config.LinkColumn("Analyse", display_text="Open in Analyser")} if analyser_url else {}
            st.dataframe(show.round(2), hide_index=True, use_container_width=True, column_config=cfgc)
            st.download_button("Download shortlist (CSV)", show.round(2).to_csv(index=False).encode(), "swing_shortlist.csv", "text/csv")
            c1, c2 = st.columns(2)
            top_n = c1.slider("Log top N picks to Track Record", 3, 20, 10)
            if c2.button("Log today's picks"):
                pk = stocks[stocks["setup"].isin(["Catalyst + price confirmed", "Catalyst, awaiting price confirmation"])].head(top_n)
                pk = pk[pk["price"].notna()]
                new = pd.DataFrame({"date": datetime.now().strftime("%Y-%m-%d"), "symbol": pk["symbol"], "entry_price": pk["price"].round(2),
                                    "swing_score": pk["swing_score"].round(1), "setup": pk["setup"]})
                st.session_state["log"] = pd.concat([st.session_state["log"], new]).drop_duplicates(["date", "symbol"])
                st.success(f"Logged {len(new)} pick(s).")
            st.subheader("Why each top stock is on the list")
            for r in stocks.head(10).itertuples():
                with st.expander(f"{r.symbol} - {r.setup} (swing {r.swing_score:.0f})"):
                    rows = res["stock_rows"][res["stock_rows"]["symbol"] == r.symbol].sort_values("age_days")
                    for x in rows.itertuples():
                        st.markdown(f"- [{x.title}]({x.link}) - *{x.source}, {x.age_days:.1f}d ago, score {x.eff:+.2f}* {fmt_events(x.events)}")
        avoid = res.get("avoid", pd.DataFrame())
        if not avoid.empty:
            st.subheader("Negative news flow (avoid / review if you hold)")
            st.dataframe(avoid.assign(Catalysts=avoid["events"].apply(fmt_events))[["symbol", "name", "industry", "net", "headlines", "Catalysts"]]
                         .rename(columns={"symbol": "Symbol", "name": "Company", "industry": "Industry", "net": "Net news", "headlines": "Headlines"}).round(2),
                         hide_index=True, use_container_width=True)
        st.caption("A high score means 'fresh catalyst + trend confirmation', not a prediction. Check the stock in your Value Analyser before acting.")

    with tabs[1]:
        st.subheader("Industry heat (from stock-level news)")
        st.dataframe(res["industries"], hide_index=True, use_container_width=True) if not res["industries"].empty else st.info("No data.")
        st.subheader("Theme heat (from headline keywords)")
        st.dataframe(res["themes"], hide_index=True, use_container_width=True) if not res["themes"].empty else st.info("No theme headlines found.")
        st.caption("Net catalyst = recency-weighted sum of headline scores; Buzz = recency-weighted headline count.")

    with tabs[2]:
        it = res["items"].copy()
        sym_f = st.text_input("Filter by symbol (optional)", "").strip().upper()
        if sym_f:
            it = it[it["symbols"].apply(lambda s: sym_f in s)]
        it["Stocks"] = it["symbols"].apply(lambda s: ", ".join(s))
        it["Events"] = it["events"].apply(fmt_events)
        st.dataframe(it[["published", "source", "title", "Stocks", "Events", "score", "link"]].sort_values("published", ascending=False),
                     hide_index=True, use_container_width=True, column_config={"link": st.column_config.LinkColumn("link")})

    with tabs[3]:
        st.dataframe(res["feeds"], hide_index=True, use_container_width=True)
        st.caption("Feeds change or block requests from time to time. A FAILED row is normal - edit the feed list in the sidebar.")
    _render_log_tab(tabs[4])


def _render_log_tab(tab: Any) -> None:
    with tab:
        st.subheader("Does this screener actually work? Measure it.")
        st.caption("Log picks, come back in 1-3 months and compare against the Nifty. Streamlit Cloud wipes local files on reboot, "
                   "so download the log regularly and re-upload it here.")
        up = st.file_uploader("Restore log (CSV)", type="csv", key="log_up")
        if up is not None and st.session_state.get("_log_sig") != (up.name, up.size):
            try:
                st.session_state["log"] = pd.read_csv(up)[LOG_COLS]
                st.session_state["_log_sig"] = (up.name, up.size)
            except Exception as exc:
                st.error(f"Could not read that CSV: {exc}")
        log = st.session_state["log"]
        st.dataframe(log, hide_index=True, use_container_width=True)
        if not log.empty:
            st.download_button("Download log (CSV)", log.to_csv(index=False).encode(), "swing_pick_log.csv", "text/csv")
            if st.button("Evaluate vs Nifty"):
                ev = evaluate_log(log)
                if ev.empty:
                    st.info("Could not fetch prices to evaluate.")
                else:
                    st.dataframe(ev, hide_index=True, use_container_width=True)
                    mature = ev[ev["days_held"] >= 20]
                    if not mature.empty:
                        st.metric("Avg excess return vs Nifty (picks held 20+ days)", f"{mature['excess_%'].mean():+.1f}%",
                                  f"hit rate {(mature['excess_%'] > 0).mean() * 100:.0f}% over {len(mature)} picks")
                    else:
                        st.caption("Picks are still younger than 20 days - too early to judge.")


if __name__ == "__main__":
    main()