import os, time, math, json, threading, datetime as dt
from datetime import timezone, timedelta
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request

# ─────────────────────────────────────────────────────────────
# Boot & ENV
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL","").strip()
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY","").strip()
TWELVE_KEY = os.getenv("TWELVE_API_KEY","").strip()
PORT = int(os.getenv("PORT","10000"))

# Schwab (Phase 3)
SCHWAB_CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID","").strip()
SCHWAB_CLIENT_SECRET = os.getenv("SCHWAB_CLIENT_SECRET","").strip()
REDIRECT_URI = os.getenv("REDIRECT_URI","").strip()
SCHWAB_ACCOUNT_ID = os.getenv("SCHWAB_ACCOUNT_ID","").strip()
TRADE_MODE = os.getenv("TRADE_MODE","PAPER").upper().strip() # PAPER|LIVE

# Universe handling
DEFAULT_UNIVERSE = "GME,AMC,IONQ,NAK,SMR,FFIE,BNZI,SPY,QQQ,TSLA,NVDA,AMD,AAPL,META,MSFT"
UNIVERSE = [s.strip().upper() for s in (os.getenv("UNIVERSE", DEFAULT_UNIVERSE)).split(",") if s.strip()]

# Scanner cadence & params
SCAN_INTERVAL_SEC = 60
RVOL_LOOKBACK = 20
RVOL_MIN = 2.0
BB_LEN = 20
KC_LEN = 20
KC_MULT = 1.5
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9

# ─────────────────────────────────────────────────────────────
# Flask heartbeat and minimal OAuth helper endpoints
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def root():
    return "BeastMode Trading Bot — live", 200

# Optional: receive Schwab OAuth code (manual once), exchange for tokens
SCHWAB_TOKEN_CACHE = {"access_token": None, "refresh_token": None, "exp": 0}

@app.route("/callback")
def schwab_callback():
    code = request.args.get("code")
    if not code:
        return "Missing code", 400
    ok, msg = schwab_exchange_code(code)
    return ("OK" if ok else "ERROR") + ": " + msg, 200 if ok else 500

def run_server():
    app.run(host="0.0.0.0", port=PORT, debug=False)

# ─────────────────────────────────────────────────────────────
# Alpha Vantage helpers
# ─────────────────────────────────────────────────────────────
def alpha_intraday(symbol: str, interval="5min", output="compact"):
    url = ("https://www.alphavantage.co/query"
           f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
           f"&interval={interval}&outputsize={output}&apikey={ALPHA_KEY}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    key = f"Time Series ({interval})"
    if key not in js:
        raise RuntimeError(f"Alpha error {symbol}: {js}")
    df = pd.DataFrame(js[key]).T.rename(columns={
        "1. open":"open","2. high":"high","3. low":"low",
        "4. close":"close","5. volume":"volume"
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def alpha_quote(symbol: str):
    url = ("https://www.alphavantage.co/query"
           f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json().get("Global Quote", {})
    px = float(js.get("05. price", "0") or 0)
    return px

def alpha_most_actives(limit=10):
    return ["SPY","QQQ","TSLA","NVDA","AAPL","AMD","META","MSFT"][:limit]

# ─────────────────────────────────────────────────────────────
# Twelve Data helpers + API endpoints
# ─────────────────────────────────────────────────────────────
def twelve_quote(symbol: str):
    """Fetch realtime quote from Twelve Data."""
    if not TWELVE_KEY:
        raise RuntimeError("TWELVE_API_KEY missing")
    url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={TWELVE_KEY}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    js = r.json()
    if "code" in js:
        raise RuntimeError(f"TwelveData error: {js}")
    return {
        "symbol": js.get("symbol"),
        "name": js.get("name"),
        "exchange": js.get("exchange"),
        "price": float(js.get("price", 0)),
        "percent_change": js.get("percent_change"),
        "change": js.get("change"),
        "volume": js.get("volume"),
        "open": js.get("open"),
        "high": js.get("high"),
        "low": js.get("low"),
        "previous_close": js.get("previous_close"),
        "fifty_two_week_high": js.get("fifty_two_week", {}).get("high")
            if isinstance(js.get("fifty_two_week"), dict) else None,
        "fifty_two_week_low": js.get("fifty_two_week", {}).get("low")
            if isinstance(js.get("fifty_two_week"), dict) else None,
        "datetime": js.get("datetime")
    }

def twelve_batch(symbols):
    """Fetch multiple symbols in one call."""
    if not TWELVE_KEY:
        raise RuntimeError("TWELVE_API_KEY missing")
    symlist = ",".join([s.strip().upper() for s in symbols.split(",") if s.strip()])
    url = f"https://api.twelvedata.com/quote?symbol={symlist}&apikey={TWELVE_KEY}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json()
    if "code" in js:
        raise RuntimeError(f"TwelveData error: {js}")
    return js

@app.route("/api/quote")
def api_quote():
    sym = request.args.get("symbol","AMC").upper()
    try:
        data = twelve_quote(sym)
        return data, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/api/screener")
def api_screener():
    syms = request.args.get("symbols","AMC,GME").upper()
    try:
        data = twelve_batch(syms)
        return data, 200
    except Exception as e:
        return {"error": str(e)}, 500

# ─────────────────────────────────────────────────────────────
# Indicators & Signals
# ─────────────────────────────────────────────────────────────
def calc_rvol(vol, lookback=20):
    avg = pd.Series(vol).rolling(lookback).mean()
    return pd.Series(vol) / avg

def bollinger(close, length=20, mult=2.0):
    ma = pd.Series(close).rolling(length).mean()
    sd = pd.Series(close).rolling(length).std(ddof=0)
    return ma, ma + mult*sd, ma - mult*sd

def keltner(high, low, close, length=20, mult=1.5):
    ema = pd.Series(close).ewm(span=length, adjust=False).mean()
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = (pd.Series(high) - pd.Series(close).shift(1)).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=length, adjust=False).mean()
    return ema, ema + mult*atr, ema - mult*atr

def macd_signal(close):
    fast = pd.Series(close).ewm(span=MACD_FAST, adjust=False).mean()
    slow = pd.Series(close).ewm(span=MACD_SLOW, adjust=False).mean()
    macd = fast - slow
    sig = macd.ewm(span=MACD_SIG, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def is_ttm_squeeze(df):
    ma, bb_u, bb_l = bollinger(df["close"], BB_LEN, 2.0)
    kc_m, kc_u, kc_l = keltner(df["high"], df["low"], df["close"], KC_LEN, KC_MULT)
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    squeeze_off = (bb_u > kc_u) | (bb_l < kc_l)
    sig_bull = squeeze_on.shift(1) & squeeze_off & (df["close"] > ma)
    sig_bear = squeeze_on.shift(1) & squeeze_off & (df["close"] < ma)
    bb_width = bb_u - bb_l
    kc_width = kc_u - kc_l
    return squeeze_on, sig_bull, sig_bear, bb_width, kc_width

def patterns(df):
    tags = []
    N = 20
    if df["close"].iloc[-1] >= df["high"].rolling(N).max().iloc[-2]:
        tags.append("Breakout")
    ma20 = pd.Series(df["close"]).rolling(20).mean()
    if df["low"].iloc[-2] <= ma20.iloc[-2] and df["close"].iloc[-1] > ma20.iloc[-1]:
        tags.append("PB-Bounce")
    if df["close"].iloc[-1] > df["close"].iloc[-4] and df["volume"].iloc[-1] < df["volume"].iloc[-4]:
        tags.append("Weak-Vol-Push")
    ema20 = pd.Series(df["close"]).ewm(span=20, adjust=False).mean()
    if df["close"].iloc[-1] > ema20.iloc[-1] and df["close"].iloc[-1] > df["close"].iloc[-6]:
        tags.append("MTF-Up")
    return tags

def score_signal(rvol_now, hist_now, bull_trig, bear_trig):
    score = 0
    if rvol_now >= RVOL_MIN: score += 2
    if hist_now > 0: score += 1
    if hist_now < 0: score -= 1
    if bull_trig: score += 2
    if bear_trig: score -= 2
    return max(-5, min(5, score))

# (Rest of your Schwab + Discord + scanning logic stays exactly as before)
