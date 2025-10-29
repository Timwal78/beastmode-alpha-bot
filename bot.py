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
# Alpha Vantage helpers (rate-limited)
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
    url = ( "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}" )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json().get("Global Quote", {})
    px = float(js.get("05. price", "0") or 0)
    return px

def alpha_most_actives(limit=10):
    # Poor-man's “actives”: we’ll just return a stable list (extend later with a separate source)
    return ["SPY","QQQ","TSLA","NVDA","AAPL","AMD","META","MSFT"][:limit]

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
    # Pullback to MA20 with bounce
    ma20 = pd.Series(df["close"]).rolling(20).mean()
    if df["low"].iloc[-2] <= ma20.iloc[-2] and df["close"].iloc[-1] > ma20.iloc[-1]:
        tags.append("PB-Bounce")
    # Weak volume push
    if df["close"].iloc[-1] > df["close"].iloc[-4] and df["volume"].iloc[-1] < df["volume"].iloc[-4]:
        tags.append("Weak-Vol-Push")
    # Multi-TF align: current close above 20EMA and last 30m trend up (approx)
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

# ─────────────────────────────────────────────────────────────
# Options: heuristic + Schwab chain (Phase 3)
# ─────────────────────────────────────────────────────────────
def next_friday(from_dt=None):
    if from_dt is None: from_dt = dt.datetime.now(timezone.utc)
    days_ahead = (4 - from_dt.weekday()) % 7
    if days_ahead == 0 and from_dt.hour >= 20:
        days_ahead = 7
    return (from_dt + timedelta(days=days_ahead)).date().isoformat()

def heuristic_option(symbol, last_price, direction):
    pct = 0.05
    strike = round((1+pct)*last_price, 2) if direction=="CALL" else round((1-pct)*last_price, 2)
    return {"symbol":symbol, "side":"C" if direction=="CALL" else "P", "strike":strike, "expiry":next_friday()}

# NOTE: Schwab API specifics vary by app; we keep a safe wrapper that returns None if not available.
SCHWAB_BASE = "https://api.schwabapi.com" # placeholder base

def schwab_exchange_code(code: str):
    # Exchange authorization code for tokens (one-time in browser via /callback)
    try:
        data = {
            "grant_type":"authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": SCHWAB_CLIENT_ID
        }
        r = requests.post(f"{SCHWAB_BASE}/v1/oauth/token", data=data, timeout=30)
        if r.status_code != 200:
            return False, f"Token exchange failed: {r.text}"
        js = r.json()
        SCHWAB_TOKEN_CACHE["access_token"] = js.get("access_token")
        SCHWAB_TOKEN_CACHE["refresh_token"] = js.get("refresh_token")
        SCHWAB_TOKEN_CACHE["exp"] = time.time() + int(js.get("expires_in", 1800)) - 60
        return True, "Tokens stored (memory)"
    except Exception as e:
        return False, str(e)

def schwab_refresh():
    if not SCHWAB_TOKEN_CACHE.get("refresh_token"):
        return False
    try:
        data = {
            "grant_type":"refresh_token",
            "refresh_token": SCHWAB_TOKEN_CACHE["refresh_token"],
            "client_id": SCHWAB_CLIENT_ID
        }
        r = requests.post(f"{SCHWAB_BASE}/v1/oauth/token", data=data, timeout=30)
        if r.status_code != 200:
            return False
        js = r.json()
        SCHWAB_TOKEN_CACHE["access_token"] = js.get("access_token")
        SCHWAB_TOKEN_CACHE["refresh_token"] = js.get("refresh_token", SCHWAB_TOKEN_CACHE["refresh_token"])
        SCHWAB_TOKEN_CACHE["exp"] = time.time() + int(js.get("expires_in", 1800)) - 60
        return True
    except:
        return False

def schwab_headers():
    tok = SCHWAB_TOKEN_CACHE.get("access_token")
    if not tok or time.time() > SCHWAB_TOKEN_CACHE.get("exp",0):
        schwab_refresh()
        tok = SCHWAB_TOKEN_CACHE.get("access_token")
    if not tok: return None
    return {"Authorization": f"Bearer {tok}"}

def schwab_option_chain(symbol: str, direction: str, last_price: float):
    """
    Tries to fetch an option chain and pick ~0.30–0.40 delta contract for next Friday,
    min OI, tight spread. Returns dict or None.
    """
    try:
        hdr = schwab_headers()
        if not hdr: 
            return None # not authenticated yet
        # Placeholder endpoint/params – adjust to your Schwab app spec if needed.
        expiry = next_friday()
        r = requests.get(
            f"{SCHWAB_BASE}/marketdata/v1/chains",
            params={"symbol": symbol, "contractType": "CALL" if direction=="CALL" else "PUT", "fromDate": expiry, "toDate": expiry},
            headers=hdr, timeout=30
        )
        if r.status_code != 200:
            return None
        js = r.json()
        # Find closest delta ~0.35, min OI, small spread
        best = None
        target_delta = 0.35
        def score(opt):
            d = abs(abs(opt.get("delta",0.0)) - target_delta)
            spread = abs(opt.get("ask",0)-opt.get("bid",0))
            oi = -(opt.get("openInterest",0))
            return (d, spread, oi)
        # Traverse example structure cautiously (shape can vary)
        for date_group in js.get("callExpDateMap" if direction=="CALL" else "putExpDateMap", {}).values():
            for strike_str, contracts in date_group.items():
                for opt in contracts:
                    # build normalized record
                    rec = {
                        "strike": float(opt.get("strikePrice", strike_str)),
                        "expiry": opt.get("expirationDate", expiry)[:10],
                        "bid": float(opt.get("bid",0)),
                        "ask": float(opt.get("ask",0)),
                        "delta": abs(float(opt.get("delta",0))),
                        "openInterest": int(opt.get("openInterest",0))
                    }
                    if best is None or score(rec) < score(best):
                        best = rec
        if not best: 
            return None
        side = "C" if direction=="CALL" else "P"
        return {"symbol": symbol, "side": side, "strike": round(best["strike"],2), "expiry": best["expiry"]}
    except Exception as e:
        print("schwab_option_chain error:", e)
        return None

# (Optional) Order placement stub (LIVE/PAPER switch). No-ops if not configured.
def schwab_place_order(symbol: str, direction: str, qty: int, use_options=False, opt=None):
    if TRADE_MODE not in ("PAPER","LIVE"): 
        return False, "invalid TRADE_MODE"
    hdr = schwab_headers()
    if not hdr or not SCHWAB_ACCOUNT_ID:
        return False, "not authenticated"
    try:
        if use_options and opt:
            payload = {"orderType":"LIMIT","session":"NORMAL","duration":"DAY",
                       "orderStrategyType":"SINGLE",
                       "orderLegCollection":[{"instruction":"BUY_TO_OPEN" if direction=="CALL" else "BUY_TO_OPEN",
                                             "quantity":qty,
                                             "instrument":{"symbol":f"{symbol} {opt['expiry']} {opt['side']}{opt['strike']}",
                                                           "assetType":"OPTION"}}]}
        else:
            payload = {"orderType":"MARKET","session":"NORMAL","duration":"DAY",
                       "orderStrategyType":"SINGLE",
                       "orderLegCollection":[{"instruction":"BUY" if direction=="CALL" else "SELL_SHORT",
                                             "quantity":qty,
                                             "instrument":{"symbol":symbol,"assetType":"EQUITY"}}]}
        # Placeholder endpoint:
        r = requests.post(f"{SCHWAB_BASE}/trader/v1/accounts/{SCHWAB_ACCOUNT_ID}/orders", headers=hdr, json=payload, timeout=30)
        if r.status_code in (200,201):
            return True, "order placed"
        return False, f"order fail {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)

# ─────────────────────────────────────────────────────────────
# Discord helpers
# ─────────────────────────────────────────────────────────────
def post_discord(payload):
    if not DISCORD_WEBHOOK_URL: return
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print("Discord error:", e)

def embed_signal(sig):
    color = 0x15C39A if sig["bias"]=="BULL" else 0xE03A3A
    fields = [
        {"name":"Price","value":f"${sig['last_price']:.2f}","inline":True},
        {"name":"RVOL","value":f"{sig['rvol']:.2f}x","inline":True},
        {"name":"Score","value":f"{sig['score']}/5","inline":True},
        {"name":"Pattern","value":", ".join(sig['patterns']) or "—","inline":True},
        {"name":"Stop","value":f\"${sig['stop']:.2f}\",\"inline\":True},
        {"name":"Target","value":f\"${sig['target']:.2f}\",\"inline\":True},
    ]
    if sig.get("option"):
        o = sig["option"]
        fields.append({"name":"Option","value":f\"{sig['ticker']} {o['expiry']} {o['side']}{o['strike']}\",\"inline\":False})
    return {
        "title": f"🚀 {sig['ticker']} — {sig['bias']} Signal",
        "description": "BeastMode Squeeze Engine",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Mode: {TRADE_MODE} — Not financial advice"},
        "timestamp": dt.datetime.now(timezone.utc).isoformat()
    }

# ─────────────────────────────────────────────────────────────
# Core scan
# ─────────────────────────────────────────────────────────────
def scan_one(symbol):
    try:
        df = alpha_intraday(symbol, "5min", "compact")
        if len(df) < 60: return None
        rvol = calc_rvol(df["volume"], RVOL_LOOKBACK)
        squeeze_on, s_bull, s_bear, bb_w, kc_w = is_ttm_squeeze(df)
        macd, sig, hist = macd_signal(df["close"])

        last = df["close"].iloc[-1]
        rvol_now = float(rvol.iloc[-1]) if not np.isnan(rvol.iloc[-1]) else 0.0
        bull_trig = bool(s_bull.iloc[-1])
        bear_trig = bool(s_bear.iloc[-1])
        hist_now = float(hist.iloc[-1])

        if rvol_now < RVOL_MIN and not bull_trig and not bear_trig:
            return None

        bias = "BULL" if (hist_now > 0 or bull_trig) else "BEAR"
        sc = score_signal(rvol_now, hist_now, bull_trig, bear_trig)
        pats = patterns(df)

        # Risk box via KC width
        atr_proxy = float((kc_w.iloc[-1]) / 2.0) if not np.isnan(kc_w.iloc[-1]) else max(0.02*last, 0.1)
        stop = last - atr_proxy if bias=="BULL" else last + atr_proxy
        target = last + 2*atr_proxy if bias=="BULL" else last - 2*atr_proxy

        # Option via Schwab if tokens available; otherwise heuristic
        direction = "CALL" if bias=="BULL" else "PUT"
        opt = schwab_option_chain(symbol, direction, last) or heuristic_option(symbol, last, direction)

        return {
            "ticker": symbol,
            "last_price": last,
            "rvol": rvol_now,
            "score": sc,
            "bias": bias,
            "patterns": pats,
            "stop": stop,
            "target": target,
            "option": opt
        }
    except Exception as e:
        print(f"[{symbol}] scan error:", e)
        return None

def expand_universe(base):
    # Add top most-actives to widen coverage without editing .env
    try:
        add = alpha_most_actives(8)
        merged = list(dict.fromkeys(base + add))
        return merged
    except:
        return base

def scan_loop():
    post_discord({"embeds":[{
        "title":"🟢 BeastMode Bot ONLINE",

