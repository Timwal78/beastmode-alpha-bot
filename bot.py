import os, time, math, json, threading, datetime as dt
from datetime import timezone, timedelta
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask

# ─────────────────────────────────────────────────────────────
# Boot & ENV
# ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
ALPHA_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "10000"))

# Schwab creds (for phase 3 live trading – placeholders here)
SCHWAB_CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID", "").strip()
SCHWAB_CLIENT_SECRET = os.getenv("SCHWAB_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv("REDIRECT_URI", "").strip()

# You can edit this universe in code or load from an env/file later.
UNIVERSE = [
    # Squeeze hunting core list (edit freely):
    "GME","AMC","IONQ","NAK","SMR","FFIE","BNZI",
    # Add a few liquid options names for pattern confirmations:
    "SPY","QQQ","TSLA","NVDA","AMD","AAPL","META","MSFT"
]

# Scan cadence & thresholds
SCAN_INTERVAL_SEC = 60 # run loop every minute
RVOL_LOOKBACK = 20
RVOL_MIN = 2.0 # relative volume minimum
BB_LEN = 20
KC_LEN = 20
KC_MULT = 1.5
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG = 9

# ─────────────────────────────────────────────────────────────
# Heartbeat server (keeps Render web service happy)
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def root():
    return "BeastMode Alpha Bot — live", 200

def run_server():
    app.run(host="0.0.0.0", port=PORT, debug=False)

# ─────────────────────────────────────────────────────────────
# Data helpers (Alpha Vantage — free key is rate-limited)
# ─────────────────────────────────────────────────────────────
def alpha_intraday(symbol: str, interval="5min", output="compact"):
    """Return OHLCV DataFrame (most recent first index to oldest)."""
    url = ("https://www.alphavantage.co/query"
           f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
           f"&interval={interval}&outputsize={output}&apikey={ALPHA_KEY}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    key = f"Time Series ({interval})"
    if key not in js:
        raise RuntimeError(f"AlphaVantage error for {symbol}: {js}")
    df = pd.DataFrame(js[key]).T.rename(columns={
        "1. open":"open","2. high":"high","3. low":"low",
        "4. close":"close","5. volume":"volume"
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True) # oldest → newest
    return df

def alpha_quote(symbol: str):
    """Last trade price for quick option calc."""
    url = ( "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_KEY}" )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    js = r.json().get("Global Quote", {})
    px = float(js.get("05. price", "0") or 0)
    return px

# ─────────────────────────────────────────────────────────────
# Indicators & Signals
# ─────────────────────────────────────────────────────────────
def calc_rvol(vol, lookback=20):
    # RVOL = today vol / avg vol
    avg = pd.Series(vol).rolling(lookback).mean()
    rvol = pd.Series(vol) / avg
    return rvol

def bollinger(close, length=20, mult=2.0):
    ma = pd.Series(close).rolling(length).mean()
    sd = pd.Series(close).rolling(length).std(ddof=0)
    upper = ma + mult*sd
    lower = ma - mult*sd
    return ma, upper, lower

def keltner(high, low, close, length=20, mult=1.5):
    ema = pd.Series(close).ewm(span=length, adjust=False).mean()
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = (pd.Series(high) - pd.Series(close).shift(1)).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=length, adjust=False).mean()
    upper = ema + mult*atr
    lower = ema - mult*atr
    return ema, upper, lower

def is_ttm_squeeze(df):
    # Classic definition: BB inside KC → squeeze ON; BB breaks out → squeeze OFF trigger
    ma, bb_u, bb_l = bollinger(df["close"], BB_LEN, 2.0)
    kc_m, kc_u, kc_l = keltner(df["high"], df["low"], df["close"], KC_LEN, KC_MULT)
    bb_width = bb_u - bb_l
    kc_width = kc_u - kc_l
    squeeze_on = (bb_u < kc_u) & (bb_l > kc_l)
    squeeze_off = (bb_u > kc_u) | (bb_l < kc_l)
    # Trigger = squeeze just turned off and price > ma (bull) or < ma (bear)
    sig_bull = squeeze_on.shift(1) & squeeze_off & (df["close"] > ma)
    sig_bear = squeeze_on.shift(1) & squeeze_off & (df["close"] < ma)
    return squeeze_on, sig_bull, sig_bear, bb_width, kc_width

def macd_signal(close):
    fast = pd.Series(close).ewm(span=MACD_FAST, adjust=False).mean()
    slow = pd.Series(close).ewm(span=MACD_SLOW, adjust=False).mean()
    macd = fast - slow
    sig = macd.ewm(span=MACD_SIG, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def patterns(df):
    # Lightweight pattern tags (extend later)
    tags = []
    # Breakout: last close > N-day high
    N = 20
    if df["close"].iloc[-1] >= df["high"].rolling(N).max().iloc[-2]:
        tags.append("Breakout")
    # Divergence proxy: price up while volume down (last few bars)
    if df["close"].iloc[-1] > df["close"].iloc[-4] and df["volume"].iloc[-1] < df["volume"].iloc[-4]:
        tags.append("Weak-Vol-Push")
    return tags

def score_signal(rvol_now, hist_now, squeeze_bull, squeeze_bear):
    score = 0
    if rvol_now >= RVOL_MIN: score += 2
    if hist_now > 0: score += 1
    if hist_now < 0: score -= 1
    if squeeze_bull: score += 2
    if squeeze_bear: score -= 2
    # Cap range
    score = max(-5, min(5, score))
    return score

# ─────────────────────────────────────────────────────────────
# Option helper (suggested strikes/expiry if Schwab chain not used yet)
# ─────────────────────────────────────────────────────────────
def next_friday(from_dt=None):
    if from_dt is None: from_dt = dt.datetime.now(timezone.utc)
    # Friday = 4 (Mon=0)
    days_ahead = (4 - from_dt.weekday()) % 7
    if days_ahead == 0 and from_dt.hour >= 20: # past 8pm UTC → pick next week
        days_ahead = 7
    d = (from_dt + timedelta(days=days_ahead)).date()
    return d.isoformat()

def suggested_option(symbol, last_price, direction):
    # Direction: "CALL" or "PUT"
    # Heuristic strike selection near 0.30–0.40 delta ≈ ~5% OTM for fast movers
    otm_pct = 0.05
    if direction == "CALL":
        strike = round((1+otm_pct) * last_price, 2)
        side = "C"
    else:
        strike = round((1-otm_pct) * last_price, 2)
        side = "P"
    expiry = next_friday()
    return {"symbol": symbol, "side": side, "strike": strike, "expiry": expiry}

# ─────────────────────────────────────────────────────────────
# Discord
# ─────────────────────────────────────────────────────────────
def post_discord(embed):
    if not DISCORD_WEBHOOK_URL:
        return
    payload = {"embeds": [embed]}
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print("Discord error:", e)

def make_embed(signal):
    color = 0x15C39A if signal["bias"] == "BULL" else 0xE03A3A
    fields = [
        {"name":"Price", "value": f"${signal['last_price']:.2f}", "inline": True},
        {"name":"RVOL", "value": f"{signal['rvol']:.2f}x", "inline": True},
        {"name":"Score", "value": f"{signal['score']}/5", "inline": True},
        {"name":"Pattern", "value": ", ".join(signal['patterns']) or "—", "inline": True},
        {"name":"Stop", "value": f"${signal['stop']:.2f}", "inline": True},
        {"name":"Target", "value": f"${signal['target']:.2f}", "inline": True},
    ]
    if signal.get("option"):
        o = signal["option"]
        fields.append({"name":"Option", "value": f"{signal['ticker']} {o['expiry']} {o['side']}{o['strike']}", "inline": False})

    return {
        "title": f"🚀 {signal['ticker']} — {signal['bias']} Signal",
        "description": "BeastMode Squeeze Scanner",
        "color": color,
        "footer": {"text": "Not financial advice — educational signals"},
        "timestamp": dt.datetime.now(timezone.utc).isoformat()
    } | {"fields": fields}

# ─────────────────────────────────────────────────────────────
# Core scan
# ─────────────────────────────────────────────────────────────
def scan_one(symbol):
    try:
        df = alpha_intraday(symbol, interval="5min", output="compact")
        if len(df) < 60: return None
        rvol = calc_rvol(df["volume"], RVOL_LOOKBACK)
        squeeze_on, s_bull, s_bear, bb_w, kc_w = is_ttm_squeeze(df)
        macd, sig, hist = macd_signal(df["close"])

        last_price = df["close"].iloc[-1]
        rvol_now = float(rvol.iloc[-1]) if not np.isnan(rvol.iloc[-1]) else 0.0
        bull_trig = bool(s_bull.iloc[-1])
        bear_trig = bool(s_bear.iloc[-1])
        hist_now = float(hist.iloc[-1])

        if rvol_now < RVOL_MIN and not bull_trig and not bear_trig:
            return None # ignore weak noise

        bias = "BULL" if (hist_now > 0 or bull_trig) else "BEAR"
        sc = score_signal(rvol_now, hist_now, bull_trig, bear_trig)
        pats = patterns(df)

        # Risk box (ATR-ish using KC width)
        atr_proxy = float((kc_w.iloc[-1]) / 2.0) if not np.isnan(kc_w.iloc[-1]) else max(0.02*last_price, 0.1)
        stop = last_price - atr_proxy if bias=="BULL" else last_price + atr_proxy
        target = last_price + 2*atr_proxy if bias=="BULL" else last_price - 2*atr_proxy

        # Suggested option (heuristic now; swap to Schwab chain when ready)
        opt = suggested_option(symbol, last_price, "CALL" if bias=="BULL" else "PUT")

        signal = {
            "ticker": symbol,
            "last_price": last_price,
            "rvol": rvol_now,
            "score": sc,
            "bias": bias,
            "patterns": pats,
            "stop": stop,
            "target": target,
            "option": opt
        }
        return signal
    except Exception as e:
        print(f"[{symbol}] scan error:", e)
        return None

def scan_loop():
    print("BeastMode loop started.")
    # Initial heartbeat to Discord
    post_discord({
        "title": "🟢 BeastMode Alpha Bot is ONLINE",
        "description": f"Tracking {len(UNIVERSE)} tickers now…",
        "color": 0x2ECC71
    })
    i = 0
    while True:
        start = time.time()
        # Respect Alpha Vantage rate limits: ~5 req/min free. We stagger symbols.
        # Batch 5 per minute.
        batch = UNIVERSE[(i*5)%len(UNIVERSE): ((i*5)%len(UNIVERSE))+5]
        if not batch:
            batch = UNIVERSE[:5]
        for sym in batch:
            sig = scan_one(sym)
            if sig:
                embed = make_embed(sig)
                post_discord(embed)
            time.sleep(1.0) # short spacing between calls

        i += 1
        # sleep to complete ~60s cadence
        elapsed = time.time() - start
        time.sleep(max(5, SCAN_INTERVAL_SEC - elapsed))

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run web heartbeat + scanner loop in parallel
    threading.Thread(target=run_server, daemon=True).start()
    scan_loop()
