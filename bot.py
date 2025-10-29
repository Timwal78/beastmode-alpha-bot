import os, time, math, json, random, threading, http.server, socketserver
from datetime import datetime, timezone
from collections import deque

import requests
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Env & Settings
# ─────────────────────────────────────────────────────────────
# ALPHA_KEYS supports one or many keys (comma separated). Fallback to ALPHA_KEY.
_keys_raw = os.getenv("ALPHA_KEYS") or os.getenv("ALPHA_KEY") or ""
API_KEYS = [k.strip() for k in _keys_raw.split(",") if k.strip()]
if not API_KEYS:
    raise SystemExit("Missing ALPHA_KEY(S). Set ALPHA_KEY or ALPHA_KEYS in Render > Environment.")

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
if not DISCORD_WEBHOOK:
    raise SystemExit("Missing DISCORD_WEBHOOK env var.")

TICKERS = [t.strip().upper() for t in (os.getenv("TICKERS") or "AMC,GME").split(",") if t.strip()]

SCAN_INTERVAL_SEC = int(os.getenv("SCAN_INTERVAL_SEC", "15")) # per-symbol delay
HEARTBEAT_MIN = int(os.getenv("HEARTBEAT_MIN", "60")) # heartbeat cadence

# Alpha Vantage limits (free ≈ 5 calls/min). We’ll process ~1 symbol / 12s by default.
ALPHA_BASE = "https://www.alphavantage.co/query"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "beastmode-alpha-bot/1.0"})

# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def pick_key() -> str:
    return random.choice(API_KEYS)

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def post_discord(content=None, embed=None):
    payload = {}
    if content:
        payload["content"] = content
    if embed:
        payload["embeds"] = [embed]
    try:
        r = SESSION.post(DISCORD_WEBHOOK, json=payload, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[{ts()}] Discord error: {e}")

def pct(a, b):
    try:
        return 100.0 * (a - b) / b if b else np.nan
    except Exception:
        return np.nan

def bb_stats(close_series, length=20, mult=2.0):
    if len(close_series) < length:
        return np.nan, np.nan, np.nan
    s = pd.Series(close_series)
    ma = s.rolling(length).mean()
    sd = s.rolling(length).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    width = (upper - lower) / ma # relative width
    return float(ma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1]), float(width.iloc[-1])

def percent_rank(series, value):
    arr = np.array(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan
    rank = (arr < value).sum() / len(arr)
    return float(rank)

# ─────────────────────────────────────────────────────────────
# Alpha Vantage fetchers with backoff
# ─────────────────────────────────────────────────────────────
def av_get(params, max_retries=3, backoff=8):
    params = {**params, "apikey": pick_key()}
    for i in range(max_retries):
        try:
            r = SESSION.get(ALPHA_BASE, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                # API limit note handling
                if "Note" in data or "Information" in data:
                    wait = backoff * (i + 1)
                    print(f"[{ts()}] Alpha note/info: throttled. Sleeping {wait}s")
                    time.sleep(wait)
                    continue
                return data
            else:
                print(f"[{ts()}] HTTP {r.status_code} from AlphaVantage; retry {i+1}")
        except Exception as e:
            print(f"[{ts()}] av_get error: {e}; retry {i+1}")
        time.sleep(backoff * (i + 1))
    return None

def get_daily(symbol, outputsize="compact"):
    data = av_get({"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize})
    if not data or "Time Series (Daily)" not in data:
        return None
    tsd = data["Time Series (Daily)"]
    rows = []
    for k, v in tsd.items():
        rows.append({
            "date": k,
            "open": float(v["1. open"]),
            "high": float(v["2. high"]),
            "low": float(v["3. low"]),
            "close": float(v["4. close"]),
            "volume": float(v["5. volume"])
        })
    df = pd.DataFrame(rows).sort_values("date")
    return df

# ─────────────────────────────────────────────────────────────
# Signal Logic (Phase 3: RVOL + BB squeeze + Trend filter)
# ─────────────────────────────────────────────────────────────
def analyze_symbol(sym: str):
    df = get_daily(sym, outputsize="compact") # last ~100 bars
    if df is None or len(df) < 40:
        return None

    close = df["close"].values
    vol = df["volume"].values

    sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    sma5 = pd.Series(close).rolling(5).mean().iloc[-1]
    last_close = close[-1]
    last_vol = vol[-1]
    avg_vol20 = pd.Series(vol).rolling(20).mean().iloc[-1]
    rvol = (last_vol / avg_vol20) if avg_vol20 and not math.isnan(avg_vol20) else np.nan

    _, _, _, bb_width_now = bb_stats(close, length=20, mult=2.0)
    # Percentile of current BB width vs last 60 days
    widths = []
    s = pd.Series(close)
    for i in range(20, min(len(s), 100)):
        window = s.iloc[:i]
        ma = window.rolling(20).mean()
        sd = window.rolling(20).std(ddof=0)
        if pd.notna(ma.iloc[-1]) and pd.notna(sd.iloc[-1]):
            w = (ma.iloc[-1] + 2*sd.iloc[-1] - (ma.iloc[-1] - 2*sd.iloc[-1])) / ma.iloc[-1]
            widths.append(w)
    pr_width = percent_rank(widths[-60:], bb_width_now) if len(widths) >= 5 else np.nan

    trend_ok = last_close > sma20 and sma5 > sma20
    squeeze_forming = (not math.isnan(pr_width)) and pr_width <= 0.20 # bottom 20% width
    momentum_ok = pct(last_close, df["close"].iloc[-2]) >= 1.5 # ≥ +1.5% vs yesterday
    rvol_ok = (not math.isnan(rvol)) and rvol >= 1.5

    score = sum([bool(squeeze_forming), bool(rvol_ok), bool(trend_ok and momentum_ok)])
    return {
        "symbol": sym,
        "last_close": round(last_close, 4),
        "rvol": round(float(rvol), 2) if not math.isnan(rvol) else None,
        "pr_width": round(float(pr_width), 2) if not math.isnan(pr_width) else None,
        "squeeze": bool(squeeze_forming),
        "trend_ok": bool(trend_ok),
        "momentum_ok": bool(momentum_ok),
        "score": int(score),
    }

def embed_from_signal(sig):
    color = 0x2ecc71 if sig["score"] >= 2 else 0xf1c40f if sig["score"] == 1 else 0xe74c3c
    lines = []
    lines.append(f"**Price:** ${sig['last_close']}")
    if sig["rvol"] is not None:
        lines.append(f"**RVOL (20d):** {sig['rvol']}x")
    if sig["pr_width"] is not None:
        lines.append(f"**BB Width %Rank (60d):** {int(sig['pr_width']*100)}th pct")
    checks = []
    checks.append("✅ Squeeze forming" if sig["squeeze"] else "▫️ Squeeze")
    checks.append("✅ Trend+Mom" if sig["trend_ok"] and sig["momentum_ok"] else "▫️ Trend+Mom")
    checks.append("✅ RVOL" if sig["rvol"] and sig["rvol"] >= 1.5 else "▫️ RVOL")
    lines.append(" / ".join(checks))

    return {
        "title": f"🚀 {sig['symbol']} — Momentum Building ({sig['score']}/3)",
        "description": "\n".join(lines),
        "timestamp": datetime.utcnow().isoformat(),
        "color": color,
        "footer": {"text": "Beastmode Alpha Scanner"},
    }

# ─────────────────────────────────────────────────────────────
# Scanner Loop
# ─────────────────────────────────────────────────────────────
def scanner_loop():
    post_discord(content="🟢 **Beastmode Alpha Bot is ONLINE** — tracking tickers now…")
    q = deque(TICKERS)
    last_heartbeat = time.time()

    while True:
        # Heartbeat
        if time.time() - last_heartbeat >= HEARTBEAT_MIN * 60:
            post_discord(content=f"⏱️ Heartbeat {ts()} — monitoring {len(TICKERS)} tickers.")
            last_heartbeat = time.time()

        if not q:
            q = deque(TICKERS)

        sym = q.popleft()
        try:
            sig = analyze_symbol(sym)
            if sig and sig["score"] >= 2:
                post_discord(embed=embed_from_signal(sig))
            else:
                # Quiet pass; uncomment to see every scan:
                # post_discord(content=f"ℹ️ {sym} scanned — no strong setup.")
                pass
        except Exception as e:
            print(f"[{ts()}] Scan error {sym}: {e}")

        time.sleep(SCAN_INTERVAL_SEC)

# ─────────────────────────────────────────────────────────────
# Minimal HTTP server (keeps free Render web dyno alive)
# ─────────────────────────────────────────────────────────────
def start_http_server():
    port = int(os.getenv("PORT", "10000"))
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args): # quieter logs
            return
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"[{ts()}] http server running on {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    # Thread 1: tiny HTTP server
    t1 = threading.Thread(target=start_http_server, daemon=True)
    t1.start()

    # Thread 2: scanner
    scanner_loop()
