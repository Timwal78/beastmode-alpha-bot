# ==============================================
# BEASTMODE ALPHA RADAR – Discord Auto Alerts
# Author: Timwal78 | Version: 1.0 (Stable)
# ==============================================

import os, time, requests, pandas as pd
from discord_webhook import DiscordWebhook
import schedule

# -------------------------------
# 🔐 Load environment variables
# -------------------------------
ALPHA_KEY = os.getenv("ALPHA_KEY")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
TICKERS = os.getenv("TICKERS", "AMC,GME,IONQ,SMR,NAK,BNZI,FFIE").split(",")

ALPHA_URL = "https://www.alphavantage.co/query"

# -------------------------------
# ⚙️ Function: get quote
# -------------------------------
def get_quote(symbol):
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.strip(),
        "apikey": ALPHA_KEY
    }
    r = requests.get(ALPHA_URL, params=params)
    data = r.json().get("Global Quote", {})
    if not data:
        return None
    return {
        "symbol": data.get("01. symbol"),
        "price": float(data.get("05. price", 0.0)),
        "change": float(data.get("09. change", 0.0)),
        "percent": data.get("10. change percent", "0%")
    }

# -------------------------------
# 💬 Function: post to Discord
# -------------------------------
def send_to_discord(msg):
    try:
        webhook = DiscordWebhook(url=DISCORD_WEBHOOK, content=msg)
        webhook.execute()
    except Exception as e:
        print(f"[Error] Discord send failed: {e}")

# -------------------------------
# 🧠 Function: scan all tickers
# -------------------------------
def scan_tickers():
    print("🔄 Running Alpha scan...")
    results = []
    for t in TICKERS:
        quote = get_quote(t)
        if not quote:
            continue
        change = quote["change"]
        pct = float(quote["percent"].replace("%",""))
        if abs(pct) >= 5: # simple squeeze momentum threshold
            direction = "🚀 Bullish" if change > 0 else "💀 Bearish"
            msg = (f"**{direction} Alert:** `{t}`\n"
                   f"Price: ${quote['price']:.2f}\n"
                   f"Change: {pct:.2f}%")
            send_to_discord(msg)
            results.append(t)
    if results:
        print(f"✅ Alerts sent for: {', '.join(results)}")
    else:
        print("⚪ No strong movers yet.")

# -------------------------------
# ⏰ Schedule every 5 minutes
# -------------------------------
schedule.every(5).minutes.do(scan_tickers)

# -------------------------------
# 🌀 Run forever
# -------------------------------
print("🔥 Beastmode Alpha Radar running...")
while True:
    schedule.run_pending()
    time.sleep(30)
