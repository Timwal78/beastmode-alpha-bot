import os
import time
import requests
import pandas as pd
from discord_webhook import DiscordWebhook

# --- ENVIRONMENT VARIABLES ---
ALPHA_KEY = os.getenv("ALPHA_KEY")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
TICKERS = os.getenv("TICKERS", "AMC,GME,IONQ,NAK,SMR,FFIE,BNZI").split(",")

# --- CONSTANTS ---
ALPHA_URL = "https://www.alphavantage.co/query"
REFRESH_MINUTES = 5 # how often to update (every 5 min)

# --- DISCORD ALERT FUNCTION ---
def send_discord(message):
    try:
        webhook = DiscordWebhook(url=DISCORD_WEBHOOK, content=message)
        response = webhook.execute()
        if response.status_code in (200, 204):
            print(f"✅ Sent to Discord: {message}")
        else:
            print(f"⚠️ Discord error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Discord send failed: {e}")

# --- FETCH STOCK QUOTE FUNCTION ---
def get_quote(symbol):
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.strip(),
            "apikey": ALPHA_KEY
        }
        r = requests.get(ALPHA_URL, params=params)
        data = r.json()
        price = float(data["Global Quote"]["05. price"])
        change_percent = float(data["Global Quote"]["10. change percent"].replace("%",""))
        return price, change_percent
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
        return None, None

# --- STARTUP ---
send_discord("🚀 **Beastmode Alpha Bot is ONLINE!** Tracking tickers now...")

# --- MAIN LOOP ---
while True:
    for symbol in TICKERS:
        price, change_percent = get_quote(symbol)
        if price is not None:
            msg = f"📈 {symbol}: ${price:.2f} ({change_percent:.2f}%)"
            print(msg)

            # Simple squeeze condition (change > 10%)
            if change_percent >= 10:
                alert = f"🔥 **SQUEEZE ALERT!** {symbol} is up {change_percent:.2f}% at ${price:.2f}"
                send_discord(alert)

    time.sleep(REFRESH_MINUTES * 60)
