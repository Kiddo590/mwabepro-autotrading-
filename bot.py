import asyncio
import json
import time
import math
import random
import logging
import requests
import os
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import websockets
from dotenv import load_dotenv
from news import NewsGate  # your news.py for sentiment filter

# === load env ===
load_dotenv()

# === USER SETTINGS ===
APP_ID = os.getenv("APP_ID", "")
DERIV_TOKEN = os.getenv("DERIV_TOKEN", "")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"

SYMBOL = os.getenv("SYMBOL", "R_100")
CONTRACT_TYPE = os.getenv("CONTRACT_TYPE", "CALL")
DURATION = int(os.getenv("DURATION", "60"))
TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "50.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500.0"))
MAX_HOURLY_LOSS = float(os.getenv("MAX_HOURLY_LOSS", "500.0"))

NEWS_THRESHOLD = float(os.getenv("NEWS_THRESHOLD", "0.0"))
NEWS_CHECK_INTERVAL = int(os.getenv("NEWS_CHECK_INTERVAL", "60"))
TICKS_HISTORY = int(os.getenv("TICKS_HISTORY", "100"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DERIV_WS = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
HEARTBEAT_INTERVAL = 15
RECONNECT_DELAY = 5

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("deriv-bot")

# Global variables
ticks = deque(maxlen=TICKS_HISTORY)
daily_pnl = 0.0
hourly_pnl = 0.0
last_news_check = datetime.min
last_hour_reset = datetime.now()
account_balance = 10000.0
trade_history = []
win_rate = 0.5

# pull your NewsAPI key from env
news_gate = NewsGate(api_key=os.getenv("NEWS_API_KEY", ""))

# === Simplified ML Predictor ===
class SimpleMLPredictor:
    def __init__(self):
        self.prediction_history = []
        self.is_trained = False
        
    def extract_features(self, window):
        """Extract basic features for prediction"""
        if len(window) < 10:
            return None
            
        features = {
            'short_ma': np.mean(window[-5:]),
            'long_ma': np.mean(window[-10:]),
            'volatility': np.std(window[-10:]),
            'rsi': self.compute_rsi(window),
            'momentum': window[-1] - window[-5]
        }
        return features
    
    def compute_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
            
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
        
        if len(gains) < period or len(losses) < period:
            return 50
            
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def predict(self, features):
        """Simple rule-based prediction"""
        if features is None:
            return None, 0.5
            
        # Simple rule-based prediction
        if (features['short_ma'] > features['long_ma'] and 
            30 < features['rsi'] < 70):
            return "CALL", 0.6
        elif (features['short_ma'] < features['long_ma'] and 
              30 < features['rsi'] < 70):
            return "PUT", 0.6
        else:
            return None, 0.3

# === Risk Management ===
class RiskManager:
    def __init__(self):
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trade_count_hour = 0
        
    def should_trade(self, signal_strength, volatility):
        global hourly_pnl, last_hour_reset
        
        # Reset hourly counters if hour changed
        current_time = datetime.now()
        if current_time.hour != last_hour_reset.hour:
            hourly_pnl = 0.0
            last_hour_reset = current_time
            self.trade_count_hour = 0
        
        # Basic risk checks
        if self.consecutive_losses >= 3:
            logger.warning("3 consecutive losses - trading paused")
            return False
            
        if hourly_pnl <= -abs(MAX_HOURLY_LOSS):
            logger.warning("Hourly loss limit reached")
            return False
            
        if self.trade_count_hour >= 10:  # Max 10 trades per hour
            logger.warning("Max trades per hour reached")
            return False
            
        # Volatility-based adjustments
        if volatility > 0.05:
            return signal_strength > 0.7
            
        return True
    
    def update_trade_result(self, successful, trade_size):
        if successful:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        self.trade_count_hour += 1

# Initialize components
ml_predictor = SimpleMLPredictor()
risk_manager = RiskManager()

# === helpers ===
def send_telegram(msg: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
        )
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)

def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
        
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) < period or len(losses) < period:
        return 50
        
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_signal(window):
    if len(window) < 10:
        return None
    
    # Multiple indicators
    short_ma = np.mean(window[-5:])
    long_ma = np.mean(window[-10:])
    rsi = compute_rsi(window)
    
    # Trend detection
    trend_up = all(window[i] > window[i-1] for i in range(-3, 0))
    trend_down = all(window[i] < window[i-1] for i in range(-3, 0))
    
    # Multiple confirmation signals
    if (short_ma > long_ma and 30 < rsi < 70 and trend_up):
        return "CALL"
    elif (short_ma < long_ma and 30 < rsi < 70 and trend_down):
        return "PUT"
    return None

def estimate_volatility(window):
    if len(window) < 3:
        return 0.0
    rets = [(window[i] - window[i - 1]) / window[i - 1] for i in range(1, len(window))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)

def dynamic_position_sizing(volatility, account_balance, win_rate):
    """Simplified position sizing"""
    if win_rate == 0 or volatility == 0:
        return TRADE_SIZE_USD
    
    # Volatility adjustment
    vol_adjustment = min(1.0, 0.1 / max(volatility, 0.001))
    
    # Win rate adjustment
    win_rate_adjustment = max(0.5, min(1.5, win_rate * 2))
    
    position_size = TRADE_SIZE_USD * vol_adjustment * win_rate_adjustment
    
    return max(10.0, min(position_size, account_balance * 0.05))

def detect_market_regime(window):
    if len(window) < 20:
        return "normal"
        
    returns = np.diff(window)
    volatility = np.std(returns)
    mean_return = np.mean(returns)
    
    if volatility > 0.02:
        return "high_volatility"
    elif abs(mean_return) < 0.001:
        return "ranging"
    else:
        return "trending"

def adjust_strategy_for_regime(regime, signal, confidence):
    if regime == "high_volatility":
        return signal, confidence * 0.7
    elif regime == "ranging":
        return signal, confidence * 0.8
    else:
        return signal, confidence

def combine_signals(tech_signal, ml_signal, ml_confidence, regime):
    if tech_signal == ml_signal and ml_confidence > 0.6:
        return tech_signal, ml_confidence
    elif ml_confidence > 0.7:
        return ml_signal, ml_confidence
    elif tech_signal and regime == "trending":
        return tech_signal, 0.6
    return None, 0.0

def risk_allows_trade(size_usd):
    global daily_pnl, hourly_pnl, last_hour_reset
    
    # Reset hourly PNL if hour changed
    current_hour = datetime.now().hour
    if current_hour != last_hour_reset.hour:
        hourly_pnl = 0.0
        last_hour_reset = datetime.now()
    
    if daily_pnl <= -abs(MAX_DAILY_LOSS):
        logger.warning("Daily loss cap breached.")
        return False
        
    if hourly_pnl <= -abs(MAX_HOURLY_LOSS):
        logger.warning("Hourly loss cap breached.")
        return False
        
    return True

def seasonal_adjustment():
    """Adjust strategy based on time of day"""
    hour = datetime.now().hour
    if hour in [0, 1, 2, 12, 13]:
        return 0.7
    return 1.0

async def handle_tick_message(msg):
    if "tick" not in msg:
        return
    t = msg["tick"]
    price = float(t.get("quote") or t.get("price"))
    ticks.append(price)
    logger.debug("tick %.5f len=%d", price, len(ticks))

# === core trading ===
async def place_trade(ws, symbol, trade_type, size_usd):
    global daily_pnl, hourly_pnl, win_rate, trade_history
    
    # Apply seasonal adjustment
    size_usd *= seasonal_adjustment()
    amount = round(size_usd, 2)

    if PAPER_MODE:
        entry = ticks[-1] if ticks else 100
        result_win = random.random() < 0.52
        pnl = (0.005 if result_win else -0.01) * amount
        daily_pnl += pnl
        hourly_pnl += pnl
        
        # Update win rate
        trade_history.append(1 if result_win else 0)
        if len(trade_history) > 10:
            win_rate = sum(trade_history[-10:]) / 10
        
        logger.info("PAPER TRADE %s %.2f pnl=%.4f daily=%.2f hourly=%.2f",
                    trade_type, amount, pnl, daily_pnl, hourly_pnl)
        send_telegram(f"PAPER TRADE {trade_type} {amount}USD pnl={pnl:.2f}")
        
        # Update risk manager
        risk_manager.update_trade_result(result_win, amount)
        
        return result_win

    logger.info("LIVE trade requested: %s amount=%.2f", trade_type, amount)
    # Live trading implementation would go here
    return True

async def run_bot():
    global last_news_check, last_hour_reset, win_rate, hourly_pnl, daily_pnl
    
    logger.info("Starting bot SYMBOL=%s PAPER=%s", SYMBOL, PAPER_MODE)
    logger.info("Trade size: $%.2f, Daily loss limit: $%.2f, Hourly loss limit: $%.2f", 
                TRADE_SIZE_USD, MAX_DAILY_LOSS, MAX_HOURLY_LOSS)
    
    while True:
        try:
            async with websockets.connect(DERIV_WS, ping_interval=None) as ws:
                if not PAPER_MODE:
                    auth_req = {"authorize": DERIV_TOKEN}
                    await ws.send(json.dumps(auth_req))
                    auth_resp = json.loads(await ws.recv())
                    if "error" in auth_resp:
                        logger.error("Authorization failed: %s", auth_resp["error"])
                        send_telegram(f"Auth failed: {auth_resp['error']}")
                        return
                    logger.info("Authorized loginid=%s",
                                auth_resp.get("authorize", {}).get("loginid"))

                await ws.send(json.dumps({"ticks": SYMBOL}))
                logger.info("Subscribed to ticks for %s", SYMBOL)
                last_ping = time.time()
                news_ok = True

                while True:
                    msg_task = asyncio.create_task(ws.recv())
                    done, pending = await asyncio.wait([msg_task], timeout=HEARTBEAT_INTERVAL)
                    if msg_task in done:
                        raw = msg_task.result()
                        msg = json.loads(raw)
                        if "tick" in msg:
                            await handle_tick_message(msg)
                        else:
                            logger.debug("WS msg keys=%s", list(msg.keys()))
                    else:
                        for p in pending:
                            p.cancel()

                    current_time = datetime.now()
                    
                    # Reset hourly PNL if hour changed
                    if current_time.hour != last_hour_reset.hour:
                        hourly_pnl = 0.0
                        last_hour_reset = current_time
                        risk_manager.trade_count_hour = 0
                        logger.info("Hourly PNL reset")

                    # News sentiment check
                    if (datetime.utcnow() - last_news_check).total_seconds() > NEWS_CHECK_INTERVAL:
                        last_news_check = datetime.utcnow()
                        sentiment = news_gate.get_sentiment()
                        logger.info("News sentiment %.3f", sentiment)
                        if sentiment < NEWS_THRESHOLD:
                            logger.warning("News blocks trading (%.3f<%.3f)",
                                           sentiment, NEWS_THRESHOLD)
                            send_telegram(f"NEWS BLOCK: sentiment={sentiment:.3f}")
                            news_ok = False
                        else:
                            news_ok = True

                    if len(ticks) >= 20:
                        current_window = list(ticks)
                        regime = detect_market_regime(current_window)
                        volatility = estimate_volatility(current_window)
                        
                        # Get technical signal
                        technical_signal = compute_signal(current_window)
                        
                        # Get ML prediction
                        ml_features = ml_predictor.extract_features(current_window)
                        ml_prediction, ml_confidence = ml_predictor.predict(ml_features)
                        
                        # Combine signals
                        final_signal, signal_confidence = combine_signals(
                            technical_signal, ml_prediction, ml_confidence, regime
                        )
                        final_signal, signal_confidence = adjust_strategy_for_regime(
                            regime, final_signal, signal_confidence
                        )
                        
                        # Calculate dynamic position size
                        size = dynamic_position_sizing(volatility, account_balance, win_rate)
                        
                        logger.debug("Signal=%s Confidence=%.2f Regime=%s Vol=%.6f Size=%.2f",
                                     final_signal, signal_confidence, regime, volatility, size)
                        
                        if (final_signal and news_ok and signal_confidence > 0.5 and 
                            risk_allows_trade(size) and risk_manager.should_trade(signal_confidence, volatility)):
                            
                            trade_success = await place_trade(ws, SYMBOL, final_signal, size)
                            await asyncio.sleep(2)

                    # Check loss limits
                    if daily_pnl <= -abs(MAX_DAILY_LOSS):
                        logger.critical("Daily loss cap hit, sleeping.")
                        send_telegram(f"Daily loss cap hit: {daily_pnl:.2f}.")
                        await asyncio.sleep(60 * 60 * 6)
                        
                    if hourly_pnl <= -abs(MAX_HOURLY_LOSS):
                        logger.critical("Hourly loss cap hit, sleeping.")
                        send_telegram(f"Hourly loss cap hit: {hourly_pnl:.2f}.")
                        await asyncio.sleep(60 * 30)

                    if time.time() - last_ping > HEARTBEAT_INTERVAL:
                        try:
                            await ws.ping()
                            last_ping = time.time()
                        except Exception:
                            logger.debug("Ping failed; reconnecting.")
                            break
        except Exception as e:
            logger.exception("Connection error: %s", e)
            logger.info("Reconnecting in %s sec...", RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")