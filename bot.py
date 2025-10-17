import asyncio
import json
import time
import math
import random
import logging
import requests
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import websockets
from dotenv import load_dotenv

# Import the enhanced ML components
from enhanced_ml import EnhancedMLPredictor
from ml_reporting import MLReporter

# === load env ===
load_dotenv()

# === USER SETTINGS ===
APP_ID = os.getenv("APP_ID", "65770")
DERIV_TOKEN = os.getenv("DERIV_TOKEN", "")
PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"

SYMBOL = os.getenv("SYMBOL", "R_100")
DURATION = int(os.getenv("DURATION", "60"))
TRADE_SIZE_USD = float(os.getenv("TRADE_SIZE_USD", "50.0"))
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "500.0"))
MAX_HOURLY_LOSS = float(os.getenv("MAX_HOURLY_LOSS", "500.0"))

# FIXED: Lower confidence threshold and volatility threshold for synthetic markets
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))  # Reduced from 0.3
VOLATILITY_THRESHOLD = float(os.getenv("VOLATILITY_THRESHOLD", "0.00005"))  # Much lower for synthetics

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
last_hour_reset = datetime.now()
account_balance = 10000.0
trade_history = []
win_rate = 0.5
trade_outcomes = []  # Track trade outcomes for ML learning

# === Risk Management ===
class EnhancedRiskManager:
    def __init__(self):
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.trade_count_hour = 0
        self.ml_confidence_history = []
        self.trade_results = []
        
    def should_trade(self, signal_strength, volatility, ml_confidence=0.5):
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
            
        if self.trade_count_hour >= 20:  # Increased from 10 to allow more trades
            logger.warning("Max trades per hour reached")
            return False
            
        # FIXED: Less restrictive ML confidence checks
        if len(self.ml_confidence_history) > 10:
            avg_confidence = np.mean(self.ml_confidence_history[-10:])
            if avg_confidence < 0.3:  # Reduced from 0.4
                return signal_strength > 0.5  # Reduced from 0.7
        
        # Recent performance-based adjustments
        if len(self.trade_results) > 5:
            recent_wins = sum(self.trade_results[-5:])
            if recent_wins <= 1:  # Poor recent performance
                return signal_strength > 0.4  # Reduced from 0.6
                
        # FIXED: Allow trading in all volatility conditions with adjusted thresholds
        if volatility > VOLATILITY_THRESHOLD * 2:  # More permissive
            return signal_strength > 0.3  # Reduced threshold
            
        return True
    
    def update_trade_result(self, successful, trade_size, ml_confidence):
        if successful:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        self.trade_count_hour += 1
        self.ml_confidence_history.append(ml_confidence)
        self.trade_results.append(1 if successful else 0)
        
        # Keep history manageable
        if len(self.ml_confidence_history) > 100:
            self.ml_confidence_history.pop(0)
        if len(self.trade_results) > 100:
            self.trade_results.pop(0)

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

# Enhanced ML initialization
ml_predictor = EnhancedMLPredictor()
# Try to load existing models on startup
try:
    ml_predictor.load_models()
    logger.info("Loaded pre-trained ML models")
except:
    logger.info("No pre-trained models found, starting fresh")

risk_manager = EnhancedRiskManager()
ml_reporter = MLReporter(ml_predictor, send_telegram)

def is_synthetic_symbol(symbol):
    """Check if the symbol is a synthetic instrument"""
    synthetic_patterns = ["R_", "1HZ", "BOOM", "CRASH"]
    return any(pattern in symbol for pattern in synthetic_patterns)

def synthetic_market_adjustment(signal_strength, volatility, symbol):
    """Adjust strategy for synthetic markets"""
    if not is_synthetic_symbol(symbol):
        return signal_strength
    
    # FIXED: More aggressive adjustments for synthetic markets
    # Synthetic markets need different signal strength handling
    if volatility < 0.0001:  # Even lower threshold for synthetics
        return signal_strength * 1.3  # Increase confidence more in low vol
    
    return signal_strength * 1.1  # Always slightly increase for synthetics

def compute_synthetic_signal(window):
    """Special signal detection for synthetic markets like R_100"""
    if len(window) < 5:  # Reduced from 10 for faster signals
        return None, 0.0
    
    prices = np.array(window)
    
    # FIXED: Simplified momentum detection for faster signals
    recent_prices = prices[-3:]  # Use only last 3 prices
    earlier_prices = prices[-6:-3] if len(prices) >= 6 else prices[:3]
    
    if len(recent_prices) < 3 or len(earlier_prices) < 3:
        return None, 0.0
    
    recent_avg = np.mean(recent_prices)
    earlier_avg = np.mean(earlier_prices)
    
    # Calculate momentum
    momentum = (recent_avg - earlier_avg) / earlier_avg if earlier_avg != 0 else 0
    
    # FIXED: Lower thresholds for synthetic signal generation
    volatility = np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 else 0
    
    # Generate signals with lower thresholds
    if momentum > 0.0005 and volatility > 0.00005:  # Much lower thresholds
        return "CALL", 0.4 + min(0.3, abs(momentum) * 100)
    elif momentum < -0.0005 and volatility > 0.00005:
        return "PUT", 0.4 + min(0.3, abs(momentum) * 100)
    
    # Weak momentum signals
    if momentum > 0.0002:
        return "CALL", 0.25
    elif momentum < -0.0002:
        return "PUT", 0.25
    
    return None, 0.0

def compute_rsi(prices, period=14):
    """Calculate RSI manually without TA-Lib"""
    if len(prices) < period + 1:
        return 50
    
    # Manual RSI calculation
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

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average manually"""
    if len(prices) < period:
        return np.mean(prices) if prices else prices[-1] if prices else 0
    
    # EMA calculation: weight = 2 / (period + 1)
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    
    # For the first few values where we don't have enough data, use SMA
    if len(prices) < period * 2:
        return np.convolve(prices, weights, mode='valid')[-1]
    else:
        return np.convolve(prices[-period:], weights, mode='valid')[0]

def compute_signal(window):
    """Improved signal generation with multiple technical indicators"""
    if len(window) < 5:  # Reduced from 10 for much faster signal generation
        return None, 0.0
    
    # FIXED: Always use synthetic-specific logic for R_100
    if is_synthetic_symbol(SYMBOL):
        return compute_synthetic_signal(window)
    
    prices = np.array(window)
    
    # Use EMA crossovers (manual calculation)
    ema_short = calculate_ema(prices, 5)  # Reduced periods
    ema_long = calculate_ema(prices, 12)  # Reduced periods
    
    # RSI with better handling
    rsi = compute_rsi(prices, 10)  # Reduced period
    
    # Simple momentum calculation
    momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
    
    # Generate signals with confidence scores
    signal_strength = 0
    signal = None
    
    # FIXED: More permissive signal conditions
    if ema_short > ema_long and rsi > 35:  # Reduced from 40
        signal_strength += 0.25  # Reduced from 0.3
        if momentum > 0:
            signal_strength += 0.15  # Reduced from 0.2
        signal = "CALL"
    elif ema_short < ema_long and rsi < 65:  # Increased from 60
        signal_strength += 0.25
        if momentum < 0:
            signal_strength += 0.15
        signal = "PUT"
    
    # Additional momentum-based signals
    if abs(momentum) > 0.001:  # Lower threshold
        if momentum > 0 and signal != "PUT":  # Don't contradict existing signal
            signal = "CALL"
            signal_strength = max(signal_strength, 0.3)
        elif momentum < 0 and signal != "CALL":
            signal = "PUT" 
            signal_strength = max(signal_strength, 0.3)
    
    # Apply synthetic market adjustment
    signal_strength = synthetic_market_adjustment(signal_strength, estimate_volatility(window), SYMBOL)
    
    return signal, min(signal_strength, 1.0)

def estimate_volatility(window):
    if len(window) < 3:
        return 0.0
    rets = [(window[i] - window[i - 1]) / window[i - 1] for i in range(1, len(window))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return math.sqrt(var)

def calculate_hurst_exponent(prices, max_lag=10):  # Reduced max_lag
    """Calculate Hurst exponent to detect market regime"""
    if len(prices) < max_lag * 2:
        return 0.5  # Neutral value when not enough data
    
    lags = range(2, min(max_lag, len(prices)//2))
    if len(lags) < 2:
        return 0.5
        
    # Calculate the variance of the differences
    tau = []
    for lag in lags:
        if lag >= len(prices):
            continue
        differences = np.subtract(prices[lag:], prices[:-lag])
        if len(differences) > 0:
            tau.append(np.std(differences))
    
    if len(tau) < 2:
        return 0.5
        
    # Fit to a power law and return the Hurst exponent
    try:
        poly = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)
        return poly[0] * 2.0
    except:
        return 0.5

def detect_market_regime(window):
    """Improved market regime detection"""
    if len(window) < 10:  # Reduced from 20 for faster detection
        return "normal"
    
    prices = np.array(window)
    
    # FIXED: Use simpler volatility calculation for synthetics
    volatility = estimate_volatility(window)
    
    # Adjusted thresholds for synthetic markets
    if volatility > 0.0003:  # Lower threshold for high vol
        return "high_volatility"
    elif volatility < 0.00008:  # Lower threshold for low vol
        return "low_volatility"
    
    # Check for trending vs mean-reverting using Hurst exponent
    hurst_exponent = calculate_hurst_exponent(prices)
    if hurst_exponent > 0.55:  # Reduced threshold
        return "trending"
    elif hurst_exponent < 0.45:  # Increased threshold
        return "mean_reverting"
    
    return "normal"

def adjust_strategy_for_regime(regime, signal, confidence):
    """Adjust strategy based on market regime"""
    if regime == "high_volatility":
        return signal, confidence * 0.8  # Reduced from 0.7
    elif regime == "low_volatility":
        return signal, confidence * 1.2  # Increased from 0.9
    elif regime == "trending":
        return signal, confidence * 1.3  # Increased from 1.1
    elif regime == "mean_reverting":
        # Reverse signals in mean-reverting markets
        if signal == "CALL":
            return "PUT", confidence * 1.1  # Increased from 0.9
        elif signal == "PUT":
            return "CALL", confidence * 1.1
    return signal, confidence

def combine_signals(tech_signal, tech_confidence, ml_signal, ml_confidence, regime, ml_training_samples):
    """Combine technical and ML signals with regime awareness"""
    # FIXED: More permissive ML weighting
    if ml_training_samples > 100 and ml_confidence > 0.4:  # Reduced thresholds
        ml_weight = 0.6  # Reduced from 0.7
        tech_weight = 0.4
    elif ml_training_samples > 25:  # Reduced from 50
        ml_weight = 0.5
        tech_weight = 0.5
    else:
        ml_weight = 0.4  # Increased from 0.3
        tech_weight = 0.6
    
    if tech_signal == ml_signal and ml_confidence > 0.4:  # Reduced from 0.5
        combined_confidence = (tech_confidence * tech_weight + ml_confidence * ml_weight)
        return tech_signal, combined_confidence
    elif ml_confidence > 0.5:  # Reduced from 0.6
        return ml_signal, ml_confidence * ml_weight
    elif tech_confidence > 0.4 and regime in ["trending", "mean_reverting", "normal"]:  # Added normal
        return tech_signal, tech_confidence * tech_weight
    
    # FIXED: Return technical signal even if confidence is lower
    if tech_signal and tech_confidence > 0.2:  # New condition
        return tech_signal, tech_confidence * 0.8
    
    return None, 0.0

def dynamic_position_sizing(volatility, account_balance, win_rate, consecutive_losses=0):
    """Improved position sizing with volatility adjustment"""
    if win_rate == 0 or volatility == 0:
        base_size = TRADE_SIZE_USD
    else:
        # Kelly criterion inspired sizing
        kelly_fraction = win_rate - (1 - win_rate) / (1/win_rate if win_rate > 0 else 1)
        base_size = account_balance * max(0.01, min(0.05, kelly_fraction * 0.5))
    
    # Volatility adjustment - more permissive
    vol_adjustment = min(1.5, 0.1 / max(volatility, 0.0001))  # Increased max adjustment
    
    # Reduce size after losses
    loss_adjustment = 1.0 / (1 + consecutive_losses * 0.3)  # Reduced penalty
    
    # Synthetic market adjustment
    if is_synthetic_symbol(SYMBOL):
        vol_adjustment *= 0.9  # Less reduction for synthetic markets
    
    position_size = base_size * vol_adjustment * loss_adjustment
    
    return max(10.0, min(position_size, account_balance * 0.05))  # Increased max to 5% of account

def risk_allows_trade(size_usd):
    """Check if risk management allows trading"""
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
    if hour in [0, 1, 2, 12, 13]:  # Low liquidity hours
        return 0.8  # Reduced from 0.7
    return 1.0

def collect_market_data():
    """Collect comprehensive market data for ML training"""
    if len(ticks) < 10:  # Reduced from 20 for faster data collection
        return None, None
    
    current_window = list(ticks)
    features = ml_predictor.extract_features(current_window)
    
    # Additional market context
    market_context = {
        'timestamp': datetime.now(),
        'price': current_window[-1],
        'volatility': estimate_volatility(current_window),
        'regime': detect_market_regime(current_window),
        'volume': len(ticks)  # Proxy for activity
    }
    
    return features, market_context

async def handle_tick_message(msg):
    if "tick" not in msg:
        return
    t = msg["tick"]
    price = float(t.get("quote") or t.get("price"))
    ticks.append(price)
    logger.debug("tick %.5f len=%d", price, len(ticks))

# Enhanced record_trade_outcome function
def record_trade_outcome(trade_type, entry_price, exit_price, success, ml_features, regime):
    """Enhanced trade outcome recording"""
    # Determine actual market movement
    price_change = exit_price - entry_price
    if abs(price_change) < entry_price * 0.001:  # Minimal change threshold
        actual_outcome = "neutral"
    elif price_change > 0:
        actual_outcome = "up"
    else:
        actual_outcome = "down"
    
    # Calculate confidence based on price movement magnitude
    movement_strength = abs(price_change) / entry_price
    confidence = min(0.9, movement_strength * 10)  # Scale to confidence
    
    # Record for ML learning with detailed metadata
    ml_reporter.record_prediction_result(
        ml_features, trade_type, confidence, actual_outcome, regime
    )
    
    # Enhanced trade outcome storage
    trade_outcomes.append({
        'timestamp': datetime.now(),
        'trade_type': trade_type,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'success': success,
        'price_change': price_change,
        'percent_change': (price_change / entry_price) * 100,
        'actual_outcome': actual_outcome,
        'movement_strength': movement_strength,
        'regime': regime,
        'duration': DURATION
    })
    
    # Keep only last 500 trades
    if len(trade_outcomes) > 500:
        trade_outcomes.pop(0)

# Add this function for periodic data collection
async def collect_periodic_data():
    """Periodically collect market data even when not trading"""
    while True:
        try:
            await asyncio.sleep(300)  # Collect every 5 minutes
            if len(ticks) >= 10:  # Reduced threshold
                features, context = collect_market_data()
                if features is not None:
                    # Store for later analysis
                    ml_predictor.add_training_sample(features, 'unknown', 0.5, context['regime'])
                    logger.debug("Collected market data for ML training")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Data collection error: {e}")
            await asyncio.sleep(60)

async def manage_ml_data():
    """Regularly clean and optimize ML data"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            ml_predictor.manage_training_data()
            logger.info(f"ML data managed: {len(ml_predictor.training_data)} samples")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Data management error: {e}")

def export_training_data():
    """Export collected data for external analysis"""
    if not trade_outcomes:
        return False
    
    # Create DataFrame with all trade data
    df = pd.DataFrame(trade_outcomes)
    
    # Add ML performance data
    ml_stats = ml_reporter.get_performance_stats()
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f'trading_data_{timestamp}.csv', index=False)
    
    # Save ML stats
    with open(f'ml_stats_{timestamp}.json', 'w') as f:
        json.dump(ml_stats, f, indent=2, default=str)
    
    logger.info(f"Exported {len(df)} trades to trading_data_{timestamp}.csv")
    send_telegram(f"📊 Data exported: {len(df)} trades")
    return True

# === core trading ===
async def place_trade(ws, symbol, trade_type, size_usd, ml_features=None, regime="normal"):
    global daily_pnl, hourly_pnl, win_rate, trade_history
    
    # Apply seasonal adjustment
    size_usd *= seasonal_adjustment()
    amount = round(size_usd, 2)
    entry_price = ticks[-1] if ticks else 100

    if PAPER_MODE:
        # More realistic paper trading simulation
        volatility = estimate_volatility(list(ticks))
        win_probability = 0.52  # Base win probability
        
        # Adjust win probability based on market regime
        if regime == "trending":
            win_probability = 0.55
        elif regime == "high_volatility":
            win_probability = 0.48
        
        result_win = random.random() < win_probability
        pnl = (0.005 if result_win else -0.01) * amount
        daily_pnl += pnl
        hourly_pnl += pnl
        
        # Update win rate
        trade_history.append(1 if result_win else 0)
        if len(trade_history) > 10:
            win_rate = sum(trade_history[-10:]) / 10
        
        # Simulate exit price with more realistic movement
        if result_win:
            exit_price = entry_price * (1 + random.uniform(0.003, 0.008))
        else:
            exit_price = entry_price * (1 - random.uniform(0.005, 0.012))
        
        # Record outcome for ML learning
        record_trade_outcome(trade_type, entry_price, exit_price, result_win, ml_features, regime)
        
        logger.info("PAPER TRADE %s %.2f pnl=%.4f daily=%.2f hourly=%.2f",
                    trade_type, amount, pnl, daily_pnl, hourly_pnl)
        send_telegram(f"PAPER TRADE {trade_type} {amount}USD pnl={pnl:.2f}")
        
        # Update risk manager
        risk_manager.update_trade_result(result_win, amount, ml_features[-1] if ml_features is not None else 0.5)
        
        return result_win

    logger.info("LIVE trade requested: %s amount=%.2f", trade_type, amount)
    
    # LIVE TRADING IMPLEMENTATION
    try:
        # FIXED: Use correct contract types for R_100 symbol
        # For R_100, valid contract types are "CALL" and "PUT" (uppercase)
        valid_contract_type = trade_type.upper()  # Ensure uppercase
        
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": valid_contract_type,  # Use uppercase
            "currency": "USD",
            "duration": DURATION,
            "duration_unit": "s",
            "symbol": symbol
        }
        
        logger.debug("Sending proposal request: %s", proposal_request)
        
        # Send the proposal request first
        await ws.send(json.dumps(proposal_request))
        
        # Wait for proposal response
        proposal_response = await asyncio.wait_for(ws.recv(), timeout=10)
        proposal_data = json.loads(proposal_response)
        
        logger.debug("Proposal response: %s", proposal_data)
        
        if "error" in proposal_data:
            error_msg = proposal_data["error"].get("message", "Unknown error")
            error_details = proposal_data["error"].get("details", {})
            logger.error("Proposal error: %s - Details: %s", error_msg, error_details)
            send_telegram(f"Proposal failed: {error_msg}")
            return False
            
        if "proposal" in proposal_data:
            # Now send the buy request with the received proposal ID
            buy_request = {
                "buy": proposal_data["proposal"]["id"],
                "price": amount
            }
            
            logger.debug("Sending buy request: %s", buy_request)
            await ws.send(json.dumps(buy_request))
            
            # Wait for buy response
            buy_response = await asyncio.wait_for(ws.recv(), timeout=10)
            buy_data = json.loads(buy_response)
            
            logger.debug("Buy response: %s", buy_data)
            
            if "error" in buy_data:
                error_msg = buy_data["error"].get("message", "Unknown error")
                logger.error("Buy error: %s", error_msg)
                send_telegram(f"Buy failed: {error_msg}")
                return False
                
            if "buy" in buy_data:
                contract_id = buy_data["buy"]["contract_id"]
                logger.info("LIVE trade executed: %s %s %.2f USD - Contract ID: %s", 
                            trade_type, symbol, amount, contract_id)
                send_telegram(f"LIVE TRADE {trade_type} {symbol} {amount}USD - ID: {contract_id}")
                
                # For now, just record the trade was placed successfully
                trade_history.append(1)
                if len(trade_history) > 10:
                    win_rate = sum(trade_history[-10:]) / 10
                
                # In live trading, we would need to track the contract to determine outcome
                # For now, we'll assume success and record a placeholder outcome
                exit_price = entry_price * 1.005  # Placeholder
                record_trade_outcome(trade_type, entry_price, exit_price, True, ml_features, regime)
                
                risk_manager.update_trade_result(True, amount, ml_features[-1] if ml_features is not None else 0.5)
                
                return True
            
    except asyncio.TimeoutError:
        logger.error("Trade request timed out")
        send_telegram("Trade request timed out")
        return False
    except Exception as e:
        logger.exception("Error executing trade: %s", e)
        send_telegram(f"Trade error: {str(e)}")
        return False

async def run_bot():
    global last_hour_reset, win_rate, hourly_pnl, daily_pnl
    
    logger.info("=== STARTING BOT WITH FIXED SIGNAL GENERATION ===")
    logger.info("SYMBOL=%s PAPER=%s", SYMBOL, PAPER_MODE)
    logger.info("Trade size: $%.2f, Daily loss limit: $%.2f, Hourly loss limit: $%.2f", 
                TRADE_SIZE_USD, MAX_DAILY_LOSS, MAX_HOURLY_LOSS)
    logger.info("FIXED PARAMETERS: Confidence threshold=%.2f, Volatility threshold=%.6f", 
                CONFIDENCE_THRESHOLD, VOLATILITY_THRESHOLD)
    
    # Check if trading on synthetic market
    if is_synthetic_symbol(SYMBOL):
        logger.info("Trading on SYNTHETIC market: %s", SYMBOL)
        send_telegram(f"🤖 Trading on SYNTHETIC market: {SYMBOL}")
        logger.info("Using synthetic-optimized signal generation")
    else:
        logger.info("Trading on NON-SYNTHETIC market: %s", SYMBOL)
        send_telegram(f"🤖 Trading on NON-SYNTHETIC market: {SYMBOL}")
    
    # Start ML services
    reporting_task = asyncio.create_task(ml_reporter.start_reporting())
    data_collection_task = asyncio.create_task(collect_periodic_data())
    data_management_task = asyncio.create_task(manage_ml_data())
    
    # Export data on startup if available
    if os.path.exists('models') and any(fname.endswith('.pkl') for fname in os.listdir('models')):
        export_training_data()
    
    try:
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
                    news_ok = True  # Always true for synthetic trading

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

                        # FIXED: Simplified trading logic with fewer data requirements
                        if len(ticks) >= 5:  # Reduced from 10 for much faster trading
                            # Collect market data for ML
                            features, market_context = collect_market_data()
                            
                            current_window = list(ticks)
                            regime = detect_market_regime(current_window)
                            volatility = estimate_volatility(current_window)
                            
                            # Get technical signal with confidence
                            technical_signal, tech_confidence = compute_signal(current_window)
                            
                            # Get ML prediction using enhanced ML
                            ml_prediction = None
                            ml_confidence = 0.0
                            if features is not None:
                                ml_prediction, ml_confidence = ml_predictor.predict(features)
                            
                            # Get ML training stats for weight calculation
                            ml_stats = ml_predictor.get_training_stats()
                            training_samples = ml_stats.get('training_samples', 0)
                            
                            # Combine signals with adaptive weighting
                            final_signal, signal_confidence = combine_signals(
                                technical_signal, tech_confidence, ml_prediction, 
                                ml_confidence, regime, training_samples
                            )
                            
                            final_signal, signal_confidence = adjust_strategy_for_regime(
                                regime, final_signal, signal_confidence
                            )
                            
                            # Calculate dynamic position size
                            size = dynamic_position_sizing(volatility, account_balance, win_rate, risk_manager.consecutive_losses)
                            
                            logger.debug("Signal=%s Confidence=%.2f Regime=%s Vol=%.6f Size=%.2f ML Samples=%d",
                                         final_signal, signal_confidence, regime, volatility, size, 
                                         training_samples)
                            
                            # FIXED: Use the lower confidence threshold
                            if (final_signal and signal_confidence > CONFIDENCE_THRESHOLD and 
                                risk_allows_trade(size) and risk_manager.should_trade(signal_confidence, volatility, ml_confidence)):
                                
                                logger.info("🎯 EXECUTING TRADE: %s (Confidence: %.2f, Size: $%.2f)", 
                                           final_signal, signal_confidence, size)
                                trade_success = await place_trade(ws, SYMBOL, final_signal, size, features, regime)
                                await asyncio.sleep(1)  # Reduced from 2 seconds

                        # Check loss limits
                        if daily_pnl <= -abs(MAX_DAILY_LOSS):
                            logger.critical("Daily loss cap hit, sleeping.")
                            send_telegram(f"🔴 Daily loss cap hit: {daily_pnl:.2f}.")
                            await asyncio.sleep(60 * 60 * 6)
                            
                        if hourly_pnl <= -abs(MAX_HOURLY_LOSS):
                            logger.critical("Hourly loss cap hit, sleeping.")
                            send_telegram(f"🔴 Hourly loss cap hit: {hourly_pnl:.2f}.")
                            await asyncio.sleep(60 * 30)

                        # Daily data export (at midnight)
                        if current_time.hour == 0 and current_time.minute < 5:
                            export_training_data()
                            await asyncio.sleep(300)  # Sleep 5 minutes after export

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
    finally:
        # Ensure tasks are cancelled when the bot stops
        reporting_task.cancel()
        data_collection_task.cancel()
        data_management_task.cancel()
        try:
            await asyncio.gather(reporting_task, data_collection_task, data_management_task, return_exceptions=True)
        except:
            pass
        
        # Export data on shutdown
        export_training_data()
        logger.info("ML tasks cancelled")

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        # Export data on manual stop
        export_training_data()