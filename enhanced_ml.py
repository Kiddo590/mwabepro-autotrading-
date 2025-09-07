import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta  # ADDED timedelta import
import logging
import random  # ADDED random import

logger = logging.getLogger("deriv-bot")

class EnhancedMLPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.training_data = []
        self.last_training_time = datetime.now()
        self.training_interval = timedelta(hours=1)  # Retrain every hour
        self.model_performance = {}
        self.active_model = 'random_forest'  # Default model
        self.adaptive_learning_rate = True
        self.performance_threshold = 0.55
        self.forgetting_factor = 0.99
        
    def extract_features(self, price_window):
        """Extract comprehensive features from price data"""
        prices = np.array(price_window)
        
        features = {
            'returns_std': np.std(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0,
            'returns_mean': np.mean(np.diff(prices) / prices[:-1]) if len(prices) > 1 else 0,
            'price_velocity': (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0,
            'price_acceleration': ((prices[-1] - prices[-5]) - (prices[-5] - prices[-10])) / prices[-10] if len(prices) > 10 else 0,
            'volatility': np.std(prices[-20:]) / np.mean(prices[-20:]) if len(prices) > 20 else 0,
            'rsi': self._calculate_rsi(prices),
            'macd': self._calculate_macd(prices),
            'bollinger_band_position': self._bollinger_band_position(prices),
            'volume_profile': self._volume_profile(prices) if len(prices) > 50 else 0,
            'support_resistance': self._support_resistance_level(prices),
            'time_of_day': datetime.now().hour / 24.0,
            'day_of_week': datetime.now().weekday() / 7.0
        }
        
        return np.array(list(features.values()))
    
    def _calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        gains = np.maximum(0, np.diff(prices))
        losses = np.maximum(0, -np.diff(prices))
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices):
        if len(prices) < 26:
            return 0
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        return ema12 - ema26
    
    def _bollinger_band_position(self, prices, period=20):
        if len(prices) < period:
            return 0
        middle_band = np.mean(prices[-period:])
        std_dev = np.std(prices[-period:])
        upper_band = middle_band + 2 * std_dev
        lower_band = middle_band - 2 * std_dev
        return (prices[-1] - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
    
    def _volume_profile(self, prices):
        if len(prices) < 50:
            return 0
        recent_prices = prices[-20:]
        historical_prices = prices[-50:-20]
        recent_avg = np.mean(recent_prices)
        historical_avg = np.mean(historical_prices)
        return (recent_avg - historical_avg) / historical_avg if historical_avg != 0 else 0
    
    def _support_resistance_level(self, prices, lookback=30):
        if len(prices) < lookback:
            return 0
        recent_max = np.max(prices[-lookback:])
        recent_min = np.min(prices[-lookback:])
        current_price = prices[-1]
        return (current_price - recent_min) / (recent_max - recent_min) if recent_max != recent_min else 0.5
    
    def add_training_sample(self, features, actual_outcome, confidence, regime):
        """Add a new training sample with metadata"""
        self.training_data.append({
            'features': features,
            'outcome': actual_outcome,
            'confidence': confidence,
            'regime': regime,
            'timestamp': datetime.now()
        })
        
        # Auto-train if we have enough data and it's time
        if len(self.training_data) >= 100 and datetime.now() - self.last_training_time > self.training_interval:
            self.train_models()
    
    def train_models(self):
        """Train all models on collected data"""
        if len(self.training_data) < 50:
            return False
            
        # Prepare training data
        X = np.array([sample['features'] for sample in self.training_data])
        y = np.array([1 if sample['outcome'] in ['up', 'CALL'] else 0 for sample in self.training_data])
        
        if len(np.unique(y)) < 2:  # Need both classes
            return False
        
        # Scale features
        try:
            X_scaled = self.scaler.fit_transform(X)
        except:
            return False
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train and evaluate each model
        best_score = 0
        best_model = None
        
        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                self.model_performance[name] = score
                
                if score > best_score:
                    best_score = score
                    best_model = name
                    self.active_model = name
                    
                logger.info(f"Model {name} trained with accuracy: {score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        self.last_training_time = datetime.now()
        
        # Save models
        self.save_models()
        
        return True
    
    def predict(self, features):
        """Make prediction using the best performing model"""
        if not self.models or len(self.training_data) < 10:
            return None, 0.5
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            model = self.models[self.active_model]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                confidence = max(proba)
                prediction = 1 if np.argmax(proba) == 1 else 0
            else:
                prediction = model.predict(features_scaled)[0]
                confidence = 0.6
                
            return ('CALL' if prediction == 1 else 'PUT'), confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None, 0.5
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs('models', exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, f'models/{name}_{datetime.now().strftime("%Y%m%d_%H%M")}.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
    def load_models(self):
        """Load trained models from disk"""
        try:
            for name in self.models.keys():
                model_files = [f for f in os.listdir('models') if f.startswith(name)]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    self.models[name] = joblib.load(f'models/{latest_model}')
            
            if os.path.exists('models/scaler.pkl'):
                self.scaler = joblib.load('models/scaler.pkl')
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def get_training_stats(self):
        """Get statistics about training data and model performance"""
        return {
            'training_samples': len(self.training_data),
            'active_model': self.active_model,
            'model_performance': self.model_performance,
            'last_training': self.last_training_time,
            'next_training': self.last_training_time + self.training_interval
        }
    
    def manage_training_data(self):
        """Ensure training data quality and relevance"""
        # Remove outdated samples (older than 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.training_data = [
            sample for sample in self.training_data 
            if sample['timestamp'] > cutoff_date
        ]
        
        # Balance classes if needed
        call_samples = [s for s in self.training_data if s['outcome'] in ['up', 'CALL']]
        put_samples = [s for s in self.training_data if s['outcome'] in ['down', 'PUT']]
        neutral_samples = [s for s in self.training_data if s['outcome'] == 'neutral']
        
        if len(call_samples) > len(put_samples) * 2:
            # Downsample majority class
            call_samples = random.sample(call_samples, len(put_samples) * 2)
            self.training_data = put_samples + call_samples + neutral_samples
        elif len(put_samples) > len(call_samples) * 2:
            put_samples = random.sample(put_samples, len(call_samples) * 2)
            self.training_data = call_samples + put_samples + neutral_samples
