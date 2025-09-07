import asyncio
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("deriv-bot")

class MLReporter:
    def __init__(self, ml_predictor, telegram_callback):
        self.ml_predictor = ml_predictor
        self.send_telegram = telegram_callback
        self.last_report_time = datetime.now()
        self.report_interval = timedelta(minutes=30)
        self.prediction_history = []
        
    async def start_reporting(self):
        """Start periodic reporting of ML status"""
        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                await self.send_status_report()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reporting error: {e}")
                await asyncio.sleep(300)
    
    async def send_status_report(self):
        """Send comprehensive ML status report"""
        stats = self.ml_predictor.get_training_stats()
        
        report = f"🤖 ML Status Report\n"
        report += f"📊 Training Samples: {stats['training_samples']}\n"
        report += f"⭐ Active Model: {stats['active_model']}\n"
        report += f"⏰ Last Training: {stats['last_training'].strftime('%H:%M')}\n"
        report += f"🔜 Next Training: {stats['next_training'].strftime('%H:%M')}\n"
        
        # Add model performances
        report += "📈 Model Accuracies:\n"
        for model, accuracy in stats['model_performance'].items():
            report += f"  {model.replace('_', ' ').title()}: {accuracy:.3f}\n"
        
        # Add prediction confidence trends
        if self.prediction_history:
            recent_confidences = [p['confidence'] for p in self.prediction_history[-10:]]
            avg_confidence = sum(recent_confidences) / len(recent_confidences)
            report += f"🎯 Avg Confidence: {avg_confidence:.3f}\n"
        
        self.send_telegram(report)
        self.last_report_time = datetime.now()
    
    def record_prediction_result(self, features, predicted_direction, confidence, actual_outcome, regime):
        """Record prediction results for analysis"""
        prediction_record = {
            'timestamp': datetime.now(),
            'predicted': predicted_direction,
            'confidence': confidence,
            'actual': actual_outcome,
            'regime': regime,
            'correct': predicted_direction and (
                (predicted_direction == 'CALL' and actual_outcome == 'up') or
                (predicted_direction == 'PUT' and actual_outcome == 'down')
            )
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 records
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        # Add to training data
        self.ml_predictor.add_training_sample(features, actual_outcome, confidence, regime)
        
        # Send immediate update for very wrong predictions
        if confidence > 0.7 and not prediction_record['correct']:
            alert = f"⚠️ High Confidence Error\n"
            alert += f"Predicted: {predicted_direction} ({confidence:.2f})\n"
            alert += f"Actual: {actual_outcome}\n"
            alert += f"Regime: {regime}"
            self.send_telegram(alert)
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.prediction_history:
            return {}
        
        correct_predictions = [p for p in self.prediction_history if p['correct']]
        accuracy = len(correct_predictions) / len(self.prediction_history) if self.prediction_history else 0
        
        # Accuracy by regime
        regime_stats = {}
        for regime in set(p['regime'] for p in self.prediction_history):
            regime_predictions = [p for p in self.prediction_history if p['regime'] == regime]
            regime_correct = [p for p in regime_predictions if p['correct']]
            regime_stats[regime] = {
                'total': len(regime_predictions),
                'accuracy': len(regime_correct) / len(regime_predictions) if regime_predictions else 0
            }
        
        return {
            'total_predictions': len(self.prediction_history),
            'overall_accuracy': accuracy,
            'regime_stats': regime_stats
        }