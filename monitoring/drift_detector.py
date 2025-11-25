import numpy as np
from scipy import stats
from monitoring.metrics_logger import MetricsLogger

class DriftDetector:
    """Detects data drift using statistical tests."""
    
    def __init__(self, logger=None, threshold=0.05):
        self.logger = logger or MetricsLogger()
        self.threshold = threshold  # p-value threshold for drift detection
        self.baseline_data = None
    
    def set_baseline(self, hours=168):  # Default: 1 week
        """Set baseline distribution from historical data."""
        predictions = self.logger.get_recent_predictions(limit=1000)
        
        if len(predictions) < 30:
            print("⚠️ Not enough data for baseline (need at least 30 samples)")
            return False
        
        self.baseline_data = {
            'query_lengths': [p['query_length'] for p in predictions],
            'entity_counts': [p['entity_count'] for p in predictions],
            'confidences': [p['confidence'] for p in predictions if p['confidence']],
            'top_scores': [p['top_score'] for p in predictions if p['top_score']],
            'response_times': [p['response_time'] for p in predictions if p['response_time']]
        }
        
        print(f"✓ Baseline set with {len(predictions)} samples")
        return True
    
    def detect_drift(self, recent_hours=24):
        """Detect drift in recent data compared to baseline."""
        if not self.baseline_data:
            print("⚠️ No baseline set. Call set_baseline() first.")
            return {}
        
        recent_predictions = self.logger.get_recent_predictions(limit=500)
        recent_predictions = [p for p in recent_predictions 
                            if p['timestamp']]  # Filter valid entries
        
        if len(recent_predictions) < 10:
            print("⚠️ Not enough recent data for drift detection")
            return {}
        
        recent_data = {
            'query_lengths': [p['query_length'] for p in recent_predictions],
            'entity_counts': [p['entity_count'] for p in recent_predictions],
            'confidences': [p['confidence'] for p in recent_predictions if p['confidence']],
            'top_scores': [p['top_score'] for p in recent_predictions if p['top_score']],
            'response_times': [p['response_time'] for p in recent_predictions if p['response_time']]
        }
        
        drift_results = {}
        
        for metric_name in self.baseline_data.keys():
            baseline = self.baseline_data[metric_name]
            recent = recent_data[metric_name]
            
            if len(baseline) < 10 or len(recent) < 10:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(baseline, recent)
            
            drift_detected = p_value < self.threshold
            
            drift_results[metric_name] = {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected,
                'baseline_mean': float(np.mean(baseline)),
                'recent_mean': float(np.mean(recent)),
                'baseline_std': float(np.std(baseline)),
                'recent_std': float(np.std(recent))
            }
            
            if drift_detected:
                message = f"Drift detected in {metric_name}: p-value={p_value:.4f}"
                print(f"⚠️ {message}")
                self.logger.log_drift_alert(
                    metric_name, 
                    statistic, 
                    self.threshold, 
                    message
                )
        
        return drift_results
    
    def get_drift_summary(self):
        """Get a human-readable summary of drift status."""
        drift_results = self.detect_drift()
        
        if not drift_results:
            return "No drift detection performed (insufficient data)"
        
        drifted_metrics = [k for k, v in drift_results.items() if v['drift_detected']]
        
        if not drifted_metrics:
            return "✓ No drift detected in any metrics"
        
        summary = f"⚠️ Drift detected in {len(drifted_metrics)} metric(s):\n"
        for metric in drifted_metrics:
            info = drift_results[metric]
            summary += f"  - {metric}: baseline={info['baseline_mean']:.2f}, recent={info['recent_mean']:.2f}\n"
        
        return summary
