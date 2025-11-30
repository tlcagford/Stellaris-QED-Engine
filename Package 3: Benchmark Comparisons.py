# tests/benchmark_comparison.py
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

class BenchmarkComparator:
    def __init__(self):
        self.baseline_models = {
            'biological': 'AlphaFold2',
            'economic': 'ARIMA_GARCH', 
            'ecological': 'LotkaVolterra',
            'social': 'SIR_Model'
        }
    
    def compare_performance(self, domain, test_data, qed_predictions):
        """Compare QED against domain-specific baseline"""
        baseline_predictions = self.get_baseline_predictions(domain, test_data)
        
        comparison = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mse']:
            qed_score = self.calculate_metric(metric, test_data, qed_predictions)
            baseline_score = self.calculate_metric(metric, test_data, baseline_predictions)
            
            comparison[metric] = {
                'qed': qed_score,
                'baseline': baseline_score,
                'improvement': (qed_score - baseline_score) / baseline_score
            }
        
        return comparison
    
    def statistical_significance(self, qed_scores, baseline_scores):
        """Test if QED improvement is statistically significant"""
        from scipy.stats import ttest_rel
        
        t_stat, p_value = ttest_rel(qed_scores, baseline_scores)
        return p_value < 0.05  # Significant improvement
