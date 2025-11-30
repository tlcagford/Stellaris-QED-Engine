# tests/cross_domain_battery.py
from qed_validation.experiments import DomainTester

class CrossDomainBattery:
    def __init__(self):
        self.domains = {
            'biological': ProteinFoldingTest(),
            'economic': MarketPredictionTest(),
            'ecological': PopulationDynamicsTest(),
            'social': NetworkDiffusionTest()
        }
    
    def run_battery(self, qed_engine):
        """Run all domain tests and return aggregated results"""
        results = {}
        for domain_name, tester in self.domains.items():
            print(f"Testing {domain_name}...")
            results[domain_name] = tester.validate(qed_engine)
        
        return self.aggregate_results(results)
    
    def aggregate_results(self, results):
        """Calculate overall theory performance"""
        success_rates = {}
        for domain, domain_results in results.items():
            success_rates[domain] = domain_results['accuracy']
        
        overall_success = np.mean(list(success_rates.values()))
        consistency = 1 - np.std(list(success_rates.values()))  # How consistent across domains
        
        return {
            'overall_success_rate': overall_success,
            'domain_consistency': consistency,
            'domain_breakdown': success_rates,
            'theory_status': self.assess_theory_status(overall_success, consistency)
        }
    
    def assess_theory_status(self, success_rate, consistency):
        if success_rate > 0.8 and consistency > 0.9:
            return "STRONG_EVIDENCE"
        elif success_rate > 0.7 and consistency > 0.8:
            return "MODERATE_EVIDENCE" 
        elif success_rate > 0.6:
            return "WEAK_EVIDENCE"
        else:
            return "INSUFFICIENT_EVIDENCE"
