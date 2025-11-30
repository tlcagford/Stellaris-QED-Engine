# quick_test.py
"""
5-minute quick validation for theory assessment
"""

class QuickValidation:
    def __init__(self, qed_engine):
        self.engine = qed_engine
        self.tests = [
            self.test_mathematical_axioms,
            self.test_domain_independence,
            self.test_novel_predictions,
            self.test_scale_invariance
        ]
    
    def run_quick_test(self, timeout=300):
        """Run quick test with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Quick test timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            results = {}
            for test in self.tests:
                test_name = test.__name__
                print(f"Running {test_name}...")
                results[test_name] = test()
            
            return self.assess_quick_results(results)
        finally:
            signal.alarm(0)  # Cancel timeout
    
    def assess_quick_results(self, results):
        """Assess if theory passes quick validation"""
        passed_tests = sum(1 for result in results.values() if result['passed'])
        total_tests = len(results)
        
        quick_status = "PASS" if passed_tests / total_tests > 0.75 else "FAIL"
        
        return {
            'quick_status': quick_status,
            'tests_passed': f"{passed_tests}/{total_tests}",
            'detailed_results': results,
            'recommendation': self.get_recommendation(quick_status)
        }
    
    def get_recommendation(self, status):
        if status == "PASS":
            return "Proceed with comprehensive testing - theory shows promise"
        else:
            return "Theory needs fundamental revisions before further testing"
