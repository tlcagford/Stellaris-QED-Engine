# tests/mathematical_axioms.py
import numpy as np
from scipy import stats
from qed_validation.core.axioms import AxiomTester

class MathematicalAxioms:
    def test_unitary_evolution(self, system_states):
        """Axiom 1: Time evolution must be unitary"""
        # Test conservation of probability
        initial_prob = np.sum([abs(amp)**2 for amp in system_states.values()])
        evolved_states = self.apply_time_evolution(system_states)
        final_prob = np.sum([abs(amp)**2 for amp in evolved_states.values()])
        
        assert abs(initial_prob - final_prob) < 1e-10
        return True
    
    def test_superposition_principle(self, state_a, state_b):
        """Axiom 2: Superposition must be linear"""
        combined = self.superposition(state_a, state_b)
        measured_a = self.collapse(combined, basis='state_a')
        measured_b = self.collapse(combined, basis='state_b')
        
        # Test linearity
        expected_ratio = abs(state_a.amplitude)**2 / abs(state_b.amplitude)**2
        actual_ratio = measured_a.probability / measured_b.probability
        
        return stats.ttest_1samp([actual_ratio/expected_ratio], 1.0).pvalue > 0.05
