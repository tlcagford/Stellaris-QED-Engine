from qed_validation import TheoryValidator
from qed_validation.experiments import run_full_suite

# Quick validation
validator = TheoryValidator()
results = validator.run_quick_test()

# Comprehensive testing
full_results = run_full_suite(
    domains=['physics', 'biology', 'economics', 'ecology'],
    tests=['axioms', 'predictions', 'novelty', 'robustness']
)
