#!/usr/bin/env python3
# validate_qed_theory.py

"""
One-click validation script for the QED General Theory
Usage: python validate_qed_theory.py --engine path/to/your/qed_engine
"""

import argparse
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Validate QED General Theory Claims')
    parser.add_argument('--engine', required=True, help='Path to QED engine implementation')
    parser.add_argument('--output', default='validation_results.json', help='Output file')
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    
    args = parser.parse_args()
    
    print("🔬 QED General Theory Validation Suite")
    print("=" * 50)
    
    # Import the user's QED engine
    try:
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("qed_engine", args.engine)
        qed_engine = module_from_spec(spec)
        spec.loader.exec_module(qed_engine)
        print("✅ QED engine loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load QED engine: {e}")
        return
    
    # Run validation
    validator = TheoryValidator(qed_engine)
    
    if args.quick:
        print("🚀 Running quick validation...")
        results = validator.run_quick_validation()
    else:
        print("🔍 Running comprehensive validation...")
        results = validator.run_comprehensive_validation()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Validation complete! Results saved to {args.output}")
    print(f"📊 Theory Status: {results['theory_status']}")
    print(f"🎯 Overall Success Rate: {results['overall_success_rate']:.2%}")

if __name__ == "__main__":
    main()
