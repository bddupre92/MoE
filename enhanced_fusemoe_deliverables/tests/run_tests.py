"""
Modified test runner script for the Enhanced FuseMoE system.

This script discovers and runs all tests in the tests directory,
generates a test report, and calculates code coverage.
"""

import unittest
import sys
import os
import argparse
import time
import coverage
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def discover_tests(test_dir, pattern='test_*.py'):
    """Discover all tests in the given directory."""
    loader = unittest.TestLoader()
    return loader.discover(test_dir, pattern=pattern)


def run_tests(test_suite, verbosity=2):
    """Run the test suite and return the result."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(test_suite)


def generate_report(result, start_time, end_time, cov_data=None):
    """Generate a test report."""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time': end_time - start_time,
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success': result.wasSuccessful(),
        'success_percentage': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
    }
    
    if cov_data:
        report['coverage'] = cov_data
    
    return report


def save_report(report, output_file):
    """Save the report to a file."""
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)


def print_report(report):
    """Print the report to the console."""
    print("\n===== Test Report =====")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Execution Time: {report['execution_time']:.2f} seconds")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Success: {report['success']}")
    print(f"Success Percentage: {report['success_percentage']:.2f}%")
    
    if 'coverage' in report:
        print("\n===== Coverage Report =====")
        print(f"Total Coverage: {report['coverage']['total_coverage']:.2f}%")
        print("\nCoverage by Module:")
        for module, cov in report['coverage']['module_coverage'].items():
            print(f"  {module}: {cov:.2f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run tests for Enhanced FuseMoE system')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--functional-only', action='store_true', help='Run only functional tests')
    parser.add_argument('--with-coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--output', type=str, default='test_report.json', help='Output file for test report')
    parser.add_argument('--verbosity', type=int, default=2, help='Verbosity level (1-3)')
    args = parser.parse_args()
    
    # Start coverage if requested
    cov = None
    if args.with_coverage:
        cov = coverage.Coverage(
            source=['models', 'optimization', 'utils'],
            omit=['*/tests/*', '*/notebooks/*']
        )
        cov.start()
    
    # Start timer
    start_time = time.time()
    
    # Discover tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    test_suite = unittest.TestSuite()
    
    if args.unit_only:
        unit_tests = discover_tests(os.path.join(test_dir, 'unit'))
        test_suite.addTests(unit_tests)
        print(f"Discovered {unit_tests.countTestCases()} unit tests")
    elif args.integration_only:
        integration_tests = discover_tests(os.path.join(test_dir, 'integration'))
        test_suite.addTests(integration_tests)
        print(f"Discovered {integration_tests.countTestCases()} integration tests")
    elif args.functional_only:
        functional_tests = discover_tests(os.path.join(test_dir, 'functional'))
        test_suite.addTests(functional_tests)
        print(f"Discovered {functional_tests.countTestCases()} functional tests")
    else:
        # Run all tests
        for test_type in ['unit', 'integration', 'functional']:
            test_path = os.path.join(test_dir, test_type)
            if os.path.exists(test_path):
                type_tests = discover_tests(test_path)
                test_suite.addTests(type_tests)
                print(f"Discovered {type_tests.countTestCases()} {test_type} tests")
    
    # Run tests
    print(f"\nRunning {test_suite.countTestCases()} total tests...")
    result = run_tests(test_suite, verbosity=args.verbosity)
    
    # End timer
    end_time = time.time()
    
    # Stop coverage if started
    cov_data = None
    if args.with_coverage and cov:
        cov.stop()
        cov.save()
        
        # Generate coverage report
        try:
            total_coverage = cov.report()
            
            # Get coverage by module
            module_coverage = {}
            for module in ['models', 'optimization', 'utils']:
                module_path = os.path.join(os.path.dirname(__file__), module)
                if os.path.exists(module_path):
                    try:
                        module_cov = cov.report(include=[f'{module}/*'])
                        module_coverage[module] = module_cov
                    except coverage.exceptions.NoDataError:
                        module_coverage[module] = 0.0
            
            cov_data = {
                'total_coverage': total_coverage,
                'module_coverage': module_coverage
            }
        except coverage.exceptions.NoDataError:
            print("Warning: No coverage data collected. Tests may not be executing code in the specified source directories.")
            cov_data = {
                'total_coverage': 0.0,
                'module_coverage': {
                    'models': 0.0,
                    'optimization': 0.0,
                    'utils': 0.0
                }
            }
    
    # Generate report
    report = generate_report(result, start_time, end_time, cov_data)
    
    # Save report
    save_report(report, args.output)
    
    # Print report
    print_report(report)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
