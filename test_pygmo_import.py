#!/usr/bin/env python3
"""
Test script to verify PyGMO integration module imports
"""

import sys
import os

# Add the necessary directories to the path
sys.path.append('/home/ubuntu')

try:
    print("Attempting to import PyGMO integration modules...")
    from pygmo_integration.enhanced_pygmo_optimizer import PyGMOOptimizer as EnhancedPyGMOOptimizer
    from pygmo_integration.optimization_problem import MigrainePredictionProblem
    from pygmo_integration.fitness_functions import AccuracyFitnessFunction
    from pygmo_integration.moe_integration import PyGMOMoEIntegration
    from pygmo_integration.visualization import OptimizationVisualizer
    
    print("All PyGMO integration modules imported successfully!")
    print("\nModule details:")
    print(f"- EnhancedPyGMOOptimizer: {EnhancedPyGMOOptimizer}")
    print(f"- MigrainePredictionProblem: {MigrainePredictionProblem}")
    print(f"- AccuracyFitnessFunction: {AccuracyFitnessFunction}")
    print(f"- PyGMOMoEIntegration: {PyGMOMoEIntegration}")
    print(f"- OptimizationVisualizer: {OptimizationVisualizer}")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nChecking if files exist:")
    
    files_to_check = [
        '/home/ubuntu/pygmo_integration/enhanced_pygmo_optimizer.py',
        '/home/ubuntu/pygmo_integration/optimization_problem.py',
        '/home/ubuntu/pygmo_integration/fitness_functions.py',
        '/home/ubuntu/pygmo_integration/moe_integration.py',
        '/home/ubuntu/pygmo_integration/visualization.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} does not exist")
