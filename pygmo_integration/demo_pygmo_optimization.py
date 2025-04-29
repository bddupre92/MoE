"""
Demonstration script for PyGMO optimization of MoE model

This script demonstrates the integration of PyGMO optimization with
the Mixture of Experts (MoE) model for migraine prediction. It shows
how to optimize the model parameters, visualize the results, and
compare the performance of the baseline and optimized models.

Author: Manus AI
Date: April 28, 2025
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Union, Optional, Any

# Import local modules
from enhanced_pygmo_optimizer import PyGMOOptimizer, OptimizationConfig
from optimization_problem import EndToEndMoEOptimizationProblem
from fitness_functions import FitnessFunctions
from moe_integration import MoEPyGMOIntegration
from parallel_evaluation import ParallelEvaluator, IslandModelParallelizer
from visualization import OptimizationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyGMO optimization of MoE model')
    
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output files')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--n-generations', type=int, default=20,
                        help='Number of generations for optimization')
    parser.add_argument('--population-size', type=int, default=50,
                        help='Population size for optimization')
    parser.add_argument('--n-islands', type=int, default=5,
                        help='Number of islands for parallel optimization')
    parser.add_argument('--algorithm', type=str, default='nsga2',
                        choices=['nsga2', 'moead', 'de', 'pso'],
                        help='Optimization algorithm')
    parser.add_argument('--objectives', type=str, nargs='+',
                        default=['accuracy', 'f1', 'false_positive_rate', 'expert_balance'],
                        help='Optimization objectives')
    parser.add_argument('--compare', action='store_true',
                        help='Compare baseline and optimized models')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize optimization results')
    parser.add_argument('--save-model', action='store_true',
                        help='Save optimized model')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load model from file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def setup_environment(args):
    """Set up environment for demonstration."""
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Set up environment with seed {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")


def create_optimization_config(args):
    """Create optimization configuration."""
    config = OptimizationConfig(
        optimization_type='end_to_end',
        algorithm_type=args.algorithm,
        population_size=args.population_size,
        generations=args.n_generations,
        islands=args.n_islands,
        objectives=args.objectives,
        output_dir=args.output_dir
    )
    
    logger.info(f"Created optimization configuration: {config}")
    return config


def run_optimization(args, config):
    """Run PyGMO optimization of MoE model."""
    logger.info("Starting PyGMO optimization of MoE model")
    
    # Create MoE-PyGMO integration
    integration = MoEPyGMOIntegration(config=config)
    
    # Create baseline model (for comparison)
    baseline_model = integration.moe_model
    
    # Optimize model
    start_time = time.time()
    optimized_model = integration.optimize_model(optimization_type='end_to_end')
    optimization_time = time.time() - start_time
    
    logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
    
    # Evaluate models
    baseline_metrics = integration.evaluate_model(baseline_model)
    optimized_metrics = integration.evaluate_model(optimized_model)
    
    # Compare models
    if args.compare:
        comparison = integration.compare_models(baseline_model, optimized_model)
        logger.info(f"Model comparison:\n{comparison}")
    
    # Visualize results
    if args.visualize:
        visualizer = OptimizationVisualizer(output_dir=args.output_dir)
        
        # Get optimization results
        optimization_results = {
            'fitness_history': integration.optimizer.fitness_history,
            'objective_names': args.objectives,
            'expert_weights': [np.random.randn(10) for _ in range(3)],  # Mock data
            'gating_weights': np.random.randn(3)  # Mock data
        }
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Baseline': baseline_metrics,
            'Optimized': optimized_metrics
        })
        metrics_df['Improvement'] = metrics_df['Optimized'] - metrics_df['Baseline']
        metrics_df['Improvement (%)'] = (metrics_df['Improvement'] / metrics_df['Baseline'] * 100).round(2)
        
        # Get expert contributions
        expert_contributions = integration.moe_model.get_expert_contributions(
            integration.data_handler.get_test_data()[0])
        
        # Create dashboard figures
        figure_paths = visualizer.create_dashboard_figures(
            optimization_results=optimization_results,
            metrics_df=metrics_df,
            expert_contributions=expert_contributions,
            expert_names=['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert'],
            feature_names=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
        )
        
        logger.info(f"Created visualization figures: {list(figure_paths.keys())}")
    
    # Save model
    if args.save_model:
        saved_files = integration.save_results()
        logger.info(f"Saved model and results: {saved_files}")
    
    return {
        'baseline_model': baseline_model,
        'optimized_model': optimized_model,
        'baseline_metrics': baseline_metrics,
        'optimized_metrics': optimized_metrics,
        'comparison': comparison if args.compare else None,
        'optimization_time': optimization_time
    }


def print_results(results):
    """Print optimization results."""
    print("\n" + "="*80)
    print("PyGMO Optimization Results".center(80))
    print("="*80)
    
    # Print optimization time
    print(f"\nOptimization Time: {results['optimization_time']:.2f} seconds")
    
    # Print baseline metrics
    print("\nBaseline Model Metrics:")
    for metric, value in results['baseline_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Print optimized metrics
    print("\nOptimized Model Metrics:")
    for metric, value in results['optimized_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Print comparison
    if results['comparison'] is not None:
        print("\nPerformance Improvement:")
        for metric, row in results['comparison'].iterrows():
            print(f"  {metric}: {row['Improvement']:.4f} ({row['Improvement (%)']:+.2f}%)")
    
    print("\n" + "="*80)


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up environment
    setup_environment(args)
    
    # Create optimization configuration
    config = create_optimization_config(args)
    
    # Run optimization
    results = run_optimization(args, config)
    
    # Print results
    print_results(results)
    
    logger.info("Demonstration completed successfully!")
    return results


if __name__ == "__main__":
    main()
