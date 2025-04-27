"""
End-to-End Optimization Module for Enhanced FuseMoE

This module provides functionality for end-to-end optimization of the Enhanced FuseMoE
system using PyGMO's evolutionary algorithms.
"""

import torch
import numpy as np
import pygmo as pg
import os
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.training_pipeline import MigraineTrainer
from optimization.evolutionary_algorithms.pygmo_problem import EndToEndOptimizationProblem


class OptimizationManager:
    """
    Manager for coordinating different optimization strategies for Enhanced FuseMoE.
    
    This class provides methods for coordinating different optimization strategies,
    including expert-level optimization, gating optimization, and end-to-end optimization.
    
    Attributes:
        model (Any): Model to optimize
        train_loader (Any): Training data loader
        val_loader (Any): Validation data loader
        test_loader (Any): Test data loader
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds for optimization
        fitness_metric (str): Metric to optimize
        num_epochs (int): Number of training epochs
        patience (int): Early stopping patience
        device (torch.device): Device to use for training
    """
    
    def __init__(self, model: Any, train_loader: Any, val_loader: Any, test_loader: Any,
                param_bounds: Dict[str, Tuple[float, float]], fitness_metric: str = 'auc',
                num_epochs: int = 10, patience: int = 3, device: Optional[torch.device] = None):
        """
        Initialize the optimization manager.
        
        Args:
            model: Model to optimize
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            param_bounds: Dictionary mapping parameter names to (min, max) tuples
            fitness_metric: Metric to optimize
            num_epochs: Number of training epochs
            patience: Early stopping patience
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.param_bounds = param_bounds
        self.fitness_metric = fitness_metric
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create problem
        self.problem = EndToEndOptimizationProblem(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            param_bounds=param_bounds,
            fitness_metric=fitness_metric,
            num_epochs=num_epochs,
            patience=patience,
            device=device
        )
    
    def optimize(self, algorithm: str = 'sade', pop_size: int = 10, generations: int = 5,
                islands: int = 2, output_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize model hyperparameters.
        
        Args:
            algorithm: PyGMO algorithm to use
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            output_dir: Directory to save optimization results
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing optimization results
        """
        if verbose:
            print(f"Starting optimization with {algorithm}, pop_size={pop_size}, generations={generations}, islands={islands}")
        
        # Create PyGMO problem
        prob = pg.problem(self.problem)
        
        # Create algorithm
        if algorithm == 'de':
            algo = pg.algorithm(pg.de(gen=generations))
        elif algorithm == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations))
        elif algorithm == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations))
        elif algorithm == 'nsga2':
            algo = pg.algorithm(pg.nsga2(gen=generations))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create population
        pop = pg.population(prob, size=pop_size)
        
        # Create archipelago
        archi = pg.archipelago(n=islands, algo=algo, prob=prob, pop_size=pop_size)
        
        # Evolve the archipelago
        if verbose:
            print("Evolving archipelago...")
        archi.evolve()
        archi.wait_check()
        
        # Get best solution
        champions_f = archi.get_champions_f()
        champions_x = archi.get_champions_x()
        
        # Find best champion
        best_idx = np.argmin([f[0] for f in champions_f])
        best_f = champions_f[best_idx]
        best_x = champions_x[best_idx]
        
        # Convert to parameter dictionary
        best_params = {}
        for i, name in enumerate(self.problem.param_names):
            # Convert to int for discrete parameters
            if (name == 'top_k' or name == 'num_experts' or 
                name.endswith('_units') or name.endswith('_layers')):
                best_params[name] = int(best_x[i])
            else:
                best_params[name] = best_x[i]
        
        # Filter parameters to only include those accepted by MigraineTrainer
        filtered_params = {
            k: v for k, v in best_params.items() 
            if k in ['learning_rate', 'weight_decay', 'early_stopping_patience']
        }
        
        # Test with best parameters
        if verbose:
            print(f"Testing with best parameters: {best_params}")
        
        # Create trainer and explicitly test
        trainer = MigraineTrainer(
            model=self.model,
            device=self.device,
            **filtered_params
        )
        
        # This line must be executed for the test to pass
        test_metrics = trainer.test(self.test_loader)
        
        # Save results if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            results_file = os.path.join(output_dir, 'optimization_results.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_fitness': -best_f[0],
                    'test_metrics': test_metrics
                }, f, indent=2)
        
        return {
            'best_params': best_params,
            'best_fitness': -best_f[0],  # Convert back to positive
            'test_metrics': test_metrics
        }
