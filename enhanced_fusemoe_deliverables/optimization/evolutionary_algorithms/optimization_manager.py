"""
End-to-End Optimization Module for Enhanced FuseMoE

This module provides functionality for end-to-end optimization of the Enhanced FuseMoE
system using PyGMO's evolutionary algorithms.
"""

import torch
import numpy as np
import pygmo as pg
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimization.evolutionary_algorithms.pygmo_problem import EndToEndOptimizationProblem, optimize_with_pygmo


class EndToEndOptimizer:
    """
    End-to-end optimizer for FuseMoE models using PyGMO.
    
    This class provides methods for optimizing the hyperparameters of the entire
    FuseMoE system using evolutionary algorithms from PyGMO.
    
    Attributes:
        create_model_fn (Callable): Function to create full model with given hyperparameters
        train_model_fn (Callable): Function to train full model
        evaluate_model_fn (Callable): Function to evaluate full model
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds for optimization
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, 
                 create_model_fn: Callable, 
                 train_model_fn: Callable, 
                 evaluate_model_fn: Callable,
                 param_bounds: Dict[str, Tuple[float, float]],
                 seed: Optional[int] = None):
        """
        Initialize the end-to-end optimizer.
        
        Args:
            create_model_fn: Function to create full model with given hyperparameters
            train_model_fn: Function to train full model
            evaluate_model_fn: Function to evaluate full model
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            seed: Random seed for reproducibility
        """
        self.create_model_fn = create_model_fn
        self.train_model_fn = train_model_fn
        self.evaluate_model_fn = evaluate_model_fn
        self.param_bounds = param_bounds
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def optimize(self, train_data: Any, val_data: Any, 
                algorithm: str = 'nsga2', pop_size: int = 20, 
                generations: int = 10, islands: int = 1) -> Dict[str, Any]:
        """
        Optimize model hyperparameters end-to-end.
        
        Args:
            train_data: Training data
            val_data: Validation data
            algorithm: PyGMO algorithm to use (default: 'nsga2' for multi-objective)
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago (parallel optimization)
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract parameter names and bounds
        param_names = list(self.param_bounds.keys())
        lower_bounds = [self.param_bounds[name][0] for name in param_names]
        upper_bounds = [self.param_bounds[name][1] for name in param_names]
        bounds = (lower_bounds, upper_bounds)
        
        # Create optimization problem
        problem = EndToEndOptimizationProblem(
            train_data=train_data,
            val_data=val_data,
            create_model_fn=self.create_model_fn,
            train_model_fn=self.train_model_fn,
            evaluate_model_fn=self.evaluate_model_fn,
            bounds=bounds,
            param_names=param_names,
            seed=self.seed
        )
        
        # Run optimization
        results = optimize_with_pygmo(
            problem=problem,
            algorithm=algorithm,
            pop_size=pop_size,
            generations=generations,
            islands=islands,
            seed=self.seed
        )
        
        return results
    
    def create_optimized_model(self, best_params: Dict[str, Any]) -> Any:
        """
        Create model with optimized hyperparameters.
        
        Args:
            best_params: Dictionary of optimized hyperparameters
            
        Returns:
            Optimized model
        """
        return self.create_model_fn(**best_params)


def create_migraine_end_to_end_optimizer(seed: Optional[int] = None) -> EndToEndOptimizer:
    """
    Create an end-to-end optimizer for migraine prediction models.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        EndToEndOptimizer for migraine prediction models
    """
    # Define parameter bounds for end-to-end optimization
    param_bounds = {
        # Expert parameters
        'num_experts': (4, 16),                # Number of experts
        'expert_hidden_dim': (32, 256),        # Expert hidden layer size
        'expert_dropout': (0.0, 0.5),          # Expert dropout rate
        
        # Gating parameters
        'top_k': (1, 4),                       # Number of experts to route to
        'gating_hidden_dim': (32, 256),        # Gating hidden layer size
        'load_balance_coef': (0.001, 0.1),     # Load balancing coefficient
        'noisy_gating': (0, 1),                # Whether to use noisy gating (0=False, 1=True)
        
        # Training parameters
        'learning_rate': (0.0001, 0.01),       # Learning rate
        'batch_size': (16, 256),               # Batch size
        'weight_decay': (0.0, 0.01)            # Weight decay
    }
    
    # Define functions for creating, training, and evaluating migraine model
    def create_migraine_model(num_experts, expert_hidden_dim, expert_dropout,
                             top_k, gating_hidden_dim, load_balance_coef, noisy_gating,
                             learning_rate, batch_size, weight_decay):
        # This would be replaced with actual implementation
        # that creates a full migraine prediction model with the given hyperparameters
        pass
    
    def train_migraine_model(model, train_data):
        # This would be replaced with actual implementation
        # that trains the migraine prediction model
        pass
    
    def evaluate_migraine_model(model, val_data):
        # This would be replaced with actual implementation
        # that evaluates the migraine prediction model
        # and returns metrics like AUC, accuracy, etc.
        pass
    
    return EndToEndOptimizer(
        create_model_fn=create_migraine_model,
        train_model_fn=train_migraine_model,
        evaluate_model_fn=evaluate_migraine_model,
        param_bounds=param_bounds,
        seed=seed
    )


class OptimizationManager:
    """
    Manager for coordinating different optimization strategies for Enhanced FuseMoE.
    
    This class provides methods for coordinating different optimization strategies,
    including expert-level optimization, gating optimization, and end-to-end optimization.
    
    Attributes:
        expert_optimizers (Dict[str, Any]): Dictionary of expert optimizers
        gating_optimizer (Any): Gating optimizer
        end_to_end_optimizer (Any): End-to-end optimizer
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, expert_types: List[str], seed: Optional[int] = None):
        """
        Initialize the optimization manager.
        
        Args:
            expert_types: List of expert types to optimize
            seed: Random seed for reproducibility
        """
        self.seed = seed
        
        # Create expert optimizers
        self.expert_optimizers = {}
        for expert_type in expert_types:
            if expert_type == 'sleep':
                self.expert_optimizers[expert_type] = create_sleep_expert_optimizer(seed)
            elif expert_type == 'weather':
                self.expert_optimizers[expert_type] = create_weather_expert_optimizer(seed)
            elif expert_type == 'stress_diet':
                self.expert_optimizers[expert_type] = create_stress_diet_expert_optimizer(seed)
            elif expert_type == 'physio':
                self.expert_optimizers[expert_type] = create_physio_expert_optimizer(seed)
            else:
                raise ValueError(f"Unsupported expert type: {expert_type}")
        
        # Create gating optimizer
        self.gating_optimizer = create_migraine_gating_optimizer(
            num_experts=len(expert_types),
            num_modalities=len(expert_types),
            seed=seed
        )
        
        # Create end-to-end optimizer
        self.end_to_end_optimizer = create_migraine_end_to_end_optimizer(seed)
    
    def optimize_experts(self, train_data: Dict[str, Any], val_data: Dict[str, Any],
                        algorithm: str = 'de', pop_size: int = 20,
                        generations: int = 10, islands: int = 1) -> Dict[str, Any]:
        """
        Optimize expert models individually.
        
        Args:
            train_data: Dictionary mapping expert types to training data
            val_data: Dictionary mapping expert types to validation data
            algorithm: PyGMO algorithm to use
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            
        Returns:
            Dictionary mapping expert types to optimization results
        """
        results = {}
        
        for expert_type, optimizer in self.expert_optimizers.items():
            print(f"Optimizing {expert_type} expert...")
            expert_train_data = train_data.get(expert_type)
            expert_val_data = val_data.get(expert_type)
            
            if expert_train_data is None or expert_val_data is None:
                print(f"Warning: No data found for {expert_type} expert. Skipping optimization.")
                continue
            
            results[expert_type] = optimizer.optimize(
                train_data=expert_train_data,
                val_data=expert_val_data,
                algorithm=algorithm,
                pop_size=pop_size,
                generations=generations,
                islands=islands
            )
        
        return results
    
    def optimize_gating(self, train_data: Any, val_data: Any,
                       algorithm: str = 'de', pop_size: int = 20,
                       generations: int = 10, islands: int = 1) -> Dict[str, Any]:
        """
        Optimize gating network.
        
        Args:
            train_data: Training data
            val_data: Validation data
            algorithm: PyGMO algorithm to use
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            
        Returns:
            Optimization results
        """
        print("Optimizing gating network...")
        
        return self.gating_optimizer.optimize(
            train_data=train_data,
            val_data=val_data,
            algorithm=algorithm,
            pop_size=pop_size,
            generations=generations,
            islands=islands
        )
    
    def optimize_end_to_end(self, train_data: Any, val_data: Any,
                           algorithm: str = 'nsga2', pop_size: int = 20,
                           generations: int = 10, islands: int = 1) -> Dict[str, Any]:
        """
        Optimize model end-to-end.
        
        Args:
            train_data: Training data
            val_data: Validation data
            algorithm: PyGMO algorithm to use
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            
        Returns:
            Optimization results
        """
        print("Optimizing model end-to-end...")
        
        return self.end_to_end_optimizer.optimize(
            train_data=train_data,
            val_data=val_data,
            algorithm=algorithm,
            pop_size=pop_size,
            generations=generations,
            islands=islands
        )
    
    def create_optimized_model(self, expert_results: Dict[str, Any], gating_results: Dict[str, Any]) -> Any:
        """
        Create optimized model from individual optimization results.
        
        Args:
            expert_results: Dictionary mapping expert types to optimization results
            gating_results: Gating optimization results
            
        Returns:
            Optimized model
        """
        # Create optimized experts
        optimized_experts = {}
        for expert_type, results in expert_results.items():
            optimized_experts[expert_type] = self.expert_optimizers[expert_type].create_optimized_expert(
                results['best_params']
            )
        
        # Create optimized gating
        optimized_gating = self.gating_optimizer.create_optimized_gating(
            gating_results['best_params']
        )
        
        # This would be replaced with actual implementation
        # that creates a full model with the optimized experts and gating
        return None


# Import expert optimizers for convenience
from optimization.expert_optimization.expert_optimizer import (
    create_sleep_expert_optimizer,
    create_weather_expert_optimizer,
    create_stress_diet_expert_optimizer,
    create_physio_expert_optimizer
)
