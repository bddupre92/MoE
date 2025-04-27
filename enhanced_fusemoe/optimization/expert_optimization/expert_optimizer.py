"""
Expert Optimizer Module for Enhanced FuseMoE

This module provides functionality for optimizing expert models in the Enhanced FuseMoE
system using PyGMO's evolutionary algorithms.
"""

import torch
import numpy as np
import pygmo as pg
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimization.evolutionary_algorithms.pygmo_problem import ExpertOptimizationProblem, optimize_with_pygmo


class ExpertOptimizer:
    """
    Optimizer for FuseMoE expert models using PyGMO.
    
    This class provides methods for optimizing the hyperparameters of expert models
    in the Enhanced FuseMoE system using evolutionary algorithms from PyGMO.
    
    Attributes:
        experts (Dict[str, Any]): Dictionary of expert models to optimize
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds for optimization
        fitness_metric (str): Metric to optimize (e.g., 'auc', 'accuracy')
        num_epochs (int): Number of epochs for training
        patience (int): Patience for early stopping
        device: Device to use for training (CPU or GPU)
        optimization_results (Dict): Results of optimization
    """
    
    def __init__(self, experts: Dict[str, Any], 
                 train_loader: Any, 
                 val_loader: Any,
                 test_loader: Any,
                 param_bounds: Dict[str, Tuple[float, float]],
                 fitness_metric: str = 'auc',
                 num_epochs: int = 10,
                 patience: int = 3,
                 device: Optional[torch.device] = None,
                 seed: Optional[int] = None):
        """
        Initialize the expert optimizer.
        
        Args:
            experts: Dictionary mapping expert names to expert models
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            fitness_metric: Metric to optimize (e.g., 'auc', 'accuracy')
            num_epochs: Number of epochs for training
            patience: Patience for early stopping
            device: Device to use for training (CPU or GPU)
            seed: Random seed for reproducibility
        """
        self.experts = experts
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.param_bounds = param_bounds
        self.fitness_metric = fitness_metric
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.optimization_results = {}
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def optimize_expert(self, expert_name: str, 
                       algorithm: str = 'de', 
                       pop_size: int = 20, 
                       generations: int = 10, 
                       islands: int = 1,
                       output_dir: Optional[str] = None,
                       verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize a specific expert model's hyperparameters.
        
        Args:
            expert_name: Name of the expert to optimize
            algorithm: PyGMO algorithm to use ('de', 'pso', 'sade', etc.)
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago (parallel optimization)
            output_dir: Directory to save optimization results
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing optimization results
        """
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not found in experts dictionary")
        
        # Extract parameter names and bounds
        param_names = list(self.param_bounds.keys())
        lower_bounds = [self.param_bounds[name][0] for name in param_names]
        upper_bounds = [self.param_bounds[name][1] for name in param_names]
        bounds = (lower_bounds, upper_bounds)
        
        # Create optimization problem
        problem = ExpertOptimizationProblem(
            expert_type=expert_name,
            expert_model=self.experts[expert_name],
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            bounds=bounds,
            param_names=param_names,
            fitness_metric=self.fitness_metric,
            num_epochs=self.num_epochs,
            patience=self.patience,
            device=self.device,
            seed=self.seed
        )
        
        # Run optimization
        if verbose:
            print(f"Optimizing {expert_name} expert...")
        
        results = optimize_with_pygmo(
            problem=problem,
            algorithm=algorithm,
            pop_size=pop_size,
            generations=generations,
            islands=islands,
            seed=self.seed
        )
        
        # Save results
        self.optimization_results[expert_name] = results
        
        # Save to file if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{expert_name}_optimization_results.pt")
            torch.save(results, output_file)
            if verbose:
                print(f"Saved optimization results to {output_file}")
        
        return results
    
    def optimize_all_experts(self, 
                           algorithm: str = 'de', 
                           pop_size: int = 20, 
                           generations: int = 10, 
                           islands: int = 1,
                           output_dir: Optional[str] = None,
                           verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Optimize all expert models' hyperparameters.
        
        Args:
            algorithm: PyGMO algorithm to use ('de', 'pso', 'sade', etc.)
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago (parallel optimization)
            output_dir: Directory to save optimization results
            verbose: Whether to print progress information
            
        Returns:
            Dictionary mapping expert names to optimization results
        """
        results = {}
        
        for expert_name in self.experts:
            expert_results = self.optimize_expert(
                expert_name=expert_name,
                algorithm=algorithm,
                pop_size=pop_size,
                generations=generations,
                islands=islands,
                output_dir=output_dir,
                verbose=verbose
            )
            results[expert_name] = expert_results
        
        return results
    
    def get_best_experts(self, top_k: int = None) -> Dict[str, Any]:
        """
        Get the best expert models based on optimization results.
        
        Args:
            top_k: Number of top experts to return. If None, return all experts.
            
        Returns:
            Dictionary mapping expert names to optimized expert models
        """
        if not self.optimization_results:
            raise ValueError("No optimization results available. Run optimize_all_experts first.")
        
        # Sort experts by fitness
        sorted_experts = sorted(
            self.optimization_results.items(),
            key=lambda x: x[1]['best_fitness'],
            reverse=True
        )
        
        # Select top_k experts
        if top_k is not None:
            sorted_experts = sorted_experts[:top_k]
        
        # Create dictionary of best experts
        best_experts = {}
        for expert_name, _ in sorted_experts:
            best_experts[expert_name] = self.experts[expert_name]
        
        return best_experts


# Legacy functions for backward compatibility

def create_sleep_expert_optimizer(seed: Optional[int] = None) -> ExpertOptimizer:
    """
    Create an optimizer for sleep expert models.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        ExpertOptimizer for sleep expert models
    """
    # This function is maintained for backward compatibility
    # In the new implementation, use the ExpertOptimizer class directly
    from models.experts.sleep_expert import SleepExpert
    
    # Create a dummy expert
    expert = SleepExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
    
    # Define parameter bounds for sleep expert
    param_bounds = {
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0, 0.001),
        'dropout_rate': (0.0, 0.5),
        'hidden_dim': (32, 128)
    }
    
    # Create mock data loaders
    class MockDataLoader:
        def __init__(self, size=100):
            self.dataset = type('obj', (object,), {'__len__': lambda self: size})()
    
    train_loader = MockDataLoader(100)
    val_loader = MockDataLoader(50)
    test_loader = MockDataLoader(50)
    
    return ExpertOptimizer(
        experts={'sleep': expert},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_bounds=param_bounds,
        fitness_metric='auc',
        num_epochs=10,
        patience=3,
        device=torch.device('cpu'),
        seed=seed
    )


def create_weather_expert_optimizer(seed: Optional[int] = None) -> ExpertOptimizer:
    """
    Create an optimizer for weather expert models.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        ExpertOptimizer for weather expert models
    """
    # This function is maintained for backward compatibility
    # In the new implementation, use the ExpertOptimizer class directly
    from models.experts.weather_expert import WeatherExpert
    
    # Create a dummy expert
    expert = WeatherExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
    
    # Define parameter bounds for weather expert
    param_bounds = {
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0, 0.001),
        'dropout_rate': (0.0, 0.5),
        'hidden_dim': (32, 128)
    }
    
    # Create mock data loaders
    class MockDataLoader:
        def __init__(self, size=100):
            self.dataset = type('obj', (object,), {'__len__': lambda self: size})()
    
    train_loader = MockDataLoader(100)
    val_loader = MockDataLoader(50)
    test_loader = MockDataLoader(50)
    
    return ExpertOptimizer(
        experts={'weather': expert},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_bounds=param_bounds,
        fitness_metric='auc',
        num_epochs=10,
        patience=3,
        device=torch.device('cpu'),
        seed=seed
    )


def create_stress_diet_expert_optimizer(seed: Optional[int] = None) -> ExpertOptimizer:
    """
    Create an optimizer for stress/diet expert models.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        ExpertOptimizer for stress/diet expert models
    """
    # This function is maintained for backward compatibility
    # In the new implementation, use the ExpertOptimizer class directly
    from models.experts.stress_diet_expert import StressDietExpert
    
    # Create a dummy expert
    expert = StressDietExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
    
    # Define parameter bounds for stress/diet expert
    param_bounds = {
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0, 0.001),
        'dropout_rate': (0.0, 0.5),
        'hidden_dim': (32, 128)
    }
    
    # Create mock data loaders
    class MockDataLoader:
        def __init__(self, size=100):
            self.dataset = type('obj', (object,), {'__len__': lambda self: size})()
    
    train_loader = MockDataLoader(100)
    val_loader = MockDataLoader(50)
    test_loader = MockDataLoader(50)
    
    return ExpertOptimizer(
        experts={'stress_diet': expert},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_bounds=param_bounds,
        fitness_metric='auc',
        num_epochs=10,
        patience=3,
        device=torch.device('cpu'),
        seed=seed
    )


def create_physio_expert_optimizer(seed: Optional[int] = None) -> ExpertOptimizer:
    """
    Create an optimizer for physiological expert models.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        ExpertOptimizer for physiological expert models
    """
    # This function is maintained for backward compatibility
    # In the new implementation, use the ExpertOptimizer class directly
    from models.experts.physio_expert import PhysioExpert
    
    # Create a dummy expert
    expert = PhysioExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
    
    # Define parameter bounds for physiological expert
    param_bounds = {
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0, 0.001),
        'dropout_rate': (0.0, 0.5),
        'hidden_dim': (32, 128)
    }
    
    # Create mock data loaders
    class MockDataLoader:
        def __init__(self, size=100):
            self.dataset = type('obj', (object,), {'__len__': lambda self: size})()
    
    train_loader = MockDataLoader(100)
    val_loader = MockDataLoader(50)
    test_loader = MockDataLoader(50)
    
    return ExpertOptimizer(
        experts={'physio': expert},
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        param_bounds=param_bounds,
        fitness_metric='auc',
        num_epochs=10,
        patience=3,
        device=torch.device('cpu'),
        seed=seed
    )
