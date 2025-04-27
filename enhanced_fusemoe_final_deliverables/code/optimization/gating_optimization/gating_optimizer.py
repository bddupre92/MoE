"""
Gating Optimizer Module for Enhanced FuseMoE

This module provides functionality for optimizing gating networks in the Enhanced FuseMoE
system using PyGMO's evolutionary algorithms.
"""

import torch
import numpy as np
import pygmo as pg
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from optimization.evolutionary_algorithms.pygmo_problem import GatingOptimizationProblem, optimize_with_pygmo


class GatingOptimizer:
    """
    Optimizer for FuseMoE gating networks using PyGMO.
    
    This class provides methods for optimizing the hyperparameters of gating networks
    in the Enhanced FuseMoE system using evolutionary algorithms from PyGMO.
    
    Attributes:
        gating_network: The gating network to optimize
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
    
    def __init__(self, gating_network: Any, 
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
        Initialize the gating optimizer.
        
        Args:
            gating_network: The gating network to optimize
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
        self.gating_network = gating_network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.param_bounds = param_bounds
        self.fitness_metric = fitness_metric
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.optimization_results = None
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def optimize(self, algorithm: str = 'de', 
                pop_size: int = 20, 
                generations: int = 10, 
                islands: int = 1,
                output_dir: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize gating network hyperparameters.
        
        Args:
            algorithm: PyGMO algorithm to use ('de', 'pso', 'sade', etc.)
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago (parallel optimization)
            output_dir: Directory to save optimization results
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract parameter names and bounds
        param_names = list(self.param_bounds.keys())
        lower_bounds = [self.param_bounds[name][0] for name in param_names]
        upper_bounds = [self.param_bounds[name][1] for name in param_names]
        bounds = (lower_bounds, upper_bounds)
        
        # Create optimization problem
        problem = GatingOptimizationProblem(
            gating_network=self.gating_network,
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
            print(f"Optimizing gating network...")
        
        results = optimize_with_pygmo(
            problem=problem,
            algorithm=algorithm,
            pop_size=pop_size,
            generations=generations,
            islands=islands,
            seed=self.seed
        )
        
        # Save results
        self.optimization_results = results
        
        # Save to file if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "gating_optimization_results.pt")
            torch.save(results, output_file)
            if verbose:
                print(f"Saved optimization results to {output_file}")
        
        return results
    
    def get_optimized_gating_network(self) -> Any:
        """
        Get the gating network with optimized hyperparameters.
        
        Returns:
            Optimized gating network
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available. Run optimize first.")
        
        # Get best parameters
        best_params = self.optimization_results['best_params']
        
        # Create a new gating network with the best parameters
        # This assumes the gating network class has a constructor that accepts these parameters
        gating_class = self.gating_network.__class__
        
        # Get the current parameters from the gating network
        current_params = {
            key: getattr(self.gating_network, key)
            for key in best_params.keys()
            if hasattr(self.gating_network, key)
        }
        
        # Update with best parameters
        current_params.update(best_params)
        
        # Create a new gating network with the best parameters
        optimized_gating = gating_class(**current_params)
        
        return optimized_gating
    
    def analyze_gating_decisions(self, data_loader: Any, 
                               num_samples: int = 100,
                               feature_names: Optional[List[str]] = None,
                               expert_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze gating network decisions on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset to analyze
            num_samples: Number of samples to analyze
            feature_names: Names of input features
            expert_names: Names of expert models
            
        Returns:
            Dictionary containing analysis results
        """
        # Move gating network to device
        self.gating_network.to(self.device)
        
        # Set gating network to evaluation mode
        self.gating_network.eval()
        
        # Initialize counters
        expert_selection_count = {}
        co_selection_count = np.zeros((self.gating_network.num_experts, self.gating_network.num_experts))
        
        # Initialize feature importance
        input_feature_importance = {}
        
        # Generate default feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.gating_network.input_dim)]
        
        # Generate default expert names if not provided
        if expert_names is None:
            expert_names = [f"expert_{i}" for i in range(self.gating_network.num_experts)]
        
        # Initialize feature importance dictionary
        for feature in feature_names:
            input_feature_importance[feature] = 0.0
        
        # Initialize expert selection frequency dictionary
        for expert in expert_names:
            expert_selection_count[expert] = 0
        
        # Collect samples
        samples_processed = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Extract inputs and targets
                inputs, _ = batch
                
                # Move inputs to device
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                elif isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass through gating network
                gates, _ = self.gating_network(inputs)
                
                # Get top-k experts for each sample
                _, top_indices = torch.topk(gates, self.gating_network.top_k, dim=1)
                
                # Update expert selection count
                for i in range(top_indices.shape[0]):
                    for j in range(self.gating_network.top_k):
                        expert_idx = top_indices[i, j].item()
                        expert_selection_count[expert_names[expert_idx]] += 1
                
                # Update co-selection count
                for i in range(top_indices.shape[0]):
                    for j in range(self.gating_network.top_k):
                        for k in range(j+1, self.gating_network.top_k):
                            expert1 = top_indices[i, j].item()
                            expert2 = top_indices[i, k].item()
                            co_selection_count[expert1, expert2] += 1
                            co_selection_count[expert2, expert1] += 1
                
                # Update samples processed
                samples_processed += inputs.shape[0] if isinstance(inputs, torch.Tensor) else len(next(iter(inputs.values())))
                
                # Break if we've processed enough samples
                if samples_processed >= num_samples:
                    break
        
        # Normalize expert selection count
        total_selections = sum(expert_selection_count.values())
        expert_selection_frequency = {
            expert: count / total_selections
            for expert, count in expert_selection_count.items()
        }
        
        # Normalize co-selection count
        total_co_selections = co_selection_count.sum()
        if total_co_selections > 0:
            co_selection_matrix = co_selection_count / total_co_selections
        else:
            co_selection_matrix = co_selection_count
        
        # Calculate feature importance (placeholder implementation)
        # In a real implementation, this would use techniques like SHAP or permutation importance
        for i, feature in enumerate(feature_names):
            input_feature_importance[feature] = np.random.uniform(0.05, 0.2)  # Placeholder
        
        # Normalize feature importance
        total_importance = sum(input_feature_importance.values())
        input_feature_importance = {
            feature: importance / total_importance
            for feature, importance in input_feature_importance.items()
        }
        
        # Return analysis results
        return {
            'expert_selection_frequency': expert_selection_frequency,
            'co_selection_matrix': co_selection_matrix,
            'input_feature_importance': input_feature_importance
        }


# Legacy function for backward compatibility

def create_migraine_gating_optimizer(num_experts: int = 4, num_modalities: int = 4, seed: Optional[int] = None) -> GatingOptimizer:
    """
    Create an optimizer for migraine gating networks.
    
    Args:
        num_experts: Number of expert models
        num_modalities: Number of data modalities
        seed: Random seed for reproducibility
        
    Returns:
        GatingOptimizer for migraine gating networks
    """
    # This function is maintained for backward compatibility
    # In the new implementation, use the GatingOptimizer class directly
    from models.gating.migraine_gating import MigraineGating
    
    # Create a dummy gating network
    gating_network = MigraineGating(
        input_dim=10,
        num_experts=num_experts,
        hidden_dim=64,
        top_k=2,
        dropout_rate=0.2
    )
    
    # Define parameter bounds for migraine gating
    param_bounds = {
        'learning_rate': (0.0001, 0.01),
        'weight_decay': (0.0, 0.001),
        'top_k': (1, num_experts),
        'hidden_dim': (32, 128),
        'dropout_rate': (0.0, 0.5)
    }
    
    # Create mock data loaders
    class MockDataLoader:
        def __init__(self, size=100):
            self.dataset = type('obj', (object,), {'__len__': lambda self: size})()
    
    train_loader = MockDataLoader(100)
    val_loader = MockDataLoader(50)
    test_loader = MockDataLoader(50)
    
    return GatingOptimizer(
        gating_network=gating_network,
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
