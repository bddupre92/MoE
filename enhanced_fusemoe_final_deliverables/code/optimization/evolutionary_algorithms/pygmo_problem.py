"""
PyGMO Problem Formulation for FuseMoE Optimization

This module provides base classes and implementations for defining optimization problems
using PyGMO (Python Parallel Global Multiobjective Optimizer) for the Enhanced FuseMoE system.
"""

import numpy as np
import pygmo as pg
import torch
from typing import List, Dict, Tuple, Any, Optional, Union, Callable


class BasePyGMOProblem:
    """
    Base class for PyGMO optimization problems for FuseMoE.
    
    This class provides the foundation for defining optimization problems
    that can be solved using PyGMO's evolutionary algorithms.
    
    Attributes:
        bounds (Tuple[List[float], List[float]]): Lower and upper bounds for parameters
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, bounds: Tuple[List[float], List[float]], seed: Optional[int] = None):
        """
        Initialize the base PyGMO problem.
        
        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) for parameters
            seed: Random seed for reproducibility
        """
        self.bounds = bounds
        self.seed = seed
        
        # Validate bounds
        if len(bounds[0]) != len(bounds[1]):
            raise ValueError("Lower and upper bounds must have the same length")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
    
    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """
        Get the bounds of the problem.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.bounds
    
    def get_name(self) -> str:
        """
        Get the name of the problem.
        
        Returns:
            Name of the problem
        """
        return self.__class__.__name__
    
    def get_extra_info(self) -> str:
        """
        Get extra information about the problem.
        
        Returns:
            String with extra information
        """
        return f"Dimensions: {len(self.bounds[0])}, Seed: {self.seed}"
    
    def fitness(self, x: List[float]) -> List[float]:
        """
        Compute the fitness of a solution.
        
        This method must be implemented by subclasses.
        
        Args:
            x: Solution vector
            
        Returns:
            List of fitness values (objectives)
        """
        raise NotImplementedError("Subclasses must implement fitness method")
    
    def get_nobj(self) -> int:
        """
        Get the number of objectives.
        
        Returns:
            Number of objectives
        """
        return 1  # Default is single-objective, override for multi-objective problems


class ExpertOptimizationProblem(BasePyGMOProblem):
    """
    PyGMO problem for optimizing expert model hyperparameters.
    
    This class defines an optimization problem for tuning the hyperparameters
    of expert models in the FuseMoE architecture.
    
    Attributes:
        expert_type (str): Type of expert model to optimize
        train_data (Any): Training data
        val_data (Any): Validation data
        create_expert_fn (Callable): Function to create expert model with given hyperparameters
        train_expert_fn (Callable): Function to train expert model
        evaluate_expert_fn (Callable): Function to evaluate expert model
        bounds (Tuple[List[float], List[float]]): Lower and upper bounds for parameters
        seed (int): Random seed for reproducibility
        param_names (List[str]): Names of parameters being optimized
    """
    
    def __init__(self, expert_type: str, train_data: Any, val_data: Any,
                 create_expert_fn: Callable, train_expert_fn: Callable, evaluate_expert_fn: Callable,
                 bounds: Tuple[List[float], List[float]], param_names: List[str],
                 seed: Optional[int] = None):
        """
        Initialize the expert optimization problem.
        
        Args:
            expert_type: Type of expert model to optimize
            train_data: Training data
            val_data: Validation data
            create_expert_fn: Function to create expert model with given hyperparameters
            train_expert_fn: Function to train expert model
            evaluate_expert_fn: Function to evaluate expert model
            bounds: Tuple of (lower_bounds, upper_bounds) for parameters
            param_names: Names of parameters being optimized
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.expert_type = expert_type
        self.train_data = train_data
        self.val_data = val_data
        self.create_expert_fn = create_expert_fn
        self.train_expert_fn = train_expert_fn
        self.evaluate_expert_fn = evaluate_expert_fn
        self.param_names = param_names
        
        # Validate param_names
        if len(param_names) != len(bounds[0]):
            raise ValueError("Number of parameter names must match number of bounds")
    
    def fitness(self, x: List[float]) -> List[float]:
        """
        Compute the fitness of a solution.
        
        Args:
            x: Solution vector representing hyperparameters
            
        Returns:
            List containing negative validation metric (to be minimized)
        """
        # Convert parameters to appropriate types and create dictionary
        params = {}
        for i, name in enumerate(self.param_names):
            # Convert to int for discrete parameters
            if name.endswith('_units') or name.endswith('_layers') or name.endswith('_size'):
                params[name] = int(x[i])
            else:
                params[name] = x[i]
        
        try:
            # Create expert model with given hyperparameters
            expert = self.create_expert_fn(**params)
            
            # Train expert model
            self.train_expert_fn(expert, self.train_data)
            
            # Evaluate expert model
            metrics = self.evaluate_expert_fn(expert, self.val_data)
            
            # Return negative metric (we want to maximize metrics, but PyGMO minimizes)
            return [-metrics['auc']]  # Use AUC as primary metric
        except Exception as e:
            # Return worst possible fitness in case of error
            print(f"Error in fitness evaluation: {e}")
            return [float('inf')]
    
    def get_name(self) -> str:
        """
        Get the name of the problem.
        
        Returns:
            Name of the problem
        """
        return f"ExpertOptimization_{self.expert_type}"
    
    def get_extra_info(self) -> str:
        """
        Get extra information about the problem.
        
        Returns:
            String with extra information
        """
        return (f"Expert Type: {self.expert_type}, "
                f"Parameters: {', '.join(self.param_names)}, "
                f"Dimensions: {len(self.bounds[0])}, "
                f"Seed: {self.seed}")


class GatingOptimizationProblem(BasePyGMOProblem):
    """
    PyGMO problem for optimizing gating network hyperparameters.
    
    This class defines an optimization problem for tuning the hyperparameters
    of the gating network in the FuseMoE architecture.
    
    Attributes:
        train_data (Any): Training data
        val_data (Any): Validation data
        create_gating_fn (Callable): Function to create gating network with given hyperparameters
        train_model_fn (Callable): Function to train full model with gating network
        evaluate_model_fn (Callable): Function to evaluate full model
        bounds (Tuple[List[float], List[float]]): Lower and upper bounds for parameters
        seed (int): Random seed for reproducibility
        param_names (List[str]): Names of parameters being optimized
    """
    
    def __init__(self, train_data: Any, val_data: Any,
                 create_gating_fn: Callable, train_model_fn: Callable, evaluate_model_fn: Callable,
                 bounds: Tuple[List[float], List[float]], param_names: List[str],
                 seed: Optional[int] = None):
        """
        Initialize the gating optimization problem.
        
        Args:
            train_data: Training data
            val_data: Validation data
            create_gating_fn: Function to create gating network with given hyperparameters
            train_model_fn: Function to train full model with gating network
            evaluate_model_fn: Function to evaluate full model
            bounds: Tuple of (lower_bounds, upper_bounds) for parameters
            param_names: Names of parameters being optimized
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.train_data = train_data
        self.val_data = val_data
        self.create_gating_fn = create_gating_fn
        self.train_model_fn = train_model_fn
        self.evaluate_model_fn = evaluate_model_fn
        self.param_names = param_names
        
        # Validate param_names
        if len(param_names) != len(bounds[0]):
            raise ValueError("Number of parameter names must match number of bounds")
    
    def fitness(self, x: List[float]) -> List[float]:
        """
        Compute the fitness of a solution.
        
        Args:
            x: Solution vector representing hyperparameters
            
        Returns:
            List containing negative validation metric (to be minimized)
        """
        # Convert parameters to appropriate types and create dictionary
        params = {}
        for i, name in enumerate(self.param_names):
            # Convert to int for discrete parameters
            if name == 'top_k' or name.endswith('_units') or name.endswith('_layers'):
                params[name] = int(x[i])
            else:
                params[name] = x[i]
        
        try:
            # Create gating network with given hyperparameters
            gating = self.create_gating_fn(**params)
            
            # Train full model with this gating
            model = self.train_model_fn(gating, self.train_data)
            
            # Evaluate full model
            metrics = self.evaluate_model_fn(model, self.val_data)
            
            # Return negative metric (we want to maximize metrics, but PyGMO minimizes)
            return [-metrics['auc']]  # Use AUC as primary metric
        except Exception as e:
            # Return worst possible fitness in case of error
            print(f"Error in fitness evaluation: {e}")
            return [float('inf')]


class EndToEndOptimizationProblem(BasePyGMOProblem):
    """
    PyGMO problem for end-to-end optimization of the FuseMoE model.
    
    This class defines a multi-objective optimization problem for tuning
    both expert and gating network hyperparameters simultaneously.
    
    Attributes:
        model (Any): Model to optimize
        train_loader (Any): Training data loader
        val_loader (Any): Validation data loader
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds
        fitness_metric (str): Metric to optimize
        num_epochs (int): Number of training epochs
        patience (int): Early stopping patience
        device (torch.device): Device to use for training
    """
    
    def __init__(self, model: Any, train_loader: Any, val_loader: Any,
                 param_bounds: Dict[str, Tuple[float, float]], fitness_metric: str = 'auc',
                 num_epochs: int = 10, patience: int = 3, device: Optional[torch.device] = None):
        """
        Initialize the end-to-end optimization problem.
        
        Args:
            model: Model to optimize
            train_loader: Training data loader
            val_loader: Validation data loader
            param_bounds: Dictionary mapping parameter names to (min, max) tuples
            fitness_metric: Metric to optimize
            num_epochs: Number of training epochs
            patience: Early stopping patience
            device: Device to use for training
        """
        # Extract parameter names and bounds
        param_names = list(param_bounds.keys())
        lb = [param_bounds[name][0] for name in param_names]
        ub = [param_bounds[name][1] for name in param_names]
        bounds = (lb, ub)
        
        super().__init__(bounds, None)
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.param_bounds = param_bounds
        self.param_names = param_names
        self.fitness_metric = fitness_metric
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize best parameters and fitness
        self.best_params = None
        self.best_fitness = float('-inf')
        
        # Validate parameter bounds
        self.dim = len(param_names)
    
    def fitness(self, x: List[float]) -> List[float]:
        """
        Compute the fitness of a solution.
        
        Args:
            x: Solution vector representing hyperparameters
            
        Returns:
            List containing [negative validation metric]
        """
        # Import here to avoid circular imports
        from utils.training_pipeline import MigraineTrainer
        
        # Convert parameters to appropriate types and create dictionary
        params = {}
        for i, name in enumerate(self.param_names):
            # Convert to int for discrete parameters
            if (name == 'top_k' or name == 'num_experts' or 
                name.endswith('_units') or name.endswith('_layers')):
                params[name] = int(x[i])
            else:
                params[name] = x[i]
        
        try:
            # Create trainer with current model
            trainer = MigraineTrainer(
                model=self.model,
                device=self.device,
                **params
            )
            
            # Train model
            history = trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                num_epochs=self.num_epochs,
                patience=self.patience
            )
            
            # Get best validation metric
            best_val_metric = max(epoch_metrics[self.fitness_metric] for epoch_metrics in history['val_history'])
            
            # Update best parameters and fitness if this is better
            if best_val_metric > self.best_fitness:
                self.best_fitness = best_val_metric
                self.best_params = params.copy()
            
            # Return negative metric (we want to maximize metrics, but PyGMO minimizes)
            return [-best_val_metric]
        except Exception as e:
            # Return worst possible fitness in case of error
            print(f"Error in fitness evaluation: {e}")
            return [float('inf')]
    
    def get_name(self) -> str:
        """
        Get the name of the problem.
        
        Returns:
            Name of the problem
        """
        return "End-to-End Optimization Problem"
    
    def get_extra_info(self) -> str:
        """
        Get extra information about the problem.
        
        Returns:
            String with extra information
        """
        info = f"Dimensions: {self.dim}, Metric: {self.fitness_metric}\n"
        
        if self.best_params is not None:
            info += "Best parameters:\n"
            for name, value in self.best_params.items():
                info += f"  {name}: {value}\n"
            info += f"Best fitness: {self.best_fitness}"
        else:
            info += "No optimization performed yet."
        
        return info


def optimize_with_pygmo(problem: Any, algorithm: str = 'sade', pop_size: int = 10, 
                       generations: int = 5, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Optimize a problem using PyGMO.
    
    Args:
        problem: PyGMO problem
        algorithm: PyGMO algorithm to use
        pop_size: Population size for evolutionary algorithm
        generations: Number of generations to evolve
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing optimization results
    """
    # Create PyGMO problem
    prob = pg.problem(problem)
    
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
    
    # Set seed if provided
    if seed is not None:
        algo.set_seed(seed)
    
    # Create population
    pop = pg.population(prob, size=pop_size)
    if seed is not None:
        pop.set_seed(seed)
    
    # Evolve the population
    pop = algo.evolve(pop)
    
    # Get best solution
    best_idx = pop.best_idx()
    best_x = pop.get_x()[best_idx]
    best_f = pop.get_f()[best_idx]
    
    # Convert to parameter dictionary
    if hasattr(problem, 'param_names'):
        best_params = {}
        for i, name in enumerate(problem.param_names):
            # Convert to int for discrete parameters
            if (name == 'top_k' or name == 'num_experts' or 
                name.endswith('_units') or name.endswith('_layers')):
                best_params[name] = int(best_x[i])
            else:
                best_params[name] = best_x[i]
    else:
        best_params = best_x
    
    return {
        'best_params': best_params,
        'best_fitness': -best_f[0],  # Convert back to positive
        'best_x': best_x,
        'best_f': best_f,
        'population': pop
    }
