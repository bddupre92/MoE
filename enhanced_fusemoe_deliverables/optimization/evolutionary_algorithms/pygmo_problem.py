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
        train_data (Any): Training data
        val_data (Any): Validation data
        create_model_fn (Callable): Function to create full model with given hyperparameters
        train_model_fn (Callable): Function to train full model
        evaluate_model_fn (Callable): Function to evaluate full model
        bounds (Tuple[List[float], List[float]]): Lower and upper bounds for parameters
        seed (int): Random seed for reproducibility
        param_names (List[str]): Names of parameters being optimized
    """
    
    def __init__(self, train_data: Any, val_data: Any,
                 create_model_fn: Callable, train_model_fn: Callable, evaluate_model_fn: Callable,
                 bounds: Tuple[List[float], List[float]], param_names: List[str],
                 seed: Optional[int] = None):
        """
        Initialize the end-to-end optimization problem.
        
        Args:
            train_data: Training data
            val_data: Validation data
            create_model_fn: Function to create full model with given hyperparameters
            train_model_fn: Function to train full model
            evaluate_model_fn: Function to evaluate full model
            bounds: Tuple of (lower_bounds, upper_bounds) for parameters
            param_names: Names of parameters being optimized
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.train_data = train_data
        self.val_data = val_data
        self.create_model_fn = create_model_fn
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
            List containing [negative validation metric, model complexity]
        """
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
            # Create full model with given hyperparameters
            model = self.create_model_fn(**params)
            
            # Calculate model complexity (number of parameters)
            num_params = sum(p.numel() for p in model.parameters())
            
            # Train full model
            self.train_model_fn(model, self.train_data)
            
            # Evaluate full model
            metrics = self.evaluate_model_fn(model, self.val_data)
            
            # Return multi-objective fitness: [negative metric, complexity]
            # We want to maximize metrics and minimize complexity
            return [-metrics['auc'], num_params / 1e6]  # Normalize complexity to millions of parameters
        except Exception as e:
            # Return worst possible fitness in case of error
            print(f"Error in fitness evaluation: {e}")
            return [float('inf'), float('inf')]
    
    def get_nobj(self) -> int:
        """
        Get the number of objectives.
        
        Returns:
            Number of objectives
        """
        return 2  # Multi-objective: performance and complexity


def optimize_with_pygmo(problem: BasePyGMOProblem, algorithm: str = 'de', 
                       pop_size: int = 20, generations: int = 10, 
                       islands: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Optimize a problem using PyGMO.
    
    Args:
        problem: PyGMO problem to optimize
        algorithm: Algorithm to use ('de', 'pso', 'sade', 'nsga2', etc.)
        pop_size: Population size
        generations: Number of generations
        islands: Number of islands for archipelago (parallel optimization)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing optimization results
    """
    # Create PyGMO problem
    prob = pg.problem(problem)
    
    # Create algorithm
    if algorithm == 'de':
        algo = pg.algorithm(pg.de(gen=generations, seed=seed))
    elif algorithm == 'pso':
        algo = pg.algorithm(pg.pso(gen=generations, seed=seed))
    elif algorithm == 'sade':
        algo = pg.algorithm(pg.sade(gen=generations, seed=seed))
    elif algorithm == 'nsga2':
        algo = pg.algorithm(pg.nsga2(gen=generations, seed=seed))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    if islands == 1:
        # Single island optimization
        pop = pg.population(prob, size=pop_size, seed=seed)
        pop = algo.evolve(pop)
        
        # Get best solution
        if problem.get_nobj() == 1:
            # Single-objective
            best_idx = pop.best_idx()
            best_x = pop.get_x()[best_idx]
            best_f = pop.get_f()[best_idx][0]
            
            # Convert parameters to dictionary
            best_params = {}
            for i, name in enumerate(problem.param_names):
                # Convert to int for discrete parameters
                if (name == 'top_k' or name == 'num_experts' or 
                    name.endswith('_units') or name.endswith('_layers')):
                    best_params[name] = int(best_x[i])
                else:
                    best_params[name] = best_x[i]
            
            return {
                'best_params': best_params,
                'best_fitness': -best_f,  # Convert back to positive metric
                'population': pop
            }
        else:
            # Multi-objective - return Pareto front
            pareto_front = pg.non_dominated_front_2d(pop.get_f())
            pareto_x = [pop.get_x()[i] for i in pareto_front]
            pareto_f = [pop.get_f()[i] for i in pareto_front]
            
            # Convert parameters to dictionary for each Pareto solution
            pareto_params = []
            for x in pareto_x:
                params = {}
                for i, name in enumerate(problem.param_names):
                    # Convert to int for discrete parameters
                    if (name == 'top_k' or name == 'num_experts' or 
                        name.endswith('_units') or name.endswith('_layers')):
                        params[name] = int(x[i])
                    else:
                        params[name] = x[i]
                pareto_params.append(params)
            
            return {
                'pareto_params': pareto_params,
                'pareto_fitness': [[-f[0], f[1]] for f in pareto_f],  # Convert first objective back to positive
                'population': pop
            }
    else:
        # Multi-island optimization
        archi = pg.archipelago(n=islands, algo=algo, prob=prob, pop_size=pop_size, seed=seed)
        archi.evolve()
        archi.wait()
        
        # Collect results from all islands
        all_pops = [isl.get_population() for isl in archi]
        
        if problem.get_nobj() == 1:
            # Single-objective - find best across all islands
            best_f = float('inf')
            best_x = None
            best_pop = None
            
            for pop in all_pops:
                idx = pop.best_idx()
                f = pop.get_f()[idx][0]
                if f < best_f:
                    best_f = f
                    best_x = pop.get_x()[idx]
                    best_pop = pop
            
            # Convert parameters to dictionary
            best_params = {}
            for i, name in enumerate(problem.param_names):
                # Convert to int for discrete parameters
                if (name == 'top_k' or name == 'num_experts' or 
                    name.endswith('_units') or name.endswith('_layers')):
                    best_params[name] = int(best_x[i])
                else:
                    best_params[name] = best_x[i]
            
            return {
                'best_params': best_params,
                'best_fitness': -best_f,  # Convert back to positive metric
                'archipelago': archi
            }
        else:
            # Multi-objective - combine populations and extract Pareto front
            combined_f = np.vstack([pop.get_f() for pop in all_pops])
            combined_x = np.vstack([pop.get_x() for pop in all_pops])
            
            # Extract Pareto front
            pareto_front = pg.non_dominated_front_2d(combined_f)
            pareto_x = [combined_x[i] for i in pareto_front]
            pareto_f = [combined_f[i] for i in pareto_front]
            
            # Convert parameters to dictionary for each Pareto solution
            pareto_params = []
            for x in pareto_x:
                params = {}
                for i, name in enumerate(problem.param_names):
                    # Convert to int for discrete parameters
                    if (name == 'top_k' or name == 'num_experts' or 
                        name.endswith('_units') or name.endswith('_layers')):
                        params[name] = int(x[i])
                    else:
                        params[name] = x[i]
                pareto_params.append(params)
            
            return {
                'pareto_params': pareto_params,
                'pareto_fitness': [[-f[0], f[1]] for f in pareto_f],  # Convert first objective back to positive
                'archipelago': archi
            }
