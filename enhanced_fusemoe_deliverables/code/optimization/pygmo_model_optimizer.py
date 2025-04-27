"""
PyGMO Model Optimizer for Enhanced FuseMoE

This module provides functionality for optimizing the Enhanced FuseMoE system
using PyGMO's evolutionary algorithms to achieve high performance metrics.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import os
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.experts.expert_registry import DynamicMigraineMoE, ScalableExpertPool, ExpertRegistry
from models.gating.migraine_gating import MigraineGating, create_migraine_gating
from models.fusion.migraine_fusion import MigraineFusion, create_migraine_fusion, MigraineFusionMoE
from models.experts.sleep_expert import SleepExpert, create_sleep_expert
from models.experts.weather_expert import WeatherExpert, create_weather_expert
from models.experts.stress_diet_expert import StressDietExpert, create_stress_diet_expert
from models.experts.physio_expert import PhysioExpert, create_physio_expert
from utils.evaluation.metrics import calculate_metrics, generate_classification_report
from utils.training_pipeline import MigraineTrainer
from optimization.evolutionary_algorithms.pygmo_problem import EndToEndOptimizationProblem


class PyGMOModelOptimizer:
    """
    Generic PyGMO-based model optimizer for Enhanced FuseMoE.
    
    This class provides a generic interface for optimizing any model in the
    Enhanced FuseMoE system using PyGMO's evolutionary algorithms.
    
    Attributes:
        model_class (type): Class of the model to optimize
        param_space (Dict[str, Tuple[float, float]]): Parameter space for optimization
        fitness_function (Callable): Function to evaluate fitness of a model
        device (torch.device): Device to use for computation
        output_dir (str): Directory to save optimization results
        seed (int): Random seed for reproducibility
        best_params (Dict[str, Any]): Best parameters found during optimization
        best_fitness (float): Best fitness value found during optimization
        optimization_history (List[Dict[str, Any]]): History of optimization runs
    """
    
    def __init__(self, model_class: type, 
                param_space: Dict[str, Tuple[float, float]],
                fitness_function: Callable[[Any, Dict[str, Any]], float],
                device: Optional[torch.device] = None,
                output_dir: Optional[str] = None,
                seed: Optional[int] = None):
        """
        Initialize the PyGMO model optimizer.
        
        Args:
            model_class: Class of the model to optimize
            param_space: Parameter space for optimization, mapping parameter names to (min, max) tuples
            fitness_function: Function to evaluate fitness of a model, taking (model, params) and returning a float
            device: Device to use for computation (default: cuda if available, else cpu)
            output_dir: Directory to save optimization results
            seed: Random seed for reproducibility
        """
        self.model_class = model_class
        self.param_space = param_space
        self.fitness_function = fitness_function
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Create output directory if provided
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize optimization results
        self.best_params = None
        self.best_fitness = -float('inf')
        self.optimization_history = []
    
    def create_model(self, params: Dict[str, Any]) -> Any:
        """
        Create a model with the given parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Created model
        """
        # Create model using model_class and params
        model = self.model_class(**params)
        
        # Move model to device if it's a torch.nn.Module
        if isinstance(model, torch.nn.Module):
            model.to(self.device)
        
        return model
    
    def evaluate_model(self, model: Any, params: Dict[str, Any]) -> float:
        """
        Evaluate a model with the given parameters.
        
        Args:
            model: Model to evaluate
            params: Dictionary of model parameters
            
        Returns:
            Fitness value (higher is better)
        """
        return self.fitness_function(model, params)
    
    def optimize(self, algorithm: str = 'sade', 
                pop_size: int = 20, 
                generations: int = 10,
                islands: int = 1,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize the model using PyGMO.
        
        Args:
            algorithm: PyGMO algorithm to use ('sade', 'de', 'pso', etc.)
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            import pygmo as pg
        except ImportError:
            print("PyGMO is not installed. Please install it with 'pip install pygmo'.")
            # Return a default result
            return {
                'best_params': {},
                'best_fitness': 0.0,
                'message': "PyGMO is not installed. Please install it with 'pip install pygmo'."
            }
        
        # Extract parameter names and bounds
        param_names = list(self.param_space.keys())
        lower_bounds = [self.param_space[name][0] for name in param_names]
        upper_bounds = [self.param_space[name][1] for name in param_names]
        bounds = (lower_bounds, upper_bounds)
        
        # Define problem
        class ModelOptimizationProblem:
            def __init__(self, optimizer, param_names, bounds):
                self.optimizer = optimizer
                self.param_names = param_names
                self.bounds = bounds
                self.dim = len(param_names)
                
                # Keep track of best solution
                self.best_fitness = -np.inf
                self.best_params = None
                self.best_model = None
                self.evaluations = 0
            
            def fitness(self, x):
                # Convert x to parameters dictionary
                params = {name: value for name, value in zip(self.param_names, x)}
                
                # Create model
                model = self.optimizer.create_model(params)
                
                # Evaluate model
                fitness = self.optimizer.evaluate_model(model, params)
                
                # Update best solution if this is better
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_params = params.copy()
                    self.best_model = model
                
                # Increment evaluation counter
                self.evaluations += 1
                
                # Print progress if verbose
                if verbose and self.evaluations % 5 == 0:
                    print(f"Evaluation {self.evaluations}: Best fitness = {self.best_fitness:.4f}")
                
                # Return negative fitness (PyGMO minimizes)
                return [-fitness]
            
            def get_bounds(self):
                return self.bounds
            
            def get_name(self):
                return "Model Optimization Problem"
            
            def get_extra_info(self):
                return "\n".join([f"{name}: {self.best_params[name]}" for name in self.param_names])
        
        # Create problem
        prob = ModelOptimizationProblem(self, param_names, bounds)
        
        # Create algorithm
        if algorithm == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations))
        elif algorithm == 'de':
            algo = pg.algorithm(pg.de(gen=generations))
        elif algorithm == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Set algorithm seed if provided
        if self.seed is not None:
            algo.set_seed(self.seed)
        
        # Create archipelago if using multiple islands
        if islands > 1:
            archi = pg.archipelago(n=islands, algo=algo, prob=pg.problem(prob), pop_size=pop_size)
            
            # Set seed for each island if provided
            if self.seed is not None:
                for i, isl in enumerate(archi):
                    isl.set_seed(self.seed + i)
            
            # Evolve archipelago
            if verbose:
                print(f"Evolving archipelago with {islands} islands for {generations} generations...")
            
            archi.evolve()
            archi.wait()
            
            # Get best solution from all islands
            best_isl = max(archi, key=lambda isl: isl.get_population().champion_f[0])
            best_x = best_isl.get_population().champion_x
            best_f = -best_isl.get_population().champion_f[0]  # Negate to get original fitness
        else:
            # Create population
            pop = pg.population(pg.problem(prob), size=pop_size)
            
            # Set seed if provided
            if self.seed is not None:
                pop.set_seed(self.seed)
            
            # Evolve population
            if verbose:
                print(f"Evolving population for {generations} generations...")
            
            pop = algo.evolve(pop)
            
            # Get best solution
            best_x = pop.champion_x
            best_f = -pop.champion_f[0]  # Negate to get original fitness
        
        # Convert best solution to parameters dictionary
        best_params = {name: value for name, value in zip(param_names, best_x)}
        
        # Update best solution if this is better
        if best_f > self.best_fitness:
            self.best_fitness = best_f
            self.best_params = best_params
        
        # Create best model
        best_model = self.create_model(self.best_params)
        
        # Save optimization results
        results = {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'best_model': best_model,
            'algorithm': algorithm,
            'pop_size': pop_size,
            'generations': generations,
            'islands': islands,
            'param_space': self.param_space,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add to optimization history
        self.optimization_history.append(results)
        
        # Save results to file if output directory is provided
        if self.output_dir is not None:
            # Save results as JSON
            import json
            
            # Convert results to JSON-serializable format
            json_results = {
                'best_params': self.best_params,
                'best_fitness': float(self.best_fitness),
                'algorithm': algorithm,
                'pop_size': pop_size,
                'generations': generations,
                'islands': islands,
                'param_space': {k: list(v) for k, v in self.param_space.items()},
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to file
            with open(os.path.join(self.output_dir, 'optimization_results.json'), 'w') as f:
                json.dump(json_results, f, indent=4)
            
            # Save model if it's a torch.nn.Module
            if isinstance(best_model, torch.nn.Module):
                torch.save(best_model.state_dict(), os.path.join(self.output_dir, 'best_model.pt'))
        
        return results
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.optimization_history:
            print("No optimization history to plot.")
            return
        
        # Extract fitness values from history
        fitness_values = [run['best_fitness'] for run in self.optimization_history]
        timestamps = [run['timestamp'] for run in self.optimization_history]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(fitness_values)), fitness_values, marker='o')
        plt.xlabel('Optimization Run')
        plt.ylabel('Best Fitness')
        plt.title('Optimization History')
        plt.grid(True)
        
        # Add timestamps as x-tick labels
        plt.xticks(range(len(timestamps)), [ts.split()[1] for ts in timestamps], rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if save_path is provided
        if save_path is not None:
            plt.savefig(save_path)
        
        # Show plot
        plt.show()
    
    def get_best_model(self) -> Any:
        """
        Get the best model found during optimization.
        
        Returns:
            Best model
        """
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet.")
        
        return self.create_model(self.best_params)


class MigraineMoEOptimizer:
    """
    Optimizer for the Enhanced FuseMoE system for migraine prediction.
    
    This class provides methods for optimizing the Enhanced FuseMoE system
    using PyGMO's evolutionary algorithms to achieve high performance metrics.
    
    Attributes:
        device (torch.device): Device to use for training
        seed (int): Random seed for reproducibility
        expert_types (List[str]): List of expert types to use
        expert_dims (Dict[str, int]): Dictionary mapping expert types to input dimensions
        output_dim (int): Output dimension of expert models
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds for optimization
    """
    
    def __init__(self, expert_types: List[str] = None, 
                expert_dims: Dict[str, int] = None,
                output_dim: int = 32,
                device: Optional[torch.device] = None,
                seed: Optional[int] = None):
        """
        Initialize the migraine MoE optimizer.
        
        Args:
            expert_types: List of expert types to use (default: ['sleep', 'weather', 'stress_diet', 'physio'])
            expert_dims: Dictionary mapping expert types to input dimensions
            output_dim: Output dimension of expert models
            device: Device to use for training (default: cuda if available, else cpu)
            seed: Random seed for reproducibility
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Set expert types
        self.expert_types = expert_types or ['sleep', 'weather', 'stress_diet', 'physio']
        
        # Set expert dimensions
        self.expert_dims = expert_dims or {
            'sleep': 6,
            'weather': 5,
            'stress_diet': 6,
            'physio': 5
        }
        
        # Set output dimension
        self.output_dim = output_dim
        
        # Define parameter bounds for optimization
        self.param_bounds = {
            # Expert parameters
            'expert_hidden_dim': (32, 256),        # Expert hidden layer size
            'expert_num_layers': (1, 4),           # Number of expert layers
            'expert_dropout': (0.0, 0.5),          # Expert dropout rate
            
            # Gating parameters
            'gating_hidden_dim': (32, 256),        # Gating hidden layer size
            'top_k': (1, 4),                       # Number of experts to route to
            'load_balance_coef': (0.001, 0.1),     # Load balancing coefficient
            'noisy_gating': (0, 1),                # Whether to use noisy gating (0=False, 1=True)
            
            # Fusion parameters
            'fusion_hidden_dim': (32, 256),        # Fusion hidden layer size
            'fusion_dropout': (0.0, 0.5),          # Fusion dropout rate
            
            # Training parameters
            'learning_rate': (0.0001, 0.01),       # Learning rate
            'weight_decay': (0.0, 0.01),           # Weight decay
            'batch_size': (16, 128)                # Batch size
        }
    
    def create_model(self, params: Dict[str, Any]) -> DynamicMigraineMoE:
        """
        Create a migraine MoE model with the given parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Initialized migraine MoE model
        """
        # Create expert registry and pool
        registry = ExpertRegistry()
        expert_pool = ScalableExpertPool(registry)
        
        # Add experts to pool
        for expert_type in self.expert_types:
            if expert_type == 'sleep':
                expert_pool.add_expert(
                    name=expert_type,
                    input_dim=self.expert_dims[expert_type],
                    hidden_dim=int(params['expert_hidden_dim']),
                    output_dim=self.output_dim,
                    num_layers=int(params['expert_num_layers']),
                    dropout_rate=params['expert_dropout']
                )
            elif expert_type == 'weather':
                expert_pool.add_expert(
                    name=expert_type,
                    input_dim=self.expert_dims[expert_type],
                    hidden_dim=int(params['expert_hidden_dim']),
                    output_dim=self.output_dim,
                    num_layers=int(params['expert_num_layers']),
                    dropout_rate=params['expert_dropout']
                )
            elif expert_type == 'stress_diet':
                expert_pool.add_expert(
                    name=expert_type,
                    input_dim=self.expert_dims[expert_type],
                    hidden_dim=int(params['expert_hidden_dim']),
                    output_dim=self.output_dim,
                    num_layers=int(params['expert_num_layers']),
                    dropout_rate=params['expert_dropout']
                )
            elif expert_type == 'physio':
                expert_pool.add_expert(
                    name=expert_type,
                    input_dim=self.expert_dims[expert_type],
                    hidden_dim=int(params['expert_hidden_dim']),
                    output_dim=self.output_dim,
                    num_layers=int(params['expert_num_layers']),
                    dropout_rate=params['expert_dropout']
                )
        
        # Create gating network
        input_dims = [self.expert_dims[expert_type] for expert_type in self.expert_types]
        gating = MigraineGating(
            input_dims=input_dims,
            hidden_dim=int(params['gating_hidden_dim']),
            num_experts=len(self.expert_types),
            top_k=int(params['top_k']),
            dropout_rate=params['expert_dropout'],
            noisy_gating=bool(int(params['noisy_gating']))
        )
        gating.load_balance_coef = params['load_balance_coef']
        
        # Create fusion mechanism
        fusion = MigraineFusion(
            expert_output_dim=self.output_dim,
            hidden_dim=int(params['fusion_hidden_dim']),
            num_experts=len(self.expert_types),
            dropout_rate=params['fusion_dropout']
        )
        
        # Create dynamic migraine MoE
        model = DynamicMigraineMoE(expert_pool, gating, fusion)
        
        # Move model to device
        model.to(self.device)
        
        return model
    
    def train_model(self, model: DynamicMigraineMoE, train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader, params: Dict[str, Any],
                   num_epochs: int = 50, patience: int = 10,
                   verbose: bool = True) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, float]]:
        """
        Train a migraine MoE model.
        
        Args:
            model: Migraine MoE model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            params: Dictionary of training parameters
            num_epochs: Maximum number of epochs to train for
            patience: Number of epochs to wait for improvement before early stopping
            verbose: Whether to print progress
            
        Returns:
            Tuple of (training_history, best_val_metrics)
        """
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=verbose
        )
        
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            early_stopping_patience=patience
        )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            verbose=verbose
        )
        
        # Get best validation metrics
        best_val_metrics = max(history['val_history'], key=lambda x: x['auc'])
        
        return history, best_val_metrics
    
    def evaluate_model(self, model: DynamicMigraineMoE, test_loader: torch.utils.data.DataLoader,
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a migraine MoE model.
        
        Args:
            model: Migraine MoE model
            test_loader: DataLoader for test data
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            device=self.device
        )
        
        # Evaluate model
        test_metrics = trainer.test(test_loader)
        
        # Collect predictions and targets
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                # Move inputs to device
                for modality in inputs:
                    inputs[modality] = inputs[modality].to(self.device)
                
                # Forward pass
                predictions = model(inputs, training=False)
                
                # Store predictions and targets
                all_predictions.append(torch.sigmoid(predictions).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Generate classification report
        report = generate_classification_report(
            y_true=all_targets,
            y_pred=all_predictions,
            output_dir=output_dir
        )
        
        # Return results
        return {
            'metrics': test_metrics,
            'report': report,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def optimize(self, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                algorithm: str = 'sade',
                pop_size: int = 20,
                generations: int = 10,
                islands: int = 1,
                output_dir: Optional[str] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Optimize the migraine MoE model using PyGMO.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            algorithm: PyGMO algorithm to use
            pop_size: Population size for evolutionary algorithm
            generations: Number of generations to evolve
            islands: Number of islands for archipelago
            output_dir: Directory to save optimization results
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            import pygmo as pg
        except ImportError:
            print("PyGMO is not installed. Please install it with 'pip install pygmo'.")
            # Return a default result
            return {
                'best_params': {},
                'best_fitness': 0.0,
                'message': "PyGMO is not installed. Please install it with 'pip install pygmo'."
            }
        
        # Create output directory if it doesn't exist
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract parameter names and bounds
        param_names = list(self.param_bounds.keys())
        lower_bounds = [self.param_bounds[name][0] for name in param_names]
        upper_bounds = [self.param_bounds[name][1] for name in param_names]
        bounds = (lower_bounds, upper_bounds)
        
        # Define problem
        class MigraineMoEProblem:
            def __init__(self, optimizer, train_loader, val_loader, param_names, bounds):
                self.optimizer = optimizer
                self.train_loader = train_loader
                self.val_loader = val_loader
                self.param_names = param_names
                self.bounds = bounds
                self.dim = len(param_names)
                
                # Keep track of best solution
                self.best_fitness = -np.inf
                self.best_params = None
                self.best_model = None
                self.evaluations = 0
            
            def fitness(self, x):
                # Convert x to parameters dictionary
                params = {name: value for name, value in zip(self.param_names, x)}
                
                # Create model
                model = self.optimizer.create_model(params)
                
                # Train model
                _, best_val_metrics = self.optimizer.train_model(
                    model=model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    params=params,
                    num_epochs=20,  # Reduced for optimization
                    patience=5,     # Reduced for optimization
                    verbose=False
                )
                
                # Get fitness (negative because PyGMO minimizes)
                fitness = -best_val_metrics['auc']
                
                # Update best solution if this is better
                if fitness < -self.best_fitness:
                    self.best_fitness = -fitness
                    self.best_params = params.copy()
                    self.best_model = model
                
                # Increment evaluation counter
                self.evaluations += 1
                
                # Print progress if verbose
                if verbose and self.evaluations % 5 == 0:
                    print(f"Evaluation {self.evaluations}: Best AUC = {self.best_fitness:.4f}")
                
                return [fitness]
            
            def get_bounds(self):
                return self.bounds
            
            def get_name(self):
                return "Migraine MoE Optimization Problem"
            
            def get_extra_info(self):
                return "\n".join([f"{name}: {self.best_params[name]}" for name in self.param_names])
        
        # Create problem
        prob = MigraineMoEProblem(self, train_loader, val_loader, param_names, bounds)
        
        # Create algorithm
        if algorithm == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations))
        elif algorithm == 'de':
            algo = pg.algorithm(pg.de(gen=generations))
        elif algorithm == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Set algorithm seed if provided
        if self.seed is not None:
            algo.set_seed(self.seed)
        
        # Create archipelago if using multiple islands
        if islands > 1:
            archi = pg.archipelago(n=islands, algo=algo, prob=pg.problem(prob), pop_size=pop_size)
            
            # Set seed for each island if provided
            if self.seed is not None:
                for i, isl in enumerate(archi):
                    isl.set_seed(self.seed + i)
            
            # Evolve archipelago
            if verbose:
                print(f"Evolving archipelago with {islands} islands for {generations} generations...")
            
            archi.evolve()
            archi.wait()
            
            # Get best solution from all islands
            best_isl = max(archi, key=lambda isl: isl.get_population().champion_f[0])
            best_x = best_isl.get_population().champion_x
            best_f = -best_isl.get_population().champion_f[0]  # Negate to get original fitness
        else:
            # Create population
            pop = pg.population(pg.problem(prob), size=pop_size)
            
            # Set seed if provided
            if self.seed is not None:
                pop.set_seed(self.seed)
            
            # Evolve population
            if verbose:
                print(f"Evolving population for {generations} generations...")
            
            pop = algo.evolve(pop)
            
            # Get best solution
            best_x = pop.champion_x
            best_f = -pop.champion_f[0]  # Negate to get original fitness
        
        # Convert best solution to parameters dictionary
        best_params = {name: value for name, value in zip(param_names, best_x)}
        
        # Create best model
        best_model = self.create_model(best_params)
        
        # Evaluate best model on test set
        test_results = self.evaluate_model(best_model, test_loader, output_dir)
        
        # Save optimization results
        results = {
            'best_params': best_params,
            'best_fitness': best_f,
            'best_model': best_model,
            'test_results': test_results,
            'algorithm': algorithm,
            'pop_size': pop_size,
            'generations': generations,
            'islands': islands
        }
        
        # Save results to file if output directory is provided
        if output_dir is not None:
            # Save model
            torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            
            # Save parameters
            np.savez(
                os.path.join(output_dir, 'optimization_results.npz'),
                best_params=best_params,
                best_fitness=best_f,
                test_metrics=test_results['metrics']
            )
            
            # Save report
            with open(os.path.join(output_dir, 'optimization_report.txt'), 'w') as f:
                f.write(f"Optimization Results\n")
                f.write(f"===================\n\n")
                f.write(f"Algorithm: {algorithm}\n")
                f.write(f"Population Size: {pop_size}\n")
                f.write(f"Generations: {generations}\n")
                f.write(f"Islands: {islands}\n\n")
                f.write(f"Best Fitness (AUC): {best_f:.4f}\n\n")
                f.write(f"Best Parameters:\n")
                for name, value in best_params.items():
                    f.write(f"  {name}: {value}\n")
                f.write(f"\nTest Results:\n")
                for metric, value in test_results['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write(f"\nClassification Report:\n")
                f.write(test_results['report'])
        
        return results
