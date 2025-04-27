"""
PyGMO Model Optimizer for Enhanced FuseMoE

This module provides functionality for optimizing the Enhanced FuseMoE system
using PyGMO's evolutionary algorithms to achieve high performance metrics.
"""

import torch
import numpy as np
import pygmo as pg
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
from models.fusion.migraine_fusion import MigraineFusion, create_migraine_fusion
from models.experts.sleep_expert import SleepExpert, create_sleep_expert
from models.experts.weather_expert import WeatherExpert, create_weather_expert
from models.experts.stress_diet_expert import StressDietExpert, create_stress_diet_expert
from models.experts.physio_expert import PhysioExpert, create_physio_expert
from utils.evaluation.metrics import calculate_metrics, generate_classification_report
from utils.training_pipeline import MigraineTrainer
from optimization.evolutionary_algorithms.pygmo_problem import EndToEndOptimizationProblem


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
        problem = MigraineMoEProblem(
            optimizer=self,
            train_loader=train_loader,
            val_loader=val_loader,
            param_names=param_names,
            bounds=bounds
        )
        
        # Create PyGMO problem
        prob = pg.problem(problem)
        
        # Create algorithm
        if algorithm == 'de':
            algo = pg.algorithm(pg.de(gen=generations))
        elif algorithm == 'sade':
            algo = pg.algorithm(pg.sade(gen=generations))
        elif algorithm == 'pso':
            algo = pg.algorithm(pg.pso(gen=generations))
        elif algorithm == 'cmaes':
            algo = pg.algorithm(pg.cmaes(gen=generations))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Set algorithm seed if provided
        if self.seed is not None:
            algo.set_seed(self.seed)
        
        # Create islands
        if islands == 1:
            # Single island
            pop = pg.population(prob, size=pop_size)
            
            # Set population seed if provided
            if self.seed is not None:
                pop.set_seed(self.seed)
            
            # Evolve population
            start_time = time.time()
            pop = algo.evolve(pop)
            end_time = time.time()
            
            # Get best solution
            best_idx = pop.best_idx()
            best_x = pop.get_x()[best_idx]
            best_f = pop.get_f()[best_idx]
            
            # Convert to parameters dictionary
            best_params = {name: value for name, value in zip(param_names, best_x)}
        else:
            # Multiple islands
            archi = pg.archipelago(n=islands, algo=algo, prob=prob, pop_size=pop_size)
            
            # Set archipelago seed if provided
            if self.seed is not None:
                archi.set_seed(self.seed)
            
            # Evolve archipelago
            start_time = time.time()
            archi.evolve()
            archi.wait_check()
            end_time = time.time()
            
            # Get best solution
            best_island_idx = 0
            best_f = float('inf')
            
            for i, island in enumerate(archi):
                island_best_idx = island.get_population().best_idx()
                island_best_f = island.get_population().get_f()[island_best_idx]
                
                if island_best_f < best_f:
                    best_f = island_best_f
                    best_island_idx = i
            
            best_island = archi[best_island_idx].get_population()
            best_idx = best_island.best_idx()
            best_x = best_island.get_x()[best_idx]
            
            # Convert to parameters dictionary
            best_params = {name: value for name, value in zip(param_names, best_x)}
        
        # Create best model
        best_model = self.create_model(best_params)
        
        # Train best model with full training schedule
        print("Training best model with full training schedule...")
        history, best_val_metrics = self.train_model(
            model=best_model,
            train_loader=train_loader,
            val_loader=val_loader,
            params=best_params,
            num_epochs=100,  # Full training
            patience=10,     # Full patience
            verbose=verbose
        )
        
        # Evaluate best model
        print("Evaluating best model...")
        evaluation_results = self.evaluate_model(
            model=best_model,
            test_loader=test_loader,
            output_dir=output_dir
        )
        
        # Save optimization results
        if output_dir is not None:
            # Save best parameters
            with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
                for name, value in best_params.items():
                    f.write(f"{name}: {value}\n")
            
            # Save best model
            torch.save(best_model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            
            # Save optimization time
            with open(os.path.join(output_dir, 'optimization_time.txt'), 'w') as f:
                f.write(f"Optimization time: {end_time - start_time:.2f} seconds\n")
        
        # Return results
        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_fitness': -best_f[0],
            'history': history,
            'evaluation': evaluation_results,
            'optimization_time': end_time - start_time
        }
    
    def plot_optimization_results(self, results: Dict[str, Any], 
                                 figsize: Tuple[int, int] = (12, 8),
                                 save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Plot optimization results.
        
        Args:
            results: Dictionary containing optimization results
            figsize: Figure size
            save_path: Directory to save plots (if None, plots are not saved)
            
        Returns:
            Dictionary mapping plot names to Matplotlib figures
        """
        # Create output directory if it doesn't exist
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        
        # Initialize plots dictionary
        plots = {}
        
        # Training history
        fig_history, ax_history = plt.subplots(figsize=figsize)
        
        # Extract training history
        train_auc = [epoch['auc'] for epoch in results['history']['train_history']]
        val_auc = [epoch['auc'] for epoch in results['history']['val_history']]
        epochs = range(1, len(train_auc) + 1)
        
        # Plot training history
        ax_history.plot(epochs, train_auc, 'b-', label='Training AUC')
        ax_history.plot(epochs, val_auc, 'r-', label='Validation AUC')
        
        # Add target line at 0.95
        ax_history.axhline(y=0.95, color='g', linestyle='--', label='Target AUC (0.95)')
        
        # Set labels and limits
        ax_history.set_xlabel('Epoch')
        ax_history.set_ylabel('AUC')
        ax_history.set_title('Training History')
        ax_history.set_ylim([0.5, 1.0])
        ax_history.legend(loc='best')
        ax_history.grid(True, alpha=0.3)
        
        # Save plot if save_path is provided
        if save_path is not None:
            fig_history.savefig(os.path.join(save_path, 'training_history.png'), 
                              bbox_inches='tight', dpi=300)
        
        plots['training_history'] = fig_history
        
        # Parameter importance
        fig_params, ax_params = plt.subplots(figsize=figsize)
        
        # Extract best parameters
        best_params = results['best_params']
        
        # Normalize parameters to [0, 1] range
        normalized_params = {}
        for name, value in best_params.items():
            min_val, max_val = self.param_bounds[name]
            normalized_params[name] = (value - min_val) / (max_val - min_val)
        
        # Sort parameters by value
        sorted_params = sorted(normalized_params.items(), key=lambda x: x[1], reverse=True)
        param_names = [item[0] for item in sorted_params]
        param_values = [item[1] for item in sorted_params]
        
        # Plot parameter importance
        ax_params.barh(param_names, param_values)
        
        # Set labels
        ax_params.set_xlabel('Normalized Parameter Value')
        ax_params.set_ylabel('Parameter')
        ax_params.set_title('Parameter Importance')
        ax_params.set_xlim([0.0, 1.0])
        ax_params.grid(True, alpha=0.3)
        
        # Save plot if save_path is provided
        if save_path is not None:
            fig_params.savefig(os.path.join(save_path, 'parameter_importance.png'), 
                             bbox_inches='tight', dpi=300)
        
        plots['parameter_importance'] = fig_params
        
        # Performance metrics
        fig_metrics, ax_metrics = plt.subplots(figsize=figsize)
        
        # Extract evaluation metrics
        metrics = results['evaluation']['metrics']
        
        # Select metrics to plot
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        metric_values = [metrics[metric] for metric in metrics_to_plot]
        
        # Plot performance metrics
        ax_metrics.bar(metrics_to_plot, metric_values)
        
        # Add target line at 0.95
        ax_metrics.axhline(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        
        # Set labels and limits
        ax_metrics.set_xlabel('Metric')
        ax_metrics.set_ylabel('Value')
        ax_metrics.set_title('Performance Metrics')
        ax_metrics.set_ylim([0.0, 1.0])
        ax_metrics.legend(loc='best')
        ax_metrics.grid(True, alpha=0.3)
        
        # Save plot if save_path is provided
        if save_path is not None:
            fig_metrics.savefig(os.path.join(save_path, 'performance_metrics.png'), 
                              bbox_inches='tight', dpi=300)
        
        plots['performance_metrics'] = fig_metrics
        
        return plots
