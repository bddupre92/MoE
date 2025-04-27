"""
Training Pipeline for Enhanced FuseMoE

This module provides functionality for training the Enhanced FuseMoE system
for migraine prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
import os
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models.experts.expert_registry import DynamicMigraineMoE, ExpertRegistry
from models.fusion.migraine_fusion import MigraineFusionMoE
from models.gating.migraine_gating import MigraineGating
from utils.evaluation.metrics import calculate_metrics


class MigraineTrainer:
    """
    Trainer for the Enhanced FuseMoE system for migraine prediction.
    
    This class provides methods for training, validating, and testing the
    Enhanced FuseMoE system.
    
    Attributes:
        model (DynamicMigraineMoE): The migraine prediction model
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        scheduler (Optional[Any]): Learning rate scheduler
        early_stopping_patience (int): Number of epochs to wait for improvement before early stopping
        best_val_metric (float): Best validation metric value
        epochs_without_improvement (int): Number of epochs without improvement
    """
    
    def __init__(self, model: DynamicMigraineMoE, 
                criterion: Optional[nn.Module] = None,
                optimizer: Optional[optim.Optimizer] = None,
                learning_rate: float = 0.001,
                weight_decay: float = 0.0,
                device: Optional[torch.device] = None,
                scheduler: Optional[Any] = None,
                early_stopping_patience: int = 10,
                checkpoint_dir: Optional[str] = None):
        """
        Initialize the migraine trainer.
        
        Args:
            model: The migraine prediction model
            criterion: Loss function (default: BCEWithLogitsLoss)
            optimizer: Optimizer (default: Adam)
            learning_rate: Learning rate for optimizer if not provided
            weight_decay: Weight decay for optimizer if not provided
            device: Device to use for training (default: cuda if available, else cpu)
            scheduler: Learning rate scheduler
            early_stopping_patience: Number of epochs to wait for improvement before early stopping
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        
        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if provided
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize early stopping variables
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        self.best_model_path = None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_load_balancing_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output, load_balancing_loss = self.model(data)
            
            # Calculate loss
            loss = self.criterion(output, target)
            if load_balancing_loss is not None:
                loss += load_balancing_loss
                total_load_balancing_loss += load_balancing_loss.item()
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Store predictions and targets for metrics calculation
            all_targets.append(target.cpu().numpy())
            all_predictions.append(torch.sigmoid(output).detach().cpu().numpy())
        
        # Calculate metrics
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        metrics = calculate_metrics(all_targets, all_predictions)
        
        # Add loss to metrics
        metrics['loss'] = total_loss / len(train_loader)
        if total_load_balancing_loss > 0:
            metrics['load_balancing_loss'] = total_load_balancing_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                all_targets.append(target.cpu().numpy())
                all_predictions.append(torch.sigmoid(output).detach().cpu().numpy())
        
        # Handle empty validation set (for testing purposes)
        if len(all_targets) == 0 or len(all_predictions) == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            
        # Calculate metrics
        all_targets = np.concatenate(all_targets) if len(all_targets) > 0 else np.array([])
        all_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
        
        # Handle empty arrays (for testing purposes)
        if len(all_targets) == 0 or len(all_predictions) == 0:
            metrics = {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        else:
            metrics = calculate_metrics(all_targets, all_predictions)
        
        # Add loss to metrics
        if len(val_loader.dataset) > 0:
            metrics['loss'] = total_loss / len(val_loader.dataset)
        else:
            metrics['loss'] = 0.0
        
        return metrics
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing test metrics
        """
        return self.validate(test_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 10,
             checkpoint_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, List[Dict[str, float]]]:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training and validation metrics for each epoch
        """
        train_history = []
        val_history = []
        
        # Use provided checkpoint directory or instance checkpoint directory
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_history.append(val_metrics)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val AUC: {val_metrics.get('auc', 0.0):.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")
            
            # Save checkpoint
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, checkpoint_path)
            
            # Early stopping
            current_val_metric = val_metrics.get('auc', 0.0)
            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.epochs_without_improvement = 0
                
                # Save best model
                if checkpoint_dir is not None:
                    self.best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }, self.best_model_path)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_history': train_history,
            'val_history': val_history
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


class PyGMOTrainingPipeline:
    """
    Training pipeline for the Enhanced FuseMoE system with PyGMO optimization.
    
    This class provides methods for training the Enhanced FuseMoE system with
    PyGMO optimization for hyperparameter tuning.
    
    Attributes:
        model_class: Class of the model to train
        expert_registry: Registry of expert models
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        device: Device to use for training
        output_dir: Directory to save outputs
    """
    
    def __init__(self, model_class: type, 
                expert_registry: ExpertRegistry,
                train_data: Tuple[torch.Tensor, torch.Tensor],
                val_data: Tuple[torch.Tensor, torch.Tensor],
                test_data: Tuple[torch.Tensor, torch.Tensor],
                device: Optional[torch.device] = None,
                output_dir: Optional[str] = None):
        """
        Initialize the PyGMO training pipeline.
        
        Args:
            model_class: Class of the model to train
            expert_registry: Registry of expert models
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            test_data: Tuple of (X_test, y_test)
            device: Device to use for training
            output_dir: Directory to save outputs
        """
        self.model_class = model_class
        self.expert_registry = expert_registry
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Create output directory if provided
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize optimization results
        self.optimization_results = None
    
    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """
        Create a model with the given parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Created model
        """
        # Extract parameters
        hidden_dim = params.get('hidden_dim', 64)
        top_k = params.get('top_k', 2)
        
        # Create gating network
        input_dim = self.train_data[0].shape[-1]
        num_experts = len(self.expert_registry)
        
        # Fix: Use input_dims as a list instead of input_dim
        gating = MigraineGating(
            input_dims=[input_dim],  # Wrap input_dim in a list to match the expected interface
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k
        )
        
        # Create model
        model = self.model_class(
            expert_registry=self.expert_registry,
            gating=gating,
            output_dim=1
        )
        
        return model
    
    def create_trainer(self, model: nn.Module, params: Dict[str, Any]) -> MigraineTrainer:
        """
        Create a trainer for the given model and parameters.
        
        Args:
            model: Model to train
            params: Dictionary of training parameters
            
        Returns:
            Created trainer
        """
        # Extract parameters
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 0.0)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler,
            checkpoint_dir=os.path.join(self.output_dir, 'checkpoints') if self.output_dir else None
        )
        
        return trainer
    
    def create_dataloaders(self, params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for the given parameters.
        
        Args:
            params: Dictionary of DataLoader parameters
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Extract parameters
        batch_size = params.get('batch_size', 32)
        
        # Create DataLoaders
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(self.train_data[0], self.train_data[1]),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(self.val_data[0], self.val_data[1]),
            batch_size=batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(self.test_data[0], self.test_data[1]),
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def evaluate_model(self, params: Dict[str, Any]) -> float:
        """
        Evaluate a model with the given parameters.
        
        Args:
            params: Dictionary of model and training parameters
            
        Returns:
            Validation metric value (higher is better)
        """
        # Create model
        model = self.create_model(params)
        
        # Create trainer
        trainer = self.create_trainer(model, params)
        
        # Create DataLoaders
        train_loader, val_loader, _ = self.create_dataloaders(params)
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=params.get('num_epochs', 10),
            verbose=False
        )
        
        # Get best validation metric
        val_metrics = [epoch_metrics.get('auc', 0.0) for epoch_metrics in history['val_history']]
        best_val_metric = max(val_metrics) if val_metrics else 0.0
        
        return best_val_metric
    
    def run_optimization(self, num_generations: int = 10, population_size: int = 20) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Run PyGMO optimization to find the best hyperparameters.
        
        Args:
            num_generations: Number of generations for PyGMO optimization
            population_size: Population size for PyGMO optimization
            
        Returns:
            Tuple of (optimized_model, optimization_results)
        """
        try:
            import pygmo as pg
        except ImportError:
            print("PyGMO is not installed. Please install it with 'pip install pygmo'.")
            # Return a default model
            default_params = {
                'hidden_dim': 64,
                'top_k': 2,
                'learning_rate': 0.001,
                'weight_decay': 0.0,
                'batch_size': 32,
                'num_epochs': 10
            }
            model = self.create_model(default_params)
            return model, {'best_params': default_params, 'best_fitness': 0.0}
        
        # Define parameter space
        param_space = {
            'hidden_dim': (32, 128),
            'top_k': (1, 4),
            'learning_rate': (0.0001, 0.01),
            'weight_decay': (0.0, 0.001),
            'batch_size': (16, 64),
            'num_epochs': (5, 20)
        }
        
        # Define PyGMO problem
        class MigraineProblem:
            def __init__(self, pipeline, param_space):
                self.pipeline = pipeline
                self.param_space = param_space
                self.dim = len(param_space)
                self.best_params = None
                self.best_fitness = -float('inf')
            
            def fitness(self, x):
                # Convert normalized parameters to actual values
                params = {}
                for i, (param_name, (min_val, max_val)) in enumerate(self.param_space.items()):
                    if param_name in ['hidden_dim', 'top_k', 'batch_size', 'num_epochs']:
                        # Integer parameters
                        params[param_name] = int(min_val + x[i] * (max_val - min_val))
                    else:
                        # Float parameters
                        params[param_name] = min_val + x[i] * (max_val - min_val)
                
                # Evaluate model
                fitness = self.pipeline.evaluate_model(params)
                
                # Update best parameters
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_params = params
                
                # Return negative fitness (PyGMO minimizes)
                return [-fitness]
            
            def get_bounds(self):
                return ([0.0] * self.dim, [1.0] * self.dim)
        
        # Create PyGMO problem
        prob = MigraineProblem(self, param_space)
        
        # Create PyGMO algorithm
        algo = pg.algorithm(pg.sade(gen=num_generations))
        
        # Create PyGMO population
        pop = pg.population(pg.problem(prob), size=population_size, seed=42)
        
        # Run optimization
        pop = algo.evolve(pop)
        
        # Get best parameters
        best_params = prob.best_params
        best_fitness = prob.best_fitness
        
        # Create optimized model
        optimized_model = self.create_model(best_params)
        
        # Train optimized model
        trainer = self.create_trainer(optimized_model, best_params)
        train_loader, val_loader, _ = self.create_dataloaders(best_params)
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=best_params.get('num_epochs', 10),
            verbose=True
        )
        
        # Save optimization results
        self.optimization_results = {
            'best_params': best_params,
            'best_fitness': best_fitness,
            'champion_x': pop.champion_x,
            'champion_f': pop.champion_f
        }
        
        if self.output_dir is not None:
            results_path = os.path.join(self.output_dir, 'optimization_results.pt')
            torch.save(self.optimization_results, results_path)
        
        return optimized_model, self.optimization_results
    
    def evaluate_optimized_model(self) -> Dict[str, float]:
        """
        Evaluate the optimized model on the test set.
        
        Returns:
            Dictionary containing test metrics
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available. Run optimization first.")
        
        # Create optimized model
        optimized_model = self.create_model(self.optimization_results['best_params'])
        
        # Create trainer
        trainer = self.create_trainer(optimized_model, self.optimization_results['best_params'])
        
        # Create DataLoaders
        _, _, test_loader = self.create_dataloaders(self.optimization_results['best_params'])
        
        # Load best model if available
        if trainer.best_model_path is not None and os.path.exists(trainer.best_model_path):
            trainer.load_checkpoint(trainer.best_model_path)
        
        # Evaluate on test set
        test_metrics = trainer.test(test_loader)
        
        # Save test metrics
        if self.output_dir is not None:
            metrics_path = os.path.join(self.output_dir, 'test_metrics.pt')
            torch.save(test_metrics, metrics_path)
        
        return test_metrics
