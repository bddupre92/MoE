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
            # Move data and target to device
            if isinstance(data, dict):
                for key in data:
                    data[key] = data[key].to(self.device)
            else:
                data = data.to(self.device)
            
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Handle different return types from model
            model_output = self.model(data, training=True)
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                output, load_balancing_loss = model_output
            else:
                output = model_output
                load_balancing_loss = None
            
            # Ensure output has the right shape for the loss function
            if output.dim() > 1 and output.size(1) > 1:
                # Multi-class output
                output = output.squeeze()
            else:
                # Binary output
                output = output.view(-1)
            
            # Ensure target has the right shape
            if target.dim() > 1:
                target = target.squeeze()
            
            # Calculate loss
            loss = self.criterion(output, target)
            if load_balancing_loss is not None:
                # Reduce load_balancing_loss to a scalar by taking the mean
                if isinstance(load_balancing_loss, torch.Tensor) and load_balancing_loss.dim() > 0:
                    load_balancing_loss = load_balancing_loss.mean()
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
        try:
            all_targets = np.concatenate(all_targets)
            all_predictions = np.concatenate(all_predictions)
            metrics = calculate_metrics(all_targets, all_predictions)
        except (ValueError, TypeError) as e:
            # Handle empty arrays or other errors
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
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
                # Move data and target to device
                if isinstance(data, dict):
                    for key in data:
                        data[key] = data[key].to(self.device)
                else:
                    data = data.to(self.device)
                
                target = target.to(self.device)
                
                # Handle different return types from model
                model_output = self.model(data, training=False)
                if isinstance(model_output, tuple) and len(model_output) >= 1:
                    output = model_output[0]
                else:
                    output = model_output
                
                # Ensure output has the right shape for the loss function
                if output.dim() > 1 and output.size(1) > 1:
                    # Multi-class output
                    output = output.squeeze()
                else:
                    # Binary output
                    output = output.view(-1)
                
                # Ensure target has the right shape
                if target.dim() > 1:
                    target = target.squeeze()
                
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
        try:
            all_targets = np.concatenate(all_targets) if len(all_targets) > 0 else np.array([])
            all_predictions = np.concatenate(all_predictions) if len(all_predictions) > 0 else np.array([])
            
            # Handle empty arrays (for testing purposes)
            if len(all_targets) == 0 or len(all_predictions) == 0:
                metrics = {'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            else:
                metrics = calculate_metrics(all_targets, all_predictions)
        except (ValueError, TypeError) as e:
            # Handle errors in metrics calculation
            metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
        
        # Add loss to metrics
        if len(val_loader) > 0:
            metrics['loss'] = total_loss / len(val_loader)
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
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load early stopping variables if available
        if 'best_val_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_val_metric']
        if 'epochs_without_improvement' in checkpoint:
            self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        return checkpoint
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint file
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_metric': self.best_val_metric,
            'epochs_without_improvement': self.epochs_without_improvement
        }, checkpoint_path)
    
    def get_learning_rate(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def get_best_model_path(self) -> Optional[str]:
        """
        Get the path to the best model checkpoint.
        
        Returns:
            Path to the best model checkpoint, or None if not available
        """
        return self.best_model_path
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'model_type': type(self.model).__name__,
            'criterion_type': type(self.criterion).__name__,
            'optimizer_type': type(self.optimizer).__name__,
            'learning_rate': self.get_learning_rate(),
            'device': str(self.device),
            'early_stopping_patience': self.early_stopping_patience,
            'best_val_metric': self.best_val_metric
        }
    
    def load_best_model(self) -> None:
        """
        Load the best model from checkpoint.
        
        Raises:
            ValueError: If best model path is not available
        """
        if self.best_model_path is None:
            raise ValueError("Best model path is not available. Train the model first.")
        
        self.load_checkpoint(self.best_model_path)


class PyGMOTrainingPipeline:
    """
    Training pipeline for the Enhanced FuseMoE system with PyGMO optimization.
    
    This class provides methods for training the Enhanced FuseMoE system with
    PyGMO optimization for hyperparameter tuning.
    
    Attributes:
        create_model_fn: Function to create model with given parameters
        train_fn: Function to train model
        evaluate_fn: Function to evaluate model
        optimization_manager: Manager for optimization process
    """
    
    def __init__(self, create_model_fn: Callable,
                train_fn: Callable,
                evaluate_fn: Callable,
                optimization_manager: Any,
                device: Optional[torch.device] = None,
                output_dir: Optional[str] = None):
        """
        Initialize the PyGMO training pipeline.
        
        Args:
            create_model_fn: Function to create model with given parameters
            train_fn: Function to train model
            evaluate_fn: Function to evaluate model
            optimization_manager: Manager for optimization process
            device: Device to use for training (default: cuda if available, else cpu)
            output_dir: Directory to save outputs (default: None)
        """
        self.create_model_fn = create_model_fn
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.optimization_manager = optimization_manager
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Create output directory if provided
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def optimize_and_train(self, train_data: Dict[str, np.ndarray],
                         train_targets: np.ndarray,
                         val_data: Dict[str, np.ndarray],
                         val_targets: np.ndarray,
                         test_data: Dict[str, np.ndarray],
                         test_targets: np.ndarray,
                         optimization_config: Dict[str, Any],
                         training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters and train the model.
        
        Args:
            train_data: Dictionary mapping expert names to training features
            train_targets: Training targets
            val_data: Dictionary mapping expert names to validation features
            val_targets: Validation targets
            test_data: Dictionary mapping expert names to test features
            test_targets: Test targets
            optimization_config: Configuration for optimization
            training_config: Configuration for training
            
        Returns:
            Dictionary containing optimization and training results
        """
        # Optimize expert models
        expert_results = self.optimization_manager.optimize_experts(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            **optimization_config
        )
        
        # Optimize gating network
        gating_results = self.optimization_manager.optimize_gating(
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            **optimization_config
        )
        
        # Create optimized model
        model = self.optimization_manager.create_optimized_model(
            expert_results=expert_results,
            gating_results=gating_results
        )
        
        # Train model
        training_history = self.train_fn(
            model=model,
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            **training_config
        )
        
        # Evaluate model
        test_metrics = self.evaluate_fn(
            model=model,
            test_data=test_data,
            test_targets=test_targets
        )
        
        # Return results
        return {
            'expert_results': expert_results,
            'gating_results': gating_results,
            'training_history': training_history,
            'test_metrics': test_metrics,
            'model': model
        }
    
    def create_model(self, input_dim: int, hidden_dim: int, num_experts: int, top_k: int) -> DynamicMigraineMoE:
        """
        Create a model with given parameters.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_experts: Number of experts
            top_k: Number of experts to route to
            
        Returns:
            Initialized model
        """
        # Create expert registry
        expert_registry = ExpertRegistry()
        
        # Create experts
        for i in range(num_experts):
            expert = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            expert_registry.register(f"expert_{i}", expert)
        
        # Create gating network
        gating = MigraineGating(
            input_dims=[input_dim],
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            top_k=top_k
        )
        
        # Create fusion mechanism
        fusion = MigraineFusionMoE(
            expert_registry=expert_registry,
            gating=gating,
            output_dim=1
        )
        
        # Create model
        model = DynamicMigraineMoE(
            expert_registry=expert_registry,
            gating=gating,
            fusion=fusion
        )
        
        return model
    
    def train_model(self, model: DynamicMigraineMoE, train_data: Dict[str, np.ndarray],
                  train_targets: np.ndarray, val_data: Dict[str, np.ndarray],
                  val_targets: np.ndarray, num_epochs: int = 10,
                  batch_size: int = 32, learning_rate: float = 0.001,
                  weight_decay: float = 0.0) -> Dict[str, List[Dict[str, float]]]:
        """
        Train a model.
        
        Args:
            model: Model to train
            train_data: Dictionary mapping expert names to training features
            train_targets: Training targets
            val_data: Dictionary mapping expert names to validation features
            val_targets: Validation targets
            num_epochs: Number of epochs to train for
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            
        Returns:
            Dictionary containing training and validation metrics for each epoch
        """
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device,
            early_stopping_patience=5,
            checkpoint_dir=self.output_dir
        )
        
        # Create DataLoader
        # This is a simplified version; in practice, you would need to create a proper Dataset
        # that handles the dictionary of inputs
        
        # Train model
        history = trainer.train(
            train_loader=None,  # Replace with actual DataLoader
            val_loader=None,    # Replace with actual DataLoader
            num_epochs=num_epochs,
            verbose=True
        )
        
        return history
    
    def evaluate_model(self, model: DynamicMigraineMoE, test_data: Dict[str, np.ndarray],
                     test_targets: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a model.
        
        Args:
            model: Model to evaluate
            test_data: Dictionary mapping expert names to test features
            test_targets: Test targets
            
        Returns:
            Dictionary containing test metrics
        """
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            device=self.device
        )
        
        # Create DataLoader
        # This is a simplified version; in practice, you would need to create a proper Dataset
        # that handles the dictionary of inputs
        
        # Evaluate model
        metrics = trainer.test(None)  # Replace with actual DataLoader
        
        return metrics


def create_migraine_trainer(model: DynamicMigraineMoE, learning_rate: float = 0.001,
                          weight_decay: float = 0.0, device: Optional[torch.device] = None,
                          early_stopping_patience: int = 10,
                          checkpoint_dir: Optional[str] = None) -> MigraineTrainer:
    """
    Create a migraine trainer.
    
    Args:
        model: The migraine prediction model
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to use for training
        early_stopping_patience: Number of epochs to wait for improvement before early stopping
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Initialized migraine trainer
    """
    return MigraineTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir
    )


def create_pygmo_training_pipeline(model_class: Any, expert_registry: ExpertRegistry,
                                 train_data: Dict[str, np.ndarray],
                                 val_data: Dict[str, np.ndarray],
                                 test_data: Dict[str, np.ndarray],
                                 device: Optional[torch.device] = None,
                                 output_dir: Optional[str] = None) -> PyGMOTrainingPipeline:
    """
    Create a PyGMO training pipeline.
    
    Args:
        model_class: Model class to use
        expert_registry: Registry of expert models
        train_data: Dictionary mapping expert names to training features
        val_data: Dictionary mapping expert names to validation features
        test_data: Dictionary mapping expert names to test features
        device: Device to use for training
        output_dir: Directory to save outputs
        
    Returns:
        Initialized PyGMO training pipeline
    """
    # Create functions for PyGMOTrainingPipeline
    def create_model_fn(params):
        return model_class(**params)
    
    def train_fn(model, train_data, train_targets, val_data, val_targets, **kwargs):
        trainer = MigraineTrainer(
            model=model,
            device=device,
            **kwargs
        )
        return trainer.train(None, None, **kwargs)  # Replace with actual DataLoaders
    
    def evaluate_fn(model, test_data, test_targets):
        trainer = MigraineTrainer(
            model=model,
            device=device
        )
        return trainer.test(None)  # Replace with actual DataLoader
    
    # Create optimization manager
    from optimization.evolutionary_algorithms.optimization_manager import OptimizationManager
    optimization_manager = OptimizationManager(
        expert_registry=expert_registry,
        train_data=train_data,
        val_data=val_data,
        device=device
    )
    
    # Create pipeline
    pipeline = PyGMOTrainingPipeline(
        create_model_fn=create_model_fn,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
        optimization_manager=optimization_manager,
        device=device,
        output_dir=output_dir
    )
    
    return pipeline
