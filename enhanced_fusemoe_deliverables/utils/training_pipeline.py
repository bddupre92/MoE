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
from models.experts.expert_registry import DynamicMigraineMoE
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
                early_stopping_patience: int = 10):
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
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize early stopping variables
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
    
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
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move inputs and targets to device
            for modality in inputs:
                inputs[modality] = inputs[modality].to(self.device)
            targets = targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, load_balancing_loss = self.model(inputs, training=True)
            
            # Calculate loss
            loss = self.criterion(predictions, targets.unsqueeze(1))
            total_loss += loss.item() * targets.size(0)
            total_load_balancing_loss += load_balancing_loss.item() * targets.size(0)
            
            # Combined loss
            combined_loss = loss + load_balancing_loss
            
            # Backward pass and optimize
            combined_loss.backward()
            self.optimizer.step()
            
            # Store predictions and targets for metrics calculation
            all_predictions.append(torch.sigmoid(predictions).detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader.dataset)
        avg_load_balancing_loss = total_load_balancing_loss / len(train_loader.dataset)
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        metrics['load_balancing_loss'] = avg_load_balancing_loss
        
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
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Move inputs and targets to device
                for modality in inputs:
                    inputs[modality] = inputs[modality].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs, training=False)
                
                # Calculate loss
                loss = self.criterion(predictions, targets.unsqueeze(1))
                total_loss += loss.item() * targets.size(0)
                
                # Store predictions and targets for metrics calculation
                all_predictions.append(torch.sigmoid(predictions).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader.dataset)
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Test the model.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary containing test metrics
        """
        # Testing is the same as validation
        return self.validate(test_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int = 100, verbose: bool = True,
             checkpoint_dir: Optional[str] = None) -> Dict[str, List[Dict[str, float]]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train for
            verbose: Whether to print progress
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary containing training history
        """
        train_history = []
        val_history = []
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            if val_metrics['auc'] > self.best_val_metric:
                self.best_val_metric = val_metrics['auc']
                self.epochs_without_improvement = 0
                
                # Save checkpoint
                if checkpoint_dir is not None:
                    self.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                self.epochs_without_improvement += 1
            
            # Print progress
            if verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} - "
                     f"Time: {epoch_time:.2f}s - "
                     f"Train Loss: {train_metrics['loss']:.4f} - "
                     f"Train AUC: {train_metrics['auc']:.4f} - "
                     f"Val Loss: {val_metrics['loss']:.4f} - "
                     f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Store metrics
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model
        if checkpoint_dir is not None:
            self.load_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))
        
        return {'train_history': train_history, 'val_history': val_history}
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save a checkpoint of the model.
        
        Args:
            filepath: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load a checkpoint of the model.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_metric = checkpoint['best_val_metric']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class PyGMOTrainingPipeline:
    """
    Training pipeline with PyGMO optimization for the Enhanced FuseMoE system.
    
    This class provides methods for training the Enhanced FuseMoE system with
    PyGMO optimization.
    
    Attributes:
        create_model_fn (Callable): Function to create the model
        train_fn (Callable): Function to train the model
        evaluate_fn (Callable): Function to evaluate the model
        optimization_manager (Any): PyGMO optimization manager
    """
    
    def __init__(self, create_model_fn: Callable, train_fn: Callable, evaluate_fn: Callable,
                optimization_manager: Any):
        """
        Initialize the PyGMO training pipeline.
        
        Args:
            create_model_fn: Function to create the model
            train_fn: Function to train the model
            evaluate_fn: Function to evaluate the model
            optimization_manager: PyGMO optimization manager
        """
        self.create_model_fn = create_model_fn
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.optimization_manager = optimization_manager
    
    def optimize_and_train(self, train_data: Dict[str, np.ndarray], train_targets: np.ndarray,
                          val_data: Dict[str, np.ndarray], val_targets: np.ndarray,
                          test_data: Dict[str, np.ndarray], test_targets: np.ndarray,
                          optimization_config: Dict[str, Any],
                          training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize and train the model.
        
        Args:
            train_data: Dictionary mapping modality names to training data arrays
            train_targets: Training target values
            val_data: Dictionary mapping modality names to validation data arrays
            val_targets: Validation target values
            test_data: Dictionary mapping modality names to test data arrays
            test_targets: Test target values
            optimization_config: Configuration for optimization
            training_config: Configuration for training
            
        Returns:
            Dictionary containing results
        """
        # Prepare data for optimization
        optimization_data = {
            'train_data': train_data,
            'train_targets': train_targets,
            'val_data': val_data,
            'val_targets': val_targets
        }
        
        # Optimize experts
        print("Optimizing expert models...")
        expert_results = self.optimization_manager.optimize_experts(
            train_data=optimization_data,
            val_data=optimization_data,
            algorithm=optimization_config.get('algorithm', 'de'),
            pop_size=optimization_config.get('pop_size', 20),
            generations=optimization_config.get('generations', 10),
            islands=optimization_config.get('islands', 1)
        )
        
        # Optimize gating
        print("Optimizing gating network...")
        gating_results = self.optimization_manager.optimize_gating(
            train_data=optimization_data,
            val_data=optimization_data,
            algorithm=optimization_config.get('algorithm', 'de'),
            pop_size=optimization_config.get('pop_size', 20),
            generations=optimization_config.get('generations', 10),
            islands=optimization_config.get('islands', 1)
        )
        
        # Create optimized model
        print("Creating optimized model...")
        model = self.optimization_manager.create_optimized_model(
            expert_results=expert_results,
            gating_results=gating_results
        )
        
        # Train optimized model
        print("Training optimized model...")
        training_history = self.train_fn(
            model=model,
            train_data=train_data,
            train_targets=train_targets,
            val_data=val_data,
            val_targets=val_targets,
            **training_config
        )
        
        # Evaluate optimized model
        print("Evaluating optimized model...")
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
