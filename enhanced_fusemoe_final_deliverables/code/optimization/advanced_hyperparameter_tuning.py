"""
Advanced Hyperparameter Tuning for Enhanced FuseMoE

This module provides advanced hyperparameter tuning functionality for the
Enhanced FuseMoE system using Bayesian optimization.
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import Bayesian optimizer
from optimization.bayesian_optimizer import BayesianOptimizer

# Import other modules
from models.experts.sleep_expert import create_sleep_expert
from models.experts.weather_expert import create_weather_expert
from models.experts.stress_diet_expert import create_stress_diet_expert
from models.experts.physio_expert import create_physio_expert
from models.experts.input_adapter import wrap_expert
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusionMoE
from models.experts.expert_registry import ExpertRegistry
from utils.training_pipeline import MigraineTrainer


class AdvancedHyperparameterTuner:
    """
    Advanced hyperparameter tuner for Enhanced FuseMoE.
    
    This class provides functionality for tuning hyperparameters of the
    Enhanced FuseMoE system using Bayesian optimization.
    
    Attributes:
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        test_loader (torch.utils.data.DataLoader): Test data loader
        expert_names (List[str]): List of expert names
        device (torch.device): Device to use for training
        output_dir (str): Directory to save results
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print progress
    """
    
    def __init__(self, train_loader, val_loader, test_loader, expert_names,
                device=None, output_dir="tuning_results", random_state=42, verbose=True):
        """
        Initialize the advanced hyperparameter tuner.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            expert_names: List of expert names
            device: Device to use for training (default: cuda if available, else cpu)
            output_dir: Directory to save results
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.expert_names = expert_names
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        self.random_state = random_state
        self.verbose = verbose
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize results storage
        self.tuning_history = []
        self.best_params = None
        self.best_score = float('-inf')
        self.best_model = None
    
    def _create_model(self, params):
        """
        Create model with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Tuple of (model, optimizer, criterion)
        """
        # Extract parameters
        hidden_dim = params.get('hidden_dim', 64)
        dropout_rate = params.get('dropout_rate', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 0.0001)
        top_k = params.get('top_k', 2)
        
        # Get input dimensions from first batch
        sample_batch, _ = next(iter(self.train_loader))
        input_dims = [sample_batch[name].shape[1] for name in self.expert_names]
        
        # Create expert registry
        expert_registry = ExpertRegistry()
        
        # Create expert models with optimized parameters
        expert_models = []
        for i, name in enumerate(self.expert_names):
            if 'sleep' in name:
                expert = create_sleep_expert(
                    input_dim=input_dims[i],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim // 2,
                    dropout_rate=dropout_rate
                )
            elif 'weather' in name:
                expert = create_weather_expert(
                    input_dim=input_dims[i],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim // 2,
                    dropout_rate=dropout_rate
                )
            elif 'stress' in name or 'diet' in name:
                expert = create_stress_diet_expert(
                    input_dim=input_dims[i],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim // 2,
                    dropout_rate=dropout_rate
                )
            else:  # physiological
                expert = create_physio_expert(
                    input_dim=input_dims[i],
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim // 2,
                    dropout_rate=dropout_rate
                )
            
            # Wrap expert
            wrapped_expert = wrap_expert(expert, name)
            expert_registry.register_expert(name, wrapped_expert)
            expert_models.append(expert)
        
        # Create gating network with optimized parameters
        gating = MigraineGating(
            input_dims=input_dims,
            hidden_dim=hidden_dim,
            num_experts=len(self.expert_names),
            top_k=top_k,
            dropout_rate=dropout_rate,
            noisy_gating=True
        )
        
        # Create fusion model
        model = MigraineFusionMoE(
            expert_registry=expert_registry,
            gating=gating,
            output_dim=1
        )
        
        # Move model to device
        model.to(self.device)
        
        # Create optimizer with optimized parameters
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create criterion
        criterion = torch.nn.BCEWithLogitsLoss()
        
        return model, optimizer, criterion
    
    def _train_and_evaluate(self, params):
        """
        Train and evaluate model with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Validation score (ROC AUC)
        """
        # Create model, optimizer, and criterion
        model, optimizer, criterion = self._create_model(params)
        
        # Extract training parameters
        num_epochs = params.get('num_epochs', 20)
        patience = params.get('patience', 5)
        batch_size = params.get('batch_size', 32)
        
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            checkpoint_dir=os.path.join(self.output_dir, 'checkpoints')
        )
        
        # Train model
        history = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=num_epochs,
            patience=patience
        )
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(self.val_loader)
        val_score = val_metrics.get('roc_auc', 0.0)
        
        # Store results
        result = {
            'params': params.copy(),
            'val_score': val_score,
            'history': history
        }
        self.tuning_history.append(result)
        
        # Update best model if needed
        if val_score > self.best_score:
            self.best_score = val_score
            self.best_params = params.copy()
            self.best_model = model.state_dict()
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(self.output_dir, 'best_model.pt'))
        
        return val_score
    
    def tune(self, param_bounds=None, n_iterations=20, n_initial_points=5):
        """
        Tune hyperparameters using Bayesian optimization.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            n_iterations: Number of optimization iterations
            n_initial_points: Number of initial random points to evaluate
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Default parameter bounds if not provided
        if param_bounds is None:
            param_bounds = {
                'learning_rate': (0.00001, 0.1),
                'weight_decay': (0.00001, 0.01),
                'batch_size': (8, 64),
                'hidden_dim': (32, 256),
                'dropout_rate': (0.0, 0.7),
                'top_k': (1, 4),
                'num_epochs': (10, 50),
                'patience': (3, 10)
            }
        
        # Create Bayesian optimizer
        optimizer = BayesianOptimizer(
            param_bounds=param_bounds,
            objective_function=self._train_and_evaluate,
            n_initial_points=n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Run optimization
        best_params, best_score = optimizer.optimize(n_iterations=n_iterations)
        
        # Plot optimization history
        self._plot_optimization_history()
        
        return best_params, best_score
    
    def _plot_optimization_history(self):
        """
        Plot optimization history.
        """
        # Extract scores and iterations
        scores = [result['val_score'] for result in self.tuning_history]
        iterations = list(range(1, len(scores) + 1))
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'o-', color='blue')
        plt.axhline(y=self.best_score, color='red', linestyle='--', label=f'Best score: {self.best_score:.4f}')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Score (ROC AUC)')
        plt.title('Hyperparameter Optimization History')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'optimization_history.png'))
        
        # Plot parameter importance if we have enough data
        if len(self.tuning_history) >= 10:
            self._plot_parameter_importance()
    
    def _plot_parameter_importance(self):
        """
        Plot parameter importance based on correlation with scores.
        """
        # Extract parameters and scores
        param_names = list(self.tuning_history[0]['params'].keys())
        param_values = {name: [] for name in param_names}
        scores = []
        
        for result in self.tuning_history:
            for name in param_names:
                param_values[name].append(result['params'].get(name, 0))
            scores.append(result['val_score'])
        
        # Calculate correlation with scores
        correlations = {}
        for name in param_names:
            values = np.array(param_values[name])
            if len(np.unique(values)) > 1:  # Only calculate if we have variation
                corr = np.corrcoef(values, scores)[0, 1]
                correlations[name] = abs(corr)  # Use absolute correlation
        
        # Sort by correlation
        sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        names = [item[0] for item in sorted_params]
        values = [item[1] for item in sorted_params]
        
        plt.barh(names, values, color='skyblue')
        plt.xlabel('Absolute Correlation with Validation Score')
        plt.ylabel('Hyperparameter')
        plt.title('Hyperparameter Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'))
    
    def evaluate_best_model(self):
        """
        Evaluate the best model on the test set.
        
        Returns:
            Dictionary of test metrics
        """
        if self.best_model is None:
            raise ValueError("No best model available. Run tune() first.")
        
        # Create model with best parameters
        model, optimizer, criterion = self._create_model(self.best_params)
        
        # Load best model weights
        model.load_state_dict(self.best_model)
        
        # Create trainer
        trainer = MigraineTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device
        )
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(self.test_loader)
        
        # Print test metrics
        if self.verbose:
            print("\nTest Metrics:")
            for name, value in test_metrics.items():
                print(f"  {name}: {value:.4f}")
        
        # Save test metrics
        with open(os.path.join(self.output_dir, 'test_metrics.txt'), 'w') as f:
            f.write("Test Metrics:\n")
            for name, value in test_metrics.items():
                f.write(f"  {name}: {value:.4f}\n")
        
        return test_metrics
    
    def get_learning_curves(self):
        """
        Get learning curves for the best model.
        
        Returns:
            Dictionary containing learning curves
        """
        if not self.tuning_history:
            raise ValueError("No tuning history available. Run tune() first.")
        
        # Find result with best validation score
        best_result = max(self.tuning_history, key=lambda x: x['val_score'])
        
        # Extract learning curves
        history = best_result['history']
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        train_loss = [epoch_metrics['loss'] for epoch_metrics in history['train_history']]
        val_loss = [epoch_metrics['loss'] for epoch_metrics in history['val_history']]
        epochs = list(range(1, len(train_loss) + 1))
        
        plt.plot(epochs, train_loss, 'o-', color='blue', label='Training')
        plt.plot(epochs, val_loss, 'o-', color='orange', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        train_acc = [epoch_metrics.get('accuracy', 0) for epoch_metrics in history['train_history']]
        val_acc = [epoch_metrics.get('accuracy', 0) for epoch_metrics in history['val_history']]
        
        plt.plot(epochs, train_acc, 'o-', color='blue', label='Training')
        plt.plot(epochs, val_acc, 'o-', color='orange', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'learning_curves.png'))
        
        return history
    
    def save_results(self):
        """
        Save tuning results to files.
        
        Returns:
            Dictionary containing paths to saved files
        """
        # Save best parameters
        with open(os.path.join(self.output_dir, 'best_params.txt'), 'w') as f:
            f.write(f"Best Validation Score: {self.best_score:.4f}\n\n")
            f.write("Best Parameters:\n")
            for name, value in self.best_params.items():
                f.write(f"  {name}: {value}\n")
        
        # Save tuning history
        np.save(os.path.join(self.output_dir, 'tuning_history.npy'), self.tuning_history)
        
        # Return paths
        return {
            'best_params': os.path.join(self.output_dir, 'best_params.txt'),
            'best_model': os.path.join(self.output_dir, 'best_model.pt'),
            'tuning_history': os.path.join(self.output_dir, 'tuning_history.npy'),
            'optimization_history_plot': os.path.join(self.output_dir, 'optimization_history.png'),
            'parameter_importance_plot': os.path.join(self.output_dir, 'parameter_importance.png'),
            'learning_curves_plot': os.path.join(self.output_dir, 'learning_curves.png'),
            'test_metrics': os.path.join(self.output_dir, 'test_metrics.txt')
        }


def run_hyperparameter_tuning(train_loader, val_loader, test_loader, expert_names,
                             output_dir="tuning_results", n_iterations=20,
                             random_state=42, verbose=True):
    """
    Run hyperparameter tuning for Enhanced FuseMoE.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        expert_names: List of expert names
        output_dir: Directory to save results
        n_iterations: Number of optimization iterations
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_params, best_score, file_paths)
    """
    # Create tuner
    tuner = AdvancedHyperparameterTuner(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        expert_names=expert_names,
        output_dir=output_dir,
        random_state=random_state,
        verbose=verbose
    )
    
    # Define expanded parameter bounds
    param_bounds = {
        'learning_rate': (0.00001, 0.1),
        'weight_decay': (0.00001, 0.01),
        'batch_size': (8, 64),
        'hidden_dim': (32, 256),
        'dropout_rate': (0.0, 0.7),
        'top_k': (1, 4),
        'num_epochs': (10, 50),
        'patience': (3, 10)
    }
    
    # Run tuning
    best_params, best_score = tuner.tune(
        param_bounds=param_bounds,
        n_iterations=n_iterations,
        n_initial_points=5
    )
    
    # Evaluate best model
    test_metrics = tuner.evaluate_best_model()
    
    # Get learning curves
    tuner.get_learning_curves()
    
    # Save results
    file_paths = tuner.save_results()
    
    return best_params, best_score, file_paths
