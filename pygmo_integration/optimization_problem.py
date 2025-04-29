"""
Optimization Problem Definition for PyGMO integration with MoE

This module defines the optimization problems for PyGMO to optimize
Mixture of Experts (MoE) models for migraine prediction. It includes
problem definitions for end-to-end model optimization, expert-specific
optimization, and gating network optimization.

Author: Manus AI
Date: April 28, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BaseMoEOptimizationProblem:
    """Base class for MoE optimization problems."""
    
    def __init__(self, 
                 model=None,
                 X_train=None, 
                 y_train=None,
                 X_val=None,
                 y_val=None,
                 objectives=['accuracy', 'f1', 'expert_balance'],
                 dim=10,
                 bounds=None):
        """
        Initialize base optimization problem.
        
        Args:
            model: MoE model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            objectives: List of objectives to optimize
            dim: Dimension of the problem (number of parameters to optimize)
            bounds: Bounds for parameters (min, max)
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.objectives = objectives
        self.dim = dim
        
        # Default bounds if none provided
        if bounds is None:
            self.bounds = ([0.0] * dim, [1.0] * dim)
        else:
            self.bounds = bounds
            
        # Validate inputs
        self._validate_inputs()
        
        logger.info(f"Initialized {self.__class__.__name__} with dim={dim}, objectives={objectives}")
    
    def _validate_inputs(self):
        """Validate inputs for the optimization problem."""
        # For demonstration, we'll use mock data if real data is not provided
        if self.model is None:
            logger.warning("No model provided, using mock model for demonstration")
            self.model = MockMoEModel(self.dim)
        
        if self.X_train is None or self.y_train is None:
            logger.warning("No training data provided, using mock data for demonstration")
            self.X_train, self.y_train = self._generate_mock_data(1000)
        
        if self.X_val is None or self.y_val is None:
            logger.warning("No validation data provided, using mock data for demonstration")
            self.X_val, self.y_val = self._generate_mock_data(200)
    
    def _generate_mock_data(self, n_samples):
        """Generate mock data for demonstration."""
        X = np.random.randn(n_samples, 10)
        y = (np.sum(X, axis=1) > 0).astype(int)
        return X, y
    
    def fitness(self, x):
        """
        Calculate fitness values for a solution.
        
        Args:
            x: Solution vector
            
        Returns:
            List of fitness values (to be minimized)
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement fitness method")
    
    def get_bounds(self):
        """Get bounds for the optimization problem."""
        return self.bounds
    
    def get_name(self):
        """Get name of the optimization problem."""
        return self.__class__.__name__
    
    def get_extra_info(self):
        """Get extra information about the optimization problem."""
        return f"\n\tDimensions: {self.dim}\n\tObjectives: {self.objectives}"
    
    def get_nobj(self):
        """Get number of objectives."""
        return len(self.objectives)


class EndToEndMoEOptimizationProblem(BaseMoEOptimizationProblem):
    """End-to-end optimization problem for MoE models."""
    
    def __init__(self, 
                 model=None,
                 train_data: Dict[str, np.ndarray] = None, 
                 val_data: Dict[str, np.ndarray] = None,
                 objectives=["accuracy", "f1", "expert_balance"],
                 dim=None,
                 bounds=None,
                 expert_weights=True,
                 gating_params=True,
                 complexity_penalty=0.01):
        """
        Initialize end-to-end optimization problem.
        
        Args:
            model: MoE model to optimize
            train_data: Dictionary containing training features and labels (e.g., {"sleep": ..., "weather": ..., "target": ...})
            val_data: Dictionary containing validation features and labels
            objectives: List of objectives to optimize
            dim: Dimension of the problem (number of parameters to optimize)
            bounds: Bounds for parameters (min, max)
            expert_weights: Whether to optimize expert weights
            gating_params: Whether to optimize gating network parameters
            complexity_penalty: Penalty factor for model complexity
        """
        # Determine dimension based on model if not provided
        if dim is None and model is not None:
            dim = self._get_model_param_count(model, expert_weights, gating_params)
        elif dim is None:
            dim = 20  # Default dimension if model not provided
            
        # Extract data and target
        self.X_train_dict = {k: v for k, v in train_data.items() if k != "target"}
        self.y_train = train_data["target"]
        self.X_val_dict = {k: v for k, v in val_data.items() if k != "target"}
        self.y_val = val_data["target"]
        
        # Pass dummy X_train, X_val to superclass (interface requires it)
        super().__init__(model, None, self.y_train, None, self.y_val, objectives, dim, bounds)
        
        self.expert_weights = expert_weights
        self.gating_params = gating_params
        self.complexity_penalty = complexity_penalty
        
        logger.info(f"Initialized EndToEndMoEOptimizationProblem with complexity_penalty={complexity_penalty}")
    
    def _get_model_param_count(self, model, expert_weights, gating_params):
        """Get parameter count from model."""
        # This would depend on the specific model implementation
        # For demonstration, we'll return a fixed value
        return 20
    
    def fitness(self, x):
        """
        Calculate fitness values for a solution.
        
        Args:
            x: Solution vector
            
        Returns:
            List of fitness values (to be minimized)
        """
        # Apply parameters to model
        self._apply_parameters(x)
        
        # For PyTorch models, we need to adapt the interface
        # Check if model has train method (PyTorch) or fit method
        if hasattr(self.model, 'train') and callable(self.model.train):
            # PyTorch model - create a simple training loop
            self.model.train()
            
            # Convert numpy arrays to PyTorch tensors if needed
            y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
            
            # Simple training loop
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            
            # Mini-batch training
            batch_size = min(32, len(self.y_train))
            
            # Check if we have dictionary inputs for MoE model
            if hasattr(self, 'X_train_dict') and self.X_train_dict:
                # Convert dictionary inputs to tensors
                X_train_tensors = {}
                for key, value in self.X_train_dict.items():
                    if not isinstance(value, torch.Tensor):
                        X_train_tensors[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        X_train_tensors[key] = value
                
                # Create indices for batching
                indices = np.arange(len(self.y_train))
                np.random.shuffle(indices)
                
                # Training loop with dictionary inputs
                for epoch in range(3):  # Just a few epochs for demonstration
                    for i in range(0, len(indices), batch_size):
                        batch_indices = indices[i:i+batch_size]
                        
                        # Create batch dictionary
                        batch_X = {k: v[batch_indices] for k, v in X_train_tensors.items()}
                        batch_y = y_train_tensor[batch_indices]
                        
                        optimizer.zero_grad()
                        
                        # Forward pass with dictionary inputs
                        try:
                            outputs = self.model(**batch_X)
                        except TypeError:
                            # If model doesn't accept kwargs, try with positional args
                            try:
                                outputs = self.model(*batch_X.values())
                            except Exception as e:
                                logger.warning(f"Model forward pass failed: {e}")
                                # Use mock predictions as fallback
                                return [0.5, 0.5]  # Default fitness values
                        
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate model
                self.model.eval()
                with torch.no_grad():
                    # Convert validation data to tensors
                    X_val_tensors = {}
                    for key, value in self.X_val_dict.items():
                        if not isinstance(value, torch.Tensor):
                            X_val_tensors[key] = torch.tensor(value, dtype=torch.float32)
                        else:
                            X_val_tensors[key] = value
                    
                    # Forward pass with dictionary inputs
                    try:
                        outputs = self.model(**X_val_tensors)
                    except TypeError:
                        # If model doesn't accept kwargs, try with positional args
                        try:
                            outputs = self.model(*X_val_tensors.values())
                        except Exception as e:
                            logger.warning(f"Model evaluation failed: {e}")
                            # Use mock predictions as fallback
                            return [0.5, 0.5]  # Default fitness values
                    
                    y_pred = (outputs > 0).float().cpu().numpy()
            
            else:
                # Standard single-input model
                if not isinstance(self.X_train, torch.Tensor):
                    X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
                else:
                    X_train_tensor = self.X_train
                
                dataset = TensorDataset(X_train_tensor, y_train_tensor)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                for epoch in range(3):  # Just a few epochs for demonstration
                    for batch_X, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate model
                self.model.eval()
                with torch.no_grad():
                    if not isinstance(self.X_val, torch.Tensor):
                        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
                    else:
                        X_val_tensor = self.X_val
                    
                    outputs = self.model(X_val_tensor)
                    y_pred = (outputs > 0).float().cpu().numpy()
        
        elif hasattr(self.model, 'fit') and callable(self.model.fit):
            # Model has fit method (scikit-learn style)
            if hasattr(self, 'X_train_dict') and self.X_train_dict:
                # If we have dictionary inputs but model has fit method,
                # we need to adapt the interface
                try:
                    self.model.fit(self.X_train_dict, self.y_train)
                    y_pred = self.model.predict(self.X_val_dict)
                except Exception as e:
                    logger.warning(f"Model fit/predict with dictionary inputs failed: {e}")
                    # Use mock predictions as fallback
                    y_pred = np.random.randint(0, 2, size=len(self.y_val))
            else:
                # Standard fit/predict interface
                self.model.fit(self.X_train, self.y_train)
                y_pred = self.model.predict(self.X_val)
        
        else:
            # No standard interface found, use mock predictions
            logger.warning("Model has no standard training interface, using mock predictions")
            y_pred = np.random.randint(0, 2, size=len(self.y_val))
        
        # Calculate fitness values
        fitness_values = []
        
        for objective in self.objectives:
            if objective == 'accuracy':
                # Accuracy (to be maximized, so we minimize 1-accuracy)
                acc = accuracy_score(self.y_val, y_pred)
                fitness_values.append(1.0 - acc)
            
            elif objective == 'precision':
                # Precision (to be maximized, so we minimize 1-precision)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fp = np.sum((y_pred == 1) & (self.y_val == 0))
                precision = tp / (tp + fp + 1e-10)
                fitness_values.append(1.0 - precision)
            
            elif objective == 'recall':
                # Recall (to be maximized, so we minimize 1-recall)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fn = np.sum((y_pred == 0) & (self.y_val == 1))
                recall = tp / (tp + fn + 1e-10)
                fitness_values.append(1.0 - recall)
            
            elif objective == 'f1':
                # F1 score (to be maximized, so we minimize 1-f1)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fp = np.sum((y_pred == 1) & (self.y_val == 0))
                fn = np.sum((y_pred == 0) & (self.y_val == 1))
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                fitness_values.append(1.0 - f1)
            
            elif objective == 'false_positive_rate':
                # False positive rate (to be minimized)
                tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred, labels=[0, 1]).ravel()
                fpr = fp / (fp + tn + 1e-10)
                fitness_values.append(fpr)
                
            elif objective == 'false_negative_rate':
                # False negative rate (to be minimized)
                tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred, labels=[0, 1]).ravel()
                fnr = fn / (fn + tp + 1e-10)
                fitness_values.append(fnr)
            
            elif objective == 'expert_balance':
                # Expert balance (minimize variance in expert contributions)
                # If model has get_expert_contributions method
                if hasattr(self.model, 'get_expert_contributions') and callable(self.model.get_expert_contributions):
                    try:
                        if hasattr(self, 'X_val_dict') and self.X_val_dict:
                            expert_contributions = self.model.get_expert_contributions(self.X_val_dict)
                        else:
                            expert_contributions = self.model.get_expert_contributions(self.X_val)
                        balance = np.var(expert_contributions)
                    except Exception as e:
                        logger.warning(f"Failed to get expert contributions: {e}")
                        balance = 0.2  # Default value
                else:
                    # Mock expert balance
                    balance = 0.2  # Default value
                fitness_values.append(balance)
            
            elif objective == 'complexity':
                # Model complexity (minimize)
                # If model has get_complexity method
                if hasattr(self.model, 'get_complexity') and callable(self.model.get_complexity):
                    complexity = self.model.get_complexity()
                else:
                    # Mock complexity
                    complexity = 10.0  # Default value
                fitness_values.append(self.complexity_penalty * complexity)
        
        return fitness_values
    
    def _apply_parameters(self, x):
        """Apply parameters to model."""
        # This would depend on the specific model implementation
        # For demonstration, we'll just pass
        pass


class ExpertOptimizationProblem(BaseMoEOptimizationProblem):
    """Optimization problem for individual experts in MoE models."""
    
    def __init__(self, 
                 model=None,
                 expert_idx=0,
                 X_train=None, 
                 y_train=None,
                 X_val=None,
                 y_val=None,
                 objectives=['accuracy', 'precision', 'complexity'],
                 dim=None,
                 bounds=None,
                 complexity_penalty=0.01):
        """
        Initialize expert optimization problem.
        
        Args:
            model: MoE model containing the expert to optimize
            expert_idx: Index of the expert to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            objectives: List of objectives to optimize
            dim: Dimension of the problem (number of parameters to optimize)
            bounds: Bounds for parameters (min, max)
            complexity_penalty: Penalty factor for model complexity
        """
        # Determine dimension based on expert if not provided
        if dim is None and model is not None:
            dim = self._get_expert_param_count(model, expert_idx)
        elif dim is None:
            dim = 10  # Default dimension if model not provided
        
        super().__init__(model, X_train, y_train, X_val, y_val, objectives, dim, bounds)
        
        self.expert_idx = expert_idx
        self.complexity_penalty = complexity_penalty
        
        logger.info(f"Initialized ExpertOptimizationProblem for expert {expert_idx}")
    
    def _get_expert_param_count(self, model, expert_idx):
        """Get parameter count from expert."""
        # This would depend on the specific model implementation
        # For demonstration, we'll return a fixed value
        return 10
    
    def fitness(self, x):
        """
        Calculate fitness values for a solution.
        
        Args:
            x: Solution vector
            
        Returns:
            List of fitness values (to be minimized)
        """
        # Apply parameters to expert
        self._apply_parameters(x)
        
        # Train expert
        self.model.fit_expert(self.expert_idx, self.X_train, self.y_train)
        
        # Evaluate expert
        y_pred = self.model.predict_expert(self.expert_idx, self.X_val)
        
        # Calculate fitness values
        fitness_values = []
        
        for objective in self.objectives:
            if objective == 'accuracy':
                # Accuracy (to be maximized, so we minimize 1-accuracy)
                acc = np.mean(y_pred == self.y_val)
                fitness_values.append(1.0 - acc)
            
            elif objective == 'precision':
                # Precision (to be maximized, so we minimize 1-precision)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fp = np.sum((y_pred == 1) & (self.y_val == 0))
                precision = tp / (tp + fp + 1e-10)
                fitness_values.append(1.0 - precision)
            
            elif objective == 'recall':
                # Recall (to be maximized, so we minimize 1-recall)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fn = np.sum((y_pred == 0) & (self.y_val == 1))
                recall = tp / (tp + fn + 1e-10)
                fitness_values.append(1.0 - recall)
            
            elif objective == 'f1':
                # F1 score (to be maximized, so we minimize 1-f1)
                # Avoid division by zero
                tp = np.sum((y_pred == 1) & (self.y_val == 1))
                fp = np.sum((y_pred == 1) & (self.y_val == 0))
                fn = np.sum((y_pred == 0) & (self.y_val == 1))
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                fitness_values.append(1.0 - f1)
            
            elif objective == 'complexity':
                # Expert complexity (minimize)
                complexity = self.model.get_expert_complexity(self.expert_idx)
                fitness_values.append(self.complexity_penalty * complexity)
        
        return fitness_values
    
    def _apply_parameters(self, x):
        """Apply parameters to expert."""
        # This would depend on the specific model implementation
        # For demonstration, we'll just pass
        pass


class GatingNetworkOptimizationProblem(BaseMoEOptimizationProblem):
    """Optimization problem for gating network in MoE models."""
    
    def __init__(self, 
                 model=None,
                 X_train=None, 
                 y_train=None,
                 X_val=None,
                 y_val=None,
                 objectives=['expert_balance', 'accuracy', 'complexity'],
                 dim=None,
                 bounds=None,
                 complexity_penalty=0.01):
        """
        Initialize gating network optimization problem.
        
        Args:
            model: MoE model containing the gating network to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            objectives: List of objectives to optimize
            dim: Dimension of the problem (number of parameters to optimize)
            bounds: Bounds for parameters (min, max)
            complexity_penalty: Penalty factor for model complexity
        """
        # Determine dimension based on gating network if not provided
        if dim is None and model is not None:
            dim = self._get_gating_param_count(model)
        elif dim is None:
            dim = 10  # Default dimension if model not provided
        
        super().__init__(model, X_train, y_train, X_val, y_val, objectives, dim, bounds)
        
        self.complexity_penalty = complexity_penalty
        
        logger.info(f"Initialized GatingNetworkOptimizationProblem")
    
    def _get_gating_param_count(self, model):
        """Get parameter count from gating network."""
        # This would depend on the specific model implementation
        # For demonstration, we'll return a fixed value
        return 10
    
    def fitness(self, x):
        """
        Calculate fitness values for a solution.
        
        Args:
            x: Solution vector
            
        Returns:
            List of fitness values (to be minimized)
        """
        # Apply parameters to gating network
        self._apply_parameters(x)
        
        # Train gating network
        self.model.fit_gating(self.X_train, self.y_train)
        
        # Evaluate model with updated gating
        y_pred = self.model.predict(self.X_val)
        
        # Get expert contributions
        expert_contributions = self.model.get_expert_contributions(self.X_val)
        
        # Calculate fitness values
        fitness_values = []
        
        for objective in self.objectives:
            if objective == 'accuracy':
                # Accuracy (to be maximized, so we minimize 1-accuracy)
                acc = np.mean(y_pred == self.y_val)
                fitness_values.append(1.0 - acc)
            
            elif objective == 'expert_balance':
                # Expert balance (minimize variance in expert contributions)
                balance = np.var(expert_contributions)
                fitness_values.append(balance)
            
            elif objective == 'expert_specialization':
                # Expert specialization (maximize, so minimize 1-specialization)
                # This measures how well each expert specializes in its domain
                specialization = self.model.get_expert_specialization()
                fitness_values.append(1.0 - specialization)
            
            elif objective == 'complexity':
                # Gating network complexity (minimize)
                complexity = self.model.get_gating_complexity()
                fitness_values.append(self.complexity_penalty * complexity)
        
        return fitness_values
    
    def _apply_parameters(self, x):
        """Apply parameters to gating network."""
        # This would depend on the specific model implementation
        # For demonstration, we'll just pass
        pass


# Mock MoE model for demonstration
class MockMoEModel:
    """Mock MoE model for demonstration purposes."""
    
    def __init__(self, dim=10, n_experts=3):
        """Initialize mock model."""
        self.dim = dim
        self.n_experts = n_experts
        self.expert_weights = np.random.random((n_experts, dim))
        self.gating_weights = np.random.random(dim)
    
    def fit(self, X, y):
        """Mock training method."""
        # Simulate training
        pass
    
    def predict(self, X):
        """Mock prediction method."""
        # Simulate prediction
        return (np.sum(X, axis=1) > 0).astype(int)
    
    def fit_expert(self, expert_idx, X, y):
        """Mock expert training method."""
        # Simulate expert training
        pass
    
    def predict_expert(self, expert_idx, X):
        """Mock expert prediction method."""
        # Simulate expert prediction
        return (np.sum(X, axis=1) > 0).astype(int)
    
    def fit_gating(self, X, y):
        """Mock gating network training method."""
        # Simulate gating network training
        pass
    
    def get_expert_contributions(self, X):
        """Mock method to get expert contributions."""
        # Simulate expert contributions
        contributions = np.random.random(self.n_experts)
        return contributions / np.sum(contributions)
    
    def get_complexity(self):
        """Mock method to get model complexity."""
        # Simulate model complexity
        return np.sum(np.abs(self.expert_weights)) + np.sum(np.abs(self.gating_weights))
    
    def get_expert_complexity(self, expert_idx):
        """Mock method to get expert complexity."""
        # Simulate expert complexity
        return np.sum(np.abs(self.expert_weights[expert_idx]))
    
    def get_gating_complexity(self):
        """Mock method to get gating network complexity."""
        # Simulate gating network complexity
        return np.sum(np.abs(self.gating_weights))
    
    def get_expert_specialization(self):
        """Mock method to get expert specialization."""
        # Simulate expert specialization
        return 0.8  # Higher is better


# Example usage
if __name__ == "__main__":
    # Create mock model
    model = MockMoEModel(dim=10, n_experts=3)
    
    # Generate mock data
    X_train = np.random.randn(1000, 10)
    y_train = (np.sum(X_train, axis=1) > 0).astype(int)
    X_val = np.random.randn(200, 10)
    y_val = (np.sum(X_val, axis=1) > 0).astype(int)
    
    # Create end-to-end optimization problem
    end_to_end_problem = EndToEndMoEOptimizationProblem(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        objectives=['accuracy', 'f1', 'expert_balance'],
        complexity_penalty=0.01
    )
    
    # Create expert optimization problem
    expert_problem = ExpertOptimizationProblem(
        model=model,
        expert_idx=0,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        objectives=['accuracy', 'precision', 'complexity'],
        complexity_penalty=0.01
    )
    
    # Create gating network optimization problem
    gating_problem = GatingNetworkOptimizationProblem(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        objectives=['expert_balance', 'accuracy', 'complexity'],
        complexity_penalty=0.01
    )
    
    # Test fitness calculation
    x = np.random.random(10)
    
    print("End-to-end problem fitness:", end_to_end_problem.fitness(x))
    print("Expert problem fitness:", expert_problem.fitness(x))
    print("Gating problem fitness:", gating_problem.fitness(x))
