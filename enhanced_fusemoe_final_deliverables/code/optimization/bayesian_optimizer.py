"""
Bayesian Optimizer for Enhanced FuseMoE

This module provides a Bayesian optimization implementation for hyperparameter tuning
of the Enhanced FuseMoE system for migraine prediction.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    This class implements Bayesian optimization with Gaussian Process regression
    and Expected Improvement acquisition function for efficient hyperparameter tuning.
    
    Attributes:
        param_bounds (Dict[str, Tuple[float, float]]): Parameter bounds for optimization
        objective_function (Callable): Function to optimize
        n_initial_points (int): Number of initial random points
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print progress
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                objective_function: Callable, 
                n_initial_points: int = 5,
                random_state: int = 42,
                verbose: bool = True):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) bounds
            objective_function: Function to optimize (should return a scalar value)
            n_initial_points: Number of initial random points to evaluate
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.param_bounds = param_bounds
        self.objective_function = objective_function
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        self.verbose = verbose
        
        # Set random seed
        np.random.seed(random_state)
        
        # Initialize Gaussian Process with Matern kernel
        self.kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=random_state
        )
        
        # Initialize data structures
        self.X_observed = []  # List of parameter dictionaries
        self.y_observed = []  # List of objective function values
        self.best_params = None
        self.best_score = float('-inf')
        
        # Parameter names and bounds
        self.param_names = list(param_bounds.keys())
        self.param_mins = np.array([param_bounds[name][0] for name in self.param_names])
        self.param_maxs = np.array([param_bounds[name][1] for name in self.param_names])
        
        # Special handling for log-uniform parameters
        self.log_uniform_params = []
        for name in self.param_names:
            if name in ['learning_rate', 'weight_decay'] or 'lr' in name:
                self.log_uniform_params.append(name)
        
        # Special handling for integer parameters
        self.integer_params = []
        for name in self.param_names:
            if name in ['batch_size', 'hidden_dim', 'num_layers', 'top_k']:
                self.integer_params.append(name)
    
    def _dict_to_array(self, params_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert parameter dictionary to array.
        
        Args:
            params_dict: Dictionary mapping parameter names to values
            
        Returns:
            Array of parameter values
        """
        return np.array([params_dict[name] for name in self.param_names])
    
    def _array_to_dict(self, params_array: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter array to dictionary.
        
        Args:
            params_array: Array of parameter values
            
        Returns:
            Dictionary mapping parameter names to values
        """
        params_dict = {}
        for i, name in enumerate(self.param_names):
            value = params_array[i]
            
            # Convert to integer if needed
            if name in self.integer_params:
                value = int(value)
            
            params_dict[name] = value
        
        return params_dict
    
    def _sample_random_params(self) -> Dict[str, float]:
        """
        Sample random parameters within bounds.
        
        Returns:
            Dictionary of randomly sampled parameters
        """
        params_array = np.zeros(len(self.param_names))
        
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_bounds[name]
            
            # Log-uniform sampling for learning rate and weight decay
            if name in self.log_uniform_params:
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                value = 10 ** np.random.uniform(log_min, log_max)
            else:
                value = np.random.uniform(min_val, max_val)
            
            # Convert to integer if needed
            if name in self.integer_params:
                value = int(value)
            
            params_array[i] = value
        
        return self._array_to_dict(params_array)
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate expected improvement acquisition function.
        
        Args:
            X: Array of parameter values to evaluate
            
        Returns:
            Expected improvement values
        """
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Predict mean and std with GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = self.gp.predict(X, return_std=True)
        
        # Calculate improvement
        improvement = mu - self.best_score
        
        # Calculate expected improvement
        Z = np.zeros_like(improvement)
        mask = sigma > 0
        Z[mask] = improvement[mask] / sigma[mask]
        ei = improvement * 0.5 * (1 + np.sign(improvement))
        ei[mask] = improvement[mask] * (0.5 * (1 + np.sign(improvement[mask]))) + sigma[mask] * np.exp(-(Z[mask]**2) / 2) / np.sqrt(2 * np.pi)
        
        return ei
    
    def _find_next_point(self) -> Dict[str, float]:
        """
        Find the next point to evaluate using expected improvement.
        
        Returns:
            Dictionary of parameters to evaluate next
        """
        # Generate random candidate points
        n_candidates = 10000
        X_candidates = np.random.uniform(
            self.param_mins, self.param_maxs, size=(n_candidates, len(self.param_names))
        )
        
        # Apply log-uniform sampling for specific parameters
        for i, name in enumerate(self.param_names):
            if name in self.log_uniform_params:
                log_min = np.log10(self.param_mins[i])
                log_max = np.log10(self.param_maxs[i])
                X_candidates[:, i] = 10 ** np.random.uniform(log_min, log_max, size=n_candidates)
        
        # Calculate expected improvement for all candidates
        ei_values = self._expected_improvement(X_candidates)
        
        # Find candidate with highest expected improvement
        best_idx = np.argmax(ei_values)
        best_candidate = X_candidates[best_idx]
        
        # Convert to dictionary
        params_dict = self._array_to_dict(best_candidate)
        
        return params_dict
    
    def optimize(self, n_iterations: int = 20) -> Tuple[Dict[str, float], float]:
        """
        Run Bayesian optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Tuple of (best_params, best_score)
        """
        # Initial random exploration
        if self.verbose:
            print("Starting initial random exploration...")
        
        for i in range(self.n_initial_points):
            # Sample random parameters
            params = self._sample_random_params()
            
            # Evaluate objective function
            score = self.objective_function(params)
            
            # Store results
            self.X_observed.append(params)
            self.y_observed.append(score)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
            
            if self.verbose:
                print(f"Initial point {i+1}/{self.n_initial_points}: score = {score:.4f}")
        
        # Convert observations to arrays for GP
        X_array = np.array([self._dict_to_array(params) for params in self.X_observed])
        y_array = np.array(self.y_observed)
        
        # Fit Gaussian Process
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X_array, y_array)
        
        # Bayesian optimization iterations
        if self.verbose:
            print("\nStarting Bayesian optimization...")
        
        for i in range(n_iterations):
            # Find next point to evaluate
            next_params = self._find_next_point()
            
            # Evaluate objective function
            score = self.objective_function(next_params)
            
            # Store results
            self.X_observed.append(next_params)
            self.y_observed.append(score)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = next_params.copy()
            
            # Update Gaussian Process
            X_array = np.array([self._dict_to_array(params) for params in self.X_observed])
            y_array = np.array(self.y_observed)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X_array, y_array)
            
            if self.verbose:
                print(f"Iteration {i+1}/{n_iterations}: score = {score:.4f}, best = {self.best_score:.4f}")
        
        if self.verbose:
            print("\nOptimization complete!")
            print(f"Best score: {self.best_score:.4f}")
            print("Best parameters:")
            for name, value in self.best_params.items():
                print(f"  {name}: {value}")
        
        return self.best_params, self.best_score
