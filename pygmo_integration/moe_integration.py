"""
PyGMO Integration with MoE Model

This module provides the integration between PyGMO optimization
and the Mixture of Experts (MoE) model for migraine prediction.
It connects the optimization framework with the model architecture
and handles data flow between the two components.

Author: Manus AI
Date: April 28, 2025
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import torch

# Import local modules
from enhanced_pygmo_optimizer import PyGMOOptimizer, OptimizationConfig
from optimization_problem import EndToEndMoEOptimizationProblem, ExpertOptimizationProblem, GatingNetworkOptimizationProblem
from fitness_functions import FitnessFunctions, FitnessFunctionFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoEPyGMOIntegration:
    """Integration between MoE model and PyGMO optimization."""
    
    def __init__(self, 
                 moe_model=None,
                 config: OptimizationConfig = None,
                 data_handler=None):
        """
        Initialize MoE-PyGMO integration.
        
        Args:
            moe_model: MoE model to optimize
            config: Optimization configuration
            data_handler: Data handler for loading and preprocessing data
        """
        self.moe_model = moe_model
        self.config = config or OptimizationConfig()
        self.data_handler = data_handler
        self.optimizer = PyGMOOptimizer(self.config)
        self.optimization_results = None
        
        # Create mock model and data if not provided
        if self.moe_model is None:
            logger.warning("No MoE model provided, using mock model for demonstration")
            self.moe_model = self._create_mock_model()
        
        if self.data_handler is None:
            logger.warning("No data handler provided, using mock data for demonstration")
            self.data_handler = self._create_mock_data_handler()
        
        logger.info(f"Initialized MoE-PyGMO integration with config: {self.config}")
    
    def _create_mock_model(self):
        """Create mock MoE model for demonstration."""
        # This is a placeholder - in a real implementation, this would create a proper mock model
        # For demonstration purposes, we'll create a simple object with the necessary methods
        class MockMoEModel:
            def __init__(self):
                self.experts = [f"Expert_{i}" for i in range(3)]
                self.expert_weights = [np.random.random(10) for _ in range(3)]
                self.gating_weights = np.random.random(10)
                self.trained = False
            
            def fit(self, X, y):
                """Train the model."""
                logger.info(f"Training mock MoE model with {len(X)} samples")
                self.trained = True
                return self
            
            def predict(self, X):
                """Make predictions."""
                return (np.random.random(len(X)) > 0.5).astype(int)
            
            def predict_proba(self, X):
                """Predict probabilities."""
                return np.random.random((len(X), 2))
            
            def get_expert_contributions(self, X):
                """Get expert contributions."""
                contributions = np.random.random(len(self.experts))
                return contributions / np.sum(contributions)
            
            def get_expert_outputs(self, X):
                """Get outputs from each expert."""
                return [np.random.random(len(X)) for _ in range(len(self.experts))]
            
            def get_complexity(self):
                """Get model complexity."""
                return np.sum([np.sum(np.abs(w)) for w in self.expert_weights]) + np.sum(np.abs(self.gating_weights))
            
            def set_parameters(self, params):
                """Set model parameters."""
                logger.info(f"Setting model parameters: {params[:5]}...")
                # In a real implementation, this would update the model parameters
                return self
            
            def get_parameters(self):
                """Get model parameters."""
                # In a real implementation, this would return the actual model parameters
                return np.concatenate([w.flatten() for w in self.expert_weights] + [self.gating_weights.flatten()])
        
        return MockMoEModel()
    
    def _create_mock_data_handler(self):
        """Create mock data handler for demonstration."""
        # This is a placeholder - in a real implementation, this would create a proper mock data handler
        # For demonstration purposes, we'll create a simple object with the necessary methods
        class MockDataHandler:
            def __init__(self):
                self.X_train = np.random.random((1000, 10))
                self.y_train = (np.random.random(1000) > 0.5).astype(int)
                self.X_val = np.random.random((200, 10))
                self.y_val = (np.random.random(200) > 0.5).astype(int)
                self.X_test = np.random.random((100, 10))
                self.y_test = (np.random.random(100) > 0.5).astype(int)
            
            def get_train_data(self):
                """Get training data."""
                return self.X_train, self.y_train
            
            def get_val_data(self):
                """Get validation data."""
                return self.X_val, self.y_val
            
            def get_test_data(self):
                """Get test data."""
                return self.X_test, self.y_test
            
            def preprocess(self, X, y=None):
                """Preprocess data."""
                if y is not None:
                    return X, y
                return X
        
        return MockDataHandler()
    
    def optimize_model(self, optimization_type='end_to_end'):
        """
        Optimize MoE model using PyGMO.
        
        Args:
            optimization_type: Type of optimization ('end_to_end', 'expert', 'gating')
            
        Returns:
            Optimized model
        """
        logger.info(f"Starting {optimization_type} optimization")
        
        # Get data
        X_train, y_train = self.data_handler.get_train_data()
        X_val, y_val = self.data_handler.get_val_data()
        
        # Create optimization problem based on type
        if optimization_type == 'end_to_end':
            problem_class = EndToEndMoEOptimizationProblem
            problem_args = {
                'model': self.moe_model,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'objectives': self.config.objectives,
                'expert_weights': self.config.expert_weights,
                'gating_params': self.config.gating_params,
                'complexity_penalty': 0.01
            }
        
        elif optimization_type == 'expert':
            problem_class = ExpertOptimizationProblem
            problem_args = {
                'model': self.moe_model,
                'expert_idx': 0,  # Could be parameterized to optimize specific expert
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'objectives': self.config.objectives,
                'complexity_penalty': 0.01
            }
        
        elif optimization_type == 'gating':
            problem_class = GatingNetworkOptimizationProblem
            problem_args = {
                'model': self.moe_model,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'objectives': self.config.objectives,
                'complexity_penalty': 0.01
            }
        
        else:
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
        
        # Run optimization
        self.optimization_results = self.optimizer.optimize(problem_class, **problem_args)
        
        # Apply optimized parameters to model
        optimized_model = self.apply_optimization_results()
        
        logger.info(f"Completed {optimization_type} optimization")
        return optimized_model
    
    def apply_optimization_results(self, results=None):
        """
        Apply optimization results to model.
        
        Args:
            results: Optimization results to apply (default: use stored results)
            
        Returns:
            Updated model
        """
        if results is None:
            if self.optimization_results is None:
                logger.warning("No optimization results to apply")
                return self.moe_model
            results = self.optimization_results
        
        # Extract parameters from results
        params = results[0]
        
        # Apply parameters to model
        self.moe_model.set_parameters(params)
        
        logger.info("Applied optimization results to model")
        return self.moe_model
    
    def evaluate_model(self, model=None, data=None):
        """
        Evaluate model performance.
        
        Args:
            model: Model to evaluate (default: use stored model)
            data: Data to evaluate on (default: use test data)
            
        Returns:
            Dictionary of performance metrics
        """
        if model is None:
            model = self.moe_model
        
        if data is None:
            X_test, y_test = self.data_handler.get_test_data()
        else:
            X_test, y_test = data
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = FitnessFunctions.accuracy(y_test, y_pred, minimize=False)
        metrics['precision'] = FitnessFunctions.precision(y_test, y_pred, minimize=False)
        metrics['recall'] = FitnessFunctions.recall(y_test, y_pred, minimize=False)
        metrics['f1'] = FitnessFunctions.f1(y_test, y_pred, minimize=False)
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['auc'] = FitnessFunctions.auc(y_test, y_proba, minimize=False)
            except Exception as e:
                logger.warning(f"Error calculating AUC: {e}")
                metrics['auc'] = 0.5
        
        # False positive/negative rates
        metrics['false_positive_rate'] = FitnessFunctions.false_positive_rate(y_test, y_pred, minimize=False)
        metrics['false_negative_rate'] = FitnessFunctions.false_negative_rate(y_test, y_pred, minimize=False)
        
        # Expert-related metrics
        expert_contributions = model.get_expert_contributions(X_test)
        metrics['expert_balance'] = FitnessFunctions.expert_balance(expert_contributions, minimize=False)
        
        # Model complexity
        metrics['model_complexity'] = model.get_complexity()
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def compare_models(self, baseline_model, optimized_model, data=None):
        """
        Compare performance of baseline and optimized models.
        
        Args:
            baseline_model: Baseline model
            optimized_model: Optimized model
            data: Data to evaluate on (default: use test data)
            
        Returns:
            DataFrame with performance comparison
        """
        if data is None:
            X_test, y_test = self.data_handler.get_test_data()
        else:
            X_test, y_test = data
        
        # Evaluate models
        baseline_metrics = self.evaluate_model(baseline_model, (X_test, y_test))
        optimized_metrics = self.evaluate_model(optimized_model, (X_test, y_test))
        
        # Create comparison DataFrame
        metrics_df = pd.DataFrame({
            'Baseline': baseline_metrics,
            'Optimized': optimized_metrics
        })
        
        # Calculate improvement
        metrics_df['Improvement'] = metrics_df['Optimized'] - metrics_df['Baseline']
        metrics_df['Improvement (%)'] = (metrics_df['Improvement'] / metrics_df['Baseline'] * 100).round(2)
        
        logger.info(f"Model comparison:\n{metrics_df}")
        return metrics_df
    
    def save_results(self, output_dir=None):
        """
        Save optimization results and model.
        
        Args:
            output_dir: Directory to save results (default: use config output_dir)
            
        Returns:
            Dictionary of saved file paths
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save optimization results
        if self.optimization_results is not None:
            results_path = os.path.join(output_dir, 'optimization_results.npz')
            np.savez(results_path, 
                     parameters=self.optimization_results[0], 
                     fitness=self.optimization_results[1])
            saved_files['optimization_results'] = results_path
        
        # Save model if it has a save method
        if hasattr(self.moe_model, 'save'):
            model_path = os.path.join(output_dir, 'optimized_model.pkl')
            self.moe_model.save(model_path)
            saved_files['model'] = model_path
        
        # Save configuration
        config_path = os.path.join(output_dir, 'optimization_config.json')
        pd.Series(self.config.to_dict()).to_json(config_path)
        saved_files['config'] = config_path
        
        logger.info(f"Saved results to {output_dir}")
        return saved_files
    
    def load_results(self, input_dir=None):
        """
        Load optimization results and model.
        
        Args:
            input_dir: Directory to load results from (default: use config output_dir)
            
        Returns:
            Loaded model
        """
        input_dir = input_dir or self.config.output_dir
        
        # Load optimization results
        results_path = os.path.join(input_dir, 'optimization_results.npz')
        if os.path.exists(results_path):
            results_data = np.load(results_path)
            self.optimization_results = (results_data['parameters'], results_data['fitness'])
        
        # Load model if it has a load method
        if hasattr(self.moe_model, 'load'):
            model_path = os.path.join(input_dir, 'optimized_model.pkl')
            if os.path.exists(model_path):
                self.moe_model.load(model_path)
        
        # Load configuration
        config_path = os.path.join(input_dir, 'optimization_config.json')
        if os.path.exists(config_path):
            config_dict = pd.read_json(config_path, typ='series').to_dict()
            self.config = OptimizationConfig.from_dict(config_dict)
            self.optimizer = PyGMOOptimizer(self.config)
        
        logger.info(f"Loaded results from {input_dir}")
        return self.moe_model


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = OptimizationConfig(
        optimization_type='end_to_end',
        algorithm_type='nsga2',
        population_size=50,
        generations=20,
        islands=3,
        objectives=['accuracy', 'f1', 'expert_balance'],
        output_dir='./output'
    )
    
    # Create integration
    integration = MoEPyGMOIntegration(config=config)
    
    # Optimize model
    optimized_model = integration.optimize_model()
    
    # Evaluate model
    metrics = integration.evaluate_model()
    
    # Compare with baseline
    baseline_model = integration._create_mock_model()
    comparison = integration.compare_models(baseline_model, optimized_model)
    
    # Save results
    saved_files = integration.save_results()
    
    print("Optimization completed successfully!")
    print(f"Metrics: {metrics}")
    print(f"Saved files: {saved_files}")
