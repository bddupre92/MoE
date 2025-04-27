"""
Unit tests for the PyGMO integration in the Enhanced FuseMoE system.

This module contains unit tests for the EndToEndOptimizationProblem class and related functions.
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import PyGMO integration
from optimization.evolutionary_algorithms.pygmo_problem import EndToEndOptimizationProblem
from optimization.evolutionary_algorithms.optimization_manager import OptimizationManager


class TestEndToEndOptimizationProblem(unittest.TestCase):
    """Test cases for the EndToEndOptimizationProblem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        
        # Create mock data loaders
        self.mock_train_loader = MagicMock()
        self.mock_val_loader = MagicMock()
        
        # Create parameter bounds
        self.param_bounds = {
            'learning_rate': (0.0001, 0.01),
            'weight_decay': (0.0, 0.001),
            'dropout_rate': (0.0, 0.5),
            'hidden_dim': (32, 128)
        }
        
        # Create problem
        self.problem = EndToEndOptimizationProblem(
            model=self.mock_model,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
    
    def test_initialization(self):
        """Test problem initialization."""
        self.assertEqual(self.problem.model, self.mock_model)
        self.assertEqual(self.problem.train_loader, self.mock_train_loader)
        self.assertEqual(self.problem.val_loader, self.mock_val_loader)
        self.assertEqual(self.problem.param_bounds, self.param_bounds)
        self.assertEqual(self.problem.fitness_metric, 'auc')
        self.assertEqual(self.problem.num_epochs, 5)
        self.assertEqual(self.problem.patience, 2)
        self.assertEqual(self.problem.device, torch.device('cpu'))
        
        # Check parameter names and bounds
        self.assertEqual(self.problem.param_names, list(self.param_bounds.keys()))
        self.assertEqual(self.problem.dim, len(self.param_bounds))
        
        # Check bounds
        lb, ub = self.problem.get_bounds()
        self.assertEqual(len(lb), len(self.param_bounds))
        self.assertEqual(len(ub), len(self.param_bounds))
        for i, name in enumerate(self.problem.param_names):
            self.assertEqual(lb[i], self.param_bounds[name][0])
            self.assertEqual(ub[i], self.param_bounds[name][1])
    
    @patch('optimization.evolutionary_algorithms.pygmo_problem.MigraineTrainer')
    def test_fitness(self, mock_trainer_class):
        """Test fitness function."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mock history
        mock_history = {
            'train_history': [{'loss': 0.5, 'accuracy': 0.8}],
            'val_history': [
                {'loss': 0.4, 'accuracy': 0.85, 'auc': 0.9},
                {'loss': 0.35, 'accuracy': 0.87, 'auc': 0.92}
            ]
        }
        mock_trainer.train.return_value = mock_history
        
        # Create test parameters
        x = [0.001, 0.0005, 0.2, 64]
        
        # Calculate fitness
        fitness = self.problem.fitness(x)
        
        # Check that trainer was created with correct parameters
        mock_trainer_class.assert_called_once()
        self.assertEqual(mock_trainer_class.call_args[1]['model'], self.mock_model)
        
        # Check that train was called with correct parameters
        mock_trainer.train.assert_called_once()
        self.assertEqual(mock_trainer.train.call_args[1]['train_loader'], self.mock_train_loader)
        self.assertEqual(mock_trainer.train.call_args[1]['val_loader'], self.mock_val_loader)
        self.assertEqual(mock_trainer.train.call_args[1]['num_epochs'], 5)
        
        # Check returned fitness (negative of best AUC)
        self.assertEqual(fitness[0], -0.92)
        
        # Check that best parameters and fitness were updated
        self.assertEqual(self.problem.best_fitness, 0.92)
        self.assertEqual(len(self.problem.best_params), len(self.param_bounds))
        for name, value in zip(self.problem.param_names, x):
            self.assertEqual(self.problem.best_params[name], value)
    
    def test_get_name(self):
        """Test get_name method."""
        self.assertEqual(self.problem.get_name(), "End-to-End Optimization Problem")
    
    def test_get_extra_info(self):
        """Test get_extra_info method."""
        # Set best parameters
        self.problem.best_params = {
            'learning_rate': 0.001,
            'weight_decay': 0.0005,
            'dropout_rate': 0.2,
            'hidden_dim': 64
        }
        self.problem.best_fitness = 0.92
        
        # Get extra info
        info = self.problem.get_extra_info()
        
        # Check that info contains parameter names and values
        for name, value in self.problem.best_params.items():
            self.assertIn(f"{name}: {value}", info)
        
        # Check that info contains best fitness
        self.assertIn(f"Best fitness: {self.problem.best_fitness}", info)


class TestOptimizationManager(unittest.TestCase):
    """Test cases for the OptimizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model
        self.mock_model = MagicMock()
        
        # Create mock data loaders
        self.mock_train_loader = MagicMock()
        self.mock_val_loader = MagicMock()
        self.mock_test_loader = MagicMock()
        
        # Create parameter bounds
        self.param_bounds = {
            'learning_rate': (0.0001, 0.01),
            'weight_decay': (0.0, 0.001),
            'dropout_rate': (0.0, 0.5),
            'hidden_dim': (32, 128)
        }
        
        # Create optimization manager
        self.manager = OptimizationManager(
            model=self.mock_model,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.model, self.mock_model)
        self.assertEqual(self.manager.train_loader, self.mock_train_loader)
        self.assertEqual(self.manager.val_loader, self.mock_val_loader)
        self.assertEqual(self.manager.test_loader, self.mock_test_loader)
        self.assertEqual(self.manager.param_bounds, self.param_bounds)
        self.assertEqual(self.manager.fitness_metric, 'auc')
        self.assertEqual(self.manager.num_epochs, 5)
        self.assertEqual(self.manager.patience, 2)
        self.assertEqual(self.manager.device, torch.device('cpu'))
    
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.problem')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.algorithm')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.population')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.archipelago')
    @patch('optimization.evolutionary_algorithms.optimization_manager.MigraineTrainer')
    def test_optimize(self, mock_trainer_class, mock_archipelago, mock_population, mock_algorithm, mock_problem):
        """Test optimize method."""
        # Skip this test if pygmo is not available
        try:
            import pygmo
        except ImportError:
            self.skipTest("pygmo not available")
        
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mock history
        mock_history = {
            'train_history': [{'loss': 0.5, 'accuracy': 0.8}],
            'val_history': [{'loss': 0.4, 'accuracy': 0.85, 'auc': 0.9}]
        }
        mock_trainer.train.return_value = mock_history
        
        # Set up mock test results
        mock_test_results = {'loss': 0.3, 'accuracy': 0.9, 'auc': 0.95}
        mock_trainer.test.return_value = mock_test_results
        
        # Set up mock archipelago
        mock_archi = MagicMock()
        mock_archipelago.return_value = mock_archi
        
        # Set up mock champion
        mock_champion = MagicMock()
        mock_champion.x = [0.001, 0.0005, 0.2, 64]
        mock_champion.f = [-0.92]
        mock_archi.get_champions_f.return_value = [mock_champion.f]
        mock_archi.get_champions_x.return_value = [mock_champion.x]
        
        # Run optimization
        results = self.manager.optimize(
            algorithm='sade',
            pop_size=10,
            generations=5,
            islands=2,
            output_dir=None,
            verbose=False
        )
        
        # Check that problem, algorithm, population, and archipelago were created
        mock_problem.assert_called_once()
        mock_algorithm.assert_called_once()
        mock_population.assert_called()
        mock_archipelago.assert_called_once()
        
        # Check that archipelago was evolved
        mock_archi.evolve.assert_called_once()
        
        # Check that test was called
        mock_trainer.test.assert_called_once_with(self.mock_test_loader)
        
        # Check returned results
        self.assertIn('best_params', results)
        self.assertIn('best_fitness', results)
        self.assertIn('test_metrics', results)
        self.assertEqual(results['best_fitness'], 0.92)
        self.assertEqual(results['test_metrics'], mock_test_results)
        
        # Check best parameters
        for name, value in zip(self.manager.problem.param_names, mock_champion.x):
            self.assertEqual(results['best_params'][name], value)


if __name__ == '__main__':
    unittest.main()
