"""
Unit tests for the PyGMO model optimizer in the Enhanced FuseMoE system.

This module contains unit tests for the MigraineMoEOptimizer class and related functions.
"""

import unittest
import torch
import numpy as np
import pygmo as pg
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import PyGMO model optimizer
from optimization.pygmo_model_optimizer import MigraineMoEOptimizer
from models.experts.expert_registry import DynamicMigraineMoE, ScalableExpertPool, ExpertRegistry
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusion


class TestMigraineMoEOptimizer(unittest.TestCase):
    """Test cases for the MigraineMoEOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expert_types = ['sleep', 'weather']
        self.expert_dims = {
            'sleep': 4,
            'weather': 3
        }
        self.output_dim = 8
        self.device = torch.device('cpu')
        self.seed = 42
        
        # Create optimizer
        self.optimizer = MigraineMoEOptimizer(
            expert_types=self.expert_types,
            expert_dims=self.expert_dims,
            output_dim=self.output_dim,
            device=self.device,
            seed=self.seed
        )
        
        # Create test parameters
        self.test_params = {
            'expert_hidden_dim': 64,
            'expert_num_layers': 2,
            'expert_dropout': 0.2,
            'gating_hidden_dim': 32,
            'top_k': 1,
            'load_balance_coef': 0.01,
            'noisy_gating': 0,
            'fusion_hidden_dim': 32,
            'fusion_dropout': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'batch_size': 32
        }
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.expert_types, self.expert_types)
        self.assertEqual(self.optimizer.expert_dims, self.expert_dims)
        self.assertEqual(self.optimizer.output_dim, self.output_dim)
        self.assertEqual(self.optimizer.device, self.device)
        self.assertEqual(self.optimizer.seed, self.seed)
        
        # Check parameter bounds
        self.assertIn('expert_hidden_dim', self.optimizer.param_bounds)
        self.assertIn('learning_rate', self.optimizer.param_bounds)
        self.assertIn('batch_size', self.optimizer.param_bounds)
    
    def test_create_model(self):
        """Test model creation."""
        # Create model
        model = self.optimizer.create_model(self.test_params)
        
        # Check model type
        self.assertIsInstance(model, DynamicMigraineMoE)
        
        # Check expert pool
        self.assertEqual(len(model.expert_pool.list_experts()), len(self.expert_types))
        for expert_type in self.expert_types:
            self.assertIn(expert_type, model.expert_pool.list_experts())
        
        # Check gating network
        self.assertIsInstance(model.gating, MigraineGating)
        self.assertEqual(model.gating.num_experts, len(self.expert_types))
        self.assertEqual(model.gating.top_k, self.test_params['top_k'])
        
        # Check fusion mechanism
        self.assertIsInstance(model.fusion, MigraineFusion)
        self.assertEqual(model.fusion.num_experts, len(self.expert_types))
    
    @patch('optimization.pygmo_model_optimizer.MigraineTrainer')
    def test_train_model(self, mock_trainer_class):
        """Test model training."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mock history
        mock_history = {
            'train_history': [{'loss': 0.5, 'accuracy': 0.8}],
            'val_history': [{'loss': 0.4, 'accuracy': 0.85, 'auc': 0.9}]
        }
        mock_trainer.train.return_value = mock_history
        
        # Create model
        model = self.optimizer.create_model(self.test_params)
        
        # Create mock data loaders
        train_loader = MagicMock()
        val_loader = MagicMock()
        
        # Train model
        history, best_val_metrics = self.optimizer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            params=self.test_params,
            num_epochs=5,
            patience=2,
            verbose=False
        )
        
        # Check that trainer was created with correct parameters
        mock_trainer_class.assert_called_once()
        self.assertEqual(mock_trainer_class.call_args[1]['model'], model)
        
        # Check that train was called with correct parameters
        mock_trainer.train.assert_called_once()
        self.assertEqual(mock_trainer.train.call_args[1]['train_loader'], train_loader)
        self.assertEqual(mock_trainer.train.call_args[1]['val_loader'], val_loader)
        self.assertEqual(mock_trainer.train.call_args[1]['num_epochs'], 5)
        
        # Check returned history and metrics
        self.assertEqual(history, mock_history)
        self.assertEqual(best_val_metrics, mock_history['val_history'][0])
    
    @patch('optimization.pygmo_model_optimizer.MigraineTrainer')
    @patch('optimization.pygmo_model_optimizer.generate_classification_report')
    def test_evaluate_model(self, mock_generate_report, mock_trainer_class):
        """Test model evaluation."""
        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Set up mock test metrics
        mock_test_metrics = {'loss': 0.3, 'accuracy': 0.9, 'auc': 0.95}
        mock_trainer.test.return_value = mock_test_metrics
        
        # Set up mock classification report
        mock_report = {'accuracy': 0.9, 'precision': 0.92, 'recall': 0.88, 'f1': 0.9}
        mock_generate_report.return_value = mock_report
        
        # Create model
        model = self.optimizer.create_model(self.test_params)
        
        # Create mock data loader
        test_loader = MagicMock()
        test_loader.__iter__.return_value = [
            (
                {'sleep': torch.randn(2, 4), 'weather': torch.randn(2, 3)},
                torch.tensor([[0], [1]], dtype=torch.float32)
            )
        ]
        
        # Evaluate model
        results = self.optimizer.evaluate_model(
            model=model,
            test_loader=test_loader,
            output_dir=None
        )
        
        # Check that trainer was created with correct parameters
        mock_trainer_class.assert_called_once()
        self.assertEqual(mock_trainer_class.call_args[1]['model'], model)
        
        # Check that test was called with correct parameters
        mock_trainer.test.assert_called_once_with(test_loader)
        
        # Check that classification report was generated
        mock_generate_report.assert_called_once()
        
        # Check returned results
        self.assertEqual(results['metrics'], mock_test_metrics)
        self.assertEqual(results['report'], mock_report)
        self.assertIn('predictions', results)
        self.assertIn('targets', results)
    
    @patch.object(MigraineMoEOptimizer, 'create_model')
    @patch.object(MigraineMoEOptimizer, 'train_model')
    @patch.object(MigraineMoEOptimizer, 'evaluate_model')
    def test_optimize(self, mock_evaluate, mock_train, mock_create):
        """Test model optimization."""
        # Skip this test if pygmo is not available
        try:
            import pygmo
        except ImportError:
            self.skipTest("pygmo not available")
        
        # Create mock model
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        # Set up mock training results
        mock_history = {
            'train_history': [{'loss': 0.5, 'accuracy': 0.8}],
            'val_history': [{'loss': 0.4, 'accuracy': 0.85, 'auc': 0.9}]
        }
        mock_best_val_metrics = {'loss': 0.4, 'accuracy': 0.85, 'auc': 0.9}
        mock_train.return_value = (mock_history, mock_best_val_metrics)
        
        # Set up mock evaluation results
        mock_eval_results = {
            'metrics': {'loss': 0.3, 'accuracy': 0.9, 'auc': 0.95},
            'report': {'accuracy': 0.9, 'precision': 0.92, 'recall': 0.88, 'f1': 0.9},
            'predictions': np.array([[0.1], [0.9]]),
            'targets': np.array([[0], [1]])
        }
        mock_evaluate.return_value = mock_eval_results
        
        # Create mock data loaders
        train_loader = MagicMock()
        val_loader = MagicMock()
        test_loader = MagicMock()
        
        # Run optimization with minimal settings to speed up test
        results = self.optimizer.optimize(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            algorithm='sade',
            pop_size=2,
            generations=2,
            islands=1,
            output_dir=None,
            verbose=False
        )
        
        # Check that model creation, training, and evaluation were called
        mock_create.assert_called()
        mock_train.assert_called()
        mock_evaluate.assert_called_once()
        
        # Check returned results
        self.assertIn('best_params', results)
        self.assertIn('best_model', results)
        self.assertIn('best_fitness', results)
        self.assertIn('optimization_history', results)
        self.assertIn('test_results', results)
        self.assertEqual(results['test_results'], mock_eval_results)


if __name__ == '__main__':
    unittest.main()
