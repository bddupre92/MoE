"""
Unit tests for the training pipeline in the Enhanced FuseMoE system.

This module contains unit tests for the training pipeline including:
- MigraineTrainer class
- PyGMOTrainingPipeline class
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import training pipeline
from utils.training_pipeline import MigraineTrainer, PyGMOTrainingPipeline
from models.experts.expert_registry import DynamicMigraineMoE


class TestMigraineTrainer(unittest.TestCase):
    """Test cases for the MigraineTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model
        self.model = MagicMock(spec=DynamicMigraineMoE)
        self.model.to = MagicMock(return_value=self.model)
        
        # Create mock criterion, optimizer, and scheduler
        self.criterion = MagicMock(spec=nn.BCEWithLogitsLoss)
        self.optimizer = MagicMock(spec=optim.Adam)
        self.scheduler = MagicMock(spec=optim.lr_scheduler.ReduceLROnPlateau)
        
        # Create trainer
        self.trainer = MigraineTrainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=torch.device('cpu'),
            scheduler=self.scheduler,
            early_stopping_patience=5
        )
        
        # Create sample data
        self.batch_size = 4
        self.num_features = {
            'sleep': 10,
            'weather': 8,
            'stress_diet': 12,
            'physio': 15
        }
        
        # Create sample inputs and targets
        self.inputs = {
            'sleep': torch.randn(self.batch_size, self.num_features['sleep']),
            'weather': torch.randn(self.batch_size, self.num_features['weather']),
            'stress_diet': torch.randn(self.batch_size, self.num_features['stress_diet']),
            'physio': torch.randn(self.batch_size, self.num_features['physio'])
        }
        self.targets = torch.randint(0, 2, (self.batch_size,), dtype=torch.float32)
        
        # Create DataLoader
        self.dataset = TensorDataset(
            self.inputs['sleep'],
            self.inputs['weather'],
            self.inputs['stress_diet'],
            self.inputs['physio'],
            self.targets
        )
        
        # Custom collate function to create dictionary of inputs
        def collate_fn(batch):
            sleep = torch.stack([item[0] for item in batch])
            weather = torch.stack([item[1] for item in batch])
            stress_diet = torch.stack([item[2] for item in batch])
            physio = torch.stack([item[3] for item in batch])
            targets = torch.stack([item[4] for item in batch])
            
            inputs = {
                'sleep': sleep,
                'weather': weather,
                'stress_diet': stress_diet,
                'physio': physio
            }
            
            return inputs, targets
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def test_initialization(self):
        """Test initialization of MigraineTrainer."""
        # Test with default parameters
        trainer = MigraineTrainer(model=self.model)
        
        # Check that model was moved to device
        self.model.to.assert_called_once()
        
        # Check default values
        self.assertIsInstance(trainer.criterion, nn.BCEWithLogitsLoss)
        self.assertIsInstance(trainer.optimizer, optim.Optimizer)
        self.assertEqual(trainer.early_stopping_patience, 10)
        self.assertEqual(trainer.best_val_metric, 0.0)
        self.assertEqual(trainer.epochs_without_improvement, 0)
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        # Mock model forward pass
        predictions = torch.randn(self.batch_size, 1)
        load_balancing_loss = torch.tensor(0.1)
        self.model.return_value = (predictions, load_balancing_loss)
        
        # Mock criterion
        loss = torch.tensor(0.5)
        self.criterion.return_value = loss
        
        # Train for one epoch
        metrics = self.trainer.train_epoch(self.dataloader)
        
        # Check that optimizer.zero_grad was called
        self.optimizer.zero_grad.assert_called()
        
        # Check that optimizer.step was called
        self.optimizer.step.assert_called()
        
        # Check that model was called with training=True
        self.model.assert_called_with(self.inputs, training=True)
        
        # Check that metrics were calculated
        self.assertIn('loss', metrics)
        self.assertIn('load_balancing_loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
    
    def test_validate(self):
        """Test validation."""
        # Mock model forward pass
        predictions = torch.randn(self.batch_size, 1)
        self.model.return_value = predictions
        
        # Mock criterion
        loss = torch.tensor(0.5)
        self.criterion.return_value = loss
        
        # Validate
        metrics = self.trainer.validate(self.dataloader)
        
        # Check that model was called with training=False
        self.model.assert_called_with(self.inputs, training=False)
        
        # Check that metrics were calculated
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
    
    def test_test(self):
        """Test testing."""
        # Mock validate method
        self.trainer.validate = MagicMock(return_value={'loss': 0.5, 'auc': 0.8})
        
        # Test
        metrics = self.trainer.test(self.dataloader)
        
        # Check that validate was called
        self.trainer.validate.assert_called_once_with(self.dataloader)
        
        # Check that metrics were returned
        self.assertEqual(metrics, {'loss': 0.5, 'auc': 0.8})
    
    def test_train(self):
        """Test training for multiple epochs."""
        # Mock train_epoch and validate methods
        self.trainer.train_epoch = MagicMock(return_value={'loss': 0.5, 'auc': 0.7})
        self.trainer.validate = MagicMock(return_value={'loss': 0.6, 'auc': 0.65})
        
        # Create temporary directory for checkpoints
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train for 3 epochs
            history = self.trainer.train(
                train_loader=self.dataloader,
                val_loader=self.dataloader,
                num_epochs=3,
                verbose=False,
                checkpoint_dir=temp_dir
            )
            
            # Check that train_epoch and validate were called 3 times
            self.assertEqual(self.trainer.train_epoch.call_count, 3)
            self.assertEqual(self.trainer.validate.call_count, 3)
            
            # Check that scheduler.step was called 3 times
            self.assertEqual(self.scheduler.step.call_count, 3)
            
            # Check that history was returned
            self.assertIn('train_history', history)
            self.assertIn('val_history', history)
            self.assertEqual(len(history['train_history']), 3)
            self.assertEqual(len(history['val_history']), 3)
            
            # Check that checkpoint file was created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'best_model.pt')))
    
    def test_early_stopping(self):
        """Test early stopping."""
        # Mock train_epoch and validate methods
        self.trainer.train_epoch = MagicMock(return_value={'loss': 0.5, 'auc': 0.7})
        self.trainer.validate = MagicMock(return_value={'loss': 0.6, 'auc': 0.65})
        
        # Set early stopping patience to 2
        self.trainer.early_stopping_patience = 2
        
        # Train for 10 epochs (should stop after 3 due to early stopping)
        history = self.trainer.train(
            train_loader=self.dataloader,
            val_loader=self.dataloader,
            num_epochs=10,
            verbose=False
        )
        
        # Check that train_epoch and validate were called 3 times
        self.assertEqual(self.trainer.train_epoch.call_count, 3)
        self.assertEqual(self.trainer.validate.call_count, 3)
        
        # Check that history was returned with 3 entries
        self.assertEqual(len(history['train_history']), 3)
        self.assertEqual(len(history['val_history']), 3)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
            
            # Mock model and optimizer state_dict
            self.model.state_dict.return_value = {'weight': torch.tensor([1.0])}
            self.optimizer.state_dict.return_value = {'lr': 0.001}
            self.scheduler.state_dict.return_value = {'factor': 0.1}
            
            # Set best_val_metric and epochs_without_improvement
            self.trainer.best_val_metric = 0.8
            self.trainer.epochs_without_improvement = 2
            
            # Save checkpoint
            self.trainer.save_checkpoint(checkpoint_path)
            
            # Check that checkpoint file was created
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Reset trainer
            self.trainer.best_val_metric = 0.0
            self.trainer.epochs_without_improvement = 0
            
            # Mock torch.load
            checkpoint = {
                'model_state_dict': {'weight': torch.tensor([2.0])},
                'optimizer_state_dict': {'lr': 0.002},
                'scheduler_state_dict': {'factor': 0.2},
                'best_val_metric': 0.9,
                'epochs_without_improvement': 3
            }
            
            with patch('torch.load', return_value=checkpoint):
                # Load checkpoint
                self.trainer.load_checkpoint(checkpoint_path)
                
                # Check that model and optimizer state_dict were loaded
                self.model.load_state_dict.assert_called_once_with({'weight': torch.tensor([2.0])})
                self.optimizer.load_state_dict.assert_called_once_with({'lr': 0.002})
                self.scheduler.load_state_dict.assert_called_once_with({'factor': 0.2})
                
                # Check that best_val_metric and epochs_without_improvement were loaded
                self.assertEqual(self.trainer.best_val_metric, 0.9)
                self.assertEqual(self.trainer.epochs_without_improvement, 3)


class TestPyGMOTrainingPipeline(unittest.TestCase):
    """Test cases for the PyGMOTrainingPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock functions
        self.create_model_fn = MagicMock()
        self.train_fn = MagicMock()
        self.evaluate_fn = MagicMock()
        self.optimization_manager = MagicMock()
        
        # Create pipeline
        self.pipeline = PyGMOTrainingPipeline(
            create_model_fn=self.create_model_fn,
            train_fn=self.train_fn,
            evaluate_fn=self.evaluate_fn,
            optimization_manager=self.optimization_manager
        )
        
        # Create sample data
        self.train_data = {
            'sleep': np.random.randn(100, 10),
            'weather': np.random.randn(100, 8),
            'stress_diet': np.random.randn(100, 12),
            'physio': np.random.randn(100, 15)
        }
        self.train_targets = np.random.randint(0, 2, (100,))
        
        self.val_data = {
            'sleep': np.random.randn(20, 10),
            'weather': np.random.randn(20, 8),
            'stress_diet': np.random.randn(20, 12),
            'physio': np.random.randn(20, 15)
        }
        self.val_targets = np.random.randint(0, 2, (20,))
        
        self.test_data = {
            'sleep': np.random.randn(30, 10),
            'weather': np.random.randn(30, 8),
            'stress_diet': np.random.randn(30, 12),
            'physio': np.random.randn(30, 15)
        }
        self.test_targets = np.random.randint(0, 2, (30,))
        
        # Create sample configs
        self.optimization_config = {
            'algorithm': 'de',
            'pop_size': 10,
            'generations': 5,
            'islands': 2
        }
        
        self.training_config = {
            'num_epochs': 20,
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 0.0001
        }
    
    def test_initialization(self):
        """Test initialization of PyGMOTrainingPipeline."""
        # Check that attributes were set correctly
        self.assertEqual(self.pipeline.create_model_fn, self.create_model_fn)
        self.assertEqual(self.pipeline.train_fn, self.train_fn)
        self.assertEqual(self.pipeline.evaluate_fn, self.evaluate_fn)
        self.assertEqual(self.pipeline.optimization_manager, self.optimization_manager)
    
    def test_optimize_and_train(self):
        """Test optimize_and_train method."""
        # Mock optimization_manager methods
        expert_results = {'expert_params': {'hidden_dim': 64}}
        gating_results = {'gating_params': {'top_k': 2}}
        self.optimization_manager.optimize_experts.return_value = expert_results
        self.optimization_manager.optimize_gating.return_value = gating_results
        
        # Mock model
        model = MagicMock()
        self.optimization_manager.create_optimized_model.return_value = model
        
        # Mock train_fn and evaluate_fn
        training_history = {'train_history': [], 'val_history': []}
        self.train_fn.return_value = training_history
        
        test_metrics = {'accuracy': 0.85, 'auc': 0.9}
        self.evaluate_fn.return_value = test_metrics
        
        # Optimize and train
        results = self.pipeline.optimize_and_train(
            train_data=self.train_data,
            train_targets=self.train_targets,
            val_data=self.val_data,
            val_targets=self.val_targets,
            test_data=self.test_data,
            test_targets=self.test_targets,
            optimization_config=self.optimization_config,
            training_config=self.training_config
        )
        
        # Check that optimization_manager methods were called
        self.optimization_manager.optimize_experts.assert_called_once()
        self.optimization_manager.optimize_gating.assert_called_once()
        self.optimization_manager.create_optimized_model.assert_called_once_with(
            expert_results=expert_results,
            gating_results=gating_results
        )
        
        # Check that train_fn was called
        self.train_fn.assert_called_once_with(
            model=model,
            train_data=self.train_data,
            train_targets=self.train_targets,
            val_data=self.val_data,
            val_targets=self.val_targets,
            **self.training_config
        )
        
        # Check that evaluate_fn was called
        self.evaluate_fn.assert_called_once_with(
            model=model,
            test_data=self.test_data,
            test_targets=self.test_targets
        )
        
        # Check that results were returned
        self.assertIn('expert_results', results)
        self.assertIn('gating_results', results)
        self.assertIn('model', results)
        self.assertIn('training_history', results)
        self.assertIn('test_metrics', results)
        self.assertEqual(results['expert_results'], expert_results)
        self.assertEqual(results['gating_results'], gating_results)
        self.assertEqual(results['model'], model)
        self.assertEqual(results['training_history'], training_history)
        self.assertEqual(results['test_metrics'], test_metrics)
    
    def test_optimize_and_train_with_default_config(self):
        """Test optimize_and_train method with default config."""
        # Mock optimization_manager methods
        expert_results = {'expert_params': {'hidden_dim': 64}}
        gating_results = {'gating_params': {'top_k': 2}}
        self.optimization_manager.optimize_experts.return_value = expert_results
        self.optimization_manager.optimize_gating.return_value = gating_results
        
        # Mock model
        model = MagicMock()
        self.optimization_manager.create_optimized_model.return_value = model
        
        # Mock train_fn and evaluate_fn
        training_history = {'train_history': [], 'val_history': []}
        self.train_fn.return_value = training_history
        
        test_metrics = {'accuracy': 0.85, 'auc': 0.9}
        self.evaluate_fn.return_value = test_metrics
        
        # Optimize and train with empty configs
        results = self.pipeline.optimize_and_train(
            train_data=self.train_data,
            train_targets=self.train_targets,
            val_data=self.val_data,
            val_targets=self.val_targets,
            test_data=self.test_data,
            test_targets=self.test_targets,
            optimization_config={},
            training_config={}
        )
        
        # Check that optimization_manager methods were called with default values
        self.optimization_manager.optimize_experts.assert_called_once()
        self.optimization_manager.optimize_gating.assert_called_once()
        
        # Check that results were returned
        self.assertIn('expert_results', results)
        self.assertIn('gating_results', results)
        self.assertIn('model', results)
        self.assertIn('training_history', results)
        self.assertIn('test_metrics', results)


if __name__ == '__main__':
    unittest.main()
