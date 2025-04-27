"""
Unit tests for the evaluation metrics in the Enhanced FuseMoE system.

This module contains unit tests for the evaluation metrics components.
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import evaluation components
from utils.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    calculate_expert_contribution_metrics,
    MigrainePerformanceMetrics
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for the evaluation metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data for classification metrics
        self.y_true_cls = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred_cls = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        self.y_prob_cls = np.array([0.2, 0.8, 0.3, 0.4, 0.1, 0.9, 0.6, 0.7, 0.2, 0.3])
        
        # Create test data for regression metrics
        self.y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.y_pred_reg = np.array([1.2, 2.1, 2.8, 4.2, 4.9, 6.1, 7.2, 7.8, 9.2, 9.7])
        
        # Create test data for expert contribution metrics
        self.expert_gates = np.array([
            [0.7, 0.3, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3, 0.7],
            [0.4, 0.3, 0.2, 0.1]
        ])
        self.expert_names = ['sleep', 'weather', 'stress_diet', 'physio']
    
    def test_calculate_classification_metrics(self):
        """Test calculate_classification_metrics function."""
        # Calculate metrics
        metrics = calculate_classification_metrics(
            y_true=self.y_true_cls,
            y_pred=self.y_pred_cls,
            y_prob=self.y_prob_cls
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check specific metric values
        self.assertAlmostEqual(metrics['accuracy'], 0.7, places=2)
        self.assertAlmostEqual(metrics['precision'], 0.75, places=2)
        self.assertAlmostEqual(metrics['recall'], 0.6, places=2)
        self.assertAlmostEqual(metrics['f1'], 0.6667, places=2)
        
        # Check confusion matrix shape
        self.assertEqual(metrics['confusion_matrix'].shape, (2, 2))
    
    def test_calculate_regression_metrics(self):
        """Test calculate_regression_metrics function."""
        # Calculate metrics
        metrics = calculate_regression_metrics(
            y_true=self.y_true_reg,
            y_pred=self.y_pred_reg
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'mse', 'rmse', 'mae', 'r2', 'explained_variance'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check specific metric values
        self.assertLess(metrics['mse'], 0.1)
        self.assertLess(metrics['rmse'], 0.3)
        self.assertLess(metrics['mae'], 0.25)
        self.assertGreater(metrics['r2'], 0.95)
        self.assertGreater(metrics['explained_variance'], 0.95)
    
    def test_calculate_expert_contribution_metrics(self):
        """Test calculate_expert_contribution_metrics function."""
        # Calculate metrics
        metrics = calculate_expert_contribution_metrics(
            expert_gates=self.expert_gates,
            expert_names=self.expert_names
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'expert_selection_frequency', 'expert_average_weight', 'expert_max_weight'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check specific metric values
        self.assertEqual(len(metrics['expert_selection_frequency']), 4)
        self.assertEqual(len(metrics['expert_average_weight']), 4)
        self.assertEqual(len(metrics['expert_max_weight']), 4)
        
        # Check that sleep expert has highest average weight
        self.assertEqual(
            np.argmax([metrics['expert_average_weight'][name] for name in self.expert_names]),
            0  # Index of 'sleep' in expert_names
        )


class TestMigrainePerformanceMetrics(unittest.TestCase):
    """Test cases for the MigrainePerformanceMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock data loaders
        self.mock_train_loader = MagicMock()
        self.mock_val_loader = MagicMock()
        self.mock_test_loader = MagicMock()
        
        # Create mock model
        self.mock_model = MagicMock()
        
        # Create test data
        self.X_train = [torch.randn(10, 10) for _ in range(4)]  # 4 experts, 10 samples each
        self.y_train = torch.randint(0, 2, (10,))
        
        self.X_val = [torch.randn(5, 10) for _ in range(4)]  # 4 experts, 5 samples each
        self.y_val = torch.randint(0, 2, (5,))
        
        self.X_test = [torch.randn(8, 10) for _ in range(4)]  # 4 experts, 8 samples each
        self.y_test = torch.randint(0, 2, (8,))
        
        # Set up mock data loaders to return test data
        def train_loader_iter():
            yield self.X_train, self.y_train
        
        def val_loader_iter():
            yield self.X_val, self.y_val
        
        def test_loader_iter():
            yield self.X_test, self.y_test
        
        self.mock_train_loader.__iter__ = train_loader_iter
        self.mock_val_loader.__iter__ = val_loader_iter
        self.mock_test_loader.__iter__ = test_loader_iter
        
        # Create performance metrics
        self.metrics = MigrainePerformanceMetrics(
            model=self.mock_model,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            device=torch.device('cpu')
        )
    
    def test_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.model, self.mock_model)
        self.assertEqual(self.metrics.train_loader, self.mock_train_loader)
        self.assertEqual(self.metrics.val_loader, self.mock_val_loader)
        self.assertEqual(self.metrics.test_loader, self.mock_test_loader)
        self.assertEqual(self.metrics.device, torch.device('cpu'))
    
    @patch('utils.evaluation.metrics.calculate_classification_metrics')
    def test_calculate_metrics(self, mock_calculate_metrics):
        """Test calculate_metrics method."""
        # Set up mock to return predefined metrics
        mock_calculate_metrics.return_value = {
            'accuracy': 0.9,
            'precision': 0.85,
            'recall': 0.88,
            'f1': 0.86,
            'auc': 0.95,
            'confusion_matrix': np.array([[4, 1], [0, 3]])
        }
        
        # Set up mock model to return predefined outputs
        self.mock_model.return_value = (
            torch.tensor([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6]),  # Probabilities
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),  # Predictions
            None  # Expert gates (not used in this test)
        )
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(data_loader=self.mock_test_loader)
        
        # Check that mock was called
        mock_calculate_metrics.assert_called_once()
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'confusion_matrix'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check specific metric values
        self.assertEqual(metrics['accuracy'], 0.9)
        self.assertEqual(metrics['precision'], 0.85)
        self.assertEqual(metrics['recall'], 0.88)
        self.assertEqual(metrics['f1'], 0.86)
        self.assertEqual(metrics['auc'], 0.95)
    
    @patch('utils.evaluation.metrics.calculate_expert_contribution_metrics')
    def test_calculate_expert_contributions(self, mock_calculate_contributions):
        """Test calculate_expert_contributions method."""
        # Set up mock to return predefined metrics
        mock_calculate_contributions.return_value = {
            'expert_selection_frequency': {
                'expert_0': 0.35,
                'expert_1': 0.25,
                'expert_2': 0.20,
                'expert_3': 0.20
            },
            'expert_average_weight': {
                'expert_0': 0.4,
                'expert_1': 0.3,
                'expert_2': 0.2,
                'expert_3': 0.1
            },
            'expert_max_weight': {
                'expert_0': 0.8,
                'expert_1': 0.7,
                'expert_2': 0.6,
                'expert_3': 0.5
            }
        }
        
        # Set up mock model to return predefined outputs
        self.mock_model.return_value = (
            torch.tensor([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6]),  # Probabilities
            torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),  # Predictions
            torch.tensor([  # Expert gates
                [0.7, 0.3, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.3, 0.7],
                [0.4, 0.3, 0.2, 0.1],
                [0.1, 0.2, 0.3, 0.4],
                [0.3, 0.3, 0.3, 0.1],
                [0.2, 0.2, 0.2, 0.4]
            ])
        )
        
        # Calculate expert contributions
        contributions = self.metrics.calculate_expert_contributions(
            data_loader=self.mock_test_loader,
            expert_names=['sleep', 'weather', 'stress_diet', 'physio']
        )
        
        # Check that mock was called
        mock_calculate_contributions.assert_called_once()
        
        # Check that all expected metrics are present
        expected_metrics = [
            'expert_selection_frequency', 'expert_average_weight', 'expert_max_weight'
        ]
        for metric in expected_metrics:
            self.assertIn(metric, contributions)
        
        # Check specific metric values
        self.assertEqual(contributions['expert_selection_frequency']['expert_0'], 0.35)
        self.assertEqual(contributions['expert_average_weight']['expert_1'], 0.3)
        self.assertEqual(contributions['expert_max_weight']['expert_2'], 0.6)
    
    @patch('numpy.savez')
    def test_save_metrics(self, mock_savez):
        """Test _save_metrics method."""
        # Set up test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.6, 0.9, 0.1])
        
        # Call _save_metrics method
        self.metrics._save_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            output_dir='/tmp',
            X_test_list=self.X_test
        )
        
        # Check that savez was called with correct arguments
        mock_savez.assert_called_once()
        args, kwargs = mock_savez.call_args
        
        # Check that the first argument is the correct file path
        self.assertEqual(args[0], '/tmp/test_predictions.npz')
        
        # Check that the correct arrays were saved
        self.assertIn('y_test', kwargs)
        self.assertIn('y_pred_test', kwargs)
        self.assertIn('y_prob_test', kwargs)
        
        # Check array values
        np.testing.assert_array_equal(kwargs['y_test'], y_true)
        np.testing.assert_array_equal(kwargs['y_pred_test'], y_pred)
        np.testing.assert_array_equal(kwargs['y_prob_test'], y_prob)


if __name__ == '__main__':
    unittest.main()
