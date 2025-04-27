"""
Unit tests for the evaluation metrics in the Enhanced FuseMoE system.

This module contains unit tests for the metrics calculation functions and classification report generation.
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import evaluation metrics
from utils.evaluation.metrics import (
    calculate_metrics, 
    generate_classification_report,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for the evaluation metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test predictions and targets
        self.y_pred = np.array([[0.1], [0.9], [0.4], [0.6], [0.3], [0.8], [0.2], [0.7]])
        self.y_true = np.array([[0], [1], [0], [1], [0], [1], [0], [0]])
        
        # Create binary predictions
        self.y_pred_binary = (self.y_pred > 0.5).astype(int)
        
        # Create temporary directory for output
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_output')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Calculate metrics
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        # Check that all expected metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
        
        # Check metric values
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)
        self.assertGreaterEqual(metrics['recall'], 0.0)
        self.assertLessEqual(metrics['recall'], 1.0)
        self.assertGreaterEqual(metrics['f1'], 0.0)
        self.assertLessEqual(metrics['f1'], 1.0)
        self.assertGreaterEqual(metrics['auc'], 0.0)
        self.assertLessEqual(metrics['auc'], 1.0)
        
        # Check specific values
        # For this test data, we expect:
        # - 5/8 correct predictions (accuracy)
        # - 3/4 true positives out of all positive predictions (precision)
        # - 3/3 true positives out of all actual positives (recall)
        # - F1 score is the harmonic mean of precision and recall
        self.assertAlmostEqual(metrics['accuracy'], 0.625, places=3)
        self.assertAlmostEqual(metrics['precision'], 0.75, places=3)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=3)
        self.assertAlmostEqual(metrics['f1'], 0.857, places=3)
        
        # AUC is more complex, but should be high for this example
        self.assertGreater(metrics['auc'], 0.8)
    
    @patch('utils.evaluation.metrics.plot_roc_curve')
    @patch('utils.evaluation.metrics.plot_precision_recall_curve')
    @patch('utils.evaluation.metrics.plot_confusion_matrix')
    @patch('utils.evaluation.metrics.plot_calibration_curve')
    def test_generate_classification_report(self, mock_calibration, mock_confusion, mock_pr, mock_roc):
        """Test classification report generation."""
        # Generate report
        report = generate_classification_report(
            y_true=self.y_true,
            y_pred=self.y_pred,
            output_dir=self.temp_dir
        )
        
        # Check that all expected metrics are present
        self.assertIn('accuracy', report)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1', report)
        self.assertIn('auc', report)
        
        # Check that all plots were generated
        mock_roc.assert_called_once()
        mock_pr.assert_called_once()
        mock_confusion.assert_called_once()
        mock_calibration.assert_called_once()
        
        # Check that plot files were saved
        for plot_name in ['roc_curve.png', 'precision_recall_curve.png', 
                         'confusion_matrix.png', 'calibration_curve.png']:
            plot_path = os.path.join(self.temp_dir, plot_name)
            self.assertTrue(os.path.exists(plot_path))
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot ROC curve
        plot_roc_curve(self.y_true, self.y_pred, ax=ax)
        
        # Save figure
        fig.savefig(os.path.join(self.temp_dir, 'test_roc_curve.png'))
        
        # Check that file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_roc_curve.png')))
        
        # Close figure
        plt.close(fig)
    
    def test_plot_precision_recall_curve(self):
        """Test precision-recall curve plotting."""
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot precision-recall curve
        plot_precision_recall_curve(self.y_true, self.y_pred, ax=ax)
        
        # Save figure
        fig.savefig(os.path.join(self.temp_dir, 'test_pr_curve.png'))
        
        # Check that file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_pr_curve.png')))
        
        # Close figure
        plt.close(fig)
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot confusion matrix
        plot_confusion_matrix(self.y_true, self.y_pred_binary, ax=ax)
        
        # Save figure
        fig.savefig(os.path.join(self.temp_dir, 'test_confusion_matrix.png'))
        
        # Check that file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_confusion_matrix.png')))
        
        # Close figure
        plt.close(fig)
    
    def test_plot_calibration_curve(self):
        """Test calibration curve plotting."""
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot calibration curve
        plot_calibration_curve(self.y_true, self.y_pred, ax=ax)
        
        # Save figure
        fig.savefig(os.path.join(self.temp_dir, 'test_calibration_curve.png'))
        
        # Check that file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'test_calibration_curve.png')))
        
        # Close figure
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
