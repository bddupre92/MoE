"""
Unit tests for visualization components in the Enhanced FuseMoE system.

This module contains unit tests for the visualization components.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import MagicMock, patch, mock_open

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import visualization components
from utils.visualization.visualization_components import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_expert_contributions,
    plot_optimization_progress,
    plot_parameter_importance,
    plot_performance_comparison,
    plot_expert_selection_heatmap,
    plot_feature_importance,
    plot_gating_network_visualization,
    save_visualization
)


class TestVisualizationComponents(unittest.TestCase):
    """Test cases for the visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data for training history
        self.training_history = {
            'train_history': [
                {'loss': 0.5, 'accuracy': 0.8},
                {'loss': 0.4, 'accuracy': 0.85},
                {'loss': 0.3, 'accuracy': 0.9}
            ],
            'val_history': [
                {'loss': 0.6, 'accuracy': 0.75},
                {'loss': 0.45, 'accuracy': 0.82},
                {'loss': 0.35, 'accuracy': 0.88}
            ]
        }
        
        # Create test data for confusion matrix
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
        self.confusion_matrix = np.array([[4, 1], [2, 3]])
        
        # Create test data for ROC curve
        self.y_prob = np.array([0.2, 0.8, 0.3, 0.4, 0.1, 0.9, 0.6, 0.7, 0.2, 0.3])
        
        # Create test data for expert contributions
        self.expert_contributions = {
            'expert_selection_frequency': {
                'sleep': 0.35,
                'weather': 0.25,
                'stress_diet': 0.20,
                'physio': 0.20
            },
            'expert_average_weight': {
                'sleep': 0.4,
                'weather': 0.3,
                'stress_diet': 0.2,
                'physio': 0.1
            },
            'expert_max_weight': {
                'sleep': 0.8,
                'weather': 0.7,
                'stress_diet': 0.6,
                'physio': 0.5
            }
        }
        
        # Create test data for optimization progress
        self.optimization_history = {
            'generations': [1, 2, 3, 4, 5],
            'best_fitness': [0.8, 0.85, 0.9, 0.92, 0.95],
            'mean_fitness': [0.7, 0.75, 0.8, 0.85, 0.9],
            'diversity': [0.5, 0.4, 0.3, 0.25, 0.2]
        }
        
        # Create test data for parameter importance
        self.parameter_importance = {
            'learning_rate': 0.3,
            'weight_decay': 0.1,
            'dropout_rate': 0.2,
            'hidden_dim': 0.25,
            'top_k': 0.15
        }
        
        # Create test data for performance comparison
        self.original_metrics = {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.7,
            'f1': 0.72,
            'auc': 0.85
        }
        self.optimized_metrics = {
            'accuracy': 0.9,
            'precision': 0.88,
            'recall': 0.85,
            'f1': 0.86,
            'auc': 0.95
        }
    
    def test_plot_training_history(self):
        """Test plot_training_history function."""
        # Create plot
        fig = plot_training_history(self.training_history)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has correct number of subplots
        self.assertEqual(len(fig.axes), 2)  # Loss and accuracy
    
    def test_plot_confusion_matrix(self):
        """Test plot_confusion_matrix function."""
        # Create plot
        fig = plot_confusion_matrix(self.confusion_matrix, class_names=['No Migraine', 'Migraine'])
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_roc_curve(self):
        """Test plot_roc_curve function."""
        # Create plot
        fig = plot_roc_curve(self.y_true, self.y_prob)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_precision_recall_curve(self):
        """Test plot_precision_recall_curve function."""
        # Create plot
        fig = plot_precision_recall_curve(self.y_true, self.y_prob)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_expert_contributions(self):
        """Test plot_expert_contributions function."""
        # Create plot
        fig = plot_expert_contributions(self.expert_contributions)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has correct number of subplots
        self.assertEqual(len(fig.axes), 3)  # Selection frequency, average weight, max weight
    
    def test_plot_optimization_progress(self):
        """Test plot_optimization_progress function."""
        # Create plot
        fig = plot_optimization_progress(self.optimization_history)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has correct number of subplots
        self.assertEqual(len(fig.axes), 2)  # Fitness and diversity
    
    def test_plot_parameter_importance(self):
        """Test plot_parameter_importance function."""
        # Create plot
        fig = plot_parameter_importance(self.parameter_importance)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_performance_comparison(self):
        """Test plot_performance_comparison function."""
        # Create plot
        fig = plot_performance_comparison(self.original_metrics, self.optimized_metrics)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_expert_selection_heatmap(self):
        """Test plot_expert_selection_heatmap function."""
        # Create test data
        expert_gates = np.array([
            [0.7, 0.3, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3, 0.7],
            [0.4, 0.3, 0.2, 0.1]
        ])
        expert_names = ['sleep', 'weather', 'stress_diet', 'physio']
        
        # Create plot
        fig = plot_expert_selection_heatmap(expert_gates, expert_names)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_feature_importance(self):
        """Test plot_feature_importance function."""
        # Create test data
        feature_importance = {
            'sleep_duration': 0.15,
            'sleep_quality': 0.12,
            'temperature': 0.18,
            'humidity': 0.10,
            'stress_level': 0.08,
            'caffeine_intake': 0.07,
            'heart_rate': 0.09,
            'blood_pressure': 0.06,
            'respiratory_rate': 0.08,
            'oxygen_saturation': 0.07
        }
        
        # Create plot
        fig = plot_feature_importance(feature_importance)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_gating_network_visualization(self):
        """Test plot_gating_network_visualization function."""
        # Create test data
        gating_weights = np.array([
            [0.2, 0.3, -0.1, 0.5],
            [0.1, -0.2, 0.4, 0.3],
            [-0.3, 0.4, 0.2, -0.1]
        ])
        feature_names = ['feature_1', 'feature_2', 'feature_3']
        expert_names = ['sleep', 'weather', 'stress_diet', 'physio']
        
        # Create plot
        fig = plot_gating_network_visualization(gating_weights, feature_names, expert_names)
        
        # Check that figure is created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one subplot
        self.assertEqual(len(fig.axes), 1)
    
    @patch('matplotlib.pyplot.savefig')
    def test_save_visualization(self, mock_savefig):
        """Test save_visualization function."""
        # Create test figure
        fig = plt.figure()
        
        # Save visualization
        save_visualization(fig, 'test_figure', '/tmp', dpi=300, format='png')
        
        # Check that savefig was called with correct arguments
        mock_savefig.assert_called_once()
        args, kwargs = mock_savefig.call_args
        
        # Check that the first argument is the correct file path
        self.assertEqual(args[0], '/tmp/test_figure.png')
        
        # Check that the correct DPI was used
        self.assertEqual(kwargs['dpi'], 300)


if __name__ == '__main__':
    unittest.main()
