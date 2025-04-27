"""
Enhanced unit tests for visualization components in the Enhanced FuseMoE system.

This module contains enhanced unit tests for the visualization components including:
- Performance visualization functions
- Expert contribution visualization
- Optimization progress visualization
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock, call

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
    MigraineVisualization
)


class TestVisualizationComponents(unittest.TestCase):
    """Test cases for the visualization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample training history
        self.history = {
            'train_history': [
                {'loss': 0.5, 'accuracy': 0.7, 'auc': 0.75},
                {'loss': 0.4, 'accuracy': 0.8, 'auc': 0.8},
                {'loss': 0.3, 'accuracy': 0.85, 'auc': 0.85}
            ],
            'val_history': [
                {'loss': 0.6, 'accuracy': 0.65, 'auc': 0.7},
                {'loss': 0.5, 'accuracy': 0.75, 'auc': 0.75},
                {'loss': 0.4, 'accuracy': 0.8, 'auc': 0.8}
            ]
        }
        
        # Create sample confusion matrix
        self.confusion_matrix = (45, 5, 10, 40)  # tn, fp, fn, tp
        
        # Create sample ROC curve data
        self.y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
        self.y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.6])
        
        # Create sample expert contributions
        self.expert_contributions = {
            'sleep': 0.3,
            'weather': 0.2,
            'stress_diet': 0.25,
            'physio': 0.25
        }
        
        # Create sample optimization progress
        self.optimization_progress = {
            'generations': list(range(10)),
            'best_fitness': [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.84, 0.85],
            'mean_fitness': [0.5, 0.55, 0.6, 0.63, 0.65, 0.68, 0.7, 0.72, 0.74, 0.75]
        }
        
        # Create sample parameter importance
        self.parameter_importance = {
            'expert_hidden_dim': 0.2,
            'expert_num_layers': 0.1,
            'expert_dropout': 0.05,
            'gating_hidden_dim': 0.15,
            'top_k': 0.1,
            'learning_rate': 0.3,
            'weight_decay': 0.1
        }
        
        # Create sample performance comparison
        self.performance_comparison = {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'original': [0.7, 0.65, 0.6, 0.62, 0.75],
            'optimized': [0.9, 0.88, 0.85, 0.86, 0.92]
        }
        
        # Create sample model outputs for expert contributions
        self.model_outputs = {
            'expert_weights': np.array([
                [0.3, 0.2, 0.25, 0.25],
                [0.4, 0.1, 0.3, 0.2],
                [0.2, 0.3, 0.2, 0.3]
            ]),
            'expert_names': ['sleep', 'weather', 'stress_diet', 'physio']
        }
        
        # Create temporary directory for saving plots
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, 'test_plot.png')
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_history(self, mock_savefig, mock_subplots, mock_figure):
        """Test plotting training history."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot training history
        fig = plot_training_history(
            history=self.history,
            metrics=['loss', 'accuracy', 'auc'],
            figsize=(10, 6),
            save_path=self.save_path
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(3, 1, figsize=(10, 6), sharex=True)
        
        # Check that each axis had plot called twice (train and val)
        for ax in mock_axes:
            self.assertEqual(ax.plot.call_count, 2)
            ax.set_ylabel.assert_called_once()
            ax.legend.assert_called_once()
            ax.grid.assert_called_once_with(True, alpha=0.3)
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix(self, mock_savefig, mock_colorbar, mock_figure):
        """Test plotting confusion matrix."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock imshow return
        mock_im = MagicMock()
        mock_ax.imshow.return_value = mock_im
        
        # Plot confusion matrix
        fig = plot_confusion_matrix(
            confusion_matrix=self.confusion_matrix,
            figsize=(8, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(8, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that imshow was called with correct parameters
        mock_ax.imshow.assert_called_once()
        
        # Check that colorbar was called
        mock_colorbar.assert_called_once_with(mock_im, ax=mock_ax)
        
        # Check that text annotations were added (one for each cell)
        self.assertEqual(mock_ax.text.call_count, 4)
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('sklearn.metrics.roc_curve')
    @patch('sklearn.metrics.auc')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_roc_curve(self, mock_savefig, mock_auc, mock_roc_curve, mock_figure):
        """Test plotting ROC curve."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock ROC curve calculation
        mock_roc_curve.return_value = (
            np.array([0, 0.1, 0.5, 1]),  # fpr
            np.array([0, 0.5, 0.9, 1]),  # tpr
            np.array([1, 0.9, 0.5, 0])   # thresholds
        )
        
        # Mock AUC calculation
        mock_auc.return_value = 0.85
        
        # Plot ROC curve
        fig = plot_roc_curve(
            y_true=self.y_true,
            y_pred_proba=self.y_pred_proba,
            figsize=(8, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(8, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that ROC curve was calculated
        mock_roc_curve.assert_called_once_with(self.y_true, self.y_pred_proba)
        
        # Check that AUC was calculated
        mock_auc.assert_called_once()
        
        # Check that ROC curve was plotted
        mock_ax.plot.assert_any_call(
            np.array([0, 0.1, 0.5, 1]),  # fpr
            np.array([0, 0.5, 0.9, 1]),  # tpr
            label=f'ROC curve (AUC = {0.85:.3f})'
        )
        
        # Check that diagonal line was plotted
        mock_ax.plot.assert_any_call([0, 1], [0, 1], 'k--')
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('sklearn.metrics.precision_recall_curve')
    @patch('sklearn.metrics.average_precision_score')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_precision_recall_curve(self, mock_savefig, mock_ap, mock_pr_curve, mock_figure):
        """Test plotting precision-recall curve."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock precision-recall curve calculation
        mock_pr_curve.return_value = (
            np.array([0.5, 0.8, 0.9, 1]),  # precision
            np.array([1, 0.8, 0.5, 0]),    # recall
            np.array([0, 0.5, 0.8, 1])     # thresholds
        )
        
        # Mock average precision calculation
        mock_ap.return_value = 0.75
        
        # Plot precision-recall curve
        fig = plot_precision_recall_curve(
            y_true=self.y_true,
            y_pred_proba=self.y_pred_proba,
            figsize=(8, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(8, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that precision-recall curve was calculated
        mock_pr_curve.assert_called_once_with(self.y_true, self.y_pred_proba)
        
        # Check that average precision was calculated
        mock_ap.assert_called_once_with(self.y_true, self.y_pred_proba)
        
        # Check that precision-recall curve was plotted
        mock_ax.plot.assert_called_once_with(
            np.array([1, 0.8, 0.5, 0]),    # recall
            np.array([0.5, 0.8, 0.9, 1]),  # precision
            label=f'Precision-Recall curve (AP = {0.75:.3f})'
        )
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_expert_contributions(self, mock_savefig, mock_figure):
        """Test plotting expert contributions."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Mock pie chart return values
        mock_wedges = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_texts = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_autotexts = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_ax.pie.return_value = (mock_wedges, mock_texts, mock_autotexts)
        
        # Plot expert contributions
        fig = plot_expert_contributions(
            expert_contributions=self.expert_contributions,
            figsize=(8, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(8, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that pie chart was created with correct values
        mock_ax.pie.assert_called_once_with(
            list(self.expert_contributions.values()),
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Check that legend was added
        mock_ax.legend.assert_called_once_with(
            mock_wedges,
            list(self.expert_contributions.keys()),
            title="Experts",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Check that axis was set to equal
        mock_ax.axis.assert_called_once_with('equal')
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_optimization_progress(self, mock_savefig, mock_figure):
        """Test plotting optimization progress."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Plot optimization progress
        fig = plot_optimization_progress(
            generations=self.optimization_progress['generations'],
            best_fitness=self.optimization_progress['best_fitness'],
            mean_fitness=self.optimization_progress['mean_fitness'],
            figsize=(10, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(10, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that best fitness line was plotted
        mock_ax.plot.assert_any_call(
            self.optimization_progress['generations'],
            self.optimization_progress['best_fitness'],
            'b-',
            label='Best Fitness'
        )
        
        # Check that mean fitness line was plotted
        mock_ax.plot.assert_any_call(
            self.optimization_progress['generations'],
            self.optimization_progress['mean_fitness'],
            'r-',
            label='Mean Fitness'
        )
        
        # Check that target line was added
        mock_ax.axhline.assert_called_once_with(y=0.95, color='g', linestyle='--', label='Target (0.95)')
        
        # Check that legend was added
        mock_ax.legend.assert_called_once_with(loc='best')
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.xticks')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_parameter_importance(self, mock_savefig, mock_tight_layout, mock_xticks, mock_figure):
        """Test plotting parameter importance."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Sort parameters by importance
        sorted_params = sorted(self.parameter_importance.items(), key=lambda x: x[1], reverse=True)
        param_names = [p[0] for p in sorted_params]
        importance = [p[1] for p in sorted_params]
        
        # Plot parameter importance
        fig = plot_parameter_importance(
            parameter_importance=self.parameter_importance,
            figsize=(10, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(10, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that bar chart was created with correct values
        mock_ax.bar.assert_called_once_with(param_names, importance)
        
        # Check that x-axis labels were rotated
        mock_xticks.assert_called_once_with(rotation=45, ha='right')
        
        # Check that layout was adjusted
        mock_tight_layout.assert_called_once()
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_performance_comparison(self, mock_savefig, mock_figure):
        """Test plotting performance comparison."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Plot performance comparison
        fig = plot_performance_comparison(
            metrics=self.performance_comparison['metrics'],
            original_performance=self.performance_comparison['original'],
            optimized_performance=self.performance_comparison['optimized'],
            figsize=(12, 6),
            save_path=self.save_path
        )
        
        # Check that figure was created with correct size
        mock_figure.assert_called_once_with(figsize=(12, 6))
        
        # Check that axes were created
        mock_fig.add_subplot.assert_called_once_with(111)
        
        # Check that bar charts were created
        self.assertEqual(mock_ax.bar.call_count, 2)
        
        # Check that target line was added
        mock_ax.axhline.assert_called_once_with(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        
        # Check that legend was added
        mock_ax.legend.assert_called_once()
        
        # Check that savefig was called if save_path was provided
        mock_savefig.assert_called_once_with(self.save_path, bbox_inches='tight', dpi=300)
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)


class TestMigraineVisualization(unittest.TestCase):
    """Test cases for the MigraineVisualization class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create visualization instance
        self.viz = MigraineVisualization(
            output_dir=self.output_dir,
            figsize=(10, 6),
            dpi=300
        )
        
        # Create sample training history
        self.history = {
            'train_history': [
                {'loss': 0.5, 'accuracy': 0.7, 'auc': 0.75},
                {'loss': 0.4, 'accuracy': 0.8, 'auc': 0.8},
                {'loss': 0.3, 'accuracy': 0.85, 'auc': 0.85}
            ],
            'val_history': [
                {'loss': 0.6, 'accuracy': 0.65, 'auc': 0.7},
                {'loss': 0.5, 'accuracy': 0.75, 'auc': 0.75},
                {'loss': 0.4, 'accuracy': 0.8, 'auc': 0.8}
            ]
        }
        
        # Create sample model outputs for expert contributions
        self.model_outputs = {
            'expert_weights': np.array([
                [0.3, 0.2, 0.25, 0.25],
                [0.4, 0.1, 0.3, 0.2],
                [0.2, 0.3, 0.2, 0.3]
            ]),
            'expert_names': ['sleep', 'weather', 'stress_diet', 'physio']
        }
        
        # Create sample feature importance
        self.feature_importance = {
            'sleep': np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05]),
            'weather': np.array([0.4, 0.3, 0.2, 0.1]),
            'stress_diet': np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
            'physio': np.array([0.2, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05])
        }
        
        # Create sample embeddings and labels
        self.embeddings = np.random.randn(100, 20)
        self.labels = np.random.randint(0, 2, 100)
        
        # Create sample optimization history
        self.optimization_history = [
            {'best_fitness': 0.6, 'avg_fitness': 0.5},
            {'best_fitness': 0.65, 'avg_fitness': 0.55},
            {'best_fitness': 0.7, 'avg_fitness': 0.6},
            {'best_fitness': 0.75, 'avg_fitness': 0.65},
            {'best_fitness': 0.8, 'avg_fitness': 0.7}
        ]
        
        # Create sample parameter history
        self.parameter_history = {
            'learning_rate': [0.01, 0.008, 0.006, 0.004, 0.002],
            'hidden_dim': [32, 64, 64, 128, 128],
            'dropout': [0.2, 0.3, 0.3, 0.4, 0.4]
        }
        
        # Create sample model results
        self.model_results = {
            'original': {
                'metrics': {
                    'accuracy': 0.7,
                    'precision': 0.65,
                    'recall': 0.6,
                    'f1': 0.62,
                    'auc': 0.75
                }
            },
            'optimized': {
                'metrics': {
                    'accuracy': 0.9,
                    'precision': 0.88,
                    'recall': 0.85,
                    'f1': 0.86,
                    'auc': 0.92
                }
            }
        }
        
        # Create sample predictions and targets
        self.predictions = np.random.rand(100, 1)
        self.targets = np.random.randint(0, 2, 100)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_history(self, mock_savefig, mock_subplots):
        """Test plotting training history."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot training history
        fig = self.viz.plot_training_history(
            history=self.history,
            metrics=['loss', 'accuracy', 'auc'],
            title='Training History',
            save_name='training_history.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(3, 1, figsize=(10, 6), sharex=True)
        
        # Check that each axis had plot called twice (train and val)
        for ax in mock_axes:
            self.assertEqual(ax.plot.call_count, 2)
            ax.set_ylabel.assert_called_once()
            ax.legend.assert_called_once()
            ax.grid.assert_called_once_with(True, alpha=0.3)
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'training_history.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_expert_contributions(self, mock_savefig, mock_subplots):
        """Test plotting expert contributions."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot expert contributions
        fig = self.viz.plot_expert_contributions(
            model_outputs=self.model_outputs,
            title='Expert Contributions',
            save_name='expert_contributions.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(1, 2, figsize=(10, 6))
        
        # Check that bar chart was created for average weights
        mock_axes[0].bar.assert_called_once()
        
        # Check that boxplot was created for weight distribution
        mock_axes[1].boxplot.assert_called_once()
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'expert_contributions.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, mock_subplots):
        """Test plotting feature importance."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot feature importance
        fig = self.viz.plot_feature_importance(
            feature_importance=self.feature_importance,
            title='Feature Importance',
            save_name='feature_importance.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(
            4, 1,
            figsize=(10, 6 * 4 / 2)
        )
        
        # Check that each axis had barh called
        for ax in mock_axes:
            ax.barh.assert_called_once()
            ax.set_xlabel.assert_called_once_with('Importance')
            ax.set_title.assert_called_once()
            ax.grid.assert_called_once_with(True, alpha=0.3)
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'feature_importance.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.colorbar')
    @patch('matplotlib.pyplot.savefig')
    @patch('sklearn.manifold.TSNE')
    def test_plot_embedding_visualization_tsne(self, mock_tsne, mock_savefig, mock_colorbar, mock_subplots):
        """Test plotting embedding visualization with t-SNE."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Mock t-SNE
        mock_tsne_instance = MagicMock()
        mock_tsne.return_value = mock_tsne_instance
        mock_tsne_instance.fit_transform.return_value = np.random.randn(100, 2)
        
        # Mock scatter return
        mock_scatter = MagicMock()
        mock_ax.scatter.return_value = mock_scatter
        
        # Plot embedding visualization
        fig = self.viz.plot_embedding_visualization(
            embeddings=self.embeddings,
            labels=self.labels,
            method='tsne',
            title='Embedding Visualization',
            save_name='embedding_visualization.png'
        )
        
        # Check that t-SNE was created with correct parameters
        mock_tsne.assert_called_once_with(n_components=2, random_state=42)
        
        # Check that t-SNE was applied to embeddings
        mock_tsne_instance.fit_transform.assert_called_once_with(self.embeddings)
        
        # Check that scatter was called
        mock_ax.scatter.assert_called_once()
        
        # Check that colorbar was added
        mock_colorbar.assert_called_once_with(mock_scatter, ax=mock_ax, label='Label')
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'embedding_visualization.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_optimization_progress(self, mock_savefig, mock_subplots):
        """Test plotting optimization progress."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Plot optimization progress
        fig = self.viz.plot_optimization_progress(
            optimization_history=self.optimization_history,
            title='Optimization Progress',
            save_name='optimization_progress.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(figsize=(10, 6))
        
        # Check that lines were plotted
        self.assertEqual(mock_ax.plot.call_count, 2)
        
        # Check that target line was added
        mock_ax.axhline.assert_called_once_with(y=0.95, color='g', linestyle='--', label='Target (0.95)')
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'optimization_progress.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    def test_plot_parameter_evolution(self, mock_savefig, mock_subplots):
        """Test plotting parameter evolution."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot parameter evolution
        fig = self.viz.plot_parameter_evolution(
            parameter_history=self.parameter_history,
            title='Parameter Evolution',
            save_name='parameter_evolution.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(
            3, 1,
            figsize=(10, 6 * 3 / 2),
            sharex=True
        )
        
        # Check that each axis had plot called
        for ax in mock_axes:
            ax.plot.assert_called_once()
            ax.set_ylabel.assert_called_once()
            ax.grid.assert_called_once_with(True, alpha=0.3)
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'parameter_evolution.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('pandas.DataFrame.plot')
    def test_plot_model_comparison(self, mock_df_plot, mock_savefig, mock_subplots):
        """Test plotting model comparison."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Plot model comparison
        fig = self.viz.plot_model_comparison(
            model_results=self.model_results,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc'],
            title='Model Comparison',
            save_name='model_comparison.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(figsize=(10, 6))
        
        # Check that DataFrame plot was called
        mock_df_plot.assert_called_once_with(kind='bar', ax=mock_ax)
        
        # Check that target line was added
        mock_ax.axhline.assert_called_once_with(y=0.95, color='r', linestyle='--', label='Target (0.95)')
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'model_comparison.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('seaborn.kdeplot')
    def test_plot_prediction_distribution(self, mock_kdeplot, mock_savefig, mock_subplots):
        """Test plotting prediction distribution."""
        # Create mock figure and axes
        mock_fig = MagicMock()
        mock_axes = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Plot prediction distribution
        fig = self.viz.plot_prediction_distribution(
            predictions=self.predictions,
            targets=self.targets,
            title='Prediction Distribution',
            save_name='prediction_distribution.png'
        )
        
        # Check that subplots was called with correct parameters
        mock_subplots.assert_called_once_with(1, 2, figsize=(10, 6))
        
        # Check that histograms were plotted
        self.assertEqual(mock_axes[0].hist.call_count, 2)
        
        # Check that KDE plots were created
        self.assertEqual(mock_kdeplot.call_count, 2)
        
        # Check that savefig was called if save_name was provided
        mock_savefig.assert_called_once_with(
            os.path.join(self.output_dir, 'prediction_distribution.png'),
            bbox_inches='tight',
            dpi=300
        )
        
        # Check that figure was returned
        self.assertEqual(fig, mock_fig)


if __name__ == '__main__':
    unittest.main()
