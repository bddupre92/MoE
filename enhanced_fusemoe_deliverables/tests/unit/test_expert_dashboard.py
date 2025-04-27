"""
Unit tests for the expert dashboard component of the Enhanced FuseMoE system.

This module provides comprehensive tests for the expert dashboard visualization
components in the Enhanced FuseMoE system.
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import torch
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils.visualization.expert_dashboard import ExpertDashboard


class TestExpertDashboard(unittest.TestCase):
    """
    Test case for the ExpertDashboard class.
    
    This class tests the functionality of the ExpertDashboard class, including
    network diagram visualization, expert weight visualization, decision tree
    visualization, and performance comparison visualization.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create expert names
        self.expert_names = ["Sleep", "Weather", "Stress/Diet", "Physiological"]
        
        # Create sample expert weights
        self.expert_weights = np.array([0.3, 0.2, 0.4, 0.1])
        
        # Create sample input features and expert gates
        self.num_samples = 50
        self.num_features = 5
        self.X = np.random.randn(self.num_samples, self.num_features)
        self.expert_gates = np.random.rand(self.num_samples, len(self.expert_names))
        self.expert_gates = self.expert_gates / self.expert_gates.sum(axis=1, keepdims=True)
        
        # Create sample metrics
        self.optimized_metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.94,
            'f1': 0.93,
            'auc': 0.97
        }
        
        self.baseline_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.84,
            'f1': 0.83,
            'auc': 0.87
        }
        
        # Create mock model
        self.model = MagicMock()
        self.model.get_expert_weights.return_value = self.expert_weights
        self.model.get_expert_gates.return_value = self.expert_gates
        
        # Create dashboard
        self.dashboard = ExpertDashboard(
            model=self.model,
            expert_names=self.expert_names,
            test_data=(self.X, None),
            optimized_metrics=self.optimized_metrics,
            baseline_metrics=self.baseline_metrics
        )
    
    def test_initialization(self):
        """
        Test initialization of ExpertDashboard.
        """
        # Test with all parameters
        dashboard = ExpertDashboard(
            model=self.model,
            expert_names=self.expert_names,
            test_data=(self.X, None),
            optimized_metrics=self.optimized_metrics,
            baseline_metrics=self.baseline_metrics
        )
        
        self.assertEqual(dashboard.model, self.model)
        self.assertEqual(dashboard.expert_names, self.expert_names)
        self.assertEqual(dashboard.optimized_metrics, self.optimized_metrics)
        self.assertEqual(dashboard.baseline_metrics, self.baseline_metrics)
        
        # Test with default parameters
        dashboard = ExpertDashboard()
        
        self.assertIsNone(dashboard.model)
        self.assertEqual(dashboard.expert_names, ["Sleep", "Weather", "Stress/Diet", "Physiological"])
        self.assertIsNone(dashboard.test_data)
        self.assertIsNone(dashboard.optimized_metrics)
        self.assertIsNone(dashboard.baseline_metrics)
    
    def test_plot_network_diagram(self):
        """
        Test plot_network_diagram method.
        """
        # Test with expert weights
        fig = self.dashboard.plot_network_diagram(self.expert_weights)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test without expert weights
        fig = self.dashboard.plot_network_diagram()
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test with custom figsize
        figsize = (12, 10)
        fig = self.dashboard.plot_network_diagram(figsize=figsize)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_size_inches()[0], figsize[0])
        self.assertEqual(fig.get_size_inches()[1], figsize[1])
    
    def test_plot_expert_weights(self):
        """
        Test plot_expert_weights method.
        """
        # Test with expert weights
        fig = self.dashboard.plot_expert_weights(self.expert_weights)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test without expert weights
        fig = self.dashboard.plot_expert_weights()
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test with custom figsize
        figsize = (12, 8)
        fig = self.dashboard.plot_expert_weights(figsize=figsize)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_size_inches()[0], figsize[0])
        self.assertEqual(fig.get_size_inches()[1], figsize[1])
    
    @patch('graphviz.Source')
    @patch('sklearn.tree.export_graphviz')
    def test_plot_decision_tree_visualization(self, mock_export_graphviz, mock_source):
        """
        Test plot_decision_tree_visualization method.
        """
        # Mock graphviz.Source.render to avoid file operations
        mock_source.return_value.render.return_value = "temp_tree.png"
        
        # Mock plt.imread to avoid file operations
        with patch('matplotlib.pyplot.imread', return_value=np.zeros((100, 100, 3))):
            # Test with input features and expert gates
            fig = self.dashboard.plot_decision_tree_visualization(self.X, self.expert_gates)
            
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 1)
            
            # Test without input features and expert gates
            fig = self.dashboard.plot_decision_tree_visualization()
            
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(len(fig.axes), 1)
            
            # Test with custom max_depth and figsize
            max_depth = 2
            figsize = (14, 10)
            fig = self.dashboard.plot_decision_tree_visualization(max_depth=max_depth, figsize=figsize)
            
            self.assertIsInstance(fig, plt.Figure)
            self.assertEqual(fig.get_size_inches()[0], figsize[0])
            self.assertEqual(fig.get_size_inches()[1], figsize[1])
    
    def test_plot_performance_comparison(self):
        """
        Test plot_performance_comparison method.
        """
        # Test with default metrics
        fig = self.dashboard.plot_performance_comparison()
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test with custom metrics
        metrics = ['accuracy', 'f1', 'auc']
        fig = self.dashboard.plot_performance_comparison(metrics=metrics)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        # Test with custom figsize
        figsize = (14, 10)
        fig = self.dashboard.plot_performance_comparison(figsize=figsize)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_size_inches()[0], figsize[0])
        self.assertEqual(fig.get_size_inches()[1], figsize[1])
        
        # Test with no metrics provided
        dashboard = ExpertDashboard()
        fig = dashboard.plot_performance_comparison()
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
    
    def test_get_expert_weights(self):
        """
        Test _get_expert_weights method.
        """
        # Test with model that has get_expert_weights method
        weights = self.dashboard._get_expert_weights()
        
        np.testing.assert_array_equal(weights, self.expert_weights)
        
        # Test with model that doesn't have get_expert_weights method
        dashboard = ExpertDashboard(expert_names=self.expert_names)
        weights = dashboard._get_expert_weights()
        
        self.assertEqual(len(weights), len(self.expert_names))
        self.assertAlmostEqual(np.sum(weights), 1.0)
    
    def test_get_features_and_gates(self):
        """
        Test _get_features_and_gates method.
        """
        # Test with model that has get_expert_gates method
        X, gates = self.dashboard._get_features_and_gates()
        
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(gates, self.expert_gates)
        
        # Test with model that doesn't have get_expert_gates method
        dashboard = ExpertDashboard(expert_names=self.expert_names)
        X, gates = dashboard._get_features_and_gates()
        
        self.assertEqual(X.shape[1], 5)  # Default num_features
        self.assertEqual(gates.shape[1], len(self.expert_names))
        self.assertTrue(np.allclose(np.sum(gates, axis=1), 1.0))
    
    @patch('streamlit.title')
    @patch('streamlit.tabs')
    def test_render_streamlit_dashboard(self, mock_tabs, mock_title):
        """
        Test render_streamlit_dashboard method.
        """
        # Mock streamlit.tabs to return a list of context managers
        mock_tab = MagicMock()
        mock_tab.__enter__ = MagicMock(return_value=None)
        mock_tab.__exit__ = MagicMock(return_value=None)
        mock_tabs.return_value = [mock_tab, mock_tab, mock_tab, mock_tab]
        
        # Mock streamlit.pyplot to avoid rendering
        with patch('streamlit.pyplot', return_value=None):
            # Mock streamlit.header to avoid rendering
            with patch('streamlit.header', return_value=None):
                # Test render_streamlit_dashboard
                self.dashboard.render_streamlit_dashboard()
                
                # Verify that streamlit.title was called
                mock_title.assert_called_once()
                
                # Verify that streamlit.tabs was called with the correct arguments
                mock_tabs.assert_called_once_with(["Network Diagram", "Expert Weights", "Decision Tree", "Performance Comparison"])


class TestExpertDashboardIntegration(unittest.TestCase):
    """
    Integration tests for the ExpertDashboard class.
    
    This class tests the integration of the ExpertDashboard class with other
    components of the Enhanced FuseMoE system.
    """
    
    def setUp(self):
        """
        Set up test fixtures.
        """
        # Skip if torch is not available
        if not torch:
            self.skipTest("PyTorch not available")
        
        # Create expert names
        self.expert_names = ["Sleep", "Weather", "Stress/Diet", "Physiological"]
        
        # Create sample input data
        self.batch_size = 10
        self.seq_len = 7
        self.input_dim = 6
        self.X = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Create sample target data
        self.y = torch.randint(0, 2, (self.batch_size, 1)).float()
        
        # Create sample test data
        self.test_data = (self.X, self.y)
    
    @unittest.skip("Integration test requires actual model implementation")
    def test_integration_with_model(self):
        """
        Test integration with actual model implementation.
        
        This test is skipped by default as it requires an actual model implementation.
        """
        # Import model implementation
        from models.fusion.migraine_fusion import MigraineFusionMoE
        from models.experts.expert_registry import ExpertRegistry
        from models.gating.migraine_gating import MigraineGating
        
        # Create expert registry
        expert_registry = ExpertRegistry()
        
        # Register experts
        for name in self.expert_names:
            expert_registry.register_expert(name, MagicMock())
        
        # Create gating network
        gating = MigraineGating(
            input_dim=self.input_dim,
            num_experts=len(self.expert_names),
            hidden_dim=64,
            top_k=2
        )
        
        # Create fusion model
        model = MigraineFusionMoE(
            expert_registry=expert_registry,
            gating=gating,
            output_dim=1
        )
        
        # Create dashboard
        dashboard = ExpertDashboard(
            model=model,
            expert_names=self.expert_names,
            test_data=self.test_data
        )
        
        # Test visualization methods
        fig1 = dashboard.plot_network_diagram()
        fig2 = dashboard.plot_expert_weights()
        fig3 = dashboard.plot_performance_comparison()
        
        self.assertIsInstance(fig1, plt.Figure)
        self.assertIsInstance(fig2, plt.Figure)
        self.assertIsInstance(fig3, plt.Figure)


if __name__ == '__main__':
    unittest.main()
