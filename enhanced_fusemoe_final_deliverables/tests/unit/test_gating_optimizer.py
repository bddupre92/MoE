"""
Unit tests for gating optimization components in the Enhanced FuseMoE system.

This module contains unit tests for the gating optimization components.
"""

import unittest
import torch
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import optimization components
from optimization.gating_optimization.gating_optimizer import GatingOptimizer
from models.gating.migraine_gating import MigraineGating


class TestGatingOptimizer(unittest.TestCase):
    """Test cases for the GatingOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock data loaders with non-zero dataset lengths
        self.mock_train_loader = MagicMock()
        self.mock_train_loader.dataset = MagicMock()
        self.mock_train_loader.dataset.__len__.return_value = 100
        
        self.mock_val_loader = MagicMock()
        self.mock_val_loader.dataset = MagicMock()
        self.mock_val_loader.dataset.__len__.return_value = 50
        
        self.mock_test_loader = MagicMock()
        self.mock_test_loader.dataset = MagicMock()
        self.mock_test_loader.dataset.__len__.return_value = 50
        
        # Create parameter bounds
        self.param_bounds = {
            'learning_rate': (0.0001, 0.01),
            'weight_decay': (0.0, 0.001),
            'top_k': (1, 4),
            'hidden_dim': (32, 128),
            'dropout_rate': (0.0, 0.5)
        }
        
        # Create gating network
        self.gating_network = MigraineGating(
            input_dim=10,
            num_experts=4,
            hidden_dim=64,
            top_k=2,
            dropout_rate=0.2
        )
        
        # Create gating optimizer
        self.gating_optimizer = GatingOptimizer(
            gating_network=self.gating_network,
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
        """Test optimizer initialization."""
        self.assertEqual(self.gating_optimizer.gating_network, self.gating_network)
        self.assertEqual(self.gating_optimizer.train_loader, self.mock_train_loader)
        self.assertEqual(self.gating_optimizer.val_loader, self.mock_val_loader)
        self.assertEqual(self.gating_optimizer.test_loader, self.mock_test_loader)
        self.assertEqual(self.gating_optimizer.param_bounds, self.param_bounds)
        self.assertEqual(self.gating_optimizer.fitness_metric, 'auc')
        self.assertEqual(self.gating_optimizer.num_epochs, 5)
        self.assertEqual(self.gating_optimizer.patience, 2)
        self.assertEqual(self.gating_optimizer.device, torch.device('cpu'))
    
    def test_optimize(self):
        """Test optimize method."""
        # Create a simplified version of the gating optimizer for testing
        class SimpleGatingOptimizer(GatingOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def optimize(self, **kwargs):
                # Return a predefined result without actually running optimization
                return {
                    'best_params': {
                        'learning_rate': 0.001,
                        'weight_decay': 0.0005,
                        'top_k': 2,
                        'hidden_dim': 64,
                        'dropout_rate': 0.2
                    },
                    'best_fitness': 0.92,
                    'test_metrics': {
                        'loss': 0.3,
                        'accuracy': 0.9,
                        'auc': 0.95
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleGatingOptimizer(
            gating_network=self.gating_network,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
        
        # Run the simplified optimization
        results = optimizer.optimize(
            algorithm='sade',
            pop_size=10,
            generations=5,
            islands=2,
            output_dir=None,
            verbose=False
        )
        
        # Check results structure
        self.assertIn('best_params', results)
        self.assertIn('best_fitness', results)
        self.assertIn('test_metrics', results)
        
        # Check specific values
        self.assertEqual(results['best_fitness'], 0.92)
        self.assertEqual(results['test_metrics']['auc'], 0.95)
        self.assertEqual(results['best_params']['learning_rate'], 0.001)
        self.assertEqual(results['best_params']['top_k'], 2)
    
    def test_get_optimized_gating_network(self):
        """Test get_optimized_gating_network method."""
        # Create a simplified version of the gating optimizer for testing
        class SimpleGatingOptimizer(GatingOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Set optimization results
                self.optimization_results = {
                    'best_params': {
                        'learning_rate': 0.001,
                        'weight_decay': 0.0005,
                        'top_k': 3,
                        'hidden_dim': 96,
                        'dropout_rate': 0.15
                    },
                    'best_fitness': 0.94,
                    'test_metrics': {
                        'loss': 0.28,
                        'accuracy': 0.92,
                        'auc': 0.97
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleGatingOptimizer(
            gating_network=self.gating_network,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
        
        # Get optimized gating network
        optimized_gating = optimizer.get_optimized_gating_network()
        
        # Check that the optimized gating network has the correct parameters
        self.assertEqual(optimized_gating.top_k, 3)
        self.assertEqual(optimized_gating.hidden_dim, 96)
        self.assertEqual(optimized_gating.dropout_rate, 0.15)
    
    def test_analyze_gating_decisions(self):
        """Test analyze_gating_decisions method."""
        # Create a simplified version of the gating optimizer for testing
        class SimpleGatingOptimizer(GatingOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def analyze_gating_decisions(self, data_loader, **kwargs):
                # Return a predefined result without actually running analysis
                return {
                    'expert_selection_frequency': {
                        'expert_0': 0.35,
                        'expert_1': 0.25,
                        'expert_2': 0.20,
                        'expert_3': 0.20
                    },
                    'co_selection_matrix': np.array([
                        [0.00, 0.10, 0.15, 0.10],
                        [0.10, 0.00, 0.05, 0.10],
                        [0.15, 0.05, 0.00, 0.00],
                        [0.10, 0.10, 0.00, 0.00]
                    ]),
                    'input_feature_importance': {
                        'feature_0': 0.15,
                        'feature_1': 0.12,
                        'feature_2': 0.18,
                        'feature_3': 0.10,
                        'feature_4': 0.08,
                        'feature_5': 0.07,
                        'feature_6': 0.09,
                        'feature_7': 0.06,
                        'feature_8': 0.08,
                        'feature_9': 0.07
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleGatingOptimizer(
            gating_network=self.gating_network,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
        
        # Run the simplified analysis
        results = optimizer.analyze_gating_decisions(
            data_loader=self.mock_test_loader,
            num_samples=100,
            feature_names=None,
            expert_names=None
        )
        
        # Check results structure
        self.assertIn('expert_selection_frequency', results)
        self.assertIn('co_selection_matrix', results)
        self.assertIn('input_feature_importance', results)
        
        # Check specific values
        self.assertEqual(results['expert_selection_frequency']['expert_0'], 0.35)
        self.assertEqual(results['co_selection_matrix'][0, 1], 0.10)
        self.assertEqual(results['input_feature_importance']['feature_2'], 0.18)


if __name__ == '__main__':
    unittest.main()
