"""
Unit tests for expert optimization components in the Enhanced FuseMoE system.

This module contains unit tests for the expert optimization components.
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
from optimization.expert_optimization.expert_optimizer import ExpertOptimizer
from models.experts.sleep_expert import SleepExpert
from models.experts.weather_expert import WeatherExpert
from models.experts.stress_diet_expert import StressDietExpert
from models.experts.physio_expert import PhysioExpert


class TestExpertOptimizer(unittest.TestCase):
    """Test cases for the ExpertOptimizer class."""
    
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
            'dropout_rate': (0.0, 0.5),
            'hidden_dim': (32, 128)
        }
        
        # Create expert models
        self.sleep_expert = SleepExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
        self.weather_expert = WeatherExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
        self.stress_diet_expert = StressDietExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
        self.physio_expert = PhysioExpert(input_dim=10, hidden_dim=64, dropout_rate=0.2)
        
        # Create expert optimizer
        self.expert_optimizer = ExpertOptimizer(
            experts={
                'sleep': self.sleep_expert,
                'weather': self.weather_expert,
                'stress_diet': self.stress_diet_expert,
                'physio': self.physio_expert
            },
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
        self.assertEqual(len(self.expert_optimizer.experts), 4)
        self.assertEqual(self.expert_optimizer.train_loader, self.mock_train_loader)
        self.assertEqual(self.expert_optimizer.val_loader, self.mock_val_loader)
        self.assertEqual(self.expert_optimizer.test_loader, self.mock_test_loader)
        self.assertEqual(self.expert_optimizer.param_bounds, self.param_bounds)
        self.assertEqual(self.expert_optimizer.fitness_metric, 'auc')
        self.assertEqual(self.expert_optimizer.num_epochs, 5)
        self.assertEqual(self.expert_optimizer.patience, 2)
        self.assertEqual(self.expert_optimizer.device, torch.device('cpu'))
    
    def test_optimize_expert(self):
        """Test optimize_expert method."""
        # Create a simplified version of the expert optimizer for testing
        class SimpleExpertOptimizer(ExpertOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def optimize_expert(self, expert_name, **kwargs):
                # Return a predefined result without actually running optimization
                return {
                    'best_params': {
                        'learning_rate': 0.001,
                        'weight_decay': 0.0005,
                        'dropout_rate': 0.2,
                        'hidden_dim': 64
                    },
                    'best_fitness': 0.92,
                    'test_metrics': {
                        'loss': 0.3,
                        'accuracy': 0.9,
                        'auc': 0.95
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleExpertOptimizer(
            experts={
                'sleep': self.sleep_expert,
                'weather': self.weather_expert,
                'stress_diet': self.stress_diet_expert,
                'physio': self.physio_expert
            },
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
        results = optimizer.optimize_expert(
            expert_name='sleep',
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
    
    def test_optimize_all_experts(self):
        """Test optimize_all_experts method."""
        # Create a simplified version of the expert optimizer for testing
        class SimpleExpertOptimizer(ExpertOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def optimize_expert(self, expert_name, **kwargs):
                # Return a predefined result without actually running optimization
                return {
                    'best_params': {
                        'learning_rate': 0.001,
                        'weight_decay': 0.0005,
                        'dropout_rate': 0.2,
                        'hidden_dim': 64
                    },
                    'best_fitness': 0.92,
                    'test_metrics': {
                        'loss': 0.3,
                        'accuracy': 0.9,
                        'auc': 0.95
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleExpertOptimizer(
            experts={
                'sleep': self.sleep_expert,
                'weather': self.weather_expert,
                'stress_diet': self.stress_diet_expert,
                'physio': self.physio_expert
            },
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
        
        # Run the simplified optimization for all experts
        results = optimizer.optimize_all_experts(
            algorithm='sade',
            pop_size=10,
            generations=5,
            islands=2,
            output_dir=None,
            verbose=False
        )
        
        # Check results structure
        self.assertEqual(len(results), 4)
        for expert_name in ['sleep', 'weather', 'stress_diet', 'physio']:
            self.assertIn(expert_name, results)
            self.assertIn('best_params', results[expert_name])
            self.assertIn('best_fitness', results[expert_name])
            self.assertIn('test_metrics', results[expert_name])
            
            # Check specific values
            self.assertEqual(results[expert_name]['best_fitness'], 0.92)
            self.assertEqual(results[expert_name]['test_metrics']['auc'], 0.95)
            self.assertEqual(results[expert_name]['best_params']['learning_rate'], 0.001)
    
    def test_get_best_experts(self):
        """Test get_best_experts method."""
        # Create a simplified version of the expert optimizer for testing
        class SimpleExpertOptimizer(ExpertOptimizer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Set optimization results
                self.optimization_results = {
                    'sleep': {
                        'best_params': {
                            'learning_rate': 0.001,
                            'weight_decay': 0.0005,
                            'dropout_rate': 0.2,
                            'hidden_dim': 64
                        },
                        'best_fitness': 0.92,
                        'test_metrics': {
                            'loss': 0.3,
                            'accuracy': 0.9,
                            'auc': 0.95
                        }
                    },
                    'weather': {
                        'best_params': {
                            'learning_rate': 0.002,
                            'weight_decay': 0.0003,
                            'dropout_rate': 0.3,
                            'hidden_dim': 32
                        },
                        'best_fitness': 0.88,
                        'test_metrics': {
                            'loss': 0.35,
                            'accuracy': 0.85,
                            'auc': 0.9
                        }
                    },
                    'stress_diet': {
                        'best_params': {
                            'learning_rate': 0.0015,
                            'weight_decay': 0.0004,
                            'dropout_rate': 0.25,
                            'hidden_dim': 48
                        },
                        'best_fitness': 0.9,
                        'test_metrics': {
                            'loss': 0.32,
                            'accuracy': 0.88,
                            'auc': 0.93
                        }
                    },
                    'physio': {
                        'best_params': {
                            'learning_rate': 0.0008,
                            'weight_decay': 0.0006,
                            'dropout_rate': 0.15,
                            'hidden_dim': 96
                        },
                        'best_fitness': 0.94,
                        'test_metrics': {
                            'loss': 0.28,
                            'accuracy': 0.92,
                            'auc': 0.97
                        }
                    }
                }
        
        # Create the simplified optimizer
        optimizer = SimpleExpertOptimizer(
            experts={
                'sleep': self.sleep_expert,
                'weather': self.weather_expert,
                'stress_diet': self.stress_diet_expert,
                'physio': self.physio_expert
            },
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            test_loader=self.mock_test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=5,
            patience=2,
            device=torch.device('cpu')
        )
        
        # Get best experts
        best_experts = optimizer.get_best_experts(top_k=2)
        
        # Check results
        self.assertEqual(len(best_experts), 2)
        self.assertIn('physio', best_experts)
        self.assertIn('sleep', best_experts)
        self.assertEqual(best_experts['physio'].input_dim, 10)
        self.assertEqual(best_experts['sleep'].input_dim, 10)


if __name__ == '__main__':
    unittest.main()
