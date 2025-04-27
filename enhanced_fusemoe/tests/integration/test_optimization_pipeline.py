"""
Integration tests for the optimization pipeline in the Enhanced FuseMoE system.

This module contains integration tests for the end-to-end optimization pipeline
using PyGMO to optimize the Enhanced FuseMoE model for migraine prediction.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components
from models.experts.expert_registry import DynamicMigraineMoE, ScalableExpertPool, ExpertRegistry
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusion
from optimization.pygmo_model_optimizer import MigraineMoEOptimizer
from optimization.evolutionary_algorithms.optimization_manager import OptimizationManager
from utils.preprocessing.data_generator import MigraineSyntheticDataGenerator
from utils.preprocessing.data_preprocessor import MigraineDataPreprocessor
from utils.evaluation.metrics import calculate_metrics


class TestEndToEndOptimizationPipeline(unittest.TestCase):
    """Test cases for the end-to-end optimization pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set device
        self.device = torch.device('cpu')
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create synthetic data
        self.num_samples = 50
        self.generator = MigraineSyntheticDataGenerator(seed=42)
        self.dataset = self.generator.generate_dataset(self.num_samples)
        
        # Preprocess data
        self.preprocessor = MigraineDataPreprocessor()
        self.processed_data = self.preprocessor.fit_transform(self.dataset)
        
        # Create torch datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self.preprocessor.create_torch_datasets(
            self.processed_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Create dataloaders
        self.batch_size = 8
        self.train_loader, self.val_loader, self.test_loader = self.preprocessor.create_dataloaders(
            self.train_dataset, self.val_dataset, self.test_dataset, batch_size=self.batch_size
        )
        
        # Define expert types and dimensions
        self.expert_types = ['sleep', 'weather', 'stress_diet', 'physio']
        self.expert_dims = {
            'sleep': 6,
            'weather': 5,
            'stress_diet': 6,
            'physio': 5
        }
        
        # Define parameter bounds for optimization
        self.param_bounds = {
            'expert_hidden_dim': (16, 64),
            'expert_num_layers': (1, 2),
            'expert_dropout': (0.1, 0.3),
            'gating_hidden_dim': (16, 64),
            'top_k': (1, 2),
            'load_balance_coef': (0.001, 0.1),
            'noisy_gating': (0, 1),
            'fusion_hidden_dim': (16, 64),
            'fusion_dropout': (0.1, 0.3),
            'learning_rate': (0.0005, 0.005),
            'weight_decay': (0.0, 0.001),
            'batch_size': (4, 16)
        }
    
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.problem')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.algorithm')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.population')
    @patch('optimization.evolutionary_algorithms.optimization_manager.pg.archipelago')
    def test_optimization_pipeline(self, mock_archipelago, mock_population, mock_algorithm, mock_problem):
        """Test the end-to-end optimization pipeline."""
        # Skip this test if pygmo is not available
        try:
            import pygmo
        except ImportError:
            self.skipTest("pygmo not available")
        
        # Create registry and expert pool
        registry = ExpertRegistry()
        expert_pool = ScalableExpertPool(registry)
        
        # Add experts to pool
        for expert_type in self.expert_types:
            expert_pool.add_expert(
                name=expert_type,
                input_dim=self.expert_dims[expert_type],
                hidden_dim=32,
                output_dim=16,
                num_layers=1,
                dropout_rate=0.2
            )
        
        # Create gating network
        input_dims = [self.expert_dims[expert_type] for expert_type in self.expert_types]
        gating = MigraineGating(
            input_dims=input_dims,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            top_k=1,
            dropout_rate=0.2,
            noisy_gating=False
        )
        
        # Create fusion mechanism
        fusion = MigraineFusion(
            expert_output_dim=16,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            dropout_rate=0.2
        )
        
        # Create MoE model
        model = DynamicMigraineMoE(expert_pool, gating, fusion)
        model.to(self.device)
        
        # Set up mock archipelago
        mock_archi = MagicMock()
        mock_archipelago.return_value = mock_archi
        
        # Set up mock champion
        mock_champion = MagicMock()
        mock_champion.x = [32, 1, 0.2, 32, 1, 0.01, 0, 32, 0.2, 0.001, 0.0001, 8]
        mock_champion.f = [-0.85]  # Negative because PyGMO minimizes
        mock_archi.get_champions_f.return_value = [mock_champion.f]
        mock_archi.get_champions_x.return_value = [mock_champion.x]
        
        # Create optimization manager
        manager = OptimizationManager(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            param_bounds=self.param_bounds,
            fitness_metric='auc',
            num_epochs=2,  # Small number for testing
            patience=1,
            device=self.device
        )
        
        # Run optimization
        results = manager.optimize(
            algorithm='sade',
            pop_size=2,  # Small population for testing
            generations=2,  # Small number for testing
            islands=1,
            output_dir=None,
            verbose=False
        )
        
        # Check that optimization was performed
        mock_problem.assert_called_once()
        mock_algorithm.assert_called_once()
        mock_population.assert_called()
        mock_archipelago.assert_called_once()
        mock_archi.evolve.assert_called_once()
        
        # Check results
        self.assertIn('best_params', results)
        self.assertIn('best_fitness', results)
        self.assertIn('test_metrics', results)
        
        # Check best parameters
        for name, value in zip(manager.problem.param_names, mock_champion.x):
            self.assertEqual(results['best_params'][name], value)
        
        # Check best fitness
        self.assertEqual(results['best_fitness'], 0.85)
    
    def test_model_creation_and_training(self):
        """Test model creation and training without optimization."""
        # Create registry and expert pool
        registry = ExpertRegistry()
        expert_pool = ScalableExpertPool(registry)
        
        # Add experts to pool
        for expert_type in self.expert_types:
            expert_pool.add_expert(
                name=expert_type,
                input_dim=self.expert_dims[expert_type],
                hidden_dim=32,
                output_dim=16,
                num_layers=1,
                dropout_rate=0.2
            )
        
        # Create gating network
        input_dims = [self.expert_dims[expert_type] for expert_type in self.expert_types]
        gating = MigraineGating(
            input_dims=input_dims,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            top_k=1,
            dropout_rate=0.2,
            noisy_gating=False
        )
        
        # Create fusion mechanism
        fusion = MigraineFusion(
            expert_output_dim=16,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            dropout_rate=0.2
        )
        
        # Create MoE model
        model = DynamicMigraineMoE(expert_pool, gating, fusion)
        model.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few steps
        model.train()
        for epoch in range(2):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # Move inputs and targets to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Calculate loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Only run a few batches for testing
                if batch_idx >= 2:
                    break
        
        # Evaluate model
        model.eval()
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Move inputs and targets to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Store outputs and targets
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate outputs and targets
        all_outputs = np.concatenate(all_outputs)
        all_targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_outputs)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('auc', metrics)
        
        # Check that metrics are valid
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)


if __name__ == '__main__':
    unittest.main()
