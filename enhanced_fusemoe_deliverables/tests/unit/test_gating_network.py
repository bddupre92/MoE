"""
Unit tests for the gating network in the Enhanced FuseMoE system.

This module contains unit tests for the MigraineGating class and related functions.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import gating network
from models.gating.migraine_gating import MigraineGating, create_migraine_gating


class TestMigraineGating(unittest.TestCase):
    """Test cases for the MigraineGating network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dims = [6, 5, 6, 5]  # Dimensions for each expert input
        self.hidden_dim = 32
        self.num_experts = 4
        self.top_k = 2
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = MigraineGating(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout_rate=0.1,
            noisy_gating=True
        ).to(self.device)
        
        # Create random inputs
        self.inputs = [
            torch.randn(self.batch_size, dim).to(self.device)
            for dim in self.input_dims
        ]
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, MigraineGating)
        self.assertEqual(self.model.input_dims, self.input_dims)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.num_experts, self.num_experts)
        self.assertEqual(self.model.top_k, self.top_k)
        self.assertTrue(self.model.noisy_gating)
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        # Set model to training mode
        self.model.train()
        
        # Run forward pass
        gates, load, importance = self.model(self.inputs, training=True)
        
        # Check output shapes
        self.assertEqual(gates.shape, (self.batch_size, self.num_experts))
        self.assertEqual(load.shape, (self.num_experts,))
        self.assertEqual(importance.shape, (self.batch_size, self.num_experts))
        
        # Check that gates sum to top_k for each sample
        gates_sum = gates.sum(dim=1)
        self.assertTrue(torch.allclose(gates_sum, torch.tensor([self.top_k] * self.batch_size, 
                                                              dtype=torch.float32, 
                                                              device=self.device)))
        
        # Check that importance is valid probability distribution
        importance_sum = importance.sum(dim=1)
        self.assertTrue(torch.allclose(importance_sum, torch.tensor([1.0] * self.batch_size, 
                                                                   dtype=torch.float32, 
                                                                   device=self.device), 
                                       atol=1e-5))
    
    def test_forward_pass_evaluation(self):
        """Test forward pass in evaluation mode."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            gates1, load1, importance1 = self.model(self.inputs, training=False)
            gates2, load2, importance2 = self.model(self.inputs, training=False)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(gates1, gates2))
        self.assertTrue(torch.allclose(load1, load2))
        self.assertTrue(torch.allclose(importance1, importance2))
    
    def test_top_k_routing(self):
        """Test that top-k routing works correctly."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            gates, _, _ = self.model(self.inputs, training=False)
        
        # Check that each sample has exactly top_k non-zero values
        for i in range(self.batch_size):
            non_zero = (gates[i] > 0).sum().item()
            self.assertEqual(non_zero, self.top_k)
    
    def test_load_balancing(self):
        """Test load balancing coefficient."""
        # Set load balancing coefficient
        self.model.load_balance_coef = 0.1
        
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Run forward pass
        optimizer.zero_grad()
        gates, load, importance = self.model(self.inputs, training=True)
        
        # Calculate load balancing loss
        load_loss = self.model.load_balance_coef * ((self.num_experts * load) - 1.0).pow(2).mean()
        
        # Check that load loss is non-zero
        self.assertGreater(load_loss.item(), 0.0)
        
        # Backpropagate loss
        load_loss.backward()
        optimizer.step()
        
        # Check that gradients were computed
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_noisy_gating(self):
        """Test noisy gating."""
        # Create model with noisy gating
        model_noisy = MigraineGating(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout_rate=0.1,
            noisy_gating=True
        ).to(self.device)
        
        # Create model without noisy gating
        model_clean = MigraineGating(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dropout_rate=0.1,
            noisy_gating=False
        ).to(self.device)
        
        # Set both models to training mode
        model_noisy.train()
        model_clean.train()
        
        # Run forward pass for both models
        gates_noisy, _, _ = model_noisy(self.inputs, training=True)
        gates_clean, _, _ = model_clean(self.inputs, training=True)
        
        # Check that outputs are different (due to noise)
        self.assertFalse(torch.allclose(gates_noisy, gates_clean))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_migraine_gating(
            input_dims=self.input_dims,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k
        )
        
        # Check model type
        self.assertIsInstance(model, MigraineGating)
        
        # Check model parameters
        self.assertEqual(model.input_dims, self.input_dims)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.num_experts, self.num_experts)
        self.assertEqual(model.top_k, self.top_k)


if __name__ == '__main__':
    unittest.main()
