"""
Unit tests for the fusion mechanism in the Enhanced FuseMoE system.

This module contains unit tests for the MigraineFusion class and related functions.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import fusion mechanism
from models.fusion.migraine_fusion import MigraineFusion, create_migraine_fusion


class TestMigraineFusion(unittest.TestCase):
    """Test cases for the MigraineFusion mechanism."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expert_output_dim = 16
        self.hidden_dim = 32
        self.num_experts = 4
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = MigraineFusion(
            expert_output_dim=self.expert_output_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create random expert outputs and gates
        self.expert_outputs = torch.randn(self.batch_size, self.num_experts, self.expert_output_dim).to(self.device)
        self.gates = torch.softmax(torch.randn(self.batch_size, self.num_experts), dim=1).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, MigraineFusion)
        self.assertEqual(self.model.expert_output_dim, self.expert_output_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.num_experts, self.num_experts)
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        # Set model to training mode
        self.model.train()
        
        # Run forward pass
        output = self.model(self.expert_outputs, self.gates, training=True)
        
        # Check output shape (batch_size, 1) for binary classification
        self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_forward_pass_evaluation(self):
        """Test forward pass in evaluation mode."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.expert_outputs, self.gates, training=False)
            output2 = self.model(self.expert_outputs, self.gates, training=False)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_weighted_combination(self):
        """Test that weighted combination works correctly."""
        # Create simple expert outputs and gates
        expert_outputs = torch.ones(1, self.num_experts, self.expert_output_dim).to(self.device)
        gates = torch.zeros(1, self.num_experts).to(self.device)
        gates[0, 0] = 1.0  # Only use first expert
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            # Get output with only first expert
            output1 = self.model(expert_outputs, gates, training=False)
            
            # Change gates to use only second expert
            gates[0, 0] = 0.0
            gates[0, 1] = 1.0
            output2 = self.model(expert_outputs, gates, training=False)
        
        # Check that outputs are different (different experts used)
        self.assertFalse(torch.allclose(output1, output2))
    
    def test_attention_mechanism(self):
        """Test attention mechanism."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randint(0, 2, (self.batch_size, 1)).float().to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.expert_outputs, self.gates, training=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        
        # Check that attention weights have gradients
        self.assertIsNotNone(self.model.attention.weight.grad)
        
        # Apply gradients
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_migraine_fusion(
            expert_output_dim=self.expert_output_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts
        )
        
        # Check model type
        self.assertIsInstance(model, MigraineFusion)
        
        # Check model parameters
        self.assertEqual(model.expert_output_dim, self.expert_output_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.num_experts, self.num_experts)
    
    def test_dropout(self):
        """Test dropout behavior."""
        # Create model with high dropout
        model_dropout = MigraineFusion(
            expert_output_dim=self.expert_output_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            dropout_rate=0.5
        ).to(self.device)
        
        # Set model to training mode
        model_dropout.train()
        
        # Run forward pass multiple times
        output1 = model_dropout(self.expert_outputs, self.gates, training=True)
        output2 = model_dropout(self.expert_outputs, self.gates, training=True)
        
        # Check that outputs are different (due to dropout)
        self.assertFalse(torch.allclose(output1, output2))
        
        # Set model to evaluation mode
        model_dropout.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = model_dropout(self.expert_outputs, self.gates, training=False)
            output2 = model_dropout(self.expert_outputs, self.gates, training=False)
        
        # Check that outputs are identical (no dropout in eval mode)
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
