"""
Unit tests for the expert registry in the Enhanced FuseMoE system.

This module contains unit tests for the ExpertRegistry, ScalableExpertPool, 
and DynamicMigraineMoE classes.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import expert registry
from models.experts.expert_registry import ExpertRegistry, ScalableExpertPool, DynamicMigraineMoE
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusion


class TestExpertRegistry(unittest.TestCase):
    """Test cases for the ExpertRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ExpertRegistry()
    
    def test_register_expert(self):
        """Test registering an expert."""
        # Create a simple expert model
        expert = torch.nn.Linear(10, 5)
        
        # Register expert
        self.registry.register('test_expert', expert)
        
        # Check that expert is registered
        self.assertTrue('test_expert' in self.registry.experts)
        self.assertEqual(self.registry.experts['test_expert'], expert)
    
    def test_get_expert(self):
        """Test getting an expert."""
        # Create a simple expert model
        expert = torch.nn.Linear(10, 5)
        
        # Register expert
        self.registry.register('test_expert', expert)
        
        # Get expert
        retrieved_expert = self.registry.get('test_expert')
        
        # Check that retrieved expert is correct
        self.assertEqual(retrieved_expert, expert)
    
    def test_get_nonexistent_expert(self):
        """Test getting a nonexistent expert."""
        # Try to get nonexistent expert
        with self.assertRaises(KeyError):
            self.registry.get('nonexistent_expert')
    
    def test_list_experts(self):
        """Test listing experts."""
        # Create simple expert models
        expert1 = torch.nn.Linear(10, 5)
        expert2 = torch.nn.Linear(8, 4)
        
        # Register experts
        self.registry.register('expert1', expert1)
        self.registry.register('expert2', expert2)
        
        # List experts
        expert_list = self.registry.list()
        
        # Check that list is correct
        self.assertEqual(set(expert_list), {'expert1', 'expert2'})
    
    def test_remove_expert(self):
        """Test removing an expert."""
        # Create a simple expert model
        expert = torch.nn.Linear(10, 5)
        
        # Register expert
        self.registry.register('test_expert', expert)
        
        # Remove expert
        self.registry.remove('test_expert')
        
        # Check that expert is removed
        self.assertFalse('test_expert' in self.registry.experts)
    
    def test_remove_nonexistent_expert(self):
        """Test removing a nonexistent expert."""
        # Try to remove nonexistent expert
        with self.assertRaises(KeyError):
            self.registry.remove('nonexistent_expert')


class TestScalableExpertPool(unittest.TestCase):
    """Test cases for the ScalableExpertPool class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ExpertRegistry()
        self.expert_pool = ScalableExpertPool(self.registry)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_add_expert(self):
        """Test adding an expert."""
        # Add expert
        self.expert_pool.add_expert(
            name='test_expert',
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_layers=2,
            dropout_rate=0.1
        )
        
        # Check that expert is added
        self.assertTrue('test_expert' in self.expert_pool.list_experts())
    
    def test_get_expert(self):
        """Test getting an expert."""
        # Add expert
        self.expert_pool.add_expert(
            name='test_expert',
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_layers=2,
            dropout_rate=0.1
        )
        
        # Get expert
        expert = self.expert_pool.get_expert('test_expert')
        
        # Check that expert is correct
        self.assertEqual(expert.input_dim, 10)
        self.assertEqual(expert.hidden_dim, 20)
        self.assertEqual(expert.output_dim, 5)
    
    def test_remove_expert(self):
        """Test removing an expert."""
        # Add expert
        self.expert_pool.add_expert(
            name='test_expert',
            input_dim=10,
            hidden_dim=20,
            output_dim=5,
            num_layers=2,
            dropout_rate=0.1
        )
        
        # Remove expert
        self.expert_pool.remove_expert('test_expert')
        
        # Check that expert is removed
        self.assertFalse('test_expert' in self.expert_pool.list_experts())
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Add experts
        self.expert_pool.add_expert(
            name='expert1',
            input_dim=5,
            hidden_dim=10,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        self.expert_pool.add_expert(
            name='expert2',
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        # Create inputs
        inputs = {
            'expert1': torch.randn(2, 5).to(self.device),
            'expert2': torch.randn(2, 4).to(self.device)
        }
        
        # Move expert pool to device
        self.expert_pool.to(self.device)
        
        # Run forward pass
        outputs = self.expert_pool(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (2, 2, 3))  # [batch_size, num_experts, output_dim]
    
    def test_dynamic_expert_addition(self):
        """Test dynamic addition of experts."""
        # Add initial expert
        self.expert_pool.add_expert(
            name='expert1',
            input_dim=5,
            hidden_dim=10,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        # Create inputs
        inputs = {
            'expert1': torch.randn(2, 5).to(self.device)
        }
        
        # Move expert pool to device
        self.expert_pool.to(self.device)
        
        # Run forward pass
        outputs1 = self.expert_pool(inputs)
        
        # Add another expert
        self.expert_pool.add_expert(
            name='expert2',
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        # Update inputs
        inputs['expert2'] = torch.randn(2, 4).to(self.device)
        
        # Run forward pass again
        outputs2 = self.expert_pool(inputs)
        
        # Check output shapes
        self.assertEqual(outputs1.shape, (2, 1, 3))  # [batch_size, num_experts, output_dim]
        self.assertEqual(outputs2.shape, (2, 2, 3))  # [batch_size, num_experts, output_dim]


class TestDynamicMigraineMoE(unittest.TestCase):
    """Test cases for the DynamicMigraineMoE class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ExpertRegistry()
        self.expert_pool = ScalableExpertPool(self.registry)
        self.batch_size = 4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Add experts
        self.expert_pool.add_expert(
            name='expert1',
            input_dim=5,
            hidden_dim=10,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        self.expert_pool.add_expert(
            name='expert2',
            input_dim=4,
            hidden_dim=8,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        # Create gating network
        self.gating = MigraineGating(
            input_dims=[5, 4],
            hidden_dim=6,
            num_experts=2,
            top_k=1,
            dropout_rate=0.1,
            noisy_gating=False
        )
        
        # Create fusion mechanism
        self.fusion = MigraineFusion(
            expert_output_dim=3,
            hidden_dim=6,
            num_experts=2,
            dropout_rate=0.1
        )
        
        # Create MoE model
        self.model = DynamicMigraineMoE(
            expert_pool=self.expert_pool,
            gating=self.gating,
            fusion=self.fusion
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Create inputs
        self.inputs = {
            'expert1': torch.randn(self.batch_size, 5).to(self.device),
            'expert2': torch.randn(self.batch_size, 4).to(self.device)
        }
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, DynamicMigraineMoE)
        self.assertEqual(self.model.expert_pool, self.expert_pool)
        self.assertEqual(self.model.gating, self.gating)
        self.assertEqual(self.model.fusion, self.fusion)
    
    def test_forward_pass_training(self):
        """Test forward pass in training mode."""
        # Set model to training mode
        self.model.train()
        
        # Run forward pass
        output = self.model(self.inputs, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))  # [batch_size, 1] for binary classification
    
    def test_forward_pass_evaluation(self):
        """Test forward pass in evaluation mode."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.inputs, training=False)
            output2 = self.model(self.inputs, training=False)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))
    
    def test_dynamic_expert_addition(self):
        """Test dynamic addition of experts."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with initial experts
        with torch.no_grad():
            output1 = self.model(self.inputs, training=False)
        
        # Add another expert
        self.expert_pool.add_expert(
            name='expert3',
            input_dim=6,
            hidden_dim=12,
            output_dim=3,
            num_layers=1,
            dropout_rate=0.1
        )
        
        # Update gating network
        new_gating = MigraineGating(
            input_dims=[5, 4, 6],
            hidden_dim=6,
            num_experts=3,
            top_k=1,
            dropout_rate=0.1,
            noisy_gating=False
        )
        
        # Update fusion mechanism
        new_fusion = MigraineFusion(
            expert_output_dim=3,
            hidden_dim=6,
            num_experts=3,
            dropout_rate=0.1
        )
        
        # Update model
        self.model.update_components(
            gating=new_gating,
            fusion=new_fusion
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Update inputs
        new_inputs = self.inputs.copy()
        new_inputs['expert3'] = torch.randn(self.batch_size, 6).to(self.device)
        
        # Run forward pass with new expert
        with torch.no_grad():
            output2 = self.model(new_inputs, training=False)
        
        # Check output shapes
        self.assertEqual(output1.shape, (self.batch_size, 1))
        self.assertEqual(output2.shape, (self.batch_size, 1))
    
    def test_model_training(self):
        """Test model training."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randint(0, 2, (self.batch_size, 1)).float().to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.inputs, training=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
