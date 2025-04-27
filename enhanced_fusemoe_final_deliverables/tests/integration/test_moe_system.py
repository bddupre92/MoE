"""
Integration tests for the MoE system in the Enhanced FuseMoE system.

This module contains integration tests for the complete Mixture of Experts system
including expert models, gating network, and fusion mechanism.
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
from models.experts.sleep_expert import SleepExpert
from models.experts.weather_expert import WeatherExpert
from models.experts.stress_diet_expert import StressDietExpert
from models.experts.physio_expert import PhysioExpert
from utils.preprocessing.data_generator import MigraineSyntheticDataGenerator
from utils.preprocessing.data_preprocessor import MigraineDataPreprocessor


class TestMoESystem(unittest.TestCase):
    """Test cases for the complete MoE system."""
    
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
        self.output_dim = 16
        
        # Create registry and expert pool
        self.registry = ExpertRegistry()
        self.expert_pool = ScalableExpertPool(self.registry)
        
        # Add experts to pool
        self.expert_pool.add_expert(
            name='sleep',
            input_dim=self.expert_dims['sleep'],
            hidden_dim=32,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.2
        )
        
        self.expert_pool.add_expert(
            name='weather',
            input_dim=self.expert_dims['weather'],
            hidden_dim=32,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.2
        )
        
        self.expert_pool.add_expert(
            name='stress_diet',
            input_dim=self.expert_dims['stress_diet'],
            hidden_dim=32,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.2
        )
        
        self.expert_pool.add_expert(
            name='physio',
            input_dim=self.expert_dims['physio'],
            hidden_dim=32,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.2
        )
        
        # Create gating network
        input_dims = [self.expert_dims[expert_type] for expert_type in self.expert_types]
        self.gating = MigraineGating(
            input_dims=input_dims,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            top_k=2,
            dropout_rate=0.2,
            noisy_gating=True
        )
        
        # Create fusion mechanism
        self.fusion = MigraineFusion(
            expert_output_dim=self.output_dim,
            hidden_dim=32,
            num_experts=len(self.expert_types),
            dropout_rate=0.2
        )
        
        # Create MoE model
        self.model = DynamicMigraineMoE(self.expert_pool, self.gating, self.fusion)
        self.model.to(self.device)
    
    def test_forward_pass(self):
        """Test forward pass through the complete MoE system."""
        # Get a batch from the dataloader
        for inputs, targets in self.train_loader:
            # Move inputs to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Check output shape
            self.assertEqual(outputs.shape, (self.batch_size, 1))
            break
    
    def test_training_step(self):
        """Test a training step with the complete MoE system."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Set model to training mode
        self.model.train()
        
        # Get a batch from the dataloader
        for inputs, targets in self.train_loader:
            # Move inputs and targets to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Check that gradients are computed
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)
            
            # Update weights
            optimizer.step()
            break
    
    def test_evaluation_step(self):
        """Test an evaluation step with the complete MoE system."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Get a batch from the dataloader
        for inputs, targets in self.val_loader:
            # Move inputs and targets to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Calculate loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
            
            # Check that loss is a scalar
            self.assertEqual(loss.dim(), 0)
            break
    
    def test_expert_selection(self):
        """Test that the gating network selects the appropriate experts."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Get a batch from the dataloader
        for inputs, _ in self.train_loader:
            # Move inputs to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Get expert outputs and gates
            with torch.no_grad():
                # Forward pass through experts
                expert_outputs = {}
                for expert_name in self.expert_pool.list_experts():
                    expert = self.expert_pool.get_expert(expert_name)
                    expert_input = inputs[expert_name]
                    expert_outputs[expert_name] = expert(expert_input)
                
                # Forward pass through gating network
                gates = self.gating([inputs[expert_name] for expert_name in self.expert_types])
                
                # Check that gates sum to top_k for each sample
                gates_sum = torch.sum(gates, dim=1)
                self.assertTrue(torch.allclose(gates_sum, torch.tensor([self.gating.top_k] * self.batch_size, 
                                                                      dtype=torch.float32)))
                
                # Check that gates are between 0 and 1
                self.assertTrue((gates >= 0).all())
                self.assertTrue((gates <= 1).all())
            break
    
    def test_fusion_mechanism(self):
        """Test that the fusion mechanism combines expert outputs correctly."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Get a batch from the dataloader
        for inputs, _ in self.train_loader:
            # Move inputs to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Get expert outputs and gates
            with torch.no_grad():
                # Forward pass through experts
                expert_outputs = {}
                for expert_name in self.expert_pool.list_experts():
                    expert = self.expert_pool.get_expert(expert_name)
                    expert_input = inputs[expert_name]
                    expert_outputs[expert_name] = expert(expert_input)
                
                # Stack expert outputs
                stacked_outputs = torch.stack([expert_outputs[name] for name in self.expert_types], dim=1)
                
                # Forward pass through gating network
                gates = self.gating([inputs[expert_name] for expert_name in self.expert_types])
                
                # Forward pass through fusion mechanism
                fused_output = self.fusion(stacked_outputs, gates)
                
                # Check output shape
                self.assertEqual(fused_output.shape, (self.batch_size, 1))
            break
    
    def test_end_to_end_training(self):
        """Test end-to-end training for a few epochs."""
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train for a few epochs
        num_epochs = 2
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                # Move inputs and targets to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update running loss
                train_loss += loss.item()
                
                # Only run a few batches for testing
                if batch_idx >= 2:
                    break
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    # Move inputs and targets to device
                    for key in inputs:
                        inputs[key] = inputs[key].to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                    
                    # Update running loss
                    val_loss += loss.item()
                    
                    # Only run a few batches for testing
                    if batch_idx >= 2:
                        break
        
        # Check that model parameters have been updated
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Create a copy of the parameter
                param_copy = param.clone().detach()
                
                # Train for one more batch
                self.model.train()
                for inputs, targets in self.train_loader:
                    # Move inputs and targets to device
                    for key in inputs:
                        inputs[key] = inputs[key].to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    break
                
                # Check that parameter has been updated
                self.assertFalse(torch.allclose(param, param_copy))
                break


if __name__ == '__main__':
    unittest.main()
