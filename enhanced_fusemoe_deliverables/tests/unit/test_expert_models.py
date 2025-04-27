"""
Unit tests for expert models in the Enhanced FuseMoE system.

This module contains unit tests for the expert models including:
- SleepExpert
- WeatherExpert
- StressDietExpert
- PhysioExpert
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import expert models
from models.experts.sleep_expert import SleepExpert, create_sleep_expert
from models.experts.weather_expert import WeatherExpert, create_weather_expert
from models.experts.stress_diet_expert import StressDietExpert, create_stress_diet_expert
from models.experts.physio_expert import PhysioExpert, create_physio_expert


class TestSleepExpert(unittest.TestCase):
    """Test cases for the SleepExpert model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 6
        self.hidden_dim = 32
        self.output_dim = 16
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = SleepExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create random input
        self.input = torch.randn(self.batch_size, self.input_dim).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, SleepExpert)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Run forward pass
        output = self.model(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_sleep_expert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Check model type
        self.assertIsInstance(model, SleepExpert)
        
        # Check model parameters
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
    
    def test_model_training(self):
        """Test model training."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randn(self.batch_size, self.output_dim).to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.input)
            output2 = self.model(self.input)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))


class TestWeatherExpert(unittest.TestCase):
    """Test cases for the WeatherExpert model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.hidden_dim = 32
        self.output_dim = 16
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = WeatherExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create random input
        self.input = torch.randn(self.batch_size, self.input_dim).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, WeatherExpert)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Run forward pass
        output = self.model(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_weather_expert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Check model type
        self.assertIsInstance(model, WeatherExpert)
        
        # Check model parameters
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
    
    def test_model_training(self):
        """Test model training."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randn(self.batch_size, self.output_dim).to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.input)
            output2 = self.model(self.input)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))


class TestStressDietExpert(unittest.TestCase):
    """Test cases for the StressDietExpert model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 6
        self.hidden_dim = 32
        self.output_dim = 16
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = StressDietExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create random input
        self.input = torch.randn(self.batch_size, self.input_dim).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, StressDietExpert)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Run forward pass
        output = self.model(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_stress_diet_expert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Check model type
        self.assertIsInstance(model, StressDietExpert)
        
        # Check model parameters
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
    
    def test_model_training(self):
        """Test model training."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randn(self.batch_size, self.output_dim).to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.input)
            output2 = self.model(self.input)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))


class TestPhysioExpert(unittest.TestCase):
    """Test cases for the PhysioExpert model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 5
        self.hidden_dim = 32
        self.output_dim = 16
        self.batch_size = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = PhysioExpert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=2,
            dropout_rate=0.1
        ).to(self.device)
        
        # Create random input
        self.input = torch.randn(self.batch_size, self.input_dim).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, PhysioExpert)
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        self.assertEqual(self.model.output_dim, self.output_dim)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Run forward pass
        output = self.model(self.input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
    
    def test_factory_function(self):
        """Test factory function."""
        # Create model using factory function
        model = create_physio_expert(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # Check model type
        self.assertIsInstance(model, PhysioExpert)
        
        # Check model parameters
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
    
    def test_model_training(self):
        """Test model training."""
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create random target
        target = torch.randn(self.batch_size, self.output_dim).to(self.device)
        
        # Run training step
        optimizer.zero_grad()
        output = self.model(self.input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Set model to evaluation mode
        self.model.eval()
        
        # Run forward pass with no grad
        with torch.no_grad():
            output1 = self.model(self.input)
            output2 = self.model(self.input)
        
        # Check that outputs are identical (deterministic in eval mode)
        self.assertTrue(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
