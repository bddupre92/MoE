"""
Training data generator module for the enhanced data pipeline.

This module provides functionality for generating training data using the enhanced data pipeline.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional

class TrainingDataGenerator:
    """
    Training data generator for the enhanced data pipeline.
    
    This class generates training data using the enhanced data pipeline and prepares
    it for use with the MoE model.
    """
    
    def __init__(self, integration, output_dir=None):
        """
        Initialize the training data generator.
        
        Args:
            integration: EnhancedDataPipelineIntegration instance
            output_dir: Optional output directory for generated data
        """
        self.integration = integration
        self.output_dir = output_dir
        print("TrainingDataGenerator initialized")
    
    def generate_training_data(self, num_samples=None):
        """
        Generate training data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary containing training data loaders and raw data
        """
        if num_samples is None:
            num_samples = 100  # Default number of samples
        
        print(f"Generating training data with {num_samples} samples...")
        
        # Generate data using the integration
        raw_data = self.integration.generate_data(num_samples)
        
        # Split data into train, validation, and test sets
        train_data, val_data, test_data = self._split_data(raw_data)
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data)
        val_loader = self._create_data_loader(val_data)
        test_loader = self._create_data_loader(test_data)
        
        print("Training data generation complete")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'raw_data': raw_data
        }
    
    def _split_data(self, data):
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Dictionary containing data to split
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        num_samples = data['target'].shape[0]
        indices = np.random.permutation(num_samples)
        
        # Use 70% for training, 15% for validation, 15% for testing
        train_idx = indices[:int(0.7 * num_samples)]
        val_idx = indices[int(0.7 * num_samples):int(0.85 * num_samples)]
        test_idx = indices[int(0.85 * num_samples):]
        
        # Split each modality
        train_data = {k: v[train_idx] for k, v in data.items()}
        val_data = {k: v[val_idx] for k, v in data.items()}
        test_data = {k: v[test_idx] for k, v in data.items()}
        
        return train_data, val_data, test_data
    
    def _create_data_loader(self, data, batch_size=32):
        """
        Create a PyTorch DataLoader from the data.
        
        Args:
            data: Dictionary containing data
            batch_size: Batch size for the DataLoader
            
        Returns:
            PyTorch DataLoader
        """
        # Convert NumPy arrays to PyTorch tensors
        tensors = [torch.tensor(data[k], dtype=torch.float32) for k in ['sleep', 'weather', 'stress_diet', 'physio']]
        target_tensor = torch.tensor(data['target'], dtype=torch.float32)
        
        # Create dataset and data loader
        dataset = TensorDataset(*tensors, target_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
