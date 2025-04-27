"""
Data Preprocessor for Enhanced FuseMoE

This module provides functionality for preprocessing data for the Enhanced FuseMoE
system for migraine prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class MigraineDataPreprocessor:
    """
    Preprocessor for migraine prediction data.
    
    This class provides methods for preprocessing data from different modalities
    for use with the Enhanced FuseMoE system.
    
    Attributes:
        feature_columns (Dict[str, List[str]]): Dictionary mapping modality names to feature column names
        sequence_length (int): Length of sequences for temporal data
        normalize (bool): Whether to normalize features
        means (Dict[str, np.ndarray]): Dictionary mapping modality names to feature means
        stds (Dict[str, np.ndarray]): Dictionary mapping modality names to feature standard deviations
    """
    
    def __init__(self, feature_columns: Dict[str, List[str]], 
                sequence_length: int = 7, normalize: bool = True):
        """
        Initialize the migraine data preprocessor.
        
        Args:
            feature_columns: Dictionary mapping modality names to feature column names
            sequence_length: Length of sequences for temporal data
            normalize: Whether to normalize features
        """
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.means = {}
        self.stds = {}
    
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Fit the preprocessor to the data.
        
        Args:
            data: Dictionary mapping modality names to DataFrames
        """
        if self.normalize:
            for modality, df in data.items():
                if modality in self.feature_columns:
                    features = df[self.feature_columns[modality]].values
                    self.means[modality] = np.mean(features, axis=0)
                    self.stds[modality] = np.std(features, axis=0)
                    # Replace zero standard deviations with 1.0
                    self.stds[modality] = np.where(self.stds[modality] < 1e-7, 1.0, self.stds[modality])
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Transform the data.
        
        Args:
            data: Dictionary mapping modality names to DataFrames
            
        Returns:
            Dictionary mapping modality names to preprocessed arrays
        """
        preprocessed = {}
        
        for modality, df in data.items():
            if modality in self.feature_columns:
                features = df[self.feature_columns[modality]].values
                
                # Normalize if enabled
                if self.normalize and modality in self.means and modality in self.stds:
                    features = (features - self.means[modality]) / self.stds[modality]
                
                # Handle temporal data
                if modality in ['sleep']:
                    # Create sequences
                    sequences = []
                    for i in range(len(features) - self.sequence_length + 1):
                        sequences.append(features[i:i+self.sequence_length])
                    
                    if sequences:
                        preprocessed[modality] = np.array(sequences)
                    else:
                        # If not enough data for a sequence, pad with zeros
                        pad_length = self.sequence_length - len(features)
                        padded = np.vstack([np.zeros((pad_length, features.shape[1])), features])
                        preprocessed[modality] = np.array([padded])
                else:
                    # Non-temporal data
                    preprocessed[modality] = features
        
        return preprocessed
    
    def fit_transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Fit the preprocessor to the data and transform it.
        
        Args:
            data: Dictionary mapping modality names to DataFrames
            
        Returns:
            Dictionary mapping modality names to preprocessed arrays
        """
        self.fit(data)
        return self.transform(data)
    
    def get_feature_dims(self) -> Dict[str, int]:
        """
        Get the feature dimensions for each modality.
        
        Returns:
            Dictionary mapping modality names to feature dimensions
        """
        dims = {}
        
        for modality, columns in self.feature_columns.items():
            dims[modality] = len(columns)
        
        return dims
    
    def save(self, filepath: str) -> None:
        """
        Save the preprocessor to a file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        state = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'normalize': self.normalize,
            'means': self.means,
            'stds': self.stds
        }
        torch.save(state, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'MigraineDataPreprocessor':
        """
        Load a preprocessor from a file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        state = torch.load(filepath)
        preprocessor = cls(
            feature_columns=state['feature_columns'],
            sequence_length=state['sequence_length'],
            normalize=state['normalize']
        )
        preprocessor.means = state['means']
        preprocessor.stds = state['stds']
        return preprocessor


class MigraineDataset(Dataset):
    """
    Dataset for migraine prediction.
    
    This class provides a PyTorch Dataset for migraine prediction data.
    
    Attributes:
        data (Dict[str, np.ndarray]): Dictionary mapping modality names to data arrays
        targets (np.ndarray): Target values (migraine occurrence)
        modalities (List[str]): List of modality names
    """
    
    def __init__(self, data: Dict[str, np.ndarray], targets: np.ndarray):
        """
        Initialize the migraine dataset.
        
        Args:
            data: Dictionary mapping modality names to data arrays
            targets: Target values (migraine occurrence)
        """
        self.data = data
        self.targets = targets
        self.modalities = list(data.keys())
        
        # Validate data
        self.validate_data()
    
    def validate_data(self) -> None:
        """
        Validate the dataset.
        
        Raises:
            ValueError: If data is invalid
        """
        # Check that all modalities have the same number of samples
        num_samples = None
        
        for modality, array in self.data.items():
            if num_samples is None:
                num_samples = len(array)
            elif len(array) != num_samples:
                raise ValueError(f"Modality '{modality}' has {len(array)} samples, but expected {num_samples}")
        
        # Check that targets have the same number of samples
        if len(self.targets) != num_samples:
            raise ValueError(f"Targets have {len(self.targets)} samples, but expected {num_samples}")
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (inputs, target)
        """
        inputs = {}
        
        for modality in self.modalities:
            inputs[modality] = torch.tensor(self.data[modality][idx], dtype=torch.float32)
        
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return inputs, target


def create_dataloaders(train_data: Dict[str, np.ndarray], train_targets: np.ndarray,
                      val_data: Dict[str, np.ndarray], val_targets: np.ndarray,
                      test_data: Dict[str, np.ndarray], test_targets: np.ndarray,
                      batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_data: Dictionary mapping modality names to training data arrays
        train_targets: Training target values
        val_data: Dictionary mapping modality names to validation data arrays
        val_targets: Validation target values
        test_data: Dictionary mapping modality names to test data arrays
        test_targets: Test target values
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MigraineDataset(train_data, train_targets)
    val_dataset = MigraineDataset(val_data, val_targets)
    test_dataset = MigraineDataset(test_data, test_targets)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
