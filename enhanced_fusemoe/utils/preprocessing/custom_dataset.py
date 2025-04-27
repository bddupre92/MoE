"""
Custom Dataset implementation for Enhanced FuseMoE system.

This module provides a custom dataset implementation for the Enhanced FuseMoE system
that properly formats inputs for the MigraineFusionMoE model and supports device transfer.
"""

import torch
from typing import List, Dict, Tuple, Any, Optional, Union


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for the Enhanced FuseMoE system.
    
    This dataset formats the data for the MigraineFusionMoE model, which expects
    a dictionary mapping expert names to input tensors. It also provides device
    transfer support to move data between devices.
    
    Attributes:
        expert_inputs: List of input tensors for each expert
        targets: Target tensor
        expert_names: List of expert names
        num_samples: Number of samples in the dataset
        device: Device to store the data on
    """
    
    def __init__(self, expert_inputs: List[torch.Tensor], 
                targets: torch.Tensor, 
                expert_names: List[str],
                device: Optional[torch.device] = None):
        """
        Initialize the custom dataset.
        
        Args:
            expert_inputs: List of input tensors for each expert
            targets: Target tensor
            expert_names: List of expert names
            device: Device to store the data on (default: None, keeps data on original device)
        """
        self.expert_names = expert_names
        self.device = device
        
        # Move data to device if specified
        if device is not None:
            self.expert_inputs = [inputs.to(device) for inputs in expert_inputs]
            self.targets = targets.to(device)
        else:
            self.expert_inputs = expert_inputs
            self.targets = targets
        
        # Ensure all inputs have the same number of samples
        self.num_samples = len(targets)
        for inputs in self.expert_inputs:
            assert len(inputs) == self.num_samples, "All inputs must have the same number of samples"
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
                - Dictionary mapping expert names to input tensors
                - Target tensor
        """
        # Create input dictionary
        inputs = {
            name: self.expert_inputs[i][idx]
            for i, name in enumerate(self.expert_names)
        }
        
        # Get target
        target = self.targets[idx]
        
        return inputs, target
    
    def to(self, device: torch.device) -> 'CustomDataset':
        """
        Move the dataset to the specified device.
        
        Args:
            device: Device to move the data to
            
        Returns:
            Self with data moved to the specified device
        """
        # Create a new dataset with data moved to the specified device
        return CustomDataset(
            expert_inputs=[inputs.to(device) for inputs in self.expert_inputs],
            targets=self.targets.to(device),
            expert_names=self.expert_names,
            device=device
        )
    
    @staticmethod
    def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function for DataLoader.
        
        This function collates a batch of samples into a single batch.
        
        Args:
            batch: List of samples, each containing a dictionary of inputs and a target
            
        Returns:
            Tuple containing:
                - Dictionary mapping expert names to batched input tensors
                - Batched target tensor
        """
        # Extract inputs and targets
        inputs_list = [sample[0] for sample in batch]
        targets = torch.stack([sample[1] for sample in batch])
        
        # Get expert names from first sample
        expert_names = list(inputs_list[0].keys())
        
        # Create batched inputs
        batched_inputs = {
            name: torch.stack([inputs[name] for inputs in inputs_list])
            for name in expert_names
        }
        
        return batched_inputs, targets
    
    def get_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader for this dataset
        """
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    
    @classmethod
    def from_dict(cls, data_dict: Dict[str, torch.Tensor], expert_names: List[str], 
                target_name: str = 'target', device: Optional[torch.device] = None) -> 'CustomDataset':
        """
        Create a dataset from a dictionary of tensors.
        
        Args:
            data_dict: Dictionary mapping names to tensors
            expert_names: List of expert names to extract from the dictionary
            target_name: Name of the target tensor in the dictionary
            device: Device to store the data on
            
        Returns:
            CustomDataset created from the dictionary
        """
        # Extract expert inputs and target
        expert_inputs = [data_dict[name] for name in expert_names]
        targets = data_dict[target_name]
        
        # Create dataset
        return cls(
            expert_inputs=expert_inputs,
            targets=targets,
            expert_names=expert_names,
            device=device
        )
    
    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
             test_ratio: float = 0.15, shuffle: bool = True) -> Tuple['CustomDataset', 'CustomDataset', 'CustomDataset']:
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio: Ratio of samples to use for training
            val_ratio: Ratio of samples to use for validation
            test_ratio: Ratio of samples to use for testing
            shuffle: Whether to shuffle the data before splitting
            
        Returns:
            Tuple containing:
                - Training dataset
                - Validation dataset
                - Test dataset
        """
        # Check that ratios sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        # Create indices
        indices = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        
        # Calculate split points
        train_size = int(train_ratio * len(self))
        val_size = int(val_ratio * len(self))
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = self._subset(train_indices)
        val_dataset = self._subset(val_indices)
        test_dataset = self._subset(test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _subset(self, indices: torch.Tensor) -> 'CustomDataset':
        """
        Create a subset of the dataset.
        
        Args:
            indices: Indices to include in the subset
            
        Returns:
            Subset of the dataset
        """
        # Extract subset of expert inputs and targets
        subset_expert_inputs = [inputs[indices] for inputs in self.expert_inputs]
        subset_targets = self.targets[indices]
        
        # Create dataset
        return CustomDataset(
            expert_inputs=subset_expert_inputs,
            targets=subset_targets,
            expert_names=self.expert_names,
            device=self.device
        )
