"""
Input shape adapter for expert models in the Enhanced FuseMoE system.

This module provides utilities for handling input shape compatibility between
different expert models in the Enhanced FuseMoE system.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class InputShapeAdapter:
    """
    Adapter for handling input shape compatibility between different expert models.
    
    This class provides methods for transforming input data to match the expected
    shape of different expert models in the Enhanced FuseMoE system.
    
    Attributes:
        input_shapes (Dict[str, Tuple]): Dictionary mapping expert names to expected input shapes
    """
    
    def __init__(self, input_shapes: Optional[Dict[str, Tuple]] = None):
        """
        Initialize the input shape adapter.
        
        Args:
            input_shapes: Dictionary mapping expert names to expected input shapes
                          (default: predefined shapes for standard experts)
        """
        # Default input shapes for standard experts
        self.input_shapes = input_shapes or {
            'sleep_expert': (None, 7, 6),  # [batch_size, sequence_length, input_dim]
            'weather_expert': (None, 6),   # [batch_size, input_dim]
            'stress_diet_expert': (None, 6),  # [batch_size, input_dim]
            'physio_expert': (None, 6)     # [batch_size, input_dim]
        }
    
    def adapt_input(self, x: torch.Tensor, expert_name: str) -> torch.Tensor:
        """
        Adapt input tensor to match the expected shape of the specified expert.
        
        Args:
            x: Input tensor
            expert_name: Name of the expert model
            
        Returns:
            Adapted input tensor with shape matching the expert's expected input shape
        """
        if expert_name not in self.input_shapes:
            raise ValueError(f"Unknown expert name: {expert_name}")
        
        expected_shape = self.input_shapes[expert_name]
        
        # Get current shape
        current_shape = x.shape
        
        # Check if shapes already match (ignoring batch dimension)
        if len(current_shape) == len(expected_shape) and all(
            c == e or e is None for c, e in zip(current_shape, expected_shape)
        ):
            return x
        
        # Handle 3D to 2D conversion (for non-sequence experts)
        if len(current_shape) == 3 and len(expected_shape) == 2:
            # If input is [batch_size, sequence_length, features]
            # and expert expects [batch_size, features]
            
            # Option 1: Take the last time step
            # return x[:, -1, :]
            
            # Option 2: Average across time steps
            return torch.mean(x, dim=1)
        
        # Handle 2D to 3D conversion (for sequence experts)
        if len(current_shape) == 2 and len(expected_shape) == 3:
            # If input is [batch_size, features]
            # and expert expects [batch_size, sequence_length, features]
            
            # Expand to add sequence dimension with length 1
            return x.unsqueeze(1).expand(-1, expected_shape[1], -1)
        
        # Handle dimension mismatch in features
        if len(current_shape) == len(expected_shape):
            if len(current_shape) == 2:  # 2D tensors
                batch_size, features = current_shape
                _, expected_features = expected_shape
                
                if features != expected_features and expected_features is not None:
                    # Use linear projection to match feature dimensions
                    adapter = nn.Linear(features, expected_features).to(x.device)
                    return adapter(x)
            
            elif len(current_shape) == 3:  # 3D tensors
                batch_size, seq_len, features = current_shape
                _, expected_seq_len, expected_features = expected_shape
                
                # Handle sequence length mismatch
                if seq_len != expected_seq_len and expected_seq_len is not None:
                    # Interpolate to match sequence length
                    x = torch.nn.functional.interpolate(
                        x.transpose(1, 2),  # [batch_size, features, seq_len]
                        size=expected_seq_len,
                        mode='linear'
                    ).transpose(1, 2)  # [batch_size, expected_seq_len, features]
                
                # Handle feature dimension mismatch
                if features != expected_features and expected_features is not None:
                    # Use linear projection to match feature dimensions
                    adapter = nn.Linear(features, expected_features).to(x.device)
                    return adapter(x.reshape(-1, features)).reshape(batch_size, -1, expected_features)
        
        # If we can't adapt, return original tensor and log warning
        print(f"Warning: Could not adapt input shape {current_shape} to expected shape {expected_shape} for {expert_name}")
        return x


class ExpertInputWrapper(nn.Module):
    """
    Wrapper for expert models to handle input shape compatibility.
    
    This wrapper ensures that input data is properly transformed to match
    the expected shape of the wrapped expert model.
    
    Attributes:
        expert (nn.Module): Expert model to wrap
        expert_name (str): Name of the expert model
        adapter (InputShapeAdapter): Input shape adapter
    """
    
    def __init__(self, expert: nn.Module, expert_name: str, adapter: Optional[InputShapeAdapter] = None):
        """
        Initialize the expert input wrapper.
        
        Args:
            expert: Expert model to wrap
            expert_name: Name of the expert model
            adapter: Input shape adapter (default: new InputShapeAdapter instance)
        """
        super(ExpertInputWrapper, self).__init__()
        
        self.expert = expert
        self.expert_name = expert_name
        self.adapter = adapter or InputShapeAdapter()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped expert model with input shape adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor from the expert model
        """
        # Adapt input shape
        adapted_x = self.adapter.adapt_input(x, self.expert_name)
        
        # Forward through expert
        return self.expert(adapted_x)
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped expert model.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from the wrapped expert model
        """
        if name in ['expert', 'expert_name', 'adapter']:
            return super().__getattr__(name)
        
        return getattr(self.expert, name)


def wrap_expert(expert: nn.Module, expert_name: str) -> ExpertInputWrapper:
    """
    Wrap an expert model with input shape handling.
    
    Args:
        expert: Expert model to wrap
        expert_name: Name of the expert model
        
    Returns:
        Wrapped expert model with input shape handling
    """
    return ExpertInputWrapper(expert, expert_name)
