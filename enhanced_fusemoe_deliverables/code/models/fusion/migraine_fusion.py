"""
Migraine Fusion Mechanism for Enhanced FuseMoE

This module provides the implementation of the fusion mechanism for the
Enhanced FuseMoE system for migraine prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class MigraineFusion(nn.Module):
    """
    Fusion mechanism for migraine prediction in the Enhanced FuseMoE system.
    
    This model combines outputs from multiple expert models to make the final
    migraine prediction. It uses a weighted combination of expert outputs
    based on the gating network's routing decisions.
    
    Attributes:
        expert_output_dim (int): Output dimension of expert models
        hidden_dim (int): Size of hidden layers
        num_experts (int): Number of expert models
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, expert_output_dim: int = 32, hidden_dim: int = 64, 
                num_experts: int = 4, dropout_rate: float = 0.2):
        """
        Initialize the migraine fusion mechanism.
        
        Args:
            expert_output_dim: Output dimension of expert models
            hidden_dim: Size of hidden layers
            num_experts: Number of expert models
            dropout_rate: Dropout probability
        """
        super(MigraineFusion, self).__init__()
        
        self.expert_output_dim = expert_output_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        
        # Expert-specific transformations
        self.expert_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(num_experts)
        ])
        
        # Attention mechanism for expert outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Final prediction layers
        self.prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, expert_outputs: torch.Tensor, gates: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Forward pass through the migraine fusion mechanism.
        
        Args:
            expert_outputs: Tensor of shape [batch_size, num_experts, expert_output_dim] containing outputs from expert models
            gates: Tensor of shape [batch_size, num_experts] containing gating weights for each expert
            training: Whether the model is in training mode
            
        Returns:
            Migraine prediction logits of shape [batch_size, 1]
        """
        batch_size = expert_outputs.size(0)
        
        # Transform expert outputs for each expert
        transformed_outputs = torch.zeros(
            batch_size, self.num_experts, self.hidden_dim, 
            device=expert_outputs.device
        )
        
        for i in range(self.num_experts):
            # Extract outputs for current expert
            expert_output = expert_outputs[:, i, :]  # [batch_size, expert_output_dim]
            
            # Apply expert-specific transformation
            transformed = self.expert_transforms[i](expert_output)  # [batch_size, hidden_dim]
            
            # Store transformed output
            transformed_outputs[:, i, :] = transformed
        
        # Apply attention mechanism
        attention_scores = self.attention(transformed_outputs).squeeze(-1)  # [batch_size, num_experts]
        
        # Combine attention scores with gating weights
        combined_weights = attention_scores * gates
        combined_weights = F.softmax(combined_weights, dim=1).unsqueeze(2)  # [batch_size, num_experts, 1]
        
        # Weighted combination of expert outputs
        weighted_outputs = transformed_outputs * combined_weights
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Apply dropout during training
        if training:
            combined_output = F.dropout(combined_output, p=self.dropout_rate, training=True)
        
        # Final prediction
        logits = self.prediction(combined_output)  # [batch_size, 1]
        
        return logits
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the migraine fusion mechanism.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'expert_output_dim': self.expert_output_dim,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'dropout_rate': self.dropout_rate
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MigraineFusion':
        """
        Create a migraine fusion mechanism from configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Initialized migraine fusion mechanism
        """
        return cls(
            expert_output_dim=config.get('expert_output_dim', 32),
            hidden_dim=config.get('hidden_dim', 64),
            num_experts=config.get('num_experts', 4),
            dropout_rate=config.get('dropout_rate', 0.2)
        )


class MigraineFusionMoE(nn.Module):
    """
    Mixture of Experts fusion mechanism for migraine prediction.
    
    This model extends the basic MigraineFusion mechanism by incorporating
    a mixture of experts approach, allowing for more specialized fusion
    strategies depending on the input data characteristics.
    
    Attributes:
        expert_output_dim (int): Output dimension of expert models
        hidden_dim (int): Size of hidden layers
        num_experts (int): Number of expert models
        num_fusion_experts (int): Number of fusion experts
        dropout_rate (float): Dropout probability
    """
    
    def __init__(self, expert_registry=None, gating=None, output_dim=None, 
                expert_output_dim: int = 32, hidden_dim: int = 64, 
                num_experts: int = 4, num_fusion_experts: int = 2,
                dropout_rate: float = 0.2):
        """
        Initialize the migraine fusion MoE mechanism.
        
        Args:
            expert_registry: Registry of expert models (for backward compatibility)
            gating: Gating network (for backward compatibility)
            output_dim: Output dimension (for backward compatibility)
            expert_output_dim: Output dimension of expert models
            hidden_dim: Size of hidden layers
            num_experts: Number of expert models
            num_fusion_experts: Number of fusion experts
            dropout_rate: Dropout probability
        """
        super(MigraineFusionMoE, self).__init__()
        
        # Handle backward compatibility
        if expert_registry is not None:
            num_experts = len(expert_registry)
        
        self.expert_registry = expert_registry
        self.gating = gating
        self.output_dim = output_dim or 1
        
        self.expert_output_dim = expert_output_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_fusion_experts = num_fusion_experts
        self.dropout_rate = dropout_rate
        
        # Create multiple fusion experts
        self.fusion_experts = nn.ModuleList([
            MigraineFusion(
                expert_output_dim=expert_output_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                dropout_rate=dropout_rate
            ) for _ in range(num_fusion_experts)
        ])
        
        # Fusion expert gating network
        self.fusion_gating = nn.Sequential(
            nn.Linear(expert_output_dim * num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_fusion_experts)
        )
        
        # Final output layer
        self.output_layer = nn.Linear(1, self.output_dim)
    
    def forward(self, inputs=None, training: bool = True):
        """
        Forward pass through the migraine fusion MoE mechanism.
        
        This method supports two calling conventions:
        1. With expert_outputs and gates for direct fusion
        2. With a dictionary of inputs for end-to-end processing
        
        Args:
            inputs: Either:
                   - Dictionary mapping expert names to input tensors (for end-to-end)
                   - Tuple of (expert_outputs, gates) for direct fusion
                   - expert_outputs tensor if gates is provided separately
            training: Whether the model is in training mode
            
        Returns:
            Migraine prediction logits
        """
        # Handle different input formats
        if isinstance(inputs, dict) and self.expert_registry is not None and self.gating is not None:
            # End-to-end processing
            # Get list of available experts
            expert_names = self.expert_registry.list()
            
            # Prepare inputs for gating network
            gating_inputs = [inputs[name] for name in expert_names if name in inputs]
            
            # Forward pass through gating network
            gates, load_balancing_loss, _ = self.gating(gating_inputs, training)
            
            # Process each expert
            batch_size = next(iter(inputs.values())).size(0)
            expert_outputs = torch.zeros(
                batch_size, self.num_experts, self.expert_output_dim, 
                device=next(iter(inputs.values())).device
            )
            
            for i, name in enumerate(expert_names):
                if name in inputs:
                    expert = self.expert_registry.get(name)
                    expert_output = expert(inputs[name])
                    expert_outputs[:, i, :] = expert_output
            
            # Forward pass through fusion mechanism
            logits, fusion_gates = self._fusion_forward(expert_outputs, gates, training)
            
            return logits, load_balancing_loss
        
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            # Direct fusion with expert_outputs and gates
            expert_outputs, gates = inputs
            logits, fusion_gates = self._fusion_forward(expert_outputs, gates, training)
            return logits
        
        elif torch.is_tensor(inputs):
            # Direct fusion with expert_outputs provided as first argument
            # This assumes gates is provided as a second positional argument
            expert_outputs = inputs
            gates = training  # In this case, the second argument is actually gates
            training = True   # Default to training mode
            logits, fusion_gates = self._fusion_forward(expert_outputs, gates, training)
            return logits
        
        else:
            raise ValueError("Invalid input format")
    
    def _fusion_forward(self, expert_outputs: torch.Tensor, gates: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal forward pass through the fusion mechanism.
        
        Args:
            expert_outputs: Tensor of shape [batch_size, num_experts, expert_output_dim]
            gates: Tensor of shape [batch_size, num_experts]
            training: Whether the model is in training mode
            
        Returns:
            Tuple of (logits, fusion_gates)
        """
        batch_size = expert_outputs.size(0)
        
        # Flatten expert outputs for fusion gating
        flattened_outputs = expert_outputs.view(batch_size, -1)  # [batch_size, num_experts * expert_output_dim]
        
        # Compute fusion expert gates
        fusion_gates = self.fusion_gating(flattened_outputs)  # [batch_size, num_fusion_experts]
        fusion_gates = F.softmax(fusion_gates, dim=1)
        
        # Apply each fusion expert
        fusion_outputs = torch.zeros(batch_size, self.num_fusion_experts, 1, device=expert_outputs.device)
        for i in range(self.num_fusion_experts):
            fusion_outputs[:, i, :] = self.fusion_experts[i](expert_outputs, gates, training)
        
        # Weighted combination of fusion expert outputs
        weighted_fusion = fusion_outputs * fusion_gates.unsqueeze(2)  # [batch_size, num_fusion_experts, 1]
        combined_fusion = weighted_fusion.sum(dim=1)  # [batch_size, 1]
        
        # Final output transformation
        logits = self.output_layer(combined_fusion)  # [batch_size, output_dim]
        
        return logits, fusion_gates
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the migraine fusion MoE mechanism.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'expert_output_dim': self.expert_output_dim,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'num_fusion_experts': self.num_fusion_experts,
            'dropout_rate': self.dropout_rate,
            'output_dim': self.output_dim
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MigraineFusionMoE':
        """
        Create a migraine fusion MoE mechanism from configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Initialized migraine fusion MoE mechanism
        """
        return cls(
            expert_output_dim=config.get('expert_output_dim', 32),
            hidden_dim=config.get('hidden_dim', 64),
            num_experts=config.get('num_experts', 4),
            num_fusion_experts=config.get('num_fusion_experts', 2),
            dropout_rate=config.get('dropout_rate', 0.2),
            output_dim=config.get('output_dim', 1)
        )


def create_migraine_fusion(expert_output_dim: int = 32, hidden_dim: int = 64,
                          num_experts: int = 4, dropout_rate: float = 0.2,
                          learning_rate: float = 0.001) -> MigraineFusion:
    """
    Create a migraine fusion mechanism.
    
    Args:
        expert_output_dim: Output dimension of expert models
        hidden_dim: Size of hidden layers
        num_experts: Number of expert models
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer (not used in return value)
        
    Returns:
        Initialized migraine fusion mechanism
    """
    model = MigraineFusion(
        expert_output_dim=expert_output_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        dropout_rate=dropout_rate
    )
    
    return model


def create_migraine_fusion_moe(expert_output_dim: int = 32, hidden_dim: int = 64,
                              num_experts: int = 4, num_fusion_experts: int = 2,
                              dropout_rate: float = 0.2,
                              learning_rate: float = 0.001) -> MigraineFusionMoE:
    """
    Create a migraine fusion MoE mechanism.
    
    Args:
        expert_output_dim: Output dimension of expert models
        hidden_dim: Size of hidden layers
        num_experts: Number of expert models
        num_fusion_experts: Number of fusion experts
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer (not used in return value)
        
    Returns:
        Initialized migraine fusion MoE mechanism
    """
    model = MigraineFusionMoE(
        expert_output_dim=expert_output_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_fusion_experts=num_fusion_experts,
        dropout_rate=dropout_rate
    )
    
    return model
