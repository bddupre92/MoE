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
    
    def forward(self, expert_outputs: List[torch.Tensor], gates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the migraine fusion mechanism.
        
        Args:
            expert_outputs: List of output tensors from expert models
            gates: Gating weights for each expert
            
        Returns:
            Migraine prediction probabilities
        """
        batch_size = expert_outputs[0].size(0)
        
        # Transform expert outputs
        transformed_outputs = []
        for i, output in enumerate(expert_outputs):
            transformed = self.expert_transforms[i](output)
            transformed_outputs.append(transformed)
        
        # Stack transformed outputs
        stacked_outputs = torch.stack(transformed_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        # Apply attention mechanism
        attention_scores = self.attention(stacked_outputs).squeeze(-1)  # [batch_size, num_experts]
        
        # Combine attention scores with gating weights
        combined_weights = attention_scores * gates
        combined_weights = F.softmax(combined_weights, dim=1).unsqueeze(2)  # [batch_size, num_experts, 1]
        
        # Weighted combination of expert outputs
        weighted_outputs = stacked_outputs * combined_weights
        combined_output = weighted_outputs.sum(dim=1)  # [batch_size, hidden_dim]
        
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


def create_migraine_fusion(expert_output_dim: int = 32, hidden_dim: int = 64,
                          num_experts: int = 4, dropout_rate: float = 0.2,
                          learning_rate: float = 0.001) -> Tuple[MigraineFusion, Dict[str, Any]]:
    """
    Create a migraine fusion mechanism with optimizer.
    
    Args:
        expert_output_dim: Output dimension of expert models
        hidden_dim: Size of hidden layers
        num_experts: Number of expert models
        dropout_rate: Dropout probability
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (model, optimizer_dict)
    """
    model = MigraineFusion(
        expert_output_dim=expert_output_dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        dropout_rate=dropout_rate
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, {'optimizer': optimizer, 'learning_rate': learning_rate}
