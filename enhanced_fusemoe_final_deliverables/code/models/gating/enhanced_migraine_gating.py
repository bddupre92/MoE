"""
Enhanced Migraine Gating Network for FuseMoE

This module provides an enhanced implementation of the migraine gating network
for the FuseMoE system with improved routing mechanisms and better load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax activation for more decisive expert selection.
    
    This implements the Gumbel-Softmax trick which allows for discrete sampling
    with straight-through gradients during training.
    
    Attributes:
        temperature (float): Temperature parameter for controlling discreteness
        hard (bool): Whether to use hard sampling
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = False):
        """
        Initialize Gumbel-Softmax activation.
        
        Args:
            temperature: Temperature parameter for controlling discreteness
            hard: Whether to use hard sampling
        """
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard
        self.eps = 1e-10
    
    def forward(self, logits: torch.Tensor, tau: float = None) -> torch.Tensor:
        """
        Forward pass through Gumbel-Softmax activation.
        
        Args:
            logits: Input logits of shape [batch_size, num_classes]
            tau: Temperature parameter (overrides self.temperature if provided)
            
        Returns:
            Gumbel-Softmax samples of shape [batch_size, num_classes]
        """
        if tau is None:
            tau = self.temperature
        
        # Sample from Gumbel distribution
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        gumbels = (logits + gumbels) / tau
        
        # Apply softmax
        y_soft = F.softmax(gumbels, dim=-1)
        
        # Straight-through if hard is True
        if self.hard:
            # One-hot encoding
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
            
            # Straight-through estimator
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        return ret


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for the gating network.
    
    This module first decides domain importance, then expert selection within each domain.
    
    Attributes:
        domain_attention: Attention mechanism for domain importance
        expert_attention: Attention mechanism for expert selection
    """
    
    def __init__(self, hidden_dim: int, num_domains: int, num_experts: int):
        """
        Initialize hierarchical attention mechanism.
        
        Args:
            hidden_dim: Hidden dimension
            num_domains: Number of domains
            num_experts: Number of experts
        """
        super(HierarchicalAttention, self).__init__()
        
        # Domain attention
        self.domain_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_domains)
        )
        
        # Expert attention (one per domain)
        self.expert_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, num_experts // num_domains)
            ) for _ in range(num_domains)
        ])
        
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.num_experts = num_experts
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Tuple of (domain_weights, expert_weights)
                domain_weights: Tensor of shape [batch_size, num_domains]
                expert_weights: Tensor of shape [batch_size, num_experts]
        """
        # Calculate domain attention
        domain_logits = self.domain_attention(x)
        domain_weights = F.softmax(domain_logits, dim=1)
        
        # Calculate expert attention for each domain
        expert_logits_list = []
        for i in range(self.num_domains):
            domain_expert_logits = self.expert_attention[i](x)
            expert_logits_list.append(domain_expert_logits)
        
        # Combine expert logits
        expert_logits = torch.cat(expert_logits_list, dim=1)
        expert_weights = F.softmax(expert_logits, dim=1)
        
        # Scale expert weights by domain weights
        experts_per_domain = self.num_experts // self.num_domains
        for i in range(self.num_domains):
            start_idx = i * experts_per_domain
            end_idx = start_idx + experts_per_domain
            expert_weights[:, start_idx:end_idx] *= domain_weights[:, i].unsqueeze(1)
        
        return domain_weights, expert_weights


class EnhancedMigraineGating(nn.Module):
    """
    Enhanced gating network for migraine prediction in the FuseMoE system.
    
    This model uses a more sophisticated routing mechanism with Gumbel-Softmax
    for more decisive expert selection and hierarchical attention for better
    domain-specific routing.
    
    Attributes:
        input_dims (List[int]): List of input dimensions for each modality
        hidden_dim (int): Size of hidden layers
        num_experts (int): Number of expert models
        top_k (int): Number of experts to route to
        dropout_rate (float): Dropout probability
        temperature (float): Temperature for Gumbel-Softmax
        use_hierarchical (bool): Whether to use hierarchical attention
    """
    
    def __init__(self, input_dims: List[int] = None, input_dim: int = None, hidden_dim: int = 128, num_experts: int = 4,
                top_k: int = 3, dropout_rate: float = 0.2, temperature: float = 0.5, use_hierarchical: bool = True):
        """
        Initialize the enhanced migraine gating network.
        
        Args:
            input_dims: List of input dimensions for each modality
            input_dim: Single input dimension (for backward compatibility)
            hidden_dim: Size of hidden layers
            num_experts: Number of expert models
            top_k: Number of experts to route to
            dropout_rate: Dropout probability
            temperature: Temperature for Gumbel-Softmax
            use_hierarchical: Whether to use hierarchical attention
        """
        super(EnhancedMigraineGating, self).__init__()
        
        # Handle backward compatibility with input_dim parameter
        if input_dims is None and input_dim is not None:
            input_dims = [input_dim]
        elif input_dims is None:
            raise ValueError("Either input_dims or input_dim must be provided")
        
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self.use_hierarchical = use_hierarchical
        
        # Process each input modality with more sophisticated encoders
        self.modality_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for dim in input_dims
        ])
        
        # Multi-head cross-modality attention
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        assert self.head_dim * self.num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Combine processed inputs with residual connections
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Gating mechanism
        if use_hierarchical:
            # Assume 4 domains (sleep, weather, stress/diet, physiological)
            self.num_domains = 4
            self.hierarchical_attention = HierarchicalAttention(
                hidden_dim, self.num_domains, num_experts
            )
        else:
            self.gate_proj = nn.Linear(hidden_dim, num_experts)
        
        # Gumbel-Softmax for more decisive expert selection
        self.gumbel_softmax = GumbelSoftmax(temperature=temperature, hard=False)
        
        # Dynamic temperature adjustment
        self.temperature_proj = nn.Linear(hidden_dim, 1)
        
        # Load balancing loss coefficient
        self.load_balance_coef = 0.05  # Increased from 0.01 to encourage more balanced expert utilization
    
    def _multi_head_attention(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            queries: Query tensor of shape [batch_size, seq_len_q, hidden_dim]
            keys: Key tensor of shape [batch_size, seq_len_k, hidden_dim]
            values: Value tensor of shape [batch_size, seq_len_k, hidden_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len_q, hidden_dim]
        """
        batch_size = queries.size(0)
        
        # Project inputs to queries, keys, and values
        q = self.query_proj(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, v)
        
        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Project output
        output = self.output_proj(output)
        
        return output
    
    def forward(self, inputs: List[torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the enhanced migraine gating network.
        
        Args:
            inputs: List of input tensors for each modality
            training: Whether the model is in training mode
            
        Returns:
            Tuple of (gates, load, importance)
                gates: Tensor of shape [batch_size, num_experts] containing routing weights
                load: Tensor of shape [num_experts] containing the fraction of routing sent to each expert
                importance: Tensor of shape [batch_size, num_experts] containing raw routing probabilities
        """
        batch_size = inputs[0].size(0)
        
        # Process each modality
        processed_inputs = []
        for i, processor in enumerate(self.modality_processors):
            processed = processor(inputs[i])
            processed_inputs.append(processed)
        
        # Apply multi-head cross-modality attention
        if len(processed_inputs) > 1:
            attended_inputs = []
            for i, input_i in enumerate(processed_inputs):
                queries = input_i.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                # Collect keys and values from all modalities
                keys = torch.stack([input_j for j, input_j in enumerate(processed_inputs)], dim=1)
                values = keys.clone()
                
                # Apply multi-head attention
                attended = self._multi_head_attention(queries, keys, values).squeeze(1)
                
                # Combine with original input using residual connection
                attended_inputs.append(input_i + attended)
            
            processed_inputs = attended_inputs
        
        # Combine all modalities
        combined = torch.cat(processed_inputs, dim=1)  # [batch_size, hidden_dim * num_modalities]
        combined = self.combiner(combined)  # [batch_size, hidden_dim]
        
        # Dynamic temperature adjustment
        temperature = F.softplus(self.temperature_proj(combined)) + 0.1
        
        # Generate gating logits
        if self.use_hierarchical:
            # Use hierarchical attention
            domain_weights, importance = self.hierarchical_attention(combined)
        else:
            # Use standard gating
            logits = self.gate_proj(combined)
            importance = F.softmax(logits, dim=1)
        
        # Apply Gumbel-Softmax for more decisive expert selection
        if training:
            if self.use_hierarchical:
                # Convert importance to logits for Gumbel-Softmax
                eps = 1e-10
                logits = torch.log(importance + eps)
            
            # Apply Gumbel-Softmax with dynamic temperature
            gumbel_weights = self.gumbel_softmax(logits, tau=temperature.squeeze())
            
            # Get top-k experts
            top_values, top_indices = gumbel_weights.topk(min(self.top_k, self.num_experts), dim=1)
            
            # Create dispatch tensor
            zeros = torch.zeros_like(gumbel_weights, requires_grad=True)
            gates = zeros.scatter(1, top_indices, top_values)
        else:
            # During inference, directly use top-k from importance
            top_values, top_indices = importance.topk(min(self.top_k, self.num_experts), dim=1)
            
            # Create dispatch tensor
            zeros = torch.zeros_like(importance, requires_grad=True)
            gates = zeros.scatter(1, top_indices, top_values)
        
        # Ensure gates sum to top_k for each sample
        gates = gates * self.top_k / gates.sum(dim=1, keepdim=True)
        
        # Calculate load (fraction of routing sent to each expert)
        load = gates.sum(0) / gates.sum()
        
        return gates, load, importance
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the enhanced migraine gating network.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'input_dims': self.input_dims,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature,
            'use_hierarchical': self.use_hierarchical,
            'load_balance_coef': self.load_balance_coef
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'EnhancedMigraineGating':
        """
        Create an enhanced migraine gating network from configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Initialized enhanced migraine gating network
        """
        gating = cls(
            input_dims=config.get('input_dims'),
            hidden_dim=config.get('hidden_dim', 128),
            num_experts=config.get('num_experts', 4),
            top_k=config.get('top_k', 3),
            dropout_rate=config.get('dropout_rate', 0.2),
            temperature=config.get('temperature', 0.5),
            use_hierarchical=config.get('use_hierarchical', True)
        )
        gating.load_balance_coef = config.get('load_balance_coef', 0.05)
        return gating


def create_enhanced_migraine_gating(input_dims: List[int] = None, input_dim: int = None, 
                                  hidden_dim: int = 128, num_experts: int = 4,
                                  top_k: int = 3, dropout_rate: float = 0.2, 
                                  temperature: float = 0.5, use_hierarchical: bool = True,
                                  load_balance_coef: float = 0.05) -> EnhancedMigraineGating:
    """
    Create an enhanced migraine gating network.
    
    Args:
        input_dims: List of input dimensions for each modality
        input_dim: Single input dimension (for backward compatibility)
        hidden_dim: Size of hidden layers
        num_experts: Number of expert models
        top_k: Number of experts to route to
        dropout_rate: Dropout probability
        temperature: Temperature for Gumbel-Softmax
        use_hierarchical: Whether to use hierarchical attention
        load_balance_coef: Coefficient for load balancing loss
        
    Returns:
        Initialized enhanced migraine gating network
    """
    # Handle backward compatibility with input_dim parameter
    if input_dims is None and input_dim is not None:
        input_dims = [input_dim]
    elif input_dims is None:
        raise ValueError("Either input_dims or input_dim must be provided")
    
    model = EnhancedMigraineGating(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        dropout_rate=dropout_rate,
        temperature=temperature,
        use_hierarchical=use_hierarchical
    )
    model.load_balance_coef = load_balance_coef
    
    return model


# For backward compatibility with existing FuseMoE architecture
class MigraineGating(nn.Module):
    """
    Backward-compatible wrapper for the enhanced migraine gating network.
    
    This class maintains the same interface as the original MigraineGating class
    but uses the enhanced implementation internally.
    """
    
    def __init__(self, input_dims: List[int] = None, input_dim: int = None, hidden_dim: int = 64, num_experts: int = 4,
                top_k: int = 2, dropout_rate: float = 0.2, noisy_gating: bool = True):
        """
        Initialize the migraine gating network.
        
        Args:
            input_dims: List of input dimensions for each modality
            input_dim: Single input dimension (for backward compatibility)
            hidden_dim: Size of hidden layers
            num_experts: Number of expert models
            top_k: Number of experts to route to
            dropout_rate: Dropout probability
            noisy_gating: Whether to use noisy gating (ignored, always uses Gumbel-Softmax)
        """
        super(MigraineGating, self).__init__()
        
        # Create enhanced gating network
        self.enhanced_gating = EnhancedMigraineGating(
            input_dims=input_dims,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout_rate=dropout_rate,
            temperature=0.5,  # Default temperature
            use_hierarchical=True  # Use hierarchical attention by default
        )
        
        # Store parameters for config
        self.input_dims = input_dims if input_dims is not None else [input_dim]
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_rate = dropout_rate
        self.noisy_gating = noisy_gating
        self.load_balance_coef = self.enhanced_gating.load_balance_coef
    
    def forward(self, inputs: List[torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the migraine gating network.
        
        Args:
            inputs: List of input tensors for each modality
            training: Whether the model is in training mode
            
        Returns:
            Tuple of (gates, load, importance)
                gates: Tensor of shape [batch_size, num_experts] containing routing weights
                load: Tensor of shape [num_experts] containing the fraction of routing sent to each expert
                importance: Tensor of shape [batch_size, num_experts] containing raw routing probabilities
        """
        return self.enhanced_gating(inputs, training)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the migraine gating network.
        
        Returns:
            Dictionary containing model configuration
        """
        return {
            'input_dims': self.input_dims,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'dropout_rate': self.dropout_rate,
            'noisy_gating': self.noisy_gating,
            'load_balance_coef': self.load_balance_coef
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MigraineGating':
        """
        Create a migraine gating network from configuration.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            Initialized migraine gating network
        """
        gating = cls(
            input_dims=config.get('input_dims'),
            hidden_dim=config.get('hidden_dim', 64),
            num_experts=config.get('num_experts', 4),
            top_k=config.get('top_k', 2),
            dropout_rate=config.get('dropout_rate', 0.2),
            noisy_gating=config.get('noisy_gating', True)
        )
        gating.load_balance_coef = config.get('load_balance_coef', 0.01)
        return gating


def create_migraine_gating(input_dims: List[int] = None, input_dim: int = None, 
                          hidden_dim: int = 64, num_experts: int = 4,
                          top_k: int = 2, dropout_rate: float = 0.2, 
                          noisy_gating: bool = True,
                          load_balance_coef: float = 0.01, 
                          learning_rate: float = 0.001) -> MigraineGating:
    """
    Create a migraine gating network.
    
    Args:
        input_dims: List of input dimensions for each modality
        input_dim: Single input dimension (for backward compatibility)
        hidden_dim: Size of hidden layers
        num_experts: Number of expert models
        top_k: Number of experts to route to
        dropout_rate: Dropout probability
        noisy_gating: Whether to use noisy gating
        load_balance_coef: Coefficient for load balancing loss
        learning_rate: Learning rate for optimizer
        
    Returns:
        Initialized migraine gating network
    """
    # Handle backward compatibility with input_dim parameter
    if input_dims is None and input_dim is not None:
        input_dims = [input_dim]
    elif input_dims is None:
        raise ValueError("Either input_dims or input_dim must be provided")
    
    model = MigraineGating(
        input_dims=input_dims,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        dropout_rate=dropout_rate,
        noisy_gating=noisy_gating
    )
    model.load_balance_coef = load_balance_coef
    
    return model
