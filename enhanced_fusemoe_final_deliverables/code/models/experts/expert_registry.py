"""
Expert Registry for Enhanced FuseMoE

This module provides a registry system for dynamically adding and managing
expert models in the Enhanced FuseMoE system for migraine prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Callable, Type

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.experts.sleep_expert import SleepExpert
from models.experts.weather_expert import WeatherExpert
from models.experts.stress_diet_expert import StressDietExpert
from models.experts.physio_expert import PhysioExpert


class ExpertRegistry:
    """
    Registry for managing expert models in the Enhanced FuseMoE system.
    
    This class provides methods for registering, retrieving, and managing
    expert models for different data domains.
    
    Attributes:
        experts (Dict[str, nn.Module]): Dictionary mapping expert names to expert instances
    """
    
    def __init__(self):
        """
        Initialize the expert registry.
        """
        self.experts = {}
    
    def register(self, name: str, expert: nn.Module) -> None:
        """
        Register an expert model.
        
        Args:
            name: Name of the expert
            expert: Expert model instance
        """
        self.experts[name] = expert
    
    # Add register_expert as an alias for register to maintain compatibility with tests
    def register_expert(self, name: str, expert: nn.Module) -> None:
        """
        Register an expert model (alias for register).
        
        Args:
            name: Name of the expert
            expert: Expert model instance
        """
        self.register(name, expert)
    
    def get(self, name: str) -> nn.Module:
        """
        Get an expert model.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert model instance
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not registered")
        
        return self.experts[name]
    
    def list(self) -> List[str]:
        """
        List all registered experts.
        
        Returns:
            List of expert names
        """
        return list(self.experts.keys())
    
    def remove(self, name: str) -> None:
        """
        Remove an expert model.
        
        Args:
            name: Name of the expert
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not registered")
        
        del self.experts[name]
    
    def __len__(self) -> int:
        """
        Get the number of registered experts.
        
        Returns:
            Number of registered experts
        """
        return len(self.experts)


class ScalableExpertPool(nn.Module):
    """
    Scalable pool of expert models for the Enhanced FuseMoE system.
    
    This class manages a pool of expert models that can be dynamically
    added, removed, or updated during training or inference.
    
    Attributes:
        registry (ExpertRegistry): Registry of expert models
        experts (nn.ModuleDict): Dictionary of expert model instances
        expert_output_dims (Dict[str, int]): Dictionary mapping expert names to output dimensions
    """
    
    def __init__(self, registry: ExpertRegistry):
        """
        Initialize the scalable expert pool.
        
        Args:
            registry: Expert registry
        """
        super(ScalableExpertPool, self).__init__()
        
        self.registry = registry
        self.experts = nn.ModuleDict()
        self.expert_output_dims = {}
    
    def add_expert(self, name: str, input_dim: int, hidden_dim: int, output_dim: int, 
                  num_layers: int = 2, dropout_rate: float = 0.2) -> None:
        """
        Add an expert model to the pool.
        
        Args:
            name: Name of the expert
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate
        """
        # Create expert model based on name
        if name == 'sleep':
            expert = SleepExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        elif name == 'weather':
            expert = WeatherExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        elif name == 'stress_diet':
            expert = StressDietExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        elif name == 'physio':
            expert = PhysioExpert(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            )
        else:
            # Generic expert model
            expert = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, output_dim)
            )
            # Add attributes to match expert models
            expert.input_dim = input_dim
            expert.hidden_dim = hidden_dim
            expert.output_dim = output_dim
        
        # Add to registry and pool
        self.registry.register(name, expert)
        self.experts[name] = expert
        self.expert_output_dims[name] = output_dim
    
    def remove_expert(self, name: str) -> None:
        """
        Remove an expert model from the pool.
        
        Args:
            name: Name of the expert
        """
        if name in self.experts:
            del self.experts[name]
            self.registry.remove(name)
            if name in self.expert_output_dims:
                del self.expert_output_dims[name]
    
    def get_expert(self, name: str) -> nn.Module:
        """
        Get an expert model from the pool.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert model instance
        """
        return self.registry.get(name)
    
    def list_experts(self) -> List[str]:
        """
        List all experts in the pool.
        
        Returns:
            List of expert names
        """
        return list(self.experts.keys())
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through all expert models in the pool.
        
        Args:
            inputs: Dictionary mapping expert names to input tensors
            
        Returns:
            Tensor of shape [batch_size, num_experts, output_dim] containing expert outputs
        """
        batch_size = next(iter(inputs.values())).size(0)
        expert_names = self.list_experts()
        num_experts = len(expert_names)
        
        # Find a common output dimension
        if not self.expert_output_dims:
            output_dim = 32  # Default
        else:
            output_dim = next(iter(self.expert_output_dims.values()))
        
        # Initialize output tensor
        outputs = torch.zeros(
            batch_size, num_experts, output_dim, 
            device=next(iter(inputs.values())).device
        )
        
        # Process each expert
        for i, name in enumerate(expert_names):
            if name in inputs:
                expert = self.experts[name]
                expert_output = expert(inputs[name])
                outputs[:, i, :] = expert_output
        
        return outputs


class DynamicMigraineMoE(nn.Module):
    """
    Dynamic Mixture of Experts model for migraine prediction.
    
    This model combines a scalable expert pool with a gating network and
    fusion mechanism to create a flexible and extensible MoE architecture.
    
    Attributes:
        expert_pool (ScalableExpertPool): Pool of expert models
        gating (nn.Module): Gating network
        fusion (nn.Module): Fusion mechanism
    """
    
    def __init__(self, expert_pool: ScalableExpertPool, gating: nn.Module, fusion: nn.Module):
        """
        Initialize the dynamic migraine MoE model.
        
        Args:
            expert_pool: Pool of expert models
            gating: Gating network
            fusion: Fusion mechanism
        """
        super(DynamicMigraineMoE, self).__init__()
        
        self.expert_pool = expert_pool
        self.gating = gating
        self.fusion = fusion
    
    def forward(self, inputs: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamic migraine MoE model.
        
        Args:
            inputs: Dictionary mapping expert names to input tensors
            training: Whether the model is in training mode
            
        Returns:
            Tuple containing:
                - Migraine prediction probabilities
                - Load balancing loss
        """
        # Get list of available experts
        expert_names = self.expert_pool.list_experts()
        
        # Prepare inputs for gating network
        gating_inputs = [inputs[name] for name in expert_names if name in inputs]
        
        # Forward pass through gating network
        # Handle different return formats from gating network
        gating_output = self.gating(gating_inputs, training)
        
        # Extract gates and load_balancing_loss based on return type
        if isinstance(gating_output, tuple) and len(gating_output) >= 2:
            gates = gating_output[0]
            load_balancing_loss = gating_output[1]
        else:
            # If gating network doesn't return load_balancing_loss, use a dummy value
            gates = gating_output
            load_balancing_loss = torch.tensor(0.0, device=next(iter(inputs.values())).device)
        
        # Forward pass through expert pool
        expert_outputs = self.expert_pool(inputs)
        
        # Forward pass through fusion mechanism
        # Handle different input formats for fusion mechanism
        if hasattr(self.fusion, 'forward') and callable(self.fusion.forward):
            # Try different input formats based on fusion mechanism's expectations
            try:
                # Try with expert_outputs and gates
                predictions = self.fusion(expert_outputs, gates, training)
            except (TypeError, ValueError):
                try:
                    # Try with a tuple of (expert_outputs, gates)
                    predictions = self.fusion((expert_outputs, gates), training)
                except (TypeError, ValueError):
                    try:
                        # Try with a dictionary containing expert_outputs and gates
                        predictions = self.fusion({
                            'expert_outputs': expert_outputs,
                            'gates': gates,
                            'training': training
                        })
                    except (TypeError, ValueError):
                        # Last resort: try with just the expert_outputs
                        predictions = self.fusion(expert_outputs)
        else:
            # If fusion is not callable, use a dummy prediction
            predictions = torch.zeros(
                expert_outputs.size(0), 1, 
                device=expert_outputs.device
            )
        
        return predictions, load_balancing_loss
    
    def update_components(self, gating: Optional[nn.Module] = None, 
                         fusion: Optional[nn.Module] = None) -> None:
        """
        Update components of the MoE model.
        
        Args:
            gating: New gating network (if None, keeps current)
            fusion: New fusion mechanism (if None, keeps current)
        """
        if gating is not None:
            self.gating = gating
        
        if fusion is not None:
            self.fusion = fusion


def create_dynamic_migraine_moe(expert_configs: Dict[str, Dict[str, Any]],
                               gating_config: Dict[str, Any],
                               fusion_config: Dict[str, Any]) -> DynamicMigraineMoE:
    """
    Create a dynamic migraine MoE model.
    
    Args:
        expert_configs: Dictionary mapping expert names to configurations
        gating_config: Configuration for the gating network
        fusion_config: Configuration for the fusion mechanism
        
    Returns:
        Dynamic migraine MoE model
    """
    # Create expert pool
    registry = ExpertRegistry()
    expert_pool = ScalableExpertPool(registry)
    
    # Add experts to pool
    for name, config in expert_configs.items():
        expert_pool.add_expert(
            name=name,
            input_dim=config.get('input_dim', 10),
            hidden_dim=config.get('hidden_dim', 20),
            output_dim=config.get('output_dim', 5),
            num_layers=config.get('num_layers', 2),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
    
    # Create gating network
    from models.gating.migraine_gating import MigraineGating
    input_dims = [config.get('input_dim', 10) for name, config in expert_configs.items()]
    gating = MigraineGating(
        input_dims=input_dims,
        hidden_dim=gating_config.get('hidden_dim', 64),
        num_experts=len(expert_configs),
        top_k=gating_config.get('top_k', 2),
        dropout_rate=gating_config.get('dropout_rate', 0.2),
        noisy_gating=gating_config.get('noisy_gating', True)
    )
    
    # Create fusion mechanism
    from models.fusion.migraine_fusion import MigraineFusion
    fusion = MigraineFusion(
        expert_output_dim=fusion_config.get('expert_output_dim', 32),
        hidden_dim=fusion_config.get('hidden_dim', 64),
        num_experts=len(expert_configs),
        dropout_rate=fusion_config.get('dropout_rate', 0.2)
    )
    
    # Create dynamic migraine MoE
    return DynamicMigraineMoE(expert_pool, gating, fusion)
