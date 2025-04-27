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
        experts (Dict[str, Type[nn.Module]]): Dictionary mapping expert names to expert classes
        expert_configs (Dict[str, Dict[str, Any]]): Dictionary mapping expert names to configurations
        expert_factories (Dict[str, Callable]): Dictionary mapping expert names to factory functions
    """
    
    def __init__(self):
        """
        Initialize the expert registry.
        """
        self.experts = {}
        self.expert_configs = {}
        self.expert_factories = {}
        
        # Register default experts
        self.register_expert('sleep', SleepExpert)
        self.register_expert('weather', WeatherExpert)
        self.register_expert('stress_diet', StressDietExpert)
        self.register_expert('physio', PhysioExpert)
    
    def register_expert(self, name: str, expert_class: Type[nn.Module], 
                       config: Optional[Dict[str, Any]] = None,
                       factory_fn: Optional[Callable] = None) -> None:
        """
        Register an expert model.
        
        Args:
            name: Name of the expert
            expert_class: Expert model class
            config: Default configuration for the expert
            factory_fn: Factory function for creating expert instances
        """
        self.experts[name] = expert_class
        self.expert_configs[name] = config or {}
        
        if factory_fn is None:
            # Default factory function
            def default_factory(**kwargs):
                merged_config = {**self.expert_configs[name], **kwargs}
                return expert_class(**merged_config)
            
            self.expert_factories[name] = default_factory
        else:
            self.expert_factories[name] = factory_fn
    
    def get_expert_class(self, name: str) -> Type[nn.Module]:
        """
        Get an expert model class.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert model class
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not registered")
        
        return self.experts[name]
    
    def get_expert_config(self, name: str) -> Dict[str, Any]:
        """
        Get the default configuration for an expert model.
        
        Args:
            name: Name of the expert
            
        Returns:
            Default configuration for the expert
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.expert_configs:
            raise KeyError(f"Expert '{name}' is not registered")
        
        return self.expert_configs[name].copy()
    
    def create_expert(self, name: str, **kwargs) -> nn.Module:
        """
        Create an instance of an expert model.
        
        Args:
            name: Name of the expert
            **kwargs: Additional configuration parameters
            
        Returns:
            Expert model instance
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.expert_factories:
            raise KeyError(f"Expert '{name}' is not registered")
        
        return self.expert_factories[name](**kwargs)
    
    def list_experts(self) -> List[str]:
        """
        List all registered experts.
        
        Returns:
            List of expert names
        """
        return list(self.experts.keys())
    
    def unregister_expert(self, name: str) -> None:
        """
        Unregister an expert model.
        
        Args:
            name: Name of the expert
            
        Raises:
            KeyError: If expert is not registered
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not registered")
        
        del self.experts[name]
        del self.expert_configs[name]
        del self.expert_factories[name]


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
    
    def __init__(self, registry: Optional[ExpertRegistry] = None):
        """
        Initialize the scalable expert pool.
        
        Args:
            registry: Expert registry (creates a new one if None)
        """
        super(ScalableExpertPool, self).__init__()
        
        self.registry = registry or ExpertRegistry()
        self.experts = nn.ModuleDict()
        self.expert_output_dims = {}
    
    def add_expert(self, name: str, **kwargs) -> None:
        """
        Add an expert model to the pool.
        
        Args:
            name: Name of the expert
            **kwargs: Configuration parameters for the expert
            
        Raises:
            KeyError: If expert is already in the pool
        """
        if name in self.experts:
            raise KeyError(f"Expert '{name}' is already in the pool")
        
        expert = self.registry.create_expert(name, **kwargs)
        self.experts[name] = expert
        
        # Store output dimension
        if hasattr(expert, 'output_dim'):
            self.expert_output_dims[name] = expert.output_dim
        else:
            # Default output dimension
            self.expert_output_dims[name] = 32
    
    def remove_expert(self, name: str) -> None:
        """
        Remove an expert model from the pool.
        
        Args:
            name: Name of the expert
            
        Raises:
            KeyError: If expert is not in the pool
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not in the pool")
        
        del self.experts[name]
        del self.expert_output_dims[name]
    
    def get_expert(self, name: str) -> nn.Module:
        """
        Get an expert model from the pool.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert model instance
            
        Raises:
            KeyError: If expert is not in the pool
        """
        if name not in self.experts:
            raise KeyError(f"Expert '{name}' is not in the pool")
        
        return self.experts[name]
    
    def list_experts(self) -> List[str]:
        """
        List all experts in the pool.
        
        Returns:
            List of expert names
        """
        return list(self.experts.keys())
    
    def get_expert_output_dim(self, name: str) -> int:
        """
        Get the output dimension of an expert model.
        
        Args:
            name: Name of the expert
            
        Returns:
            Output dimension of the expert
            
        Raises:
            KeyError: If expert is not in the pool
        """
        if name not in self.expert_output_dims:
            raise KeyError(f"Expert '{name}' is not in the pool")
        
        return self.expert_output_dims[name]
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all expert models in the pool.
        
        Args:
            inputs: Dictionary mapping expert names to input tensors
            
        Returns:
            Dictionary mapping expert names to output tensors
        """
        outputs = {}
        
        for name, expert in self.experts.items():
            if name in inputs:
                outputs[name] = expert(inputs[name])
        
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
    
    def forward(self, inputs: Dict[str, torch.Tensor], training: bool = True) -> torch.Tensor:
        """
        Forward pass through the dynamic migraine MoE model.
        
        Args:
            inputs: Dictionary mapping expert names to input tensors
            training: Whether the model is in training mode
            
        Returns:
            Migraine prediction probabilities
        """
        # Get list of available experts
        expert_names = self.expert_pool.list_experts()
        
        # Prepare inputs for gating network
        gating_inputs = [inputs[name] for name in expert_names if name in inputs]
        
        # Forward pass through gating network
        gates, indices, load_balancing_loss = self.gating(gating_inputs, training)
        
        # Forward pass through expert pool
        expert_outputs = self.expert_pool(inputs)
        
        # Prepare expert outputs for fusion
        fusion_inputs = [expert_outputs[name] for name in expert_names if name in expert_outputs]
        
        # Forward pass through fusion mechanism
        predictions = self.fusion(fusion_inputs, gates)
        
        if training:
            return predictions, load_balancing_loss
        else:
            return predictions
    
    def add_expert(self, name: str, **kwargs) -> None:
        """
        Add an expert model to the MoE.
        
        Args:
            name: Name of the expert
            **kwargs: Configuration parameters for the expert
        """
        self.expert_pool.add_expert(name, **kwargs)
    
    def remove_expert(self, name: str) -> None:
        """
        Remove an expert model from the MoE.
        
        Args:
            name: Name of the expert
        """
        self.expert_pool.remove_expert(name)
    
    def list_experts(self) -> List[str]:
        """
        List all experts in the MoE.
        
        Returns:
            List of expert names
        """
        return self.expert_pool.list_experts()
    
    def get_expert(self, name: str) -> nn.Module:
        """
        Get an expert model from the MoE.
        
        Args:
            name: Name of the expert
            
        Returns:
            Expert model instance
        """
        return self.expert_pool.get_expert(name)


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
        expert_pool.add_expert(name, **config)
    
    # Create gating network
    from models.gating.migraine_gating import MigraineGating
    input_dims = [expert_pool.get_expert_output_dim(name) for name in expert_pool.list_experts()]
    gating = MigraineGating(
        input_dims=input_dims,
        hidden_dim=gating_config.get('hidden_dim', 64),
        num_experts=len(expert_pool.list_experts()),
        top_k=gating_config.get('top_k', 2),
        dropout_rate=gating_config.get('dropout_rate', 0.2),
        noisy_gating=gating_config.get('noisy_gating', True)
    )
    
    # Create fusion mechanism
    from models.fusion.migraine_fusion import MigraineFusion
    fusion = MigraineFusion(
        expert_output_dim=fusion_config.get('expert_output_dim', 32),
        hidden_dim=fusion_config.get('hidden_dim', 64),
        num_experts=len(expert_pool.list_experts()),
        dropout_rate=fusion_config.get('dropout_rate', 0.2)
    )
    
    # Create dynamic migraine MoE
    return DynamicMigraineMoE(expert_pool, gating, fusion)
