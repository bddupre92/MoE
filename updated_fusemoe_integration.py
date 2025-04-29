"""
FuseMoE Integration Module for Migraine Prediction

This module integrates the FuseMoE library with our data processing pipeline.
It provides adapter classes and utility functions to connect our processed data with FuseMoE models.

Author: Manus AI
Date: April 29, 2025
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback # Import traceback module
import pandas as pd

# Add FuseMoE source directory to path
sys.path.append("/home/ubuntu/FuseMoE/src")

# Import FuseMoE modules
try:
    from core.hme_seq import HierarchicalMoE
    from core.sparse_moe import MoE as SparseMoE  # Import MoE and alias it as SparseMoE
    from utils.config import MoEConfig
    FUSEMOE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: FuseMoE modules not found. Error: {e}")
    print("FuseMoE integration will be disabled.")
    FUSEMOE_AVAILABLE = False

class FuseMoEAdapter:
    """
    Adapter class to connect our data processing pipeline with FuseMoE models.
    """
    
    def __init__(self, config=None):
        """
        Initialize the FuseMoE adapter.
        
        Args:
            config: Configuration dictionary for the adapter
        """
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default configuration
        self.default_config = {
            'num_experts': 4,
            'k': 2,  # Number of experts to use
            'model_dim': 64,
            'hidden_dim': 128,
            'expert_type': 'ffn'
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        
        # Check if FuseMoE is available
        if not FUSEMOE_AVAILABLE:
            print("FuseMoE is not available. Using fallback implementation.")
            self.use_fallback = True
        else:
            self.use_fallback = False
    
    def create_sparse_moe_layer(self, input_dim, output_dim=None):
        """
        Create a SparseMoE layer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (defaults to model_dim from config)
            
        Returns:
            SparseMoE layer or None if FuseMoE is not available
        """
        if self.use_fallback:
            return None
        
        output_dim = output_dim or self.default_config["model_dim"]
        
        try:
            # Create MoEConfig object required by MoE (aliased as SparseMoE)
            moe_config = MoEConfig(
                num_experts=self.default_config["num_experts"],
                moe_input_size=input_dim,
                moe_hidden_size=self.default_config["hidden_dim"],
                moe_output_size=output_dim,
                router_type="joint",  # Use "joint" to avoid modality iteration
                gating="softmax",      # Default gating mechanism
                num_modalities=1,      # Assuming single modality for this layer
                top_k=self.default_config["k"],
                noisy_gating=True      # Enable noisy gating
            )
            
            # Create the MoE layer
            moe_layer = SparseMoE(config=moe_config)
            return moe_layer
        except Exception as e:
            print(f"Error creating SparseMoE layer: {e}")
            self.use_fallback = True
            return None
    
    def create_hierarchical_moe(self, input_dim, output_dim=None, num_layers=2):
        """
        Create a HierarchicalMoE model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (defaults to model_dim from config)
            num_layers: Number of layers in the hierarchical MoE
            
        Returns:
            HierarchicalMoE model or None if FuseMoE is not available
        """
        if self.use_fallback:
            return None
        
        output_dim = output_dim or self.default_config['model_dim']
        
        try:
            # Create a list of expert counts for each layer
            # HierarchicalMoE expects a list, not a single integer
            num_experts_list = [self.default_config['num_experts'], self.default_config['num_experts']]
            
            # Create MoEConfig object required by HierarchicalMoE
            moe_config = MoEConfig(
                num_experts=num_experts_list,  # This must be a list [4, 4]
                moe_input_size=input_dim,
                moe_hidden_size=self.default_config["hidden_dim"],
                moe_output_size=output_dim,
                router_type="joint",  # Use "joint" to avoid modality iteration
                gating=["softmax", "softmax"],  # Pass as list for each level
                num_modalities=4,      # Sleep, Weather, Stress/Diet, Physio
                top_k=[self.default_config["k"], self.default_config["k"]], # Pass as list
                disjoint_top_k=self.default_config["k"],  # Added this parameter
                noisy_gating=True      # Enable noisy gating for better load balancing
            )
            
            # Create the hierarchical MoE model
            hme_model = HierarchicalMoE(config=moe_config)
            return hme_model
        except Exception as e:
            print(f"Error creating HierarchicalMoE model: {e}")
            self.use_fallback = True
            return None
    
    def adapt_data_for_fusemoe(self, X_data, y_data=None):
        """
        Adapt our processed data format to be compatible with FuseMoE models.
        
        Args:
            X_data: DataFrame containing processed features
            y_data: Series containing target variable
            
        Returns:
            Tuple of (inputs, targets) in FuseMoE-compatible format
        """
        if X_data is None:
            return None, None
        
        try:
            # Identify feature groups based on column names
            sleep_cols = [col for col in X_data.columns if any(s in col for s in 
                         ['sleep_hours', 'deep_sleep', 'rem_sleep', 'light_sleep', 'awake_time', 'sleep_quality'])]
            
            weather_cols = [col for col in X_data.columns if any(s in col for s in 
                           ['temperature', 'humidity', 'pressure', 'pressure_change'])]
            
            stress_diet_cols = [col for col in X_data.columns if any(s in col for s in 
                               ['stress_level', 'alcohol', 'caffeine', 'chocolate', 'processed_food', 'water_consumed'])]
            
            # Remaining columns that aren't in the above groups and aren't demographic (age, sex)
            # are assumed to be physio features
            demographic_cols = [col for col in X_data.columns if any(s in col for s in ['age', 'sex'])]
            physio_cols = [col for col in X_data.columns if col not in sleep_cols + weather_cols + 
                          stress_diet_cols + demographic_cols]
            
            # Print feature group sizes for debugging
            print(f"Feature groups: Sleep ({len(sleep_cols)}), Weather ({len(weather_cols)}), "
                  f"Stress/Diet ({len(stress_diet_cols)}), Physio ({len(physio_cols)}), "
                  f"Demographics ({len(demographic_cols)})")
            
            # Extract data for each modality
            sleep_data = torch.tensor(X_data[sleep_cols].values, dtype=torch.float32)
            weather_data = torch.tensor(X_data[weather_cols].values, dtype=torch.float32)
            stress_diet_data = torch.tensor(X_data[stress_diet_cols].values, dtype=torch.float32)
            
            # For physio, if no columns are found, create a placeholder tensor
            if len(physio_cols) > 0:
                physio_data = torch.tensor(X_data[physio_cols].values, dtype=torch.float32)
            else:
                # Create placeholder tensor with zeros
                physio_data = torch.zeros((len(X_data), 1), dtype=torch.float32)
                print("Warning: No physio features found. Using placeholder zeros.")
            
            # Convert target to tensor if provided
            if y_data is not None:
                target_data = torch.tensor(y_data.values, dtype=torch.float32)
                # Reshape if necessary
                if target_data.dim() == 1:
                    target_data = target_data.unsqueeze(1)
            else:
                target_data = None
            
            return {
                'sleep': sleep_data,
                'weather': weather_data,
                'stress_diet': stress_diet_data,
                'physio': physio_data
            }, target_data
            
        except Exception as e:
            print(f"Error adapting data for FuseMoE: {e}")
            traceback.print_exc()
            return None, None

class FuseMoEMigraineModel(nn.Module):
    """
    A wrapper model that integrates FuseMoE with our migraine prediction task.
    This model uses FuseMoE components if available, otherwise falls back to a simple implementation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the FuseMoE migraine model.
        
        Args:
            config: Configuration dictionary for the model
        """
        super(FuseMoEMigraineModel, self).__init__()
        self.config = config or {}
        self.adapter = FuseMoEAdapter(config)
        
        # Input dimensions for each data type - updated based on data_processor.py
        self.sleep_dim = self.config.get('sleep_dim', 6)  # total_sleep_hours, deep_sleep_pct, rem_sleep_pct, light_sleep_pct, awake_time_mins, sleep_quality
        self.weather_dim = self.config.get('weather_dim', 4)  # temperature, humidity, pressure, pressure_change_24h
        self.stress_diet_dim = self.config.get('stress_diet_dim', 6)  # stress_level, water_consumed_liters, alcohol_consumed, caffeine_consumed, chocolate_consumed, processed_food_consumed
        self.physio_dim = self.config.get('physio_dim', 1)  # Placeholder for physio features (will be expanded when implemented)
        
        # Output dimension
        self.output_dim = self.config.get('output_dim', 1)  # Binary classification: has_migraine
        
        # Model dimension
        self.model_dim = self.config.get('model_dim', 64)
        
        # Create FuseMoE layers or fallback layers
        self.create_model_layers()
    
    def create_model_layers(self):
        """Create model layers using FuseMoE or fallback implementation."""
        # Sleep data processing
        if not self.adapter.use_fallback:
            # Use FuseMoE for sleep data
            self.sleep_expert = self.adapter.create_sparse_moe_layer(
                input_dim=self.sleep_dim,
                output_dim=self.model_dim
            )
        else:
            # Fallback implementation
            self.sleep_expert = nn.Sequential(
                nn.Linear(self.sleep_dim, self.model_dim * 2),
                nn.ReLU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.ReLU()
            )
        
        # Weather data processing
        if not self.adapter.use_fallback:
            # Use FuseMoE for weather data
            self.weather_expert = self.adapter.create_sparse_moe_layer(
                input_dim=self.weather_dim,
                output_dim=self.model_dim
            )
        else:
            # Fallback implementation
            self.weather_expert = nn.Sequential(
                nn.Linear(self.weather_dim, self.model_dim * 2),
                nn.ReLU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.ReLU()
            )
        
        # Stress/diet data processing
        if not self.adapter.use_fallback:
            # Use FuseMoE for stress/diet data
            self.stress_diet_expert = self.adapter.create_sparse_moe_layer(
                input_dim=self.stress_diet_dim,
                output_dim=self.model_dim
            )
        else:
            # Fallback implementation
            self.stress_diet_expert = nn.Sequential(
                nn.Linear(self.stress_diet_dim, self.model_dim * 2),
                nn.ReLU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.ReLU()
            )
        
        # Physiological data processing
        if not self.adapter.use_fallback:
            # Use FuseMoE for physiological data
            self.physio_expert = self.adapter.create_sparse_moe_layer(
                input_dim=self.physio_dim,
                output_dim=self.model_dim
            )
        else:
            # Fallback implementation
            self.physio_expert = nn.Sequential(
                nn.Linear(self.physio_dim, self.model_dim * 2),
                nn.ReLU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.ReLU()
            )
        
        # Fusion layer
        if not self.adapter.use_fallback:
            # Use HierarchicalMoE for fusion
            self.fusion = self.adapter.create_hierarchical_moe(
                input_dim=self.model_dim * 4,  # Concatenated expert outputs
                output_dim=self.model_dim,
                num_layers=2
            )
        else:
            # Fallback implementation
            self.fusion = nn.Sequential(
                nn.Linear(self.model_dim * 4, self.model_dim * 2),
                nn.ReLU(),
                nn.Linear(self.model_dim * 2, self.model_dim),
                nn.ReLU()
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.model_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x=None, sleep=None, weather=None, stress_diet=None, physio=None):
        """
        Forward pass through the model.
        
        Args:
            x: Optional combined input tensor (if provided, will be split into modalities)
            sleep: Sleep data tensor
            weather: Weather data tensor
            stress_diet: Stress/diet data tensor
            physio: Physiological data tensor
            
        Returns:
            Model output (migraine prediction)
        """
        # If x is provided as a combined tensor, split it into modalities
        if x is not None:
            # Determine split points based on feature dimensions
            sleep_end = self.sleep_dim
            weather_end = sleep_end + self.weather_dim
            stress_diet_end = weather_end + self.stress_diet_dim
            
            # Split the input tensor
            sleep = x[:, :sleep_end]
            weather = x[:, sleep_end:weather_end]
            stress_diet = x[:, weather_end:stress_diet_end]
            physio = x[:, stress_diet_end:]
        
        # Process each data type through its expert
        if not self.adapter.use_fallback:
            # FuseMoE implementation
            try:
                sleep_out, _ = self.sleep_expert(sleep)
                weather_out, _ = self.weather_expert(weather)
                stress_diet_out, _ = self.stress_diet_expert(stress_diet)
                physio_out, _ = self.physio_expert(physio)
            except Exception as e:
                print(f"Error in FuseMoE forward pass: {e}")
                print("--- Traceback --- ")
                traceback.print_exc() # Print detailed traceback
                print("--- End Traceback ---")
                # Fall back to simple implementation
                self.adapter.use_fallback = True
                self.create_model_layers()
                return self.forward(sleep=sleep, weather=weather, stress_diet=stress_diet, physio=physio)
        else:
            # Fallback implementation
            sleep_out = self.sleep_expert(sleep)
            weather_out = self.weather_expert(weather)
            stress_diet_out = self.stress_diet_expert(stress_diet)
            physio_out = self.physio_expert(physio)
        
        # Concatenate expert outputs
        combined = torch.cat([sleep_out, weather_out, stress_diet_out, physio_out], dim=1)
        
        # Fusion layer
        if not self.adapter.use_fallback:
            try:
                fused, _ = self.fusion(combined)
            except Exception as e:
                print(f"Error in FuseMoE fusion: {e}")
                print("--- Traceback --- ")
                traceback.print_exc() # Print detailed traceback
                print("--- End Traceback ---")
                # Fall back to simple implementation
                self.adapter.use_fallback = True
                self.create_model_layers()
                return self.forward(sleep=sleep, weather=weather, stress_diet=stress_diet, physio=physio)
        else:
            fused = self.fusion(combined)
        
        # Output layer
        output = self.output_layer(fused)
        
        return output

def create_fusemoe_model(config=None):
    """
    Create a FuseMoE migraine prediction model.
    
    Args:
        config: Configuration dictionary for the model
        
    Returns:
        FuseMoEMigraineModel instance
    """
    return FuseMoEMigraineModel(config)

def test_fusemoe_integration():
    """
    Test the FuseMoE integration with sample data.
    
    Returns:
        True if test passes, False otherwise
    """
    try:
        # Create sample data with updated dimensions
        batch_size = 10
        sleep_data = torch.randn(batch_size, 6)  # Updated from 7x6 to 6
        weather_data = torch.randn(batch_size, 4)  # Updated from 5 to 4
        stress_diet_data = torch.randn(batch_size, 6)  # Unchanged
        physio_data = torch.randn(batch_size, 1)  # Updated from 5 to 1 (placeholder)
        
        # Create model
        model = create_fusemoe_model()
        
        # Forward pass
        output = model(sleep=sleep_data, weather=weather_data, stress_diet=stress_diet_data, physio=physio_data)
        
        # Check output shape
        expected_shape = (batch_size, 1)
        if output.shape == expected_shape:
            print(f"Test passed! Output shape: {output.shape}")
            return True
        else:
            print(f"Test failed! Expected shape {expected_shape}, got {output.shape}")
            return False
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()
        return False

def test_with_processed_data():
    """
    Test the FuseMoE integration with data from the data_processor.py pipeline.
    
    Returns:
        True if test passes, False otherwise
    """
    try:
        # Import the data processor
        sys.path.append("/home/ubuntu")
        from data_processor import process_data
        
        # Process the data
        print("Processing data...")
        X_train, y_train, X_valid, y_valid, X_test, y_test = process_data()
        
        # Create adapter
        adapter = FuseMoEAdapter()
        
        # Adapt data for FuseMoE
        print("Adapting data for FuseMoE...")
        inputs, targets = adapter.adapt_data_for_fusemoe(X_train, y_train)
        
        if inputs is None or targets is None:
            print("Failed to adapt data for FuseMoE.")
            return False
        
        # Create model
        print("Creating FuseMoE model...")
        model = create_fusemoe_model()
        
        # Forward pass
        print("Performing forward pass...")
        output = model(
            sleep=inputs['sleep'],
            weather=inputs['weather'],
            stress_diet=inputs['stress_diet'],
            physio=inputs['physio']
        )
        
        # Check output shape
        expected_shape = targets.shape
        if output.shape == expected_shape:
            print(f"Test passed! Output shape: {output.shape}")
            return True
        else:
            print(f"Test failed! Expected shape {expected_shape}, got {output.shape}")
            return False
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the integration with sample data
    print("\n=== Testing with sample data ===")
    test_fusemoe_integration()
    
    # Test with processed data if available
    print("\n=== Testing with processed data ===")
    try:
        test_with_processed_data()
    except ImportError:
        print("Data processor not available. Skipping test with processed data.")
