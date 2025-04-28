"""
Configuration Manager for the Enhanced Data Generation Pipeline.

This module provides centralized management of all generator parameters,
supports configuration profiles for different scenarios, and enables
reproducible data generation with seed management.
"""

import os
import json
import yaml
import copy
from typing import Dict, Any, Optional, Union


class ConfigManager:
    """
    Configuration Manager for the Enhanced Data Generation Pipeline.
    
    This class provides centralized management of all generator parameters,
    supports configuration profiles for different scenarios, and enables
    reproducible data generation with seed management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Configuration Manager.
        
        Args:
            config_path: Optional path to a configuration file (JSON or YAML).
                         If not provided, default configuration will be used.
        """
        # Load default configuration
        self.default_config = self._load_default_config()
        
        # Initialize with default configuration
        self.config = copy.deepcopy(self.default_config)
        
        # If config_path is provided, load and merge with defaults
        if config_path:
            self.load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load the default configuration.
        
        Returns:
            Dict containing the default configuration.
        """
        return {
            "general": {
                "seed": 42,
                "num_samples": 1000,
                "time_periods": 7,
                "output_format": "pytorch"
            },
            "sleep": {
                "duration_mean": 7.0,
                "duration_std": 1.5,
                "quality_mean": 6.0,
                "quality_std": 2.0,
                "deep_sleep_mean": 20.0,
                "deep_sleep_std": 5.0,
                "rem_sleep_mean": 25.0,
                "rem_sleep_std": 5.0,
                "interruptions_lambda": 2.0,
                "time_to_sleep_lambda": 15.0,
                "circadian_rhythm_enabled": True,
                "weekend_variation_enabled": True
            },
            "weather": {
                "temperature_mean": 20.0,
                "temperature_std": 8.0,
                "humidity_mean": 60.0,
                "humidity_std": 15.0,
                "pressure_mean": 1013.0,
                "pressure_std": 10.0,
                "precipitation_lambda": 2.0,
                "wind_speed_lambda": 10.0,
                "seasonal_variation_enabled": True,
                "daily_fluctuation_enabled": True,
                "extreme_events_enabled": True,
                "barometric_pressure_changes_enabled": True
            },
            "stress_diet": {
                "stress_mean": 5.0,
                "stress_std": 2.0,
                "alcohol_lambda": 1.0,
                "meal_regularity_mean": 6.0,
                "meal_regularity_std": 2.0,
                "hydration_mean": 6.0,
                "hydration_std": 2.0,
                "exercise_lambda": 30.0,
                "work_life_cycle_enabled": True,
                "weekend_variation_enabled": True,
                "stress_diet_correlation_enabled": True
            },
            "physiological": {
                "heart_rate_mean": 75.0,
                "heart_rate_std": 10.0,
                "systolic_mean": 120.0,
                "systolic_std": 15.0,
                "diastolic_mean": 80.0,
                "diastolic_std": 10.0,
                "temperature_mean": 36.8,
                "temperature_std": 0.5,
                "respiratory_mean": 16.0,
                "respiratory_std": 3.0,
                "circadian_rhythm_enabled": True,
                "exercise_effect_enabled": True,
                "parameter_correlations_enabled": True
            },
            "migraine": {
                "sleep_weights": {
                    "sleep_duration": -0.3,
                    "sleep_quality": -0.4,
                    "deep_sleep_percentage": -0.2,
                    "rem_sleep_percentage": 0.1,
                    "sleep_interruptions": 0.3,
                    "time_to_sleep": 0.2
                },
                "weather_weights": {
                    "temperature": 0.1,
                    "humidity": 0.2,
                    "pressure": 0.4,
                    "precipitation": 0.2,
                    "wind_speed": 0.1,
                    "pressure_change_rate": 0.5
                },
                "stress_diet_weights": {
                    "stress_level": 0.4,
                    "caffeine_intake": 0.3,
                    "alcohol_consumption": 0.2,
                    "meal_regularity": -0.3,
                    "hydration": -0.4
                },
                "physiological_weights": {
                    "heart_rate_variability": 0.3,
                    "blood_pressure": 0.2,
                    "body_temperature": 0.1
                },
                "temporal_effects": {
                    "lag_hours": [6, 12, 24, 48],
                    "lag_weights": [0.2, 0.4, 0.3, 0.1],
                    "cumulative_effect_enabled": True,
                    "cumulative_effect_decay": 0.8
                },
                "individual_variability": {
                    "enabled": True,
                    "sensitivity_std": 0.5,
                    "threshold_mean": 0.7,
                    "threshold_std": 0.1
                }
            },
            "validation": {
                "enabled": True,
                "check_physiological_plausibility": True,
                "check_temporal_consistency": True,
                "check_correlations": True,
                "check_class_balance": True
            },
            "augmentation": {
                "enabled": False,
                "noise_level": 0.05,
                "edge_case_percentage": 0.1
            }
        }
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file and merge with defaults.
        
        Args:
            config_path: Path to a configuration file (JSON or YAML).
        
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file format is not supported.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format based on extension
        _, ext = os.path.splitext(config_path)
        
        # Load configuration based on file format
        if ext.lower() in ['.json']:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
        
        # Merge with default configuration
        self._merge_config(user_config)
    
    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """
        Merge user configuration with default configuration.
        
        Args:
            user_config: User-provided configuration dictionary.
        """
        # Deep merge of nested dictionaries
        for key, value in user_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self._merge_dict(self.config[key], value)
            else:
                self.config[key] = value
    
    def _merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries.
        
        Args:
            target: Target dictionary to merge into.
            source: Source dictionary to merge from.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_dict(target[key], value)
            else:
                target[key] = value
    
    def get_config(self, section: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get the current configuration.
        
        Args:
            section: Optional section name to retrieve specific configuration.
                    If not provided, the entire configuration is returned.
        
        Returns:
            Configuration dictionary or specific section.
        
        Raises:
            KeyError: If the specified section does not exist.
        """
        if section is None:
            return copy.deepcopy(self.config)
        
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        
        return copy.deepcopy(self.config[section])
    
    def set_config(self, section: str, key: str, value: Any) -> None:
        """
        Set a specific configuration value.
        
        Args:
            section: Section name.
            key: Configuration key.
            value: Configuration value.
        
        Raises:
            KeyError: If the specified section does not exist.
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        
        self.config[section][key] = value
    
    def reset_to_defaults(self) -> None:
        """Reset the configuration to default values."""
        self.config = copy.deepcopy(self.default_config)
    
    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path to save the configuration file.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        # Determine file format based on extension
        _, ext = os.path.splitext(config_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save configuration based on file format
        if ext.lower() in ['.json']:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif ext.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")
    
    def get_seed(self) -> int:
        """
        Get the current random seed.
        
        Returns:
            Random seed value.
        """
        return self.config["general"]["seed"]
    
    def set_seed(self, seed: int) -> None:
        """
        Set the random seed.
        
        Args:
            seed: Random seed value.
        """
        self.config["general"]["seed"] = seed
    
    def get_num_samples(self) -> int:
        """
        Get the number of samples to generate.
        
        Returns:
            Number of samples.
        """
        return self.config["general"]["num_samples"]
    
    def set_num_samples(self, num_samples: int) -> None:
        """
        Set the number of samples to generate.
        
        Args:
            num_samples: Number of samples.
        """
        self.config["general"]["num_samples"] = num_samples
