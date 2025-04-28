"""
Pipeline Integration module for the Enhanced Data Generation Pipeline.

This module integrates all components of the pipeline and provides a simple interface
for generating synthetic migraine data.
"""

import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple, Union

from moe_data_pipeline.config.config_manager import ConfigManager
from moe_data_pipeline.generators.sleep_data_generator import SleepDataGenerator
from moe_data_pipeline.generators.weather_data_generator import WeatherDataGenerator
from moe_data_pipeline.generators.stress_diet_generator import StressDietGenerator
from moe_data_pipeline.generators.physiological_generator import PhysiologicalGenerator
from moe_data_pipeline.orchestration.temporal_orchestrator import TemporalOrchestrator
from moe_data_pipeline.validation.data_validator import DataValidator
from moe_data_pipeline.formatting.output_formatter import OutputFormatter


class MigraineDataPipeline:
    """
    Migraine Data Pipeline for generating synthetic migraine data.
    
    This class integrates all components of the pipeline and provides a simple interface
    for generating synthetic migraine data.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        """
        Initialize the Migraine Data Pipeline.
        
        Args:
            config_path: Path to the configuration file.
            config: Configuration dictionary (alternative to config_path).
            seed: Optional random seed for reproducibility.
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path, config)
        self.config = self.config_manager.get_config()
        
        # Set random seed
        self.seed = seed if seed is not None else np.random.randint(0, 2**32 - 1)
        
        # Initialize components
        self.sleep_generator = SleepDataGenerator(self.config, self.seed)
        self.weather_generator = WeatherDataGenerator(self.config, self.seed + 1)
        self.stress_diet_generator = StressDietGenerator(self.config, self.seed + 2)
        self.physiological_generator = PhysiologicalGenerator(self.config, self.seed + 3)
        self.temporal_orchestrator = TemporalOrchestrator(self.config, self.seed + 4)
        self.data_validator = DataValidator(self.config)
        self.output_formatter = OutputFormatter(self.config)
    
    def generate_data(self, num_samples: int, time_periods: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Generate synthetic migraine data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            
        Returns:
            Dictionary containing the generated data for each domain.
        """
        # Generate sleep data
        print("Generating sleep data...")
        sleep_data = self.sleep_generator.generate(num_samples, time_periods)
        
        # Generate weather data
        print("Generating weather data...")
        weather_data = self.weather_generator.generate(num_samples, time_periods)
        
        # Generate stress and diet data
        print("Generating stress and diet data...")
        stress_diet_data = self.stress_diet_generator.generate(num_samples, time_periods)
        
        # Generate physiological data
        print("Generating physiological data...")
        physiological_data = self.physiological_generator.generate(
            num_samples, time_periods, 
            stress_level=stress_diet_data.get("stress_level"),
            exercise_duration=stress_diet_data.get("exercise_duration")
        )
        
        # Orchestrate temporal patterns and generate migraine data
        print("Orchestrating temporal patterns and generating migraine data...")
        migraine_data = self.temporal_orchestrator.orchestrate(
            sleep_data, weather_data, stress_diet_data, physiological_data
        )
        
        # Validate data
        print("Validating data...")
        validation_results = self.data_validator.validate(
            sleep_data, weather_data, stress_diet_data, physiological_data, migraine_data
        )
        
        if not validation_results["valid"]:
            print("Data validation failed:")
            for error in validation_results["errors"]:
                print(f"- {error}")
        else:
            print("Data validation passed.")
        
        # Return generated data
        return {
            "sleep": sleep_data,
            "weather": weather_data,
            "stress_diet": stress_diet_data,
            "physiological": physiological_data,
            "migraine": migraine_data,
            "validation": validation_results
        }
    
    def format_and_save_data(self, 
                           generated_data: Dict[str, Dict[str, np.ndarray]], 
                           output_dir: str,
                           prefix: str = "migraine_data") -> Dict[str, str]:
        """
        Format and save the generated data.
        
        Args:
            generated_data: Dictionary containing the generated data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Format data
        print("Formatting data...")
        formatted_data = self.output_formatter.format_data(
            generated_data["sleep"],
            generated_data["weather"],
            generated_data["stress_diet"],
            generated_data["physiological"],
            generated_data["migraine"]
        )
        
        # Save data
        print("Saving data...")
        paths = self.output_formatter.save_data(formatted_data, output_dir, prefix)
        
        print(f"Data saved to {output_dir}")
        for split, path in paths.items():
            print(f"- {split}: {path}")
        
        return paths
    
    def generate_and_save_data(self, 
                             num_samples: int, 
                             time_periods: int, 
                             output_dir: str,
                             prefix: str = "migraine_data") -> Dict[str, str]:
        """
        Generate, format, and save synthetic migraine data.
        
        Args:
            num_samples: Number of samples to generate.
            time_periods: Number of time periods for time-series data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Generate data
        generated_data = self.generate_data(num_samples, time_periods)
        
        # Format and save data
        paths = self.format_and_save_data(generated_data, output_dir, prefix)
        
        return paths
