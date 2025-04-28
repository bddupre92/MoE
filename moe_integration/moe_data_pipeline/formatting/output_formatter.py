"""
Output Formatter module for the Enhanced Data Generation Pipeline.

This module formats generated data for compatibility with the MoE system,
supports multiple output formats, and implements data splitting for
training/validation/testing.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Union
import torch


class OutputFormatter:
    """
    Output Formatter for the Enhanced Data Generation Pipeline.
    
    This class formats generated data for compatibility with the MoE system,
    supports multiple output formats, and implements data splitting for
    training/validation/testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Output Formatter.
        
        Args:
            config: Configuration dictionary for the formatter.
        """
        self.config = config
        self.general_config = config.get("general", {})
        self.output_format = self.general_config.get("output_format", "pytorch")
        
        # Default split ratios
        self.train_ratio = self.general_config.get("train_ratio", 0.7)
        self.val_ratio = self.general_config.get("val_ratio", 0.15)
        self.test_ratio = self.general_config.get("test_ratio", 0.15)
    
    def format_data(self, 
                   sleep_data: Dict[str, np.ndarray],
                   weather_data: Dict[str, np.ndarray],
                   stress_diet_data: Dict[str, np.ndarray],
                   physiological_data: Dict[str, np.ndarray],
                   migraine_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format the generated data for compatibility with the MoE system.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            migraine_data: Dictionary containing migraine data.
            
        Returns:
            Dictionary containing the formatted data in the specified output format.
        """
        # Extract dimensions
        num_samples, time_periods = next(iter(sleep_data.values())).shape
        
        # Split data into train/val/test sets
        indices = self._split_indices(num_samples)
        
        # Format data based on output format
        if self.output_format.lower() == "pytorch":
            return self._format_pytorch(
                sleep_data, weather_data, stress_diet_data, physiological_data, migraine_data, indices
            )
        elif self.output_format.lower() == "numpy":
            return self._format_numpy(
                sleep_data, weather_data, stress_diet_data, physiological_data, migraine_data, indices
            )
        elif self.output_format.lower() == "pandas":
            return self._format_pandas(
                sleep_data, weather_data, stress_diet_data, physiological_data, migraine_data, indices
            )
        elif self.output_format.lower() == "json":
            return self._format_json(
                sleep_data, weather_data, stress_diet_data, physiological_data, migraine_data, indices
            )
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
    
    def save_data(self, 
                 formatted_data: Dict[str, Any], 
                 output_dir: str,
                 prefix: str = "migraine_data") -> Dict[str, str]:
        """
        Save the formatted data to disk.
        
        Args:
            formatted_data: Dictionary containing the formatted data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data based on output format
        if self.output_format.lower() == "pytorch":
            return self._save_pytorch(formatted_data, output_dir, prefix)
        elif self.output_format.lower() == "numpy":
            return self._save_numpy(formatted_data, output_dir, prefix)
        elif self.output_format.lower() == "pandas":
            return self._save_pandas(formatted_data, output_dir, prefix)
        elif self.output_format.lower() == "json":
            return self._save_json(formatted_data, output_dir, prefix)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
    
    def _split_indices(self, num_samples: int) -> Dict[str, np.ndarray]:
        """
        Split data indices into train/val/test sets.
        
        Args:
            num_samples: Number of samples to split.
            
        Returns:
            Dictionary containing the indices for each split.
        """
        # Create permuted indices
        indices = np.random.permutation(num_samples)
        
        # Calculate split sizes
        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }
    
    def _format_pytorch(self, 
                       sleep_data: Dict[str, np.ndarray],
                       weather_data: Dict[str, np.ndarray],
                       stress_diet_data: Dict[str, np.ndarray],
                       physiological_data: Dict[str, np.ndarray],
                       migraine_data: Dict[str, np.ndarray],
                       indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format data as PyTorch tensors.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            migraine_data: Dictionary containing migraine data.
            indices: Dictionary containing the indices for each split.
            
        Returns:
            Dictionary containing the formatted data as PyTorch tensors.
        """
        # Prepare data for each expert
        sleep_features = self._prepare_expert_features(sleep_data)
        weather_features = self._prepare_expert_features(weather_data)
        stress_diet_features = self._prepare_expert_features(stress_diet_data)
        physiological_features = self._prepare_expert_features(physiological_data)
        
        # Prepare target data (migraine intensity)
        target = migraine_data["intensity"]
        
        # Convert to PyTorch tensors
        sleep_tensor = torch.tensor(sleep_features, dtype=torch.float32)
        weather_tensor = torch.tensor(weather_features, dtype=torch.float32)
        stress_diet_tensor = torch.tensor(stress_diet_features, dtype=torch.float32)
        physiological_tensor = torch.tensor(physiological_features, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        # Split data
        result = {}
        for split, split_indices in indices.items():
            result[split] = {
                "sleep": sleep_tensor[split_indices],
                "weather": weather_tensor[split_indices],
                "stress_diet": stress_diet_tensor[split_indices],
                "physiological": physiological_tensor[split_indices],
                "target": target_tensor[split_indices]
            }
            
            # Add combined input for MoE model
            result[split]["X"] = [
                result[split]["sleep"],
                result[split]["weather"],
                result[split]["stress_diet"],
                result[split]["physiological"]
            ]
            result[split]["y"] = result[split]["target"]
        
        # Add metadata
        result["metadata"] = {
            "sleep_features": list(sleep_data.keys()),
            "weather_features": list(weather_data.keys()),
            "stress_diet_features": list(stress_diet_data.keys()),
            "physiological_features": list(physiological_data.keys()),
            "target_feature": "migraine_intensity",
            "num_samples": {split: len(split_indices) for split, split_indices in indices.items()},
            "time_periods": target.shape[1]
        }
        
        return result
    
    def _format_numpy(self, 
                     sleep_data: Dict[str, np.ndarray],
                     weather_data: Dict[str, np.ndarray],
                     stress_diet_data: Dict[str, np.ndarray],
                     physiological_data: Dict[str, np.ndarray],
                     migraine_data: Dict[str, np.ndarray],
                     indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format data as NumPy arrays.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            migraine_data: Dictionary containing migraine data.
            indices: Dictionary containing the indices for each split.
            
        Returns:
            Dictionary containing the formatted data as NumPy arrays.
        """
        # Prepare data for each expert
        sleep_features = self._prepare_expert_features(sleep_data)
        weather_features = self._prepare_expert_features(weather_data)
        stress_diet_features = self._prepare_expert_features(stress_diet_data)
        physiological_features = self._prepare_expert_features(physiological_data)
        
        # Prepare target data (migraine intensity)
        target = migraine_data["intensity"]
        
        # Split data
        result = {}
        for split, split_indices in indices.items():
            result[split] = {
                "sleep": sleep_features[split_indices],
                "weather": weather_features[split_indices],
                "stress_diet": stress_diet_features[split_indices],
                "physiological": physiological_features[split_indices],
                "target": target[split_indices]
            }
            
            # Add combined input for MoE model
            result[split]["X"] = [
                result[split]["sleep"],
                result[split]["weather"],
                result[split]["stress_diet"],
                result[split]["physiological"]
            ]
            result[split]["y"] = result[split]["target"]
            
            # Add test predictions file format for dashboard compatibility
            if split == "test":
                result["test_predictions"] = {
                    "y_test": result[split]["target"],
                    "y_pred_test": np.zeros_like(result[split]["target"])  # Placeholder for predictions
                }
        
        # Add metadata
        result["metadata"] = {
            "sleep_features": list(sleep_data.keys()),
            "weather_features": list(weather_data.keys()),
            "stress_diet_features": list(stress_diet_data.keys()),
            "physiological_features": list(physiological_data.keys()),
            "target_feature": "migraine_intensity",
            "num_samples": {split: len(split_indices) for split, split_indices in indices.items()},
            "time_periods": target.shape[1]
        }
        
        return result
    
    def _format_pandas(self, 
                      sleep_data: Dict[str, np.ndarray],
                      weather_data: Dict[str, np.ndarray],
                      stress_diet_data: Dict[str, np.ndarray],
                      physiological_data: Dict[str, np.ndarray],
                      migraine_data: Dict[str, np.ndarray],
                      indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format data as Pandas DataFrames.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            migraine_data: Dictionary containing migraine data.
            indices: Dictionary containing the indices for each split.
            
        Returns:
            Dictionary containing the formatted data as Pandas DataFrames.
        """
        # Extract dimensions
        num_samples, time_periods = next(iter(sleep_data.values())).shape
        
        # Create DataFrames for each domain
        dfs = {}
        
        # Process each split
        for split, split_indices in indices.items():
            # Create list to hold DataFrames for each time period
            split_dfs = []
            
            for t in range(time_periods):
                # Create data for this time period
                data = {}
                
                # Add sample ID
                data["sample_id"] = split_indices
                
                # Add time period
                data["time_period"] = t
                
                # Add sleep features
                for feature, values in sleep_data.items():
                    data[f"sleep_{feature}"] = values[split_indices, t]
                
                # Add weather features
                for feature, values in weather_data.items():
                    data[f"weather_{feature}"] = values[split_indices, t]
                
                # Add stress/diet features
                for feature, values in stress_diet_data.items():
                    data[f"stress_diet_{feature}"] = values[split_indices, t]
                
                # Add physiological features
                for feature, values in physiological_data.items():
                    data[f"physiological_{feature}"] = values[split_indices, t]
                
                # Add target
                data["migraine_intensity"] = migraine_data["intensity"][split_indices, t]
                
                # Create DataFrame for this time period
                df = pd.DataFrame(data)
                split_dfs.append(df)
            
            # Concatenate DataFrames for all time periods
            dfs[split] = pd.concat(split_dfs, ignore_index=True)
        
        # Add metadata
        metadata = {
            "sleep_features": list(sleep_data.keys()),
            "weather_features": list(weather_data.keys()),
            "stress_diet_features": list(stress_diet_data.keys()),
            "physiological_features": list(physiological_data.keys()),
            "target_feature": "migraine_intensity",
            "num_samples": {split: len(split_indices) for split, split_indices in indices.items()},
            "time_periods": time_periods
        }
        
        return {"dataframes": dfs, "metadata": metadata}
    
    def _format_json(self, 
                    sleep_data: Dict[str, np.ndarray],
                    weather_data: Dict[str, np.ndarray],
                    stress_diet_data: Dict[str, np.ndarray],
                    physiological_data: Dict[str, np.ndarray],
                    migraine_data: Dict[str, np.ndarray],
                    indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Format data as JSON-serializable dictionaries.
        
        Args:
            sleep_data: Dictionary containing sleep data.
            weather_data: Dictionary containing weather data.
            stress_diet_data: Dictionary containing stress and diet data.
            physiological_data: Dictionary containing physiological data.
            migraine_data: Dictionary containing migraine data.
            indices: Dictionary containing the indices for each split.
            
        Returns:
            Dictionary containing the formatted data as JSON-serializable dictionaries.
        """
        # Extract dimensions
        num_samples, time_periods = next(iter(sleep_data.values())).shape
        
        # Create JSON data for each split
        json_data = {}
        
        for split, split_indices in indices.items():
            # Create list of samples
            samples = []
            
            for i, idx in enumerate(split_indices):
                # Create sample dictionary
                sample = {
                    "sample_id": int(idx),
                    "time_series": []
                }
                
                # Add time series data
                for t in range(time_periods):
                    time_point = {
                        "time_period": t,
                        "sleep": {},
                        "weather": {},
                        "stress_diet": {},
                        "physiological": {},
                        "migraine_intensity": float(migraine_data["intensity"][idx, t])
                    }
                    
                    # Add sleep features
                    for feature, values in sleep_data.items():
                        time_point["sleep"][feature] = float(values[idx, t])
                    
                    # Add weather features
                    for feature, values in weather_data.items():
                        time_point["weather"][feature] = float(values[idx, t])
                    
                    # Add stress/diet features
                    for feature, values in stress_diet_data.items():
                        time_point["stress_diet"][feature] = float(values[idx, t])
                    
                    # Add physiological features
                    for feature, values in physiological_data.items():
                        time_point["physiological"][feature] = float(values[idx, t])
                    
                    sample["time_series"].append(time_point)
                
                samples.append(sample)
            
            json_data[split] = samples
        
        # Add metadata
        metadata = {
            "sleep_features": list(sleep_data.keys()),
            "weather_features": list(weather_data.keys()),
            "stress_diet_features": list(stress_diet_data.keys()),
            "physiological_features": list(physiological_data.keys()),
            "target_feature": "migraine_intensity",
            "num_samples": {split: len(split_indices) for split, split_indices in indices.items()},
            "time_periods": time_periods
        }
        
        return {"data": json_data, "metadata": metadata}
    
    def _prepare_expert_features(self, domain_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Prepare features for a specific expert by stacking arrays.
        
        Args:
            domain_data: Dictionary containing domain-specific data.
            
        Returns:
            NumPy array of shape (num_samples, time_periods, num_features) containing stacked features.
        """
        # Extract dimensions
        num_samples, time_periods = next(iter(domain_data.values())).shape
        num_features = len(domain_data)
        
        # Initialize feature array
        features = np.zeros((num_samples, time_periods, num_features))
        
        # Stack features
        for i, (feature, values) in enumerate(domain_data.items()):
            features[:, :, i] = values
        
        return features
    
    def _save_pytorch(self, 
                     formatted_data: Dict[str, Any], 
                     output_dir: str,
                     prefix: str) -> Dict[str, str]:
        """
        Save PyTorch tensors to disk.
        
        Args:
            formatted_data: Dictionary containing the formatted data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Create paths
        paths = {}
        
        # Save each split
        for split in ["train", "val", "test"]:
            if split in formatted_data:
                split_path = os.path.join(output_dir, f"{prefix}_{split}.pt")
                torch.save(formatted_data[split], split_path)
                paths[split] = split_path
        
        # Save metadata
        if "metadata" in formatted_data:
            metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(formatted_data["metadata"], f, indent=2)
            paths["metadata"] = metadata_path
        
        return paths
    
    def _save_numpy(self, 
                   formatted_data: Dict[str, Any], 
                   output_dir: str,
                   prefix: str) -> Dict[str, str]:
        """
        Save NumPy arrays to disk.
        
        Args:
            formatted_data: Dictionary containing the formatted data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Create paths
        paths = {}
        
        # Save each split
        for split in ["train", "val", "test"]:
            if split in formatted_data:
                split_path = os.path.join(output_dir, f"{prefix}_{split}.npz")
                np.savez(split_path, **formatted_data[split])
                paths[split] = split_path
        
        # Save test predictions for dashboard compatibility
        if "test_predictions" in formatted_data:
            pred_path = os.path.join(output_dir, "test_predictions.npz")
            np.savez(pred_path, **formatted_data["test_predictions"])
            paths["test_predictions"] = pred_path
        
        # Save metadata
        if "metadata" in formatted_data:
            metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(formatted_data["metadata"], f, indent=2)
            paths["metadata"] = metadata_path
        
        return paths
    
    def _save_pandas(self, 
                    formatted_data: Dict[str, Any], 
                    output_dir: str,
                    prefix: str) -> Dict[str, str]:
        """
        Save Pandas DataFrames to disk.
        
        Args:
            formatted_data: Dictionary containing the formatted data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Create paths
        paths = {}
        
        # Save each split
        for split, df in formatted_data["dataframes"].items():
            split_path = os.path.join(output_dir, f"{prefix}_{split}.csv")
            df.to_csv(split_path, index=False)
            paths[split] = split_path
        
        # Save metadata
        if "metadata" in formatted_data:
            metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(formatted_data["metadata"], f, indent=2)
            paths["metadata"] = metadata_path
        
        return paths
    
    def _save_json(self, 
                  formatted_data: Dict[str, Any], 
                  output_dir: str,
                  prefix: str) -> Dict[str, str]:
        """
        Save JSON data to disk.
        
        Args:
            formatted_data: Dictionary containing the formatted data.
            output_dir: Directory to save the data.
            prefix: Prefix for the output files.
            
        Returns:
            Dictionary containing the paths to the saved files.
        """
        # Create paths
        paths = {}
        
        # Save each split
        for split, data in formatted_data["data"].items():
            split_path = os.path.join(output_dir, f"{prefix}_{split}.json")
            with open(split_path, "w") as f:
                json.dump(data, f, indent=2)
            paths[split] = split_path
        
        # Save metadata
        if "metadata" in formatted_data:
            metadata_path = os.path.join(output_dir, f"{prefix}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(formatted_data["metadata"], f, indent=2)
            paths["metadata"] = metadata_path
        
        # Save combined file
        combined_path = os.path.join(output_dir, f"{prefix}_all.json")
        with open(combined_path, "w") as f:
            json.dump(formatted_data, f, indent=2)
        paths["combined"] = combined_path
        
        return paths
