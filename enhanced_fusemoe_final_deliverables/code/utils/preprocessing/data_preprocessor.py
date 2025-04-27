"""
Data Preprocessor for Enhanced FuseMoE

This module provides functionality for preprocessing data for the
Enhanced FuseMoE system for migraine prediction.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Base class for data preprocessing.
    
    This class provides methods for preprocessing data for the
    Enhanced FuseMoE system.
    
    Attributes:
        scaler_dict (Dict[str, StandardScaler]): Dictionary mapping feature names to scalers
        encoder_dict (Dict[str, OneHotEncoder]): Dictionary mapping feature names to encoders
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(self):
        """
        Initialize the data preprocessor.
        """
        self.scaler_dict = None
        self.encoder_dict = None
        self.is_fitted = False
    
    def fit_transform(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit preprocessor to dataset and transform it.
        
        Args:
            dataset: Dataset to fit and transform
            
        Returns:
            Transformed dataset
        """
        # Initialize scaler and encoder dictionaries
        self.scaler_dict = {}
        self.encoder_dict = {}
        
        # Create copy of dataset
        transformed_dataset = {}
        
        # Process each modality
        for key, data in dataset.items():
            if key == 'labels':
                # Copy labels as is
                transformed_dataset[key] = data
            elif isinstance(data, pd.DataFrame):
                # Normalize dataframe
                normalized_df, scaler = normalize_dataframe(data)
                self.scaler_dict[key] = scaler
                
                # Store transformed data
                transformed_dataset[key] = normalized_df
            else:
                # Copy data as is
                transformed_dataset[key] = data
        
        # Mark as fitted
        self.is_fitted = True
        
        return transformed_dataset
    
    def transform(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform dataset using fitted preprocessor.
        
        Args:
            dataset: Dataset to transform
            
        Returns:
            Transformed dataset
        """
        # Check if preprocessor is fitted
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Create copy of dataset
        transformed_dataset = {}
        
        # Process each modality
        for key, data in dataset.items():
            if key == 'labels':
                # Copy labels as is
                transformed_dataset[key] = data
            elif isinstance(data, pd.DataFrame) and key in self.scaler_dict:
                # Apply normalization
                normalized_df = pd.DataFrame(index=data.index)
                
                # Apply scaler to each column
                for col, scaler in self.scaler_dict[key].items():
                    if col in data.columns:
                        # Apply scaler
                        normalized_df[col] = (data[col] - scaler['mean']) / scaler['std']
                    else:
                        # Column not found, use zeros
                        normalized_df[col] = 0.0
                
                # Store transformed data
                transformed_dataset[key] = normalized_df
            else:
                # Copy data as is
                transformed_dataset[key] = data
        
        return transformed_dataset
    
    def create_torch_datasets(self, dataset: Dict[str, Any], train_ratio: float = 0.7,
                            val_ratio: float = 0.15, test_ratio: float = 0.15,
                            stratify: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create PyTorch datasets from preprocessed dataset.
        
        Args:
            dataset: Preprocessed dataset
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            stratify: Whether to stratify the split based on labels
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Check that ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1")
        
        # Create custom dataset
        inputs = {}
        for key, data in dataset.items():
            if key != 'labels':
                if isinstance(data, pd.DataFrame):
                    inputs[key] = data.values
                else:
                    inputs[key] = data
        
        labels = dataset['labels']
        
        # Create custom dataset
        full_dataset = CustomDataset(inputs, labels)
        
        # Calculate split sizes
        num_samples = len(full_dataset)
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size
        
        # Split dataset
        if stratify and 'labels' in dataset:
            # Stratified split
            # This is a simplified implementation that ensures the same proportion of labels
            # in each split, but doesn't use sklearn's stratified split
            
            # Get indices for each class
            labels = dataset['labels']
            pos_indices = np.where(labels == 1)[0]
            neg_indices = np.where(labels == 0)[0]
            
            # Calculate split sizes for each class
            pos_train_size = int(train_ratio * len(pos_indices))
            pos_val_size = int(val_ratio * len(pos_indices))
            pos_test_size = len(pos_indices) - pos_train_size - pos_val_size
            
            neg_train_size = int(train_ratio * len(neg_indices))
            neg_val_size = int(val_ratio * len(neg_indices))
            neg_test_size = len(neg_indices) - neg_train_size - neg_val_size
            
            # Shuffle indices
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)
            
            # Split indices
            pos_train_indices = pos_indices[:pos_train_size]
            pos_val_indices = pos_indices[pos_train_size:pos_train_size+pos_val_size]
            pos_test_indices = pos_indices[pos_train_size+pos_val_size:]
            
            neg_train_indices = neg_indices[:neg_train_size]
            neg_val_indices = neg_indices[neg_train_size:neg_train_size+neg_val_size]
            neg_test_indices = neg_indices[neg_train_size+neg_val_size:]
            
            # Combine indices
            train_indices = np.concatenate([pos_train_indices, neg_train_indices])
            val_indices = np.concatenate([pos_val_indices, neg_val_indices])
            test_indices = np.concatenate([pos_test_indices, neg_test_indices])
            
            # Shuffle indices
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            np.random.shuffle(test_indices)
            
            # Create subset datasets
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        else:
            # Random split
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset,
                         batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch dataloaders from datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader


class CustomDataset(Dataset):
    """
    Custom dataset for the Enhanced FuseMoE system.
    
    This class provides a PyTorch dataset for the Enhanced FuseMoE system.
    
    Attributes:
        inputs (Dict[str, torch.Tensor]): Input data for each modality
        targets (torch.Tensor): Target labels
    """
    
    def __init__(self, inputs: Dict[str, np.ndarray], targets: np.ndarray):
        """
        Initialize the custom dataset.
        
        Args:
            inputs: Input data for each modality
            targets: Target labels
        """
        # Convert inputs to tensors
        self.inputs = {}
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                self.inputs[key] = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, pd.DataFrame):
                self.inputs[key] = torch.tensor(value.values, dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported input type: {type(value)}")
        
        # Convert targets to tensor
        if isinstance(targets, np.ndarray):
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported target type: {type(targets)}")
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Dataset length
        """
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (inputs, target)
        """
        # Get inputs for each modality
        item_inputs = {}
        for key, value in self.inputs.items():
            item_inputs[key] = value[idx]
        
        # Get target
        target = self.targets[idx]
        
        return item_inputs, target
    
    def to(self, device: torch.device) -> 'CustomDataset':
        """
        Move dataset to device.
        
        Args:
            device: Device to move dataset to
            
        Returns:
            Dataset moved to device
        """
        # Move inputs to device
        for key in self.inputs:
            self.inputs[key] = self.inputs[key].to(device)
        
        # Move targets to device
        self.targets = self.targets.to(device)
        
        return self


class MigraineDataPreprocessor(DataPreprocessor):
    """
    Data preprocessor for migraine prediction.
    
    This class provides methods for preprocessing data for migraine prediction
    using the Enhanced FuseMoE system.
    
    Attributes:
        time_periods (int): Number of time periods for time series data
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        random_seed (int): Random seed for reproducibility
        scaler_dict (Dict[str, StandardScaler]): Dictionary mapping feature names to scalers
        encoder_dict (Dict[str, OneHotEncoder]): Dictionary mapping feature names to encoders
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(self, time_periods: int = 7, test_size: float = 0.2, 
                val_size: float = 0.1, random_seed: int = 42):
        """
        Initialize the migraine data preprocessor.
        
        Args:
            time_periods: Number of time periods for time series data
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_seed: Random seed for reproducibility
        """
        super().__init__()
        self.time_periods = time_periods
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def normalize_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Normalize data using min-max scaling.
        
        Args:
            data: Data to normalize
            
        Returns:
            Tuple of (normalized_data, scaler)
        """
        # Create scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Fit and transform data
        normalized_data = scaler.fit_transform(data)
        
        # Ensure all values are within the expected range
        normalized_data = np.clip(normalized_data, -1, 1)
        
        return normalized_data, scaler
    
    def create_time_series_features(self, data: np.ndarray, time_periods: int) -> np.ndarray:
        """
        Create time series features from data.
        
        Args:
            data: Data to create time series features from
            time_periods: Number of time periods to include
            
        Returns:
            Time series features
        """
        # Get data dimensions
        n_samples, n_features = data.shape
        
        # Create time series features
        time_series_data = np.zeros((n_samples - time_periods + 1, time_periods, n_features))
        
        for i in range(n_samples - time_periods + 1):
            time_series_data[i] = data[i:i+time_periods]
        
        return time_series_data
    
    def create_domain_specific_features(self, sleep_data: pd.DataFrame, weather_data: pd.DataFrame,
                                      stress_diet_data: pd.DataFrame, physio_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create domain-specific features from raw data.
        
        Args:
            sleep_data: Sleep data
            weather_data: Weather data
            stress_diet_data: Stress and diet data
            physio_data: Physiological data
            
        Returns:
            Dictionary mapping domain names to feature arrays
        """
        # Create domain-specific features
        domain_features = {}
        
        # Sleep features
        sleep_features = sleep_data.copy()
        # Add derived features
        sleep_features['sleep_efficiency'] = sleep_features['deep_sleep_percentage'] + sleep_features['rem_sleep_percentage']
        sleep_features['sleep_quality_score'] = sleep_features['sleep_quality'] * sleep_features['sleep_duration'] / 10
        domain_features['sleep_features'] = sleep_features
        
        # Weather features
        weather_features = weather_data.copy()
        # Add derived features
        if 'temperature' in weather_features.columns and 'humidity' in weather_features.columns:
            # Calculate heat index (simplified)
            weather_features['heat_index'] = weather_features['temperature'] + 0.05 * weather_features['humidity']
        if 'pressure' in weather_features.columns:
            # Calculate pressure change (if time series data)
            weather_features['pressure_change'] = weather_features['pressure'].diff().fillna(0)
        domain_features['weather_features'] = weather_features
        
        # Stress and diet features
        stress_diet_features = stress_diet_data.copy()
        # Add derived features
        if 'stress_level' in stress_diet_features.columns and 'exercise_duration' in stress_diet_features.columns:
            # Calculate stress-exercise balance
            stress_diet_features['stress_exercise_balance'] = stress_diet_features['stress_level'] - stress_diet_features['exercise_duration'] / 10
        if 'caffeine_intake' in stress_diet_features.columns and 'alcohol_intake' in stress_diet_features.columns:
            # Calculate total stimulant intake
            stress_diet_features['total_stimulants'] = stress_diet_features['caffeine_intake'] + stress_diet_features['alcohol_intake'] * 2
        domain_features['stress_diet_features'] = stress_diet_features
        
        # Physiological features
        physio_features = physio_data.copy()
        # Add derived features
        if 'blood_pressure_systolic' in physio_features.columns and 'blood_pressure_diastolic' in physio_features.columns:
            # Calculate pulse pressure
            physio_features['pulse_pressure'] = physio_features['blood_pressure_systolic'] - physio_features['blood_pressure_diastolic']
            # Calculate mean arterial pressure
            physio_features['mean_arterial_pressure'] = (
                physio_features['blood_pressure_diastolic'] + 
                physio_features['pulse_pressure'] / 3
            )
        domain_features['physio_features'] = physio_features
        
        return domain_features
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Calculate train size
        train_size = 1 - self.test_size - self.val_size
        
        # Split data into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y
        )
        
        # Split train+val into train and val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=self.random_seed, stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data_for_experts(self, sleep_data: pd.DataFrame, weather_data: pd.DataFrame,
                               stress_diet_data: pd.DataFrame, physio_data: pd.DataFrame,
                               labels: pd.Series) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for expert models.
        
        Args:
            sleep_data: Sleep data
            weather_data: Weather data
            stress_diet_data: Stress and diet data
            physio_data: Physiological data
            labels: Migraine labels
            
        Returns:
            Tuple of (expert_data, labels)
        """
        # Normalize data
        sleep_norm, _ = self.normalize_data(sleep_data)
        weather_norm, _ = self.normalize_data(weather_data)
        stress_diet_norm, _ = self.normalize_data(stress_diet_data)
        physio_norm, _ = self.normalize_data(physio_data)
        
        # Create time series features
        sleep_ts = self.create_time_series_features(sleep_norm, self.time_periods)
        weather_ts = self.create_time_series_features(weather_norm, self.time_periods)
        stress_diet_ts = self.create_time_series_features(stress_diet_norm, self.time_periods)
        physio_ts = self.create_time_series_features(physio_norm, self.time_periods)
        
        # Create expert data dictionary
        expert_data = {
            'sleep_expert': sleep_ts,
            'weather_expert': weather_ts,
            'stress_diet_expert': stress_diet_ts,
            'physio_expert': physio_ts
        }
        
        # Adjust labels for time series
        labels_array = labels.values if isinstance(labels, pd.Series) else labels
        adjusted_labels = labels_array[self.time_periods-1:]
        
        return expert_data, adjusted_labels
    
    def create_data_loaders(self, expert_data: Dict[str, np.ndarray], labels: np.ndarray,
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for expert models.
        
        Args:
            expert_data: Dictionary mapping expert names to data
            labels: Migraine labels
            batch_size: Batch size
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Get data dimensions
        n_samples = labels.shape[0]
        
        # Create indices for train, val, and test sets
        indices = np.arange(n_samples)
        train_size = int((1 - self.test_size - self.val_size) * n_samples)
        val_size = int(self.val_size * n_samples)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Create datasets
        train_dataset = ExpertDataset(expert_data, labels, train_indices)
        val_dataset = ExpertDataset(expert_data, labels, val_indices)
        test_dataset = ExpertDataset(expert_data, labels, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader


class ExpertDataset(Dataset):
    """
    Dataset for expert models.
    
    This class provides a PyTorch dataset for expert models in the
    Enhanced FuseMoE system.
    
    Attributes:
        expert_data (Dict[str, np.ndarray]): Data for each expert
        labels (np.ndarray): Migraine labels
        indices (np.ndarray): Indices to use
    """
    
    def __init__(self, expert_data: Dict[str, np.ndarray], labels: np.ndarray,
                indices: np.ndarray):
        """
        Initialize the expert dataset.
        
        Args:
            expert_data: Data for each expert
            labels: Migraine labels
            indices: Indices to use
        """
        self.expert_data = expert_data
        self.labels = labels
        self.indices = indices
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Dataset length
        """
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (expert_inputs, label)
        """
        # Get index
        index = self.indices[idx]
        
        # Get expert inputs
        expert_inputs = {}
        for expert_name, expert_data in self.expert_data.items():
            expert_inputs[expert_name] = torch.tensor(expert_data[index], dtype=torch.float32)
        
        # Get label
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        
        return expert_inputs, label


# Standalone functions for data preprocessing

def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize dataframe using z-score normalization.
    
    Args:
        df: Dataframe to normalize
        
    Returns:
        Tuple of (normalized_df, scaler_dict)
    """
    # Create copy of dataframe
    normalized_df = pd.DataFrame(index=df.index)
    
    # Create scaler dictionary
    scaler_dict = {}
    
    # Normalize each column
    for col in df.columns:
        # Calculate mean and standard deviation
        mean = df[col].mean()
        std = df[col].std()
        
        # Handle zero standard deviation
        if std == 0:
            std = 1.0
        
        # Normalize column
        normalized_df[col] = (df[col] - mean) / std
        
        # Store scaler parameters
        scaler_dict[col] = {'mean': mean, 'std': std}
    
    return normalized_df, scaler_dict


def normalize_data(data: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize data using min-max scaling.
    
    Args:
        data: Data to normalize
        
    Returns:
        Tuple of (normalized_data, scaler)
    """
    # Create scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Fit and transform data
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, OneHotEncoder]]:
    """
    Encode categorical features using one-hot encoding.
    
    Args:
        df: Dataframe containing categorical features
        
    Returns:
        Tuple of (encoded_df, encoder_dict)
    """
    # Create copy of dataframe
    encoded_df = pd.DataFrame(index=df.index)
    
    # Create encoder dictionary
    encoder_dict = {}
    
    # Process each column
    for col in df.columns:
        # Check if column is categorical
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Create encoder
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            # Fit encoder
            encoder.fit(df[[col]])
            
            # Transform column
            encoded_cols = encoder.transform(df[[col]])
            
            # Get feature names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            
            # Add encoded columns to dataframe
            for i, name in enumerate(feature_names):
                encoded_df[name] = encoded_cols[:, i]
            
            # Store encoder
            encoder_dict[col] = encoder
        else:
            # Copy numeric column as is
            encoded_df[col] = df[col]
    
    return encoded_df, encoder_dict


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df: Dataframe containing missing values
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'zero')
        
    Returns:
        Dataframe with missing values handled
    """
    # Create copy of dataframe
    filled_df = df.copy()
    
    # Handle missing values in each column
    for col in filled_df.columns:
        if filled_df[col].isna().any():
            if strategy == 'mean':
                filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
            elif strategy == 'median':
                filled_df[col] = filled_df[col].fillna(filled_df[col].median())
            elif strategy == 'mode':
                filled_df[col] = filled_df[col].fillna(filled_df[col].mode()[0])
            elif strategy == 'zero':
                filled_df[col] = filled_df[col].fillna(0)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")
    
    return filled_df


def create_feature_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Create feature vector from dataframe.
    
    Args:
        df: Dataframe to convert to feature vector
        
    Returns:
        Feature vector as numpy array
    """
    # Convert dataframe to numpy array
    feature_vector = df.values
    
    return feature_vector


def create_time_series_features(data: np.ndarray, time_periods: int) -> np.ndarray:
    """
    Create time series features from data.
    
    Args:
        data: Data to create time series features from
        time_periods: Number of time periods to include
        
    Returns:
        Time series features
    """
    # Get data dimensions
    n_samples, n_features = data.shape
    
    # Create time series features
    time_series_data = np.zeros((n_samples - time_periods + 1, time_periods, n_features))
    
    for i in range(n_samples - time_periods + 1):
        time_series_data[i] = data[i:i+time_periods]
    
    return time_series_data


def create_domain_specific_features(sleep_data: pd.DataFrame, weather_data: pd.DataFrame,
                                  stress_diet_data: pd.DataFrame, physio_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create domain-specific features from raw data.
    
    Args:
        sleep_data: Sleep data
        weather_data: Weather data
        stress_diet_data: Stress and diet data
        physio_data: Physiological data
        
    Returns:
        Dictionary mapping domain names to feature dataframes
    """
    # Create domain-specific features
    domain_features = {}
    
    # Sleep features
    sleep_features = sleep_data.copy()
    # Add derived features
    if 'deep_sleep_percentage' in sleep_features.columns and 'rem_sleep_percentage' in sleep_features.columns:
        sleep_features['sleep_efficiency'] = sleep_features['deep_sleep_percentage'] + sleep_features['rem_sleep_percentage']
    if 'sleep_quality' in sleep_features.columns and 'sleep_duration' in sleep_features.columns:
        sleep_features['sleep_quality_score'] = sleep_features['sleep_quality'] * sleep_features['sleep_duration'] / 10
    domain_features['sleep_features'] = sleep_features
    
    # Weather features
    weather_features = weather_data.copy()
    # Add derived features
    if 'temperature' in weather_features.columns and 'humidity' in weather_features.columns:
        # Calculate heat index (simplified)
        weather_features['heat_index'] = weather_features['temperature'] + 0.05 * weather_features['humidity']
    if 'pressure' in weather_features.columns:
        # Calculate pressure change (if time series data)
        weather_features['pressure_change'] = weather_features['pressure'].diff().fillna(0)
    domain_features['weather_features'] = weather_features
    
    # Stress and diet features
    stress_diet_features = stress_diet_data.copy()
    # Add derived features
    if 'stress_level' in stress_diet_features.columns and 'exercise_duration' in stress_diet_features.columns:
        # Calculate stress-exercise balance
        stress_diet_features['stress_exercise_balance'] = stress_diet_features['stress_level'] - stress_diet_features['exercise_duration'] / 10
    if 'caffeine_intake' in stress_diet_features.columns and 'alcohol_intake' in stress_diet_features.columns:
        # Calculate total stimulant intake
        stress_diet_features['total_stimulants'] = stress_diet_features['caffeine_intake'] + stress_diet_features['alcohol_intake'] * 2
    domain_features['stress_diet_features'] = stress_diet_features
    
    # Physiological features
    physio_features = physio_data.copy()
    # Add derived features
    if 'blood_pressure_systolic' in physio_features.columns and 'blood_pressure_diastolic' in physio_features.columns:
        # Calculate pulse pressure
        physio_features['pulse_pressure'] = physio_features['blood_pressure_systolic'] - physio_features['blood_pressure_diastolic']
        # Calculate mean arterial pressure
        physio_features['mean_arterial_pressure'] = (
            physio_features['blood_pressure_diastolic'] + 
            physio_features['pulse_pressure'] / 3
        )
    domain_features['physio_features'] = physio_features
    
    return domain_features
