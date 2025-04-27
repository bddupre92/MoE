"""
Enhanced Data Preprocessing for Migraine Prediction

This module provides advanced data preprocessing techniques for the Enhanced FuseMoE
system to improve model performance beyond 95%.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from imblearn.over_sampling import SMOTE

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering for migraine prediction.
    
    This class implements domain-specific feature engineering techniques
    to extract more predictive features from the raw data.
    
    Attributes:
        window_size: Size of the window for temporal features
        use_derivatives: Whether to use derivatives of features
        use_interactions: Whether to use feature interactions
        use_temporal_patterns: Whether to use temporal pattern features
    """
    
    def __init__(self, window_size: int = 3, use_derivatives: bool = True,
                use_interactions: bool = True, use_temporal_patterns: bool = True):
        """
        Initialize advanced feature engineering.
        
        Args:
            window_size: Size of the window for temporal features
            use_derivatives: Whether to use derivatives of features
            use_interactions: Whether to use feature interactions
            use_temporal_patterns: Whether to use temporal pattern features
        """
        self.window_size = window_size
        self.use_derivatives = use_derivatives
        self.use_interactions = use_interactions
        self.use_temporal_patterns = use_temporal_patterns
    
    def transform_sleep_data(self, sleep_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform sleep data with advanced features.
        
        Args:
            sleep_data: DataFrame containing sleep data
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df = sleep_data.copy()
        
        # Calculate sleep pattern variability
        if self.use_temporal_patterns and len(df) >= self.window_size:
            # Rolling standard deviation of sleep duration
            df['sleep_duration_variability'] = df['sleep_duration'].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0)
            
            # Rolling standard deviation of sleep quality
            df['sleep_quality_variability'] = df['sleep_quality'].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0)
            
            # Sleep regularity index (consistency of sleep timing)
            df['sleep_regularity'] = 1.0 - df['sleep_onset_time'].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0) / 24.0  # Normalize by 24 hours
        
        # Calculate derivatives (rate of change)
        if self.use_derivatives and len(df) >= 2:
            df['sleep_duration_change'] = df['sleep_duration'].diff().fillna(0)
            df['sleep_quality_change'] = df['sleep_quality'].diff().fillna(0)
            df['rem_percentage_change'] = df['rem_percentage'].diff().fillna(0)
        
        # Calculate interaction features
        if self.use_interactions:
            # Interaction between sleep duration and quality
            df['duration_quality_interaction'] = df['sleep_duration'] * df['sleep_quality']
            
            # Interaction between REM percentage and deep sleep percentage
            if 'deep_sleep_percentage' in df.columns:
                df['rem_deep_interaction'] = df['rem_percentage'] * df['deep_sleep_percentage']
        
        return df
    
    def transform_weather_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform weather data with advanced features.
        
        Args:
            weather_data: DataFrame containing weather data
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df = weather_data.copy()
        
        # Calculate barometric pressure trend
        if self.use_derivatives and len(df) >= 2:
            if 'barometric_pressure' in df.columns:
                df['pressure_change'] = df['barometric_pressure'].diff().fillna(0)
                
                # Rapid pressure changes are known migraine triggers
                if len(df) >= self.window_size:
                    df['pressure_change_rate'] = df['barometric_pressure'].diff(
                        periods=self.window_size
                    ).fillna(0) / self.window_size
            
            # Temperature changes
            if 'temperature' in df.columns:
                df['temperature_change'] = df['temperature'].diff().fillna(0)
                
                # Rapid temperature changes
                if len(df) >= self.window_size:
                    df['temperature_change_rate'] = df['temperature'].diff(
                        periods=self.window_size
                    ).fillna(0) / self.window_size
        
        # Calculate weather instability index
        if self.use_temporal_patterns and len(df) >= self.window_size:
            weather_vars = ['temperature', 'humidity', 'barometric_pressure']
            weather_vars = [var for var in weather_vars if var in df.columns]
            
            if weather_vars:
                # Normalize each variable
                normalized_vars = {}
                for var in weather_vars:
                    normalized_vars[var] = (df[var] - df[var].mean()) / (df[var].std() + 1e-8)
                
                # Calculate instability as the sum of rolling standard deviations
                instability = 0
                for var in weather_vars:
                    instability += normalized_vars[var].rolling(
                        window=self.window_size, min_periods=1
                    ).std().fillna(0)
                
                df['weather_instability'] = instability / len(weather_vars)
        
        # Calculate interaction features
        if self.use_interactions:
            # Interaction between temperature and humidity (heat index)
            if 'temperature' in df.columns and 'humidity' in df.columns:
                df['heat_index'] = 0.5 * (
                    df['temperature'] + 61.0 + ((df['temperature'] - 68.0) * 1.2) + (df['humidity'] * 0.094)
                )
            
            # Interaction between pressure and humidity
            if 'barometric_pressure' in df.columns and 'humidity' in df.columns:
                df['pressure_humidity_interaction'] = df['barometric_pressure'] * df['humidity'] / 100.0
        
        return df
    
    def transform_stress_diet_data(self, stress_diet_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform stress and diet data with advanced features.
        
        Args:
            stress_diet_data: DataFrame containing stress and diet data
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df = stress_diet_data.copy()
        
        # Calculate stress accumulation
        if self.use_temporal_patterns and len(df) >= self.window_size:
            if 'stress_level' in df.columns:
                df['stress_accumulation'] = df['stress_level'].rolling(
                    window=self.window_size, min_periods=1
                ).sum().fillna(0)
                
                # Stress variability
                df['stress_variability'] = df['stress_level'].rolling(
                    window=self.window_size, min_periods=1
                ).std().fillna(0)
        
        # Calculate derivatives
        if self.use_derivatives and len(df) >= 2:
            if 'stress_level' in df.columns:
                df['stress_change'] = df['stress_level'].diff().fillna(0)
            
            if 'caffeine_intake' in df.columns:
                df['caffeine_change'] = df['caffeine_intake'].diff().fillna(0)
            
            if 'alcohol_consumption' in df.columns:
                df['alcohol_change'] = df['alcohol_consumption'].diff().fillna(0)
        
        # Calculate interaction features
        if self.use_interactions:
            # Interaction between stress and caffeine
            if 'stress_level' in df.columns and 'caffeine_intake' in df.columns:
                df['stress_caffeine_interaction'] = df['stress_level'] * df['caffeine_intake']
            
            # Interaction between stress and alcohol
            if 'stress_level' in df.columns and 'alcohol_consumption' in df.columns:
                df['stress_alcohol_interaction'] = df['stress_level'] * df['alcohol_consumption']
            
            # Interaction between hydration and alcohol
            if 'hydration_level' in df.columns and 'alcohol_consumption' in df.columns:
                df['hydration_alcohol_interaction'] = (1.0 - df['hydration_level']) * df['alcohol_consumption']
            
            # Meal regularity and stress
            if 'meal_regularity' in df.columns and 'stress_level' in df.columns:
                df['meal_stress_interaction'] = (1.0 - df['meal_regularity']) * df['stress_level']
        
        # Calculate dietary trigger index
        if 'caffeine_intake' in df.columns and 'alcohol_consumption' in df.columns:
            # Weighted sum of known dietary triggers
            df['dietary_trigger_index'] = (
                0.4 * df['caffeine_intake'] + 
                0.4 * df['alcohol_consumption'] + 
                0.2 * (1.0 - df.get('meal_regularity', 0.5))
            )
        
        return df
    
    def transform_physiological_data(self, physiological_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform physiological data with advanced features.
        
        Args:
            physiological_data: DataFrame containing physiological data
            
        Returns:
            DataFrame with engineered features
        """
        # Create a copy to avoid modifying the original
        df = physiological_data.copy()
        
        # Calculate heart rate variability
        if self.use_temporal_patterns and len(df) >= self.window_size:
            if 'heart_rate' in df.columns:
                df['heart_rate_variability'] = df['heart_rate'].rolling(
                    window=self.window_size, min_periods=1
                ).std().fillna(0)
        
        # Calculate blood pressure features
        if 'blood_pressure_systolic' in df.columns and 'blood_pressure_diastolic' in df.columns:
            # Pulse pressure
            df['pulse_pressure'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']
            
            # Mean arterial pressure
            df['mean_arterial_pressure'] = (
                df['blood_pressure_diastolic'] + 
                (df['pulse_pressure'] / 3)
            )
        
        # Calculate derivatives
        if self.use_derivatives and len(df) >= 2:
            if 'heart_rate' in df.columns:
                df['heart_rate_change'] = df['heart_rate'].diff().fillna(0)
            
            if 'blood_pressure_systolic' in df.columns:
                df['systolic_change'] = df['blood_pressure_systolic'].diff().fillna(0)
            
            if 'blood_pressure_diastolic' in df.columns:
                df['diastolic_change'] = df['blood_pressure_diastolic'].diff().fillna(0)
        
        # Calculate interaction features
        if self.use_interactions:
            # Interaction between heart rate and blood pressure
            if 'heart_rate' in df.columns and 'mean_arterial_pressure' in df.columns:
                df['rate_pressure_product'] = df['heart_rate'] * df['mean_arterial_pressure'] / 100.0
            
            # Interaction between oxygen saturation and heart rate
            if 'oxygen_saturation' in df.columns and 'heart_rate' in df.columns:
                df['oxygen_heart_interaction'] = df['oxygen_saturation'] * df['heart_rate'] / 100.0
        
        # Calculate physiological stress index
        if 'heart_rate' in df.columns and 'mean_arterial_pressure' in df.columns:
            # Normalized physiological stress
            hr_norm = (df['heart_rate'] - 60) / 40  # Normalize around resting HR
            bp_norm = (df['mean_arterial_pressure'] - 80) / 20  # Normalize around normal MAP
            
            df['physiological_stress_index'] = (hr_norm + bp_norm) / 2
        
        return df


class AdvancedDataPreprocessor:
    """
    Advanced data preprocessor for migraine prediction.
    
    This class implements advanced preprocessing techniques including
    quantile transformation, feature selection, and data augmentation.
    
    Attributes:
        feature_engineering: Feature engineering component
        quantile_transformer: Quantile transformer for normalization
        feature_selector: Feature selector for selecting most predictive features
        use_smote: Whether to use SMOTE for data augmentation
        random_state: Random state for reproducibility
    """
    
    def __init__(self, window_size: int = 3, n_features_to_select: int = 20,
                use_derivatives: bool = True, use_interactions: bool = True,
                use_temporal_patterns: bool = True, use_smote: bool = True,
                random_state: int = 42):
        """
        Initialize advanced data preprocessor.
        
        Args:
            window_size: Size of the window for temporal features
            n_features_to_select: Number of features to select
            use_derivatives: Whether to use derivatives of features
            use_interactions: Whether to use feature interactions
            use_temporal_patterns: Whether to use temporal pattern features
            use_smote: Whether to use SMOTE for data augmentation
            random_state: Random state for reproducibility
        """
        self.feature_engineering = AdvancedFeatureEngineering(
            window_size=window_size,
            use_derivatives=use_derivatives,
            use_interactions=use_interactions,
            use_temporal_patterns=use_temporal_patterns
        )
        
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal',
            random_state=random_state
        )
        
        self.feature_selector = SelectKBest(
            mutual_info_classif,
            k=n_features_to_select
        )
        
        self.use_smote = use_smote
        self.random_state = random_state
        
        # Fitted flag
        self.fitted = False
        
        # Store column names
        self.feature_names = None
        self.selected_feature_names = None
    
    def fit_transform(self, data: Dict[str, pd.DataFrame], labels: pd.Series) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            data: Dictionary of DataFrames for each modality
            labels: Series of labels
            
        Returns:
            Tuple of (transformed_data, transformed_labels)
        """
        # Apply feature engineering
        engineered_data = {}
        for modality, df in data.items():
            if modality == 'sleep_data':
                engineered_data[modality] = self.feature_engineering.transform_sleep_data(df)
            elif modality == 'weather_data':
                engineered_data[modality] = self.feature_engineering.transform_weather_data(df)
            elif modality == 'stress_diet_data':
                engineered_data[modality] = self.feature_engineering.transform_stress_diet_data(df)
            elif modality == 'physiological_data':
                engineered_data[modality] = self.feature_engineering.transform_physiological_data(df)
            else:
                engineered_data[modality] = df.copy()
        
        # Combine all features for normalization and feature selection
        combined_df = pd.concat(engineered_data.values(), axis=1)
        
        # Store column names
        self.feature_names = combined_df.columns.tolist()
        
        # Apply quantile transformation
        normalized_data = self.quantile_transformer.fit_transform(combined_df)
        
        # Apply feature selection
        selected_data = self.feature_selector.fit_transform(normalized_data, labels)
        
        # Get selected feature indices
        selected_indices = self.feature_selector.get_support(indices=True)
        
        # Store selected feature names
        self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        
        # Apply SMOTE for data augmentation
        if self.use_smote:
            smote = SMOTE(random_state=self.random_state)
            selected_data, labels = smote.fit_resample(selected_data, labels)
        
        # Split back into modalities
        transformed_data = {}
        start_idx = 0
        for modality, df in engineered_data.items():
            end_idx = start_idx + df.shape[1]
            modality_indices = [i for i in selected_indices if start_idx <= i < end_idx]
            
            if modality_indices:
                # Adjust indices to be relative to the modality
                relative_indices = [i - start_idx for i in modality_indices]
                
                # Select features for this modality
                transformed_data[modality] = normalized_data[:, modality_indices]
            else:
                # No features selected for this modality
                transformed_data[modality] = np.zeros((normalized_data.shape[0], 0))
            
            start_idx = end_idx
        
        self.fitted = True
        
        return transformed_data, labels.values
    
    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Transform the data using the fitted preprocessor.
        
        Args:
            data: Dictionary of DataFrames for each modality
            
        Returns:
            Dictionary of transformed data for each modality
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")
        
        # Apply feature engineering
        engineered_data = {}
        for modality, df in data.items():
            if modality == 'sleep_data':
                engineered_data[modality] = self.feature_engineering.transform_sleep_data(df)
            elif modality == 'weather_data':
                engineered_data[modality] = self.feature_engineering.transform_weather_data(df)
            elif modality == 'stress_diet_data':
                engineered_data[modality] = self.feature_engineering.transform_stress_diet_data(df)
            elif modality == 'physiological_data':
                engineered_data[modality] = self.feature_engineering.transform_physiological_data(df)
            else:
                engineered_data[modality] = df.copy()
        
        # Combine all features for normalization and feature selection
        combined_df = pd.concat(engineered_data.values(), axis=1)
        
        # Apply quantile transformation
        normalized_data = self.quantile_transformer.transform(combined_df)
        
        # Apply feature selection
        selected_data = self.feature_selector.transform(normalized_data)
        
        # Get selected feature indices
        selected_indices = self.feature_selector.get_support(indices=True)
        
        # Split back into modalities
        transformed_data = {}
        start_idx = 0
        for modality, df in engineered_data.items():
            end_idx = start_idx + df.shape[1]
            modality_indices = [i for i in selected_indices if start_idx <= i < end_idx]
            
            if modality_indices:
                # Adjust indices to be relative to the modality
                relative_indices = [i - start_idx for i in modality_indices]
                
                # Select features for this modality
                transformed_data[modality] = normalized_data[:, modality_indices]
            else:
                # No features selected for this modality
                transformed_data[modality] = np.zeros((normalized_data.shape[0], 0))
            
            start_idx = end_idx
        
        return transformed_data
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")
        
        # Get feature importance scores
        importance_scores = self.feature_selector.scores_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Mark selected features
        importance_df['selected'] = importance_df['feature'].isin(self.selected_feature_names)
        
        return importance_df


class EnhancedDataLoader:
    """
    Enhanced data loader for migraine prediction.
    
    This class implements an enhanced data loader with support for
    sequential data, batch normalization, and data validation.
    
    Attributes:
        batch_size: Batch size for data loading
        shuffle: Whether to shuffle the data
        sequence_length: Length of sequences for sequential data
        device: Device for PyTorch tensors
    """
    
    def __init__(self, batch_size: int = 32, shuffle: bool = True,
                sequence_length: int = 7, device: torch.device = None):
        """
        Initialize enhanced data loader.
        
        Args:
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle the data
            sequence_length: Length of sequences for sequential data
            device: Device for PyTorch tensors
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_sequences(self, data: Dict[str, np.ndarray], labels: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Create sequences from data.
        
        Args:
            data: Dictionary of arrays for each modality
            labels: Array of labels
            
        Returns:
            Tuple of (sequenced_data, sequenced_labels)
        """
        # Get number of samples
        num_samples = next(iter(data.values())).shape[0]
        
        # Create sequences
        sequenced_data = {}
        for modality, arr in data.items():
            # Initialize sequences
            sequences = []
            
            # Create sequences
            for i in range(num_samples - self.sequence_length + 1):
                sequence = arr[i:i+self.sequence_length]
                sequences.append(sequence)
            
            # Convert to tensor
            sequenced_data[modality] = torch.tensor(
                np.array(sequences),
                dtype=torch.float32
            ).to(self.device)
        
        # Create sequenced labels (use the label at the end of each sequence)
        sequenced_labels = torch.tensor(
            labels[self.sequence_length-1:],
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)
        
        return sequenced_data, sequenced_labels
    
    def create_data_loader(self, data: Dict[str, np.ndarray], labels: np.ndarray,
                          is_sequential: bool = True) -> torch.utils.data.DataLoader:
        """
        Create a data loader.
        
        Args:
            data: Dictionary of arrays for each modality
            labels: Array of labels
            is_sequential: Whether to create sequences
            
        Returns:
            PyTorch DataLoader
        """
        # Create sequences if needed
        if is_sequential:
            data, labels = self.create_sequences(data, labels)
        else:
            # Convert to tensors
            tensor_data = {}
            for modality, arr in data.items():
                tensor_data[modality] = torch.tensor(
                    arr,
                    dtype=torch.float32
                ).to(self.device)
            
            data = tensor_data
            labels = torch.tensor(
                labels,
                dtype=torch.float32
            ).unsqueeze(1).to(self.device)
        
        # Create dataset
        dataset = EnhancedDataset(data, labels)
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        
        return data_loader


class EnhancedDataset(torch.utils.data.Dataset):
    """
    Enhanced dataset for migraine prediction.
    
    Attributes:
        data: Dictionary of tensors for each modality
        labels: Tensor of labels
    """
    
    def __init__(self, data: Dict[str, torch.Tensor], labels: torch.Tensor):
        """
        Initialize enhanced dataset.
        
        Args:
            data: Dictionary of tensors for each modality
            labels: Tensor of labels
        """
        self.data = data
        self.labels = labels
        
        # Validate data
        self._validate_data()
    
    def _validate_data(self):
        """
        Validate the data.
        
        Raises:
            ValueError: If data is invalid
        """
        # Check that all modalities have the same number of samples
        num_samples = self.labels.size(0)
        for modality, tensor in self.data.items():
            if tensor.size(0) != num_samples:
                raise ValueError(
                    f"Modality {modality} has {tensor.size(0)} samples, "
                    f"but labels has {num_samples} samples"
                )
        
        # Check for NaN values
        for modality, tensor in self.data.items():
            if torch.isnan(tensor).any():
                raise ValueError(f"Modality {modality} contains NaN values")
        
        if torch.isnan(self.labels).any():
            raise ValueError("Labels contain NaN values")
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return self.labels.size(0)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (inputs, label)
        """
        # Get inputs for each modality
        inputs = {modality: tensor[idx] for modality, tensor in self.data.items()}
        
        # Get label
        label = self.labels[idx]
        
        return inputs, label
