"""
Unit tests for the data preprocessor in the Enhanced FuseMoE system.

This module contains unit tests for the data preprocessing functions.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import data preprocessor
from utils.preprocessing.data_preprocessor import (
    MigraineDataPreprocessor,
    normalize_dataframe,
    encode_categorical_features,
    handle_missing_values,
    create_feature_vector
)


class TestMigraineDataPreprocessor(unittest.TestCase):
    """Test cases for the MigraineDataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sleep_data = pd.DataFrame({
            'sleep_duration': [7.5, 6.2, 8.1, 5.5, 7.0],
            'sleep_quality': [8, 6, 9, 4, 7],
            'deep_sleep_percentage': [25, 20, 30, 15, 22],
            'rem_sleep_percentage': [20, 18, 25, 15, 19],
            'sleep_interruptions': [2, 3, 1, 4, 2],
            'time_to_sleep': [15, 30, 10, 45, 20]
        })
        
        self.weather_data = pd.DataFrame({
            'temperature': [25.5, 28.2, 22.1, 30.5, 24.0],
            'humidity': [65, 70, 60, 75, 68],
            'pressure': [1012, 1008, 1015, 1005, 1010],
            'precipitation': [0, 5, 0, 10, 2],
            'wind_speed': [10, 15, 8, 20, 12]
        })
        
        self.stress_diet_data = pd.DataFrame({
            'stress_level': [6, 8, 4, 9, 5],
            'caffeine_intake': [200, 300, 100, 350, 150],
            'alcohol_intake': [1, 2, 0, 3, 1],
            'meal_regularity': [7, 5, 8, 4, 6],
            'hydration_level': [8, 6, 9, 5, 7],
            'exercise_duration': [30, 15, 45, 0, 20]
        })
        
        self.physio_data = pd.DataFrame({
            'heart_rate': [72, 80, 68, 85, 75],
            'blood_pressure_systolic': [120, 130, 115, 140, 125],
            'blood_pressure_diastolic': [80, 85, 75, 90, 82],
            'body_temperature': [36.6, 36.8, 36.5, 37.0, 36.7],
            'respiratory_rate': [14, 16, 13, 18, 15]
        })
        
        self.labels = np.array([0, 1, 0, 1, 0])
        
        # Create dataset
        self.dataset = {
            'sleep': self.sleep_data,
            'weather': self.weather_data,
            'stress_diet': self.stress_diet_data,
            'physio': self.physio_data,
            'labels': self.labels
        }
        
        # Create preprocessor
        self.preprocessor = MigraineDataPreprocessor()
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsNone(self.preprocessor.scaler_dict)
        self.assertIsNone(self.preprocessor.encoder_dict)
        self.assertFalse(self.preprocessor.is_fitted)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        # Fit and transform data
        processed_data = self.preprocessor.fit_transform(self.dataset)
        
        # Check that preprocessor is fitted
        self.assertTrue(self.preprocessor.is_fitted)
        self.assertIsNotNone(self.preprocessor.scaler_dict)
        
        # Check processed data
        self.assertIsInstance(processed_data, dict)
        self.assertIn('sleep', processed_data)
        self.assertIn('weather', processed_data)
        self.assertIn('stress_diet', processed_data)
        self.assertIn('physio', processed_data)
        self.assertIn('labels', processed_data)
        
        # Check that data is normalized
        for key in ['sleep', 'weather', 'stress_diet', 'physio']:
            df = processed_data[key]
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), len(self.dataset[key]))
            
            # Check that values are normalized (between -3 and 3 for most features)
            for col in df.columns:
                self.assertTrue(df[col].min() >= -5)
                self.assertTrue(df[col].max() <= 5)
    
    def test_transform(self):
        """Test transform method."""
        # Fit preprocessor
        self.preprocessor.fit_transform(self.dataset)
        
        # Transform new data
        new_data = {
            'sleep': self.sleep_data.copy(),
            'weather': self.weather_data.copy(),
            'stress_diet': self.stress_diet_data.copy(),
            'physio': self.physio_data.copy(),
            'labels': self.labels.copy()
        }
        
        processed_data = self.preprocessor.transform(new_data)
        
        # Check processed data
        self.assertIsInstance(processed_data, dict)
        for key in ['sleep', 'weather', 'stress_diet', 'physio']:
            self.assertIsInstance(processed_data[key], pd.DataFrame)
            self.assertEqual(len(processed_data[key]), len(new_data[key]))
    
    def test_transform_without_fit(self):
        """Test transform method without fitting first."""
        # Try to transform without fitting
        with self.assertRaises(ValueError):
            self.preprocessor.transform(self.dataset)
    
    def test_create_torch_datasets(self):
        """Test create_torch_datasets method."""
        # Fit and transform data
        processed_data = self.preprocessor.fit_transform(self.dataset)
        
        # Create torch datasets
        train_dataset, val_dataset, test_dataset = self.preprocessor.create_torch_datasets(
            processed_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Check dataset types
        self.assertIsInstance(train_dataset, torch.utils.data.Dataset)
        self.assertIsInstance(val_dataset, torch.utils.data.Dataset)
        self.assertIsInstance(test_dataset, torch.utils.data.Dataset)
        
        # Check dataset sizes
        total_samples = len(self.dataset['sleep'])
        self.assertEqual(len(train_dataset) + len(val_dataset) + len(test_dataset), total_samples)
        
        # Check dataset item format
        if len(train_dataset) > 0:
            train_item = train_dataset[0]
            self.assertIsInstance(train_item, tuple)
            self.assertEqual(len(train_item), 2)
            
            # Check input format
            inputs, target = train_item
            self.assertIsInstance(inputs, dict)
            self.assertIn('sleep', inputs)
            self.assertIn('weather', inputs)
            self.assertIn('stress_diet', inputs)
            self.assertIn('physio', inputs)
            
            # Check target format
            self.assertIsInstance(target, torch.Tensor)
            self.assertEqual(target.shape, (1,))
    
    def test_create_dataloaders(self):
        """Test create_dataloaders method."""
        # Fit and transform data
        processed_data = self.preprocessor.fit_transform(self.dataset)
        
        # Create torch datasets
        train_dataset, val_dataset, test_dataset = self.preprocessor.create_torch_datasets(
            processed_data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Create dataloaders
        batch_size = 2
        train_loader, val_loader, test_loader = self.preprocessor.create_dataloaders(
            train_dataset, val_dataset, test_dataset, batch_size=batch_size
        )
        
        # Check dataloader types
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Check batch size
        self.assertEqual(train_loader.batch_size, batch_size)
        self.assertEqual(val_loader.batch_size, batch_size)
        self.assertEqual(test_loader.batch_size, batch_size)


# Test standalone functions
class TestStandaloneFunctions(unittest.TestCase):
    """Test cases for standalone preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.numeric_df = pd.DataFrame({
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [1.5, 2.5, 3.5, 4.5, 5.5],
            'feature3': [100, 200, 300, 400, 500]
        })
        
        self.categorical_df = pd.DataFrame({
            'feature1': ['A', 'B', 'A', 'C', 'B'],
            'feature2': [1, 2, 1, 3, 2],
            'feature3': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        self.missing_df = pd.DataFrame({
            'feature1': [10, np.nan, 30, 40, 50],
            'feature2': [1.5, 2.5, np.nan, 4.5, 5.5],
            'feature3': [100, 200, 300, np.nan, 500]
        })
    
    def test_normalize_dataframe(self):
        """Test normalize_dataframe function."""
        # Normalize dataframe
        normalized_df, scaler = normalize_dataframe(self.numeric_df)
        
        # Check output types
        self.assertIsInstance(normalized_df, pd.DataFrame)
        self.assertIsInstance(scaler, dict)
        
        # Check that dataframe has same shape
        self.assertEqual(normalized_df.shape, self.numeric_df.shape)
        
        # Check that values are normalized
        for col in normalized_df.columns:
            self.assertAlmostEqual(normalized_df[col].mean(), 0, delta=0.1)
            self.assertAlmostEqual(normalized_df[col].std(), 1, delta=0.1)
    
    def test_encode_categorical_features(self):
        """Test encode_categorical_features function."""
        # Encode categorical features
        encoded_df, encoder = encode_categorical_features(self.categorical_df)
        
        # Check output types
        self.assertIsInstance(encoded_df, pd.DataFrame)
        self.assertIsInstance(encoder, dict)
        
        # Check that dataframe has same number of rows
        self.assertEqual(len(encoded_df), len(self.categorical_df))
        
        # Check that categorical columns are encoded
        self.assertGreater(len(encoded_df.columns), len(self.categorical_df.columns))
        
        # Check that encoder contains mapping for each categorical column
        self.assertIn('feature1', encoder)
        self.assertIn('feature3', encoder)
    
    def test_handle_missing_values(self):
        """Test handle_missing_values function."""
        # Handle missing values
        filled_df = handle_missing_values(self.missing_df)
        
        # Check output type
        self.assertIsInstance(filled_df, pd.DataFrame)
        
        # Check that dataframe has same shape
        self.assertEqual(filled_df.shape, self.missing_df.shape)
        
        # Check that there are no missing values
        self.assertEqual(filled_df.isna().sum().sum(), 0)
    
    def test_create_feature_vector(self):
        """Test create_feature_vector function."""
        # Create feature vector
        feature_vector = create_feature_vector(self.numeric_df)
        
        # Check output type
        self.assertIsInstance(feature_vector, np.ndarray)
        
        # Check shape
        self.assertEqual(feature_vector.shape, (len(self.numeric_df), len(self.numeric_df.columns)))


if __name__ == '__main__':
    unittest.main()
