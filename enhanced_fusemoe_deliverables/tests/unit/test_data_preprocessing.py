"""
Unit tests for data preprocessing components in the Enhanced FuseMoE system.

This module contains unit tests for the data preprocessing components.
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

# Import preprocessing components
from utils.preprocessing.data_generator import (
    SyntheticMigraineDataGenerator,
    generate_sleep_data,
    generate_weather_data,
    generate_stress_diet_data,
    generate_physiological_data
)
from utils.preprocessing.data_preprocessor import (
    MigraineDataPreprocessor,
    normalize_data,
    create_time_series_features,
    create_domain_specific_features
)


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test cases for the SyntheticMigraineDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create data generator
        self.generator = SyntheticMigraineDataGenerator(
            num_samples=100,
            time_periods=7,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.num_samples, 100)
        self.assertEqual(self.generator.time_periods, 7)
        self.assertEqual(self.generator.random_seed, 42)
        
        # Check that random state is initialized
        self.assertIsNotNone(self.generator.random_state)
    
    def test_generate_sleep_data(self):
        """Test generate_sleep_data method."""
        # Generate sleep data
        sleep_data = self.generator.generate_sleep_data()
        
        # Check data shape and type
        self.assertIsInstance(sleep_data, pd.DataFrame)
        self.assertEqual(sleep_data.shape[0], 100)
        
        # Check that required columns are present
        required_columns = [
            'sleep_duration', 'sleep_quality', 'deep_sleep_percentage',
            'rem_sleep_percentage', 'sleep_interruptions'
        ]
        for column in required_columns:
            self.assertIn(column, sleep_data.columns)
        
        # Check that values are within expected ranges
        self.assertTrue((sleep_data['sleep_duration'] >= 0).all())
        self.assertTrue((sleep_data['sleep_duration'] <= 12).all())
        self.assertTrue((sleep_data['sleep_quality'] >= 0).all())
        self.assertTrue((sleep_data['sleep_quality'] <= 10).all())
    
    def test_generate_weather_data(self):
        """Test generate_weather_data method."""
        # Generate weather data
        weather_data = self.generator.generate_weather_data()
        
        # Check data shape and type
        self.assertIsInstance(weather_data, pd.DataFrame)
        self.assertEqual(weather_data.shape[0], 100)
        
        # Check that required columns are present
        required_columns = [
            'temperature', 'humidity', 'pressure', 'precipitation',
            'wind_speed', 'cloud_cover'
        ]
        for column in required_columns:
            self.assertIn(column, weather_data.columns)
        
        # Check that values are within expected ranges
        self.assertTrue((weather_data['temperature'] >= -20).all())
        self.assertTrue((weather_data['temperature'] <= 45).all())
        self.assertTrue((weather_data['humidity'] >= 0).all())
        self.assertTrue((weather_data['humidity'] <= 100).all())
    
    def test_generate_stress_diet_data(self):
        """Test generate_stress_diet_data method."""
        # Generate stress and diet data
        stress_diet_data = self.generator.generate_stress_diet_data()
        
        # Check data shape and type
        self.assertIsInstance(stress_diet_data, pd.DataFrame)
        self.assertEqual(stress_diet_data.shape[0], 100)
        
        # Check that required columns are present
        required_columns = [
            'stress_level', 'caffeine_intake', 'alcohol_consumption',
            'meal_regularity', 'hydration_level', 'exercise_duration'
        ]
        for column in required_columns:
            self.assertIn(column, stress_diet_data.columns)
        
        # Check that values are within expected ranges
        self.assertTrue((stress_diet_data['stress_level'] >= 0).all())
        self.assertTrue((stress_diet_data['stress_level'] <= 10).all())
        self.assertTrue((stress_diet_data['caffeine_intake'] >= 0).all())
    
    def test_generate_physiological_data(self):
        """Test generate_physiological_data method."""
        # Generate physiological data
        physio_data = self.generator.generate_physiological_data()
        
        # Check data shape and type
        self.assertIsInstance(physio_data, pd.DataFrame)
        self.assertEqual(physio_data.shape[0], 100)
        
        # Check that required columns are present
        required_columns = [
            'heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'body_temperature', 'respiratory_rate', 'oxygen_saturation'
        ]
        for column in required_columns:
            self.assertIn(column, physio_data.columns)
        
        # Check that values are within expected ranges
        self.assertTrue((physio_data['heart_rate'] >= 40).all())
        self.assertTrue((physio_data['heart_rate'] <= 180).all())
        self.assertTrue((physio_data['blood_pressure_systolic'] >= 80).all())
        self.assertTrue((physio_data['blood_pressure_systolic'] <= 200).all())
    
    def test_generate_migraine_labels(self):
        """Test generate_migraine_labels method."""
        # Generate domain data
        sleep_data = self.generator.generate_sleep_data()
        weather_data = self.generator.generate_weather_data()
        stress_diet_data = self.generator.generate_stress_diet_data()
        physio_data = self.generator.generate_physiological_data()
        
        # Generate migraine labels
        labels = self.generator.generate_migraine_labels(
            sleep_data=sleep_data,
            weather_data=weather_data,
            stress_diet_data=stress_diet_data,
            physio_data=physio_data
        )
        
        # Check data shape and type
        self.assertIsInstance(labels, pd.Series)
        self.assertEqual(labels.shape[0], 100)
        
        # Check that values are binary
        self.assertTrue(set(labels.unique()).issubset({0, 1}))
    
    def test_generate_dataset(self):
        """Test generate_dataset method."""
        # Generate complete dataset
        dataset = self.generator.generate_dataset()
        
        # Check that all expected components are present
        self.assertIn('sleep_data', dataset)
        self.assertIn('weather_data', dataset)
        self.assertIn('stress_diet_data', dataset)
        self.assertIn('physio_data', dataset)
        self.assertIn('labels', dataset)
        
        # Check that all components have the same number of samples
        self.assertEqual(dataset['sleep_data'].shape[0], 100)
        self.assertEqual(dataset['weather_data'].shape[0], 100)
        self.assertEqual(dataset['stress_diet_data'].shape[0], 100)
        self.assertEqual(dataset['physio_data'].shape[0], 100)
        self.assertEqual(dataset['labels'].shape[0], 100)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for the MigraineDataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data
        generator = SyntheticMigraineDataGenerator(
            num_samples=100,
            time_periods=7,
            random_seed=42
        )
        self.dataset = generator.generate_dataset()
        
        # Create preprocessor
        self.preprocessor = MigraineDataPreprocessor(
            time_periods=7,
            test_size=0.2,
            val_size=0.1,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertEqual(self.preprocessor.time_periods, 7)
        self.assertEqual(self.preprocessor.test_size, 0.2)
        self.assertEqual(self.preprocessor.val_size, 0.1)
        self.assertEqual(self.preprocessor.random_seed, 42)
    
    def test_normalize_data(self):
        """Test normalize_data method."""
        # Normalize sleep data
        sleep_data = self.dataset['sleep_data']
        normalized_data, scaler = self.preprocessor.normalize_data(sleep_data)
        
        # Check data shape and type
        self.assertIsInstance(normalized_data, np.ndarray)
        self.assertEqual(normalized_data.shape[0], 100)
        self.assertEqual(normalized_data.shape[1], sleep_data.shape[1])
        
        # Check that values are normalized (between 0 and 1 or -1 and 1)
        self.assertTrue(np.all(normalized_data >= -1))
        self.assertTrue(np.all(normalized_data <= 1))
        
        # Check that scaler is returned
        self.assertIsNotNone(scaler)
    
    def test_create_time_series_features(self):
        """Test create_time_series_features method."""
        # Create time series features for sleep data
        sleep_data = self.dataset['sleep_data'].values
        time_series_data = self.preprocessor.create_time_series_features(
            data=sleep_data,
            time_periods=3  # Use smaller time periods for testing
        )
        
        # Check data shape
        self.assertEqual(time_series_data.shape[0], 100 - 3 + 1)  # Adjusted for time periods
        self.assertEqual(time_series_data.shape[1], 3)  # Time periods
        self.assertEqual(time_series_data.shape[2], sleep_data.shape[1])  # Features
    
    def test_create_domain_specific_features(self):
        """Test create_domain_specific_features method."""
        # Create domain-specific features
        domain_features = self.preprocessor.create_domain_specific_features(
            sleep_data=self.dataset['sleep_data'],
            weather_data=self.dataset['weather_data'],
            stress_diet_data=self.dataset['stress_diet_data'],
            physio_data=self.dataset['physio_data']
        )
        
        # Check that all domains are present
        self.assertIn('sleep_features', domain_features)
        self.assertIn('weather_features', domain_features)
        self.assertIn('stress_diet_features', domain_features)
        self.assertIn('physio_features', domain_features)
        
        # Check that features have been created
        self.assertGreater(domain_features['sleep_features'].shape[1], 0)
        self.assertGreater(domain_features['weather_features'].shape[1], 0)
        self.assertGreater(domain_features['stress_diet_features'].shape[1], 0)
        self.assertGreater(domain_features['physio_features'].shape[1], 0)
    
    def test_split_data(self):
        """Test split_data method."""
        # Create dummy data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.split_data(X, y)
        
        # Check data shapes
        self.assertEqual(X_train.shape[0], 70)  # 70% train
        self.assertEqual(X_val.shape[0], 10)    # 10% val
        self.assertEqual(X_test.shape[0], 20)   # 20% test
        self.assertEqual(y_train.shape[0], 70)
        self.assertEqual(y_val.shape[0], 10)
        self.assertEqual(y_test.shape[0], 20)
    
    def test_prepare_data_for_experts(self):
        """Test prepare_data_for_experts method."""
        # Prepare data for experts
        expert_data, labels = self.preprocessor.prepare_data_for_experts(
            sleep_data=self.dataset['sleep_data'],
            weather_data=self.dataset['weather_data'],
            stress_diet_data=self.dataset['stress_diet_data'],
            physio_data=self.dataset['physio_data'],
            labels=self.dataset['labels']
        )
        
        # Check that all expert data is present
        self.assertIn('sleep_expert', expert_data)
        self.assertIn('weather_expert', expert_data)
        self.assertIn('stress_diet_expert', expert_data)
        self.assertIn('physio_expert', expert_data)
        
        # Check that data has been prepared with time series features
        expected_shape = (100 - self.preprocessor.time_periods + 1, 
                         self.preprocessor.time_periods, 
                         self.dataset['sleep_data'].shape[1])
        self.assertEqual(expert_data['sleep_expert'].shape, expected_shape)
        
        # Check that labels have been adjusted for time series
        self.assertEqual(labels.shape[0], 100 - self.preprocessor.time_periods + 1)
    
    def test_create_data_loaders(self):
        """Test create_data_loaders method."""
        # Create dummy expert data
        expert_data = {
            'sleep_expert': np.random.rand(100, 7, 5),
            'weather_expert': np.random.rand(100, 7, 6),
            'stress_diet_expert': np.random.rand(100, 7, 6),
            'physio_expert': np.random.rand(100, 7, 6)
        }
        labels = np.random.randint(0, 2, 100)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.preprocessor.create_data_loaders(
            expert_data=expert_data,
            labels=labels,
            batch_size=16
        )
        
        # Check that data loaders are created
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Check batch size
        for X, y in train_loader:
            self.assertLessEqual(X[0].shape[0], 16)  # Batch size
            self.assertEqual(len(X), 4)  # 4 experts
            break


if __name__ == '__main__':
    unittest.main()
