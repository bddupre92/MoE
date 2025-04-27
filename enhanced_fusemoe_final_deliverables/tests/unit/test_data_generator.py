"""
Unit tests for the data generator in the Enhanced FuseMoE system.

This module contains unit tests for the synthetic data generation functions.
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

# Import data generator
from utils.preprocessing.data_generator import (
    MigraineSyntheticDataGenerator,
    generate_sleep_data,
    generate_weather_data,
    generate_stress_diet_data,
    generate_physiological_data,
    generate_migraine_labels
)


class TestMigraineSyntheticDataGenerator(unittest.TestCase):
    """Test cases for the MigraineSyntheticDataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.seed = 42
        self.generator = MigraineSyntheticDataGenerator(seed=self.seed)
        
        # Create temporary directory for output
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_output')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.seed, self.seed)
        self.assertIsNotNone(self.generator.sleep_params)
        self.assertIsNotNone(self.generator.weather_params)
        self.assertIsNotNone(self.generator.stress_diet_params)
        self.assertIsNotNone(self.generator.physio_params)
        self.assertIsNotNone(self.generator.migraine_params)
    
    def test_generate_sleep_data(self):
        """Test sleep data generation."""
        # Generate data
        sleep_data = self.generator.generate_sleep_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(sleep_data, pd.DataFrame)
        self.assertEqual(len(sleep_data), self.num_samples)
        
        # Check columns
        expected_columns = ['sleep_duration', 'sleep_quality', 'deep_sleep_percentage', 
                           'rem_sleep_percentage', 'sleep_interruptions', 'time_to_sleep']
        for col in expected_columns:
            self.assertIn(col, sleep_data.columns)
        
        # Check value ranges
        self.assertTrue((sleep_data['sleep_duration'] >= 0).all())
        self.assertTrue((sleep_data['sleep_duration'] <= 12).all())
        self.assertTrue((sleep_data['sleep_quality'] >= 0).all())
        self.assertTrue((sleep_data['sleep_quality'] <= 10).all())
        self.assertTrue((sleep_data['deep_sleep_percentage'] >= 0).all())
        self.assertTrue((sleep_data['deep_sleep_percentage'] <= 100).all())
        self.assertTrue((sleep_data['rem_sleep_percentage'] >= 0).all())
        self.assertTrue((sleep_data['rem_sleep_percentage'] <= 100).all())
        self.assertTrue((sleep_data['sleep_interruptions'] >= 0).all())
        self.assertTrue((sleep_data['time_to_sleep'] >= 0).all())
    
    def test_generate_weather_data(self):
        """Test weather data generation."""
        # Generate data
        weather_data = self.generator.generate_weather_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(weather_data, pd.DataFrame)
        self.assertEqual(len(weather_data), self.num_samples)
        
        # Check columns
        expected_columns = ['temperature', 'humidity', 'pressure', 'precipitation', 'wind_speed']
        for col in expected_columns:
            self.assertIn(col, weather_data.columns)
        
        # Check value ranges
        self.assertTrue((weather_data['humidity'] >= 0).all())
        self.assertTrue((weather_data['humidity'] <= 100).all())
        self.assertTrue((weather_data['pressure'] >= 900).all())
        self.assertTrue((weather_data['pressure'] <= 1100).all())
        self.assertTrue((weather_data['precipitation'] >= 0).all())
        self.assertTrue((weather_data['wind_speed'] >= 0).all())
    
    def test_generate_stress_diet_data(self):
        """Test stress and diet data generation."""
        # Generate data
        stress_diet_data = self.generator.generate_stress_diet_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(stress_diet_data, pd.DataFrame)
        self.assertEqual(len(stress_diet_data), self.num_samples)
        
        # Check columns
        expected_columns = ['stress_level', 'caffeine_intake', 'alcohol_intake', 
                           'meal_regularity', 'hydration_level', 'exercise_duration']
        for col in expected_columns:
            self.assertIn(col, stress_diet_data.columns)
        
        # Check value ranges
        self.assertTrue((stress_diet_data['stress_level'] >= 0).all())
        self.assertTrue((stress_diet_data['stress_level'] <= 10).all())
        self.assertTrue((stress_diet_data['caffeine_intake'] >= 0).all())
        self.assertTrue((stress_diet_data['alcohol_intake'] >= 0).all())
        self.assertTrue((stress_diet_data['meal_regularity'] >= 0).all())
        self.assertTrue((stress_diet_data['meal_regularity'] <= 10).all())
        self.assertTrue((stress_diet_data['hydration_level'] >= 0).all())
        self.assertTrue((stress_diet_data['hydration_level'] <= 10).all())
        self.assertTrue((stress_diet_data['exercise_duration'] >= 0).all())
    
    def test_generate_physiological_data(self):
        """Test physiological data generation."""
        # Generate data
        physio_data = self.generator.generate_physiological_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(physio_data, pd.DataFrame)
        self.assertEqual(len(physio_data), self.num_samples)
        
        # Check columns
        expected_columns = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                           'body_temperature', 'respiratory_rate']
        for col in expected_columns:
            self.assertIn(col, physio_data.columns)
        
        # Check value ranges
        self.assertTrue((physio_data['heart_rate'] >= 40).all())
        self.assertTrue((physio_data['heart_rate'] <= 200).all())
        self.assertTrue((physio_data['blood_pressure_systolic'] >= 80).all())
        self.assertTrue((physio_data['blood_pressure_systolic'] <= 200).all())
        self.assertTrue((physio_data['blood_pressure_diastolic'] >= 40).all())
        self.assertTrue((physio_data['blood_pressure_diastolic'] <= 120).all())
        self.assertTrue((physio_data['body_temperature'] >= 35).all())
        self.assertTrue((physio_data['body_temperature'] <= 42).all())
        self.assertTrue((physio_data['respiratory_rate'] >= 8).all())
        self.assertTrue((physio_data['respiratory_rate'] <= 30).all())
    
    def test_generate_migraine_labels(self):
        """Test migraine label generation."""
        # Generate data
        sleep_data = self.generator.generate_sleep_data(self.num_samples)
        weather_data = self.generator.generate_weather_data(self.num_samples)
        stress_diet_data = self.generator.generate_stress_diet_data(self.num_samples)
        physio_data = self.generator.generate_physiological_data(self.num_samples)
        
        # Generate labels
        labels = self.generator.generate_migraine_labels(
            sleep_data, weather_data, stress_diet_data, physio_data
        )
        
        # Check data type and shape
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), self.num_samples)
        
        # Check value range
        self.assertTrue(np.all(np.logical_or(labels == 0, labels == 1)))
        
        # Check distribution (should have some of each class)
        self.assertTrue(np.sum(labels) > 0)
        self.assertTrue(np.sum(labels) < self.num_samples)
    
    def test_generate_dataset(self):
        """Test complete dataset generation."""
        # Generate dataset
        data = self.generator.generate_dataset(self.num_samples)
        
        # Check data type
        self.assertIsInstance(data, dict)
        
        # Check keys
        expected_keys = ['sleep', 'weather', 'stress_diet', 'physio', 'labels']
        for key in expected_keys:
            self.assertIn(key, data)
        
        # Check shapes
        self.assertEqual(len(data['sleep']), self.num_samples)
        self.assertEqual(len(data['weather']), self.num_samples)
        self.assertEqual(len(data['stress_diet']), self.num_samples)
        self.assertEqual(len(data['physio']), self.num_samples)
        self.assertEqual(len(data['labels']), self.num_samples)
    
    def test_save_and_load_dataset(self):
        """Test dataset saving and loading."""
        # Generate dataset
        data = self.generator.generate_dataset(self.num_samples)
        
        # Save dataset
        save_path = os.path.join(self.temp_dir, 'test_dataset.pkl')
        self.generator.save_dataset(data, save_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(save_path))
        
        # Load dataset
        loaded_data = self.generator.load_dataset(save_path)
        
        # Check that loaded data matches original
        self.assertEqual(len(loaded_data), len(data))
        for key in data:
            if isinstance(data[key], pd.DataFrame):
                pd.testing.assert_frame_equal(loaded_data[key], data[key])
            else:
                np.testing.assert_array_equal(loaded_data[key], data[key])
    
    def test_create_torch_datasets(self):
        """Test torch dataset creation."""
        # Generate dataset
        data = self.generator.generate_dataset(self.num_samples)
        
        # Create torch datasets
        train_dataset, val_dataset, test_dataset = self.generator.create_torch_datasets(
            data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Check dataset types
        self.assertIsInstance(train_dataset, torch.utils.data.Dataset)
        self.assertIsInstance(val_dataset, torch.utils.data.Dataset)
        self.assertIsInstance(test_dataset, torch.utils.data.Dataset)
        
        # Check dataset sizes
        self.assertAlmostEqual(len(train_dataset) / self.num_samples, 0.7, delta=0.01)
        self.assertAlmostEqual(len(val_dataset) / self.num_samples, 0.15, delta=0.01)
        self.assertAlmostEqual(len(test_dataset) / self.num_samples, 0.15, delta=0.01)
        
        # Check dataset item format
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


# Test standalone functions
class TestStandaloneFunctions(unittest.TestCase):
    """Test cases for standalone data generation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 50
        self.seed = 42
        np.random.seed(self.seed)
    
    def test_generate_sleep_data(self):
        """Test standalone sleep data generation."""
        # Generate data
        sleep_data = generate_sleep_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(sleep_data, pd.DataFrame)
        self.assertEqual(len(sleep_data), self.num_samples)
        
        # Check columns
        expected_columns = ['sleep_duration', 'sleep_quality', 'deep_sleep_percentage', 
                           'rem_sleep_percentage', 'sleep_interruptions', 'time_to_sleep']
        for col in expected_columns:
            self.assertIn(col, sleep_data.columns)
    
    def test_generate_weather_data(self):
        """Test standalone weather data generation."""
        # Generate data
        weather_data = generate_weather_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(weather_data, pd.DataFrame)
        self.assertEqual(len(weather_data), self.num_samples)
        
        # Check columns
        expected_columns = ['temperature', 'humidity', 'pressure', 'precipitation', 'wind_speed']
        for col in expected_columns:
            self.assertIn(col, weather_data.columns)
    
    def test_generate_stress_diet_data(self):
        """Test standalone stress and diet data generation."""
        # Generate data
        stress_diet_data = generate_stress_diet_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(stress_diet_data, pd.DataFrame)
        self.assertEqual(len(stress_diet_data), self.num_samples)
        
        # Check columns
        expected_columns = ['stress_level', 'caffeine_intake', 'alcohol_intake', 
                           'meal_regularity', 'hydration_level', 'exercise_duration']
        for col in expected_columns:
            self.assertIn(col, stress_diet_data.columns)
    
    def test_generate_physiological_data(self):
        """Test standalone physiological data generation."""
        # Generate data
        physio_data = generate_physiological_data(self.num_samples)
        
        # Check data type and shape
        self.assertIsInstance(physio_data, pd.DataFrame)
        self.assertEqual(len(physio_data), self.num_samples)
        
        # Check columns
        expected_columns = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                           'body_temperature', 'respiratory_rate']
        for col in expected_columns:
            self.assertIn(col, physio_data.columns)
    
    def test_generate_migraine_labels(self):
        """Test standalone migraine label generation."""
        # Generate data
        sleep_data = generate_sleep_data(self.num_samples)
        weather_data = generate_weather_data(self.num_samples)
        stress_diet_data = generate_stress_diet_data(self.num_samples)
        physio_data = generate_physiological_data(self.num_samples)
        
        # Generate labels
        labels = generate_migraine_labels(
            sleep_data, weather_data, stress_diet_data, physio_data
        )
        
        # Check data type and shape
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), self.num_samples)
        
        # Check value range
        self.assertTrue(np.all(np.logical_or(labels == 0, labels == 1)))


if __name__ == '__main__':
    unittest.main()
