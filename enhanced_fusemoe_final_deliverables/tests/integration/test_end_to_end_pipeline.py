"""
Integration tests for the end-to-end pipeline of the Enhanced FuseMoE system.

This module provides comprehensive tests for the entire pipeline of the Enhanced FuseMoE
system, including data processing, model training, optimization, and evaluation.
"""

import unittest
import numpy as np
import pandas as pd
import torch
import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import modules to test
from utils.preprocessing.data_generator import MigraineDataGenerator
from utils.preprocessing.data_preprocessor import DataPreprocessor
from utils.training_pipeline import MigraineTrainer, PyGMOTrainingPipeline
from utils.evaluation.metrics import MigrainePerformanceMetrics, SimpleMetrics
from optimization.evolutionary_algorithms.optimization_manager import OptimizationManager
from optimization.pygmo_model_optimizer import PyGMOModelOptimizer
from models.experts.expert_registry import ExpertRegistry
from models.experts.sleep_expert import create_sleep_expert
from models.experts.weather_expert import create_weather_expert
from models.experts.stress_diet_expert import create_stress_diet_expert
from models.experts.physio_expert import create_physio_expert
from models.experts.input_adapter import wrap_expert
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusionMoE


class TestEndToEndPipeline(unittest.TestCase):
    """
    Test case for the end-to-end pipeline of the Enhanced FuseMoE system.
    
    This class tests the entire pipeline of the Enhanced FuseMoE system, including
    data generation, preprocessing, model training, optimization, and evaluation.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test fixtures that are used for all tests.
        """
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test fixtures that were used for all tests.
        """
        # Remove temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """
        Set up test fixtures that are used for each test.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create data generator
        self.data_generator = MigraineDataGenerator(
            seed=42,
            num_patients=10,
            days_per_patient=10,
            migraine_probability=0.2
        )
        
        # Generate data
        self.data = self.data_generator.generate_data()
        
        # Create preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Create expert registry
        self.expert_registry = ExpertRegistry()
        
        # Create expert models
        self.sleep_expert = create_sleep_expert(
            input_dim=self.data['X_train'][0].shape[-1],
            hidden_dim=64,
            output_dim=32
        )
        
        self.weather_expert = create_weather_expert(
            input_dim=self.data['X_train'][1].shape[-1],
            hidden_dim=64,
            output_dim=32
        )
        
        self.stress_diet_expert = create_stress_diet_expert(
            input_dim=self.data['X_train'][2].shape[-1],
            hidden_dim=64,
            output_dim=32
        )
        
        self.physio_expert = create_physio_expert(
            input_dim=self.data['X_train'][3].shape[-1],
            hidden_dim=64,
            output_dim=32
        )
        
        # Wrap expert models
        self.wrapped_sleep_expert = wrap_expert(self.sleep_expert, "sleep_expert")
        self.wrapped_weather_expert = wrap_expert(self.weather_expert, "weather_expert")
        self.wrapped_stress_diet_expert = wrap_expert(self.stress_diet_expert, "stress_diet_expert")
        self.wrapped_physio_expert = wrap_expert(self.physio_expert, "physio_expert")
        
        # Register experts
        self.expert_registry.register_expert("sleep_expert", self.wrapped_sleep_expert)
        self.expert_registry.register_expert("weather_expert", self.wrapped_weather_expert)
        self.expert_registry.register_expert("stress_diet_expert", self.wrapped_stress_diet_expert)
        self.expert_registry.register_expert("physio_expert", self.wrapped_physio_expert)
        
        # Create gating network
        self.gating = MigraineGating(
            input_dim=self.data['X_train'][0].shape[-1],
            num_experts=len(self.expert_registry),
            hidden_dim=64,
            top_k=2
        )
        
        # Create fusion model
        self.model = MigraineFusionMoE(
            expert_registry=self.expert_registry,
            gating=self.gating,
            output_dim=1
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Create criterion
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def test_data_generation_and_preprocessing(self):
        """
        Test data generation and preprocessing.
        """
        # Check that data was generated
        self.assertIn('X_train', self.data)
        self.assertIn('y_train', self.data)
        self.assertIn('X_val', self.data)
        self.assertIn('y_val', self.data)
        self.assertIn('X_test', self.data)
        self.assertIn('y_test', self.data)
        
        # Check data shapes
        self.assertEqual(len(self.data['X_train']), 4)  # 4 expert inputs
        self.assertEqual(self.data['X_train'][0].shape[0], 70)  # 70% of 100 samples
        self.assertEqual(self.data['X_val'][0].shape[0], 15)    # 15% of 100 samples
        self.assertEqual(self.data['X_test'][0].shape[0], 15)   # 15% of 100 samples
        
        # Preprocess data
        preprocessed_data = self.preprocessor.preprocess(self.data)
        
        # Check that preprocessing didn't change data structure
        self.assertIn('X_train', preprocessed_data)
        self.assertIn('y_train', preprocessed_data)
        self.assertIn('X_val', preprocessed_data)
        self.assertIn('y_val', preprocessed_data)
        self.assertIn('X_test', preprocessed_data)
        self.assertIn('y_test', preprocessed_data)
    
    def test_model_forward_pass(self):
        """
        Test model forward pass.
        """
        # Get batch of data
        X_batch = [x[:5].to(self.device) for x in self.data['X_train']]  # First 5 samples of each expert input
        
        # Create input dictionary for model
        input_dict = {
            "sleep_expert": X_batch[0],
            "weather_expert": X_batch[1],
            "stress_diet_expert": X_batch[2],
            "physio_expert": X_batch[3]
        }
        
        # Forward pass
        output, _ = self.model(input_dict)
        
        # Check output shape
        self.assertEqual(output.shape[0], 5)  # Batch size
        self.assertEqual(output.shape[1], 1)  # Output dimension
    
    def test_model_training(self):
        """
        Test model training for a single epoch.
        """
        # Create data loaders with all expert inputs
        train_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_train']],
            targets=self.data['y_train'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        val_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_val']],
            targets=self.data['y_val'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Create trainer
        trainer = MigraineTrainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
        
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Check that metrics were computed
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
    
    @patch('pygmo.algorithm')
    @patch('pygmo.population')
    def test_pygmo_optimization(self, mock_population, mock_algorithm):
        """
        Test PyGMO optimization.
        """
        # Mock PyGMO components
        mock_algorithm.return_value = MagicMock()
        mock_population.return_value = MagicMock()
        
        # Create dataset with all expert inputs
        train_dataset = CustomDataset(
            expert_inputs=self.data['X_train'],
            targets=self.data['y_train'],
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        val_dataset = CustomDataset(
            expert_inputs=self.data['X_val'],
            targets=self.data['y_val'],
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        # Create optimizer
        optimizer = PyGMOModelOptimizer(
            model=self.model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=self.device,
            output_dir=os.path.join(self.temp_dir, 'optimization')
        )
        
        # Define parameter space
        param_space = {
            'learning_rate': (0.0001, 0.01),
            'batch_size': (8, 32),
            'hidden_dim': (32, 128)
        }
        
        # Run optimization (mocked)
        with patch.object(optimizer, '_evaluate_model', return_value=0.9):
            best_params, best_score = optimizer.optimize(
                param_space=param_space,
                population_size=10,
                generations=5,
                algorithm='sade',
                seed=42
            )
        
        # Check that optimization returned results
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)
    
    def test_performance_metrics(self):
        """
        Test performance metrics calculation.
        """
        # Create data loaders with all expert inputs
        test_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_test']],
            targets=self.data['y_test'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Create metrics calculator
        metrics_calculator = MigrainePerformanceMetrics(
            model=self.model,
            device=self.device
        )
        
        # Calculate metrics with mocked model forward pass
        with patch.object(self.model, 'forward', return_value=(torch.sigmoid(torch.randn(15, 1).to(self.device)), None)):
            metrics = metrics_calculator.calculate_metrics(
                test_loader,
                output_dir=os.path.join(self.temp_dir, 'metrics')
            )
        
        # Check that metrics were computed
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check that predictions file was created
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'metrics', 'test_predictions.npz')))
        
        # Test SimpleMetrics
        simple_metrics = SimpleMetrics.calculate_from_file(
            os.path.join(self.temp_dir, 'metrics', 'test_predictions.npz')
        )
        
        # Check that metrics were computed
        self.assertIn('accuracy', simple_metrics)
        self.assertIn('precision', simple_metrics)
        self.assertIn('recall', simple_metrics)
        self.assertIn('f1', simple_metrics)
    
    @unittest.skip("Full end-to-end test is time-consuming")
    def test_full_end_to_end_pipeline(self):
        """
        Test the full end-to-end pipeline.
        
        This test is skipped by default as it is time-consuming.
        """
        # Create data loaders with all expert inputs
        train_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_train']],
            targets=self.data['y_train'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        val_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_val']],
            targets=self.data['y_val'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        test_dataset = CustomDataset(
            expert_inputs=[x.to(self.device) for x in self.data['X_test']],
            targets=self.data['y_test'].to(self.device),
            expert_names=["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16,
            shuffle=False
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Create trainer
        trainer = MigraineTrainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            checkpoint_dir=os.path.join(self.temp_dir, 'checkpoints')
        )
        
        # Train for multiple epochs
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
            patience=2
        )
        
        # Load best model
        trainer.load_checkpoint(trainer.best_model_path)
        
        # Create metrics calculator
        metrics_calculator = MigrainePerformanceMetrics(
            model=self.model,
            device=self.device
        )
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_metrics(
            test_loader,
            output_dir=os.path.join(self.temp_dir, 'metrics')
        )
        
        # Check that metrics were computed
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)


class CustomDataset(torch.utils.data.Dataset):
    """
    Custom dataset for the Enhanced FuseMoE system.
    
    This dataset formats the data for the MigraineFusionMoE model, which expects
    a dictionary mapping expert names to input tensors.
    """
    
    def __init__(self, expert_inputs, targets, expert_names):
        """
        Initialize the custom dataset.
        
        Args:
            expert_inputs: List of input tensors for each expert
            targets: Target tensor
            expert_names: List of expert names
        """
        self.expert_inputs = expert_inputs
        self.targets = targets
        self.expert_names = expert_names
        
        # Ensure all inputs have the same number of samples
        self.num_samples = len(targets)
        for inputs in expert_inputs:
            assert len(inputs) == self.num_samples, "All inputs must have the same number of samples"
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple containing:
                - Dictionary mapping expert names to input tensors
                - Target tensor
        """
        # Create input dictionary
        inputs = {
            name: self.expert_inputs[i][idx]
            for i, name in enumerate(self.expert_names)
        }
        
        # Get target
        target = self.targets[idx]
        
        return inputs, target


class TestEndToEndPipelineWithMocks(unittest.TestCase):
    """
    Test case for the end-to-end pipeline of the Enhanced FuseMoE system with mocks.
    
    This class tests the end-to-end pipeline with mocked components to isolate
    specific functionality and avoid dependencies on other components.
    """
    
    def setUp(self):
        """
        Set up test fixtures that are used for each test.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create mocked model
        self.model = MagicMock()
        self.model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        
        # Create mocked optimizer
        self.optimizer = MagicMock()
    
    def tearDown(self):
        """
        Clean up test fixtures that were used for each test.
        """
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_metrics_calculation_with_mocks(self):
        """
        Test metrics calculation with mocked components.
        """
        # Create mocked data
        y_true = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32).unsqueeze(1)
        y_pred = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float32).unsqueeze(1)
        
        # Save predictions to file
        os.makedirs(os.path.join(self.temp_dir, 'metrics'), exist_ok=True)
        np.savez(
            os.path.join(self.temp_dir, 'metrics', 'test_predictions.npz'),
            y_test=y_true.numpy(),
            y_pred_test=y_pred.numpy()
        )
        
        # Calculate metrics
        metrics = SimpleMetrics.calculate_from_file(
            os.path.join(self.temp_dir, 'metrics', 'test_predictions.npz')
        )
        
        # Check that metrics were computed
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
    
    @patch('pygmo.algorithm')
    @patch('pygmo.population')
    def test_optimization_manager(self, mock_population, mock_algorithm):
        """
        Test optimization manager with mocked components.
        """
        # Mock PyGMO components
        mock_algorithm.return_value = MagicMock()
        mock_population.return_value = MagicMock()
        
        # Create optimization manager
        optimization_manager = OptimizationManager(
            model=self.model,
            device=self.device,
            output_dir=os.path.join(self.temp_dir, 'optimization')
        )
        
        # Define parameter space
        param_space = {
            'learning_rate': (0.0001, 0.01),
            'batch_size': (8, 32),
            'hidden_dim': (32, 128)
        }
        
        # Mock fitness function
        fitness_func = MagicMock(return_value=0.9)
        
        # Run optimization
        best_params, best_score = optimization_manager.optimize(
            param_space=param_space,
            fitness_func=fitness_func,
            population_size=10,
            generations=5,
            algorithm='sade',
            seed=42
        )
        
        # Check that optimization returned results
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)
    
    @patch('pygmo.algorithm')
    @patch('pygmo.population')
    def test_pygmo_training_pipeline(self, mock_population, mock_algorithm):
        """
        Test PyGMO training pipeline with mocked components.
        """
        # Mock PyGMO components
        mock_algorithm.return_value = MagicMock()
        mock_population.return_value = MagicMock()
        
        # Create mocked data
        X_train = torch.randn(10, 5)
        y_train = torch.randint(0, 2, (10, 1), dtype=torch.float32)
        X_val = torch.randn(5, 5)
        y_val = torch.randint(0, 2, (5, 1), dtype=torch.float32)
        
        # Create mocked datasets
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 10
        val_dataset = MagicMock()
        val_dataset.__len__.return_value = 5
        
        # Create PyGMO training pipeline
        pipeline = PyGMOTrainingPipeline(
            model_class=MagicMock,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=self.device,
            output_dir=os.path.join(self.temp_dir, 'training')
        )
        
        # Mock evaluate_model method
        pipeline.evaluate_model = MagicMock(return_value=0.9)
        
        # Define parameter space
        param_space = {
            'learning_rate': (0.0001, 0.01),
            'batch_size': (8, 32),
            'hidden_dim': (32, 128)
        }
        
        # Run optimization
        best_params, best_score = pipeline.run_optimization(
            param_space=param_space,
            population_size=10,
            generations=5,
            algorithm='sade',
            seed=42
        )
        
        # Check that optimization returned results
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)


if __name__ == '__main__':
    unittest.main()
