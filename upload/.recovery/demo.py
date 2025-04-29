"""
Demonstration script for the enhanced data pipeline integration with MoE.

This script demonstrates the complete integration of the enhanced data pipeline with the MoE model,
from data generation to model training and dashboard visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

# Add the integration directory to the path
sys.path.append('/home/ubuntu/moe_integration')

# Import integration components
from enhanced_data_pipeline.integration import EnhancedDataPipelineIntegration
from enhanced_data_pipeline.parameter_calibration import ParameterCalibrator
from enhanced_data_pipeline.training_data_generator import TrainingDataGenerator
from enhanced_data_pipeline.model_trainer import ModelTrainer
from dashboard.dashboard_integration import DashboardIntegration
from config.config import *

def run_demonstration(mode: str = 'full', num_samples: int = 100, num_epochs: int = 10, run_dashboard: bool = False):
    """
    Run a demonstration of the enhanced data pipeline integration.
    
    Args:
        mode: Demonstration mode ('full', 'data_only', 'dashboard_only')
        num_samples: Number of samples to generate
        num_epochs: Number of epochs to train for
        run_dashboard: Whether to run the Streamlit dashboard
    
    Returns:
        Dictionary containing demonstration results
    """
    print("=" * 80)
    print("ENHANCED DATA PIPELINE INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Create output directory
    output_dir = os.path.join('/home/ubuntu/moe_integration/output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {
        'data_generation': None,
        'model_training': None,
        'dashboard': None
    }
    
    # Step 1: Initialize integration
    print("\nStep 1: Initializing integration...")
    integration = EnhancedDataPipelineIntegration()
    
    # Initialize generators and adapters
    integration.initialize_generators()
    integration.initialize_adapters()
    print("Integration initialized successfully")
    
    # Step 2: Generate data
    if mode in ['full', 'data_only']:
        print("\nStep 2: Generating enhanced data...")
        
        # Initialize training data generator
        data_generator = TrainingDataGenerator(integration)
        
        # Generate training data
        start_time = time.time()
        data = data_generator.generate_training_data(num_samples=num_samples)
        generation_time = time.time() - start_time
        
        print(f"Data generation completed in {generation_time:.2f} seconds")
        
        # Store results
        results['data_generation'] = {
            'num_samples': num_samples,
            'generation_time': generation_time,
            'data_shapes': {
                'sleep': data['raw_data']['sleep'].shape if 'sleep' in data['raw_data'] else None,
                'weather': data['raw_data']['weather'].shape if 'weather' in data['raw_data'] else None,
                'stress_diet': data['raw_data']['stress_diet'].shape if 'stress_diet' in data['raw_data'] else None,
                'physio': data['raw_data']['physio'].shape if 'physio' in data['raw_data'] else None,
                'target': data['raw_data']['target'].shape if 'target' in data['raw_data'] else None
            }
        }
    
    # Step 3: Train model
    if mode == 'full':
        print("\nStep 3: Training MoE model with enhanced data...")
        
        # Initialize model trainer
        trainer = ModelTrainer(integration)
        
        # Initialize model
        success = trainer.initialize_model()
        
        if success:
            # Train model
            start_time = time.time()
            history = trainer.train(
                data['train_loader'], 
                data['val_loader'],
                num_epochs=num_epochs
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            test_metrics = trainer.evaluate(data['test_loader'])
            
            print(f"Model training completed in {training_time:.2f} seconds")
            print(f"Test metrics: {test_metrics}")
            
            # Store results
            results['model_training'] = {
                'num_epochs': num_epochs,
                'training_time': training_time,
                'test_metrics': test_metrics,
                'history': history
            }
        else:
            print("Model initialization failed, skipping training")
    
    # Step 4: Create dashboard
    print("\nStep 4: Creating Streamlit dashboard...")
    
    # Initialize dashboard integration
    dashboard = DashboardIntegration(integration)
    
    # Create dashboard
    dashboard.create_dashboard()
    
    print("Dashboard created successfully")
    
    # Store results
    results['dashboard'] = {
        'dashboard_dir': dashboard.dashboard_dir,
        'pages': [
            'Home.py',
            'pages/1_Data_Visualization.py',
            'pages/2_Model_Performance.py',
            'pages/3_Configuration.py',
            'pages/4_About.py'
        ]
    }
    
    # Run dashboard if requested
    if run_dashboard:
        print("\nStep 5: Running Streamlit dashboard...")
        dashboard.run_dashboard()
    else:
        print("\nTo run the dashboard, execute the following command:")
        print(f"cd {dashboard.dashboard_dir} && streamlit run Home.py")
    
    print("\nDemonstration completed successfully!")
    return results

def main():
    """Parse command line arguments and run demonstration."""
    parser = argparse.ArgumentParser(description='Run enhanced data pipeline integration demonstration')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'data_only', 'dashboard_only'],
                        help='Demonstration mode')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--run-dashboard', action='store_true',
                        help='Run the Streamlit dashboard')
    
    args = parser.parse_args()
    
    run_demonstration(
        mode=args.mode,
        num_samples=args.samples,
        num_epochs=args.epochs,
        run_dashboard=args.run_dashboard
    )

if __name__ == "__main__":
    main()
