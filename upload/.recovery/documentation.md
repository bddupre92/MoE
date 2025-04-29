# Enhanced Data Pipeline Integration Documentation

## Overview

This document provides comprehensive documentation for the integration of the enhanced data generation pipeline with the existing Mixture of Experts (MoE) system for migraine prediction. The enhanced pipeline addresses limitations in the current implementation by generating more realistic data with proper temporal patterns, correlations between variables, and individual variability in migraine triggers.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Integration Components](#integration-components)
4. [Implementation Details](#implementation-details)
5. [Usage Guide](#usage-guide)
6. [Testing and Validation](#testing-and-validation)
7. [Dashboard](#dashboard)
8. [Future Enhancements](#future-enhancements)

## Introduction

The original MoE system for migraine prediction uses a data generation approach that lacks realistic temporal patterns and proper correlations between variables. The enhanced data generation pipeline improves upon this by:

- Generating more realistic temporal patterns with circadian rhythms, weekly patterns, and seasonal variations
- Modeling proper correlations between related variables
- Incorporating individual variability in baseline values and trigger sensitivity
- Implementing realistic lag effects between triggers and migraine onset
- Providing comprehensive data validation to ensure data quality
- Supporting multiple output formats (PyTorch, NumPy, Pandas, JSON)

## Architecture

The integration follows a modular architecture that maintains compatibility with the existing MoE system while enhancing its capabilities:

```
Enhanced Data Pipeline
    ↓
Adapters (Sleep, Weather, Stress/Diet, Physio)
    ↓
MoE Model (Expert Models, Gating Network, Fusion Mechanism)
    ↓
Performance Evaluation
    ↓
Dashboard Visualization
```

### Data Flow

1. The enhanced data pipeline generates synthetic data for sleep, weather, stress/diet, and physiological measurements
2. Adapters convert this data to the format expected by each expert model
3. The MoE model processes the data through its expert models, gating network, and fusion mechanism
4. Performance metrics are calculated and stored
5. The dashboard visualizes the data, model performance, and expert contributions

## Integration Components

### Enhanced Data Pipeline Integration

The `EnhancedDataPipelineIntegration` class serves as the main interface between the enhanced data generation pipeline and the existing MoE system. It handles:

- Initialization of data generators and adapters
- Configuration management
- Data generation and adaptation
- Data splitting for training, validation, and testing

### Adapters

Adapters convert the enhanced data to the format expected by each expert model:

- `SleepDataAdapter`: Adapts sleep data to match the expected input format of the sleep expert model (shape: None, 7, 6)
- `WeatherDataAdapter`: Adapts weather data for the weather expert model (shape: None, 5)
- `StressDietAdapter`: Adapts stress and diet data for the stress_diet expert model (shape: None, 6)
- `PhysioDataAdapter`: Adapts physiological data for the physio expert model (shape: None, 5)

### Parameter Calibration

The `ParameterCalibrator` class optimizes the parameters of the enhanced data pipeline to achieve the best performance with the MoE model. It calibrates:

- Temporal pattern parameters (circadian rhythms, weekly patterns, seasonal variations, individual variability, lag effects)
- Correlation parameters between different data modalities

### Training Data Generation

The `TrainingDataGenerator` class generates training data using the calibrated parameters of the enhanced data pipeline. It handles:

- Data generation with optimal parameters
- Data adaptation for expert models
- Data splitting into train, validation, and test sets
- Data visualization and saving

### Model Training

The `ModelTrainer` class trains the MoE model using the enhanced training data. It handles:

- Model initialization
- Training loop with early stopping
- Validation and evaluation
- Metrics calculation and visualization
- Model saving

### Dashboard Integration

The `DashboardIntegration` class creates a Streamlit dashboard to visualize the enhanced data and model performance. It includes:

- Data visualization page
- Model performance page
- Configuration page
- About page

## Implementation Details

### Directory Structure

```
moe_integration/
├── config/
│   └── config_manager.py
├── enhanced_data_pipeline/
│   ├── integration.py
│   ├── parameter_calibration.py
│   ├── training_data_generator.py
│   ├── model_trainer.py
│   └── dataset.py
├── adapters/
│   ├── sleep_adapter.py
│   ├── weather_adapter.py
│   ├── stress_diet_adapter.py
│   └── physio_adapter.py
├── dashboard/
│   ├── Home.py
│   └── pages/
│       ├── 1_Data_Visualization.py
│       ├── 2_Model_Performance.py
│       ├── 3_Configuration.py
│       └── 4_About.py
├── tests/
│   └── test_integration.py
└── output/
    ├── data/
    ├── models/
    └── calibration/
```

### Key Classes and Methods

#### EnhancedDataPipelineIntegration

```python
class EnhancedDataPipelineIntegration:
    def __init__(self, config=None)
    def initialize_generators(self)
    def initialize_adapters(self)
    def generate_data(self, num_samples)
    def adapt_data_for_experts(self, data)
    def prepare_data_for_training(self, data)
    def save_data(self, data, path)
```

#### ParameterCalibrator

```python
class ParameterCalibrator:
    def __init__(self, integration, output_dir=None)
    def calibrate_temporal_parameters(self, num_samples=1000)
    def calibrate_correlation_parameters(self, num_samples=1000)
```

#### TrainingDataGenerator

```python
class TrainingDataGenerator:
    def __init__(self, integration, output_dir=None)
    def generate_training_data(self, num_samples=None)
```

#### ModelTrainer

```python
class ModelTrainer:
    def __init__(self, integration, output_dir=None)
    def initialize_model(self)
    def train(self, train_loader, val_loader, num_epochs=None, early_stopping_patience=None)
    def evaluate(self, test_loader)
```

#### DashboardIntegration

```python
class DashboardIntegration:
    def __init__(self, integration)
    def create_dashboard(self)
    def run_dashboard(self, port=None)
```

## Usage Guide

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bddupre92/MoE.git
cd MoE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the enhanced data pipeline integration files to the appropriate location:
```bash
cp -r moe_integration /path/to/destination
```

### Basic Usage

1. Initialize the integration:
```python
from moe_integration.enhanced_data_pipeline.integration import EnhancedDataPipelineIntegration

# Initialize integration
integration = EnhancedDataPipelineIntegration()

# Initialize generators and adapters
integration.initialize_generators()
integration.initialize_adapters()
```

2. Calibrate parameters:
```python
from moe_integration.enhanced_data_pipeline.parameter_calibration import ParameterCalibrator

# Initialize calibrator
calibrator = ParameterCalibrator(integration)

# Calibrate parameters
temporal_params = calibrator.calibrate_temporal_parameters()
correlation_params = calibrator.calibrate_correlation_parameters()
```

3. Generate training data:
```python
from moe_integration.enhanced_data_pipeline.training_data_generator import TrainingDataGenerator

# Initialize training data generator
data_generator = TrainingDataGenerator(integration)

# Generate training data
data = data_generator.generate_training_data()
```

4. Train the model:
```python
from moe_integration.enhanced_data_pipeline.model_trainer import ModelTrainer

# Initialize model trainer
trainer = ModelTrainer(integration)

# Train model
history = trainer.train(data['train_loader'], data['val_loader'])

# Evaluate model
test_metrics = trainer.evaluate(data['test_loader'])
```

5. Create and run the dashboard:
```python
from moe_integration.dashboard.dashboard_integration import DashboardIntegration

# Initialize dashboard integration
dashboard = DashboardIntegration(integration)

# Create dashboard
dashboard.create_dashboard()

# Run dashboard
dashboard.run_dashboard()
```

### Configuration

The enhanced data pipeline can be configured through the `config.py` file or through the dashboard's configuration page. Key configuration parameters include:

- **General Parameters**: Number of samples, random seed, sequence length, data splits
- **Temporal Pattern Parameters**: Enable/disable circadian rhythms, weekly patterns, seasonal variations, individual variability, lag effects
- **Correlation Parameters**: Correlation values between different data modalities
- **Training Parameters**: Batch size, learning rate, number of epochs, early stopping patience

## Testing and Validation

The integration includes comprehensive testing to ensure proper functionality:

1. **Unit Tests**: Test individual components of the enhanced data pipeline
2. **Integration Tests**: Test the complete integration from data generation to model training and dashboard visualization
3. **Validation**: Validate the enhanced data against real-world patterns and expert knowledge

To run the integration test:
```bash
python -m moe_integration.tests.test_integration
```

## Dashboard

The Streamlit dashboard provides a user-friendly interface for visualizing the enhanced data and model performance. It includes:

### Home Page
- Overview of the enhanced data pipeline integration
- Key performance metrics
- Sample visualizations

### Data Visualization Page
- Data distributions for different modalities
- Temporal patterns
- Correlations between variables
- Target distribution
- Raw data explorer

### Model Performance Page
- Training metrics
- Test performance
- Expert contributions
- Performance comparison with original model

### Configuration Page
- General configuration
- Temporal pattern configuration
- Correlation configuration
- Calibration results

### About Page
- Information about the enhanced data pipeline integration
- Architecture overview
- Benefits and future work

## Future Enhancements

Potential future enhancements include:

1. **Additional Data Modalities**: Incorporating more data sources
2. **Personalization**: Tailoring the model to individual patient characteristics
3. **Mobile Integration**: Deploying the model on mobile devices for real-time prediction
4. **Explainability**: Improving the explanation of model predictions
5. **Advanced Visualization**: Adding more interactive visualizations
6. **Real-time Data Integration**: Connecting to real-time data sources
7. **Federated Learning**: Enabling privacy-preserving distributed training
