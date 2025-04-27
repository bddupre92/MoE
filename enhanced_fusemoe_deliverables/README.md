# Enhanced FuseMoE with PyGMO Integration

This repository contains the Enhanced FuseMoE system, which extends the [FuseMoE](https://github.com/aaronhan223/FuseMoE) framework with PyGMO integration for evolutionary optimization of mixture of experts models for migraine prediction.

## Overview

The Enhanced FuseMoE system combines domain-specific expert models with evolutionary optimization to improve migraine prediction accuracy. Key features include:

1. **Domain-Specific Expert Models**: Specialized models for sleep, weather, stress/diet, and physiological data
2. **PyGMO Integration**: Evolutionary optimization for hyperparameter tuning and model selection
3. **Advanced Fusion Mechanisms**: Sophisticated methods for combining expert outputs
4. **Comprehensive Evaluation**: Detailed performance metrics and visualization tools
5. **Robust Testing Framework**: Extensive unit and integration tests

## Directory Structure

- **code/**: Core implementation of the Enhanced FuseMoE system
  - **models/**: Expert models, gating networks, and fusion mechanisms
  - **optimization/**: PyGMO integration and evolutionary algorithms
  - **utils/**: Data processing, training pipelines, and evaluation metrics
- **documentation/**: System documentation and specifications
  - **PRD.md**: Product Requirements Document
  - **test_documentation.md**: Testing framework documentation
- **notebooks/**: Jupyter notebooks demonstrating the system
  - **migraine_fusemoe_pipeline.ipynb**: Complete pipeline demonstration
- **tests/**: Comprehensive test suite
  - **unit/**: Unit tests for individual components
  - **integration/**: Integration tests for the complete system

## Getting Started

To get started with the Enhanced FuseMoE system, follow these steps:

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install torch pandas numpy matplotlib scikit-learn pygmo
   ```
3. Open the Jupyter notebook in the `notebooks/` directory to see a complete demonstration of the system

## Key Components

### Expert Models

The system includes specialized expert models for different data domains:

- **Sleep Expert**: Analyzes sleep patterns and quality
- **Weather Expert**: Processes weather conditions and changes
- **Stress/Diet Expert**: Evaluates stress levels and dietary factors
- **Physiological Expert**: Examines physiological measurements

### Gating Network

The MigraineGating network determines which experts to use for each input, optimizing the mixture of experts approach.

### Fusion Mechanism

The MigraineFusionMoE mechanism combines the outputs of the selected experts to produce the final prediction.

### PyGMO Optimization

The system leverages PyGMO's evolutionary algorithms to optimize:

- Model hyperparameters
- Expert selection and weighting
- Fusion mechanism parameters

## Performance

The Enhanced FuseMoE system achieves superior performance compared to baseline models:

- Higher accuracy and precision in migraine prediction
- Better handling of multi-modal data
- Improved interpretability through expert contributions analysis

## Documentation

For detailed documentation, refer to:

- **PRD.md**: Product Requirements Document outlining the system specifications
- **test_documentation.md**: Comprehensive documentation of the testing framework

## Jupyter Notebook

The `migraine_fusemoe_pipeline.ipynb` notebook in the `notebooks/` directory provides a complete demonstration of the Enhanced FuseMoE system, including:

1. Data generation and preprocessing
2. Model creation and configuration
3. PyGMO optimization
4. Model training and evaluation
5. Visualization and interpretation

## Contributing

Contributions to the Enhanced FuseMoE system are welcome. Please ensure that all contributions adhere to the existing code style and include appropriate tests.
