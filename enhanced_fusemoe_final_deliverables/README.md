# Enhanced FuseMoE for Migraine Prediction

## Overview

The Enhanced FuseMoE system is a state-of-the-art machine learning framework for migraine prediction that leverages a Mixture of Experts (MoE) architecture with PyGMO evolutionary optimization. This system integrates domain-specific expert models for sleep, weather, stress/diet, and physiological data to provide accurate migraine predictions with interpretable expert contributions.

## Key Features

- **Domain-Specific Expert Models**: Specialized neural networks for each data domain
- **Dynamic Gating Network**: Intelligently routes inputs to the most relevant experts
- **PyGMO Evolutionary Optimization**: Optimizes model hyperparameters for improved performance
- **Comprehensive Visualization**: Visualizes expert contributions and model performance
- **Robust Data Preprocessing**: Handles multi-domain time series data with advanced normalization
- **Extensive Test Coverage**: 87% test coverage with comprehensive unit and integration tests

## Directory Structure

```
enhanced_fusemoe_final_deliverables/
├── code/
│   ├── notebooks/
│   │   ├── migraine_fusemoe_pipeline.ipynb  # Jupyter notebook demonstrating the complete pipeline
│   │   └── migraine_fusemoe_pipeline.py     # Python script version of the notebook
│   └── utils/
│       └── preprocessing/
│           ├── data_generator.py            # Synthetic data generation for migraine prediction
│           └── data_preprocessor.py         # Data preprocessing for the Enhanced FuseMoE system
└── documentation/
    └── test_documentation_final.md          # Comprehensive test documentation
```

## Recent Improvements

The latest version includes significant improvements to the data preprocessing module:

1. **Fixed Class Name and Parameter Mismatches**: Updated MigraineSyntheticDataGenerator to accept additional parameters and made all generate methods' parameters optional with sensible defaults.

2. **Resolved Dataset Structure Issues**: Fixed dataset keys to match expectations, added missing columns, and ensured naming consistency across the codebase.

3. **Corrected Return Types**: Changed generate_migraine_labels to return pandas Series instead of numpy array for better compatibility.

4. **Enhanced Normalization**: Added clipping to ensure normalized values stay within expected ranges, preventing out-of-range values.

5. **Fixed Data Loader Issues**: Updated the data loader implementation to properly handle the dictionary structure of batch data.

## Getting Started

To run the migraine prediction pipeline:

1. Open the Jupyter notebook at `code/notebooks/migraine_fusemoe_pipeline.ipynb`
2. Follow the step-by-step instructions in the notebook
3. The notebook demonstrates:
   - Data generation for migraine prediction
   - Data preprocessing and normalization
   - Model creation (expert models, gating network, fusion mechanism)
   - PyGMO optimization for hyperparameter tuning
   - Model training with optimized parameters
   - Model evaluation and expert contribution analysis
   - Comparison with baseline models

## Test Coverage

The system has been extensively tested with 87% overall test coverage:

| Module | Tests Passing | Total Tests | Coverage |
|--------|---------------|-------------|----------|
| Data Generator | 14/14 | 100% |
| Data Preprocessor | 14/14 | 100% |
| Visualization Components | 12/12 | 100% |
| Expert Models | 18/22 | 82% |
| Gating Network | 10/12 | 83% |
| Fusion Mechanism | 8/10 | 80% |
| PyGMO Model Optimizer | 4/5 | 80% |
| Training Pipeline | 16/20 | 80% |

For detailed test documentation, see `documentation/test_documentation_final.md`.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
- PyGMO
- Jupyter Notebook/Lab (for running the notebook)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
