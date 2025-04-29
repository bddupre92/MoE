# Visualization Package for Publication

## Overview
This package contains a complete set of visualizations for your publication, organized by category. The visualizations are based on the MoE (Mixture of Experts) model with PyGMO optimization for migraine prediction.

## Visualization Categories

### 1. Model Training Visualizations (Figures 1-4)
These visualizations show the learning curves and convergence analysis of the model during training. They are based on actual model training data.

- **Figure 1**: Learning curves showing loss over epochs
- **Figure 2**: Learning curves showing accuracy over epochs
- **Figure 3**: Combined learning curves showing both loss and accuracy
- **Figure 4**: Model convergence analysis showing the gap between training and validation accuracy

### 2. Performance Metrics Visualizations (Figures 5-9)
These visualizations show the performance metrics of the baseline and optimized models. They are based on actual model evaluation data.

- **Figure 5**: Confusion matrix for the baseline model
- **Figure 6**: Confusion matrix for the optimized model
- **Figure 7**: ROC curves comparing baseline and optimized models
- **Figure 8**: Bar chart comparing key performance metrics
- **Figure 9**: Comparison of error rates between baseline and optimized models

### 3. Expert Contributions Visualizations (Figures 10-14)
These visualizations show how different experts contribute to the MoE model's predictions. They are based on actual model architecture and behavior.

- **Figure 10**: Expert distribution pie chart
- **Figure 11**: Expert specialization heatmap
- **Figure 12**: MoE architecture network diagram
- **Figure 13**: Expert activation patterns across different input types
- **Figure 14**: Expert contribution weights comparison

### 4. PyGMO Optimization Visualizations (Figures 15-19)
**Note: These visualizations are based on synthetic data** due to integration issues with the PyGMO optimization module. Each visualization is clearly labeled as using synthetic data.

- **Figure 15**: Convergence plot showing fitness improvement over generations
- **Figure 16**: Pareto front visualization for multi-objective optimization
- **Figure 17**: Population diversity across generations
- **Figure 18**: Island model migration topology
- **Figure 19**: Hyperparameter importance for model performance

### 5. Comparative Analysis Visualizations (Figures 20-24)
These visualizations compare different aspects of the model's performance. They are based on actual model evaluation data.

- **Figure 20**: Radar chart comparing models across multiple metrics
- **Figure 21**: Percentage improvements from baseline to optimized model
- **Figure 22**: Trade-offs visualization between different performance aspects
- **Figure 23**: Sensitivity analysis of model to different input features
- **Figure 24**: Ablation study results showing impact of removing components

## Notes on Data Sources

- **Authentic Data**: Figures 1-14 and 20-24 are based on actual model training and evaluation data.
- **Synthetic Data**: Figures 15-19 (PyGMO Optimization) use synthetic data as placeholders due to integration issues with the PyGMO optimization module.

## File Organization

The visualizations are organized in the following directory structure:

```
full_pipeline_output/
├── model_training/            # Figures 1-4
├── performance_metrics/       # Figures 5-9
├── expert_contributions/      # Figures 10-14
├── pygmo_optimization/        # Figures 15-19 (synthetic)
└── comparative_analysis/      # Figures 20-24
```

## Usage in Publication

When using these visualizations in your publication, please note the following:

1. All visualizations are provided in high-resolution PNG format (300 DPI)
2. For Figures 15-19, please include a note that these are based on synthetic data
3. The visualizations are designed to be publication-ready with appropriate fonts, colors, and sizes

## Technical Details

- **Model**: Mixture of Experts (MoE) with multiple expert networks
- **Data**: Synthetic healthcare data for migraine prediction
- **Optimization**: PyGMO-based multi-objective optimization (synthetic results)
- **Metrics**: Accuracy, precision, recall, F1 score, AUC, false positive rate, false negative rate
