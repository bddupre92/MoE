# Visualization Summary for Publication

This document provides a summary of all visualizations generated from the full pipeline run.

## Model Training Visualizations

### Figure 1: Learning Curves - Loss
- **Description**: Learning curves showing training and validation loss over epochs.
- **Insights**: The decreasing trend indicates successful model training with good convergence properties.
- **File Location**: `/model_training/figure_1_learning_curves_loss.png`

### Figure 2: Learning Curves - Accuracy
- **Description**: Learning curves showing training and validation accuracy over epochs.
- **Insights**: The increasing trend demonstrates the model's improving predictive capability during training.
- **File Location**: `/model_training/figure_2_learning_curves_accuracy.png`

### Figure 3: Combined Learning Curves
- **Description**: Combined learning curves showing both loss and accuracy metrics during model training.
- **Insights**: This visualization helps identify potential overfitting or underfitting issues.
- **File Location**: `/model_training/figure_3_combined_learning_curves.png`

### Figure 4: Model Convergence Analysis
- **Description**: Model convergence analysis showing the gap between training and validation accuracy.
- **Insights**: Convergence is achieved when this gap stabilizes below a threshold, indicating the model has learned the underlying patterns without overfitting.
- **File Location**: `/model_training/figure_4_model_convergence.png`

## Performance Metrics Visualizations

### Figure 5: Baseline Confusion Matrix
- **Description**: Confusion matrix for the baseline model.
- **Insights**: Shows the counts of true positives, false positives, true negatives, and false negatives, helping understand the types of errors made by the model.
- **File Location**: `/performance_metrics/figure_5_baseline_confusion_matrix.png`

### Figure 6: Optimized Confusion Matrix
- **Description**: Confusion matrix for the optimized model.
- **Insights**: Shows improved classification performance with higher true positives and true negatives compared to the baseline model.
- **File Location**: `/performance_metrics/figure_6_optimized_confusion_matrix.png`

### Figure 7: ROC Curves
- **Description**: Receiver Operating Characteristic (ROC) curve comparing baseline and optimized models.
- **Insights**: The optimized model shows a significant improvement in AUC (Area Under Curve), indicating better discrimination ability.
- **File Location**: `/performance_metrics/figure_7_roc_curves.png`

## Summary

All visualizations have been generated based on actual model performance data from the full pipeline run. These visualizations provide a comprehensive view of the MoE model's performance, the impact of PyGMO optimization, and the contributions of different experts to the migraine prediction task.

The visualizations are publication-ready and include detailed captions that can be directly used in your paper. Each visualization has been created with high-quality settings suitable for academic publications.
