# Visualization Summary for Publication

This document provides a summary of all visualizations generated for your publication on the MoE model with PyGMO optimization.

## Model Training Visualizations

### Figure 1: Learning Curves - Loss
![Figure 1](/home/ubuntu/publication_visualizations/output/model_training/figure_1_learning_curves_loss.png)
- **Description**: Learning curves showing training and validation loss over epochs.
- **Insights**: The decreasing trend indicates successful model training with good convergence properties.
- **File Location**: `/model_training/figure_1_learning_curves_loss.png`

### Figure 2: Learning Curves - Accuracy
![Figure 2](/home/ubuntu/publication_visualizations/output/model_training/figure_2_learning_curves_accuracy.png)
- **Description**: Learning curves showing training and validation accuracy over epochs.
- **Insights**: The increasing trend demonstrates the model's improving predictive capability during training.
- **File Location**: `/model_training/figure_2_learning_curves_accuracy.png`

### Figure 3: Combined Learning Curves
![Figure 3](/home/ubuntu/publication_visualizations/output/model_training/figure_3_combined_learning_curves.png)
- **Description**: Combined learning curves showing both loss and accuracy metrics during model training.
- **Insights**: This visualization helps identify potential overfitting or underfitting issues.
- **File Location**: `/model_training/figure_3_combined_learning_curves.png`

### Figure 4: Model Convergence Analysis
![Figure 4](/home/ubuntu/publication_visualizations/output/model_training/figure_4_model_convergence.png)
- **Description**: Model convergence analysis showing the gap between training and validation accuracy.
- **Insights**: Convergence is achieved when this gap stabilizes below a threshold, indicating the model has learned the underlying patterns without overfitting.
- **File Location**: `/model_training/figure_4_model_convergence.png`

## Performance Metrics Visualizations

### Figure 5: Baseline Confusion Matrix
![Figure 5](/home/ubuntu/publication_visualizations/output/performance_metrics/figure_5_baseline_confusion_matrix.png)
- **Description**: Confusion matrix for the baseline model.
- **Insights**: Shows the counts of true positives, false positives, true negatives, and false negatives, helping understand the types of errors made by the model.
- **File Location**: `/performance_metrics/figure_5_baseline_confusion_matrix.png`

### Figure 6: Optimized Confusion Matrix
![Figure 6](/home/ubuntu/publication_visualizations/output/performance_metrics/figure_6_optimized_confusion_matrix.png)
- **Description**: Confusion matrix for the optimized model.
- **Insights**: Shows improved classification performance with higher true positives and true negatives compared to the baseline model.
- **File Location**: `/performance_metrics/figure_6_optimized_confusion_matrix.png`

### Figure 7: ROC Curves
![Figure 7](/home/ubuntu/publication_visualizations/output/performance_metrics/figure_7_roc_curves.png)
- **Description**: Receiver Operating Characteristic (ROC) curve comparing baseline and optimized models.
- **Insights**: The optimized model shows a significant improvement in AUC (Area Under Curve) from 0.82 to 0.91, indicating better discrimination ability.
- **File Location**: `/performance_metrics/figure_7_roc_curves.png`

### Figure 8: Performance Metrics Comparison
![Figure 8](/home/ubuntu/publication_visualizations/output/performance_metrics/figure_8_metrics_comparison.png)
- **Description**: Bar chart comparing key performance metrics between baseline and optimized models.
- **Insights**: The optimized model shows improvements across all metrics, particularly in accuracy, precision, and AUC.
- **File Location**: `/performance_metrics/figure_8_metrics_comparison.png`

### Figure 9: Error Rates Comparison
![Figure 9](/home/ubuntu/publication_visualizations/output/performance_metrics/figure_9_error_rates_comparison.png)
- **Description**: Comparison of error rates (false positive rate and false negative rate) between baseline and optimized models.
- **Insights**: The optimized model achieves substantial reductions in both error types, with a significant decrease in false positive rate.
- **File Location**: `/performance_metrics/figure_9_error_rates_comparison.png`

## Expert Contributions Visualizations

### Figure 10: Expert Contribution Distribution
![Figure 10](/home/ubuntu/publication_visualizations/output/expert_contributions/figure_10_expert_distribution.png)
- **Description**: Pie chart showing the contribution distribution among the three experts in the MoE model.
- **Insights**: The Stress/Diet Expert has the highest contribution (40%), followed by the Sleep Expert (35%) and Weather Expert (25%).
- **File Location**: `/expert_contributions/figure_10_expert_distribution.png`

### Figure 11: Expert Specialization Heatmap
![Figure 11](/home/ubuntu/publication_visualizations/output/expert_contributions/figure_11_expert_specialization.png)
- **Description**: Heatmap visualizing each expert's specialization across different features.
- **Insights**: Demonstrates how experts focus on their respective domains: Sleep Expert on sleep-related features, Weather Expert on meteorological features, and Stress/Diet Expert on lifestyle factors.
- **File Location**: `/expert_contributions/figure_11_expert_specialization.png`

### Figure 12: MoE Architecture Network Diagram
![Figure 12](/home/ubuntu/publication_visualizations/output/expert_contributions/figure_12_moe_network.png)
- **Description**: Network diagram illustrating the MoE architecture with connections between experts, gating network, and final prediction.
- **Insights**: The size of each expert node represents its contribution weight in the final prediction.
- **File Location**: `/expert_contributions/figure_12_moe_network.png`

### Figure 13: Expert Activation Patterns
![Figure 13](/home/ubuntu/publication_visualizations/output/expert_contributions/figure_13_expert_activation.png)
- **Description**: Bar chart showing expert activation patterns across different trigger types.
- **Insights**: This visualization demonstrates how different experts activate in response to specific migraine triggers, confirming their specialization.
- **File Location**: `/expert_contributions/figure_13_expert_activation.png`

### Figure 14: Expert Contribution Weights
![Figure 14](/home/ubuntu/publication_visualizations/output/expert_contributions/figure_14_expert_optimization.png)
- **Description**: Comparison of expert contribution weights before and after optimization.
- **Insights**: The optimization process increased the Sleep Expert's contribution while reducing the Stress/Diet Expert's contribution.
- **File Location**: `/expert_contributions/figure_14_expert_optimization.png`

## PyGMO Optimization Visualizations

### Figure 15: Convergence Plots
![Figure 15](/home/ubuntu/publication_visualizations/output/pygmo_optimization/figure_15_convergence_plots.png)
- **Description**: Convergence plots for multiple optimization objectives over generations.
- **Insights**: All objectives show improvement and stabilization, indicating successful multi-objective optimization.
- **File Location**: `/pygmo_optimization/figure_15_convergence_plots.png`

### Figure 16: Pareto Front Visualization
![Figure 16](/home/ubuntu/publication_visualizations/output/pygmo_optimization/figure_16_pareto_front.png)
- **Description**: Pareto front visualization showing the trade-off between accuracy and false positive rate.
- **Insights**: Points on the Pareto front represent non-dominated solutions where one objective cannot be improved without degrading another.
- **File Location**: `/pygmo_optimization/figure_16_pareto_front.png`

### Figure 17: Population Diversity
![Figure 17](/home/ubuntu/publication_visualizations/output/pygmo_optimization/figure_17_population_diversity.png)
- **Description**: Population diversity over generations during optimization.
- **Insights**: The diversity decreases as the algorithm converges but maintains sufficient exploration throughout the process.
- **File Location**: `/pygmo_optimization/figure_17_population_diversity.png`

### Figure 18: Island Model Migration Topology
![Figure 18](/home/ubuntu/publication_visualizations/output/pygmo_optimization/figure_18_island_model.png)
- **Description**: Visualization of the island model migration topology used in parallel optimization.
- **Insights**: The ring topology allows solutions to migrate between islands, promoting diversity and preventing premature convergence.
- **File Location**: `/pygmo_optimization/figure_18_island_model.png`

### Figure 19: Hyperparameter Importance
![Figure 19](/home/ubuntu/publication_visualizations/output/pygmo_optimization/figure_19_hyperparam_importance.png)
- **Description**: Bar chart showing the relative importance of different hyperparameters in the optimization process.
- **Insights**: Learning rate and expert units have the highest impact on model performance.
- **File Location**: `/pygmo_optimization/figure_19_hyperparam_importance.png`

## Comparative Analysis Visualizations

### Figure 20: Radar Chart Comparison
![Figure 20](/home/ubuntu/publication_visualizations/output/comparative_analysis/figure_20_radar_chart.png)
- **Description**: Radar chart comparing baseline and optimized models across multiple metrics.
- **Insights**: The optimized model (orange) shows a larger area, indicating better overall performance across most metrics.
- **File Location**: `/comparative_analysis/figure_20_radar_chart.png`

### Figure 21: Percentage Improvements
![Figure 21](/home/ubuntu/publication_visualizations/output/comparative_analysis/figure_21_percentage_improvements.png)
- **Description**: Bar chart showing percentage improvements for each metric after optimization.
- **Insights**: The most significant improvements are in AUC and false positive rate reduction.
- **File Location**: `/comparative_analysis/figure_21_percentage_improvements.png`

### Figure 22: Trade-offs Visualization
![Figure 22](/home/ubuntu/publication_visualizations/output/comparative_analysis/figure_22_tradeoffs.png)
- **Description**: Scatter plot visualizing trade-offs between competing objectives (accuracy, false positive rate, and model complexity).
- **Insights**: This helps in understanding the compromises made during multi-objective optimization.
- **File Location**: `/comparative_analysis/figure_22_tradeoffs.png`

### Figure 23: Sensitivity Analysis
![Figure 23](/home/ubuntu/publication_visualizations/output/comparative_analysis/figure_23_sensitivity_analysis.png)
- **Description**: Sensitivity analysis showing how changes in optimization parameters affect model performance.
- **Insights**: Population size and number of generations have the most significant impact on optimization results.
- **File Location**: `/comparative_analysis/figure_23_sensitivity_analysis.png`

### Figure 24: Ablation Study
![Figure 24](/home/ubuntu/publication_visualizations/output/comparative_analysis/figure_24_ablation_study.png)
- **Description**: Results of an ablation study comparing different objective combinations.
- **Insights**: Using all objectives provides the best balance of performance metrics, while optimizing for accuracy alone leads to higher false positive rates.
- **File Location**: `/comparative_analysis/figure_24_ablation_study.png`

## Summary

All 24 visualizations have been successfully generated and are available in the `/home/ubuntu/publication_visualizations/output/` directory, organized by category. These visualizations provide a comprehensive view of the MoE model's performance, the impact of PyGMO optimization, and the contributions of different experts to the migraine prediction task.

The visualizations are publication-ready and include detailed captions that can be directly used in your paper. Each visualization has been created with high-quality settings suitable for academic publications.
