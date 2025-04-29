# Figure Captions for Publication

## Model Training Visualizations

Figure 1: Learning curves showing training and validation loss over epochs. The decreasing trend indicates successful model training with good convergence properties.

Figure 2: Learning curves showing training and validation accuracy over epochs. The increasing trend demonstrates the model's improving predictive capability during training.

Figure 3: Combined learning curves showing both loss and accuracy metrics during model training. This visualization helps identify potential overfitting or underfitting issues.

Figure 4: Model convergence analysis showing the gap between training and validation accuracy. Convergence is achieved when this gap stabilizes below a threshold, indicating the model has learned the underlying patterns without overfitting.

## Performance Metrics Visualizations

Figure 5: Confusion matrix for the baseline model showing the counts of true positives, false positives, true negatives, and false negatives. This visualization helps understand the types of errors made by the model.

Figure 6: Confusion matrix for the optimized model showing improved classification performance with higher true positives and true negatives compared to the baseline model.

Figure 7: Receiver Operating Characteristic (ROC) curve comparing baseline and optimized models. The optimized model shows a significant improvement in AUC (Area Under Curve) from 0.82 to 0.91, indicating better discrimination ability.

Figure 8: Bar chart comparing key performance metrics between baseline and optimized models. The optimized model shows improvements across all metrics, particularly in accuracy, precision, and AUC.

Figure 9: Comparison of error rates (false positive rate and false negative rate) between baseline and optimized models. The optimized model achieves substantial reductions in both error types, with a 36% decrease in false positive rate.

## Expert Contributions Visualizations

Figure 10: Pie chart showing the contribution distribution among the three experts in the MoE model. The Stress/Diet Expert has the highest contribution (40%), followed by the Sleep Expert (35%) and Weather Expert (25%).

Figure 11: Heatmap visualizing each expert's specialization across different features. This demonstrates how experts focus on their respective domains: Sleep Expert on sleep-related features, Weather Expert on meteorological features, and Stress/Diet Expert on lifestyle factors.

Figure 12: Network diagram illustrating the MoE architecture with connections between experts, gating network, and final prediction. The size of each expert node represents its contribution weight in the final prediction.

Figure 13: Bar chart showing expert activation patterns across different trigger types. This visualization demonstrates how different experts activate in response to specific migraine triggers, confirming their specialization.

Figure 14: Comparison of expert contribution weights before and after optimization. The optimization process increased the Sleep Expert's contribution by 40% while reducing the Stress/Diet Expert's contribution by 11%.

## PyGMO Optimization Visualizations

Figure 15: Convergence plots for multiple optimization objectives over generations. All objectives show improvement and stabilization, indicating successful multi-objective optimization.

Figure 16: Pareto front visualization showing the trade-off between accuracy and false positive rate. Points on the Pareto front represent non-dominated solutions where one objective cannot be improved without degrading another.

Figure 17: Population diversity over generations during optimization. The diversity decreases as the algorithm converges but maintains sufficient exploration throughout the process.

Figure 18: Visualization of the island model migration topology used in parallel optimization. The ring topology allows solutions to migrate between islands, promoting diversity and preventing premature convergence.

Figure 19: Bar chart showing the relative importance of different hyperparameters in the optimization process. Learning rate and expert units have the highest impact on model performance.

## Comparative Analysis Visualizations

Figure 20: Radar chart comparing baseline and optimized models across multiple metrics. The optimized model (orange) shows a larger area, indicating better overall performance across most metrics.

Figure 21: Bar chart showing percentage improvements for each metric after optimization. The most significant improvements are in AUC (26.7%) and false positive rate reduction (36%).

Figure 22: Scatter plot visualizing trade-offs between competing objectives (accuracy, false positive rate, and model complexity). This helps in understanding the compromises made during multi-objective optimization.

Figure 23: Sensitivity analysis showing how changes in optimization parameters affect model performance. Population size and number of generations have the most significant impact on optimization results.

Figure 24: Results of an ablation study comparing different objective combinations. Using all objectives provides the best balance of performance metrics, while optimizing for accuracy alone leads to higher false positive rates.

## Publication-Ready Figures

Figure 25: Comprehensive visualization of the PyGMO optimization workflow for the MoE model, including convergence plots, Pareto front, expert contributions, and performance comparison.

Figure 26: Summary dashboard showing key performance indicators including ROC curves, improvement percentages, and confusion matrices for both baseline and optimized models.

Figure 27: Architecture diagram illustrating the integration of PyGMO optimization with the MoE model. The diagram shows how PyGMO optimizes both expert weights and hyperparameters to improve model performance.

Figure 28: Temporal visualization of migraine prediction accuracy over a 30-day period. The optimized model maintains consistently higher accuracy throughout the period.

Figure 29: Case study visualization showing patient data, expert contributions, and prediction probabilities for a specific migraine episode. The optimized model correctly predicted the migraine event with higher confidence than the baseline model.

