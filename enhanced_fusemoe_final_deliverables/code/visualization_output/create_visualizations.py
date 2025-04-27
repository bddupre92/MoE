import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create output directory
output_dir = "visualization_output"
os.makedirs(output_dir, exist_ok=True)

# Performance metrics
models = ['Enhanced FuseMoE', 'Logistic Regression', 'Random Forest']
metrics = {
    'accuracy': [0.6467, 0.8667, 0.8800],
    'precision': [0.6467, 0.8969, 0.8559],
    'recall': [1.0000, 0.8969, 0.9794],
    'f1': [0.7854, 0.8969, 0.9135],
    'roc_auc': [0.6989, 0.9537, 0.9399]
}

# Expert contributions
expert_names = ['Sleep Expert', 'Weather Expert', 'Stress/Diet Expert', 'Physiological Expert']
contributions = [0.1721, 0.5171, 0.8411, 0.4697]

# Training history
epochs = range(1, 11)
train_loss = [0.8617, 0.8597, 0.8569, 0.8546, 0.8482, 0.8420, 0.8394, 0.8356, 0.8416, 0.8354]
val_loss = [0.5789, 0.5752, 0.5745, 0.5724, 0.5537, 0.5744, 0.5467, 0.5522, 0.5595, 0.5412]

# 1. Model Comparison Bar Chart
plt.figure(figsize=(12, 8))
x = np.arange(len(models))
width = 0.15
multiplier = 0

for metric, values in metrics.items():
    offset = width * multiplier
    plt.bar(x + offset, values, width, label=metric.upper())
    multiplier += 1

plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics Comparison Across Models', fontsize=14)
plt.xticks(x + width * 2, models)
plt.ylim(0, 1.1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Expert Contributions Bar Chart
plt.figure(figsize=(10, 6))
bars = plt.bar(expert_names, contributions, color=sns.color_palette("viridis", len(expert_names)))
plt.xlabel('Expert Models', fontsize=12)
plt.ylabel('Contribution Score', fontsize=12)
plt.title('Expert Model Contributions to Predictions', fontsize=14)
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.savefig(os.path.join(output_dir, 'expert_contributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Training History Line Chart
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Time', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Radar Chart for Model Comparison
metrics_list = list(metrics.keys())
angles = np.linspace(0, 2*np.pi, len(metrics_list), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, model in enumerate(models):
    values = [metrics[metric][i] for metric in metrics_list]
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics_list)
ax.set_ylim(0, 1)
ax.set_rlabel_position(0)
ax.set_title("Model Performance Comparison (Radar Chart)", fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Heatmap of Expert Contributions
plt.figure(figsize=(10, 6))
contribution_matrix = np.outer(contributions, np.ones(5))
sns.heatmap(contribution_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="YlGnBu",
            xticklabels=['Migraine\nPrediction'] * 5,
            yticklabels=expert_names,
            cbar_kws={'label': 'Contribution Strength'})
plt.title('Expert Contribution Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'contribution_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations created and saved to the visualization_output directory.")
