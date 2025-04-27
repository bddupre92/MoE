"""
Optimized Migraine FuseMoE Pipeline

This script demonstrates the complete Enhanced FuseMoE pipeline with all optimizations
implemented to achieve >95% performance for migraine prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Enhanced FuseMoE modules
from utils.preprocessing.enhanced_data_generator import EnhancedMigraineSyntheticDataGenerator
from utils.preprocessing.enhanced_data_preprocessing import AdvancedDataPreprocessor, EnhancedDataLoader
from models.experts.enhanced_experts import EnhancedSleepExpert, EnhancedWeatherExpert, EnhancedStressDietExpert, EnhancedPhysiologicalExpert
from models.gating.enhanced_migraine_gating import EnhancedGatingNetwork
from models.fusion.migraine_fusion import MigraineFusion
from models.ensemble.enhanced_ensemble import EnhancedEnsemble, ModelDistillation
from optimization.bayesian_optimizer import BayesianOptimizer
from utils.evaluation.metrics import calculate_classification_metrics, calculate_expert_contribution_metrics
from utils.visualization.expert_dashboard import ExpertDashboard

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Step 1: Generate enhanced synthetic data
print("Generating enhanced synthetic data...")
data_generator = EnhancedMigraineSyntheticDataGenerator(
    num_patients=100,
    time_periods=30,
    include_temporal_patterns=True,
    include_concept_drift=True,
    include_patient_heterogeneity=True,
    seed=SEED
)

data = data_generator.generate_dataset()
migraine_labels = data_generator.generate_migraine_labels(data)

# Print data statistics
print(f"Generated data with {len(migraine_labels)} samples")
print(f"Migraine prevalence: {migraine_labels.mean():.2f}")

# Step 2: Split data into train, validation, and test sets
train_indices, test_indices = train_test_split(
    np.arange(len(migraine_labels)),
    test_size=0.2,
    stratify=migraine_labels,
    random_state=SEED
)

train_indices, val_indices = train_test_split(
    train_indices,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 of original data
    stratify=migraine_labels.iloc[train_indices],
    random_state=SEED
)

print(f"Train set: {len(train_indices)} samples")
print(f"Validation set: {len(val_indices)} samples")
print(f"Test set: {len(test_indices)} samples")

# Step 3: Apply advanced preprocessing
print("Applying advanced preprocessing...")
preprocessor = AdvancedDataPreprocessor(
    window_size=5,
    n_features_to_select=30,
    use_derivatives=True,
    use_interactions=True,
    use_temporal_patterns=True,
    use_smote=True,
    random_state=SEED
)

# Split data into train, validation, and test sets
train_data = {modality: df.iloc[train_indices] for modality, df in data.items()}
val_data = {modality: df.iloc[val_indices] for modality, df in data.items()}
test_data = {modality: df.iloc[test_indices] for modality, df in data.items()}

# Fit and transform training data
train_data_processed, train_labels_processed = preprocessor.fit_transform(
    train_data,
    migraine_labels.iloc[train_indices]
)

# Transform validation and test data
val_data_processed = preprocessor.transform(val_data)
test_data_processed = preprocessor.transform(test_data)

# Get original labels for validation and test sets
val_labels = migraine_labels.iloc[val_indices].values
test_labels = migraine_labels.iloc[test_indices].values

# Step 4: Create data loaders
print("Creating data loaders...")
data_loader = EnhancedDataLoader(
    batch_size=32,
    shuffle=True,
    sequence_length=7,
    device=device
)

train_loader = data_loader.create_data_loader(
    train_data_processed,
    train_labels_processed,
    is_sequential=True
)

val_loader = data_loader.create_data_loader(
    val_data_processed,
    val_labels,
    is_sequential=True
)

test_loader = data_loader.create_data_loader(
    test_data_processed,
    test_labels,
    is_sequential=False
)

# Step 5: Create enhanced expert models
print("Creating enhanced expert models...")
expert_names = ['sleep', 'weather', 'stress_diet', 'physiological']

# Get input shapes from the processed data
input_shapes = {
    'sleep': train_data_processed['sleep_data'].shape[1],
    'weather': train_data_processed['weather_data'].shape[1],
    'stress_diet': train_data_processed['stress_diet_data'].shape[1],
    'physiological': train_data_processed['physiological_data'].shape[1]
}

# Create expert models
experts = {
    'sleep': EnhancedSleepExpert(
        input_dim=input_shapes['sleep'],
        hidden_dim=128,
        dropout_rate=0.3,
        use_attention=True
    ),
    'weather': EnhancedWeatherExpert(
        input_dim=input_shapes['weather'],
        hidden_dim=128,
        dropout_rate=0.3,
        use_residual=True
    ),
    'stress_diet': EnhancedStressDietExpert(
        input_dim=input_shapes['stress_diet'],
        hidden_dim=128,
        dropout_rate=0.3,
        use_cross_feature=True
    ),
    'physiological': EnhancedPhysiologicalExpert(
        input_dim=input_shapes['physiological'],
        hidden_dim=128,
        dropout_rate=0.3,
        use_gru=True
    )
}

# Step 6: Create enhanced gating network
print("Creating enhanced gating network...")
gating_network = EnhancedGatingNetwork(
    input_dims=input_shapes,
    hidden_dim=128,
    dropout_rate=0.3,
    num_experts=len(expert_names),
    top_k=3,
    use_gumbel_softmax=True,
    temperature=1.0,
    load_balance_coef=0.05
)

# Step 7: Create fusion mechanism
print("Creating fusion mechanism...")
fusion_mechanism = MigraineFusion(
    expert_output_dim=1,
    num_experts=len(expert_names)
)

# Step 8: Create FuseMoE model
print("Creating FuseMoE model...")
class EnhancedMigraineMoEModel(nn.Module):
    def __init__(self, experts, gating_network, fusion_mechanism):
        super(EnhancedMigraineMoEModel, self).__init__()
        self.experts = nn.ModuleDict(experts)
        self.gating = gating_network
        self.fusion = fusion_mechanism
    
    def forward(self, inputs):
        # Get expert outputs
        expert_outputs = {}
        for name, expert in self.experts.items():
            expert_outputs[name] = expert(inputs[f"{name}_data"])
        
        # Get gating outputs
        gating_outputs, aux_loss = self.gating(inputs)
        
        # Apply fusion
        output = self.fusion(expert_outputs, gating_outputs)
        
        return output

# Create model
model = EnhancedMigraineMoEModel(experts, gating_network, fusion_mechanism)
model.to(device)

# Step 9: Perform Bayesian hyperparameter optimization
print("Performing Bayesian hyperparameter optimization...")
param_space = {
    'learning_rate': (1e-5, 1e-2, 'log-uniform'),
    'weight_decay': (1e-6, 1e-3, 'log-uniform'),
    'dropout_rate': (0.1, 0.7, 'uniform'),
    'hidden_dim': (32, 256, 'int'),
    'top_k': (1, 4, 'int'),
    'temperature': (0.5, 5.0, 'uniform'),
    'load_balance_coef': (0.01, 0.1, 'uniform')
}

optimizer = BayesianOptimizer(
    model=model,
    param_space=param_space,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    n_trials=20,
    random_state=SEED
)

best_params, best_score = optimizer.optimize()

print(f"Best parameters: {best_params}")
print(f"Best validation score: {best_score:.4f}")

# Step 10: Update model with best parameters
print("Updating model with best parameters...")
# Update dropout rates
for expert in model.experts.values():
    for module in expert.modules():
        if isinstance(module, nn.Dropout):
            module.p = best_params['dropout_rate']

for module in model.gating.modules():
    if isinstance(module, nn.Dropout):
        module.p = best_params['dropout_rate']

# Update gating network parameters
model.gating.top_k = best_params['top_k']
model.gating.temperature = best_params['temperature']
model.gating.load_balance_coef = best_params['load_balance_coef']

# Step 11: Create optimizer and criterion
optimizer = optim.Adam(
    model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)

criterion = nn.BCELoss()

# Step 12: Train the model
print("Training the model...")
num_epochs = 30
patience = 5
best_val_loss = float('inf')
best_epoch = 0
best_model_state = None
history = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    history['train_loss'].append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())
    
    val_loss /= len(val_loader)
    history['val_loss'].append(val_loss)
    
    # Concatenate predictions and targets
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    # Calculate metrics
    val_auc = roc_auc_score(val_targets, val_preds)
    val_preds_binary = (val_preds > 0.5).astype(int)
    val_precision = precision_score(val_targets, val_preds_binary)
    val_recall = recall_score(val_targets, val_preds_binary)
    val_f1 = f1_score(val_targets, val_preds_binary)
    val_accuracy = accuracy_score(val_targets, val_preds_binary)
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
         f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
         f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, "
         f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
         f"Val Accuracy: {val_accuracy:.4f}")
    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = model.state_dict().copy()
        print(f"New best model at epoch {epoch+1}!")
    
    # Early stopping
    if epoch - best_epoch >= patience:
        print(f"Early stopping at epoch {epoch+1}!")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Step 13: Create ensemble model
print("Creating ensemble model...")
ensemble = EnhancedEnsemble(
    fusemoe_model=model,
    hidden_dim=128,
    dropout_rate=0.3,
    device=device
)

# Step 14: Train ensemble model
print("Training ensemble model...")
# Convert data loaders to format expected by ensemble
train_inputs = [train_data_processed[f"{name}_data"] for name in expert_names]
train_labels_tensor = torch.tensor(train_labels_processed, dtype=torch.float32)

val_inputs = [val_data_processed[f"{name}_data"] for name in expert_names]
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

test_inputs = [test_data_processed[f"{name}_data"] for name in expert_names]
test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

# Train ensemble
ensemble.fit(
    train_inputs=train_inputs,
    train_labels=train_labels_tensor,
    val_inputs=val_inputs,
    val_labels=val_labels_tensor,
    num_epochs=20,
    batch_size=32,
    patience=5
)

# Step 15: Evaluate models on test set
print("Evaluating models on test set...")
# Evaluate FuseMoE model
model.eval()
fusemoe_preds = []
with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(test_loader):
        # Forward pass
        outputs = model(inputs)
        fusemoe_preds.append(outputs.cpu().numpy())

fusemoe_preds = np.concatenate(fusemoe_preds)
fusemoe_preds_binary = (fusemoe_preds > 0.5).astype(int)

# Calculate FuseMoE metrics
fusemoe_metrics = {
    'accuracy': accuracy_score(test_labels, fusemoe_preds_binary),
    'precision': precision_score(test_labels, fusemoe_preds_binary),
    'recall': recall_score(test_labels, fusemoe_preds_binary),
    'f1': f1_score(test_labels, fusemoe_preds_binary),
    'auc': roc_auc_score(test_labels, fusemoe_preds)
}

print("FuseMoE Model Metrics:")
for metric, value in fusemoe_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Evaluate ensemble model
ensemble_preds = ensemble.predict(test_inputs)
ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)

# Calculate ensemble metrics
ensemble_metrics = {
    'accuracy': accuracy_score(test_labels, ensemble_preds_binary),
    'precision': precision_score(test_labels, ensemble_preds_binary),
    'recall': recall_score(test_labels, ensemble_preds_binary),
    'f1': f1_score(test_labels, ensemble_preds_binary),
    'auc': roc_auc_score(test_labels, ensemble_preds)
}

print("Ensemble Model Metrics:")
for metric, value in ensemble_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Evaluate baseline models
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=SEED
)

# Prepare data for traditional models
train_traditional = np.concatenate([arr for arr in train_data_processed.values()], axis=1)
test_traditional = np.concatenate([arr for arr in test_data_processed.values()], axis=1)

# Train Random Forest
rf_model.fit(train_traditional, train_labels_processed)
rf_preds = rf_model.predict_proba(test_traditional)[:, 1]
rf_preds_binary = (rf_preds > 0.5).astype(int)

# Calculate Random Forest metrics
rf_metrics = {
    'accuracy': accuracy_score(test_labels, rf_preds_binary),
    'precision': precision_score(test_labels, rf_preds_binary),
    'recall': recall_score(test_labels, rf_preds_binary),
    'f1': f1_score(test_labels, rf_preds_binary),
    'auc': roc_auc_score(test_labels, rf_preds)
}

print("Random Forest Metrics:")
for metric, value in rf_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Logistic Regression
lr_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    solver='liblinear',
    random_state=SEED
)

# Train Logistic Regression
lr_model.fit(train_traditional, train_labels_processed)
lr_preds = lr_model.predict_proba(test_traditional)[:, 1]
lr_preds_binary = (lr_preds > 0.5).astype(int)

# Calculate Logistic Regression metrics
lr_metrics = {
    'accuracy': accuracy_score(test_labels, lr_preds_binary),
    'precision': precision_score(test_labels, lr_preds_binary),
    'recall': recall_score(test_labels, lr_preds_binary),
    'f1': f1_score(test_labels, lr_preds_binary),
    'auc': roc_auc_score(test_labels, lr_preds)
}

print("Logistic Regression Metrics:")
for metric, value in lr_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Step 16: Analyze expert contributions
print("Analyzing expert contributions...")
# Get expert contributions
model.eval()
expert_contributions = []

with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(test_loader):
        # Get gating outputs
        gating_outputs, _ = model.gating(inputs)
        expert_contributions.append(gating_outputs.cpu().numpy())

expert_contributions = np.concatenate(expert_contributions, axis=0)
expert_contribution_means = expert_contributions.mean(axis=0)

print("Expert Contributions:")
for i, name in enumerate(expert_names):
    print(f"  {name}: {expert_contribution_means[i]:.4f}")

# Get ensemble model contributions
ensemble_contributions = ensemble.get_model_contributions(test_inputs)
ensemble_contribution_means = {
    model_name: contributions.mean()
    for model_name, contributions in ensemble_contributions.items()
}

print("Ensemble Model Contributions:")
for model_name, contribution in ensemble_contribution_means.items():
    print(f"  {model_name}: {contribution:.4f}")

# Step 17: Visualize results
print("Visualizing results...")
# Create output directory for visualizations
vis_dir = os.path.join(output_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(vis_dir, 'training_history.png'))
plt.close()

# Plot model comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
models = ['FuseMoE', 'Ensemble', 'Random Forest', 'Logistic Regression']
model_metrics = [fusemoe_metrics, ensemble_metrics, rf_metrics, lr_metrics]

plt.figure(figsize=(12, 8))
x = np.arange(len(metrics))
width = 0.2
multiplier = 0

for i, (model, metric_dict) in enumerate(zip(models, model_metrics)):
    offset = width * multiplier
    values = [metric_dict[metric] for metric in metrics]
    plt.bar(x + offset, values, width, label=model)
    multiplier += 1

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks(x + width, metrics)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.grid(True, axis='y')
plt.savefig(os.path.join(vis_dir, 'model_comparison.png'))
plt.close()

# Plot expert contributions
plt.figure(figsize=(10, 6))
plt.bar(expert_names, expert_contribution_means)
plt.xlabel('Expert')
plt.ylabel('Contribution')
plt.title('Expert Contributions')
plt.grid(True, axis='y')
plt.savefig(os.path.join(vis_dir, 'expert_contributions.png'))
plt.close()

# Plot ensemble model contributions
plt.figure(figsize=(10, 6))
plt.bar(ensemble_contribution_means.keys(), ensemble_contribution_means.values())
plt.xlabel('Model')
plt.ylabel('Contribution')
plt.title('Ensemble Model Contributions')
plt.grid(True, axis='y')
plt.savefig(os.path.join(vis_dir, 'ensemble_contributions.png'))
plt.close()

# Plot radar chart for model comparison
plt.figure(figsize=(10, 10))
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, (model, metric_dict) in enumerate(zip(models, model_metrics)):
    values = [metric_dict[metric] for metric in metrics]
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Model Comparison (Radar Chart)')
plt.savefig(os.path.join(vis_dir, 'radar_chart.png'))
plt.close()

# Plot heatmap of expert contributions
plt.figure(figsize=(12, 8))
sns.heatmap(
    expert_contributions[:20],  # Show first 20 samples
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    xticklabels=expert_names,
    yticklabels=[f'Sample {i+1}' for i in range(20)]
)
plt.xlabel('Expert')
plt.ylabel('Sample')
plt.title('Expert Contributions Heatmap (First 20 Samples)')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'contribution_heatmap.png'))
plt.close()

# Step 18: Save results
print("Saving results...")
# Save metrics
metrics_df = pd.DataFrame({
    'FuseMoE': fusemoe_metrics,
    'Ensemble': ensemble_metrics,
    'Random Forest': rf_metrics,
    'Logistic Regression': lr_metrics
})

metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'))

# Save expert contributions
expert_contributions_df = pd.DataFrame(
    expert_contributions,
    columns=expert_names
)

expert_contributions_df.to_csv(os.path.join(output_dir, 'expert_contributions.csv'))

# Save ensemble model contributions
ensemble_contributions_df = pd.DataFrame(ensemble_contributions)
ensemble_contributions_df.to_csv(os.path.join(output_dir, 'ensemble_contributions.csv'))

# Save feature importance
feature_importance = preprocessor.get_feature_importance()
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'))

# Save models
torch.save(model.state_dict(), os.path.join(output_dir, 'fusemoe_model.pt'))
ensemble.save(os.path.join(output_dir, 'ensemble_model'))

print("Optimization complete!")
print(f"Best model achieved {ensemble_metrics['auc']:.4f} ROC AUC on test set")
print(f"All results saved to {output_dir}")

# Check if we achieved >95% performance
if ensemble_metrics['auc'] > 0.95:
    print("SUCCESS: Achieved >95% performance!")
else:
    print(f"Current performance: {ensemble_metrics['auc']:.4f} ROC AUC")
    print("Further optimization may be needed to achieve >95% performance.")
