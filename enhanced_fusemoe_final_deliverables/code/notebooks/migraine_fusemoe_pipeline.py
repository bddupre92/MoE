#!/usr/bin/env python
# coding: utf-8

# # Enhanced FuseMoE for Migraine Prediction
# 
# This notebook demonstrates the complete pipeline for migraine prediction using the Enhanced FuseMoE system with PyGMO integration for evolutionary optimization.
# 
# The Enhanced FuseMoE system extends the [FuseMoE](https://github.com/aaronhan223/FuseMoE) framework with:
# 1. Domain-specific expert models for migraine prediction
# 2. PyGMO integration for evolutionary optimization
# 3. Advanced fusion mechanisms
# 4. Comprehensive evaluation metrics
# 5. Visualization components
# 
# Let's walk through the entire pipeline from data generation to model evaluation and visualization.

# ## 1. Setup and Imports
# 
# First, let's import all the necessary modules and set up our environment.

# In[ ]:


import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Enhanced FuseMoE modules
from utils.preprocessing.data_generator import MigraineSyntheticDataGenerator
from utils.preprocessing.data_preprocessor import DataPreprocessor, MigraineDataPreprocessor
from utils.training_pipeline import MigraineTrainer, PyGMOTrainingPipeline
from utils.evaluation.metrics import MigrainePerformanceMetrics, SimpleMetrics
from optimization.evolutionary_algorithms.optimization_manager import OptimizationManager
from optimization.pygmo_model_optimizer import PyGMOModelOptimizer
from models.experts.expert_registry import ExpertRegistry
from models.experts.sleep_expert import create_sleep_expert
from models.experts.weather_expert import create_weather_expert
from models.experts.stress_diet_expert import create_stress_diet_expert
from models.experts.physio_expert import create_physio_expert
from models.experts.input_adapter import wrap_expert, InputShapeAdapter
from models.gating.migraine_gating import MigraineGating
from models.fusion.migraine_fusion import MigraineFusion, MigraineFusionMoE

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "migraine_prediction_results"
os.makedirs(output_dir, exist_ok=True)


# ## 2. Data Generation
# 
# Next, let's generate synthetic data for migraine prediction. The data includes features from four domains:
# 1. Sleep data
# 2. Weather data
# 3. Stress and diet data
# 4. Physiological data

# In[ ]:


# Create data generator
data_generator = MigraineSyntheticDataGenerator(seed=SEED)

# Generate data
# Since MigraineSyntheticDataGenerator doesn't have the same parameters,
# we'll need to manually generate the data for multiple patients and days
num_patients = 50
days_per_patient = 20
total_samples = num_patients * days_per_patient
migraine_probability = 0.2

# Generate synthetic data for each modality
sleep_data = data_generator.generate_sleep_data(total_samples)
weather_data = data_generator.generate_weather_data(total_samples)
stress_diet_data = data_generator.generate_stress_diet_data(total_samples)
physio_data = data_generator.generate_physiological_data(total_samples)

# Generate migraine labels
migraine_labels = data_generator.generate_migraine_labels(
    sleep_data, weather_data, stress_diet_data, physio_data
)

# Split data into train, validation, and test sets
train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
num_train = int(total_samples * train_ratio)
num_val = int(total_samples * val_ratio)
num_test = total_samples - num_train - num_val

# Create indices for splitting
indices = np.random.permutation(total_samples)
train_indices = indices[:num_train]
val_indices = indices[num_train:num_train+num_val]
test_indices = indices[num_train+num_val:]

# Split data
X_train = [
    torch.tensor(sleep_data.iloc[train_indices].values, dtype=torch.float32),
    torch.tensor(weather_data.iloc[train_indices].values, dtype=torch.float32),
    torch.tensor(stress_diet_data.iloc[train_indices].values, dtype=torch.float32),
    torch.tensor(physio_data.iloc[train_indices].values, dtype=torch.float32)
]
y_train = torch.tensor(migraine_labels.iloc[train_indices].values, dtype=torch.float32).unsqueeze(1)

X_val = [
    torch.tensor(sleep_data.iloc[val_indices].values, dtype=torch.float32),
    torch.tensor(weather_data.iloc[val_indices].values, dtype=torch.float32),
    torch.tensor(stress_diet_data.iloc[val_indices].values, dtype=torch.float32),
    torch.tensor(physio_data.iloc[val_indices].values, dtype=torch.float32)
]
y_val = torch.tensor(migraine_labels.iloc[val_indices].values, dtype=torch.float32).unsqueeze(1)

X_test = [
    torch.tensor(sleep_data.iloc[test_indices].values, dtype=torch.float32),
    torch.tensor(weather_data.iloc[test_indices].values, dtype=torch.float32),
    torch.tensor(stress_diet_data.iloc[test_indices].values, dtype=torch.float32),
    torch.tensor(physio_data.iloc[test_indices].values, dtype=torch.float32)
]
y_test = torch.tensor(migraine_labels.iloc[test_indices].values, dtype=torch.float32).unsqueeze(1)

# Create data dictionary
data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test
}

# Print data shapes
print("Data shapes:")
print(f"X_train: {[x.shape for x in data['X_train']]}")
print(f"y_train: {data['y_train'].shape}")
print(f"X_val: {[x.shape for x in data['X_val']]}")
print(f"y_val: {data['y_val'].shape}")
print(f"X_test: {[x.shape for x in data['X_test']]}")
print(f"y_test: {data['y_test'].shape}")


# ## 3. Data Preprocessing
# 
# Now, let's preprocess the data to prepare it for model training.

# In[ ]:


# Create preprocessor
preprocessor = MigraineDataPreprocessor()

# Preprocess data using fit_transform instead of preprocess
preprocessed_data = preprocessor.fit_transform(data)

# Create custom dataset for model training
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, expert_inputs, targets, expert_names):
        self.expert_inputs = expert_inputs
        self.targets = targets
        self.expert_names = expert_names
        self.num_samples = len(targets)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create input dictionary
        inputs = {
            name: self.expert_inputs[i][idx]
            for i, name in enumerate(self.expert_names)
        }
        
        # Get target
        target = self.targets[idx]
        
        return inputs, target

# Create datasets
expert_names = ["sleep_expert", "weather_expert", "stress_diet_expert", "physio_expert"]

train_dataset = CustomDataset(
    expert_inputs=preprocessed_data['X_train'],
    targets=preprocessed_data['y_train'],
    expert_names=expert_names
)

val_dataset = CustomDataset(
    expert_inputs=preprocessed_data['X_val'],
    targets=preprocessed_data['y_val'],
    expert_names=expert_names
)

test_dataset = CustomDataset(
    expert_inputs=preprocessed_data['X_test'],
    targets=preprocessed_data['y_test'],
    expert_names=expert_names
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ## 4. Model Creation
# 
# Let's create the expert models, gating network, and fusion mechanism for our Enhanced FuseMoE system.

# In[ ]:


# Create expert registry
expert_registry = ExpertRegistry()

# Create expert models
sleep_expert = create_sleep_expert(
    input_dim=preprocessed_data['X_train'][0].shape[-1],
    hidden_dim=64,
    output_dim=32
)

weather_expert = create_weather_expert(
    input_dim=preprocessed_data['X_train'][1].shape[-1],
    hidden_dim=64,
    output_dim=32
)

stress_diet_expert = create_stress_diet_expert(
    input_dim=preprocessed_data['X_train'][2].shape[-1],
    hidden_dim=64,
    output_dim=32
)

physio_expert = create_physio_expert(
    input_dim=preprocessed_data['X_train'][3].shape[-1],
    hidden_dim=64,
    output_dim=32
)

# Wrap expert models
wrapped_sleep_expert = wrap_expert(sleep_expert, "sleep_expert")
wrapped_weather_expert = wrap_expert(weather_expert, "weather_expert")
wrapped_stress_diet_expert = wrap_expert(stress_diet_expert, "stress_diet_expert")
wrapped_physio_expert = wrap_expert(physio_expert, "physio_expert")

# Register experts
expert_registry.register_expert("sleep_expert", wrapped_sleep_expert)
expert_registry.register_expert("weather_expert", wrapped_weather_expert)
expert_registry.register_expert("stress_diet_expert", wrapped_stress_diet_expert)
expert_registry.register_expert("physio_expert", wrapped_physio_expert)

# Create gating network
gating = MigraineGating(
    input_dims=[preprocessed_data['X_train'][i].shape[-1] for i in range(4)],
    num_experts=len(expert_registry),
    hidden_dim=64,
    top_k=2
)

# Create fusion model
model = MigraineFusionMoE(
    expert_registry=expert_registry,
    gating=gating,
    output_dim=1
)

# Move model to device
model.to(device)

# Print model summary
print("Model created with the following components:")
print(f"- Expert Registry: {len(expert_registry)} experts")
print(f"- Gating Network: {gating.__class__.__name__}")
print(f"- Fusion Mechanism: {model.__class__.__name__}")


# ## 5. PyGMO Optimization
# 
# Now, let's use PyGMO to optimize the hyperparameters of our model.

# In[ ]:


# Define parameter space
param_space = {
    'learning_rate': (0.0001, 0.01),
    'batch_size': (8, 64),
    'hidden_dim': (32, 128),
    'dropout': (0.0, 0.5),
    'weight_decay': (0.0, 0.01)
}

# Create data loaders for optimization
train_loader_opt = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader_opt = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader_opt = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Extract experts, gating network, and fusion mechanism from the model
experts = [expert_registry.get(name) for name in expert_names]
gating_network = model.gating
fusion_mechanism = model

# Create PyGMO model optimizer
optimizer = PyGMOModelOptimizer(
    experts=experts,
    gating_network=gating_network,
    fusion_mechanism=fusion_mechanism,
    train_loader=train_loader_opt,
    val_loader=val_loader_opt,
    test_loader=test_loader_opt,
    param_bounds=param_space,
    fitness_metric='auc',
    num_epochs=10,
    patience=3,
    device=device
)

# Run optimization - use optimize_end_to_end since there's no direct optimize method
best_params_dict = optimizer.optimize_end_to_end()
best_params = best_params_dict['best_params']
best_score = best_params_dict['best_fitness']

# Print optimization results
print("Optimization Results:")
print(f"Best Score: {best_score:.4f}")
print("Best Parameters:")
for param, value in best_params.items():
    print(f"- {param}: {value}")


# ## 6. Model Training
# 
# Let's train our model using the optimized hyperparameters.

# In[ ]:


# Create optimizer with optimized learning rate
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)

# Create criterion
criterion = torch.nn.BCEWithLogitsLoss()

# Create trainer
trainer = MigraineTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    checkpoint_dir=os.path.join(output_dir, 'checkpoints')
)

# Create data loaders with optimized batch size
# Use a default batch size of 32 if 'batch_size' is not in best_params
default_batch_size = 32
batch_size = int(best_params.get('batch_size', default_batch_size))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20
)

# Plot training history
plt.figure(figsize=(12, 4))

# Extract metrics from history
train_loss = [epoch_metrics['loss'] for epoch_metrics in history['train_history']]
val_loss = [epoch_metrics['loss'] for epoch_metrics in history['val_history']]
train_acc = [epoch_metrics.get('accuracy', 0) for epoch_metrics in history['train_history']]
val_acc = [epoch_metrics.get('accuracy', 0) for epoch_metrics in history['val_history']]

plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train')
plt.plot(val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Train')
plt.plot(val_acc, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.show()


# ## 7. Model Evaluation
# 
# Let's evaluate our trained model on the test set.

# In[ ]:


# Load best model if available, otherwise continue with the current model
if hasattr(trainer, 'best_model_path') and trainer.best_model_path is not None:
    trainer.load_checkpoint(trainer.best_model_path)
else:
    print("No best model checkpoint available, using current model state")

# Create test loader
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,  # Use the same batch_size variable defined earlier
    shuffle=False
)

# Create metrics calculator
metrics_calculator = MigrainePerformanceMetrics(
    model=model,
    device=device
)

# Calculate metrics
metrics = metrics_calculator.calculate_metrics(
    data_loader=test_loader,
    output_dir=os.path.join(output_dir, 'metrics')
)

# Print metrics
print("Test Metrics:")
for metric, value in metrics.items():
    print(f"- {metric}: {value:.4f}")

# Calculate expert contributions
contributions = metrics_calculator.calculate_expert_contributions(
    data_loader=test_loader,
    output_dir=os.path.join(output_dir, 'metrics')
)

# Print expert contributions
print("\nExpert Contributions:")
for expert, contribution in contributions.items():
    print(f"- {expert}: {contribution:.4f}")

# Plot expert contributions
plt.figure(figsize=(10, 6))
plt.bar(contributions.keys(), contributions.values())
plt.xlabel('Expert')
plt.ylabel('Average Contribution')
plt.title('Expert Contributions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'expert_contributions.png'))
plt.show()


# ## 8. Comparison with Baseline Models
# 
# Let's compare our Enhanced FuseMoE model with baseline models.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to evaluate baseline models
def evaluate_baseline(model_name, model, X_train, y_train, X_test, y_test):
    # Train model
    model.fit(X_train, y_train.flatten())
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics

# Prepare data for baseline models
# Concatenate all expert inputs
X_train_concat = np.concatenate([x.numpy() for x in preprocessed_data['X_train']], axis=1)
X_test_concat = np.concatenate([x.numpy() for x in preprocessed_data['X_test']], axis=1)
y_train_np = preprocessed_data['y_train'].numpy()
y_test_np = preprocessed_data['y_test'].numpy()

# Create and evaluate baseline models
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=SEED),
    'Random Forest': RandomForestClassifier(random_state=SEED)
}

baseline_results = {}
for name, model in baseline_models.items():
    baseline_results[name] = evaluate_baseline(
        model_name=name,
        model=model,
        X_train=X_train_concat,
        y_train=y_train_np,
        X_test=X_test_concat,
        y_test=y_test_np
    )

# Add Enhanced FuseMoE results
baseline_results['Enhanced FuseMoE'] = metrics

# Print comparison
print("Model Comparison:")
for model_name, model_metrics in baseline_results.items():
    print(f"\n{model_name}:")
    for metric, value in model_metrics.items():
        print(f"- {metric}: {value:.4f}")

# Plot comparison
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
model_names = list(baseline_results.keys())

plt.figure(figsize=(12, 8))
bar_width = 0.25
index = np.arange(len(metrics_to_plot))

for i, model_name in enumerate(model_names):
    values = [baseline_results[model_name][metric] for metric in metrics_to_plot]
    plt.bar(index + i * bar_width, values, bar_width, label=model_name)

plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Model Comparison')
plt.xticks(index + bar_width, metrics_to_plot)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
plt.show()


# ## 9. Visualization Dashboard
# 
# Finally, let's create a visualization dashboard for our migraine prediction model.

# In[ ]:


# This code would typically launch a Streamlit dashboard
# For this notebook, we'll just display some visualizations

# Load test predictions
predictions = np.load(os.path.join(output_dir, 'metrics', 'test_predictions.npz'))
y_true = predictions['y_test']
y_pred = predictions['y_pred_test']

# Plot ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.show()

# Plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_binary = (y_pred >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.xticks([0.5, 1.5], ['No Migraine', 'Migraine'])
plt.yticks([0.5, 1.5], ['No Migraine', 'Migraine'])
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Plot feature importance
# For this example, we'll use the Random Forest feature importance
rf_model = baseline_models['Random Forest']
feature_names = [f"Feature {i+1}" for i in range(X_train_concat.shape[1])]
feature_importance = rf_model.feature_importances_

# Sort feature importance
indices = np.argsort(feature_importance)[-10:]  # Top 10 features
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), feature_importance[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.show()


# ## 10. Conclusion
# 
# In this notebook, we've demonstrated the complete pipeline for migraine prediction using the Enhanced FuseMoE system with PyGMO integration. The key components of our system include:
# 
# 1. **Data Generation and Preprocessing**: We generated synthetic data for migraine prediction across four domains (sleep, weather, stress/diet, physiological) and preprocessed it for model training.
# 
# 2. **Expert Models**: We created domain-specific expert models for each data modality, wrapped them with input adapters, and registered them in an expert registry.
# 
# 3. **Gating Network**: We implemented a migraine-specific gating network that determines which experts to use for each input.
# 
# 4. **Fusion Mechanism**: We used a mixture of experts fusion mechanism to combine the outputs of the selected experts.
# 
# 5. **PyGMO Optimization**: We leveraged PyGMO's evolutionary algorithms to optimize the hyperparameters of our model.
# 
# 6. **Model Training and Evaluation**: We trained our model using the optimized hyperparameters and evaluated it on the test set.
# 
# 7. **Comparison with Baselines**: We compared our Enhanced FuseMoE model with baseline models (Logistic Regression and Random Forest).
# 
# 8. **Visualization**: We created visualizations to help understand the model's performance and behavior.
# 
# The Enhanced FuseMoE system demonstrates the power of combining domain-specific expert models with evolutionary optimization for migraine prediction. The system can be extended to include additional expert models, more sophisticated fusion mechanisms, and advanced visualization components.
