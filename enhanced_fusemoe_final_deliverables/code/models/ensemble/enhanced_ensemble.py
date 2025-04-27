"""
Enhanced Ensemble Approach for Migraine Prediction

This module implements a stacked ensemble approach that combines the Enhanced FuseMoE model
with traditional machine learning models to achieve >95% performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


class MetaLearner(nn.Module):
    """
    Meta-learner for the stacked ensemble approach.
    
    This model learns optimal weights for each base model in the ensemble.
    
    Attributes:
        input_dim: Number of base models
        hidden_dim: Size of hidden layers
        dropout_rate: Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        """
        Initialize meta-learner.
        
        Args:
            input_dim: Number of base models
            hidden_dim: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(MetaLearner, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Neural network for learning optimal weights
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for dynamic weighting
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through meta-learner.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        
        # Apply attention weights
        weighted_input = x * attention_weights
        
        # Pass through model
        output = self.model(weighted_input)
        
        return output


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    
    This loss function down-weights well-classified examples and focuses on hard examples.
    
    Attributes:
        alpha: Weighting factor for the positive class
        gamma: Focusing parameter
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Weighting factor for the positive class
            gamma: Focusing parameter
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through focal loss.
        
        Args:
            inputs: Predicted probabilities of shape [batch_size, 1]
            targets: Ground truth labels of shape [batch_size, 1]
            
        Returns:
            Loss value
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal weights
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()


class BalancedBCELoss(nn.Module):
    """
    Balanced binary cross entropy loss for addressing class imbalance.
    
    This loss function applies different weights to positive and negative examples.
    
    Attributes:
        pos_weight: Weight for positive examples
    """
    
    def __init__(self, pos_weight: float = 2.0):
        """
        Initialize balanced BCE loss.
        
        Args:
            pos_weight: Weight for positive examples
        """
        super(BalancedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through balanced BCE loss.
        
        Args:
            inputs: Predicted probabilities of shape [batch_size, 1]
            targets: Ground truth labels of shape [batch_size, 1]
            
        Returns:
            Loss value
        """
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate loss with positive weighting
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight
        
        return (weights * loss).mean()


class EnhancedEnsemble:
    """
    Enhanced ensemble for migraine prediction.
    
    This class combines the Enhanced FuseMoE model with traditional machine learning models
    using a stacked ensemble approach with a meta-learner.
    
    Attributes:
        fusemoe_model: Enhanced FuseMoE model
        rf_model: Random Forest model
        lr_model: Logistic Regression model
        meta_learner: Meta-learner for combining model predictions
        device: Device for PyTorch models
    """
    
    def __init__(self, fusemoe_model: nn.Module, hidden_dim: int = 64, dropout_rate: float = 0.2,
                device: torch.device = None):
        """
        Initialize enhanced ensemble.
        
        Args:
            fusemoe_model: Enhanced FuseMoE model
            hidden_dim: Size of hidden layers for meta-learner
            dropout_rate: Dropout probability for meta-learner
            device: Device for PyTorch models
        """
        self.fusemoe_model = fusemoe_model
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        self.lr_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        
        # Meta-learner for combining model predictions
        self.meta_learner = MetaLearner(
            input_dim=3,  # FuseMoE, Random Forest, Logistic Regression
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.balanced_bce_loss = BalancedBCELoss(pos_weight=2.0)
        
        # Device for PyTorch models
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusemoe_model.to(self.device)
        self.meta_learner.to(self.device)
        
        # Optimizer for meta-learner
        self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        
        # Trained flag
        self.trained = False
    
    def _prepare_traditional_inputs(self, inputs: List[torch.Tensor]) -> np.ndarray:
        """
        Prepare inputs for traditional machine learning models.
        
        Args:
            inputs: List of input tensors for each modality
            
        Returns:
            Numpy array of concatenated inputs
        """
        # Convert PyTorch tensors to numpy arrays
        numpy_inputs = [inp.cpu().numpy() for inp in inputs]
        
        # Flatten and concatenate
        flattened_inputs = []
        for inp in numpy_inputs:
            if len(inp.shape) > 2:
                # For sequential data, flatten the sequence dimension
                batch_size = inp.shape[0]
                flattened = inp.reshape(batch_size, -1)
            else:
                flattened = inp
            flattened_inputs.append(flattened)
        
        # Concatenate all inputs
        concatenated = np.concatenate(flattened_inputs, axis=1)
        
        return concatenated
    
    def fit(self, train_inputs: List[torch.Tensor], train_labels: torch.Tensor,
           val_inputs: List[torch.Tensor], val_labels: torch.Tensor,
           num_epochs: int = 20, batch_size: int = 32, patience: int = 5):
        """
        Train the enhanced ensemble.
        
        Args:
            train_inputs: List of input tensors for each modality (training set)
            train_labels: Ground truth labels for training set
            val_inputs: List of input tensors for each modality (validation set)
            val_labels: Ground truth labels for validation set
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
        """
        # Prepare inputs for traditional models
        train_traditional_inputs = self._prepare_traditional_inputs(train_inputs)
        val_traditional_inputs = self._prepare_traditional_inputs(val_inputs)
        
        # Convert labels to numpy arrays
        train_labels_np = train_labels.cpu().numpy()
        val_labels_np = val_labels.cpu().numpy()
        
        # Train traditional models
        print("Training Random Forest model...")
        self.rf_model.fit(train_traditional_inputs, train_labels_np.ravel())
        
        print("Training Logistic Regression model...")
        self.lr_model.fit(train_traditional_inputs, train_labels_np.ravel())
        
        # Get predictions from traditional models
        rf_train_preds = self.rf_model.predict_proba(train_traditional_inputs)[:, 1]
        lr_train_preds = self.lr_model.predict_proba(train_traditional_inputs)[:, 1]
        
        rf_val_preds = self.rf_model.predict_proba(val_traditional_inputs)[:, 1]
        lr_val_preds = self.lr_model.predict_proba(val_traditional_inputs)[:, 1]
        
        # Create data loaders for PyTorch models
        train_dataset = EnsembleDataset(train_inputs, train_labels, rf_train_preds, lr_train_preds)
        val_dataset = EnsembleDataset(val_inputs, val_labels, rf_val_preds, lr_val_preds)
        
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
        
        # Train meta-learner
        print("Training meta-learner...")
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        
        for epoch in range(num_epochs):
            # Training
            self.meta_learner.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get batch data
                inputs, labels, rf_preds, lr_preds = batch
                inputs = [inp.to(self.device) for inp in inputs]
                labels = labels.to(self.device)
                rf_preds = rf_preds.to(self.device)
                lr_preds = lr_preds.to(self.device)
                
                # Forward pass through FuseMoE model
                self.fusemoe_model.eval()  # Use eval mode for FuseMoE
                with torch.no_grad():
                    fusemoe_preds = self.fusemoe_model(inputs)
                
                # Combine predictions
                combined_preds = torch.cat([
                    fusemoe_preds, rf_preds.unsqueeze(1), lr_preds.unsqueeze(1)
                ], dim=1)
                
                # Forward pass through meta-learner
                self.meta_optimizer.zero_grad()
                outputs = self.meta_learner(combined_preds)
                
                # Calculate loss
                loss = self.focal_loss(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.meta_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.meta_learner.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Get batch data
                    inputs, labels, rf_preds, lr_preds = batch
                    inputs = [inp.to(self.device) for inp in inputs]
                    labels = labels.to(self.device)
                    rf_preds = rf_preds.to(self.device)
                    lr_preds = lr_preds.to(self.device)
                    
                    # Forward pass through FuseMoE model
                    fusemoe_preds = self.fusemoe_model(inputs)
                    
                    # Combine predictions
                    combined_preds = torch.cat([
                        fusemoe_preds, rf_preds.unsqueeze(1), lr_preds.unsqueeze(1)
                    ], dim=1)
                    
                    # Forward pass through meta-learner
                    outputs = self.meta_learner(combined_preds)
                    
                    # Calculate loss
                    loss = self.balanced_bce_loss(outputs, labels)
                    val_loss += loss.item()
                    
                    # Store predictions and labels
                    val_preds.append(outputs.cpu().numpy())
                    val_true.append(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # Concatenate predictions and labels
            val_preds = np.concatenate(val_preds)
            val_true = np.concatenate(val_true)
            
            # Calculate metrics
            val_auc = roc_auc_score(val_true, val_preds)
            val_preds_binary = (val_preds > 0.5).astype(int)
            val_precision = precision_score(val_true, val_preds_binary)
            val_recall = recall_score(val_true, val_preds_binary)
            val_f1 = f1_score(val_true, val_preds_binary)
            val_accuracy = accuracy_score(val_true, val_preds_binary)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, "
                 f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
                 f"Val Accuracy: {val_accuracy:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = self.meta_learner.state_dict().copy()
                print(f"New best model at epoch {epoch+1}!")
            
            # Early stopping
            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        
        # Load best model
        if best_state_dict is not None:
            self.meta_learner.load_state_dict(best_state_dict)
        
        self.trained = True
        
        print("Training complete!")
    
    def predict(self, inputs: List[torch.Tensor]) -> np.ndarray:
        """
        Make predictions with the enhanced ensemble.
        
        Args:
            inputs: List of input tensors for each modality
            
        Returns:
            Numpy array of predictions
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet!")
        
        # Prepare inputs for traditional models
        traditional_inputs = self._prepare_traditional_inputs(inputs)
        
        # Get predictions from traditional models
        rf_preds = self.rf_model.predict_proba(traditional_inputs)[:, 1]
        lr_preds = self.lr_model.predict_proba(traditional_inputs)[:, 1]
        
        # Convert to PyTorch tensors
        rf_preds = torch.tensor(rf_preds, dtype=torch.float32).to(self.device)
        lr_preds = torch.tensor(lr_preds, dtype=torch.float32).to(self.device)
        
        # Move inputs to device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Forward pass through FuseMoE model
        self.fusemoe_model.eval()
        with torch.no_grad():
            fusemoe_preds = self.fusemoe_model(inputs)
        
        # Combine predictions
        combined_preds = torch.cat([
            fusemoe_preds, rf_preds.unsqueeze(1), lr_preds.unsqueeze(1)
        ], dim=1)
        
        # Forward pass through meta-learner
        self.meta_learner.eval()
        with torch.no_grad():
            outputs = self.meta_learner(combined_preds)
        
        return outputs.cpu().numpy()
    
    def evaluate(self, inputs: List[torch.Tensor], labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the enhanced ensemble.
        
        Args:
            inputs: List of input tensors for each modality
            labels: Ground truth labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet!")
        
        # Get predictions
        preds = self.predict(inputs)
        
        # Convert labels to numpy array
        labels_np = labels.cpu().numpy()
        
        # Calculate metrics
        auc = roc_auc_score(labels_np, preds)
        preds_binary = (preds > 0.5).astype(int)
        precision = precision_score(labels_np, preds_binary)
        recall = recall_score(labels_np, preds_binary)
        f1 = f1_score(labels_np, preds_binary)
        accuracy = accuracy_score(labels_np, preds_binary)
        
        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
    
    def get_model_contributions(self, inputs: List[torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Get the contribution of each model to the final predictions.
        
        Args:
            inputs: List of input tensors for each modality
            
        Returns:
            Dictionary of model contributions
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet!")
        
        # Prepare inputs for traditional models
        traditional_inputs = self._prepare_traditional_inputs(inputs)
        
        # Get predictions from traditional models
        rf_preds = self.rf_model.predict_proba(traditional_inputs)[:, 1]
        lr_preds = self.lr_model.predict_proba(traditional_inputs)[:, 1]
        
        # Convert to PyTorch tensors
        rf_preds = torch.tensor(rf_preds, dtype=torch.float32).to(self.device)
        lr_preds = torch.tensor(lr_preds, dtype=torch.float32).to(self.device)
        
        # Move inputs to device
        inputs = [inp.to(self.device) for inp in inputs]
        
        # Forward pass through FuseMoE model
        self.fusemoe_model.eval()
        with torch.no_grad():
            fusemoe_preds = self.fusemoe_model(inputs)
        
        # Combine predictions
        combined_preds = torch.cat([
            fusemoe_preds, rf_preds.unsqueeze(1), lr_preds.unsqueeze(1)
        ], dim=1)
        
        # Forward pass through meta-learner to get attention weights
        self.meta_learner.eval()
        with torch.no_grad():
            # Get attention weights from meta-learner
            attention_weights = F.softmax(self.meta_learner.attention(combined_preds), dim=1)
        
        # Convert to numpy arrays
        attention_weights = attention_weights.cpu().numpy()
        
        return {
            'fusemoe': attention_weights[:, 0],
            'random_forest': attention_weights[:, 1],
            'logistic_regression': attention_weights[:, 2]
        }
    
    def save(self, path: str):
        """
        Save the enhanced ensemble.
        
        Args:
            path: Path to save the model
        """
        if not self.trained:
            raise RuntimeError("Model has not been trained yet!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FuseMoE model
        torch.save(self.fusemoe_model.state_dict(), f"{path}_fusemoe.pt")
        
        # Save meta-learner
        torch.save(self.meta_learner.state_dict(), f"{path}_meta_learner.pt")
        
        # Save traditional models
        import joblib
        joblib.dump(self.rf_model, f"{path}_rf_model.pkl")
        joblib.dump(self.lr_model, f"{path}_lr_model.pkl")
    
    @classmethod
    def load(cls, path: str, fusemoe_model: nn.Module, hidden_dim: int = 64, dropout_rate: float = 0.2,
            device: torch.device = None) -> 'EnhancedEnsemble':
        """
        Load the enhanced ensemble.
        
        Args:
            path: Path to load the model from
            fusemoe_model: Enhanced FuseMoE model architecture (will be loaded with weights)
            hidden_dim: Size of hidden layers for meta-learner
            dropout_rate: Dropout probability for meta-learner
            device: Device for PyTorch models
            
        Returns:
            Loaded enhanced ensemble
        """
        # Create ensemble
        ensemble = cls(
            fusemoe_model=fusemoe_model,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            device=device
        )
        
        # Load FuseMoE model
        fusemoe_state_dict = torch.load(f"{path}_fusemoe.pt", map_location=ensemble.device)
        ensemble.fusemoe_model.load_state_dict(fusemoe_state_dict)
        
        # Load meta-learner
        meta_learner_state_dict = torch.load(f"{path}_meta_learner.pt", map_location=ensemble.device)
        ensemble.meta_learner.load_state_dict(meta_learner_state_dict)
        
        # Load traditional models
        import joblib
        ensemble.rf_model = joblib.load(f"{path}_rf_model.pkl")
        ensemble.lr_model = joblib.load(f"{path}_lr_model.pkl")
        
        ensemble.trained = True
        
        return ensemble


class EnsembleDataset(torch.utils.data.Dataset):
    """
    Dataset for training the enhanced ensemble.
    
    Attributes:
        inputs: List of input tensors for each modality
        labels: Ground truth labels
        rf_preds: Predictions from Random Forest model
        lr_preds: Predictions from Logistic Regression model
    """
    
    def __init__(self, inputs: List[torch.Tensor], labels: torch.Tensor,
                rf_preds: np.ndarray, lr_preds: np.ndarray):
        """
        Initialize ensemble dataset.
        
        Args:
            inputs: List of input tensors for each modality
            labels: Ground truth labels
            rf_preds: Predictions from Random Forest model
            lr_preds: Predictions from Logistic Regression model
        """
        self.inputs = inputs
        self.labels = labels
        self.rf_preds = torch.tensor(rf_preds, dtype=torch.float32)
        self.lr_preds = torch.tensor(lr_preds, dtype=torch.float32)
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (inputs, label, rf_pred, lr_pred)
        """
        # Get inputs for each modality
        sample_inputs = [inp[idx] for inp in self.inputs]
        
        # Get label
        sample_label = self.labels[idx]
        
        # Get predictions from traditional models
        sample_rf_pred = self.rf_preds[idx]
        sample_lr_pred = self.lr_preds[idx]
        
        return sample_inputs, sample_label, sample_rf_pred, sample_lr_pred


class ModelDistillation:
    """
    Model distillation for transferring knowledge from ensemble to a single model.
    
    This class implements knowledge distillation to transfer the knowledge from
    the enhanced ensemble to a single optimized model.
    
    Attributes:
        teacher_model: Enhanced ensemble (teacher)
        student_model: Single model (student)
        device: Device for PyTorch models
    """
    
    def __init__(self, teacher_model: EnhancedEnsemble, student_model: nn.Module,
                temperature: float = 2.0, alpha: float = 0.5,
                device: torch.device = None):
        """
        Initialize model distillation.
        
        Args:
            teacher_model: Enhanced ensemble (teacher)
            student_model: Single model (student)
            temperature: Temperature for softening the teacher's predictions
            alpha: Weight for balancing soft and hard targets
            device: Device for PyTorch models
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Device for PyTorch models
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student_model.to(self.device)
        
        # Optimizer for student model
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=0.001)
        
        # Loss functions
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.hard_loss = nn.BCELoss()
    
    def fit(self, train_inputs: List[torch.Tensor], train_labels: torch.Tensor,
           val_inputs: List[torch.Tensor], val_labels: torch.Tensor,
           num_epochs: int = 20, batch_size: int = 32, patience: int = 5):
        """
        Train the student model using knowledge distillation.
        
        Args:
            train_inputs: List of input tensors for each modality (training set)
            train_labels: Ground truth labels for training set
            val_inputs: List of input tensors for each modality (validation set)
            val_labels: Ground truth labels for validation set
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Patience for early stopping
        """
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            *train_inputs, train_labels
        )
        val_dataset = torch.utils.data.TensorDataset(
            *val_inputs, val_labels
        )
        
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
        
        # Training
        print("Training student model with knowledge distillation...")
        best_val_loss = float('inf')
        best_epoch = 0
        best_state_dict = None
        
        for epoch in range(num_epochs):
            # Training
            self.student_model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Get batch data
                *inputs, labels = batch
                inputs = [inp.to(self.device) for inp in inputs]
                labels = labels.to(self.device)
                
                # Get teacher predictions
                teacher_preds = torch.tensor(
                    self.teacher_model.predict(inputs),
                    dtype=torch.float32
                ).to(self.device)
                
                # Forward pass through student model
                self.optimizer.zero_grad()
                student_preds = self.student_model(inputs)
                
                # Calculate soft targets (teacher predictions)
                soft_targets = torch.pow(teacher_preds, 1.0 / self.temperature)
                
                # Calculate hard targets (ground truth)
                hard_targets = labels
                
                # Calculate distillation loss
                soft_loss = self.kd_loss(
                    F.log_softmax(student_preds / self.temperature, dim=1),
                    F.softmax(soft_targets / self.temperature, dim=1)
                )
                hard_loss = self.hard_loss(student_preds, hard_targets)
                
                # Combine losses
                loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.student_model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    # Get batch data
                    *inputs, labels = batch
                    inputs = [inp.to(self.device) for inp in inputs]
                    labels = labels.to(self.device)
                    
                    # Forward pass through student model
                    student_preds = self.student_model(inputs)
                    
                    # Calculate loss
                    loss = self.hard_loss(student_preds, labels)
                    val_loss += loss.item()
                    
                    # Store predictions and labels
                    val_preds.append(student_preds.cpu().numpy())
                    val_true.append(labels.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # Concatenate predictions and labels
            val_preds = np.concatenate(val_preds)
            val_true = np.concatenate(val_true)
            
            # Calculate metrics
            val_auc = roc_auc_score(val_true, val_preds)
            val_preds_binary = (val_preds > 0.5).astype(int)
            val_precision = precision_score(val_true, val_preds_binary)
            val_recall = recall_score(val_true, val_preds_binary)
            val_f1 = f1_score(val_true, val_preds_binary)
            val_accuracy = accuracy_score(val_true, val_preds_binary)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, "
                 f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
                 f"Val Accuracy: {val_accuracy:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = self.student_model.state_dict().copy()
                print(f"New best model at epoch {epoch+1}!")
            
            # Early stopping
            if epoch - best_epoch >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
        
        # Load best model
        if best_state_dict is not None:
            self.student_model.load_state_dict(best_state_dict)
        
        print("Training complete!")
    
    def save_student(self, path: str):
        """
        Save the student model.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save student model
        torch.save(self.student_model.state_dict(), path)
