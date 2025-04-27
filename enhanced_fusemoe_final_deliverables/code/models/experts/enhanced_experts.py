"""
Enhanced Sleep Expert Model for FuseMoE

This module provides an enhanced sleep expert model with transformer architecture
for the FuseMoE system to better capture temporal patterns in sleep data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union


class TransformerBlock(nn.Module):
    """
    Transformer block for sequence modeling.
    
    Attributes:
        attention: Multi-head attention layer
        norm1: Layer normalization for attention output
        norm2: Layer normalization for feed-forward output
        feed_forward: Feed-forward network
        dropout: Dropout layer
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, embed_dim)
            
        Returns:
            Output tensor of shape (seq_len, batch_size, embed_dim)
        """
        # Multi-head attention with residual connection and layer normalization
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class FeatureWiseAttention(nn.Module):
    """
    Feature-wise attention mechanism to focus on the most predictive sleep metrics.
    
    Attributes:
        query: Query projection
        key: Key projection
        value: Value projection
        scale: Scaling factor for attention scores
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize feature-wise attention.
        
        Args:
            input_dim: Input dimension
        """
        super(FeatureWiseAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Project inputs to queries, keys, and values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(weights, v)
        
        return output


class EnhancedSleepExpert(nn.Module):
    """
    Enhanced sleep expert model with transformer architecture.
    
    This model uses a transformer architecture to better capture temporal patterns
    in sleep data, with feature-wise attention to focus on the most predictive metrics.
    
    Attributes:
        embedding: Input embedding layer
        positional_encoding: Positional encoding for transformer
        transformer_blocks: List of transformer blocks
        feature_attention: Feature-wise attention mechanism
        fc_layers: Fully connected layers for final prediction
        dropout: Dropout layer
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3,
                num_heads: int = 4, dropout: float = 0.2):
        """
        Initialize enhanced sleep expert model.
        
        Args:
            input_dim: Input dimension (number of sleep features)
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(EnhancedSleepExpert, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.register_buffer("positional_encoding", self._create_positional_encoding(100, hidden_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature-wise attention
        self.feature_attention = FeatureWiseAttention(hidden_dim)
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def _create_positional_encoding(self, max_seq_len: int, hidden_dim: int) -> torch.Tensor:
        """
        Create positional encoding for transformer.
        
        Args:
            max_seq_len: Maximum sequence length
            hidden_dim: Hidden dimension
            
        Returns:
            Positional encoding tensor of shape (max_seq_len, hidden_dim)
        """
        # Initialize positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        
        # Create position indices
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sinusoidal functions
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / hidden_dim))
        
        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Get sequence length
        seq_len = x.size(1)
        
        # Apply input embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transpose for transformer (seq_len, batch_size, hidden_dim)
        x = x.transpose(0, 1)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Transpose back (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)
        
        # Apply feature-wise attention
        x = self.feature_attention(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Apply fully connected layers with residual connections
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x + residual)  # Residual connection
        
        residual = x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final layer
        x = self.fc3(x)
        
        return x


class EnhancedWeatherExpert(nn.Module):
    """
    Enhanced weather expert model with attention to pressure changes and seasonal patterns.
    
    This model uses attention mechanisms to focus on important weather features,
    particularly pressure changes which are known migraine triggers.
    
    Attributes:
        feature_extractor: Convolutional layers for feature extraction
        attention: Attention mechanism for feature importance
        fc_layers: Fully connected layers for final prediction
        dropout: Dropout layer
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128, dropout: float = 0.2):
        """
        Initialize enhanced weather expert model.
        
        Args:
            input_dim: Input dimension (number of weather features)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(EnhancedWeatherExpert, self).__init__()
        
        # Feature extraction with 1D convolutions
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Pressure change specific attention
        self.pressure_attention = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Temperature change specific attention
        self.temp_attention = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Cross-feature interaction layer
        self.interaction_layer = nn.Bilinear(hidden_dim, hidden_dim // 2, hidden_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Extract pressure change and temperature change features
        pressure_change = x[:, 6:7]  # Assuming pressure_change is the 7th feature
        temp_change = x[:, 7:8]      # Assuming temperature_change is the 8th feature
        
        # Process pressure and temperature changes separately
        pressure_features = self.pressure_attention(pressure_change)
        temp_features = self.temp_attention(temp_change)
        
        # Combine specialized features
        specialized_features = torch.cat([pressure_features, temp_features], dim=1)
        
        # Reshape for 1D convolution (batch_size, input_dim, 1)
        x_conv = x.unsqueeze(-1)
        
        # Apply feature extraction
        features = self.feature_extractor(x_conv)
        
        # Squeeze the last dimension
        features = features.squeeze(-1)
        
        # Apply attention
        attention_weights = self.attention(features)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted_features = features * attention_weights
        
        # Apply cross-feature interaction
        interaction_features = self.interaction_layer(weighted_features, specialized_features)
        
        # Combine features
        combined_features = torch.cat([interaction_features, specialized_features], dim=1)
        
        # Apply fully connected layers
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        
        x = self.fc3(x)
        
        return x


class EnhancedStressDietExpert(nn.Module):
    """
    Enhanced stress and diet expert model with interaction modeling.
    
    This model captures interactions between stress and dietary factors,
    which are known to have complex relationships in migraine triggers.
    
    Attributes:
        stress_encoder: Neural network for encoding stress features
        diet_encoder: Neural network for encoding diet features
        interaction_layer: Layer for modeling interactions between stress and diet
        fc_layers: Fully connected layers for final prediction
        dropout: Dropout layer
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, dropout: float = 0.2):
        """
        Initialize enhanced stress and diet expert model.
        
        Args:
            input_dim: Input dimension (number of stress and diet features)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(EnhancedStressDietExpert, self).__init__()
        
        # Split input dimensions for stress and diet
        stress_dim = 1  # stress_level
        caffeine_dim = 1  # caffeine_intake
        alcohol_dim = 1  # alcohol_consumption
        diet_dim = 3  # meal_regularity, hydration_level, exercise_duration
        
        # Stress encoder
        self.stress_encoder = nn.Sequential(
            nn.Linear(stress_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Caffeine encoder
        self.caffeine_encoder = nn.Sequential(
            nn.Linear(caffeine_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Alcohol encoder
        self.alcohol_encoder = nn.Sequential(
            nn.Linear(alcohol_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Diet encoder
        self.diet_encoder = nn.Sequential(
            nn.Linear(diet_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction layers
        self.stress_caffeine_interaction = nn.Bilinear(hidden_dim // 2, hidden_dim // 4, hidden_dim // 4)
        self.stress_alcohol_interaction = nn.Bilinear(hidden_dim // 2, hidden_dim // 4, hidden_dim // 4)
        self.hydration_alcohol_interaction = nn.Bilinear(hidden_dim // 2, hidden_dim // 4, hidden_dim // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Split input into stress and diet components
        stress = x[:, 0:1]  # stress_level
        caffeine = x[:, 1:2]  # caffeine_intake
        alcohol = x[:, 2:3]  # alcohol_consumption
        diet = x[:, 3:6]  # meal_regularity, hydration_level, exercise_duration
        
        # Encode components
        stress_features = self.stress_encoder(stress)
        caffeine_features = self.caffeine_encoder(caffeine)
        alcohol_features = self.alcohol_encoder(alcohol)
        diet_features = self.diet_encoder(diet)
        
        # Calculate interactions
        stress_caffeine_interaction = self.stress_caffeine_interaction(stress_features, caffeine_features)
        stress_alcohol_interaction = self.stress_alcohol_interaction(stress_features, alcohol_features)
        hydration_alcohol_interaction = self.hydration_alcohol_interaction(diet_features, alcohol_features)
        
        # Combine features and interactions
        combined_features = torch.cat([
            stress_features, caffeine_features, alcohol_features, diet_features,
            stress_caffeine_interaction, stress_alcohol_interaction, hydration_alcohol_interaction
        ], dim=1)
        
        # Apply fully connected layers
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        
        x = self.fc3(x)
        
        return x


class EnhancedPhysiologicalExpert(nn.Module):
    """
    Enhanced physiological expert model with attention to vital signs.
    
    This model uses attention mechanisms to focus on important physiological features
    and their relationships to migraine triggers.
    
    Attributes:
        feature_extractor: Neural network for feature extraction
        attention: Attention mechanism for feature importance
        fc_layers: Fully connected layers for final prediction
        dropout: Dropout layer
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, dropout: float = 0.2):
        """
        Initialize enhanced physiological expert model.
        
        Args:
            input_dim: Input dimension (number of physiological features)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(EnhancedPhysiologicalExpert, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Blood pressure specific processing
        self.bp_processor = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Heart rate specific processing
        self.hr_processor = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Extract specific physiological features
        heart_rate = x[:, 0:1]  # heart_rate
        blood_pressure = x[:, 1:3]  # blood_pressure_systolic, blood_pressure_diastolic
        
        # Process specific features
        hr_features = self.hr_processor(heart_rate)
        bp_features = self.bp_processor(blood_pressure)
        
        # Combine specialized features
        specialized_features = torch.cat([hr_features, bp_features], dim=1)
        
        # Apply feature extraction
        features = self.feature_extractor(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = attention_weights.unsqueeze(2)
        
        # Apply attention to features
        weighted_features = features.unsqueeze(1) * attention_weights
        weighted_features = weighted_features.sum(dim=1)
        
        # Combine features
        combined_features = torch.cat([weighted_features, specialized_features], dim=1)
        
        # Apply fully connected layers
        x = self.fc1(combined_features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        
        x = self.fc3(x)
        
        return x
