import torch
import torch.nn as nn
import math
from models.image import ImageEncoder

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to inject sequence order information.
    Uses sinusoidal functions as described in "Attention is All You Need".
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VideoHybrid(nn.Module):
    """
    Hybrid Video Deepfake Detection Model combining:
    1. CNN (EfficientNet B7) - Spatial feature extraction from individual frames
    2. Bidirectional LSTM - Temporal sequence modeling with forward/backward context
    3. Transformer Encoder - Long-range dependency modeling with self-attention
    
    Architecture Flow:
    Input Video Frames → CNN Features → LSTM → Positional Encoding → Transformer → Classification
    """
    def __init__(self, args):
        super(VideoHybrid, self).__init__()
        self.args = args
        
        # 1. CNN Feature Extractor (EfficientNet B7)
        self.image_encoder = ImageEncoder(args)
        self.feature_size = 2560  # EfficientNet B7 output features
        
        # 2. Feature projection (optional: reduce dimensionality)
        self.feature_projection = nn.Linear(self.feature_size, 512)
        self.feature_norm = nn.LayerNorm(512)
        
        # 3. Bidirectional LSTM Layer
        self.lstm_hidden_size = 256  # Reduced since bidirectional doubles it
        self.lstm_num_layers = 2
        self.lstm = nn.LSTM(
            input_size=512,  # Projected feature size
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.lstm_num_layers > 1 else 0
        )
        
        # LSTM output size (bidirectional = 2 * hidden)
        self.lstm_out_size = self.lstm_hidden_size * 2
        
        # 4. Positional Encoding for Transformer
        self.positional_encoding = PositionalEncoding(
            d_model=self.lstm_out_size,
            max_len=100,  # Maximum sequence length
            dropout=0.1
        )
        
        # 5. Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.lstm_out_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',  # GELU often works better than ReLU
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=3  # Increased for better representation
        )
        
        # 6. Attention Pooling (learnable weighted average)
        self.attention_weights = nn.Linear(self.lstm_out_size, 1)
        
        # 7. Classifier Head
        self.num_classes = 2
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels, height, width)
               - batch: batch size
               - seq_len: number of frames in video
               - channels: 3 (RGB)
               - height, width: frame dimensions (e.g., 224x224)
        
        Returns:
            logits: Classification logits of shape (batch, num_classes)
        """
        b, s, c, h, w = x.shape
        
        # --- Stage 1: CNN Feature Extraction ---
        # Flatten batch and sequence dimensions for parallel CNN processing
        x = x.view(b * s, c, h, w)
        
        # Extract spatial features using pretrained EfficientNet
        with torch.no_grad():  # Freeze CNN encoder
            features = self.image_encoder.model.encoder.forward_features(x)
            features = self.image_encoder.model.avg_pool(features).flatten(1)
        
        # Project and normalize features
        features = self.feature_projection(features)
        features = self.feature_norm(features)
        
        # Reshape back to sequence: (batch, seq_len, feature_size)
        features = features.view(b, s, -1)
        
        # --- Stage 2: LSTM Temporal Modeling ---
        self.lstm.flatten_parameters()
        # lstm_out captures bidirectional temporal context
        # Shape: (batch, seq_len, lstm_out_size)
        lstm_out, _ = self.lstm(features)
        
        # --- Stage 3: Positional Encoding ---
        # Add positional information for transformer
        lstm_out = self.positional_encoding(lstm_out)
        
        # --- Stage 4: Transformer Self-Attention ---
        # Capture long-range dependencies across the sequence
        # Shape: (batch, seq_len, lstm_out_size)
        transformer_out = self.transformer_encoder(lstm_out)
        
        # --- Stage 5: Attention Pooling ---
        # Compute attention weights for each time step
        attention_scores = self.attention_weights(transformer_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum of transformer outputs
        context_vector = torch.sum(transformer_out * attention_weights, dim=1)  # (batch, lstm_out_size)
        
        # --- Stage 6: Classification ---
        logits = self.classifier(context_vector)
        
        return logits
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization.
        Useful for understanding which frames the model focuses on.
        """
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        
        with torch.no_grad():
            features = self.image_encoder.model.encoder.forward_features(x)
            features = self.image_encoder.model.avg_pool(features).flatten(1)
            features = self.feature_projection(features)
            features = self.feature_norm(features)
            features = features.view(b, s, -1)
            
            self.lstm.flatten_parameters()
            lstm_out, _ = self.lstm(features)
            lstm_out = self.positional_encoding(lstm_out)
            transformer_out = self.transformer_encoder(lstm_out)
            
            attention_scores = self.attention_weights(transformer_out)
            attention_weights = torch.softmax(attention_scores, dim=1)
        
        return attention_weights.squeeze(-1)  # (batch, seq_len)
