import torch
import torch.nn as nn
from models.image import ImageEncoder

class VideoLSTM(nn.Module):
    def __init__(self, args):
        super(VideoLSTM, self).__init__()
        self.args = args
        
        # Use the existing ImageEncoder to handle weight loading
        self.image_encoder = ImageEncoder(args)
        
        # Feature size for EfficientNet B7 is 2560
        # If a different encoder is used, this might need to be adjustable
        self.feature_size = 2560 
        
        self.lstm_hidden_size = 512
        self.num_layers = 2
        self.num_classes = 2
        
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.5 if self.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(self.lstm_hidden_size, self.num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, seq_len, channels, height, width)
        b, s, c, h, w = x.shape
        
        # Flatten batch and sequence dimensions for CNN
        x = x.view(b * s, c, h, w)
        
        # Extract features using the encoder from ImageEncoder
        # We bypass ImageEncoder.forward to get raw features
        with torch.no_grad(): # Optional: freeze CNN if desired, otherwise remove
             # Using the underlying DeepFakeClassifier's encoder
             features = self.image_encoder.model.encoder.forward_features(x)
             features = self.image_encoder.model.avg_pool(features).flatten(1)
        
        # Reshape back to sequence: (batch, seq_len, feature_size)
        features = features.view(b, s, -1)
        
        # LSTM forward
        # out shape: (batch, seq_len, hidden_size)
        # hn shape: (num_layers, batch, hidden_size)
        self.lstm.flatten_parameters()
        out, (hn, cn) = self.lstm(features)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        
        return logits
