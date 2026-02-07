import torch
import argparse
from models.LSTM import VideoLSTM

def test_lstm():
    print("Testing VideoLSTM...")
    
    # Mock arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
    parser.add_argument("--freeze_image_encoder", type=bool, default=False)
    # Add other args if ImageEncoder needs them, but these seem sufficient based on code
    args = parser.parse_args([])
    
    # Instantiate model
    try:
        model = VideoLSTM(args)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return

    # Create dummy input: (batch=1, seq_len=5, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 5, 3, 224, 224)
    print(f"Dummy input shape: {dummy_input.shape}")
    
    # Forward pass
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        if output.shape == (1, 2):
            print("Test PASSED: Output shape is correct.")
        else:
            print(f"Test FAILED: Expected output shape (1, 2), got {output.shape}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_lstm()
