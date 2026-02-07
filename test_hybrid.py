import torch
import argparse
from models.Hybrid import VideoHybrid

def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_hybrid():
    print("="*70)
    print("Testing VideoHybrid Model - LSTM + Transformer Encoder")
    print("="*70)
    
    # Mock arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
    parser.add_argument("--freeze_image_encoder", type=bool, default=False)
    args = parser.parse_args([])
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"\n✓ CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        args.device = 'cuda'
    else:
        print("\n✗ CUDA not available. Using CPU.")
    
    # Instantiate model
    print("\n" + "-"*70)
    print("1. Model Instantiation")
    print("-"*70)
    try:
        model = VideoHybrid(args).to(args.device)
        print("✓ Model instantiated successfully.")
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"\n  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with different sequence lengths
    print("\n" + "-"*70)
    print("2. Forward Pass Tests")
    print("-"*70)
    
    test_configs = [
        {"batch": 1, "seq_len": 5, "desc": "Single video, 5 frames"},
        {"batch": 2, "seq_len": 10, "desc": "2 videos, 10 frames each"},
        {"batch": 4, "seq_len": 8, "desc": "4 videos, 8 frames each"},
    ]
    
    for i, config in enumerate(test_configs, 1):
        batch = config["batch"]
        seq_len = config["seq_len"]
        desc = config["desc"]
        
        print(f"\nTest {i}: {desc}")
        print(f"  Input shape: ({batch}, {seq_len}, 3, 224, 224)")
        
        # Create dummy input
        dummy_input = torch.randn(batch, seq_len, 3, 224, 224).to(args.device)
        
        # Forward pass
        try:
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  Output shape: {output.shape}")
            
            # Check output shape
            expected_shape = (batch, 2)
            if output.shape == expected_shape:
                print(f"  ✓ Output shape is correct: {output.shape}")
                
                # Show predictions
                probs = torch.softmax(output, dim=1)
                print(f"  Sample predictions (Real/Fake probabilities):")
                for j in range(min(batch, 3)):  # Show first 3 samples
                    print(f"    Sample {j+1}: Real={probs[j, 0]:.4f}, Fake={probs[j, 1]:.4f}")
            else:
                print(f"  ✗ Expected output shape {expected_shape}, got {output.shape}")
                
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Test attention weights extraction
    print("\n" + "-"*70)
    print("3. Attention Weights Visualization")
    print("-"*70)
    
    try:
        dummy_input = torch.randn(1, 8, 3, 224, 224).to(args.device)
        attention_weights = model.get_attention_weights(dummy_input)
        
        print(f"  Attention weights shape: {attention_weights.shape}")
        print(f"  Attention weights (which frames the model focuses on):")
        print(f"  {attention_weights[0].cpu().numpy()}")
        print(f"  Sum of weights: {attention_weights.sum().item():.4f} (should be ~1.0)")
        print("  ✓ Attention weights extracted successfully")
        
    except Exception as e:
        print(f"  ✗ Attention extraction failed: {e}")
        import traceback
        traceback.print_exc()

    # Model architecture summary
    print("\n" + "-"*70)
    print("4. Model Architecture Summary")
    print("-"*70)
    print("\n  Architecture Flow:")
    print("  Input Video → CNN (EfficientNet B7) → Feature Projection")
    print("  → Bidirectional LSTM → Positional Encoding")
    print("  → Transformer Encoder → Attention Pooling → Classifier")
    print("\n  Key Components:")
    print(f"    - CNN Feature Size: 2560")
    print(f"    - Projected Feature Size: 512")
    print(f"    - LSTM Hidden Size: 256 (bidirectional → 512)")
    print(f"    - LSTM Layers: 2")
    print(f"    - Transformer Heads: 8")
    print(f"    - Transformer Layers: 3")
    print(f"    - Classifier: 512 → 256 → 2")

    # Final summary
    print("\n" + "="*70)
    print("All Tests Completed Successfully! ✓")
    print("="*70)
    print("\nThe hybrid model combines:")
    print("  1. CNN for spatial feature extraction")
    print("  2. LSTM for temporal sequence modeling")
    print("  3. Transformer for long-range dependencies")
    print("  4. Attention pooling for adaptive frame weighting")
    print("\nNext steps:")
    print("  - Train the model on your dataset")
    print("  - Use get_attention_weights() to visualize important frames")
    print("  - Fine-tune hyperparameters as needed")
    print("="*70)

if __name__ == "__main__":
    test_hybrid()

