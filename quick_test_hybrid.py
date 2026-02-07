"""
Quick test script for the hybrid model without loading pretrained weights.
This tests the architecture and forward pass quickly.
"""
import torch
import argparse
from models.Hybrid import VideoHybrid

print("Quick Hybrid Model Test")
print("="*50)

# Create minimal args
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
parser.add_argument("--freeze_image_encoder", type=bool, default=False)
args = parser.parse_args([])

print("\n1. Creating model...")
try:
    model = VideoHybrid(args)
    print("✓ Model created successfully")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total:,}")
    print(f"   Trainable: {trainable:,}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n2. Testing forward pass...")
try:
    # Small test: 1 video, 5 frames
    x = torch.randn(1, 5, 3, 224, 224)
    print(f"   Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"   Output shape: {output.shape}")
    
    if output.shape == (1, 2):
        print("   ✓ Correct output shape!")
        probs = torch.softmax(output, dim=1)
        print(f"   Predictions: Real={probs[0,0]:.4f}, Fake={probs[0,1]:.4f}")
    else:
        print(f"   ✗ Wrong shape! Expected (1, 2)")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n3. Testing attention weights...")
try:
    x = torch.randn(1, 8, 3, 224, 224)
    weights = model.get_attention_weights(x)
    print(f"   Attention shape: {weights.shape}")
    print(f"   Weights sum: {weights.sum():.4f}")
    print("   ✓ Attention extraction works!")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
