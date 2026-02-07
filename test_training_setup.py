"""
Quick test to diagnose training issues
"""
import sys
print("Starting diagnostic test...")

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"   ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"   ✓ OpenCV")
except Exception as e:
    print(f"   ✗ OpenCV import failed: {e}")
    sys.exit(1)

try:
    from models.Hybrid import VideoHybrid
    print(f"   ✓ Hybrid model import")
except Exception as e:
    print(f"   ✗ Hybrid model import failed: {e}")
    sys.exit(1)

try:
    from dataset_fakeavceleb import FakeAVCelebDataset
    print(f"   ✓ Dataset import")
except Exception as e:
    print(f"   ✗ Dataset import failed: {e}")
    sys.exit(1)

# Test 2: Check CSV file
print("\n2. Checking CSV file...")
import os
csv_file = 'datasets/fakeavceleb_100.csv'
if os.path.exists(csv_file):
    print(f"   ✓ CSV file exists: {csv_file}")
    import pandas as pd
    df = pd.read_csv(csv_file)
    print(f"   ✓ CSV has {len(df)} entries")
else:
    print(f"   ✗ CSV file not found: {csv_file}")
    sys.exit(1)

# Test 3: Load dataset (small sample)
print("\n3. Testing dataset loading (first 2 videos)...")
try:
    dataset = FakeAVCelebDataset(
        csv_file=csv_file,
        base_path='.',
        num_frames=5,
        img_size=224,
        augment=False,
        max_videos=2
    )
    print(f"   ✓ Dataset created with {len(dataset)} videos")
    
    if len(dataset) > 0:
        print("   Loading first video...")
        video, label = dataset[0]
        print(f"   ✓ Video shape: {video.shape}, Label: {label}")
    else:
        print("   ✗ No valid videos in dataset!")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create model
print("\n4. Testing model creation...")
try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
    parser.add_argument("--freeze_image_encoder", type=bool, default=True)
    args = parser.parse_args([])
    
    print("   Creating model (this may take a minute)...")
    model = VideoHybrid(args)
    print(f"   ✓ Model created successfully")
    
    # Test forward pass
    print("   Testing forward pass...")
    dummy_input = torch.randn(1, 5, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"   ✗ Model creation/forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nThe training script should work. If it's still hanging,")
print("the issue might be with video file loading or system resources.")
