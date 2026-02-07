"""
Step-by-step test to find where code hangs
"""
import sys

print("STEP 1: Script started", flush=True)
sys.stdout.flush()

print("STEP 2: Importing torch...", flush=True)
sys.stdout.flush()
import torch
print("  ✓ Torch imported", flush=True)
sys.stdout.flush()

print("STEP 3: Importing models...", flush=True)
sys.stdout.flush()
from models.Hybrid import VideoHybrid
print("  ✓ Hybrid model imported", flush=True)
sys.stdout.flush()

print("STEP 4: Creating args...", flush=True)
sys.stdout.flush()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
parser.add_argument("--freeze_image_encoder", type=bool, default=True)
args = parser.parse_args([])
print(f"  ✓ Args created, device={args.device}", flush=True)
sys.stdout.flush()

print("STEP 5: Creating model (THIS MAY TAKE TIME)...", flush=True)
sys.stdout.flush()
model = VideoHybrid(args)
print("  ✓ Model created!", flush=True)
sys.stdout.flush()

print("STEP 6: Testing forward pass...", flush=True)
sys.stdout.flush()
dummy = torch.randn(1, 5, 3, 224, 224)
with torch.no_grad():
    out = model(dummy)
print(f"  ✓ Forward pass successful! Output shape: {out.shape}", flush=True)
sys.stdout.flush()

print("\n" + "="*70, flush=True)
print("ALL STEPS PASSED! Model works!", flush=True)
print("="*70, flush=True)
sys.stdout.flush()
