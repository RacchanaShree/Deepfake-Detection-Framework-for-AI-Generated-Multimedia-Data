import torch
import os

checkpoint_path = r"c:\Users\THRIVENI GK\OneDrive\Desktop\AI-Generated-Video-Detector-main\checkpoints\hybrid_fakeavceleb\best.pth"

if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best Val Acc: {checkpoint.get('best_val_acc', 'Unknown')}")
        print(f"Train Acc: {checkpoint.get('train_acc', 'Unknown')}")
        print("Training seems to have completed successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Checkpoint not found.")
