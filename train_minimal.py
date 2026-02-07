"""
Minimal Training Script for Hybrid Model - Quick Testing Version

This script uses:
- Only 10 videos (5 real, 5 fake) for fast training
- 5 epochs only
- Minimal frames (5 per video)
- Verbose output to track progress
- No data augmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import pandas as pd
from tqdm import tqdm
from models.Hybrid import VideoHybrid
from dataset_fakeavceleb import FakeAVCelebDataset
from datetime import datetime

print("="*70)
print("MINIMAL HYBRID MODEL TRAINING - QUICK TEST")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Configuration
CSV_FILE = 'datasets/fakeavceleb_100.csv'
BASE_PATH = '.'
NUM_VIDEOS = 10  # Only use 10 videos
NUM_EPOCHS = 5
BATCH_SIZE = 2
NUM_FRAMES = 5  # Reduced from 8
LEARNING_RATE = 0.0001
CHECKPOINT_DIR = 'checkpoints/hybrid_minimal'

print(f"\nConfiguration:")
print(f"  Videos: {NUM_VIDEOS}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Frames per video: {NUM_FRAMES}")
print("="*70)

# Create checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Create minimal CSV with balanced classes
print("\n1. Creating minimal dataset...")
df = pd.read_csv(CSV_FILE)
real_videos = df[df['label'] == 'real'].head(5)
fake_videos = df[df['label'] == 'fake'].head(5)
minimal_df = pd.concat([real_videos, fake_videos]).sample(frac=1, random_state=42)

# Save minimal CSV
minimal_csv = 'datasets/fakeavceleb_minimal.csv'
minimal_df.to_csv(minimal_csv, index=False)
print(f"   ✓ Created {minimal_csv} with {len(minimal_df)} videos")
print(f"   Real: {(minimal_df['label']=='real').sum()}, Fake: {(minimal_df['label']=='fake').sum()}")

# Split into train/val (8 train, 2 val)
train_df = minimal_df.head(8)
val_df = minimal_df.tail(2)

train_csv = 'datasets/fakeavceleb_minimal_train.csv'
val_csv = 'datasets/fakeavceleb_minimal_val.csv'
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
print(f"   ✓ Train: {len(train_df)} videos, Val: {len(val_df)} videos")

# Create datasets
print("\n2. Loading datasets...")
print("   Loading training dataset...")
train_dataset = FakeAVCelebDataset(
    csv_file=train_csv,
    base_path=BASE_PATH,
    num_frames=NUM_FRAMES,
    img_size=224,
    augment=False  # No augmentation for speed
)
print(f"   ✓ Train dataset: {len(train_dataset)} videos")

print("   Loading validation dataset...")
val_dataset = FakeAVCelebDataset(
    csv_file=val_csv,
    base_path=BASE_PATH,
    num_frames=NUM_FRAMES,
    img_size=224,
    augment=False
)
print(f"   ✓ Val dataset: {len(val_dataset)} videos")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("\n✗ Error: No valid videos found!")
    exit(1)

# Create data loaders
print("\n3. Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"   ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Initialize model
print("\n4. Initializing model...")
print("   (This may take 1-2 minutes to load EfficientNet...)")

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
parser.add_argument("--freeze_image_encoder", type=bool, default=True)
args = parser.parse_args([])

model = VideoHybrid(args).to(args.device)
print(f"   ✓ Model created on device: {args.device}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-5
)

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-"*70)
    
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    print("Training...")
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.to(args.device)
        labels = labels.to(args.device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        print(f"  Batch {batch_idx+1}/{len(train_loader)}: loss={loss.item():.4f}")
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    print("Validating...")
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(args.device)
            labels = labels.to(args.device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    # Print summary
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc
    }
    
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'latest.pth'))
    print(f"  ✓ Saved checkpoint: latest.pth")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        checkpoint['best_val_acc'] = best_val_acc
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, 'best.pth'))
        print(f"  ✓ Saved best model with val_acc: {val_acc:.2f}%")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
