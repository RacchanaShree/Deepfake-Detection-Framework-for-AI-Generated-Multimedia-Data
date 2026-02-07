"""
Complete Training Script for Hybrid LSTM-Transformer Model on FakeAVCeleb Dataset

Usage:
    python train_hybrid_fakeavceleb.py --csv datasets/fakeavceleb_1k.csv --epochs 50 --batch_size 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
from models.Hybrid import VideoHybrid
from dataset_fakeavceleb import FakeAVCelebDataset, create_train_val_split
import time
from datetime import datetime


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]  ')
        for videos, labels in pbar:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss / (len(all_preds) // labels.size(0)):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    real_mask = all_labels == 0
    fake_mask = all_labels == 1
    
    real_acc = 100. * (all_preds[real_mask] == all_labels[real_mask]).sum() / real_mask.sum() if real_mask.sum() > 0 else 0
    fake_acc = 100. * (all_preds[fake_mask] == all_labels[fake_mask]).sum() / fake_mask.sum() if fake_mask.sum() > 0 else 0
    
    return val_loss, val_acc, real_acc, fake_acc


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid LSTM-Transformer Model on FakeAVCeleb')
    
    # Data arguments
    parser.add_argument('--csv', type=str, default='datasets/fakeavceleb_100.csv',
                        help='Path to CSV file with video paths and labels')
    parser.add_argument('--base_path', type=str, default='.',
                        help='Base path to FakeAVCeleb_v1.2 folder')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to extract per video')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training')
    
    # Model arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained_image_encoder', type=bool, default=False,
                        help='Use pretrained EfficientNet weights')
    parser.add_argument('--freeze_image_encoder', type=bool, default=True,
                        help='Freeze CNN encoder during training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/hybrid_fakeavceleb',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Print configuration
    print("="*80)
    print("HYBRID LSTM-TRANSFORMER TRAINING ON FAKEAVCELEB")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    print(f"  CSV file: {args.csv}")
    print(f"  Device: {args.device}")
    print(f"  Frames per video: {args.num_frames}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Pretrained CNN: {args.pretrained_image_encoder}")
    print(f"  Freeze CNN: {args.freeze_image_encoder}")
    print("="*80)
    
    # Create train/val split if needed
    if not os.path.exists(args.csv.replace('.csv', '_train.csv')):
        print("\nCreating train/val split...")
        train_csv, val_csv = create_train_val_split(
            args.csv, 
            train_ratio=args.train_ratio,
            random_seed=42
        )
    else:
        train_csv = args.csv.replace('.csv', '_train.csv')
        val_csv = args.csv.replace('.csv', '_val.csv')
        print(f"\nUsing existing split:")
        print(f"  Train: {train_csv}")
        print(f"  Val: {val_csv}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = FakeAVCelebDataset(
        csv_file=train_csv,
        base_path=args.base_path,
        num_frames=args.num_frames,
        img_size=224,
        augment=True  # Data augmentation for training
    )
    
    val_dataset = FakeAVCelebDataset(
        csv_file=val_csv,
        base_path=args.base_path,
        num_frames=args.num_frames,
        img_size=224,
        augment=False  # No augmentation for validation
    )
    
    print(f"  Train dataset: {len(train_dataset)} videos")
    print(f"  Val dataset: {len(val_dataset)} videos")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\n✗ Error: No valid videos found in dataset!")
        print("Please check that video files exist in the correct location.")
        return
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = VideoHybrid(args).to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nLoading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*80}")
        
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch+1
        )
        
        # Validate
        val_loss, val_acc, real_acc, fake_acc = validate(
            model, val_loader, criterion, args.device, epoch+1
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Real Acc: {real_acc:.2f}% | Fake Acc: {fake_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'best_val_acc': best_val_acc,
            'args': args
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
        
        # Save periodically
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth'))
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
            print(f"  ✓ Saved best model with val_acc: {val_acc:.2f}%")
    
    # Training complete
    total_time = time.time() - training_start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
