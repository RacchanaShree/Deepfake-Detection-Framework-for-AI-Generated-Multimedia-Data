"""
Monitor training progress by checking checkpoint files
"""
import os
import torch
import time
from datetime import datetime

checkpoint_dir = 'checkpoints/hybrid_fakeavceleb'

print("="*70)
print("TRAINING PROGRESS MONITOR")
print("="*70)
print(f"Monitoring: {checkpoint_dir}")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nWaiting for training to start...")
print("(This may take 2-5 minutes while the model loads)")
print("="*70)

last_epoch = -1
start_time = time.time()

while True:
    try:
        # Check if checkpoint directory exists
        if not os.path.exists(checkpoint_dir):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting for checkpoint directory...")
            time.sleep(30)
            continue
        
        # Check for latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        
        if os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location='cpu')
            epoch = checkpoint['epoch'] + 1
            
            if epoch > last_epoch:
                last_epoch = epoch
                elapsed = time.time() - start_time
                
                print(f"\n{'='*70}")
                print(f"Epoch {epoch}/30 COMPLETED")
                print(f"{'='*70}")
                print(f"Train Loss: {checkpoint['train_loss']:.4f} | Train Acc: {checkpoint['train_acc']:.2f}%")
                print(f"Val Loss: {checkpoint['val_loss']:.4f} | Val Acc: {checkpoint['val_acc']:.2f}%")
                print(f"Real Acc: {checkpoint['real_acc']:.2f}% | Fake Acc: {checkpoint['fake_acc']:.2f}%")
                print(f"Elapsed time: {elapsed/3600:.2f} hours")
                
                if os.path.exists(best_path):
                    best_checkpoint = torch.load(best_path, map_location='cpu')
                    print(f"Best Val Acc so far: {best_checkpoint['best_val_acc']:.2f}%")
                
                # Check if training is complete
                if epoch >= 30:
                    print(f"\n{'='*70}")
                    print("TRAINING COMPLETE! ✓")
                    print(f"{'='*70}")
                    print(f"Total time: {elapsed/3600:.2f} hours")
                    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
                    print(f"Checkpoints saved in: {checkpoint_dir}")
                    break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for first checkpoint...")
        
        # Wait before next check
        time.sleep(60)  # Check every minute
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        break
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(60)

print("\nMonitoring ended.")
