"""
Inference script for the Hybrid LSTM-Transformer Video Deepfake Detector

Usage:
    python inference_hybrid.py --video path/to/video.mp4 --checkpoint path/to/model.pth
"""

import torch
import torch.nn as nn
import cv2
import argparse
import numpy as np
from torchvision import transforms
from models.Hybrid import VideoHybrid
import os


class VideoProcessor:
    """Process video files for model inference"""
    
    def __init__(self, num_frames=10, img_size=224):
        self.num_frames = num_frames
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            print(f"Warning: Video has only {total_frames} frames, need {self.num_frames}")
            frame_indices = list(range(total_frames))
        else:
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transforms
                frame = self.transform(frame)
                frames.append(frame)
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1])  # Repeat last frame
        
        # Stack frames: (num_frames, C, H, W)
        frames_tensor = torch.stack(frames)
        
        # Add batch dimension: (1, num_frames, C, H, W)
        return frames_tensor.unsqueeze(0)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
    parser.add_argument("--freeze_image_encoder", type=bool, default=False)
    args = parser.parse_args([])
    
    # Initialize model
    model = VideoHybrid(args).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_video(model, video_tensor, device):
    """Make prediction on video"""
    video_tensor = video_tensor.to(device)
    
    with torch.no_grad():
        logits = model(video_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Get attention weights for visualization
        attention_weights = model.get_attention_weights(video_tensor)
    
    return probs, attention_weights


def main():
    parser = argparse.ArgumentParser(description='Hybrid Model Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to extract')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--show_attention', action='store_true', help='Show attention weights')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hybrid LSTM-Transformer Video Deepfake Detector")
    print("="*70)
    
    # ===== VALIDATION 1: Check if video file exists =====
    if not os.path.exists(args.video):
        print(f"\n✗ Error: Video file not found: {args.video}")
        print("Please check the file path and try again.")
        return
    
    # ===== VALIDATION 2: Check video file format =====
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']
    file_ext = os.path.splitext(args.video)[1].lower()
    
    if file_ext not in valid_extensions:
        print(f"\n✗ Error: Invalid video format: {file_ext}")
        print(f"\nSupported formats:")
        for ext in valid_extensions:
            print(f"  - {ext}")
        return
    
    # ===== VALIDATION 3: Check if checkpoint exists =====
    if not os.path.exists(args.checkpoint):
        print(f"\n✗ Error: Checkpoint file not found: {args.checkpoint}")
        print("Please check the checkpoint path and try again.")
        return
    
    # ===== VALIDATION 4: Check file size and warn if large =====
    file_size_mb = os.path.getsize(args.video) / (1024 * 1024)
    if file_size_mb > 50:
        print(f"\n⚠ Warning: Large video file detected ({file_size_mb:.1f} MB)")
        print("Processing may take longer than usual...")
    
    print(f"\n✓ Video: {args.video} ({file_size_mb:.1f} MB)")
    print(f"✓ Checkpoint: {args.checkpoint}")
    print(f"✓ Device: {args.device}")
    print(f"✓ Frames to extract: {args.num_frames}")
    
    # Load model
    print("\nLoading model...")
    try:
        model = load_model(args.checkpoint, args.device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process video
    print("\nProcessing video...")
    try:
        processor = VideoProcessor(num_frames=args.num_frames)
        video_tensor = processor.extract_frames(args.video)
        print(f"✓ Extracted {args.num_frames} frames")
        print(f"  Video tensor shape: {video_tensor.shape}")
    except Exception as e:
        print(f"✗ Failed to process video: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Make prediction
    print("\nMaking prediction...")
    try:
        probs, attention_weights = predict_video(model, video_tensor, args.device)
        
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nReal probability: {real_prob:.4f} ({real_prob*100:.2f}%)")
        print(f"Fake probability: {fake_prob:.4f} ({fake_prob*100:.2f}%)")
        
        if fake_prob > real_prob:
            print(f"\n⚠️  PREDICTION: FAKE (confidence: {fake_prob*100:.2f}%)")
        else:
            print(f"\n✓ PREDICTION: REAL (confidence: {real_prob*100:.2f}%)")
        
        if args.show_attention:
            print("\n" + "-"*70)
            print("Frame Attention Weights (which frames the model focuses on):")
            print("-"*70)
            weights = attention_weights[0].cpu().numpy()
            for i, weight in enumerate(weights):
                bar_length = int(weight * 50)  # Scale to 50 chars
                bar = '█' * bar_length
                print(f"Frame {i+1:2d}: {bar} {weight:.4f}")
        
        print("="*70)
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
