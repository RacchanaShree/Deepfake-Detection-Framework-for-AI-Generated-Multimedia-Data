"""
FakeAVCeleb Dataset Loader for Hybrid LSTM-Transformer Model

This dataset loader handles the FakeAVCeleb dataset which contains:
- Real videos (RealVideo-RealAudio)
- Fake videos (FakeVideo-FakeAudio)

The dataset is organized with CSV files containing video paths and labels.
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FakeAVCelebDataset(Dataset):
    """
    PyTorch Dataset for FakeAVCeleb video dataset.
    
    Args:
        csv_file (str): Path to CSV file with video paths and labels
        base_path (str): Base directory containing FakeAVCeleb_v1.2 folder
        num_frames (int): Number of frames to extract from each video
        img_size (int): Size to resize frames to (default: 224)
        augment (bool): Whether to apply data augmentation
        max_videos (int): Maximum number of videos to load (for debugging)
    """
    
    def __init__(self, csv_file, base_path, num_frames=10, img_size=224, 
                 augment=False, max_videos=None):
        self.base_path = base_path
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment
        
        # Read CSV file
        self.data = pd.read_csv(csv_file)
        
        # Limit dataset size if specified (for debugging)
        if max_videos is not None:
            self.data = self.data.head(max_videos)
        
        # Convert labels to integers: real=0, fake=1
        self.data['label'] = self.data['label'].map({'real': 0, 'fake': 1})
        
        # Define transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Verify some videos exist
        self._verify_videos()
    
    def _verify_videos(self):
        """Verify that video files exist and remove missing ones"""
        valid_indices = []
        missing_count = 0
        
        for idx in range(len(self.data)):
            video_path = self._get_video_path(idx)
            if os.path.exists(video_path):
                valid_indices.append(idx)
            else:
                missing_count += 1
                if missing_count <= 5:  # Only print first 5 missing files
                    print(f"Warning: Video not found: {video_path}")
        
        if missing_count > 0:
            print(f"\nTotal missing videos: {missing_count}/{len(self.data)}")
            print(f"Valid videos: {len(valid_indices)}/{len(self.data)}")
            self.data = self.data.iloc[valid_indices].reset_index(drop=True)
    
    def _get_video_path(self, idx):
        """Construct full video path"""
        relative_path = self.data.iloc[idx]['video_path']
        # The CSV has paths like "FakeAVCeleb/RealVideo-RealAudio/..."
        # We need to adjust to "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio/..."
        
        # Remove "FakeAVCeleb/" prefix and add correct base path
        if relative_path.startswith('FakeAVCeleb/'):
            relative_path = relative_path[len('FakeAVCeleb/'):]
        
        full_path = os.path.join(self.base_path, 'FakeAVCeleb_v1.2', 
                                 'FakeAVCeleb_v1.2', relative_path)
        return full_path
    
    def _extract_frames(self, video_path):
        """
        Extract frames from video using uniform sampling.
        
        Returns:
            frames: List of numpy arrays (H, W, C) in RGB format
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Video has 0 frames: {video_path}")
        
        # Sample frames uniformly
        if total_frames < self.num_frames:
            # If video has fewer frames than needed, sample all and repeat last
            frame_indices = list(range(total_frames))
        else:
            # Uniform sampling
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If frame read fails, use last valid frame or black frame
                if len(frames) > 0:
                    frames.append(frames[-1])
                else:
                    # Create black frame
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        cap.release()
        
        # Pad if necessary (repeat last frame)
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.num_frames]  # Ensure exactly num_frames
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a video sample.
        
        Returns:
            video_tensor: Tensor of shape (num_frames, 3, H, W)
            label: Integer label (0=real, 1=fake)
        """
        video_path = self._get_video_path(idx)
        label = self.data.iloc[idx]['label']
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            
            # Apply transforms to each frame
            transformed_frames = []
            for frame in frames:
                transformed_frame = self.transform(frame)
                transformed_frames.append(transformed_frame)
            
            # Stack frames: (num_frames, C, H, W)
            video_tensor = torch.stack(transformed_frames)
            
            return video_tensor, label
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return a dummy tensor and label
            dummy_video = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
            return dummy_video, label


def create_train_val_split(csv_file, train_ratio=0.8, random_seed=42):
    """
    Split a CSV file into train and validation sets.
    
    Args:
        csv_file (str): Path to CSV file
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
    
    Returns:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    split_idx = int(len(df) * train_ratio)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Save
    base_name = csv_file.replace('.csv', '')
    train_csv = f"{base_name}_train.csv"
    val_csv = f"{base_name}_val.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Created train/val split:")
    print(f"  Train: {len(train_df)} samples -> {train_csv}")
    print(f"  Val: {len(val_df)} samples -> {val_csv}")
    
    return train_csv, val_csv


if __name__ == '__main__':
    # Test the dataset loader
    print("Testing FakeAVCeleb Dataset Loader...")
    print("="*70)
    
    # Test with small dataset
    csv_file = 'datasets/fakeavceleb_100.csv'
    base_path = '.'
    
    print(f"\nLoading dataset from: {csv_file}")
    dataset = FakeAVCelebDataset(
        csv_file=csv_file,
        base_path=base_path,
        num_frames=10,
        img_size=224,
        augment=False,
        max_videos=5  # Only load 5 videos for testing
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\nLoading first sample...")
        video, label = dataset[0]
        print(f"Video shape: {video.shape}")
        print(f"Label: {label} ({'Fake' if label == 1 else 'Real'})")
        print(f"Video dtype: {video.dtype}")
        print(f"Video range: [{video.min():.3f}, {video.max():.3f}]")
        
        print("\n✓ Dataset loader test passed!")
    else:
        print("\n✗ No valid videos found in dataset!")
    
    print("="*70)
