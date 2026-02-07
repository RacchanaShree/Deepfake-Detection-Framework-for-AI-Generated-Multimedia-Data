import torch
import cv2
import numpy as np
import argparse
from models.LSTM import VideoLSTM

def preprocess_video_sequence(video_path, seq_len=10):
    """
    Extracts a sequence of frames from a video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Could not read frames from {video_path}")
        return None

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, seq_len).astype(int)
    
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            # Preprocess: Resize to 224x224 (EfficientNet standard) and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)) # Note: ImageEncoder might expect 256 then crop, but 224 is standard
            frame = frame / 255.0
            # Transpose to (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
            
    cap.release()
    
    # Pad if not enough frames
    while len(frames) < seq_len:
        frames.append(frames[-1])
        
    # Stack into tensor: (seq_len, C, H, W)
    frames = np.array(frames)
    frames = torch.FloatTensor(frames)
    
    # Add batch dimension: (1, seq_len, C, H, W)
    return frames.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--device", type=str, default='cpu')
    # Args required by VideoLSTM/ImageEncoder
    parser.add_argument("--pretrained_image_encoder", type=bool, default=False)
    parser.add_argument("--freeze_image_encoder", type=bool, default=False)
    
    args = parser.parse_args()
    
    print(f"Loading model on {args.device}...")
    model = VideoLSTM(args).to(args.device)
    model.eval()
    
    print(f"Processing video: {args.video_path}")
    input_tensor = preprocess_video_sequence(args.video_path)
    
    if input_tensor is None:
        return

    input_tensor = input_tensor.to(args.device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        
    # Assuming class 0 is Real, 1 is Fake (or vice versa, need to check dataset)
    # Usually 0: Real, 1: Fake in many datasets, but let's just print raw probs
    print(f"Probabilities: {probs.cpu().numpy()}")
    
    fake_prob = probs[0][1].item()
    if fake_prob > 0.5:
        print(f"Prediction: FAKE ({fake_prob*100:.2f}%)")
    else:
        print(f"Prediction: REAL ({(1-fake_prob)*100:.2f}%)")

if __name__ == "__main__":
    main()
