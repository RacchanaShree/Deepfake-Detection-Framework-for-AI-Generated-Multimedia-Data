"""
Enhanced Gradio UI for Hybrid LSTM-Transformer Video Deepfake Detector
"""
import gradio as gr
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import os
from torchvision import transforms
from models.Hybrid import VideoHybrid
import matplotlib.pyplot as plt
import io
from PIL import Image

# Configuration
CHECKPOINT_PATH = "checkpoints/hybrid_fakeavceleb/latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 10
IMG_SIZE = 224

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
        """Extract frames from video (optimized)"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Early validation
        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no frames")
        
        if total_frames < self.num_frames:
            frame_indices = list(range(total_frames))
        else:
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
    
    model = VideoHybrid(args).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def create_attention_visualization(attention_weights):
    """Create a visual representation of attention weights (optimized)"""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    weights = attention_weights[0].cpu().numpy()
    frames = np.arange(1, len(weights) + 1)
    
    colors = plt.cm.viridis(weights / weights.max())
    bars = ax.bar(frames, weights, color=colors, edgecolor='none', linewidth=0)
    
    ax.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax.set_title('Frame-wise Attention Distribution', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Convert to image with reduced DPI for faster rendering
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=75, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

# Global model variable
model = None

def initialize_model():
    """Initialize the model on startup"""
    global model
    if model is None:
        try:
            model = load_model(CHECKPOINT_PATH, DEVICE)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

def predict_video(video_path, show_attention=True):
    """Make prediction on uploaded video with comprehensive validation"""
    global model
    
    try:
        # ===== VALIDATION STEP 1: Check if file exists =====
        if video_path is None:
            error_html = """
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--warning-bg); border: 2px solid var(--warning-text);">
                <h2 style="color: var(--warning-text); margin: 10px 0;">⚠️ No File Uploaded</h2>
                <p style="color: var(--warning-text);">Please upload a video file to analyze.</p>
            </div>
            """
            return error_html, None, "No file uploaded"
        
        if not os.path.exists(video_path):
            error_html = """
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--error-bg); border: 2px solid var(--error-text);">
                <h2 style="color: var(--error-text); margin: 10px 0;">❌ File Not Found</h2>
                <p style="color: var(--error-text);">The uploaded file could not be found. Please try again.</p>
            </div>
            """
            return error_html, None, "File not found"
        
        # ===== VALIDATION STEP 2: Check file format =====
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.mpg', '.mpeg']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in valid_extensions:
            error_html = f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--error-bg); border: 2px solid var(--error-text);">
                <h2 style="color: var(--error-text); margin: 10px 0;">❌ Invalid File Format</h2>
                <p style="color: var(--error-text); font-size: 16px; margin: 15px 0;">
                    <strong>Please upload a valid video file</strong>
                </p>
                <p style="color: var(--error-text); margin: 10px 0;">
                    Detected format: <code style="background: rgba(255,255,255,0.5); padding: 2px 6px; border-radius: 3px;">{file_ext if file_ext else 'unknown'}</code>
                </p>
                <div style="margin-top: 15px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 5px; text-align: left;">
                    <strong style="color: var(--error-text);">Supported formats:</strong>
                    <ul style="color: var(--error-text); margin: 10px 0;">
                        <li>MP4 (.mp4)</li>
                        <li>AVI (.avi)</li>
                        <li>MOV (.mov)</li>
                        <li>MKV (.mkv)</li>
                        <li>WebM (.webm)</li>
                        <li>FLV (.flv)</li>
                        <li>WMV (.wmv)</li>
                    </ul>
                </div>
            </div>
            """
            return error_html, None, f"Invalid format: {file_ext}"
        
        # ===== VALIDATION STEP 3: Check file size and warn if large =====
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        status_message = f"Processing video ({file_size_mb:.1f} MB)..."
        
        if file_size_mb > 50:
            status_message = f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer..."
        
        # ===== VALIDATION STEP 4: Initialize model if needed =====
        if model is None:
            if not initialize_model():
                error_html = """
                <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--error-bg); border: 2px solid var(--error-text);">
                    <h2 style="color: var(--error-text); margin: 10px 0;">❌ Model Loading Error</h2>
                    <p style="color: var(--error-text);">Failed to load the detection model. Please check checkpoint path.</p>
                    <p style="color: var(--error-text); font-size: 14px; margin-top: 10px;">
                        Expected checkpoint: <code style="background: rgba(255,255,255,0.5); padding: 2px 6px;">{CHECKPOINT_PATH}</code>
                    </p>
                </div>
                """
                return error_html, None, "Model loading failed"
        
        # ===== PROCESSING: Extract frames and make prediction =====
        processor = VideoProcessor(num_frames=NUM_FRAMES)
        video_tensor = processor.extract_frames(video_path)
        video_tensor = video_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            logits = model(video_tensor)
            probs = torch.softmax(logits, dim=1)
            attention_weights = model.get_attention_weights(video_tensor)
        
        # Clear GPU cache to prevent memory buildup
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        
        # Create result text
        if fake_prob > real_prob:
            verdict = "🚨 FAKE"
            confidence = fake_prob * 100
            color = "red"
        else:
            verdict = "✅ REAL"
            confidence = real_prob * 100
            color = "green"
        
        # Add file size info to result
        file_info = f"<li style=\"color: var(--info-text);\"><strong style=\"color: var(--info-text);\">File Size:</strong> {file_size_mb:.1f} MB</li>"
        if file_size_mb > 50:
            file_info += "<li style=\"color: var(--info-text);\"><strong style=\"color: var(--info-text);\">Note:</strong> Large file processed successfully</li>"
        
        result_html = f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--header-bg);">
            <h1 style="color: var(--header-text); font-size: 48px; margin: 10px 0;">{verdict}</h1>
            <h2 style="color: var(--header-text); font-size: 32px; margin: 10px 0;">Confidence: {confidence:.2f}%</h2>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: var(--secondary-bg); border-radius: 8px;">
            <h3 style="margin-top: 0; color: var(--secondary-text);">📊 Detailed Probabilities</h3>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: var(--secondary-text);"><strong>Real:</strong></span>
                    <span style="color: var(--secondary-text); font-weight: 600;">{real_prob*100:.2f}%</span>
                </div>
                <div style="background: var(--border-color); border-radius: 10px; height: 25px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #28a745, #20c997); height: 100%; width: {real_prob*100}%; transition: width 0.3s;"></div>
                </div>
            </div>
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: var(--secondary-text);"><strong>Fake:</strong></span>
                    <span style="color: var(--secondary-text); font-weight: 600;">{fake_prob*100:.2f}%</span>
                </div>
                <div style="background: var(--border-color); border-radius: 10px; height: 25px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #dc3545, #fd7e14); height: 100%; width: {fake_prob*100}%; transition: width 0.3s;"></div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: var(--info-bg); border-left: 4px solid var(--info-border); border-radius: 4px;">
            <h4 style="margin-top: 0; color: var(--info-text);">ℹ️ Analysis Details</h4>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li style="color: var(--info-text);"><strong style="color: var(--info-text);">Model:</strong> Hybrid LSTM-Transformer</li>
                <li style="color: var(--info-text);"><strong style="color: var(--info-text);">Frames Analyzed:</strong> {NUM_FRAMES}</li>
                <li style="color: var(--info-text);"><strong style="color: var(--info-text);">Device:</strong> {DEVICE.upper()}</li>
                <li style="color: var(--info-text);"><strong style="color: var(--info-text);">Checkpoint:</strong> {Path(CHECKPOINT_PATH).name}</li>
                {file_info}
            </ul>
        </div>
        """
        
        # Create attention visualization if requested
        attention_img = None
        if show_attention:
            attention_img = create_attention_visualization(attention_weights)
        
        return result_html, attention_img, f"✓ Analysis complete! Processed {NUM_FRAMES} frames from {file_size_mb:.1f} MB video."
        
    except cv2.error as e:
        # Handle OpenCV-specific errors (corrupted video, codec issues, etc.)
        error_html = f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--error-bg); border: 2px solid var(--error-text);">
            <h2 style="color: var(--error-text); margin: 10px 0;">❌ Video Processing Error</h2>
            <p style="color: var(--error-text); font-size: 16px; margin: 15px 0;">
                Unable to process the video file. The file may be corrupted or use an unsupported codec.
            </p>
            <details style="margin-top: 15px; text-align: left; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                <summary style="cursor: pointer; color: var(--error-text); font-weight: bold;">Technical Details</summary>
                <pre style="color: var(--error-text); font-size: 12px; margin: 10px 0; overflow-x: auto;">{str(e)}</pre>
            </details>
        </div>
        """
        return error_html, None, "Video processing error"
        
    except Exception as e:
        # Handle any other unexpected errors
        import traceback
        error_details = traceback.format_exc()
        
        error_html = f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background: var(--error-bg); border: 2px solid var(--error-text);">
            <h2 style="color: var(--error-text); margin: 10px 0;">❌ Unexpected Error</h2>
            <p style="color: var(--error-text); font-size: 16px; margin: 15px 0;">
                An error occurred during analysis. Please try again or contact support.
            </p>
            <details style="margin-top: 15px; text-align: left; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">
                <summary style="cursor: pointer; color: var(--error-text); font-weight: bold;">Technical Details</summary>
                <pre style="color: var(--error-text); font-size: 12px; margin: 10px 0; overflow-x: auto; white-space: pre-wrap;">{str(e)}</pre>
            </details>
        </div>
        """
        return error_html, None, "Analysis failed"

# CSS for theming (minified for performance)
CUSTOM_CSS = """
:root{--header-bg:linear-gradient(135deg,#667eea 0%,#764ba2 100%);--header-text:white;--card-bg:#fff;--card-text:#212529;--secondary-bg:#f8f9fa;--secondary-text:#212529;--border-color:#dee2e6;--success-bg:#d4edda;--success-text:#155724;--error-bg:#f8d7da;--error-text:#721c24;--warning-bg:#fff3cd;--warning-text:#856404;--info-bg:#e7f3ff;--info-text:#0c5460;--info-border:#2196F3}
.dark{--header-bg:linear-gradient(135deg,#374151 0%,#1f2937 100%);--header-text:#f3f4f6;--card-bg:#1f2937;--card-text:#f3f4f6;--secondary-bg:#374151;--secondary-text:#e5e7eb;--border-color:#4b5563;--success-bg:#064e3b;--success-text:#a7f3d0;--error-bg:#7f1d1d;--error-text:#fecaca;--warning-bg:#78350f;--warning-text:#fde68a;--info-bg:#1e3a8a;--info-text:#bfdbfe;--info-border:#3b82f6}
.result-card{background:var(--card-bg);color:var(--card-text);border:1px solid var(--border-color)}
"""

# JS to toggle theme
JS_TOGGLE = """
function toggleTheme() {
    const body = document.querySelector('body');
    if (body.classList.contains('dark')) {
        body.classList.remove('dark');
        return "🌙 Switch to Dark Mode";
    } else {
        body.classList.add('dark');
        return "☀️ Switch to Light Mode";
    }
}
"""

# Create Gradio interface
demo = gr.Blocks(title="Multimedia Deepfake Detector")

with demo:
    gr.HTML(f"<style>{CUSTOM_CSS}</style>")
    # Header
    # Header
    with gr.Row():
        with gr.Column(scale=10):
            gr.HTML("""
            <div style="text-align: center; padding: 30px; background: var(--header-bg); border-radius: 15px; margin-bottom: 30px;">
                <h1 style="color: var(--header-text); margin: 0; font-size: 2.8em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎬 Multimedia Deepfake Detector</h1>
                <p style="color: var(--header-text); margin: 15px 0 0 0; font-size: 1.2em; opacity: 0.9;">Powered by Hybrid LSTM-Transformer Deep Learning Model</p>
            </div>
            """)
        with gr.Column(scale=1, min_width=150):
            theme_btn = gr.Button("☀️ Switch to Light Mode", variant="secondary")
    
    theme_btn.click(None, None, theme_btn, js=JS_TOGGLE)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Upload Video")
            video_input = gr.Video(label="Select a video file to analyze")
            
            show_attention_checkbox = gr.Checkbox(
                label="Show Frame Attention Visualization",
                value=True
            )
            
            analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")
            
            gr.Markdown("""
            ### 📋 Instructions
            1. Upload a video file (MP4, AVI, MOV, etc.)
            2. Click "Analyze Video" to start detection
            3. View the results and attention visualization
            
            ### ℹ️ About
            This tool uses a state-of-the-art **Hybrid LSTM-Transformer** model 
            to detect AI-generated or manipulated videos (deepfakes).
            
            The model analyzes temporal patterns and visual artifacts across 
            multiple frames to determine authenticity.
            """)
        
        with gr.Column():
            gr.Markdown("### 📊 Analysis Results")
            result_output = gr.HTML(
                value="<p style='text-align: center; color: var(--secondary-text); padding: 40px;'>Upload a video and click 'Analyze Video' to see results here.</p>"
            )
            
            attention_output = gr.Image(label="Frame Attention Weights", type="pil")
            
            status_output = gr.Textbox(label="Status", interactive=False)
    
    # Example videos
    gr.Markdown("### 🎥 Example Videos")
    gr.Examples(
        examples=[
            ["videos/aaa.mp4"],
            ["videos/bbb.mp4"],
            ["videos/real-1.mp4"],
        ],
        inputs=video_input
    )
    
    # Event handlers
    analyze_btn.click(
        fn=predict_video,
        inputs=[video_input, show_attention_checkbox],
        outputs=[result_output, attention_output, status_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: var(--secondary-text); padding: 10px;">
        <p><strong>Multimedia Deepfake Detector</strong> | Hybrid LSTM-Transformer Model</p>
        <p>⚠️ This tool is for research and educational purposes. Results should be verified by experts.</p>
    </div>
    """)

if __name__ == "__main__":
    print("=" * 70)
    print("Multimedia Deepfake Detector - Gradio UI")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Frames per video: {NUM_FRAMES}")
    print("=" * 70)
    
    # Initialize model on startup
    print("Loading model...")
    if initialize_model():
        print("✓ Model loaded successfully!")
    else:
        print("⚠ Warning: Model failed to load. Will retry on first prediction.")
    
    print("\nLaunching Gradio interface...")
    demo.queue(max_size=10)  # Enable queue for better performance
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        show_error=True,
        inbrowser=True
    )
