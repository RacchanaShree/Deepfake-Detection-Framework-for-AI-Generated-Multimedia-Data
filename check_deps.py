import sys
import importlib

packages = [
    "gradio",
    "torch",
    "tensorflow",
    "cv2",
    "moviepy",
    "onnx",
    "onnx2pytorch",
    "timm"
]

print(f"Python version: {sys.version}")

missing = []
for p in packages:
    try:
        importlib.import_module(p)
        print(f"[OK] {p}")
    except ImportError:
        if p == "cv2":
            try:
                importlib.import_module("cv2")
                print(f"[OK] {p}")
            except ImportError:
                print(f"[MISSING] {p}")
                missing.append(p)
        else:
            print(f"[MISSING] {p}")
            missing.append(p)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print("Please run: pip install -r requirements.txt")
else:
    print("All dependencies seem to be installed.")
