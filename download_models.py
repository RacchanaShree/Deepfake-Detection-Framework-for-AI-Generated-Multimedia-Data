import os
import urllib.request

def download_file(url, filename):
    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

def main():
    checkpoints_dir = "checkpoints"
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    files = {
        "model.pth": "https://github.com/Pranesh-2005/AI-Generated-Video-Detector/raw/main/checkpoints/model.pth",
        "efficientnet.onnx": "https://github.com/Pranesh-2005/AI-Generated-Video-Detector/raw/main/checkpoints/efficientnet.onnx"
    }

    for filename, url in files.items():
        filepath = os.path.join(checkpoints_dir, filename)
        # Check if file exists and is small (LFS pointer)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size < 1000: # It's likely a pointer
                print(f"{filename} seems to be an LFS pointer (size: {size} bytes). Downloading actual file...")
                download_file(url, filepath)
            else:
                print(f"{filename} already exists and seems large enough ({size} bytes). Skipping.")
        else:
            download_file(url, filepath)

if __name__ == "__main__":
    main()
