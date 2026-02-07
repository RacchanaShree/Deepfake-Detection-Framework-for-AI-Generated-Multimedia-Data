
import os
import shutil
import sys

def debug_move():
    target_dir = 'docs'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created {target_dir}")
    else:
        print(f"{target_dir} exists")

    files = [
        'ICSSR_final_submission.md',
        'Proposal-2025-26.docx',
        'POSTER_SVIT PE_CSE.pptx'
    ]

    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))

    for f in files:
        if os.path.exists(f):
            print(f"Found {f}, moving...")
            try:
                shutil.move(f, os.path.join(target_dir, f))
                print(f"SUCCESS: Moved {f}")
            except Exception as e:
                print(f"ERROR moving {f}: {e}")
        else:
            print(f"NOT FOUND: {f}")

if __name__ == "__main__":
    debug_move()
