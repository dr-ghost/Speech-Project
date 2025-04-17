import os
import sys
import torch
from models import vc_demo

if __name__ == "__main__":
    # Run the voice conversion demo
    src_path, dest_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Check if the source and destination paths are valid
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if not os.path.isfile(dest_path):
        raise FileNotFoundError(f"Destination file not found: {dest_path}")
    
    vc_demo(src_path, dest_path, out_path)