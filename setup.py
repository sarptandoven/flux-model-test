import os
import sys
from pathlib import Path

def main():
    """Run the complete setup process"""
    # Add current directory to path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    
    # Import setup modules
    from setup_comfyui import setup_comfyui
    from download_models import download_models
    
    # Run setup steps
    print("Setting up ComfyUI...")
    setup_comfyui()
    
    print("\nDownloading models...")
    download_models()
    
    print("\n✅ Setup complete! The model is ready to use.")

if __name__ == "__main__":
    main() 