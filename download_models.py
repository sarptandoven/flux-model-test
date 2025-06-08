import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_models():
    """Download all required models for the ComfyUI Flux workflow"""
    base = Path("ComfyUI/models")
    base.mkdir(parents=True, exist_ok=True)
    
    # Define model files to download
    files = {
        base/"checkpoints"/"robux-machine.safetensors":
            ("sarptandoven/robux-machine-v2", "robux-machine.safetensors"),
        base/"vae"/"flux"/"ae.safetensors":
            ("ffxvs/vae-flux", "ae.safetensors"),
        base/"clip"/"EVA02_CLIP_L_336_psz14_s6B.pt":
            ("QuanSun/EVA-CLIP", "EVA02_CLIP_L_336_psz14_s6B.pt"),
        base/"text_encoders"/"EVA02_CLIP_L_336_psz14_s6B.pt":
            ("QuanSun/EVA-CLIP", "EVA02_CLIP_L_336_psz14_s6B.pt"),
        base/"text_encoders"/"sd3m"/"t5xxl_fp8_e4m3fn.safetensors":
            ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
        base/"pulid"/"pulid_flux_v0.9.0.safetensors":
            ("guozinan/PuLID", "pulid_flux_v0.9.0.safetensors"),
        base/"loras"/"flux"/"lora.safetensors":
            ("XLabs-AI/flux-RealismLora", "lora.safetensors"),
        base/"loras"/"flux"/"flux.1-turbo-alpha"/"diffusion_pytorch_model.safetensors":
            ("alimama-creative/FLUX.1-Turbo-Alpha", "diffusion_pytorch_model.safetensors"),
    }
    
    # Download each model
    for dst, (repo, fn) in files.items():
        dst = Path(dst)
        if dst.exists():
            print(f"✓ {dst.name} already exists")
            continue
            
        print(f"Downloading {fn} from {repo}...")
        dst.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=repo,
            filename=fn,
            local_dir=dst.parent,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded {dst.name}")
    
    # Create CLIP alias
    src = base/"text_encoders"/"EVA02_CLIP_L_336_psz14_s6B.pt"
    alias = src.with_suffix(".1.pt")
    if src.exists() and not alias.exists():
        shutil.copy(src, alias)
        print("✓ Created CLIP alias")
    
    print("✅ All models downloaded successfully")

if __name__ == "__main__":
    download_models() 