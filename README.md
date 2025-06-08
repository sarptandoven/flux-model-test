# ComfyUI Flux Professional Portrait Generator

This model uses ComfyUI and the Flux workflow to generate professional LinkedIn-style portraits from input images. It applies face swapping while preserving the original facial structure and hair style/color.

## Features

- Professional portrait generation with business attire
- Face swapping with identity preservation
- High-quality output with realistic details
- Customizable prompts and parameters

## Usage

### Input Parameters

- `image`: Input image to process (required)
- `prompt`: Positive prompt describing the desired output (optional)
- `negative_prompt`: Negative prompt to avoid unwanted features (optional)
- `guidance_scale`: Controls how closely the output follows the prompt (default: 2.5)
- `steps`: Number of denoising steps (default: 10)
- `seed`: Random seed for reproducibility (optional)

### Example

```python
import replicate

# Run the model
output = replicate.run(
    "your-username/comfyui-flux",
    input={
        "image": "path/to/input.jpg",
        "prompt": "Professional photorealistic LinkedIn portrait of a person...",
        "negative_prompt": "deformed, dark, unrealistic...",
        "guidance_scale": 2.5,
        "steps": 10
    }
)

# The output is a base64-encoded image
```

## Model Details

This model uses several components:
- Base model: robux-machine
- VAE: flux/ae.safetensors
- CLIP: EVA02_CLIP_L_336_psz14_s6B.pt
- LoRA: flux/lora.safetensors and flux.1-turbo-alpha
- PuLID: pulid_flux_v0.9.0.safetensors

## License

This model is released under the same license as the original ComfyUI and its components.

## Credits

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- Flux: https://github.com/alimama-creative/FLUX
- PuLID: https://github.com/guozinan/PuLID 