import os
import sys
import json
import torch
import asyncio
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from cog import BasePredictor, Input, Path as CogPath

# Add ComfyUI to path
COMFY_ROOT = Path("ComfyUI")
if str(COMFY_ROOT) not in sys.path:
    sys.path.append(str(COMFY_ROOT))

# Import ComfyUI components
from nodes import (
    NODE_CLASS_MAPPINGS,
    VAELoader,
    DualCLIPLoader,
    LoraLoaderModelOnly,
    EmptyLatentImage,
    SaveImage,
    LoadImage,
    CheckpointLoaderSimple,
    VAEDecode,
)

class ComfyFluxPredictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self._init_comfy()
        self._load_models()
        
    def _init_comfy(self):
        """Initialize ComfyUI environment"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Load custom nodes
        from nodes import init_extra_nodes
        init_extra_nodes()
        
    def _load_models(self):
        """Load all required models"""
        self.vae_loader = VAELoader()
        self.vae = self.vae_loader.load_vae(vae_name="flux/ae.safetensors")
        
        self.dual_clip_loader = DualCLIPLoader()
        self.clip = self.dual_clip_loader.load_clip(
            clip_name1="EVA02_CLIP_L_336_psz14_s6B.1.pt",
            clip_name2="sd3m/t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )
        
        self.checkpoint_loader = CheckpointLoaderSimple()
        self.model = self.checkpoint_loader.load_checkpoint(
            ckpt_name="robux-machine.safetensors"
        )
        
        self.lora_loader = LoraLoaderModelOnly()
        self.model = self.lora_loader.load_lora_model_only(
            lora_name="flux/lora.safetensors",
            strength_model=0.8,
            model=self.model[0],
        )
        
        self.model = self.lora_loader.load_lora_model_only(
            lora_name="flux/flux.1-turbo-alpha/diffusion_pytorch_model.safetensors",
            strength_model=0.7,
            model=self.model[0],
        )

    def predict(
        self,
        image: CogPath = Input(description="Input image to process"),
        prompt: str = Input(description="Positive prompt", default="Professional photorealistic LinkedIn portrait of a person; faceswapped; identical hair (style & color) and original facial structure preserved; confident, approachable expression; impeccably tailored business suit & perfectly knotted plain‑color tie with visible fine fabric fibers; medium portrait captured from several meters away (subject occupies roughly one‑third of frame height, background more pronounced), head‑to‑upper‑torso visible, no distracting hands foregrounded; shot with realistic body‑to‑head ratio"),
        negative_prompt: str = Input(description="Negative prompt", default="deformed, dark, unrealistic, plastic-like skin, blurry suit, unfocused suit, blurry tie, blurry edges, asymmetrical, angry, zoomed-in, too-close, facing away, blurred body, unfocused, blurry face, ears sticking out, deformed ears, small-sized body, long neck, blurred background, animated appearance, fake appearance, small body appearance, thin body, scrawny appearance, unconfident, tiny and thin shoulders, hands, fingers"),
        guidance_scale: float = Input(description="Guidance scale", default=2.5),
        steps: int = Input(description="Number of steps", default=10),
        seed: int = Input(description="Random seed", default=None),
    ) -> str:
        """Run a single prediction on the model"""
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        # Load input image
        load_image = LoadImage()
        input_image = load_image.load_image(image=str(image))
        
        # Create empty latent
        empty_latent = EmptyLatentImage()
        latent = empty_latent.generate(
            width=512,
            height=512,
            batch_size=1,
        )
        
        # Encode prompts
        caching_clip = NODE_CLASS_MAPPINGS["CachingCLIPTextEncode"]()
        positive = caching_clip.encode(
            text=prompt,
            clip=self.clip[0],
        )
        
        negative = caching_clip.encode(
            text=negative_prompt,
            clip=self.clip[0],
        )
        
        # Setup sampling
        ksampler = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        sampler = ksampler.get_sampler(sampler_name="euler")
        
        random_noise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        noise = random_noise.get_noise(noise_seed=seed)
        
        # Apply PuLID Flux
        apply_pulid = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        model = apply_pulid.apply_pulid_flux(
            weight=0.9,
            start_at=0.1,
            end_at=1,
            model=self.model[0],
            pulid_flux=NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]().load_model("pulid_flux_v0.9.0.safetensors")[0],
            eva_clip=NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]().load_eva_clip()[0],
            face_analysis=NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]().load_insightface(provider="CUDA")[0],
            image=input_image[0],
            unique_id=seed,
        )
        
        # Forward override
        forward_override = NODE_CLASS_MAPPINGS["FluxForwardOverrider"]()
        model = forward_override.apply_patch(model=model[0])
        
        # Apply TEA cache patch
        tea_cache = NODE_CLASS_MAPPINGS["ApplyTeaCachePatch"]()
        model = tea_cache.apply_patch(
            rel_l1_thresh=0.4,
            cache_device="offload_device",
            wan_coefficients="disabled",
            model=model[0],
        )
        
        # Setup guidance
        flux_guidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        positive = flux_guidance.append(
            guidance=guidance_scale,
            conditioning=positive[0],
        )
        
        negative = flux_guidance.append(
            guidance=guidance_scale,
            conditioning=negative[0],
        )
        
        # Setup CFG
        cfg_guider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        guider = cfg_guider.get_guider(
            cfg=1,
            model=model[0],
            positive=positive[0],
            negative=negative[0],
        )
        
        # Setup sampling
        model_sampling = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        model = model_sampling.patch(
            max_shift=1,
            base_shift=0.5,
            width=512,
            height=512,
            model=self.model[0],
        )
        
        basic_scheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        sigmas = basic_scheduler.get_sigmas(
            scheduler="simple",
            steps=steps,
            denoise=1,
            model=model[0],
        )
        
        # Run sampling
        sampler_advanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        samples = sampler_advanced.sample(
            noise=noise[0],
            guider=guider[0],
            sampler=sampler[0],
            sigmas=sigmas[0],
            latent_image=latent[0],
        )
        
        # Decode
        vae_decode = VAEDecode()
        image = vae_decode.decode(
            samples=samples[0],
            vae=self.vae[0],
        )
        
        # Save image
        save_image = SaveImage()
        output_path = save_image.save_images(
            filename_prefix="output",
            images=image[0]
        )
        
        # Convert to base64
        with open(output_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode()

if __name__ == "__main__":
    predictor = ComfyFluxPredictor()
    predictor.setup()
    
    app = FastAPI()
    
    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        prompt: str = "Professional photorealistic LinkedIn portrait of a person; faceswapped; identical hair (style & color) and original facial structure preserved; confident, approachable expression; impeccably tailored business suit & perfectly knotted plain‑color tie with visible fine fabric fibers; medium portrait captured from several meters away (subject occupies roughly one‑third of frame height, background more pronounced), head‑to‑upper‑torso visible, no distracting hands foregrounded; shot with realistic body‑to‑head ratio",
        negative_prompt: str = "deformed, dark, unrealistic, plastic-like skin, blurry suit, unfocused suit, blurry tie, blurry edges, asymmetrical, angry, zoomed-in, too-close, facing away, blurred body, unfocused, blurry face, ears sticking out, deformed ears, small-sized body, long neck, blurred background, animated appearance, fake appearance, small body appearance, thin body, scrawny appearance, unconfident, tiny and thin shoulders, hands, fingers",
        guidance_scale: float = 2.5,
        steps: int = 10,
        seed: Optional[int] = None,
    ):
        # Save uploaded image
        temp_path = "temp_input.png"
        with open(temp_path, "wb") as f:
            f.write(await image.read())
            
        # Run prediction
        result = predictor.predict(
            image=CogPath(temp_path),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
        )
        
        # Cleanup
        os.remove(temp_path)
        
        return {"image": result}
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 