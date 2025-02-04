import numpy as np
import torch
from PIL import Image
import mlx.core as mx
import mlx.nn as nn

# Import your MLX-based stable diffusion classes
from .stable_diffusion.stable_diffusion import StableDiffusion, StableDiffusionXL

class MLXText2Image:
    """
    A ComfyUI node wrapping Apple's MLX text-to-image script (for M1/M2).
    Generates an image (grid if multiple) from a text prompt.
    """

    models = ["stabilityai/stable-diffusion-3.5-large"
              , "stabilityai/stable-diffusion-3.5-large-turbo"
              , "stabilityai/stable-diffusion-3.5-medium"
              , "black-forest-labs/FLUX.1-schnell"]

    @classmethod
    def INPUT_TYPES(cls):
       return {
            "required": {
                "prompt": ("STRING", {
                    "default": "A scenic lake at sunset",
                    "multiline": True,
                    "tooltip": "Text prompt describing what you want to generate.",
                }),
                "model": (MLXText2Image.models, {
                    "default": MLXText2Image.models[3],
                    "tooltip": "Choose either standard SD or SDXL.",
                }),
                "n_images": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to generate.",
                }),
                "steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Diffusion steps (0 = use script's default).",
                }),
                "cfg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "CFG scale (0 = use script's default).",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional negative prompt.",
                }),
                "n_rows": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Rows in the final image grid.",
                }),
                "decoding_batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "tooltip": "Batch size used during decoding.",
                }),
                "float16": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Use FP16 if 'true' (recommended on M1/M2).",
                }),
                "quantize": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Quantize linear layers/unet (experimental).",
                }),
                "preload_models": (["true", "false"], {
                    "default": "false",
                    "tooltip": "Force model to load into memory beforehand.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed (0 means random).",
                }),
                "verbose": (["false", "true"], {
                    "default": "false",
                    "tooltip": "Print extra memory usage info.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MLX Nodes"

    def generate(
        self,
        prompt,
        model,
        n_images,
        steps,
        cfg,
        negative_prompt,
        n_rows,
        decoding_batch_size,
        float16,
        quantize,
        preload_models,
        seed,
        verbose
    ):
        """
        Main logic: replicate the text-to-image flow from the original script.
        Returns a single ComfyUI 'IMAGE' (with shape [1, H, W, 3]) containing a grid.
        """
        print("generate==>")
        # Convert booleans
        float16 = (float16 == "true")
        quantize = (quantize == "true")
        preload_models = (preload_models == "true")
        verbose = (verbose == "true")
        if seed == 0:
            seed = None  # Let the code pick a random seed

        print("# 1. Load appropriate stable diffusion model " + model)
        sd = StableDiffusion(model, float16=float16)



        print("# 2. Optional: quantize")
        if quantize:
            nn.quantize(sd.text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear))
            nn.quantize(sd.unet, group_size=32, bits=8)

        print("# 3. Handle default steps/cfg based on the script logic")
        if cfg == 0.0:
            cfg = 7.5   # default from script
        if steps == 0:
            steps = 50  # default from script

        # 4. Optionally preload
        if preload_models:
            sd.ensure_models_are_loaded()
        
        print("# 5. Generate latents")
        latents = sd.generate_latents(
            prompt,
            n_images=n_images,
            cfg_weight=cfg,
            num_steps=steps,
            seed=seed,
            negative_text=negative_prompt,
        )

        # Evaluate in a loop to simulate tqdm steps
        for x_t in latents:
            mx.eval(x_t)

        # 6. Free up memory from text encoders / unet
        del sd.text_encoder
        del sd.unet
        del sd.sampler

        peak_mem_unet = mx.metal.get_peak_memory() / 1024**3

        # 7. Decode to images in small batches
        decoded = []
        for i in range(0, n_images, decoding_batch_size):
            chunk = sd.decode(x_t[i : i + decoding_batch_size])
            mx.eval(chunk)
            decoded.append(chunk)
        peak_mem_overall = mx.metal.get_peak_memory() / 1024**3

        # 8. Arrange them on a grid
        x = mx.concatenate(decoded, axis=0)  # shape [n_images, H, W, C]
        # Optionally add padding
        x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])  # just like script
        B, H, W, C = x.shape
        # Reshape into [n_rows, B/n_rows, H, W, C], then transpose for grid
        x = x.reshape(n_rows, B // n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
        x = x.reshape(n_rows * H, (B // n_rows) * W, C)
        x = (x * 255).astype(mx.uint8)

        # 9. (Optional) Print memory usage
        if verbose:
            print(f"[MLXText2Image] Peak memory (UNet): {peak_mem_unet:.3f} GB")
            print(f"[MLXText2Image] Peak memory overall: {peak_mem_overall:.3f} GB")

        # 10. Convert final numpy array -> torch float (ComfyUI "IMAGE" format)
        img_tensor = torch.from_numpy(x.astype(np.float32) / 255.0)
        # [H, W, 3] -> [1, H, W, 3]
        img_tensor = img_tensor.unsqueeze(0)
        print("<==generate")
        return (img_tensor, )


# Dict so ComfyUI can find/register this node
NODE_CLASS_MAPPINGS = {
    "MLXText2Image": MLXText2Image
}
