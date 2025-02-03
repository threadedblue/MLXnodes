import math
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from PIL import Image

# Make sure these imports point to Apple’s MLX stable_diffusion code:
# Option A: local stable_diffusion folder
#from .stable_diffusion import StableDiffusion, StableDiffusionXL

# Option B: direct import if you have stable_diffusion in PYTHONPATH
from .stable_diffusion.stable_diffusion import StableDiffusion, StableDiffusionXL

class MLXImg2Img:
    """
    A ComfyUI node wrapping Apple's MLX script for image-to-image generation.
    Uses a source image plus textual prompt to produce new images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE", {
                    "tooltip": "The starting image, whose content will be transformed.",
                }),
                "prompt": ("STRING", {
                    "default": "A painting of a fantasy landscape",
                    "multiline": True,
                    "tooltip": "Text prompt describing the desired transformation.",
                }),
                "model": (["sd", "sdxl"], {
                    "default": "sdxl",
                    "tooltip": "Choose standard SD or SDXL model.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly to deviate from the original image (0->almost unchanged; 1->almost new).",
                }),
                "n_images": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images (batch size).",
                }),
                "steps": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Number of diffusion steps (0 uses script defaults).",
                }),
                "cfg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Classifier-Free Guidance scale (0 uses script defaults).",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional negative prompt to remove certain elements.",
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
                    "tooltip": "Batch size for decoding latents into images.",
                }),
                "quantize": (["false", "true"], {
                    "default": "false",
                    "tooltip": "Quantize linear layers/unet for smaller memory usage (experimental).",
                }),
                "float16": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Use FP16 if 'true' (recommended on M1/M2).",
                }),
                "preload_models": (["false", "true"], {
                    "default": "false",
                    "tooltip": "Force model to fully load into memory before generation.",
                }),
                "verbose": (["false", "true"], {
                    "default": "false",
                    "tooltip": "If 'true', prints extra memory usage info.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed (0 means random).",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "MLX Nodes"

    def generate(
        self,
        source_image,
        prompt,
        model,
        strength,
        n_images,
        steps,
        cfg,
        negative_prompt,
        n_rows,
        decoding_batch_size,
        quantize,
        float16,
        preload_models,
        verbose,
        seed
    ):
        # Convert booleans
        float16 = (float16 == "true")
        quantize = (quantize == "true")
        preload_models = (preload_models == "true")
        verbose = (verbose == "true")

        # If seed == 0, we let the code pick a random seed
        if seed == 0:
            seed = None

        # 1. Load the appropriate stable diffusion model
        if model == "sdxl":
            sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=float16)
        else:
            sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=float16)

        # 2. (Optional) quantize
        if quantize:
            if model == "sdxl":
                nn.quantize(sd.text_encoder_1, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.text_encoder_2, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.unet, group_size=32, bits=8)
            else:
                nn.quantize(sd.text_encoder, class_predicate=lambda _, m: isinstance(m, nn.Linear))
                nn.quantize(sd.unet, group_size=32, bits=8)

        # 3. Script defaults for steps/cfg
        if model == "sdxl":
            if cfg == 0.0:
                cfg = 0.0
            if steps == 0:
                steps = 2
        else:
            if cfg == 0.0:
                cfg = 7.5
            if steps == 0:
                steps = 50

        # 4. Fix steps if (steps * strength) < 1
        if int(steps * strength) < 1:
            steps = int(math.ceil(1 / strength))
            if verbose:
                print(f"Strength {strength} too low => steps set to {steps}")

        # 5. Preload if requested
        if preload_models:
            sd.ensure_models_are_loaded()

        # 6. Convert ComfyUI "IMAGE" => MLX array
        #    ComfyUI images are typically [B, H, W, 3], float32 in [0..1].
        #    We'll just take the first image in the batch (index 0).
        img_batch = source_image
        if len(img_batch.shape) == 4:
            # We'll forcibly pick the 0th image if there's a batch dimension
            img = img_batch[0]  # shape [H, W, 3]
        else:
            img = img_batch  # shape [H, W, 3] if not batched

        # shape is float in [0..1], so let's convert to Nx array format
        # Apple’s script expects [H, W, C], range -1..1
        h, w, c = img.shape
        # Round down to multiples of 64
        W = w - w % 64
        H = h - h % 64
        if (W != w) or (H != h):
            if verbose:
                print(f"Warning: image shape not divisible by 64 => resizing to {W}x{H}")
            # We'll do a simple area or nearest resize
            # For a quick approach, let's do "nearest" to match script.
            pil_im = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))
            pil_im = pil_im.resize((W, H), Image.NEAREST)
            # Convert back to Nx format
            arr = np.array(pil_im).astype(np.float32) / 255.0
        else:
            arr = img.cpu().numpy().astype(np.float32)

        # Now range [0..1], so script does (img * 2 - 1)
        arr = (arr[:, :, :3] * 2.0) - 1.0  # remove any alpha channel
        # Nx array
        arr_mlx = mx.array(arr)

        # 7. Generate latents from image
        latents = sd.generate_latents_from_image(
            arr_mlx,
            prompt,
            strength=strength,
            n_images=n_images,
            cfg_weight=cfg,
            num_steps=steps,
            negative_text=negative_prompt,
            seed=seed,
        )

        # Evaluate latents (like tqdm)
        for x_t in latents:
            mx.eval(x_t)

        # 8. Free memory from text encoders / unet
        if model == "sdxl":
            del sd.text_encoder_1
            del sd.text_encoder_2
        else:
            del sd.text_encoder
        del sd.unet
        del sd.sampler
        peak_mem_unet = mx.metal.get_peak_memory() / 1024**3

        # 9. Decode latents in small batches
        decoded = []
        for i in range(0, n_images, decoding_batch_size):
            chunk = sd.decode(x_t[i : i + decoding_batch_size])
            mx.eval(chunk)
            decoded.append(chunk)

        peak_mem_overall = mx.metal.get_peak_memory() / 1024**3

        # 10. Arrange results in a grid
        x = mx.concatenate(decoded, axis=0)  # [n_images, H, W, C]
        x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
        B, HH, WW, CC = x.shape
        x = x.reshape(n_rows, B // n_rows, HH, WW, CC).transpose(0, 2, 1, 3, 4)
        x = x.reshape(n_rows * HH, (B // n_rows) * WW, CC)
        x = (x * 255).astype(mx.uint8)

        # 11. Optionally print memory usage
        if verbose:
            print(f"[MLXImg2Img] Peak memory (UNet): {peak_mem_unet:.3f}GB")
            print(f"[MLXImg2Img] Peak memory overall: {peak_mem_overall:.3f}GB")

        # 12. Convert Nx array -> ComfyUI "IMAGE" (float in [0..1], shape [1, H, W, 3])
        final_np = x.cpu().numpy().astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_np).unsqueeze(0)  # shape [1, H, W, 3]

        return (final_tensor, )


# Module-level dict so ComfyUI recognizes this node
NODE_CLASS_MAPPINGS = {
    "MLXImg2Img": MLXImg2Img,
}
