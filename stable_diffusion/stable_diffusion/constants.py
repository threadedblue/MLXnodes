_DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1-base"

_MODELS = {
    # Existing models (example from your MLX code)
    "stabilityai/sdxl-turbo": {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "text_encoder_2_config": "text_encoder_2/config.json",
        "text_encoder_2": "text_encoder_2/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
        "tokenizer_2_vocab": "tokenizer_2/vocab.json",
        "tokenizer_2_merges": "tokenizer_2/merges.txt",
    },
    "stabilityai/stable-diffusion-2-1-base": {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
    },

    # -------------------------------
    # NEW MODELS (placeholders/example)
    # -------------------------------

    # 1. stabilityai/stable-diffusion-3.5-large
    "stabilityai/stable-diffusion-3.5-large": {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
        # Depending on whether this model uses two text encoders
        # you might also need "text_encoder_2" entries:
        # "text_encoder_2_config": "text_encoder_2/config.json",
        # "text_encoder_2": "text_encoder_2/model.safetensors",
        # "tokenizer_2_vocab": "tokenizer_2/vocab.json",
        # "tokenizer_2_merges": "tokenizer_2/merges.txt",
    },

    # 2. stabilityai/stable-diffusion-3.5-large-turbo
    "stabilityai/stable-diffusion-3.5-large-turbo": {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
    },

    # 3. stabilityai/stable-diffusion-3.5-medium
    "stabilityai/stable-diffusion-3.5-medium": {
        "unet_config": "unet/config.json",
        "unet": "unet/diffusion_pytorch_model.safetensors",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder/model.safetensors",
        "vae_config": "vae/config.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "diffusion_config": "scheduler/scheduler_config.json",
        "tokenizer_vocab": "tokenizer/vocab.json",
        "tokenizer_merges": "tokenizer/merges.txt",
    },

    # 4. black-forest-labs/FLUX.1-schnell
    "black-forest-labs/FLUX.1-schnell": {
        "diffusion_config": "scheduler/scheduler_config.json",
        "text_encoder_config": "text_encoder/config.json",
        "text_encoder": "text_encoder_2/model.safetensors.index.json",        
        "tokenizer": "tokenizer/vocab.json",
        "tokenizer_2": "tokenizer_2/tokenizer.json",
        "vae": "vae/config.json",
    },
}   "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized":