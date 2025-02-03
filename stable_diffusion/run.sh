#!/usr/bin/env bash

# Ensure a mode is provided
if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 txt2img|img2img"
    exit 1
fi

MODE="$1"

# Define script paths
SD_TXT2IMG="txt2image.py"
SD_IMG2IMG="image2image.py"

# Define arguments **inside** the script
if [[ "$MODE" == "txt2img" ]]; then
    PROMPT="young woman with pink hair and an orange and white striped rugby shirt"
    CMD=("python3" "$SD_TXT2IMG" "$PROMPT")

    # Optional arguments for txt2img
    ARGS=(
        "--model" "sd"
        "--n_images" "4"
        "--steps" "25"
        "--cfg" "7.5"
        "--negative_prompt" "no background"
        "--n_rows" "1"
        "--decoding_batch_size" "1"
        "--no-float16"
        "--quantize"
        "--preload-models"
        "--output" "penny-prior.png"
        "--seed" "4759"
        "--verbose"
    )

elif [[ "$MODE" == "img2img" ]]; then
    IMAGE="input.jpg"
    PROMPT="young woman with pink hair and an orange and white striped rugby shirt"
    CMD=("python3" "$SD_IMG2IMG" "$IMAGE" "$PROMPT")

    # Optional arguments for img2img
    ARGS=(
        "--model" "sdxl"
        "--strength" "0.8"
        "--n_images" "4"
        "--steps" "25"
        "--cfg" "7.5"
        "--negative_prompt" "no background"
        "--n_rows" "1"
        "--decoding_batch_size" "1"
        "--no-float16"
        "--quantize"
        "--preload-models"
        "--output" "transformed.png"
        "--seed" "1234"
        "--verbose"
    )

else
    echo "Error: Invalid mode. Use 'txt2img' or 'img2img'."
    exit 1
fi

# Append additional arguments
CMD+=("${ARGS[@]}")

# Run the command
echo "Running: ${CMD[@]}"
"${CMD[@]}"
