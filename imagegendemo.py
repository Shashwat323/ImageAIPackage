import json

import torch
from diffusers import DDPMPipeline
from PIL import Image
import os
from safetensors.torch import load_file

from models import unet2d
from diffusers import DDPMScheduler


def generate_images(model_path, model_index_path, output_dir, num_images=16, seed=42):
    """
    Generates new images using the trained model.

    Args:
    - model_path: Path to the directory with the `.safetensors` weights files.
    - model_index_path: Path to `model_index.json` containing model component mapping.
    - output_dir: Path to save the generated images.
    - num_images: Number of images to generate.
    - seed: Random seed for reproducibility.

    Returns:
    - Saves generated images to the specified output directory.
    """

    # Load the model and its weights
    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    # Load the UNet model weights
    weights_file = str(os.path.join(model_path, model_index["unet"]))
    state_dict = load_file(weights_file)

    # Initialize the UNet model and load the weights
    model = unet2d.model
    model.load_state_dict(state_dict)

    # Move model to CUDA (if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define a noise scheduler (same as used during training)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initialize the DDPM pipeline
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

    # Generate random noise for backward diffusion
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Set the pipeline to device and evaluation mode
    pipeline.to(device)
    pipeline.unet.eval()

    # Generate sample images
    print("Generating images...")
    images = pipeline(batch_size=num_images, generator=generator).images

    # Save images
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(images):
        image_file = os.path.join(output_dir, f"generated_image_{i + 1}.png")
        image.save(image_file)
        print(f"Saved image: {image_file}")

    print("Image generation completed!")


# Example to use the function
if __name__ == "__main__":
    # Path where trained weights and model_index.json are stored
    model_path = "D:\\Other\\Repos\\ImageAIPackage\\ddpm-flowers-128\\unet"
    model_index_path = os.path.join(model_path, "model_index.json")

    # Output folder for generated images
    output_dir = "D:\\Other\\Repos\\ImageAIPackage\\ddpm-flowers-128\\samples"

    # Generate 16 images
    generate_images(model_path=model_path, model_index_path=model_index_path, output_dir=output_dir, num_images=16)