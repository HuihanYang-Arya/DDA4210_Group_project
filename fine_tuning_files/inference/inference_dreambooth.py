import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
import json
import os
import torch
import argparse
import numpy as np


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main(config):
    # Set the paths
    output_dir = config['output_dir']
    vae_path = f'{output_dir}/vae'
    text_encoder_path = f'{output_dir}/text_encoder'
    unet_path = f'{output_dir}/unet'

    # Load the individual components
    vae = AutoencoderKL.from_pretrained(vae_path)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
    tokenizer = CLIPTokenizer.from_pretrained(text_encoder_path)  # Assuming the tokenizer is located in the same folder as the text_encoder
    unet = UNet2DConditionModel.from_pretrained(unet_path)

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config['pretrained_model_name_or_path'],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
    )

    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(config["device"])
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if config['seed'] is None else torch.Generator(device=config["device"]).manual_seed(config['seed'])
    images = []
    for _ in range(config['num_validation_images']):
        with torch.autocast("cuda"):
            image = pipeline(config['validation_prompt'], num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    # Save the generated images to the output directory
    tmp = os.path.join(config["output_dir"], 'validation_result')
    prompt_output_dir = os.path.join(tmp, config['validation_prompt'].replace(" ", "_"))
    os.makedirs(prompt_output_dir, exist_ok=True)
    for i, image in enumerate(images):
        name = str(config['validation_prompt'].replace(" ", "_")) + f"_image_{i}.png"
        image.save(os.path.join(prompt_output_dir, name))

    return images

def parse_args():
    parser = argparse.ArgumentParser(description="Image generation and validation script")
    parser.add_argument(
        "--config_path",
        default="../../configuration_file/config_test.json",
        type=str,
        help="Path to the configuration file"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    main(config)

