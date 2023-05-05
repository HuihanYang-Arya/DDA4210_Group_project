import torch
import json
import os
import argparse
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main(config):
    # Load the trained pipeline
    pipeline = DiffusionPipeline.from_pretrained(torch_dtype=torch.float32)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(config["device"])

    # Load attention processors
    pipeline.unet.load_attn_procs(config["output_dir"])

    # Set the generator for random number generation
    generator = torch.Generator(device=config["device"]).manual_seed(config["seed"])

    prompt = config["validation_prompt"]
    num_validation_images = config["num_validation_images"]

    # Generate and save validation images
    images = [
        pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
        for _ in range(num_validation_images)
    ]

    for i, image in enumerate(images):
        name = f"validation_image_{i}.png"
        image.save(os.path.join(config["output_dir"], name))

def parse_args():
    parser = argparse.ArgumentParser(description="Image generation script")
    parser.add_argument(
        "--config_path",
        default="configuration_file/config_test.json",
        type=str,
        help="Path to the configuration file"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    main(config)
