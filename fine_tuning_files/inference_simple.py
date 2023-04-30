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
    unet = UNet2DConditionModel.from_pretrained(unet_path)

    # Load the components into the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        config['pretrained_model_name_or_path'],
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
    )

    # Move the pipeline to the desired device
    pipeline = pipeline.to(config["device"])

    # Set the generator for random number generation
    generator = torch.Generator(device=config["device"]).manual_seed(config["seed"])

    for prompt in config["prompts"]:
        # Generate images for the given prompt
        images = []
        for i in range(config["num_images"]):
            images.append(pipeline(prompt, num_inference_steps=config["num_inference_steps"], generator=generator).images[0])

        # Save the generated images to the output directory
        tmp = os.path.join(config["output_dir"],'pirc_result')
        prompt_output_dir = os.path.join(tmp, prompt.replace(" ", "_"))
        os.makedirs(prompt_output_dir, exist_ok=True)
        for i, image in enumerate(images):
            name = str(prompt.replace(" ", "_"))+f"image_{i}.png"
            image.save(os.path.join(prompt_output_dir, name))

if __name__ == "__main__":
    config_path = "configuration_file/config_normal_test.json"
    config = load_config(config_path)
    main(config)