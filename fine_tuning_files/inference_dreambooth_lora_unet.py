import torch
import json
import os
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main(config):
    # Load the trained pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        config["pretrained_model_name_or_path"], revision=config["revision"], torch_dtype=torch.float32
    )
    pipeline = pipeline.to(config["device"])

    # Load attention processors
    pipeline.unet.load_attn_procs(config["output_dir"])

    # Load UNet state_dict
    unet_state_dict = torch.load(os.path.join(config["output_dir"], "unet.pth"), map_location=config["device"])
    pipeline.unet.load_state_dict(unet_state_dict)

    pipeline.to("cuda")

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

import os
if __name__ == "__main__":
    config_path = "configuration_file/config_unfreezed_test.json"
    current_path = os.getcwd()
    print("Current Path: ", current_path)

    config_path = "../configuration_file/config_test.json"
    # Change the config --output_dir to "../stored_parameters_for_models/sd-model-finetuned-unfreezed-unet-last1-2"
    
    config = load_config(config_path)
    main(config)