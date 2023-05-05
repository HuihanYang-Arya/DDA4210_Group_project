import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Define the root directory containing the folders
root_dir = "path/to/your/directory"

# Initialize the total sum of logits
total_logits_sum = 0

# Loop through the 10 folders
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    
    if not os.path.isdir(folder_path):
        continue

    # Extract the text from the folder name
    text = folder_name

    # Initialize the folder sum of logits
    folder_logits_sum = 0

    # Loop through the 10 images in the folder
    image_count = 0
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)

        if not os.path.isfile(image_path):
            continue

        image = Image.open(image_path)

        # Prepare the inputs
        inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

        # Get the logits per image
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image

        # Accumulate the logits for the folder
        folder_logits_sum += logits_per_image.item()
        image_count += 1

    # Calculate the average logits for the folder
    folder_logits_avg = folder_logits_sum / image_count

    # Accumulate the folder logits
    total_logits_sum += folder_logits_avg

# Calculate the average logits for all folders
average_logits_all_folders = total_logits_sum / 10
print("Average logits for all folders:", average_logits_all_folders)
