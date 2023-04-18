import os

# Define the directory containing the images
image_directory = 'Simpson_man'

# List all files in the directory
files = os.listdir(image_directory)

# Initialize a counter for the renamed images
i = 1

# Loop through each file in the directory
for file in files:
    # Check if the file is a jpg image
    if file.endswith('.jpg') or file.endswith('.jpeg'):
        # Define the new name of the file
        new_name = f'train_{i}.jpg'

        # Rename the file
        os.rename(os.path.join(image_directory, file), os.path.join(image_directory, new_name))

        # Increment the counter
        i += 1