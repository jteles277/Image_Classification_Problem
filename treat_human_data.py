"""from PIL import Image
import numpy as np

# Load the JPEG image using Pillow
image = Image.open("my_axe.jpg")
  

# Convert the grayscale image to a 1-dimensional array
array = np.array(image).reshape(1, -1)

# Save the array as a NumPy binary file
np.save("my_axe.npy", array)
"""

import os
import numpy as np
from PIL import Image

# Define the directory containing the JPG images
dir_path = "human_data/"

# Define the directory to save the NPY files
save_dir = "human_data/"

# Get a list of all the JPG images in the directory
jpg_files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

# Loop through each JPG image and save it as an NPY file
for jpg_file in jpg_files:
    # Load the JPG image as a numpy array
    img = np.array(Image.open(os.path.join(dir_path, jpg_file))).reshape(1, -1)
 
    # Save the NPY file
    np.save(os.path.join(save_dir, jpg_file.replace(".jpg", ".npy")), img)