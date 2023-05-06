import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import sys

# Load the NumPy array from the .npy file
array = np.load('human_data/my_axe.npy')
 

# Print the shape and contents of the array
print('Array shape:', array.shape)
print('Array contents:') 
print(array)
"""
for i in range(0,5):
    # Convert the pixel values to a Pillow Image object
    image = Image.fromarray(array[i].reshape((int(np.sqrt(len(array[i]))), int(np.sqrt(len(array[i]))))).astype('uint8'))

    # Save the image to a PNG file
    image.save('my_image' +str(i) +'.png')
 """