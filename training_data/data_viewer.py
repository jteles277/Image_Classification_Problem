import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import sys

# Load the NumPy array from the .npy file
array = np.load('axe.npy') 

# Convert the pixel values to a Pillow Image object
image = Image.fromarray(array[0].reshape((int(np.sqrt(len(array[0]))), int(np.sqrt(len(array[0]))))).astype('uint8'))

    # Save the image to a PNG file
image.save('my_image' +str(1) +'.png')

# Load the NumPy array from the .npy file
array = np.load('apple.npy') 

# Convert the pixel values to a Pillow Image object
image = Image.fromarray(array[3].reshape((int(np.sqrt(len(array[3]))), int(np.sqrt(len(array[3]))))).astype('uint8'))

    # Save the image to a PNG file
image.save('my_image' +str(2) +'.png')

# Load the NumPy array from the .npy file
array = np.load('sword.npy') 

# Convert the pixel values to a Pillow Image object
image = Image.fromarray(array[3].reshape((int(np.sqrt(len(array[3]))), int(np.sqrt(len(array[3]))))).astype('uint8'))

    # Save the image to a PNG file
image.save('my_image' +str(3) +'.png')

# Load the NumPy array from the .npy file
array = np.load('house.npy') 

# Convert the pixel values to a Pillow Image object
image = Image.fromarray(array[2].reshape((int(np.sqrt(len(array[2]))), int(np.sqrt(len(array[2]))))).astype('uint8'))

    # Save the image to a PNG file
image.save('my_image' +str(4) +'.png')

# Load the NumPy array from the .npy file
array = np.load('book.npy') 

# Convert the pixel values to a Pillow Image object
image = Image.fromarray(array[2].reshape((int(np.sqrt(len(array[2]))), int(np.sqrt(len(array[2]))))).astype('uint8'))

    # Save the image to a PNG file
image.save('my_image' +str(5) +'.png')

"""
for i in range(0,5):
    # Convert the pixel values to a Pillow Image object
    image = Image.fromarray(array[i].reshape((int(np.sqrt(len(array[i]))), int(np.sqrt(len(array[i]))))).astype('uint8'))

    # Save the image to a PNG file
    image.save('my_image' +str(i) +'.png')
 """