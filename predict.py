import sys
import os
import numpy as np
from PIL import Image
from model import Model

def predict_class():
     
         
    # Define the directory containing the JPG images
    dir_path = "demonstration/"

    # Define the directory to save the NPY files
    save_dir = "demonstration/"

    # Get a list of all the JPG images in the directory
    jpg_files = [f for f in os.listdir(dir_path) if f.endswith(".jpg")]

    # Loop through each JPG image and save it as an NPY file
    for jpg_file in jpg_files:
        # Load the JPG image as a numpy array
        img = np.array(Image.open(os.path.join(dir_path, jpg_file))).reshape(1, -1)
    
        # Save the NPY file
        np.save(os.path.join(save_dir, jpg_file.replace(".jpg", ".npy")), img)


    print("\t Human Drawings:")
    # Loop through each file in the directory
    for filename in os.listdir("demonstration"):
        if filename.endswith(".npy"):
            # Load the image data
            image_to_predict = np.load(os.path.join("demonstration", filename))

            # Use the trained model to predict the label of the image
            predicted_label = Model.load_model("pika").predict(image_to_predict[:, :-1])

            # Print the predicted label and filename
            print("\t\tPredicted label for", filename, ":", predicted_label) 

    

 
predict_class()