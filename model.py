import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os

class Model:

    @staticmethod
    def train_model(data_folder):

        print("\n loading files... \n")  

        data_files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]  # get all npy files 

        # Load individual npy files containing data and create corresponding label arrays
        X = np.empty((0, 783))  # empty array to store data
        y = np.empty((0,), dtype=str)  # empty array to store labels

        for file in data_files:
            data = np.load(os.path.join(data_folder, file))
            label = file[:-4]  # extract label from filename (remove .npy extension)
            labels = np.full((len(data),), label)
            X = np.concatenate((X, data[:, :-1]), axis=0)
            y = np.concatenate((y, labels), axis=0)

        print("\n split training and testing data... \n")
        # Create training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

        
        print("\n Training Neual Network... \n") 
        # Train a neural network classifier
        nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100) # nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100, verbose=2)

        # Iterate over the data in batches of 100 and call partial_fit for each batch

        num_batches = 100
        batch_size = len(X_train) // num_batches  

        print("\n Total: " + str(num_batches))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            nn.partial_fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx], classes=np.unique(y_train)) 
            percent_complete = (i / num_batches) * 100
            print(f"[{'=' * int(percent_complete / 2):48s}] {int(percent_complete)}%", end="\r")

        # Get the confusion matrix  
        print("Calculating the Confusion Matrix...")
        
        # Get predictions for the test set
        y_pred = nn.predict(X_test)

       # Get unique labels
        labels = np.unique(y_test)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

        # Print confusion matrix with labels
        print("Confusion Matrix:")
        print("      Predicted")
        print("      ", " ".join(labels))
        for i in range(len(labels)):
            row = " ".join(str(x) for x in conf_matrix[i])
            print(f"True {labels[i]} {row}")

        return nn, conf_matrix 
    
    @staticmethod
    def train_model_2(data_folder):

        print("\n loading files... \n")  

        data_files = [f for f in os.listdir(data_folder) if f.endswith(".npy")]  # get all npy files 

        # Load individual npy files containing data and create corresponding label arrays
        X = np.empty((0, 783))  # empty array to store data
        y = np.empty((0,), dtype=str)  # empty array to store labels

        for file in data_files:
            data = np.load(os.path.join(data_folder, file))
            label = file[:-4]  # extract label from filename (remove .npy extension)
            labels = np.full((len(data),), label)
            X = np.concatenate((X, data[:, :-1]), axis=0)
            y = np.concatenate((y, labels), axis=0)

        print("\n split training and testing data... \n")
        # Create training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 

        
        print("\n Training Neual Network... \n") 
        # Train a neural network classifier
        nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100) # nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100, verbose=2)

        # Iterate over the data in batches of 100 and call partial_fit for each batch

        num_batches = 100
        batch_size = len(X_train) // num_batches  

        print("\n Total: " + str(num_batches))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            nn.partial_fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx], classes=np.unique(y_train)) 
            percent_complete = (i / num_batches) * 100
            print(f"[{'=' * int(percent_complete / 2):48s}] {int(percent_complete)}%", end="\r")

        # Get the confusion matrix  
        print("Calculating the Confusion Matrix...")
        # Get predictions for the test set
        y_pred = nn.predict(X_test)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(conf_matrix)

        return nn, conf_matrix 
    
    @staticmethod
    def save_model(model, file_name):
 
        # Save the trained model
        with open("models/"+file_name+ ".pkl", 'wb') as file:
            pickle.dump(model, file) 

        return
    
    @staticmethod
    def load_model(file_name): 
        
        # Load the saved model
        with open("models/"+file_name+ ".pkl", 'rb') as file:
            nn = pickle.load(file)

        return nn
    
    @staticmethod
    def use_model(model, folder_to_be_tested):  

        print("\t Human Drawings")
        # Loop through each file in the directory
        for filename in os.listdir(folder_to_be_tested):
            if filename.endswith(".npy"):
                # Load the image data
                image_to_predict = np.load(os.path.join(folder_to_be_tested, filename))

                # Use the trained model to predict the label of the image
                predicted_label = model.predict(image_to_predict[:, :-1])

                # Print the predicted label and filename
                print("\t\tPredicted label for", filename, ":", predicted_label) 
        
        return
    
    @staticmethod
    def get_accuracy(model, folder_used_to_test):  

        # print("\n loading files... \n")
        # Load individual npy files containing data
        data_files = [f for f in os.listdir(folder_used_to_test) if f.endswith(".npy")]  # get all npy files 

        # Load individual npy files containing data and create corresponding label arrays
        X = np.empty((0, 783))  # empty array to store data
        y = np.empty((0,), dtype=str)  # empty array to store labels

        for file in data_files:
            data = np.load(os.path.join(folder_used_to_test, file))
            label = file[:-4]  # extract label from filename (remove .npy extension)
            labels = np.full((len(data),), label)
            X = np.concatenate((X, data[:, :-1]), axis=0)
            y = np.concatenate((y, labels), axis=0) 

        # Create training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate the neural network classifier
        nn_score_test = model.score(X_test, y_test)
        nn_score_train = model.score(X_train, y_train)
        
        
        return nn_score_train, nn_score_test
    


if __name__ == "__main__":
    
    model, conf_mtrx = Model.train_model("training_data")
    Model.save_model(model, "test")

    Model.use_model(Model.load_model("test"), "human_data")
    
    #nn_score_train, nn_score_test= Model.get_accuracy(Model.load_model("ff_nn"), "training_data")
    #print("\tTraining accuracy:", nn_score_train)
    #print("\tTesting accuracy:", nn_score_test) 
    """
    
    print("\n\ntrained_nn\n")
    Model.use_model(Model.load_model("trained_nn"), "human_data")
    print("\n\nnn_bad_model\n")
    Model.use_model(Model.load_model("nn_bad_model"), "human_data")
    print("\n\nnn_great_model\n")
    Model.use_model(Model.load_model("nn_great_model"), "human_data")
    print("\n\nnn_test_model\n")
    Model.use_model(Model.load_model("nn_test_model"), "human_data")
    print("\n\nnn_graffed\n")
    Model.use_model(Model.load_model("nn_graffed"), "human_data")
    print("\n\ntest\n")
    Model.use_model(Model.load_model("test2"), "human_data")
   """

    """
    
    print("\n\ntrained_nn\n")
    Model.use_model(Model.load_model("trained_nn"), "human_data")

    nn_score_train, nn_score_test= Model.get_accuracy(Model.load_model("trained_nn"), "training_data")
    print("\tTraining accuracy:", nn_score_train)
    print("\tTesting accuracy:", nn_score_test)
    
    
    print("\n\nnn_bad_model\n")
    Model.use_model(Model.load_model("nn_bad_model"), "human_data")
    
    nn_score_train, nn_score_test= Model.get_accuracy(Model.load_model("nn_bad_model"), "training_data")
    print("\tTraining accuracy:", nn_score_train)
    print("\tTesting accuracy:", nn_score_test)
    

    print("\n\nnn_great_model\n")
    Model.use_model(Model.load_model("nn_great_model"), "human_data")
    
    nn_score_train, nn_score_test= Model.get_accuracy(Model.load_model("nn_great_model"), "training_data")
    print("\tTraining accuracy:", nn_score_train)
    print("\tTesting accuracy:", nn_score_test)

    print("\n\nnn_test_model\n")
    Model.use_model(Model.load_model("nn_test_model"), "human_data")
    
    nn_score_train, nn_score_test= Model.get_accuracy(Model.load_model("nn_test_model"), "training_data")
    print("\tTraining accuracy:", nn_score_train)
    print("\tTesting accuracy:", nn_score_test)
    

    """