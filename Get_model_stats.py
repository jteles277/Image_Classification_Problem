import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os



def plot_pls(train, test):
    plt.plot(train, color='green', alpha=0.8, label='Train')
    plt.plot(test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()

data_folder = "training_data"

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
nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100, verbose=2) # nn = MLPClassifier(hidden_layer_sizes=(784,300,60,50,40,10), max_iter=100, verbose=2)

# Iterate over the data in batches of 100 and call partial_fit for each batch

num_batches = 100
batch_size = len(X_train) // num_batches 

scores_train = []
scores_test = []

print("\n Total: " + str(num_batches))

for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(X_train))
    nn.partial_fit(X_train[start_idx:end_idx], y_train[start_idx:end_idx], classes=np.unique(y))

     
    # SCORE TRAIN
    scores_train.append(nn.score(X_train, y_train))
    # SCORE TEST
    scores_test.append(nn.score(X_test, y_test))

    print()
     
    
 
for i in scores_train:
    print(i)
print("")
for i in scores_test:
    print(i) 
 

import pickle

# Assume you have trained your neural network and called it 'nn'
# ...

# Save the trained model
with open('nn_graffed.pkl', 'wb') as file:
    pickle.dump(nn, file)

# Evaluate the neural network classifier
nn_score = nn.score(X_test, y_test)
print("Neural Network accuracy:", nn_score)

 