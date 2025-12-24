from nn import Neural_Network
from plot_chart import plot
from typing import Dict, List
import numpy as np

"""
This file consists of all the hyperparameter tuning methods. Most of them were useful during
the designing and testing of the model but are no longer used. They are displayed to show 
how the determined hypertuned variables were calculated. 
"""
"""
This method will find the index of the element where the list no longer decreases by 
the cutoff value. If the cutoff value is 0.01, we return the list index where we no longer
see a 1% decrease in value within the list. 
"""
def detect_cutoff(l: List, cutoff: float) -> int:
    # Return the int at which the values decrease by less than cutoff - 1 index
    iterator = 1
    previous_val = 0
    for iterator in range(1, len(l)):
        # Keep iterating until values decrease by less than cutoff
        if l[iterator] >= (l[previous_val] * (1 - cutoff)):
            return iterator - 1
        # If valid, keep incrementing
        previous_val = iterator
        iterator += 1

# Build the regression nn with a large number of epochs to determine optimum epoch count
"""
This method tests out the losses of a simple neural network over 1000 epochs. Once we detect that
the loss is failing to decrease by 0.1 %, that is where we draw the line for maximum number of 
epochs to run. We also draw a plot to verify if our cutoff epoch is valid. 
"""
def tune_nn_epochs (dataset: Dict, label_type: str, eta: float, plot_flag: bool) -> int:
    max_epochs = 1000
    # Tune neural network
    print("Tuning Epochs")
    # Get magnitude of error
    nn = Neural_Network(label_type, 0, 0, eta, 'loss', max_epochs)
    losses = nn.build_nn(dataset['features'], dataset['label'])
    losses = np.abs(losses)
    # Cut off epoch the moment we detect loss improving by less than 0.1%
    best_epoch = detect_cutoff(losses, 0.001)
    print(f"Best Gradient at epoch: {best_epoch}")
    # Plot the loss per epoch if bool set
    x_values = list(range(0, max_epochs))
    if plot_flag:
        plot(x_values, losses, "Loss per Epoch", "Epoch", "Loss")
    print("Hyperameter Epoch Tuned")
    return best_epoch

"""
This method tests the optimum hidden nodes combination. This is done by calculating the loss
for a NN with 2 hidden layers consisting of 1 through 15 nodes each and then storing the combination
that results in the lowest final losss value. It is expected that the loss within any neural network
decreases over time and so we disregard all but the last loss value. 
"""
def tune_nn_hidden_layers(dataset: Dict, label_type: str, eta: float, epochs: int) -> List:
    # Tune neural network with hidden layers with nodes ranging from 1-15 each. 
    # Retrieve the hidden node counts with lowest loss
    lowest_loss = 99999
    best_nodes = []
    print("Tuning hidden layer nodes")
    for hidden1 in range(1, 16):
        for hidden2 in range(1, 16):
            nn = Neural_Network(label_type, 2, [hidden1, hidden2], eta, 'loss', epochs)
            losses = nn.build_nn(dataset['features'], dataset['label'])
            # Get final loss
            loss = losses[len(losses) - 1]
            # Print result for tracking
            print(f"Hidden | [{hidden1}, {hidden2}]: {loss}")
            # Compare to best
            if loss < lowest_loss:
                best_nodes = [hidden1, hidden2]
                lowest_loss = loss
    print(f"Lowest loss at nodes {best_nodes}")

"""
Similar to the tune_nn_hidden_layers method, we tune the encoding and hidden layer node count for 
an autoencoder trained NN. This is done identically to it's sister method with the limitation that
the encoding layer must have less nodes than the input layer. 
"""
def tune_autoencoder_NN(dataset: Dict, label_type: str, eta: float, epochs: int) -> List:
    # Tune Autoencoder_NN with encoding layers STRICTLY smaller than input feature count
    max_encoding_count = len(dataset['features']) 
    # Hidden layer shall be again ranging from 1-15\
    lowest_loss = 99999
    best_nodes = []
    print("Tuning Autoencoder")
    for encoding in range(1, max_encoding_count):
        for hidden in range(1, 16):
            nn = Neural_Network(label_type, 2, [encoding, hidden], eta, 'loss', epochs)
            losses = nn.fit_autoencoder(dataset['features'], dataset['label'])
            # Get final loss
            loss = losses[len(losses) - 1]
            # Print result for tracking
            print(f"Autoencoder | [{encoding}, {hidden}]: {loss}")
            if loss < lowest_loss:
                best_nodes = [encoding, hidden]
                lowest_loss = loss
    print(f"Lowest loss at nodess {best_nodes}")

"""
This function trains all of the hyperparameters. 
"""
def tune_all (file_name: str, dataset: Dict):
    # Call all tuning mechanics from the given dataset and filename
    # Determine what type of class label we are working with
    if (file_name == "abalone.data" or
        file_name == "forestfires.data" or
        file_name == "machine.data"): 
        class_label_type = "regression"
    elif (file_name == "breast-cancer-wisconsin.data" or
          file_name == "car.data" or
          file_name == "house-votes-84.data"):
        class_label_type = "classification"
    else:
        print("Not a valid file")
        return
    
    # Tune simple nn for epochs
    eta = 0.0001
    epochs = tune_nn_epochs(dataset, class_label_type, eta, True)
    # Tune hidden layer nn for hidden nodes
    tune_nn_hidden_layers(dataset, class_label_type, eta, epochs)
    # Tune Autoencoded NN
    tune_autoencoder_NN(dataset, class_label_type, eta, epochs)

"""
After tuning all of the hyperparameters, this method serves as a reference to grab all of the 
tuned hyperparameters. These values were either determined by the above mentioned tuning methods 
or by visual analysis of the loss plots. 
"""
def get_tuned_values(file_name: str) -> List:
    # Return all the tuned hyperparameters of each file
    # Return epochs, eta, [hidden nodes], [autoencoder nodes]
    if file_name == "abalone.data":
        return "regression", 542, 0.001, [1, 8], [4, 5]
    elif file_name == "breast-cancer-wisconsin.data":
        return "classification", 408, 0.005, [15, 15], [4, 5]
    elif file_name == "car.data":
        return "classification", 500, 0.001, [3, 15], [4, 5]
    elif file_name == "forestfires.data":
        return "regression", 200, 0.01, [7, 4], [4, 5]
    elif file_name == "house-votes-84.data":
        return "classification", 293, 0.1, [5, 4], [4, 5]
    elif file_name == "machine.data":
        return "regression", 482, 0.0005, [1, 2], [4, 5]
    else: 
        print("Not a valid file")
        return None