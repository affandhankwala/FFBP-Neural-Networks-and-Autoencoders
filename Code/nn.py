from matrix_calculations import create_X_matrix
from dataset_manipulation import exclude_features
from calculations import mean, get_mse_list, multiply_list_by_val, subtract_lists
from feature import Feature
from typing import List, Dict
from copy import deepcopy
import numpy as np

"""
This class is the heart of the program. This is where the neural network is created,
trained and test. This class is capable of creating a neural network with any number of 
hidden layers and any number of hidden nodes in said hidden layers--included 0 hidden
layers. The performance of these models can be tested as well. We can also create an 
autoencoded neural network that is capable of encoding and training a standard network
after encoding. The performance of this model can also be measured.
"""
# Class for neural networks
class Neural_Network:
    """
    Initialization function
    """
    def __init__(self, type: str, hidden_layers: int, hidden_nodes: int, 
                 eta: float, return_type: str, epochs: int):
        # Type of neural network. Classification or Regression
        self.type = type
        # Number of hidden layers
        self.hidden_layers = hidden_layers
        # List of nodes per hidden layer. For autoencoder, this is limited to a 1D
        # array of length 2 with the first index correlated to the encoding nodes
        # and the second correlating to the hidden layer nodes
        self.hidden_nodes = hidden_nodes
        # Learning rate. Used during back propagation
        self.eta = eta
        # Return type of the system. If set to 'loss', we return the loss
        # of all the epochs during training/testing. If set to 'model' we  
        # return the weights and biases of the network. 
        self.return_type = return_type
        # Number of epochs that we continue building off of
        self.epochs = epochs
        # Initialize starting data to null
        self.X = []
        # Initialize starting true labels to null
        self.y_true = []
        # Initialize hiden layer values to null
        self.hidden_layers_unactivated = []
        self.hidden_layers_activated = []
        # Initialize encoded value to null
        self.X_encoded = None
    
    """
    This method initializes the weights and biases of our system.
    """
    def initialize_w_b(self, X: List, labels_values: List) -> List:
        # Check if have hidden layers.
        # Initialize input layer node count
        current_layer_count = len(X[0])
        if self.hidden_layers != 0:
            # Set weights and bias with respect to hidden layers
            all_layer_weights = []
            all_bias = []
            # For each hidden layer, create weight matrix and bias element per node
            for h in range(self.hidden_layers):
                all_layer_weights.append(np.random.randn(self.hidden_nodes[h], current_layer_count) * 0.01)
                all_bias.append(np.zeros(self.hidden_nodes[h]))
                # Increment current layer
                current_layer_count = self.hidden_nodes[h]
            # Need to get weights and biases to final layer
        # Set the weights and bias depending on current type of nn
        if self.type == 'classification':
            # Weights will be dimensions of features(col) x class_label_values (row)
            weights = np.random.randn(len(labels_values), current_layer_count) * 0.01
            # Bias will be 1D list of length class_labels_values
            bias = np.zeros(len(labels_values))
        else:
            # Weights will be 1D list of length features 
            weights = np.random.randn(current_layer_count) * 0.01
            # Bias will be initialized to 0
            bias = 0
        # Return all the weights combined if hidden layers included
        if self.hidden_layers != 0:
            all_layer_weights.append(weights)
            all_bias.append(bias)
            return all_layer_weights, all_bias
        # Otherwise just return the weights and bias from input to output layer
        return weights, bias
    
    """
    This method creates a softmaxes matrix from the given Z values.
    """
    def softmax(self, Z: List) -> List:
        # Initialize softmax
        softmaxes = []
        # Calculate softmax
        for row in Z:
            # Create list for all exponential logits
            exp_row_Z = []
            for val in row: 
                exp_row_Z.append(np.exp(val))
            # Sum the exponential logits
            sum_exp_row_Z = np.sum(exp_row_Z)
            # Calculate the softmax at row
            softmax_at_row = []
            for val in range(len(row)): 
                softmax_at_row.append(exp_row_Z[val] / sum_exp_row_Z)
            # Append soft max at row to softmaxes
            softmaxes.append(softmax_at_row)
        return softmaxes

    """
    This method returns the sigmoid of any value based on the sigmoid equation.
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    """
    This method returns the sigmoid derivative of any value based on the derivative of
    the sigmoid function
    """
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    """
    This method calculates all example's probability of being the correct class label. 
    This is done by examining the softmaxes and appending the probabiltiy of the true class
    to a list and then returning the list. 
    """
    def true_class_prob_list(self, softmaxes: List, labels: List):
        # Return all example probabilities with true class label
        # Labels is numerical encoded
        true_probs = []
        for example in range(len(softmaxes)):
            # Get probability of true label at example
            true_label = labels[example]
            # Avoid possiblity of 0 probability by adding very small value
            # Error proof log calculation
            true_prob = softmaxes[example][true_label]
            if true_prob == 0: 
                true_prob = 1e-15
            true_probs.append(softmaxes[example][true_label])
        return true_probs

    """
    This method calculates the cross entropy of a list of probabilities.
    """
    def cross_entropy (self, true_probs: List) -> float:
        # Calculate the log of each probability
        negative_logged_p = np.log(true_probs)
        # Return the negative mean of all the logs
        return -mean(negative_logged_p)

    """
    This method returns the output loss between the predicted and true class labels. 
    This is done by subtracting 1 from all the correct class label probabilities within 
    the Z_gradient matrix
    """
    def output_loss(self, y_pred: List, y_true: List) -> List:
        Z_gradient_loss = deepcopy(y_pred)
        # Iterate through all probability distributions
        for example in range(len(Z_gradient_loss)):
            # Subtract 1 from correct prediction
            true_val = y_true[example]
            Z_gradient_loss[example][true_val] -= 1
        return Z_gradient_loss
    
    """
    This method scales the output loss of the prediction matrix by dividing each element
    with the number of examples.
    """
    def average_output_loss(self, y_pred: List, y_true: List) -> List:
        # Get gradients list
        gradient_loss = self.output_loss(y_pred, y_true)
        # Average gradients down
        # Divide each element with size of label
        divider = 1 / len(y_true)
        for row in range(len(gradient_loss)):
            # Multiply by inverse
            gradient_loss[row] = multiply_list_by_val(gradient_loss[row], divider)
        return gradient_loss
    
    """
    This method either returns the original Z list or a softmax matrix dependent on 
    if the neural network is regressing or classifying, respetivelyl
    """
    def get_pred(self, Z: List) -> List:
        if self.type == "classification":
            # Return softmax on classification
            return self.softmax(Z)
        # Return original on regression
        return Z

    """
    This method calculates the loss between the predicted and true class labels. 
    This is done by either MSE or mean cross entropy loss, dependent on the neural 
    network type.
    """
    def get_loss(self, y_pred: List, y_true: List) -> float:
        if self.type == "classification":
            # Select probability of true claass for each example
            true_class_probabilities = self.true_class_prob_list(y_pred, y_true)
            # Calculate mean cross entropy loss
            mean_cross_entropy_loss = self.cross_entropy(true_class_probabilities)
            return mean_cross_entropy_loss
        else:
            # Calculate MSE
            mse = get_mse_list(y_pred, y_true)
            # Return loss = 1/2n * mse
            return mse * (1/(2*len(y_true)))
    
    """
    This method sums up all of the error per class. 
    """
    def sum_error_per_class(self, errors: List) -> List:
        totals = []
        # Iterate through each class and append each class error into list
        for c in range(len(errors[0])):
            sum = 0
            # Iterate throguh each example to find total error per class
            for e in range(len(errors)):
                sum += errors[e][c]
            totals.append(sum)
        return np.array(totals)

    """
    This method conducts the entire backpropgation of the neural network. This is done by 
    identifying if we have any hidden layers. If we do not have any hidden layers, we can
    simply calculate the gradient loss at the output node and then return the weight and 
    bias gradients for each of the input nodes. 
    If we have hidden layers, we must calculate the propagated error between each of the 
    layers and calcualte the gradient at each layer. After calculating the gradient at 
    each layer, we calculate the weight gradient at each layer and store this value. Once 
    we get traverse back to the input layer, we return all of the weight and bias gradients.
    """
    def backpropagate(self, X: List, y_true: List, y_pred: List) -> List:
        example_count = len(X)
        # Check if we have hidden layers
        if self.hidden_layers == 0:
            if self.type == "classification":
                gradient_loss = self.average_output_loss(y_pred, y_true)
            else:
                # Gradient of MSE with respect to expected
                gradient_loss = multiply_list_by_val(subtract_lists(y_true, y_pred), (-2/len(y_true)))
            # Calculate weight gradient
            gradient_weight = np.dot(X.T, gradient_loss)
            # Calculate bias gradient
            gradient_bias = np.sum(gradient_loss, axis = 0)
            # Return two gradients
            return gradient_weight, gradient_bias
        # Initialize list for all weight_gradients and biases
        # They are ordered such at index 0 implies weights and bias gradient between output and final hidden
        all_weight_gradients = []
        all_bias_gradients = []
        # Go into error and gradients per layer
        for layer in range(self.hidden_layers + 1):
            # Check if we are at output layer
            if layer == 0:
                # Error at output
                # Calculate error and do shape correct
                if self.type=="regression":
                    predictions = self.hidden_layers_activated[self.hidden_layers]
                    error_output = predictions - y_true
                    # Shape correct
                    error_output = error_output[:, np.newaxis]
                else: 
                    error_output = self.output_loss(y_pred, y_true)
                    # Shape correct
                    error_output = np.array(error_output)
            else:
                # Error prop to Hidden Layer
                # Get original z (unactivated) values
                z = self.hidden_layers_unactivated[self.hidden_layers - layer]
                # Apply derivative of activation function
                sigmoid_der_z = self.sigmoid_derivative(z)
                error_output = np.dot(error_output, weight_gradient) * sigmoid_der_z.T
            # Gradient at Layer 
            # If we are at input layer, predictions set to X
            if layer == self.hidden_layers:
                # Transpose twice to get original data
                predictions = X.T
            else: 
                predictions = self.hidden_layers_activated[self.hidden_layers - layer - 1]
            # Calculate gradient
            gradient = np.dot(error_output.T, predictions.T)
            # Calculate weight gradient
            weight_gradient = 1/example_count * gradient
            # Calculate bias gradient
            # Dependent on neural network type
            if self.type == "regression":
                bias_gradient = 1/example_count * np.sum(weight_gradient)
            else:
                bias_gradient = 1/example_count * self.sum_error_per_class(error_output)
            # Append gradients
            all_weight_gradients.append(weight_gradient)
            all_bias_gradients.append(bias_gradient)
        # Return all weight and bias gradients lists
        return all_weight_gradients, all_bias_gradients

    """
    This method is responsible for calculating new weights. If we have no hidden layers, 
    the new weights are simply the difference between the current weights and the determined
    weight and bias gradients in product with the learning rate. 
    If we have layers, we must extract all of the weight and bias gradients from the parameters
    and go through each layer to update it's weights. 
    """
    def new_weights(self, weights: List, bias, all_weight_gradients: List, all_bias_gradients) -> List:
        # Check if we have hidden layers
        if self.hidden_layers == 0:
            weights -= self.eta * all_weight_gradients.T
            bias -= self.eta * all_bias_gradients
        # If we have hidden layers, we must update weights and bias through each layer
        # all_weight_gradients and all_bias_gradients are broken apart to give us true gradients per layer
        else:
            for i in range(len(weights)):
                # Get inverse index (ith element from end)
                j = len(weights) - i - 1
                # Reshape if shape is (n,)
                weights[i] = np.array(weights[i])
                if weights[i].shape == (weights[i].size,):
                    # Reshape
                    s = weights[i].size
                    weights[i] = weights[i].reshape(1, s)
                    # Perform calculation
                    weights[i] -= self.eta * all_weight_gradients[j]
                    # Revert shape
                    weights[i] = weights[i].reshape(s,)
                else:
                    weights[i] -= self.eta * all_weight_gradients[j]
                bias[i] -= self.eta * all_bias_gradients[j]
        # Return altered weights and bias
        return weights, bias
   
    """
    This method setups of the feedforward aspect of our dataset by proving us the final layer's 
    values, weights, and bias. If there are no hidden layers, this is simply the same as the 
    parameter values. If we have hidden layers, we need to conduct feedforward algorithm to get to 
    final hidden layer and store all weights and biases. All unactivated and activated values must
    be stored within the member variables. 
    """
    def activatated_values(self, X: List, weights: List, bias: List) -> List:
        # Check if we have hidden layers
        if self.hidden_layers == 0: 
            return X, weights, bias
        # Translate the inputs throguh hidden layers
        # Initialize layer_inputs
        current_layer_inputs = X.T
        # Look at each hidden layer at a time
        for layer in range(self.hidden_layers):
            # Get the weights
            layer_weights = weights[layer]
            layer_bias = bias[layer]
            # Correct bias shape
            layer_bias = layer_bias[:, np.newaxis]
            # Calculate unactivated hidden node values
            hidden_values = np.dot(layer_weights, current_layer_inputs) + layer_bias
            # Activate hidden node
            hidden_values_activated = self.sigmoid(hidden_values)
            # Use these activated values as inputs
            current_layer_inputs = hidden_values_activated
            # Add to object's member variables
            self.hidden_layers_activated.append(hidden_values_activated)
            self.hidden_layers_unactivated.append(hidden_values)
        # Return final layer activated values transposed
        # Return the last weights and bias layer
        return current_layer_inputs.T, weights[len(weights) - 1], bias[len(bias) - 1]

    """
    This is the entry point into the construction of any neural network. After the parameters are 
    defined, this method will train any neural network and return either the loss or the weights/bias
    of the network. THis is done by first running a feed forward run through the model with all layers
    in place. Once the feed forward run is complete, we will calculate the loss that our model current 
    produces with the respective loss function. After determining this loss value, we calculate the 
    error value and use them for our backpropagation stage. During this stage, we determine the weight
    gradients and bias gradients that will be required to alter the current weights and biases of the
    network. Finally, we alter the weights and biases of the model and rerun the entire dataset again
    as a second epoch. We keep running until we hit a certain number of epochs with the goal of 
    minimizing loss every epoch. 
    """
    def build_nn(self, features: List, labels: Feature):
        # Build an appropriate neural network for the dataset
        y_true = labels.get_data()
        # Initialize weights and bias
        weights = None
        bias = None
        # Initialize losses list
        losses = []
        for _ in range(self.epochs):
            # Create X matrix to hold all examples with relevant features
            # Exclude_features removes ID and name features
            # Features (cols) x Examples (rows)
            X = np.array(create_X_matrix(features, exclude_features(features)))
            # Initialize weights
            if weights is None:
                # Initialize weights and bias respective of NN type and hidden layers
                weights, bias = self.initialize_w_b(X, labels.get_values())
            # Find input values after hidden layers if applicable
            inputs, final_weights, final_bias = self.activatated_values(X, weights, bias)
            # Conduct prediction
            Z = np.dot(inputs, final_weights.T) + final_bias
            # Add this value into both activated and unactivated hidden values 
            self.hidden_layers_activated.append(Z)
            self.hidden_layers_unactivated.append(Z)
            # Get predictions
            y_pred = self.get_pred(Z)
            # Get loss
            losses.append(self.get_loss(y_pred, y_true))
            # Back propagate
            all_weight_gradients, all_bias_gradients = self.backpropagate(X, y_true, y_pred)
            # Set new weights
            weights, bias = self.new_weights(weights, bias, all_weight_gradients, all_bias_gradients)
        if self.return_type == 'loss':
            return losses
        elif self.return_type == 'model':
            return weights, bias 
    
    """
    This method tests a built neural network by extracing the weights and biases of the model and then 
    running a feed forward approach through the layers to determine the predicted values. The loss
    is computed and reported back. There is no backpropagation as we are not altering any of the weights.
    """
    def test_nn(self, features: List, labels: Feature, model: List) -> List:
        # Return the loss values without any alteration to model weights
        weights, bias = model
        y_true = labels.get_data()
        X = np.array(create_X_matrix(features, exclude_features(features)))
        # Find input values after hidden layers if appliable
        inputs, final_weights, final_bias = self.activatated_values(X, weights, bias)
        # Predict
        Z = np.dot(inputs, final_weights.T) + final_bias
        y_pred = self.get_pred(Z)
        loss = self.get_loss(y_pred, y_true)
        # Return the loss
        return loss
    
    """
    This method is utilizing during creation of an autoencoded neural network. This is done by linking
    the input layer to an encoded layer and performing a FFBP algorithm over a set number of epochs. 
    The loss function is reserved to simply an MSE value. Once we iterate through all of the epochs, 
    we save the encoded values into memory and return the losses per epoch. 
    """
    def train_autoencoder(self, features: List) -> List:
        # Train the autoencoder based on dataset
        X = np.array(create_X_matrix(features, exclude_features(features)))
        # Initialize dimensions
        input_dimensions = X.shape[1]
        # Initialize weights
        encoded_weights = np.random.randn(input_dimensions, self.hidden_nodes[0]) * 0.01
        decoded_weights = np.random.randn(self.hidden_nodes[0], input_dimensions) * 0.01
        # Initialize bias
        encoded_bias = np.zeros(self.hidden_nodes[0])
        decoded_bias = np.zeros(input_dimensions)
        losses = []
        # Begin training autoencoder
        for _ in range(self.epochs):
            # Forward pass
            # Get unactivated Z for encoded layer
            z_encoded = np.dot(X, encoded_weights) + encoded_bias
            # Activate Z encoded
            encoded = self.sigmoid(z_encoded)
            # Get unactived Z for decoded layer
            z_decoded = np.dot(encoded, decoded_weights) + decoded_bias
            # Activate Z decoded
            decoded = self.sigmoid(z_decoded)
            # Compute loss and gradients
            losses.append(np.mean(np.square(X, decoded)))

            # Back propagation
            decoded_error = decoded - X
            decoded_gradient = np.dot(encoded.T, decoded_error)
            decoded_bias_gradient = np.sum(decoded_error, axis = 0)
            encoded_error = np.dot(decoded_error, decoded_weights.T) * self.sigmoid_derivative(encoded)
            encoded_gradient = np.dot(X.T, encoded_error)
            encoded_bias_gradient = np.sum(encoded_error, axis = 0)
            
            # Update weights
            encoded_weights -= self.eta * encoded_gradient
            decoded_weights -= self.eta * decoded_gradient
            # Update biases
            encoded_bias -= self.eta * encoded_bias_gradient
            decoded_bias -= self.eta * decoded_bias_gradient
        # Return encoded representation
        self.X_encoded = encoded
        return losses

    """
    After the training of the autoencoder, we append a hidden layer and output layer to the encoding
    layer of the autoencoder with the hopes of minimizing error over some epochs. A standard FFBP algorithm
    is applied with the initial weights between the input and the encoded layer locked. Once we iterate
    through all epochs, we return the weights and biases for testing.
    """
    def train_from_encoded(self, labels) -> List:
        # Encoded layer -> Hidden -> Output
        # Get input dimensions
        input_dim = self.X_encoded.shape[1]
        # Get output dimensions
        if self.type == "classification":
            output_dim = len(labels.get_values())
        else:
            output_dim = 1
        y_true = labels.get_data()
        # Initialize weights
        hidden_weights = np.random.randn(input_dim, self.hidden_nodes[1])
        output_weights = np.random.randn(self.hidden_nodes[1], output_dim)
        # Initialize biases
        hidden_bias = np.zeros(self.hidden_nodes[1])
        output_bias = np.zeros(output_dim)
        losses = []
        for _ in range(self.epochs):
            # Forward pass 
            # Get unactivated Z for hidden layer
            z_hidden = np.dot(self.X_encoded, hidden_weights) + hidden_bias
            # Activate Z Hidden
            hidden_activated = self.sigmoid(z_hidden)
            # Get unactivated Z for output layer
            z_output = np.dot(hidden_activated, output_weights) + output_bias
            # Activate Z Output
            output_activated = self.sigmoid(z_output)
            # Compute loss
            if self.type == "classification":
                # Cross entropy loss
                example_count = output_activated.shape[0]
                correct_class_probs = output_activated[np.arange(example_count), y_true]
                losses.append(-np.mean(np.log(correct_class_probs + 1e-15)))
            else:
                # MSE loss
                losses.append(np.mean(np.square(y_true - output_activated)))
            # Backpropagation
            if self.type == "classification":
                output_error = output_activated
                # Subtract 1 from true class probabilities
                output_error[np.arange(len(y_true)), y_true] -= 1
            else:
                y_true = np.array(y_true)
                output_error = output_activated - y_true.reshape(-1, 1)
            output_gradient = np.dot(hidden_activated.T, output_error)
            output_bias_gradient = np.sum(output_error, axis = 0)
            hidden_error = np.dot(output_error, output_weights.T) * self.sigmoid_derivative(hidden_activated)
            hidden_gradient = np.dot(self.X_encoded.T, hidden_error)
            hidden_bias_gradient = np.sum(hidden_error, axis = 0)
            # Update weights
            hidden_weights -= self.eta * hidden_gradient
            output_weights -= self.eta * output_gradient
            # Update biases
            hidden_bias -= self.eta * hidden_bias_gradient
            output_bias -= self.eta * output_bias_gradient
        # Return the weights to reproduce this model       
        return hidden_weights, output_weights, hidden_bias, output_bias
    
    """
    This method tests an autoencoder after it has finished encoding and training this is done similar to 
    the test_nn method where we pass in all of the weight and bias metrics and test our model via
    FFBP algorithm without backpropagation. After testing all labels, we return the loss
    """
    def test_autoencoded(self, labels: Feature, model: List) -> List:
        # Test the autoencoded NN 
        # Extract all values
        hidden_weights, output_weights, hidden_bias, output_bias = model
        y_true = labels.get_data()
        # Get unactivated Z for hidden layer
        z_hidden = np.dot(self.X_encoded, hidden_weights)
        # Activate Z Hidden
        hidden_activated = self.sigmoid(z_hidden)
        # Get unactivated Z for output layer
        z_output = np.dot(hidden_activated, output_weights) + output_bias
        # Activate Z Output
        output_activated = self.sigmoid(z_output)
        # Compute loss
        if self.type == "classification":
            # Cross entropy loss
            example_count = output_activated.shape[0]
            correct_class_probs = output_activated[np.arange(example_count), y_true]
            loss = (-np.mean(np.log(correct_class_probs + 1e-15)))
        else:
            # MSE loss
            loss = (np.mean(np.square(y_true - output_activated)))
        return loss
    
    """
    This method trains the autoencoder and trains it once the encoding is complete. 
    """
    def fit_autoencoder(self, features: List, labels: Feature) -> List:
        # Train autoencoder
        autoencoder_losses = self.train_autoencoder(features)
        # Use encoded values to get output weights
        encoded_weights = self.train_from_encoded(labels)
        if self.return_type == "loss":
            return autoencoder_losses
        else: 
            # Return all weights
            return encoded_weights

    

