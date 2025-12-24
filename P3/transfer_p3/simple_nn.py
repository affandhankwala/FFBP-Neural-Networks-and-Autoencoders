from calculations import get_mse, multiply_list_by_val, subtract_lists, mean
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from copy import deepcopy


def calculate_z(input_values: List, weights: List, bias: float) -> float:
    # Return the dot product of input values and weights plus the bias term
    return np.dot(input_values, weights) + bias

def calculate_weight_gradient(input_values: List, gradient: float) -> List:
    # Calculate the weight gradients for new weights
    weight_gradients = []
    for value in range(len(input_values)):
        # dL_dwi = dL_dy * xi
        weight_gradients.append(gradient * input_values[value])
    return weight_gradients


def calculate_bias_gradient(value_count: float, predicted: float, expected: float) -> float:
    return 2 / value_count * (predicted - expected)

def create_X_matrix (features: List) -> List:
    # Create matrix to hold all feature data
    feature_length = len(features)
    example_length = len(features[0].get_data())
    # Initialize arrays
    X =[]
    for example in range(example_length):
        X_example = []
        for feature in range(feature_length):
            X_example.append(features[feature].get_data()[example])
        X.append(X_example)
    return X

def calculate_softmax(Z: List) -> List:
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

def get_true_class_prob_list(softmaxes: List, labels: List):
    # Return all example probabilities with true class label
    # Labels is numerical encoded
    true_probs = []
    for example in range(len(softmaxes)):
        # Get probability of true label at example
        true_label = labels[example]
        # Avoid possiblity of 0 probability
        # Error proof log calculation
        true_prob = softmaxes[example][true_label]
        if true_prob == 0: 
            true_prob = 1e-15
        true_probs.append(softmaxes[example][true_label])
    return true_probs

def get_cross_entropy (true_probs: List):
    # Calculate the -log of each probability
    # Add 1e-15 to prevent log(0)
    negative_logged_p = np.log(true_probs)
    # Return the mean
    return -mean(negative_logged_p)

def calculate_gradient_Z(y_pred: List, y_true: List) -> List:
    Z_gradient_loss = deepcopy(y_pred)
    # Iterate through all probability distributions
    for example in range(len(Z_gradient_loss)):
        # Subtract 1 from correct prediction
        true_val = y_true[example]
        Z_gradient_loss[example][true_val] -= 1
    # Average gradients down 
    # Divide each element with size of label values
    divider = 1 / len(Z_gradient_loss[0])
    for row in range(len(Z_gradient_loss)):
        # Multiply by inverse 
        Z_gradient_loss[row] = multiply_list_by_val(Z_gradient_loss[row], divider)
    return Z_gradient_loss

def build_classifiction_nn(dataset: Dict, eta: float):
    # Implement logistic regression 
    # Number of features
    feature_count = len(dataset['features'])
    # Number of examples
    example_length = len(dataset['features'][0].get_data())
    labels = dataset['label']
    y_true = labels.get_data()
    # Initialize z array
    z_logits = []
    # Initialize weights and bias
    weights = []
    biases = []
    # Initialize gradients
    total_gradients = 0
    mean_gradients = []
    # Repeat over some epochs
    epochs = 50
    for epoch in range(epochs):
        # Check if weights are null
        if len(weights) < 1:
            # Initialize to random elements of size features x label values
            weights = np.random.randn(feature_count, len(labels.get_values())) * 0.01
        # Check if biases are null
        if len(biases) < 1:
            # Initialize to 0 elements of size labels
            biases = np.zeros(len(labels.get_values())) 
        # Create X matrix of size examples x features
        X = np.array(create_X_matrix(dataset['features']))
        # Calculate Z logits
        Z = np.dot(X, weights) + biases
        # Calculate softmaxes
        y_pred = calculate_softmax(Z)
        # Calculate cross-entropy loss
        # Select probability of true class for each example
        true_class_probabilities = get_true_class_prob_list(y_pred, y_true)
        # Calculate mean cross entropy loss
        mean_cross_entropy_loss = get_cross_entropy(true_class_probabilities)
        # Back propagate
        # gradient_loss_Z = y_pred
        # gradient_loss_Z[np.arange(len(labels.get_data())), y_true]
        # gradient_loss_Z /= len(labels.get_data())
        gradient_loss = calculate_gradient_Z(y_pred, y_true)
        # Calculate weight gradient
        gradient_weight = np.dot(X.T, gradient_loss)   # THIS RETURNS 2 X 16 ARRAY
        # Calculate bias gradient
        gradient_bias = np.sum(gradient_loss, axis = 0)
        # New weights
        # TODO: VERIFY THIS WORKS
        weights -= eta * gradient_weight
        # New bias
        biases -= eta * gradient_bias
        print(f"{epoch} | {mean_cross_entropy_loss}")

    return
# TODO: SEPARATE THE TUNING OF EPOCH FROM REGRESSION NN BUILDING
def build_regression_nn(dataset: Dict, eta: float):
    # Implement linear regression without hidden layer
    # Number of features
    feature_count = len(dataset['features'])
    # Number of examples
    example_length = len(dataset['features'][0].get_data())
    # Initialize weights and bias
    weights = []
    bias = None
    # Repeat over some epochs
    total_gradients = 0
    mean_gradients = []
    epochs = 15
    for _ in range(epochs):
        for value_iterator in range(example_length):
            # Forward propagation
            # Extract feature values of current input feature
            input_values = []
            for feature_iterator in range(feature_count):
                feature_type = dataset['features'][feature_iterator].get_type()
                # Only take nominal, discrete, and continuous values
                if (feature_type == 'nominal' or 
                    feature_type == 'continous' or
                    feature_type == 'discrete'):
                    input_values.append(dataset['features'][feature_iterator].get_data()[value_iterator])
            # Set weights and bias if they are not done so
            if len(weights) == 0:
                weights = [1 for i in range(len(input_values))]
                bias = 1
            predicted = calculate_z(input_values, weights, bias)
            expected = dataset['label'].get_data()[value_iterator]
            # Calculate loss function as derivative of MSE
            gradient = predicted - expected
            # Calculate weight gradients
            weight_gradients = calculate_weight_gradient(input_values, gradient)
            # Add in the 1/n part of MSE calculation
            weight_gradients = multiply_list_by_val(weight_gradients, 2 / example_length)
            # Calculate new weights
            for w in range(len(weights)):
                weights[w] -= eta * weight_gradients[w]
            # Calculate new bias
            # Bias gradient = gradient
            bias -= eta * gradient
            # Total all gradients to determine average gradient per epoch
            total_gradients += gradient
        # Average the gradients 
        mean_gradients.append(total_gradients / example_length)
    # Print out the epoch with the smallest gradient
    best_epoch = np.argmin(np.abs(mean_gradients))
    print(f"Smallest Gradient at epoch: {best_epoch}")
    # Plot the gradient per epoch
    x_values = list(range(0, epochs))
    plt.plot(x_values, mean_gradients)
    plt.title("Gradient per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient")
    plt.show()
    return

def build_simple_nn (dataset: Dict, eta: float):
    # Determine how many input nodes
    # Number of inputs = number of features
    input_count = len(dataset['features'])
    # Determine which type of nn will be constructed
    class_label_type = dataset['label'].get_type()
    if class_label_type == 'nominal':
        # Classifiction problem
        build_classifiction_nn(dataset, eta)
    else:
        # Regression problem
        build_regression_nn(dataset, eta)
