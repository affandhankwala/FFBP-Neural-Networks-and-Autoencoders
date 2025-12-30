from dataset_manipulation import shuffle, split_dataset_into_n, split_dataset
from nn import Neural_Network
from typing import Dict, List

"""
This file is responsible for cross validating the three neural networks. This is done by first
shuffling the dataset. Once the dataset is shuffled, we split it among 5 equal splits consisting
of X, y pairs. Within each of these pairs, we split again at an 80/20 train/test proportion.
We then train the simple NN on the training set and test it via the test dataset and record
its loss values. Similarly we train the hidden layer NN on the train dataset and test on the 
test dataset and store its losses as well. We conduct the same on the autoencoded NN. After we
Iterate through all of these networks, we apppend all three model's error to saved lists. 
Then we train/test all three models on another X, y pair until all 5 X, y pairs have been utilized
on the neural networks. At this point present the loss metrics. 
"""
# Cross validate all three networks
def cross_validate(dataset: Dict, tuned_values: List) -> None:
    # dataset is a dictionary consisting of the 'features' and 'label' keys. 
    # tuned_values is a list of all the hyperparameters that were determined at an earlier stage
    # Unpack tuned values
    class_label_type, epochs, eta, hidden_nodes, encoded_nodes = tuned_values
    # Shuffle dataset
    shuffle(dataset['features'], dataset['label'])
    # Split data into 5 training and validation sections
    X_y_pairs = split_dataset_into_n(dataset, 5)
    # Setup losses lists
    simple_nn_losses = []
    hidden_nn_losses = []
    autoencoded_nn_losses = []
    # Iterate through each dataset pair
    for pair in X_y_pairs:
        X, y = pair
        # Train Each neural network twice
        # Shuffle/Split each dataset into train and test
        shuffle(X, y)
        X_train, X_test, y_train, y_test = split_dataset(dataset, 0.8)

        # Simple Neural Network
        simple_nn = Neural_Network(class_label_type, 0, [], eta, 'model', epochs)
        model = simple_nn.build_nn(X_train, y_train)
        # Test Simple NN
        simple_nn_losses.append(round(simple_nn.test_nn(X_test, y_test, model), 4))

        # Traditional Neural Network with 2 hidden layers
        hidden_nn = Neural_Network(class_label_type, 2, hidden_nodes, eta, 'model', epochs)
        hidden_model = hidden_nn.build_nn(X_train, y_train)
        # Test Hidden Layer NN
        hidden_nn_losses.append(round(hidden_nn.test_nn(X_test, y_test, hidden_model), 4))

        # Neural Network with auto encoder
        autoencoded_nn = Neural_Network(class_label_type, 0, encoded_nodes, eta, 'model', epochs)
        model = autoencoded_nn.fit_autoencoder(X, y)
        # Test Autoencoded NN
        autoencoded_nn_losses.append(round(autoencoded_nn.test_autoencoded(y, model), 4))
    
    # Report findings
    print("Finished cross validate with below losses: ")
    print(f"Simple NN: {simple_nn_losses}")
    print(f"Hidden layer NN: {hidden_nn_losses}")
    print(f"Autoencoder NN autoencoded layer losses: {autoencoded_nn_losses}")

