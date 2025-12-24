from feature import Feature
from typing import List, Dict
import random
import math

"""
This file is responsible for all the calculations related to any manipulations to dataset. 
This includes splitting and shuffling. It also includes removal or exclusion of certain elements
"""

"""
exclude_features is given a list of features and it will return another list of features 
that contains any ID or name type features. This is critical for the building and testing
of the neural networks as ID or name features have no reason to be included within the them. 
After this method points out the indexes of the features to exclude, these indexes are 
handled elsewhere.
"""
def exclude_features(features: List) -> List:
    # Return a list of all features that will be excluded
    # ID and name will be excluded
    excluded = []
    for f in range(len(features)):
        if (features[f].get_type() == 'ID' or
            features[f].get_type() == 'name'):
            excluded.append(f)
    return excluded

""" 
This method shuffles a list of features in random order but makes sure to maintain the 
index integrity between the features with themselves and the labels. This is done by iterating
through each element of each feature and translating it into a 2D array. The label is 
likewise added to this array. This array is shuffled so that all elements are in random order
but this preserves the list index integrity. The data is then finally reassigned. 
"""
def shuffle(X: List, y: Feature) -> None:
    # Initialize 2D array
    observations = []
    # Iterate over every example within each feature
    for element in range(len(X[0].get_data())):
        # Initialize inner 1D array
        ob = []
        # Iterate over every feature of an example. To allow for proper index matching
        for feature in range(len(X)):
            ob.append(X[feature].get_data()[element])
        # Append label data
        ob.append(y.get_data()[element])
        observations.append(ob)
    # Shuffle 2D array
    random.shuffle(observations)
    # Re-assign data
    for j in range(len(observations[0]) - 1):
        # Temporary list to hold data of each feature
        feature_vals = []
        for i in range(len(observations)):
            feature_vals.append(observations[i][j])
        # Reassign each feature data
        X[j].assign_data(feature_vals)
    # Re-assign labels
    label_vals = []
    for label in range(len(observations)):
        label_vals.append(observations[label][len(observations[label]) - 1])
    y.assign_data(label_vals)
    # No need to return as we are modifying in memory

"""
This method returns a list of all the feature values at any example. Since our data structure
dictionary is structured so that we can easily pull any example's features by grabbing all 
elements at an index, this method simply goes in and creates a list of all the feature values
at the ith index. 
"""
def pull_example_at_i (features: List, i: int) -> List:
    # Return all feature values of example at i
    example = []
    for f in range(len(features)):
        example.append(features[f].get_data()[i])
    return example

""" This method returns a deepcopy of a parameter feature """
def copy_feature(feature: Feature, data: List) -> Feature:
    # Create new Feature object initialized at values of previous feature
    f = Feature(feature.get_name(), feature.get_type(), feature.get_values())
    # Transfer data
    f.assign_data(data)
    return f

"""
Upon given a list of examples, the list of labels, the original list of features and 
the original labels, this method is capable of recreating a new features list and feature
value with the passed in example values and labels. 
"""
def convert_examples_list_into_feature(examples_list: List, l: List, X: List, y: List) -> List:
    # examples_list is a 2D list of all the examples with their respective features
    # l is the labels for the examples
    # X is the original list of features that will be referenced for proper feature creation
    # y is the original label Feature that will be referenced for proper label feature creation
    # Initialize a pointer into the list of features
    X_feature_iterator = 0
    # Initialize the 2D list representing the new X dataset
    new_X = []
    # Initialize a point into the example list
    feature = 0
    # Iterate through through each feature within the nested list within the examples 
    while feature < len(examples_list[0]):
        # Save the current feature
        current_feature = X[X_feature_iterator]
        # Create a new feature with arbitrary starting data
        new_feature = copy_feature(current_feature, [])
        # Initialize 1D list to hold all data
        values = []
        # Iterate through each example of examples_list and pull the respective data at feature index
        for example in range(len(examples_list)):
            values.append(examples_list[example][feature])
        # Assign newly created feature our data
        new_feature.assign_data(values)
        # Add Feature into list of features
        new_X.append(new_feature)
        feature += 1
        X_feature_iterator += 1
    # Handle the labels and initialize 1D list to hold label values
    labels = []
    # Iterate through all labels and pull data
    for i in range(len(l)):
        labels.append(l[i])
    # Initialize new feature with newly labels data
    new_y = copy_feature(y, labels)
    # Return the list of features as X and label as y
    return new_X, new_y

"""
This method splits any dataset into n equal size X, y pairs. This is done by first 
setting up a 2D list of all the examples per each split. Once the indexes of the 
examples are stored, we call the pull_example_at_i for each example and generate a 
2D list of example values per split. This is passed into the convert_examples_list_into_feature
method which converts the split into an X, y pair. Over the course of n splits, we 
generate n X, y pairs that are returned.
"""
def split_dataset_into_n (dataset: Dict, n: int) -> List:
    features = dataset['features']
    label = dataset['label']
    example_count = len(label.get_data())
    # Initialize all pairs
    X_y_pairs = []
    # Initialize index-storers
    index_storers = []
    # Index_storers will store all lists containing example indexes of each split
    # Iterate over all starting indexes 
    for i in range(n):
        # Skip by 'n' over all examples and append to list
        indexes_of_split = [k for k in range(i, example_count, n)]
        index_storers.append(indexes_of_split)
    # Iterate through each index list
    for index_list in index_storers:
        # Iterate through each index and pull all feature values at index
        examples = []
        labels = []
        for index in index_list:
            # Get example at index
            example = pull_example_at_i(features, index)
            # Get label at index
            l = label.get_data()[index]
            examples.append(example)
            labels.append(l)
        # Convert examples into X, y
        X, y = convert_examples_list_into_feature(examples, labels, features, label)
        # Add these datasets to stored pairs
        X_y_pairs.append([X, y])
    # Return all pairs
    return X_y_pairs
    
"""
This method is takes in a dataset with list of features and labels as keys and a split 
proportion float. This split proportion indicates the split ratio between the first and second
splits. The split method is called for the splitting
"""
def split_dataset (dataset: Dict, split_proportion: float) -> List:
    # Initialize all new datasets
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    # X will be splits from the feature key within the dataset
    X_train, X_test = split(dataset['features'], split_proportion)
    # y will be splits from the label key within the dataset
    # We pass in the labels as a list because the split method only takes lists as parameter
    y_train, y_test = split([dataset['label']], split_proportion)  
    # Undo the list-encapsulating of the label splits
    y_train = y_train[0]
    y_test = y_test[0]
    return X_train, X_test, y_train, y_test

"""
This method splits a List of features at a specific split proportion and returns a List
of the two splits. This is done by first extracting the data from each feature into a two temporary
arrays that are split at the split index. These arrays then hold the data for the new features and 
copy_feature is called with these new lists to create identical features to the parameter feature
but with different data values.
""" 
def split(l: List[Feature], split_proportion: float) -> List:
    # Determine the split_index by multiplying the data length with the proportion float
    split_index = math.floor(len(l[0].get_data()) * split_proportion)
    # Initialize 2D arrays to first and second splits
    first = []
    second = []
    # Iterate through each feature in the original features list
    for f in range(len(l)):
        # Initialize temporary 1D arrays to hold split data
        temp1 = []
        temp2 = []
        # Store feature for copying later
        feature = l[f]
        # Iterate through data and apppend to respective temporary array depending on index
        for i in range(len(feature.get_data())):
            # Split at split_index such that temp1 holds first portion and temp2 holds second
            if i < split_index:
                temp1.append(feature.get_data()[i])
            else:
                temp2.append(feature.get_data()[i])
        # Create two new features with the saved feature values and new data
        first_f = copy_feature(feature, temp1)
        second_f = copy_feature(feature, temp2)
        # Append both features to list of features
        first.append(first_f)
        second.append(second_f)
    # First is a list of features with data from 0:split_index while second is a list
    # of features with data from split_index:len
    return first, second
