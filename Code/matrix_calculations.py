from typing import List

"""
This file is reserved for any method that performs calculations on matrices. 
"""

"""
This method is responsible for creating an X matrix based from the useful features
of the dataset. A feature is deemed useful if it is not ID or name type. The 
feaatures that are useless are skipped. 
"""
def create_X_matrix (features: List, exclude_features: List) -> List:
    # features is a list of all the features 
    # exclude_features is a list of indexes pointing to all the features we should skip
    # Create matrix to hold all feature data
    feature_length = len(features)
    # Store example length
    example_length = len(features[0].get_data())
    # Initialize arrays
    X =[]
    for example in range(example_length):
        X_example = []
        for feature in range(feature_length):
            # Skip all features in exclude list
            if feature in exclude_features: continue
            X_example.append(features[feature].get_data()[example])
        X.append(X_example)
    return X