from typing import List, Dict
import math
import numpy as np
"""
This file is responsible for the calculations of certain metrics. 
It is meant to be a hub to store many calculation methods
"""

"""
This method will calculate the mode of any list. This is done by 
mapping out the list into a map with each element's occurrence stored
as the value and then returning the key with the highest value
"""
def mode(l: List) -> float:
    occurrences = {}
    for value in l:
        # Create Dictionary to hold all occurrences per key
        if value in occurrences:
            occurrences[value] += 1
        else:
            occurrences[value] = 1
    # Return key with the highest values
    return max(occurrences, key=occurrences.get)

"""
This method calculates the mean of a list. This is done by dividing
the sum of all elements of th list with the length of the list
"""
def mean(l: List) -> float:
    return sum(l) / len(l)

"""
This method calcultes the median of a list. This is done by first ordering
all the elements in ascending order and then finding the middle element. 
"""
def median (l: List) -> float:
    l.sort()
    mid_index = math.floor(len(l) / 2)
    if len(l) % 2 == 1:
        return l[mid_index]
    else:
        # If the list is an even length, we return the average between the 
        # two middle elements
        return (l[mid_index] + l[mid_index - 1]) / 2

"""
This method returns the mode of any feature among a dataset. 
"""
def get_dataset_mode(dataset: Dict, col: int):
    specified_feature = dataset['features'][col].get_data()
    return mode(specified_feature)

"""
This method returns the median of any feature among a dataset.
"""
def get_dataset_median(dataset: Dict, col: int):
    specified_feature = dataset['features'][col].get_data()
    sorted_list = conv_and_sort(specified_feature)
    return str(median(sorted_list))

"""
This method converts all float values that are input as strings into floats
and then returns a sorted list of all the floats
"""
def conv_and_sort(l: List):
    sorted_list = []
    for i in range(len(l)):
        try:
            sorted_list.append(float(l[i]))
        except ValueError: 
            continue
    sorted_list.sort()
    return sorted_list

"""
This method calcualtes the mean and standard deviation of the passed in list 
via np commands
"""
def calculate_mean_std_dev(l: List):
    return np.mean(l), np.std(l)

"""
This method normalizes all the continuous and discrete values wtihin a dataset. 
This is done through z-score standardization.
"""
def normalize(dataset: Dict):
    # Normalize all continueous and discrete columns via z-score standardization
    for i in range(len(dataset['features'])):
        feature = dataset['features'][i]
        if (feature.get_type() == 'continuous' or 
            feature.get_type() == 'discrete'):
            # Calculate the mean and std_dev from helper function
            mean, std_dev = calculate_mean_std_dev(feature.get_data())
            # Scale down all values with determined mean and std_dev
            z_scores = [(x - mean) / std_dev for x in feature.get_data()]
            # Replace original data with new scaled data
            dataset['features'][i].assign_data(z_scores)

"""
This method returns the mean squared error between two values. The equation is
(predicted - actual) ^ 2.
"""
def get_mse (predicted: float, actual: float) -> float:
    return math.pow(predicted - actual, 2)


"""
This method calculates total mean squared error between two lists. This is the sum
of all the mean squared errors between each of the list elements. The sum is 
averaged down by the length of the input list. 
"""
def get_mse_list(predicted: List, actual: List) -> float:
    if len(predicted) != len(actual): return None
    se = 0
    for i in range(len(predicted)):
        # Sum all mse values
        se += get_mse(predicted[i], actual[i])
    # Scale down the sum
    return se / len(predicted)

"""
This method multiplies any list with a value. Useful when attempting to multiply
constant with a standard array.
"""
def multiply_list_by_val(l: List, val: float) -> List:
    new_l = []
    for i in l:
        # Iterate through each element and multiply in
        new_l.append(val * i)
    return new_l

"""
This method subtracts two identically sized lists. This is done in an iterative
fashion where each element in each list is subtracted from one another one at a time.
"""
def subtract_lists(l1: List, l2: List) -> List:
    # If mismatch sizes, we cannot subtract
    if len(l1) != len(l2): return None
    sub_l = []
    # Iterate through all elements and subtract them
    for i in range(len(l1)):
        sub_l.append(l1[i] - l2[i])
    return sub_l

    