from typing import List, Dict
import math
import numpy as np

def mode(l: List) -> float:
    occurrences = {}
    for value in l:
        if value in occurrences:
            occurrences[value] += 1
        else:
            occurrences[value] = 1
    return max(occurrences, key=occurrences.get)

def mean(l: List) -> float:
    return sum(l) / len(l)

def median (l: List) -> float:
    l.sort()
    mid_index = math.floor(len(l) / 2)
    if len(l) % 2 == 1:
        return l[mid_index]
    else:
        return (l[mid_index] + l[mid_index - 1]) / 2
    
def get_kmode(knn: List) -> float:
    # Separate labels from rest of data
    labels = []
    for neighbor in knn:
        labels.append(neighbor[4])
    return mode(labels)

def get_dataset_mode(dataset: Dict, col: int):
    specified_col = dataset['features'][col].get_data()
    return mode(specified_col)

def get_dataset_median(dataset: Dict, col: int):
    specified_col = dataset['features'][col].get_data()
    sorted_list = conv_and_sort(specified_col)
    return str(median(sorted_list))

def conv_and_sort(l: List):
    sorted_list = []
    for i in range(len(l)):
        try:
            sorted_list.append(float(l[i]))
        except ValueError: 
            continue
    sorted_list.sort()
    return sorted_list

def calculate_mean_std_dev(l: List):
    return np.mean(l), np.std(l)

def normalize(dataset: Dict):
    # Normalize all continueous and discrete columns via z-score standardization
    for i in range(len(dataset['features'])):
        feature = dataset['features'][i]
        if (feature.get_type() == 'continuous' or 
            feature.get_type() == 'discrete'):
            mean, std_dev = calculate_mean_std_dev(feature.get_data())
            z_scores = [(x - mean) / std_dev for x in feature.get_data()]
            dataset['features'][i].assign_data(z_scores)

def get_mse (expected: float, actual: float) -> float:
    return math.pow(expected - actual, 2)

def multiply_list_by_val(l: List, val: float) -> List:
    new_l = []
    for i in l:
        new_l.append(val * i)
    return new_l

def subtract_lists(l1: List, l2: List) -> List:
    if len(l1) != len(l2): return None
    sub_l = []
    for i in range(len(l1)):
        sub_l.append(l1[i] - l2[i])
    return sub_l

    