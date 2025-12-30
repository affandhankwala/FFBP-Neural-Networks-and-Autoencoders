from typing import List, Dict
from calculations import get_dataset_mode, get_dataset_median, normalize

"""
This method removes all null values within a Feature with the appropriate replacement. 
When working with nominal features, we replace the null value with the node. 
When working with discrete/continuous values, we replace the null value with the median.
"""
def remove_null(dataset: Dict):
    # Replace ? values with mode
    for i in range(len(dataset['features'])):
        for j in range(len(dataset['features'][i].get_data())):
            if dataset['features'][i].get_data()[j] == "?":
                # Replace with mode if nominal
                if dataset["features"][i].get_type() == 'nominal':
                    dataset['features'][i].alter_data(j, get_dataset_mode(dataset, i))
                # Otherwise replace with median
                else:
                    dataset['features'][i].alter_data(j, get_dataset_median(dataset, i))

"""
This method is a safeguard against any floats that are accidentally converted into strings.
"""
def convert_data(dataset: Dict):
    # Convert all strings that are floats into floats
    for i in range(len(dataset['features'])):
        if (dataset['features'][i].get_type() == 'continuous' or 
            dataset['features'][i].get_type() == 'discrete'):
            for j in range(len(dataset['features'][i].get_data())):
                dataset['features'][i].alter_data(j, float(dataset['features'][i].get_data()[j]))

"""
This method converts all nominal data and labels into a numerically encoded counterpart. This
is done by replacing all data values with the index of said value within their values list.
"""
def integer_converted (dataset: Dict):
    # Separate nominal data into unique columns with each nominal selection
    for i in range(len(dataset['features'])):
        feature = dataset['features'][i]
        if feature.get_type() == 'nominal':
            for j in range(len(feature.get_data())):
                # Change each value from original string to integer correlating to its value's index
                dataset['features'][i].get_data()[j] = dataset['features'][i].get_values().index(dataset['features'][i].get_data()[j])
    # Convert label into numerical values as well
    labels = dataset['label']
    if labels.get_type() == 'nominal':
        for j in range(len(labels.get_data())):
            # Change each value from original string to integer correlating to value's index
            dataset['label'].get_data()[j] = dataset['label'].get_values().index(dataset['label'].get_data()[j])

"""
This method is the entry point into pre-processing and calls the other functions. 
"""
def pre_process(dataset: Dict):
    # Remove null values
    remove_null(dataset)
    # Convert all continuous and discrete data into floats
    convert_data(dataset)
    # Normalize continuous and discrete colums
    normalize(dataset)
    # Integer convert nominal columns
    integer_converted(dataset)
    print("Preprocessing complete")