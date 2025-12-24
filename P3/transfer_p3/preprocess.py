from typing import List, Dict
from calculations import get_dataset_mode, get_dataset_median, normalize

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

def convert_data(dataset: Dict):
    # Convert all strings that are floats into floats
    for i in range(len(dataset['features'])):
        if (dataset['features'][i].get_type() == 'continuous' or 
            dataset['features'][i].get_type() == 'discrete'):
            for j in range(len(dataset['features'][i].get_data())):
                dataset['features'][i].alter_data(j, float(dataset['features'][i].get_data()[j]))

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