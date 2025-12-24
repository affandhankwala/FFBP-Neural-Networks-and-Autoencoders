from typing import Dict
import pandas as pd
from feature import Feature

# This file is responsible for loading all the data from the datasets into a dictionary 

"""
This method creates a dictionary that holds all the feature values within the 'feature' key as a 
list of Feature objects. The labels are held in a label key which is represented by a Feature. 
Each of the 6 datasets has it's own condition.
"""
def select_dataset(file_name: str) -> Dict:
    dataset = {}
    dataset['name'] = file_name
    if file_name == "abalone.data":
        dataset['features'] = [
            Feature("Sex", "nominal", values=["M", "F", "I"]),
            Feature("Length", "continuous"),
            Feature("Diameter", "continuous"),
            Feature("Height", "continuous"),
            Feature("Whole weight", "continuous"),
            Feature("Shucked weight", "continuous"),
            Feature("Viscera weight", "continuous"),
            Feature("Shell weight", "continuous")
        ]
        dataset['label'] = Feature("Rings", "discrete")
        dataset['header'] = False
    elif file_name == "breast-cancer-wisconsin.data":
        dataset['features'] = [
            Feature("Sample code Number", "ID"),
            Feature("Clump Thickness", "discrete"),
            Feature("Uniformity of Cell Size", "discrete"),
            Feature("Uniformity of Cell Shape", "discrete"),
            Feature("Marginal Adhesion", "discrete"),
            Feature("Signal Epithalial Cell Size",  "discrete"),
            Feature("Bare Nuclei", "discrete"),
            Feature("Bland Chromatin", "discrete"),
            Feature("Normal Nucleoli", "discrete"),
            Feature("Mitosis", "discrete") 
        ]
        dataset['label'] = Feature("Class", "nominal", [2, 4])
        dataset['header'] = False
    elif file_name == "car.data":
        dataset['features'] = [
            Feature("Buying", "nominal", values=["vhigh", "high", "med", "low"]),
            Feature("Maint", "nominal", values=["vhigh", "high", "med", "low"]),
            Feature("Doors", "nominal", values=["2", "3", "4", "5more"]),
            Feature("Persons", "nominal", values=["2", "4", "more"]),
            Feature("Lug_boot", "nominal", values=["small", "med", "big"]),
            Feature("Safety", "nominal", values=["low", "med", "high"])
        ]
        dataset['label'] = Feature("Class", "nominal", values=["unacc", "acc", "good", "vgood"])
        dataset['header'] = False
    elif file_name == "forestfires.data":
        dataset['features'] = [
            Feature("X", "discrete"),
            Feature("Y", "discrete"),
            Feature("Month", "nominal", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
            Feature("Day", "nominal", ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]),
            Feature("FFMC", "continuous"),
            Feature("DMC", "continuous"),
            Feature("DC", "continuous"),
            Feature("ISI", "continuous"),
            Feature("Temp", "continuous"),
            Feature("RH", "continuous"),
            Feature("Wind", "continuous"),
            Feature("Rain", "continuous")            
        ]
        dataset['label'] = Feature("Area", "continuous")
        dataset['header'] = True
    elif file_name == "house-votes-84.data":
        dataset['features'] = [
            Feature("handicapped-infants", "nominal", ["y", "n"]),
            Feature("water-project-cost-sharing", "nominal", ["y", "n"]),
            Feature("adoption-of-the-budget-resolution", "nominal", ["y", "n"]),
            Feature("physician-fee-freeze", "nominal", ["y", "n"]),
            Feature("el-salvador-aid", "nominal", ["y", "n"]),
            Feature("religious-groups-in-schools", "nominal", ["y", "n"]),
            Feature("anti-satellite-test-ban", "nominal", ["y", "n"]),
            Feature("aid-to-nicaraguan-contras", "nominal", ["y", "n"]),
            Feature("mx-missile", "nominal", ["y", "n"]),
            Feature("immigration", "nominal", ["y", "n"]),
            Feature("synfuels-corporation-cutback", "nominal", ["y", "n"]),
            Feature("education-spending", "nominal", ["y", "n"]),
            Feature("superfund-right-to-sue", "nominal", ["y", "n"]),
            Feature("crime", "nominal", ["y", "n"]),
            Feature("duty-free-exports", "nominal", ["y", "n"]),
            Feature("export-administration-act-south-africa", "nominal", ["y", "n"])
        ]
        dataset['label'] = Feature("Class Name", "nominal", ["democrat", "republican"])
        dataset['header'] = False
    elif file_name == "machine.data":
        dataset['features'] = [
            Feature("Vendor Name", "nominal", ["adviser", "amdahl", "apollo",
                                            "basf","bti","burroughs","c.r.d",
                                            "cambex", "cdc", "dec", "dg", "formation",
                                            "four-phase", "gould", "harris",
                                            "honeywell","hp", "ibm", 
                                            "ipl", "magnuson", "microdata", "nas", "ncr", 
                                            "nixdorf", "perkin-elmer", "prime", "siemens", "sperry",
                                            "sratus", "wang"]),
            Feature("Name", "name"),
            Feature("MYCT", "discrete"),
            Feature("MMIN", "discrete"),
            Feature("MMAX", "discrete"),
            Feature("CACH", "discrete"),
            Feature("CHMIN", "discrete"),
            Feature("CHMAX", "discrete"),
            Feature("ERP", "discrete")
        ]
        dataset['label'] = Feature("PRP", "discrete")
        dataset['header'] = False
    else:
        print("Dataset was not found")
        return None
    return dataset

"""
This method assigns the data to each of the features within the datasets. It is critical that 
the information is normalized. This is why we have 2 special cases due to the structure of our 
CSV dataset
"""
def combine_dataset(dataset:Dict, df) -> Dict:
    if dataset['name'] == "house-votes-84.data":
        # Special case of votes dataset
        for col in range(1, df.shape[1]):
            column_data = df.iloc[:, col].tolist()
            dataset['features'][col - 1].assign_data(column_data)
        dataset['label'].assign_data(df.iloc[:, 0].tolist())
    elif dataset['name'] == "machine.data":
        # Special case of machine dataset
        for col in range(df.shape[1] - 2):
            column_data = df.iloc[:, col].tolist()
            dataset['features'][col].assign_data(column_data)
        dataset['features'][len(dataset['features']) - 1].assign_data(df.iloc[:, -1].tolist())
        dataset['label'].assign_data(df.iloc[:, -2].tolist())
    else:
        for col in range(df.shape[1] - 1):
            column_data = df.iloc[:, col].tolist()
            dataset['features'][col].assign_data(column_data)
        dataset['label'].assign_data(df.iloc[:, -1].tolist())
    return dataset

"""
This method calls select_dataset and combine_dataset to create a dictionary holding all of the values
of the dataset CSV.
"""
def load_data (directory_name: str, file_name: str) -> Dict:
    # Get column information
    dataset = select_dataset(file_name)
    # Pull dataframe
    file_path = f"{directory_name}/{file_name}"
    if dataset['header'] == False:
        df = pd.read_csv(file_path, header=None)
    else:
        df = pd.read_csv(file_path)
    # Combine data so that each Feature has its respective values
    combine_dataset(dataset, df)
    print("Data Loaded")
    return dataset