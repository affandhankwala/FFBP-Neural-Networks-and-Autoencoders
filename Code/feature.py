from typing import List

"""
This object houses all of the data from the dataset. Each of these objects has
a set of member variables and functions that are defined below
"""
class Feature:
    def __init__(self, name, type, values=None) -> None:
        # The name of the Feature
        self.name = name
        # The type of data. Discrete, Continuous, ID, Name, Nominal
        self.type = type
        # If the Feature holds nominal data, all possible values are placed in this list
        self.values = values
        # The data of the feature
        self.data = None
    # Data setter
    def assign_data(self, data: List) -> None:
        self.data = data
    # Altering the data at any particular index
    def alter_data(self, index: int, new_value) -> None:
        self.data[index] = new_value
    # Data getter
    def get_data(self) -> List:
        return self.data
    # Name getter
    def get_name(self) -> str:
        return self.name
    # Type Getter
    def get_type(self) -> str:
        return self.type
    # Values list getter
    def get_values(self) -> List:
        return self.values
