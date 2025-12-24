from typing import List

class Feature:
    def __init__(self, name, type, values=None, values_dict=None):
        self.name = name
        self.type = type
        self.values = values
        self.values_dict = values_dict
        self.data = None
    def assign_data(self, data: List):
        self.data = data
    def alter_data(self, index: int, new_value):
        self.data[index] = new_value
    def set_mean_std(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev
    def setC(self, c: int, v: int):
        self.values_dict[v]["c"] = c
    def setCa(self, ca: List, v: int):
        self.values_dict[v]["ca"] = ca
    def get_data(self):
        return self.data
    def get_name(self):
        return self.name
    def get_type(self):
        return self.type
    def get_values(self):
        return self.values
    def get_values_dict(self):
        return self.values_dict
    def delete_data(self, index: int):
        self.data.pop(index)