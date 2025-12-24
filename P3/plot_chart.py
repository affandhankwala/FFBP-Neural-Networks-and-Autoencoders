from typing import List
import matplotlib.pyplot as plt

"""
This function will print out all plots
"""
def plot(x: List, y: List, title: str, xlabel: str, ylabel: str):
    plt.plot(x, y)
    plt.title(f"{title}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.show()