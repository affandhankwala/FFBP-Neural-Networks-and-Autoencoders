from load import load_data
from preprocess import pre_process
from cross_validate import cross_validate
from tune import get_tuned_values, tune_all
import time

"""
Serving as the starting point of the project, all of the data loading, preprocessing, 
tuning, training and evaluating is done from here.
"""
def main():
    # All datasets are stored within the 'dataset' directory, so we define this variable
    directory_name = "datasets"
    # Selected dataset name
    file_name = "car.data"
    # Start Timer
    start_time = time.time()
    # Load_data
    dataset = load_data(directory_name, file_name)
    # Preprocess
    pre_process(dataset)
    # Fetch Tuned hyperparmeters
    tuned_values = get_tuned_values(file_name)
    # Cross validate
    cross_validate(dataset, tuned_values)
    end_time = time.time()
    # Conclude
    print(f"Neural networks trained and tested: {round(end_time - start_time, 2)} s")

    
if __name__ == "__main__": 
    main() 