from load import load_data
from preprocess import pre_process
from simple_nn import build_simple_nn
import time


def main():
    # All datasets are stored within the 'dataset' directory, so we define this variable
    directory_name = "datasets"
    # Selected dataset name
    file_name = "house-votes-84.data"
    # Start Timer
    start_time = time.time()
    # Load_data
    dataset = load_data(directory_name, file_name)
    # Preprocess
    pre_process(dataset)
    # Build simple nn
    build_simple_nn(dataset, 0.1)

    # Cross validate
    #cross_validate_tree(dataset, file_name)
    # Capture end time
    end_time = time.time()
    # Print results
    print(f"Decision Tree trained, pruned, and tested: {round(end_time - start_time, 2)} s")

    
if __name__ == "__main__": 
    main()