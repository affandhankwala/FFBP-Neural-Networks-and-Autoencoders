# Feed-Forward Neural Networks and Autoencoders

A from-scratch implementation of a **feed-forward neural network trained with backpropagation**, plus an
**autoencoder**, evaluated on a range of classification and regression datasets.

## Introduction

This project was completed for Johns Hopkins University's *Introduction to Machine Learning* course
(605.649).

## Problem Statement

The task is to design, develop, train, and test a feed-forward neural network using backpropagation,
where backpropagation drives the adjustment of edge weights. An **autoencoder** is also implemented, and
its benefit as a learned feature representation for the downstream classification and regression tasks is
analyzed.

The network is built without high-level deep-learning frameworks — the forward pass, loss, gradients, and
weight updates are all implemented directly (see `nn.py`, `calculations.py`, and `matrix_calculations.py`).

## Datasets

The model is evaluated using ten-fold cross-validation across several UCI datasets (stored in
`Code/datasets/`):

| Dataset | Task |
|---------|------|
| `breast-cancer-wisconsin` | Classification |
| `car` | Classification |
| `house-votes-84` | Classification |
| `abalone` | Regression |
| `forestfires` | Regression |
| `machine` | Regression |

## Repository Structure

```
FFBP-Neural-Networks-and-Autoencoders/
├── README.md
├── Report.pdf                 # Full write-up of methodology and results
└── Code/
    ├── main.py                # Entry point: load → preprocess → tune → cross-validate
    ├── load.py                # Dataset loading
    ├── preprocess.py          # Cleaning / encoding / scaling
    ├── dataset_manipulation.py
    ├── feature.py             # Feature handling
    ├── nn.py                  # Neural network model
    ├── calculations.py        # Activation, loss, and gradient math
    ├── matrix_calculations.py # Matrix helpers
    ├── cross_validate.py      # k-fold cross-validation loop
    ├── tune.py                # Hyperparameter tuning / tuned-value lookup
    ├── plot_chart.py          # Result plotting
    ├── data.xlsx              # Tabulated results
    └── datasets/              # UCI .data / .names files
```

## Running

Select the dataset to run inside `Code/main.py` (the `file_name` variable, e.g. `"car.data"`), then:

```bash
cd Code
python main.py
```

The script loads and preprocesses the chosen dataset, retrieves tuned hyperparameters, runs
cross-validation, and reports the total training/testing time.

### Requirements

- Python 3
- `numpy`, `pandas`, `matplotlib` (and `openpyxl` for the spreadsheet output)

## Results

See [`Report.pdf`](Report.pdf) for the full report on network performance, the effect of the autoencoder
on classification and regression accuracy, and the cross-validation results.
