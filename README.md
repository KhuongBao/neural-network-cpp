# Neural Network in C++

This project implements a simple neural network in C++ using the Eigen library for matrix operations. The neural network supports various activation functions and loss functions and can be used for both binary and multi-class classification tasks.

## Features

- Supports multiple activation functions: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax
- Supports multiple loss functions: Binary Cross-Entropy, Categorical Cross-Entropy, Mean Squared Error
- Implements forward and backward propagation
- Uses the Eigen library for efficient matrix operations

## Prerequisites

- C++11 or later
- Eigen library (https://eigen.tuxfamily.org/)

<ins>(OPTIONAL)</ins>
- Python 3 with Jupyter Notebook
- Python libraries: `scikit-learn`, `matplotlib`, `numpy`, `pandas`

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/KhuongBao/neural-network-cpp.git
    cd neural-network-cpp
    ```

2. Download and install the Eigen library:
    ```
    git clone https://gitlab.com/libeigen/eigen.git	
    ```

3. Set the path to the installed Eigen library in Makefile:
    ```
    EIGEN_PATH = "C:/[path_to_eigen]"
    ```

4. Install the optional Python libraries:
    ```
    pip install scikit-learn matplotlib numpy pandas
    ```
## Example Usage

### Step 1: Generate Dataset

1. Open the `Data.ipynb` notebook in Jupyter Notebook.
2. Run the first two (2) cells to generate `train.csv` and `test.csv` datasets.

_Note: The `train.csv` and `test.csv` files are already included in the repository. You can use these files directly or generate new random datasets using the Python notebook._


### Step 2: Train the Model 

1. [Modify](https://github.com/KhuongBao/neural-network-cpp?tab=readme-ov-file#configuration) the `main.cpp` file if necessary to configure the neural network parameters.
2. Compile and run the `main.cpp` file to train the model and save predictions:
    ```
    make
    ```
_Note: Parameters are saved automatically to “parameters.csv”. These can be loaded in using model.load_parameters(filename) to skip training steps_


### Step 3: Visualize Decision Boundaries

1. Go back to the `Data.ipynb` notebook.
2. Run the remaining cells to visualize the decision boundaries for the training and test datasets.

## Configuration
You can switch configurations in `main.cpp` to customize the neural network. Here are some options:
### Activation Functions
-	ReLU: "relu"
-	Sigmoid: "sigmoid"
-	Tanh: "tanh"
-	Leaky ReLU: "leaky_relu"
-	Softmax: "softmax"
### Loss Functions
-	Binary Cross-Entropy: "binary_cross_entropy"
-	Categorical Cross-Entropy: "categorical_cross_entropy"
-	Mean Squared Error: "mean_squared_error"


