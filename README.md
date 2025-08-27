# Radial Basis Function (RBF) Neural Network with Momentum

This project implements a Radial Basis Function (RBF) neural network from scratch using Python and NumPy to approximate the sine function. The implementation includes momentum-based training with comprehensive analysis of different gamma (momentum) values.

## Project Overview

The RBF network is designed to learn the sine function over the interval [0, π] using Gaussian activation functions in the hidden layer and linear activation in the output layer. The network architecture consists of:

- **Input Layer**: 1 neuron (input values)
- **Hidden Layer**: 2 neurons with Gaussian (RBF) activation functions
- **Output Layer**: 1 neuron with linear activation function

## Features

- ✅ Custom RBF neural network implementation
- ✅ Gaussian activation functions with derivatives
- ✅ Momentum-based backpropagation training
- ✅ Comprehensive error tracking and visualization
- ✅ Comparison of different momentum values (γ = 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
- ✅ Real-time plotting of network responses vs target function
- ✅ SSE (Sum of Squared Errors) convergence visualization

## Mathematical Foundation

### Network Architecture
- **Hidden Layer**: Uses Gaussian RBF activation: `φ(n) = exp(-n²)`
- **Output Layer**: Linear activation: `f(n) = n`
- **Distance Metric**: Absolute difference weighted by bias terms

### Training Algorithm
- **Backpropagation** with momentum term
- **Learning Rate**: 0.01 (fixed)
- **Momentum Values**: 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
- **Convergence Criterion**: SSE < 0.001 or max 2000 iterations

## Requirements

```
numpy
matplotlib
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd rbf-neural-network
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

### Basic Usage

Run the main script to train the RBF network:

```bash
python Question6.py
```

### Testing Different Momentum Values

The script includes commented lines for testing different momentum values. Uncomment the desired lines to run experiments:

```python
# Uncomment to test different momentum values
train_rbf(rbf, inputs, targets, momentum_term=0.2, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
train_rbf(rbf, inputs, targets, momentum_term=0.4, learning_rate=0.01, max_iterations=2000, sse_cutoff=0.001)
# ... and so on
```

## Results Analysis

### Network Configuration
- **Training Samples**: 100 observations
- **Input Range**: [0, π]
- **Target Function**: sin(x)
- **Maximum Iterations**: 2000
- **SSE Threshold**: 0.001

### Momentum Effect Analysis

The project systematically analyzes the effect of different gamma (momentum) values:

| Gamma (γ) | Learning Rate | Hidden Neurons | Max Iterations | SSE Threshold |
|-----------|---------------|----------------|----------------|---------------|
| 0.1       | 0.01         | 2              | 2000           | 1e-3          |
| 0.2       | 0.01         | 2              | 2000           | 1e-3          |
| 0.4       | 0.01         | 2              | 2000           | 1e-3          |
| 0.6       | 0.01         | 2              | 2000           | 1e-3          |
| 0.8       | 0.01         | 2              | 2000           | 1e-3          |
| 1.0       | 0.01         | 2              | 2000           | 1e-3          |

### Visualizations

The program generates two key plots for each training session:

1. **Network Response vs Target**: Shows how well the trained network approximates the sine function
2. **SSE Convergence**: Log-scale plot showing error reduction over training iterations

## Code Structure

### Main Components

- **`RBF` Class**: Main neural network implementation
  - `forward()`: Forward propagation through the network
  - `backward()`: Backpropagation for gradient calculation
  - `update_params()`: Parameter updates with momentum
  - `update_params_1()`: Initial parameter update (first iteration)

- **Activation Functions**:
  - `guassian()`: Gaussian RBF activation
  - `guassian_derivative()`: Derivative of Gaussian function
  - `linear()`: Linear activation for output layer
  - `linear_derivative()`: Derivative of linear function

- **Training Function**:
  - `train_rbf()`: Main training loop with visualization

### Key Features

- **Momentum Implementation**: Incorporates previous weight changes for smoother convergence
- **Error Tracking**: Monitors SSE throughout training
- **Early Stopping**: Training halts when SSE threshold is reached
- **Comprehensive Logging**: Displays initial and final network parameters

## Expected Outcomes

Based on the momentum analysis, you should observe:

- **Lower Momentum Values (0.1-0.4)**: More stable but potentially slower convergence
- **Higher Momentum Values (0.6-1.0)**: Faster convergence but potential instability
- **Optimal Range**: Typically around 0.4-0.6 for this function approximation problem

## Educational Value

This implementation demonstrates:

- RBF network fundamentals
- Momentum-based optimization
- Function approximation using neural networks
- Hyperparameter sensitivity analysis
- Visualization of training dynamics

## Contributing

Feel free to contribute improvements, additional analysis, or optimizations to this educational implementation.

## License

This project is for educational purposes. Feel free to use and modify as needed.

---

**Note**: This implementation is designed for educational purposes to understand RBF networks and momentum effects. For production applications, consider using established deep learning frameworks like TensorFlow or PyTorch.
