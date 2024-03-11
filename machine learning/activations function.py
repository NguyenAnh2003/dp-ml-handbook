import matplotlib.pyplot as plt
import numpy as np

# implement these activation functions and plot their derivatives
# activation function -> activate the value
# flow: linear sum -> activation function -> input for next hidden layer in NN

# random values
x = np.linspace(-10, 10, 100) # valuate fluctuate from -10 -> 10 with 100 samples

# sigmoid can be considered as Logistic Function value fluctuate 0 - 1,
# the output of sigmoid considered as probability

# defining plot function for common activation function
def plot_function(values, derivative):
    """
    :param sigmoid_values: sigmoid values with random values
    :param sigmoid_derivative: sigmoid derivative with random values
    """
    plt.figure(figsize=(5, 5))
    plt.plot(x, values, label="Sigmoid Function")
    plt.plot(x, derivative, label="Sigmoid Derivatives")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def sigmoid_f(x):
    sigmoid_v = 1/ (1 + np.exp(-x))
    return sigmoid_v

def sigmoid_derivative(x):
    return sigmoid_f(x) * (1-sigmoid_f(x))

# Tanh - Hyperbolic tangent
def tanh_f(x):
    # formula: (e^x - e^(-x)) / (e^x + e^(-x))
    return np.tanh(x) # using numpy lib

def tanh_derivative(x):
    return 1.0 - tanh_f(x)**2

# ReLU
def relu_f(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Leaky ReLU
def leaky_relu_r(x, a=0.1):
    return np.where(x > 0, x, a*x)

def leaky_relu_derivative(x, a=0.1):
    return np.where(x > 0, 1, a)

# ELU
def elu_f(x):
    return

def elu_der(x):
    return

if __name__ == "__main__":
    # sigmoid
    sigmoid_values = sigmoid_f(x)
    sigmoid_derivatives = sigmoid_derivative(x)
    plot_function(sigmoid_values, sigmoid_derivatives)

    # tanh
    tanh_values = tanh_f(x)
    tanh_derivatives = tanh_derivative(x)
    plot_function(tanh_values, tanh_derivatives)

    # relu
    relu_v  = relu_f(x)
    relu_der = relu_derivative(x)
    plot_function(relu_v, relu_der)

    # leaky relu
    lrelu_v = leaky_relu_r(x)
    lrelu_der = leaky_relu_derivative(x)
    plot_function(lrelu_v, lrelu_der)