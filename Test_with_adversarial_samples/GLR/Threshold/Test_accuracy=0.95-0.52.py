# coding: utf-8
# Python imports
import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
# from sklearn import datasets, cross_validation, metrics # data and evaluation utils\
from sklearn.model_selection import train_test_split
from sklearn import *
from matplotlib.colors import colorConverter, ListedColormap  # some plotting functions
import itertools
import collections
import os
import pickle
# import the module related to shrink the size of image
from skimage import transform, data

def add_one(data):
    ones = np.ones(tuple(data.shape[:-1]) + (1,))
    return np.concatenate([ones, data], axis=-1)

# Define the non-linear functions used
def logistic(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)


def threshold(z):
    z[z < 0] = 0
    return np.sign(z)

# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""

    def get_params(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []

    def get_grad(self, loss):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.

        """
        return []

    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass

    def update_param(self, loss, learning_rate):
        pass

class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out, sigm, n_each):
        """
        Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables.
        sigm is the std-deviation of noise.
        """
        self.sigm = sigm
        self.W = np.random.randn(n_in + 1, n_out) * 0.1

    def get_params(self):
        """Return an iterator over the parameters."""
        return self.W

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        self.noise = np.random.randn(*(X.shape[:-1] + (self.W.shape[-1],))) * self.sigm
        self.input = add_one(X)
        return self.input.dot(self.W) + self.noise

    def get_grad(self, loss):
        """Return a list of gradients over the parameters."""
        return (np.einsum('abc,abd->abcd', (self.input * loss[:, :, np.newaxis]), self.noise) / (self.sigm ** 2)).mean(
            axis=(0, 1))

    def update_param(self, loss, learning_rate):
        self.W -= learning_rate * self.get_grad(loss)

class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        # re = logistic(X)
        re = threshold(X)
        return re

class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)

    def get_cost(self, Y, T, n_each, train=True):
        """Return the cost at the output of this output layer."""
        if train:
            T = np.stack([T] * n_each).transpose([1, 0, 2])
            return -np.multiply(T, np.log(Y)).sum(axis=-1)
        else:
            return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]

# Define the forward propagation step as a method.
def forward_step(input_samples, layers, n_each, train=True):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.
    """
    if train:
        X = np.stack([input_samples] * n_each).transpose([1, 0, 2])
    else:
        X = input_samples
    for layer in layers:
        if train:
            pass
        else:
            layer.noise = 0
        X = layer.get_output(X)
    return X

# Define a method to update the parameters
def update_params(layers, loss, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer in layers:
        layer.update_param(loss, learning_rate)

sigm = 0
n_each = 10000
# Define a sample model to be trained on the data
hidden_neurons_1 = 20  # Number of neurons in the first hidden-layer
hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
# Create the model
layers = []  # Define a list of layers
# Add hidden layer
layers.append(LinearLayer(14 * 14, hidden_neurons_1, sigm, n_each))
layers.append(LogisticLayer())
# Add output layer
layers.append(LinearLayer(hidden_neurons_2, 10, sigm, n_each))
layers.append(SoftmaxOutputLayer())

# Load pre-trained weights
weights_dict = {}
with open("weights.pkl", "rb") as f:
    weights_dict = pickle.load(f,encoding='latin1')
layers[0].W = weights_dict["layer1_weights"]
layers[2].W = weights_dict["layer2_weights"]
print("Weights Loaded!")

# load the adversarial samples
adv_dict = {}
with open("adversarial_samples_with_FGSM.pkl", "rb") as f:
    adv_dict = pickle.load(f)
xs = adv_dict["xs"]
y_trues = adv_dict["y_trues"]
y_preds = adv_dict["y_preds"]
noises  = adv_dict["noises"]
y_preds_adversarial = adv_dict["y_preds_adversarial"]

# get the adversarial samples
xs = np.array(xs).reshape(len(xs),196)
noises = np.array(noises).reshape(len(xs),196)
y_trues = np.array(y_trues).reshape(len(xs))
advs = xs + noises

# Get results of test data
activation = forward_step(xs, layers, n_each, False)
y_pred = np.argmax(activation, axis=1)
test_accuracy = metrics.accuracy_score(y_trues, y_pred)
print('The accuracy on the initial_x set is {:.2f}'.format(test_accuracy)) #0.95

# test the accuracy of adversarial samples
activation = forward_step(advs, layers, n_each, False)
y_pred = np.argmax(activation, axis=1)
adv_accuracy = metrics.accuracy_score(y_trues, y_pred)
print('The accuracy on the adversarial samples is {:.2f}'.format(adv_accuracy)) #0.52
