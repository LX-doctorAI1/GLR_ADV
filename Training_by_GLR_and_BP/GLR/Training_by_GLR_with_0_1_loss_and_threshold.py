# coding: utf-8
# Python imports
import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
from sklearn.model_selection import train_test_split
from sklearn import *
from matplotlib.colors import colorConverter, ListedColormap  # some plotting functions
import itertools
import collections
import os
import time
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

def data_transform(d):
    shape = d.shape
    d = np.reshape(d, [-1, shape[-1]])
    d[np.arange(d.shape[0]), np.argmax(d, axis=-1)] = 1
    d = np.reshape(d, shape)
    d[d != 1] = 0
    return d


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
        #         return  (self.input[:, :, np.newaxis].dot(self.noise.reshape(1, self.noise.shape[0])) * loss / self.sigm**2).mean(axis=0)
        return (np.einsum('abc,abd->abcd', (self.input * loss[:, :, np.newaxis]), self.noise) / (self.sigm ** 2)).mean(
            axis=(0, 1))

    #         print((np.einsum('abc,abd->abcd',(self.input * loss[:, :, np.newaxis]), self.noise)/(self.sigm**2)).mean(axis=(0, 1))[0])

    def update_param(self, loss, learning_rate):
        self.W -= learning_rate * self.get_grad(loss)


class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        re = threshold(X)
        return re


class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""

    def get_output(self, X):
        """Perform the forward step transformation."""
        return data_transform(softmax(X))

    # 0-1LOSS函数 :
    def get_one_zero_cost(self, Y, T, n_each, train=True):
        if train:
            T = np.stack([T] * n_each).transpose([1, 0, 2])
            max_index1 = np.argmax(Y, axis=-1).reshape(-1)
            max_index2 = np.argmax(T, axis=-1).reshape(-1)
            compar_index = 1 * (max_index1 != max_index2)
            cost = compar_index.reshape(Y.shape[0], Y.shape[1])
        else:
            max_index1 = np.argmax(Y, axis=-1).reshape(-1)
            max_index2 = np.argmax(T, axis=-1).reshape(-1)
            compar_index = 1 * (max_index1 != max_index2)
            cost = np.sum(compar_index) / float(compar_index.shape[0])
        return cost


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


def flat_trans(x):
    x.resize_(28 * 28)  
    return x


class Mnist(object):

    def __init__(self):
        self.dataname = "Mnist"
        self.dims = 28 * 28
        self.shape = [28, 28, 1]
        self.image_size = 28
        self.data, self.data_y = self.load_mnist()

    def load_mnist(self):
        data_dir = os.path.join("", "../../mnist/raw")
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        # 鍒╃敤np.fromfile璇彞灏嗚繖涓猽byte鏂囦欢璇诲彇杩涙潵
        # 闇€瑕佹敞鎰忕殑鏄敤np.uint8鐨勬牸锟?        # 杩樻湁璇诲彇杩涙潵鐨勬槸涓€涓竴缁村悜锟?        # <type 'tuple'>: (47040016,)锛岃繖灏辨槸loaded鍙橀噺鐨勮瀹屼箣鍚庣殑鏁版嵁绫诲瀷
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
        # 'train-images-idx3-ubyte'杩欎釜鏂囦欢鍓嶅崄鍏綅淇濆瓨鐨勬槸涓€浜涜鏄庡叿浣撴墦鍗扮粨鏋滃涓嬶細
        point = loaded[:16]
        # print(point)
        # [  0   0   8   3   0   0 234  96   0   0   0  28   0   0   0  28]
        # 搴忓彿锟?寮€濮嬶紝涓婅堪鏁板瓧鏈変笅闈㈣繖鍑犱釜鍏紡鐨勫惈锟?        # MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
        # ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);    绛変簬60000
        # ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12); 绛変簬28
        # ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);绛変簬28

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        point = loaded[:8]
        # print(point)
        # [  0   0   8   1   0   0 234  96]
        # 杩欎簺鏁板瓧鐨勪綔鐢ㄥ拰涓婅堪绫讳技
        # 杩欎簺鏁板瓧鐨勫姛鑳戒箣涓€灏辨槸鍙互鍒ゆ柇浣犱笅杞界殑鏁版嵁闆嗗涓嶅锛屽叏涓嶅叏

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        # 鐩殑鏄负浜嗘墦涔辨暟鎹泦
        # 杩欓噷闅忔剰鍥哄畾涓€涓猻eed锛屽彧瑕乻eed鐨勫€间竴鏍凤紝閭ｄ箞鎵撲贡鐭╅樀鐨勮寰嬪氨鏄竴鐪肩殑
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        # convert label to one-hot
        # 鎵嬪姩灏嗘暟鎹浆鎹㈡垚one-hot缂栫爜褰㈠紡
        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, int(y[i])] = 1.0
        return X / 255., y_vec

mn_object = Mnist()
XX2 = mn_object.data.reshape(70000, 784)
XX = []
YY = mn_object.data_y.reshape(70000, 10)

# change the size of image
for i in range(XX2.shape[0]):
    # (784,) -> (28,28)
    x = XX2[i].reshape(28, 28)
    # (28,28) -> (14,14)
    x_little = transform.resize(x, (14, 14))
    # (14,14) -> (324,)
    x_little = x_little.reshape(196)
    XX.append(x_little)
XX = np.array(XX)

# Divide the data into training sets and test sets
X_train, X_validation, T_train, T_validation = train_test_split(
    XX, YY, test_size=0.3)

# # Create the minibatches
batch_size = 25
nb_of_batches = int(X_train.shape[0] / batch_size)  # Number of batches=1960  49000=1960*25
# Create batches (X,Y) from the training set
X_batches = np.array_split(X_train, nb_of_batches, axis=0)  # X samples
T_batches = np.array_split(T_train, nb_of_batches, axis=0)  # Y targets

sigm = 2
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

# Perform backpropagation
# initalize some lists to store the cost for future analysis
# cost of iteration
training_costs_iteration = []
validation_costs_iteration = []
# cost of epoch
training_costs_epoch = []
validation_costs_epoch = []

max_nb_of_iterations = 30
# min_nb_of_iterations = 8
learning_rate = 0.1

# introduction
print(time.strftime('========== %Y-%m-%d %H:%M:%S ==========', time.localtime(time.time())))
print("SDENN minist training : 0-1_LOSS,threshold,maxeopch=" + str(max_nb_of_iterations) + ",sigm=" + str(
    sigm) + ",sample_size=14*14" + ",loss_times=" + str(10))

# begin to train
for iteration in range(max_nb_of_iterations):
    n_batch = 0
    for X, T in zip(X_batches, T_batches):  # For each minibatch sub-iteration
        n_batch += 1
        Y = forward_step(X, layers, n_each)  # Get the activations
        minibatch_cost = layers[-1].get_one_zero_cost(Y, T, n_each)  # Get cost
        # minibatch_costs.append(minibatch_cost.mean())
        update_params(layers, minibatch_cost, learning_rate)  # Update the parameters

        if n_batch % 100 == 0:
            # cost1 : get cost of iteration
            # Get training set cost
            activation = forward_step(X_train, layers, n_each, False)
            train_cost = layers[-1].get_one_zero_cost(activation, T_train, n_each, False)
            training_costs_iteration.append(train_cost)
            # Get validation set cost
            activation = forward_step(X_validation, layers, n_each, False)
            validation_cost = layers[-1].get_one_zero_cost(activation, T_validation, n_each, False)
            # print the current loss
            print("validation_cost  /  100batches:  {}".format(validation_cost))
            print(time.strftime('========== %Y-%m-%d %H:%M:%S ==========', time.localtime(time.time())))
            validation_costs_iteration.append(validation_cost)

    # cost2 : get cost of epoch
    # Get training set cost
    activation = forward_step(X_train, layers, n_each, False)
    train_cost = layers[-1].get_one_zero_cost(activation, T_train, n_each, False)
    training_costs_epoch.append(train_cost)
    # Get validation set cost
    activation = forward_step(X_validation, layers, n_each, False)
    validation_cost = layers[-1].get_one_zero_cost(activation, T_validation, n_each, False)
    validation_costs_epoch.append(validation_cost)

    nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    print("<<============= epoch : " + str(nb_of_iterations) + " =============>>")
    print("SDENN minist training : 0-1_LOSS,threshold,maxeopch=" + str(max_nb_of_iterations) + ",sigm=" + str(
        sigm) + ",sample_size=14*14" + ",loss_times=" + str(10))
    # Get results of validation data
    y_true = np.argmax(T_validation, axis=1)
    activation = forward_step(X_validation, layers, n_each, False)
    y_pred = np.argmax(activation, axis=1)
    test_accuracy = metrics.accuracy_score(y_true, y_pred)
    print('The accuracy on the validation set is {:.2f}'.format(test_accuracy))

    # save the result each epoch
    if (nb_of_iterations % 1) == 0:
        print("----------- begin : save result -----------")

        # Get results of validation data
        y_true = np.argmax(T_validation, axis=1)
        activation = forward_step(X_validation, layers, n_each, False)
        y_pred = np.argmax(activation, axis=1)
        test_accuracy = metrics.accuracy_score(y_true, y_pred)
        print('The accuracy on the validation set is {:.2f}'.format(test_accuracy))

        print("Dumping weights to disk")
        weights_dict = {}
        # save the weights of nn
        weights_dict["layer1_weights"] = layers[0].W
        weights_dict["layer2_weights"] = layers[2].W
        with open(
                "weights_single_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + "_threshold.pkl",
                "wb") as f:
            pickle.dump(weights_dict, f)
        print("Finished dumping to disk..")

        # Plot and save cost of iteration
        minibatch_x_inds = np.linspace(0, len(training_costs_iteration), len(training_costs_iteration))
        plt.plot(minibatch_x_inds, training_costs_iteration, 'b-', linewidth=3, label='training set')
        plt.plot(minibatch_x_inds, validation_costs_iteration, 'r--', linewidth=3, label='validation set')
        # Add labels to the plot
        plt.xlabel('iteration', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.legend()
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, len(training_costs_iteration), 0, 6.0))
        plt.grid()
        # save figure
        plt.savefig("iteration/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".png")
        # save train_loss and val_loss
        np.save("iteration/train_loss/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".npy",
                training_costs_iteration)
        np.save("iteration/val_loss/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".npy",
                validation_costs_iteration)
        plt.close()

        # Plot and save cost of epoch
        minibatch_x_inds = np.linspace(0, len(training_costs_epoch), num=len(training_costs_epoch))
        plt.plot(minibatch_x_inds, training_costs_epoch, 'b-', linewidth=3, label='training set')
        plt.plot(minibatch_x_inds, validation_costs_epoch, 'r--', linewidth=3, label='validation set')
        # Add labels to the plot
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('loss', fontsize=15)
        plt.legend()
        x1, x2, y1, y2 = plt.axis()
        plt.axis((0, len(training_costs_epoch), 0, 6.0))
        plt.grid()
        # save figure
        plt.savefig("epoch/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".png")
        # save train_loss and val_loss
        np.save("epoch/train_loss/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".npy",
                training_costs_epoch)
        np.save("epoch/val_loss/loss_acc=" + str(test_accuracy) + "_runedepoch=" + str(nb_of_iterations) + ".npy",
                validation_costs_epoch)
        plt.close()

        print("----------- end : save result -----------")






