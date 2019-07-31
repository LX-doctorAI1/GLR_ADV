
# coding: utf-8
# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# Allow matplotlib to plot inside this notebook  
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)
from sklearn.model_selection import train_test_split
from sklearn import *           
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections
import os
import pickle
# import the module related to shrink the size of image
from skimage import transform,data


def add_one(data):
    ones = np.ones(tuple(data.shape[:-1]) + (1,))
    return np.concatenate([ones, data], axis=-1)

# Define the non-linear functions used
def logistic(z): 
    return 1 / (1 + np.exp(-z))
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)

def threshold(z):
    
    #print("z.shape : " + str(z.shape))
    #print("z.type : " + str(type(z)))
    
    z[z < 0] = 0
    return np.sign(z)
    
def relu(z):
    return np.maximum(0, z)

def leaky_relu(z):
    return np.maximum(0.1*z, z)


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
    


# In[126]:


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
        self.W = np.random.randn(n_in+1, n_out) * 0.1
        
    def get_params(self):
        """Return an iterator over the parameters."""
        return self.W
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        self.noise = np.random.randn(*(X.shape[:-1] + (self.W.shape[-1], ))) * self.sigm
        self.input = add_one(X)
        return self.input.dot(self.W) + self.noise 
        
    def get_grad(self, loss):
        """Return a list of gradients over the parameters."""
#         return  (self.input[:, :, np.newaxis].dot(self.noise.reshape(1, self.noise.shape[0])) * loss / self.sigm**2).mean(axis=0)
        return  (np.einsum('abc,abd->abcd',(self.input * loss[:, :, np.newaxis]), self.noise)/(self.sigm**2)).mean(axis=(0, 1))
#         print((np.einsum('abc,abd->abcd',(self.input * loss[:, :, np.newaxis]), self.noise)/(self.sigm**2)).mean(axis=(0, 1))[0])
    
    
    def update_param(self, loss, learning_rate):
        self.W -= learning_rate * self.get_grad(loss)
    


# In[127]:


class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        re = logistic(X)
        #re = threshold(X)
        return re


# In[128]:


class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)
    
    def get_cost(self, Y, T, n_each, train=True):
        """Return the cost at the output of this output layer."""
        if train:
            T  = np.stack([T] * n_each).transpose([1, 0, 2])
            return -np.multiply(T, np.log(Y)).sum(axis=-1)
        else:
            return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]


# In[129]:


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
#     activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
#     X = input_samples
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
#         Y = layer.get_output(X)  # Get the output of the current layer
#         activations.append(Y)  # Store the output for future processing
#         X = activations[-1]  # Set the current input as the activations of the previous layer
#     return activations  # Return the activations of each layer
    return X


# In[130]:


# Define a method to update the parameters
def update_params(layers, loss, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer in layers:
        layer.update_param(loss, learning_rate)


# In[131]:


def flat_trans(x):
    x.resize_(28 * 28) #图片缩放
    return x


# In[132]:

# 定义读取minist数据集的类
class Mnist(object):

    def __init__(self):

        self.dataname = "Mnist"
        self.dims = 28*28
        self.shape = [28 , 28 , 1]
        self.image_size = 28
        self.data, self.data_y = self.load_mnist()

    def load_mnist(self):

        data_dir = os.path.join("", "mnist/raw")
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        # 利用np.fromfile语句将这个ubyte文件读取进来
        # 需要注意的是用np.uint8的格�?        # 还有读取进来的是一个一维向�?        # <type 'tuple'>: (47040016,)，这就是loaded变量的读完之后的数据类型
        loaded = np.fromfile(file=fd , dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 , 28 ,  1)).astype(np.float)
        #'train-images-idx3-ubyte'这个文件前十六位保存的是一些说明具体打印结果如下：
        point = loaded[:16]
        # print(point)
        # [  0   0   8   3   0   0 234  96   0   0   0  28   0   0   0  28]
        # 序号�?开始，上述数字有下面这几个公式的含�?        # MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
        # ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);    等于60000
        # ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12); 等于28
        # ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);等于28

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)


        point = loaded[:8]
        # print(point)
        # [  0   0   8   1   0   0 234  96]
        # 这些数字的作用和上述类似
        # 这些数字的功能之一就是可以判断你下载的数据集对不对，全不全

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 , 28 , 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)

        #目的是为了打乱数据集
        #这里随意固定一个seed，只要seed的值一样，那么打乱矩阵的规律就是一眼的
        seed = 666
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        #convert label to one-hot
        #手动将数据转换成one-hot编码形式
        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, int(y[i])] = 1.0
        return X / 255., y_vec


# mn_object = Mnist()
# XX = mn_object.data.reshape(70000,784)
# YY = mn_object.data_y.reshape(70000,10)

# 读取minist数据集
mn_object = Mnist()
XX2 = mn_object.data.reshape(70000,784)
XX = []
YY = mn_object.data_y.reshape(70000,10)

# change the size of image
for i in range(XX2.shape[0]):
    #(784,) -> (28,28)
    x = XX2[i].reshape(28,28)
    #(28,28) -> (14,14)
    x_little = transform.resize(x, (14, 14))
	#(14,14) -> (196,)
    x_little = x_little.reshape(196)
    XX.append(x_little)
XX = np.array(XX)

# print(type(XX))  # <class 'numpy.ndarray'>
# print(XX.shape)  # (70000,784)
# print(type(YY))  # <class 'numpy.ndarray'>
# print(YY.shape)  # (70000,10)

# print(x[0])
# print(y[0])

# Divide the data into training sets and test sets
X_train, X_validation, T_train, T_validation = train_test_split(
    XX, YY, test_size=0.3)
# print(X_train.shape)#(49000, 784)
# print(X_test.shape)#(360, 64)
# print(T_train.shape)#(49000, 10)
# print(T_test.shape)#(360, 10)

# # Create the minibatches
batch_size = 25
nb_of_batches = int(X_train.shape[0] / batch_size)  # Number of batches=1960  49000=1960*25
# Create batches (X,Y) from the training set
X_batches = np.array_split(X_train, nb_of_batches, axis=0)  # X samples
T_batches = np.array_split(T_train, nb_of_batches, axis=0)  # Y targets

# print(nb_of_batches)
# print(len(X_train))
# print("nb_of_batches = X_train.shape[0] / batch_size : {} = {}/{} ".format(nb_of_batches,X_train.shape[0],batch_size))
# print(X_train.shape)#(49000,784)

sigm = 2
n_each = 10000
# Define a sample model to be trained on the data
hidden_neurons_1 = 20  # Number of neurons in the first hidden-layer
hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
# Create the model
layers = [] # Define a list of layers
# Add first hidden layer
layers.append(LinearLayer(14*14, hidden_neurons_1, sigm, n_each))
layers.append(LogisticLayer())
# Add second hidden layer
# layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2, sigm, n_each))
# layers.append(LogisticLayer())
# Add output layer
layers.append(LinearLayer(hidden_neurons_2, 10, sigm, n_each))
layers.append(SoftmaxOutputLayer())


# Perform backpropagation
# initalize some lists to store the cost for future analysis        
minibatch_costs = []
training_costs = []
validation_costs = []

max_nb_of_iterations = 15 # Train for a maximum of 300 iterations
learning_rate = 0.1  # Gradient descent learning rate

# Train for the maximum number of iterations
for iteration in range(max_nb_of_iterations):
    n_batch = 0
    for X, T in zip(X_batches,T_batches):  # For each minibatch sub-iteration
        n_batch += 1
        Y = forward_step(X, layers, n_each) # Get the activations
        minibatch_cost = layers[-1].get_cost(Y, T, n_each)  # Get cost
        minibatch_costs.append(minibatch_cost.mean())
        update_params(layers, minibatch_cost, learning_rate)  # Update the parameters
        # Get full training cost for future analysis (plots)
        activation = forward_step(X_train, layers, n_each, False)
        train_cost = layers[-1].get_cost(activation, T_train, n_each, False)
        training_costs.append(train_cost)
        # Get full validation cost
        activation = forward_step(X_validation, layers, n_each, False)
        validation_cost = layers[-1].get_cost(activation, T_validation, n_each, False)
        validation_costs.append(validation_cost)
        # print the current loss
        if n_batch%100 == 0:
            print("validation_cost : {}".format(validation_cost))
    
    nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    print("epoch : " + str(nb_of_iterations))

    # Stop training if the cost on the validation set doesn't decrease
    if len(validation_costs) > 3:
        if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3] :
            break

# Get results of test data
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
with open("SDENN_minist_weights_sigmoid.pkl", "wb") as f:
    pickle.dump(weights_dict, f)
print("Finished dumping to disk..")

# plot and save the diagrams of loss
minibatch_x_inds = np.linspace(0, nb_of_batches*nb_of_iterations, num = nb_of_batches*nb_of_iterations)
# iteration_x_inds = np.linspace(0, nb_of_iterations, num = nb_of_iterations) 
# Plot the cost over the iterations
# plt.plot(minibatch_x_inds, minibatch_costs, 'r-', linewidth=2, label='cost minibatches')
plt.plot(minibatch_x_inds, training_costs, 'b-', linewidth=1.5, label='training set')
plt.plot(minibatch_x_inds, validation_costs, 'r--', linewidth=1.5, label='validation set')
# Add labels to the plot
plt.xlabel('iteration', fontsize=15)
plt.ylabel('loss', fontsize=15)
# plt.title('Decrease of cost over backprop iteration')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((0,nb_of_batches*nb_of_iterations,0,6.0))
plt.grid()
#plt.show()
# save figure
plt.savefig("SDENN_minist_sigmoid" + ".png")
plt.close()





# In[158]:


# # Get results of test data
# y_true = np.argmax(T_test, axis=1)  # Get the target outputs
# activation = forward_step(X_test, layers, n_each, False)  # Get activation of test samples
# y_pred = np.argmax(activation, axis=1)  # Get the predictions made by the network
# test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
# print('The accuracy on the test set is {:.2f}'.format(test_accuracy))


# In[ ]:


# # Plot the minibatch, full training set, and validation costs
# minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
# iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
# # Plot the cost over the iterations
# plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
# plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
# plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
# # Add labels to the plot
# plt.xlabel('iteration')
# plt.ylabel('$\\xi$', fontsize=15)
# plt.title('Decrease of cost over backprop iteration')
# plt.legend()
# x1,x2,y1,y2 = plt.axis()
# plt.axis((0,nb_of_iterations,0,2.5))
# plt.grid()
# plt.show()


# In[ ]:


# # Show confusion table
# conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
# # Plot the confusion table
# class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # Show class labels on each axis
# ax.xaxis.tick_top()
# major_ticks = range(0,10)
# minor_ticks = [x + 0.5 for x in range(0, 10)]
# ax.xaxis.set_ticks(major_ticks, minor=False)
# ax.yaxis.set_ticks(major_ticks, minor=False)
# ax.xaxis.set_ticks(minor_ticks, minor=True)
# ax.yaxis.set_ticks(minor_ticks, minor=True)
# ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
# ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
# # Set plot labels
# ax.yaxis.set_label_position("right")
# ax.set_xlabel('Predicted label')
# ax.set_ylabel('True label')
# fig.suptitle('Confusion table', y=1.03, fontsize=15)
# # Show a grid to seperate digits
# ax.grid(b=True, which=u'minor')
# # Color each grid cell according to the number classes predicted
# ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
# # Show the number of samples in each cell
# for x in xrange(conf_matrix.shape[0]):
#     for y in xrange(conf_matrix.shape[1]):
#         color = 'w' if x == y else 'k'
#         ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       
# plt.show()

