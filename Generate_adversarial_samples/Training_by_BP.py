import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from tqdm import *
from skimage import transform, data
import pickle

# define nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # my network is composed of only affine layers
        self.fc1 = nn.Linear(14 * 14, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DATA LOADERS
def flat_trans(x):
    x.resize_(28 * 28) #图片缩放
    return x

if __name__ == '__main__':
    net = Net()
    # define the loss function
    SoftmaxWithXent = nn.CrossEntropyLoss()
    # define optimization algorithm
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04)
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(flat_trans)])
    # load the data
    traindata   = torchvision.datasets.MNIST(root="../mnist", train=True, download=False, transform = mnist_transform)
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers = 2)
    testdata    = torchvision.datasets.MNIST(root="../mnist", train=False, download=False, transform = mnist_transform)
    testloader  = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers = 2)
    print("trainloader.length : {}".format(len(trainloader))) #235 batches

    # strat to train the nn
    for epoch in range(100):
        print("Epoch: {}".format(epoch))
        running_loss = 0.0
        # import ipdb; ipdb.set_trace()
        for data in tqdm(trainloader):
            # get the inputs
            inputs, labels = data

            # transform the size of sample : (28,28) -> (14,14)
            inputs = inputs.numpy()
            num_of_sample_in_batch = inputs.shape[0] #获得当前batch中的样本数量,不是所有的都是256
            inp = inputs.reshape(num_of_sample_in_batch,28,28)
            inp_little = transform.resize(inp, (num_of_sample_in_batch,14, 14))
            inp_little = inp_little.reshape(num_of_sample_in_batch,196) #14*14=196
            inputs = torch.from_numpy(inp_little)

            # wrap them in a variable
            inputs, labels = Variable(inputs.float()), Variable(labels)
            # zero the gradients
            optimizer.zero_grad()

            # forward + loss + backward
            outputs = net(inputs)  # forward pass
            loss = SoftmaxWithXent(outputs, labels)  # compute softmax -> loss
            loss.backward()  # get gradients on params
            optimizer.step()  # SGD update

            # print statistics
            running_loss += loss.data.numpy()

        print('Epoch: {} | Loss: {}'.format(epoch, running_loss / 235.0))
    print("Finished Training")

    # TEST
    correct = 0.0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.numpy()
        num_of_sample_in_batch = images.shape[0]
        inp = images.reshape(num_of_sample_in_batch, 28, 28)
        inp_little = transform.resize(inp, (num_of_sample_in_batch, 14, 14))
        inp_little = np.array(inp_little.reshape(num_of_sample_in_batch, 196))
        images = torch.from_numpy(inp_little)

        outputs = net(Variable(images.float()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("test set Accuracy: {}".format(float(correct) / total))

    print("Dumping weights to disk")
    weights_dict = {}
    # save the weights of nn with protocol=2
    for param in list(net.named_parameters()):
        print("Serializing Param", param[0])
        weights_dict[param[0]] = param[1]
    with open("weights_size=14_protocol=2.pkl", "wb") as f:
        import pickle
        pickle.dump(weights_dict, f, protocol=2)

    weights_dict2 = {}
    # save the weights of nn with protocol=3
    for param in list(net.named_parameters()):
        print("Serializing Param", param[0])
        weights_dict2[param[0]] = param[1]
    with open("weights_size=14_protocol=3.pkl", "wb") as f2:
        pickle.dump(weights_dict2, f2, protocol=3)
    print("Finished dumping to disk..")










