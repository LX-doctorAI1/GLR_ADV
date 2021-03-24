import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable
from olds import dataLoader

device = torch.device('cuda:4')

def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = Network()
    model_static = torch.load('./mnist_model/blackBP.pth', map_location='cpu')
    net.load_state_dict(model_static)
    SoftmaxWithXent = nn.CrossEntropyLoss()
    batch_size = 128
    transform = transforms.Compose([transforms.ToTensor()])
    data_dir = './'
    tranform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=tranform)
    val_dataset = torchvision.datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=tranform)

    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(dataset=val_dataset, batch_size=128, num_workers=0, shuffle=False)
    train_dataloader, test_dataloader = dataLoader.LoadMNIST('../data/MNIST', tranform, batch_size, False)
    xs = []
    y_trues = []
    for data in tqdm(test_dataloader):
        inputs, labels = data
        inputs = inputs.reshape((len(inputs), -1))
        if len(xs) == 0:
            xs = inputs
            y_trues = labels
        else:
            xs = torch.cat([xs, inputs], dim=0)
            y_trues = torch.cat([y_trues, labels], dim=0)
    xs = np.array(xs)
    y_trues = np.array(y_trues).reshape(-1)

    noises = []
    y_preds = []
    y_preds_adversarial = []
    totalMisclassifications = 0
    xs_clean = []
    y_trues_clean = []
    num_adv = 0
    import config

    epsilon = config.epsilon

    for x, y_true in tqdm(zip(xs, y_trues)):

        # Wrap x as a variable
        x = Variable(torch.FloatTensor(x.reshape(1, 28*28)), requires_grad=True)
        y_true = Variable(torch.LongTensor(np.array([y_true])), requires_grad=False)

        # Classification before Adv
        y_pred = np.argmax(net(x).data.numpy())

        # Generate Adversarial Image
        # Forward pass
        outputs = net(x)
        loss = SoftmaxWithXent(outputs, y_true)
        loss.backward()  # obtain gradients on x

        # Add perturbation
        x_grad = torch.sign(x.grad.data)
        x_adversarial = torch.clamp(x.data + epsilon * x_grad, 0, 1)

        # Classification after optimization
        y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).data.numpy())
        # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

        # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
        if y_true.data.numpy() != y_pred:
            print("WARNING: MISCLASSIFICATION ERROR")
            totalMisclassifications += 1
        else:
            if y_pred_adversarial != y_pred:
                num_adv += 1
        y_preds.append(y_pred)
        y_preds_adversarial.append(y_pred_adversarial)
        noises.append(x_adversarial.numpy())
        xs_clean.append(x.data.numpy())
        y_trues_clean.append(y_true.data.numpy())

    print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
    print("the amount of adv samples is : {}".format(num_adv))  # 576

    print("Successful!!")

    with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_FGSM.pkl", "wb") as f:
        adv_data_dict2 = {
            "xs": xs_clean,
            "y_trues": y_trues_clean,
            "y_preds": y_preds,
            "noises": noises,
            "y_preds_adversarial": y_preds_adversarial
        }
        pickle.dump(adv_data_dict2, f, protocol=3)
    print("Successful!!")