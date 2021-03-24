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
from imagecorruptions import corrupt
import skimage as sk
from skimage.filters import gaussian

device = torch.device('cuda:4')


def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(28 - c[1], c[1], -1):
            for w in range(28 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


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

    noise = ['gaussian', 'impulse', 'glass_blur', 'contrast']
    noise_function = [gaussian_noise, impulse_noise, glass_blur, contrast]
    strength = [1, 2, 3, 4, 5]
    epsilon = config.epsilon
    for i in range(len(noise)):
        for j in range(len(strength)):
            noises = []
            y_preds = []
            y_preds_adversarial = []
            totalMisclassifications = 0
            xs_clean = []
            y_trues_clean = []
            num_adv = 0
            for x, y_true in tqdm(zip(xs, y_trues)):

                # Wrap x as a variable
                xs_clean.append(np.array(x))
                xv = torch.Tensor(x.reshape(1, 28 * 28))
                # xv = nn.Parameter(torch.FloatTensor(x.reshape(1, 28 * 28)), requires_grad=True)
                y_true = Variable(torch.LongTensor(np.array([y_true])), requires_grad=False)
                # Classification before Adv
                y_pred = np.argmax(net(xv).data.numpy())

                # Generate Adversarial Image
                xv = xv.numpy() * 255
                xv = xv.reshape(28, 28).astype(np.float)
                xv = noise_function[i](xv, strength[j]).astype(np.float)
                xv /= 255
                xv = xv.reshape(1, 28 * 28)
                xv = torch.from_numpy(xv).float()
                # method = optim.LBFGS(list(xv), lr=1e-1)
                # Add perturbation
                # x_grad = torch.sign(x.grad.data)
                x_adversarial = torch.clamp(xv, 0, 1)

                # Classification after optimization
                y_pred_adversarial = np.argmax(net(Variable(x_adversarial)).data.numpy())
                # print "Before: {} | after: {}".format(y_pred, y_pred_adversarial)

                # print "Y_TRUE: {} | Y_PRED: {}".format(_y_true, y_pred)
                if y_true.data.numpy() != y_pred:
                    # print("WARNING: MISCLASSIFICATION ERROR")
                    totalMisclassifications += 1
                else:
                    if y_pred_adversarial != y_pred:
                        num_adv += 1
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append(x_adversarial.numpy())
                y_trues_clean.append(y_true.data.numpy())

            print("Total totalMisclassifications :{}/{} ".format(totalMisclassifications, len(xs)))  # 1221/1797
            print("the amount of adv samples is : {}".format(num_adv))  # 576

            print("Successful!!")

            with open("Generate_adversarial_sample_epsilon=" + str(epsilon) + "by_{}_{}.pkl".format(noise[i],
                                                                                                    strength[j]),
                      "wb") as f:
                adv_data_dict2 = {
                    "xs": xs_clean,
                    "y_trues": y_trues_clean,
                    "y_preds": y_preds,
                    "noises": noises,
                    "y_preds_adversarial": y_preds_adversarial
                }
                pickle.dump(adv_data_dict2, f, protocol=3)
            print("{}-{} Successful!!".format(noise[i], strength[j]))
