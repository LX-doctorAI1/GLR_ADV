from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import os


def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


