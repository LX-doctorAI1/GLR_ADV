import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_ratio=0.1,
                           shuffle=True,
                           download=True):

    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_valid = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_set = datasets.MNIST(root=data_dir, train=True, transform=transform_train, download=download)
    valid_set = datasets.MNIST(root=data_dir, train=True, transform=transform_valid, download=False)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_ratio * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        sampler=train_sampler,num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size,
        sampler=valid_sampler)
    return train_loader, valid_loader


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    download=True):

    # define transform
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=download)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle)

    return data_loader
