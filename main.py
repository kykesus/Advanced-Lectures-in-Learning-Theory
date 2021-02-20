#!/usr/bin/python

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

def main():

    # (Down)load MNIST dataset
    # We use tensor transfrom 
    dataset_train = MNIST('./data/', train=True, download=True, transform=ToTensor())
    dataset_test = MNIST('./data/', train=False, download=True, transform=ToTensor())

    # Load the samples to memory
    data_train = DataLoader(dataset_train)
    data_test = DataLoader(dataset_test)

    # Merge train & test data
    data = ?
    labels = ?

if __name__ == "__main__":
    main()
