#!/usr/bin/python

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def main():

    # (Down)load MNIST dataset
    # We use tensor transform
    dataset_train = MNIST('./data/', train=True,
                          download=True, transform=ToTensor())
    dataset_test = MNIST('./data/', train=False,
                         download=True, transform=ToTensor())

    # Merge train & test
    dataset = ConcatDataset([dataset_train, dataset_test])

    # Load the samples to memory
    # TODO - should we shuffle?
    data = DataLoader(dataset, batch_size=len(dataset))

    # Apply preprocessing - we decided to use PCA
    # This preprocessing keeps the input for the next stage convex
    ###
    # Preprocessing
    ###

    # Define the binary classification problems

    # Determine the loss function to work with

    # Create true labels for the problems

    # Discuss and definitions for different gradients

    # Part A - Optimization
    # Run learning, plot figures and whatever...

    # Part B - Generalization
    # Use randomized train & test sets


if __name__ == "__main__":
    main()
