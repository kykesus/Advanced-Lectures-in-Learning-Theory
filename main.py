#!/usr/bin/python

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import random


def simple_pca(X=None, dim=0, npca=0):
    if X is None:
        return None
    # centering the data
    X -= np.mean(X, axis=dim)
    if dim != 0:
        X = X.transpose()
    # Each column is different variable (pixel), and rows indicates observations (images)
    covariance_matrix = np.cov(X, rowvar=False)
    # Decompose using eigen value decomposition
    egn_val, egn_vec = linalg.eigh(covariance_matrix)
    # Sort eigen vectors based on eigenvalues
    sorting_indices = np.argsort(egn_val)[::-1]
    egn_vec = egn_vec[:, sorting_indices]
    egn_val = egn_val[sorting_indices]
    # Return only first Npca vectors
    return egn_vec[:, :npca], egn_val[:npca]


def hinge_loss_grad(target, score, input_vec):
    ty = target * score
    txi = target * input_vec
    grad = np.zeros(input_vec.shape)
    if ty < 1:
        grad = -txi
    return grad


def hinge_loss(target, score):
    '''
    Function for calculation the hinge loss of "score" versus "target"
    '''
    ty = target * score
    return np.max([0, 1 - ty])


def gd_optimizer(data, target, w_shape, initial_variance, eta, epochs):

    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)

    n_points = data.shape[0]

    # Initialize loss vector for each epoch
    loss_list = np.zeros(epochs,)

    x = data
    t = target

    for epoc in range(epochs):
        score = np.dot(x, wt).squeeze()

        for ii in range(n_points):
            out = eta*hinge_loss_grad(t[ii], score[ii], x[ii, :])/n_points
            wt -= out.reshape(-1, 1)
            loss_list[epoc] += hinge_loss(t[ii], score[ii])/n_points
        #print(epoc/epochs)

    return wt, loss_list

def rgd_optimizer(data, target, w_shape, initial_variance, eta ,lambda_coeff, epochs):

    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)

    n_points = data.shape[0]

    # Initialize loss vector for each epoch
    loss_list = np.zeros(epochs,)

    x = data
    t = target

    for epoc in range(epochs):
        score = np.dot(x, wt).squeeze()
        loss_term = 0
        for ii in range(n_points):
            if ii == 0:
                out = eta*hinge_loss_grad(t[ii], score[ii], x[ii, :])/n_points
            else:
                out += eta*hinge_loss_grad(t[ii], score[ii], x[ii, :])/n_points
            loss_term += hinge_loss(t[ii], score[ii])/n_points
        wt -= out.reshape(-1, 1) + (2 * lambda_coeff * wt)
        loss_list[epoc] = loss_term + lambda_coeff*np.sum(wt**2)
        
        #print(epoc/epochs)

    return wt, loss_list

def test_model(model,data,target):
    score = np.dot(data, model).squeeze() > 0
    target = np.array(target) > 0
    return np.sum(score == target)/len(target)

def sgd_optimizer(data, target, w_shape, initial_variance, eta, epochs):
    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)

    n_points = data.shape[0]

    # Initialize loss vector for each epoch
    loss_list = np.zeros(epochs,)

    x = data
    t = target
    n_idx = [i for i in range(n_points)]
    random.shuffle(n_idx)

    for epoc in range(epochs):
        ii = n_idx[epoc]
        score = np.dot(x[ii,:], wt).squeeze()
        out = eta*hinge_loss_grad(t[ii], score, x[ii, :])
        wt -= out.reshape(-1, 1)
        loss_list[epoc] = hinge_loss(t[ii], score)
        #print(epoc/epochs)

    return wt, loss_list

def is_prime(x):
    '''
    Map labels to 1/-1 values depending if label is prime
    '''
    new_lbl = [1 if lbl in [2, 3, 5, 7] else -1 for lbl in x]
    return np.array(new_lbl)


def main():
    # (Down)load MNIST dataset
    # We use tensor transform
    dataset_train = MNIST('./data/', train=True,
                          download=True, transform=ToTensor())
    dataset_test = MNIST('./data/', train=False,
                         download=True,
                         transform=ToTensor())

    # Extract images and labels
    img_train = dataset_train.data
    img_test = dataset_test.data
    lbl_train = dataset_train.targets
    lbl_test = dataset_test.targets

    # Merge train & test
    img_combined = torch.cat((img_train, img_test), dim=0)
    lbl_combined = torch.cat((lbl_train, lbl_test), dim=0)

    # Transform to numpy arrays
    img_combined = img_combined.numpy()
    lbl_combined = lbl_combined.numpy()
    img_combined = img_combined.reshape(img_combined.shape[0], 28**2)
    img_combined = img_combined.astype(float)

    # Apply preprocessing - we decided to use PCA
    # This preprocessing keeps the input for the next stage convex
    ###
    if 0:
        egn_vec, egn_val = simple_pca(img_combined, dim=0, npca=28**2)
        fig, ax = plt.subplots()
        ax.plot(egn_val)
        ax.grid()
        ax.set_xlabel('Eigenvalue index')
        ax.set_ylabel('Score')
        fig.show()

    egn_vec, egn_val = simple_pca(img_combined, dim=0, npca=100)
    img_combined = np.dot(img_combined, egn_vec)

    # Define the binary classification problems
    # We create 3 sets of labels with value +1/-1 for the binary classification with hinge loss.
    target_greater_than_5 = [1 if (lbl >= 5) else -1 for lbl in lbl_combined]
    target_even = [1 if (lbl % 2 == 0) else -1 for lbl in lbl_combined]
    target_prime = is_prime(lbl_combined)

    # Discuss and definitions for different gradients
    # Part A - Optimization
    epochs = int(1e1)  # number of epochs to run for
    eta = 3e-4  # just from trial and error not theoretic justified...

    # Run learning
    '''
    Simple GD
    '''
    print('Simple GD')
    print('#'*10 + 'Bigger than 5' + '#'*10)
    wt_5, loss_list_5 = gd_optimizer(data=img_combined, target=target_greater_than_5,
                                     w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + 'Number is even' + '#'*10)
    wt_ev, loss_list_ev = gd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + 'Number is prime' + '#'*10)
    wt_p, loss_list_p = gd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5)
    ax.plot(loss_list_ev)
    ax.plot(loss_list_p)
    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error (%)')
    ax.set_title('GD')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5,EVEN,PRIME = test_model(wt_5,img_combined,target_greater_than_5) , test_model(wt_ev,img_combined,target_even) , test_model(wt_p,img_combined,target_prime)  
    print('GD correctness: GT5: {} Even {} Prime {}'.format(GT5,EVEN,PRIME))

    '''
    Regularized GD
    '''
    print('Regularized GD')
    lambda_coeff = 1e-1
    print('#'*10 + 'Bigger than 5' + '#'*10)
    wt_5, loss_list_5 = rgd_optimizer(data=img_combined, target=target_greater_than_5,
                                     w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=lambda_coeff, epochs=epochs)

    print('#'*10 + 'Number is even' + '#'*10)
    wt_ev, loss_list_ev = rgd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=lambda_coeff, epochs=epochs)

    print('#'*10 + 'Number is prime' + '#'*10)
    wt_p, loss_list_p = rgd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=lambda_coeff, epochs=epochs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5)
    ax.plot(loss_list_ev)
    ax.plot(loss_list_p)
    ax.grid()
    ax.set_title('RGD')
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error (%)')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5,EVEN,PRIME = test_model(wt_5,img_combined,target_greater_than_5) , test_model(wt_ev,img_combined,target_even) , test_model(wt_p,img_combined,target_prime)  
    print('RGD correctness: GT5: {} Even {} Prime {}'.format(GT5,EVEN,PRIME))
    '''
    SGD
    '''
    print('SGD')
    eta = 1e-6
    epochs = int(5e4)
    print('#'*10 + 'Bigger than 5' + '#'*10)
    wt_5, loss_list_5 = sgd_optimizer(data=img_combined, target=target_greater_than_5,
                                     w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + 'Number is even' + '#'*10)
    wt_ev, loss_list_ev = sgd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + 'Number is prime' + '#'*10)
    wt_p, loss_list_p = sgd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5)
    ax.plot(loss_list_ev)
    ax.plot(loss_list_p)
    ax.grid()
    ax.set_title('SGD')
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error (%)')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5,EVEN,PRIME = test_model(wt_5,img_combined,target_greater_than_5) , test_model(wt_ev,img_combined,target_even) , test_model(wt_p,img_combined,target_prime)  
    print('SGD correctness: GT5: {} Even {} Prime {}'.format(GT5,EVEN,PRIME))




    # Part B - Generalization
    # Use randomized train & test sets

    input("Press any key to continue")


if __name__ == "__main__":
    main()
