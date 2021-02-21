#!/usr/bin/python

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def simple_pca(X = None,dim = 0,npca = 0):
    if X is None:
        return None
    # centering the data
    X -= np.mean(X,axis=dim)
    if dim != 0:
        X = X.transpose()
    # Each column is different variable (pixel), and rows indicates observations (images)
    covariance_matrix = np.cov(X,rowvar=False)
    # Decompose using eigen value decomposition
    egn_val,egn_vec = linalg.eigh(covariance_matrix)
    # Sort eigen vectors based on eigenvalues
    sorting_indices = np.argsort(egn_val)[::-1]
    egn_vec = egn_vec[:,sorting_indices]
    egn_val = egn_val[sorting_indices]
    # Return only first Npca vectors
    return egn_vec[:,:npca],egn_val[:npca]


def hinge_loss_grad(target,score,input_vec):
    ty = target*score
    txi = target*input_vec
    grad = np.zeros(input_vec.shape)
    if ty<1:
        grad = -txi
    return grad

def hinge_loss(target,score):
    ty = target*score
    return np.max([0,1-ty])

debug = False


def gd_optimizer(data,target,w_shape,initial_variance,eta,epochs):
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape,1)
    n_points = data.shape[0]
    loss_list = np.zeros(epochs,)
    x = data
    t = target
    for epoc in range(epochs):
        score = np.dot(x,wt).squeeze()
        for ii in range(n_points):
            out = eta*hinge_loss_grad(t[ii],score[ii],x[ii,:])/n_points
            wt -= out.reshape(-1,1)
            loss_list[epoc] += hinge_loss(t[ii],score[ii])/n_points
        print(epoc/epochs)
    return wt,loss_list

def is_prime(x):
    new_lbl = []
    for xx in x:
        if xx in [1,2,3,5,7]:
            new_lbl.append(1)
        else:
            new_lbl.append(-1)
    return np.array(new_lbl)

def main():
    import matplotlib.pyplot as plt
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
    img_combined = torch.cat((img_train,img_test),dim=0)
    lbl_combined = torch.cat((lbl_train,lbl_test),dim=0)
    # Transform to numpy arrays
    img_combined = img_combined.numpy()
    lbl_combined = lbl_combined.numpy()
    img_combined = img_combined.reshape(img_combined.shape[0],28**2)
    img_combined = img_combined.astype(float)
    # Apply preprocessing - we decided to use PCA
    # This preprocessing keeps the input for the next stage convex
    ###
    if 0:
        egn_vec,egn_val = simple_pca(img_combined,dim = 0,npca = 28**2)
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.plot(egn_val)
        ax.grid()
        ax.set_xlabel('Eigen value index')
        ax.set_ylabel('Score')
        fig.show()
    egn_vec,egn_val = simple_pca(img_combined,dim = 0,npca = 100)
    img_combined = np.dot(img_combined,egn_vec)
    # Define the binary classification problems
    target_greater_than_5 = (lbl_combined > 5) * 2 - 1
    target_even = (lbl_combined % 2) * 2 - 1
    target_prime = is_prime(lbl_combined)
    # Discuss and definitions for different gradients
    # Part A - Optimization
    epochs = int(1e2) # number of epochs to run for
    eta = 3e-4 # just from trial and error not theoretic justified...
    # Run learning, plot figures and whatever...
    fig,ax = plt.subplots()
    print('#'*10 + 'Bigger than 5' + '#'*10)
    wt_5,loss_list_5 = gd_optimizer(data=img_combined,target=target_greater_than_5,w_shape=100,initial_variance=1/100,eta=eta,epochs=epochs)
    print('#'*10 + 'Number is even' + '#'*10)
    wt_ev,loss_list_ev = gd_optimizer(data=img_combined,target=target_even,w_shape=100,initial_variance=1/100,eta=eta,epochs=epochs)
    print('#'*10 + 'Number is prime' + '#'*10)
    wt_p,loss_list_p = gd_optimizer(data=img_combined,target=target_prime,w_shape=100,initial_variance=1/100,eta=eta,epochs=epochs)
    ax.plot(loss_list_5)
    ax.plot(loss_list_ev)
    ax.plot(loss_list_p)
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error')
    ax.legend(['Bigger than 5','Number is even','Number is prime'])
    fig.show()
    # Part B - Generalization
    # Use randomized train & test sets


if __name__ == "__main__":
    main()
