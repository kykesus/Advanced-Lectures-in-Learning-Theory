#!/usr/bin/python

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import random


def list_reduction(a, b):
    return list(set(a)-set(b))


def simple_pca(X=None, dim=0, npca=0):
    if X is None:
        return None
    # centering the data
    X -= np.mean(X, axis=dim)
    if dim != 0:
        X = X.transpose()
    # Each column is different variable (pixel), and rows indicates observations (images)
    covariance_matrix = np.cov(X, rowvar=False)
    # Decompose using eigenvalue decomposition
    egn_val, egn_vec = linalg.eigh(covariance_matrix)
    # Sort eigenvectors based on eigenvalues
    sorting_indices = np.argsort(egn_val)[::-1]
    egn_vec = egn_vec[:, sorting_indices]
    egn_val = egn_val[sorting_indices]
    # Return only first Npca vectors
    return egn_vec[:, :npca], egn_val[:npca]


def hinge_loss_grad(target, ty, input_vec):
    if ty < 1:
        txi = target * input_vec
        return -txi

    return np.zeros(input_vec.shape)


def hinge_loss_grad_vec(target, ty, input_vec):
    res = -input_vec.transpose()*target.reshape(-1)
    label_condition = np.array(ty < 1)
    res = res*label_condition
    return res


def hinge_loss(ty):
    '''
    Function for calculation the hinge loss of "score" versus "target"
    '''
    return np.max([0, 1 - ty])


def project(wt, R):
    '''
    Project a vector wt onto a sphere of radius R
    '''
    norm = np.sqrt(np.dot(wt.transpose(), wt))
    if norm > R:
        return wt * R / norm
    else:
        return wt


def gd_optimizer(data, target, w_shape, initial_variance, eta, epochs, test_data=None, test_target=None):

    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)
    target = np.array(target)
    data = np.array(data)
    n_points = data.shape[0]

    test_flag = np.all(test_data != None)
    # Initialize loss vector for each epoch
    if test_flag:
        loss_list = np.zeros([epochs, 3])
    else:
        loss_list = np.zeros([epochs, 1])
    x = data
    t = target

    for epoc in range(epochs):
        score = np.dot(x, wt).squeeze()
        ty = t*score
        out = (np.sum(hinge_loss_grad_vec(t, ty, x), 1)).reshape(-1, 1)
        wt -= eta*out/n_points
        loss_list[epoc, 0] = np.sum(hinge_loss_vec(ty))/n_points
        if test_flag:
            h_loss, zone_loss = test_model(wt, test_data, test_target)
            loss_list[epoc, 1] = h_loss
            loss_list[epoc, 2] = zone_loss
    return wt, loss_list


def cgd_optimizer(data, target, w_shape, initial_variance, eta, R, epochs, test_data=None, test_target=None):
    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)
    target = np.array(target)
    data = np.array(data)
    n_points = data.shape[0]

    test_flag = np.all(test_data != None)
    # Initialize loss vector for each epoch
    if test_flag:
        loss_list = np.zeros([epochs, 3])
    else:
        loss_list = np.zeros([epochs, 1])

    x = data
    t = target

    for epoc in range(epochs):
        score = np.dot(x, wt).squeeze()
        ty = t*score
        out = (np.sum(hinge_loss_grad_vec(t, ty, x), 1)).reshape(-1, 1)
        wt -= eta*out/n_points
        loss_list[epoc, 0] = np.sum(hinge_loss_vec(ty))/n_points
        wt = project(wt, R)
        if test_flag:
            h_loss, zone_loss = test_model(wt, test_data, test_target)
            loss_list[epoc, 1] = h_loss
            loss_list[epoc, 2] = zone_loss

    return wt, loss_list


def rgd_optimizer(data, target, w_shape, initial_variance, eta, lambda_coeff, epochs, test_data=None, test_target=None):

    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)
    target = np.array(target)
    data = np.array(data)
    n_points = data.shape[0]

    test_flag = np.all(test_data != None)
    # Initialize loss vector for each epoch
    if test_flag:
        loss_list = np.zeros([epochs, 3])
    else:
        loss_list = np.zeros([epochs, 1])

    x = data
    t = target

    for epoc in range(epochs):
        score = np.dot(x, wt).squeeze()
        ty = t*score
        out = (np.sum(hinge_loss_grad_vec(t, ty, x), 1)).reshape(-1, 1)
        out = eta*out/n_points
        wt -= out.reshape(-1, 1) + (2 * lambda_coeff * wt)
        loss_list[epoc, 0] = np.sum(hinge_loss_vec(
            ty))/n_points + lambda_coeff*np.sum(wt**2)
        if test_flag:
            h_loss, zone_loss = test_model(wt, test_data, test_target)
            loss_list[epoc, 1] = h_loss
            loss_list[epoc, 2] = zone_loss

    return wt, loss_list


def hinge_loss_vec(ty):
    tylen = ty.shape[0]
    res = 1-ty
    mat = np.concatenate((np.zeros([tylen, 1]), res.reshape(-1, 1)), axis=1)
    r = np.max(mat, 1)
    return r


def test_model(model, data, target):
    score = np.dot(data, model).squeeze()
    n_points = data.shape[0]
    loss_hinge = np.sum(hinge_loss_vec(target*score))/n_points
    pd = (target == np.sign(score))
    return loss_hinge, 1 - np.sum(pd)/len(target)


def sgd_optimizer(data, target, w_shape, initial_variance, eta, epochs, test_data=None, test_target=None):
    # Initialize weights
    wt = np.sqrt(initial_variance)*np.random.randn(w_shape, 1)
    target = np.array(target)
    data = np.array(data)
    n_points = data.shape[0]

    test_flag = np.all(test_data != None)
    # Initialize loss vector for each epoch
    if test_flag:
        loss_list = np.zeros([epochs, 3])
    else:
        loss_list = np.zeros([epochs, 1])

    x = data
    t = target
    n_idx = [i for i in range(n_points)]
    random.shuffle(n_idx)

    for epoc in range(epochs):
        ii = n_idx[epoc]
        score = np.dot(x[ii, :], wt).squeeze()
        ty = t[ii] * score
        out = eta*hinge_loss_grad(t[ii], ty, x[ii, :])
        wt -= out.reshape(-1, 1)
        loss_list[epoc, 0] = hinge_loss(ty)
        if test_flag:
            h_loss, zone_loss = test_model(wt, test_data, test_target)
            loss_list[epoc, 1] = h_loss
            loss_list[epoc, 2] = zone_loss
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
                         download=True, transform=ToTensor())

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
    egn_vec, egn_val = simple_pca(img_combined, dim=0, npca=100)
    img_combined = np.dot(img_combined, egn_vec)

    # Define the binary classification problems
    # We create 3 sets of labels with value +1/-1 for the binary classification with hinge loss.
    target_greater_than_5 = [1 if (lbl >= 5) else -1 for lbl in lbl_combined]
    target_even = [1 if (lbl % 2 == 0) else -1 for lbl in lbl_combined]
    target_prime = is_prime(lbl_combined)

    # Part A - Optimization

    # ############################################################################
    # # Simple GD
    # ############################################################################

    epochs = int(1e2)  # number of epochs to run for
    eta = 3e-4  # just from trial and error not theoretic justified...

    print('Simple GD')
    print('#'*10 + ' Bigger than 5 ' + '#'*10)
    theo_wt_5_gd, loss_list_5_gd = gd_optimizer(data=img_combined, target=target_greater_than_5,
                                                w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + ' Number is even ' + '#'*10)
    theo_wt_ev_gd, loss_list_ev_gd = gd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + ' Number is prime ' + '#'*10)
    theo_wt_p_gd, loss_list_p_gd = gd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    # # Plot all binary problems with theoretical parameters
    fig, ax = plt.subplots()
    ax.plot(loss_list_5_gd)
    ax.plot(loss_list_ev_gd)
    ax.plot(loss_list_p_gd)
    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('GD')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5, EVEN, PRIME = test_model(theo_wt_5_gd, img_combined, target_greater_than_5), test_model(
        theo_wt_ev_gd, img_combined, target_even), test_model(theo_wt_p_gd, img_combined, target_prime)
    print('GD results (hinge loss, binary loss):\nGT5: {}\nEven {}\nPrime {}'.format(
        GT5, EVEN, PRIME))

    # Try other parameters values for gt5 problem
    best_loss_gd = 1e6
    best_wts_gd = np.zeros([100, 1])
    best_factor_gd = 1
    fig, ax = plt.subplots()
    for x in np.append(np.linspace(-10, 10, 9), [-90, -50, 50, 100]):
        proposed_eta = eta * (1 + (x / 100))
        proposed_wt_5, proposed_loss_list_5 = gd_optimizer(data=img_combined, target=target_greater_than_5,
                                                           w_shape=100, initial_variance=1/100, eta=proposed_eta, epochs=epochs)
        min_epoch = np.argmin(proposed_loss_list_5)
        if (current_best := proposed_loss_list_5[min_epoch]) < best_loss_gd:
            best_loss_gd = current_best
            best_wts_gd = proposed_wt_5[min_epoch]
            best_factor_gd = (1 + (x / 100))
        ax.plot(proposed_loss_list_5)

    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('GD - Greater than 5')
    ax.legend([f'{100 + x}% * Theoretical' for x in np.append(
        np.linspace(-10, 10, 9), [-90, -50, 50, 100])])
    fig.show()

    print(
        f'Best loss achieved for {best_factor_gd} * theoretical values. Loss: {best_loss_gd}')

    ############################################################################
    # Constrained GD
    ############################################################################

    # We assume that the weights are contained in a ball with radius X
    R = 1e-2

    # The subgradient for hinge loss is bounded with L=1
    L = 1

    # Theoretical eta for error at most RL/sqrt(T)
    eta = R / (L * np.sqrt(epochs))

    print('Constrained GD')
    print('#'*10 + ' Bigger than 5 ' + '#'*10)
    theo_wt_5_cgd, loss_list_5_cgd = cgd_optimizer(
        data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=eta, R=R, epochs=epochs)

    print('#'*10 + ' Number is even ' + '#'*10)
    theo_wt_ev_cgd, loss_list_ev_cgd = cgd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, R=R, epochs=epochs)

    print('#'*10 + ' Number is prime ' + '#'*10)
    theo_wt_p_cgd, loss_list_p_cgd = cgd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, R=R, epochs=epochs)

    fig.show()
    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5_cgd)
    ax.plot(loss_list_ev_cgd)
    ax.plot(loss_list_p_cgd)
    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('CGD')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5, EVEN, PRIME = test_model(theo_wt_5_cgd, img_combined, target_greater_than_5), test_model(
        theo_wt_ev_cgd, img_combined, target_even), test_model(theo_wt_p_cgd, img_combined, target_prime)
    print('CGD results (hinge loss, binary loss):\nGT5: {}\nEven {}\nPrime {}'.format(
        GT5, EVEN, PRIME))

    # Try other parameters values for gt5 problem
    best_loss_cgd = 1e6
    best_wts_cgd = np.zeros([100, 1])
    best_factor_cgd = 1
    fig, ax = plt.subplots()
    for x in np.append(np.linspace(-10, 10, 9), [-90, -50, 50, 100]):
        proposed_eta = eta * (1 + (x / 100))
        proposed_R = R * (1 + (x / 100))
        proposed_wt_5, proposed_loss_list_5 = cgd_optimizer(
            data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=proposed_eta, R=proposed_R, epochs=epochs)
        min_epoch = np.argmin(proposed_loss_list_5)
        if (current_best := proposed_loss_list_5[min_epoch]) < best_loss_cgd:
            best_loss_cgd = current_best
            best_wts_cgd = proposed_wt_5[min_epoch]
            best_factor_cgd = (1 + (x / 100))
        ax.plot(proposed_loss_list_5)

    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('CGD - Greater than 5')
    ax.legend([f'{100 + x}% * Theoretical' for x in np.append(
        np.linspace(-10, 10, 9), [-90, -50, 50, 100])])
    fig.show()

    print(
        f'Best loss achieved for {best_factor_cgd} * theoretical values. Loss: {best_loss_cgd}')

    ############################################################################
    # Regularized GD
    ############################################################################
    LAMBDA = 1 / np.sqrt(img_combined.shape[0])

    print('Regularized GD')
    print('#'*10 + ' Bigger than 5 ' + '#'*10)
    theo_wt_5_rgd, loss_list_5_rgd = rgd_optimizer(
        data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=LAMBDA, epochs=epochs)

    print('#'*10 + ' Number is even ' + '#'*10)
    theo_wt_ev_rgd, loss_list_ev_rgd = rgd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=LAMBDA, epochs=epochs)

    print('#'*10 + ' Number is prime ' + '#'*10)
    theo_wt_p_rgd, loss_list_p_rgd = rgd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, lambda_coeff=LAMBDA, epochs=epochs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5_rgd)
    ax.plot(loss_list_ev_rgd)
    ax.plot(loss_list_p_rgd)
    ax.grid()
    ax.set_title('RGD')
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5, EVEN, PRIME = test_model(theo_wt_5_rgd, img_combined, target_greater_than_5), test_model(
        theo_wt_ev_rgd, img_combined, target_even), test_model(theo_wt_p_rgd, img_combined, target_prime)
    print('RGD results (hinge loss, binary loss):\nGT5: {}\nEven {}\nPrime {}'.format(
        GT5, EVEN, PRIME))

    # Try other parameters values for gt5 problem
    best_loss_rgd = 1e6
    best_wts_rgd = np.zeros([100, 1])
    best_factor_rgd = 1
    fig, ax = plt.subplots()
    for x in np.append(np.linspace(-10, 10, 9), [-90, -50, 50, 100]):
        proposed_eta = eta * (1 + (x / 100))
        proposed_lambda = LAMBDA * (1 + (x / 100))
        proposed_wt_5, proposed_loss_list_5 = rgd_optimizer(
            data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=proposed_eta, lambda_coeff=proposed_lambda, epochs=epochs)
        min_epoch = np.argmin(proposed_loss_list_5)
        if (current_best := proposed_loss_list_5[min_epoch]) < best_loss_rgd:
            best_loss_rgd = current_best
            best_wts_rgd = proposed_wt_5[min_epoch]
            best_factor_rgd = (1 + (x / 100))
        ax.plot(proposed_loss_list_5)

    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('RGD - Greater than 5')
    ax.legend([f'{100 + x}% * Theoretical' for x in np.append(
        np.linspace(-10, 10, 9), [-90, -50, 50, 100])])
    fig.show()

    print(
        f'Best loss achieved for {best_factor_rgd} * theoretical values. Loss: {best_loss_rgd}')

    ############################################################################
    # SGD
    ############################################################################
    eta = 1e-6
    epochs = int(5e4)

    print('SGD')
    print('#'*10 + ' Bigger than 5 ' + '#'*10)
    theo_wt_5_sgd, loss_list_5_sgd = sgd_optimizer(
        data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + ' Number is even ' + '#'*10)
    theo_wt_ev_sgd, loss_list_ev_sgd = sgd_optimizer(
        data=img_combined, target=target_even, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    print('#'*10 + ' Number is prime ' + '#'*10)
    theo_wt_p_sgd, loss_list_p_sgd = sgd_optimizer(
        data=img_combined, target=target_prime, w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(loss_list_5_sgd)
    ax.plot(loss_list_ev_sgd)
    ax.plot(loss_list_p_sgd)
    ax.grid()
    ax.set_title('SGD')
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.legend(['Bigger than 5', 'Number is even', 'Number is prime'])
    fig.show()
    GT5, EVEN, PRIME = test_model(theo_wt_5_sgd, img_combined, target_greater_than_5), test_model(
        theo_wt_ev_sgd, img_combined, target_even), test_model(theo_wt_p_sgd, img_combined, target_prime)
    print('SGD results (hinge loss, binary loss):\nGT5: {}\nEven {}\nPrime {}'.format(
        GT5, EVEN, PRIME))

    # Try other parameters values for gt5 problem
    best_loss_sgd = 1e6
    best_wts_sgd = np.zeros([100, 1])
    best_factor_sgd = 1
    fig, ax = plt.subplots()
    for x in np.append(np.linspace(-10, 10, 9), [-90, -50, 50, 100]):
        proposed_eta = eta * (1 + (x / 100))
        proposed_wt_5, proposed_loss_list_5 = sgd_optimizer(
            data=img_combined, target=target_greater_than_5, w_shape=100, initial_variance=1/100, eta=proposed_eta, epochs=epochs)
        min_epoch = np.argmin(proposed_loss_list_5)
        if (current_best := proposed_loss_list_5[min_epoch]) < best_loss_sgd:
            best_loss_sgd = current_best
            best_wts_sgd = proposed_wt_5[min_epoch]
            best_factor_sgd = (1 + (x / 100))
        ax.plot(proposed_loss_list_5)

    ax.grid()
    ax.set_xlabel('Epochs (#)')
    ax.set_ylabel('Error')
    ax.set_title('SGD - Greater than 5')
    ax.legend([f'{100 + x}% * Theoretical' for x in np.append(
        np.linspace(-10, 10, 9), [-90, -50, 50, 100])])
    fig.show()

    print(
        f'Best loss achieved for {best_factor_sgd} * theoretical values. Loss: {best_loss_sgd}')

    ############################################################################
    # Part B - Generalization
    ###########################################################################

    # Use randomized train & test sets
    indices = [i for i in range(img_combined.shape[0])]
    train_test_factor = 0.85
    target_greater_than_5 = np.array(target_greater_than_5)
    train_idx = np.random.choice(indices, int(
        train_test_factor*img_combined.shape[0])).astype(int)
    test_idx = list_reduction(indices, train_idx.tolist())

    # split to test-train
    train_data = img_combined[train_idx, :]
    target_greater_than_5_train = target_greater_than_5[train_idx]
    test_data = img_combined[test_idx, :]
    target_greater_than_5_test = target_greater_than_5[test_idx]

    print('SGD')
    eta = 1e-6
    epochs = len(train_idx)
    print('#'*10 + ' Bigger than 5 ' + '#'*10)
    theo_wt_5_sgd, loss_list_5_sgd = sgd_optimizer(data=train_data, target=target_greater_than_5_train,
                                                   w_shape=100, initial_variance=1/100, eta=eta, epochs=epochs, test_data=test_data,
                                                   test_target=target_greater_than_5_test)
    train_loss = loss_list_5_sgd[:, 0]
    test_loss = loss_list_5_sgd[:, 1]
    zero_one_loss = loss_list_5_sgd[:, 2]
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(test_loss)
    ax.legend(['train loss', 'test loss'])
    ax.grid()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(zero_one_loss)
    ax.grid()
    ax.set_xlabel('epochs')
    ax.legend(['zero-one loss'])
    ax.set_ylabel('loss')
    fig.show()
    input("Press any key to continue")


if __name__ == "__main__":
    main()
