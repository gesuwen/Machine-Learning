"""
Implementation of Expectation-maximization algorithm on Gaussian Mixture Model 
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights 
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices
    
    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0] # number of objects
    C = pi.shape[0] # number of clusters
    gamma = np.zeros((N, C)) # distribution q(T)

    const_c = np.zeros(C)
    
    for i in range(C):
        pi_c = pi[i] # number
        mu_c = mu[i:i+1, :] # (1 x d)
        sigma_c = sigma[i] # (d x d)
        det_sigma_c = np.linalg.det(sigma_c)
        
        # assume determinant positive
        const_c[i] = pi_c/np.sqrt(det_sigma_c) #number
        X_centered = X - mu_c # (N x d)
        # y = (A^-1)x
        y = np.linalg.solve(sigma_c, X_centered.T)  #(d x N)
        # -1/2 * (xi-mu_c).T * sigma^-1 * (xi-mu_c) for all points together which diagonal of the dot product on right side
        gamma[:,i] = -0.5 * np.diag(np.dot(X_centered, y)) # (N)
        
    # subtract the max across column for a given row from the row (improve stability)
    gamma = gamma - gamma[range(N), np.argmax(gamma, axis=1)][:,np.newaxis]
    # take exponent of gamma
    gamma = np.exp(gamma)
    #mutiply each columns by respective constant pi_c / sqrt(det_sigma_c)                          
    gamma = np.multiply(gamma, const_c[np.newaxis,:])
    # normalize gamma
    gamma /= np.sum(gamma, axis=1)[:,np.newaxis]
   
    return gamma



def M_step(X, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    mu = np.zeros((C, d))
    sigma = np.zeros((C, d, d))
    pi = np.sum(gamma, axis=0) / (1.0 * N)
    
    for i in range(C):
        mu[i, :] = np.sum(X * gamma[:,i][:,np.newaxis], axis=0) / (pi[i]*N)
        Xi = X - mu[i, :][np.newaxis, :]
        sigma[i, :, :] = np.dot((Xi*gamma[:, i][:,np.newaxis]).T, Xi) / (pi[i]*N)

    return pi, mu, sigma


def compute_vlb(X, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    
    Returns value of variational lower bound
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    loss = 0
    for i in range(C):
        pi_i = pi[i]
        mu_i = mu[i]
        sigma_i = sigma[i]
        det_sigma_i = np.linalg.det(sigma_i)
        for j in range(N):
            loss += (gamma[j,i]+1e-20) * (np.log(pi_i) - 0.5 * np.log(det_sigma_i) - 0.5 * d * np.log(2 * np.pi))
            x_j = X[j]
            x_j = X[j] - mu_i
            y_j = np.linalg.solve(sigma_i, x_j)
            loss -= (gamma[j,i]+1e-20) * 0.5 * np.dot(x_j[np.newaxis,:], y_j[:, np.newaxis]) 
            loss -= (gamma[j,i]+1e-20) * np.log((gamma[j,i]+1e-20))

    return loss


def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.
    
    X: (N, d), data points
    C: int, number of clusters
    '''
    N, d = X.shape[0], X.shape[1] # N: number of objects; D: dimension of each object
    best_loss = float('-inf')
    best_pi = None
    best_mu = None
    best_sigma = None

    for _ in range(restarts):
        try:
            pi = np.array([1.0/C]*C,dtype=np.float32)
            mu = np.random.rand(C, d)
            sigma_ = np.random.rand(C, d, d)
            sigma = np.array([np.dot(A, A.T) for A in sigma_])
            prev_loss = None
            for i in range(max_iter):
                gamma = E_step(X, pi, mu, sigma)
                pi, mu, sigma = M_step(X, gamma)
                loss = compute_vlb(X, pi, mu, sigma, gamma)
                if not math.isnan(loss) and loss > best_loss:
                    best_loss = loss
                    best_mu = mu
                    best_pi = pi
                    best_sigma = sigma

                if prev_loss is not None:
                    diff = np.abs(loss - prev_loss)
                    if diff < rtol:
                        break
                prev_loss = loss
        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")
            pass

    return best_loss, best_pi, best_mu, best_sigma


if __name__ == '__main__':
    # Initialize and plot the data
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=0,cluster_std=[0.5,0.5,0.5])
    pi0 = np.random.random_sample((3))
    mu0 = np.random.random_sample((3,2))
    sigma0 = np.random.random_sample((3,2,2))
    plt.scatter(X[:, 0], X[:, 1], c='grey', s=30)
    plt.axis('equal')
    plt.show()
    
    # Perform E step and M step, calculate the loss
    gamma = E_step(X, pi0, mu0, sigma0)
    pi, mu, sigma = M_step(X, gamma)
    loss = compute_vlb(X, pi, mu, sigma, gamma)
    
    # Train with EM algorithm
    best_loss, best_pi, best_mu, best_sigma = train_EM(X, 3)
    
    gamma = E_step(X, best_pi, best_mu, best_sigma)
    # Classfy each point to clusters
    labels = gamma.argmax(1)
    # Visualize the result
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    plt.axis('equal')
    plt.show()