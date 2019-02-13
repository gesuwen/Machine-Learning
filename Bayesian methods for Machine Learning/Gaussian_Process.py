# -*- coding: utf-8 -*-

# Gaussian Process with  GPy and GPyOpt libraries 

import numpy as np
import GPy
import GPyOpt
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import sklearn.datasets
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import time

# Gaussian processes: GPy
# A simple regression problem to fit a Gaussian Process with RBF kernel.
def generate_points(n=25, noise_variance=0.0036):
    np.random.seed(777)
    X = np.random.uniform(-3.,3.,(n,1))
    y = np.sin(X) + np.random.randn(n,1)*noise_variance**0.5
    return X, y
    
def generate_noise(n=25, noise_variance=0.0036):
    np.random.seed(777)
    X = np.random.uniform(-3.,3.,(n,1))
    y = np.random.randn(n,1)*noise_variance**0.5
    return X, y

def f(parameters):
    parameters = parameters[0]
    score = -cross_val_score(
                SVR(C=parameters[0],
                              epsilon=parameters[2],
                              gamma=parameters[1]),
                X, y, scoring='neg_mean_squared_error').mean()
    score = np.array(score)
    return score


if __name__ == "__main__":
    # Create data points
    # Create RBF kernel with variance 1.5 and length-scale parameter 2 for 1D samples 
    # and compute value of the kernel between 6-th and 10-th points (one-based indexing system)
    X, y = generate_points()
    plt.plot(X, y, '.')
    plt.show()
    
    # Create RBF kernel with variance 1.5 and length-scale parameter 2 for 1D 
    # samples and compute value of the kernel between 6-th and 10-th points (one-based 
    # indexing system).
    kernel = GPy.kern.RBF(1, 1.5, 2) ### YOUR CODE HERE
    kernel_59 = kernel.K(np.array([X[5],X[9]]))[0,1] ### YOUR CODE HERE
    
    
    # Fit GP into generated data with previous kernel.
    model = GPy.models.GPRegression(X,y, kernel)  ### YOUR CODE HERE
    mean, variance = model.predict(np.array([[1]])) ### YOUR CODE HERE
    
    model.plot()
    plt.show()
    
    # Check current parameter fit
    model
    
    #  Optimize length-scale, variance and noise component of the model.
    model.optimize()
    lengthscale = kernel.lengthscale
    
    model.plot()
    plt.show()
    
    # Generate two datasets: sinusoid wihout noise and samples from gaussian noise. 
    # Optimize kernel parameters
    X, y = generate_noise(noise_variance=10)
    kernel = GPy.kern.RBF(1, 1.5, 2)
    model = GPy.models.GPRegression(X,y, kernel)
    model.optimize()
    noise = model.Gaussian_noise.variance
    
    X, y = generate_points(noise_variance=0)
    kernel = GPy.kern.RBF(1, 1.5, 2)
    model = GPy.models.GPRegression(X,y, kernel)
    model.optimize()
    just_signal = model.Gaussian_noise.variance
    
    # Sparse GP
    # Create a dataset of 1000 points and fit GPRegression. 
    X, y = generate_points(1000)
    # Measure time for predicting mean and variance at position  𝑥=1 . 
    start = time.time()
    ### YOUR CODE HERE
    kernel = GPy.kern.RBF(1, 1.5, 2)
    model = GPy.models.GPRegression(X,y, kernel)
    #model.optimize()
    mean, variance = model.predict(np.array([[1]]))
    time_gp = time.time()-start
    # Fit SparseGPRegression with 10 inducing inputs and repeat the experiment. 
    start = time.time()
    kernel = GPy.kern.RBF(1, 1.5, 2)
    model = GPy.models.SparseGPRegression(X,y, kernel, num_inducing=10)
    #model.optimize()
    mean, variance = model.predict(np.array([[1]]))
    time_sgp = time.time()-start
    
    model.plot()
    plt.show()
    
    # Bayesian optimization: GPyOpt 
    # use diabetes dataset provided in sklearn package
    dataset = sklearn.datasets.load_diabetes()
    X = dataset['data']
    y = dataset['target']

    # use cross validation score to estimate accuracy; tune: max_depth, learning_rate, n_estimators
    # Score. Optimizer will try to find minimum, so we will add a "-" sign.
    baseline = -cross_val_score(XGBRegressor(), X, y, scoring='neg_mean_squared_error').mean()
    baseline
    
    # Bounds (NOTE: define continuous variables first, then discrete!)
    bounds = [
                {'name': 'C', 'type': 'continuous', 'domain': (1e-5, 1000)},
                {'name': 'gamma', 'type': 'continuous', 'domain': (1e-5, 10)},
                {'name': 'epsilon', 'type': 'continuous', 'domain': (1e-5, 10)}
             ]
    
    np.random.seed(777)
    optimizer = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds,
                                                    acquisition_type ='MPI',
                                                    acquisition_par = 0.1,
                                                    exact_eval=True)
    max_iter = 50
    max_time = 60
    optimizer.run_optimization(max_iter, max_time)
    optimizer.plot_convergence()
    
    # Best values of parameters:
    optimizer.X[np.argmin(optimizer.Y)]
    
    print('MSE:', np.min(optimizer.Y), 'Gain:', baseline/np.min(optimizer.Y)*100)
    
    # Tune SVR model. Find optimal values for three parameters: C, epsilon and gamma. 
    # Use range (1e-5, 1000) for C, (1e-5, 10) for epsilon and gamma. Use MPI as 
    # acquisition function with weight 0.1.
    best_epsilon = optimizer.X[np.argmin(optimizer.Y)][2]
    
    # boost in improvement that you got after tuning hyperparameters
    performance_boost =  baseline/np.min(optimizer.Y) 