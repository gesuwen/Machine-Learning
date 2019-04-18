# Implementation of a linear model, logistic regression, mini-batch SGD, momentum and RMSprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def expand(X):
    """
    Adds quadratic features to make the circular shape data linearly separable

    For each sample (row in matrix), compute an expanded row:
    [feature0, feature1, feature0^2, feature1^2, feature0*feature1, 1]

    :param X: matrix of features, shape [n_samples,2]
    :returns: expanded features of shape [n_samples,6]
    """
    X_expanded = np.zeros((X.shape[0], 6))
    X_expanded[:, 0:2] = X[:, 0:2]
    X_expanded[:, 2:4] = X[:, 0:2] ** 2
    X_expanded[:, 4] = X[:, 0] * X[:, 1]
    X_expanded[:, 5] = 1
    return X_expanded


def probability(X, w):
    """
    Logistic regression

    Given input features and weights
    return predicted probabilities of y==1 given x, P(y=1|x)

    :param X: feature matrix X of shape [n_samples,6] (expanded)
    :param w: weight vector w of shape [6] for each of the expanded features
    :returns: an array of predicted probabilities in [0,1] interval.
    """
    X_expanded = expand(X)
    logit = X_expanded.dot(w)
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob


def compute_loss(X, y, w):
    """
    cross-entropy loss minimization of logistic regression

    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute scalar loss function using formula above.
    """
    prob = probability(X, w)
    loss = - 1.0/X.shape[0] * np.sum(y * np.log(prob) + (1-y) * np.log(1.0 - prob))
    return loss


def compute_grad(X, y, w):
    """
    compute gradients of each features

    Given feature matrix X [n_samples,6], target vector [n_samples] of 1/0,
    and weight vector w [6], compute vector [6] of derivatives of L over each weights.
    """
    prob = probability(X, w)
    grad = -1.0 / X.shape[0] * np.sum((y - prob)[:, None] * X, axis=0)
    return grad


# visualizes the predictions
def visualize(X, y, w, history, title):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = probability(expand(np.c_[xx.ravel(), yy.ravel()]), w)
    Z = Z.reshape(xx.shape)
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.grid()
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Two-dimensional classification
    # Generate non-separable training data
    numPoints = 300
    X, y = make_circles(n_samples=numPoints, factor=0.5, noise=.05)

    print("\nPlot the training data")
    print("Training data shape: ", X.shape, y.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
    plt.show()

    # Mini-batch SGD - takes a random example on each iteration, calculates a gradient of the loss and makes a step
    X_expanded = expand(X)
    w = np.array([0, 0, 0, 0, 0, 1]) # initialize weights
    eta = 0.1 # learning rate
    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12, 5))

    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)

        grad = compute_grad(X_expanded, y, w) # calculate gradients
        w = w - eta * grad # Update weights
    visualize(X, y, w, loss, "SGD")
    plt.clf()

    # SGD with momentum - helps accelerate SGD in the relevant direction and dampens oscillations
    w = np.array([0, 0, 0, 0, 0, 1])

    eta = 0.05 # learning rate
    alpha = 0.9 # momentum
    nu = np.zeros_like(w)

    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12, 5))

    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)

        grad = compute_grad(X_expanded, y, w)
        nu = alpha * nu + eta * grad # with momentum
        w = w - nu
    visualize(X, y, w, loss, "SGD with momentum")
    plt.clf()

    # RMSprop
    w = np.array([0, 0, 0, 0, 0, 1.])

    eta = 0.1  # learning rate
    alpha = 0.9  # moving average of gradient norm squared
    g2 = np.zeros_like(w)
    eps = 1e-8

    n_iter = 100
    batch_size = 4
    loss = np.zeros(n_iter)
    plt.figure(figsize=(12, 5))
    for i in range(n_iter):
        ind = np.random.choice(X_expanded.shape[0], batch_size)
        loss[i] = compute_loss(X_expanded, y, w)

        grad = compute_grad(X_expanded, y, w)
        grad2 = grad ** 2
        g2 = alpha * g2 + (1 - alpha) * grad2
        w = w - eta * grad / np.sqrt(g2 + eps)
    visualize(X, y, w, loss, "SGD with RMSprop")
    plt.clf()

