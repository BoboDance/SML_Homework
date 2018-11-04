import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

training_data = pd.read_csv("gmm.txt", sep="\s{2}")
k = 4
n_iter = 30


def EM():
    # init
    covar = np.array(
        [np.identity(training_data.shape[1]) for _ in range(k)])
    mu = np.random.uniform(training_data.min().min(), training_data.max().max(),
                           size=(k, training_data.shape[1]))
    pi = np.random.uniform(size=(k,))

    log_likelihood = np.empty((n_iter,))

    for i in range(n_iter):
        alpha = e(x=training_data.values, mu=mu, covar=covar, pi=pi)
        mu, covar, pi = m(x=training_data.values, alpha=alpha)

        # plot at given steps
        if i + 1 in [1, 3, 5, 10, 30]:
            plt.figure(i)
            visualize(mu, covar, training_data.values, i)
        log_likelihood[i] = calculate_log_likelihood(training_data.values, mu, covar, pi)

        print("Finished iteration {}".format(i))

    plt.figure(n_iter + 1)
    plt.plot(log_likelihood)
    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood Progress")
    plt.show()


def e(x, mu, covar, pi):
    alpha = np.empty((k, x.shape[0]))
    for i in range(k):
        alpha[i] = pi[i] * multivariate_gaussian(x, mu[i], covar[i])

    # sum over all models per data point
    denominator = np.sum(alpha, axis=0)

    return alpha / denominator


def m(x, alpha):
    N = np.sum(alpha, axis=1)  # sum over all data points per model

    mu = np.zeros((k, x.shape[1]))
    covar = np.zeros((k, x.shape[1], x.shape[1]))

    for i in range(k):
        # update mu
        for j, val in enumerate(x):
            mu[i] += (alpha[i, j] * val)

        mu[i] /= N[i]

        # update covariance
        for j, val in enumerate(x):
            diff = val - mu[i]
            covar[i] += alpha[i, j] * np.outer(diff, diff.T)

        covar[i] /= N[i]

    # update pi
    pi = N / x.shape[0]

    return mu, covar, pi


def multivariate_gaussian(data, mu, covar):

    out = np.empty(data.shape[0])
    denominator = np.sqrt((2 * math.pi) ** data.shape[1] * np.linalg.det(covar))
    covar_inv = np.linalg.inv(covar)

    # compute for each datapoint
    for i, x in enumerate(data):
        diff = x - mu
        out[i] = np.exp(-.5 * diff.T.dot(covar_inv).dot(diff)) / denominator

    return out


def visualize(mu, covar, data, iteration):

    steps = 100

    x_data = data[:, 0]
    y_data = data[:, 1]

    x_min = x_data.min()
    x_max = x_data.max()
    y_min = y_data.min()
    y_max = y_data.max()

    x = np.arange(x_min - 1, x_max + 1, (x_max - x_min + 2) / steps)
    y = np.arange(y_min - 1, y_max + 1, (y_max - y_min + 2) / steps)

    Y, X = np.meshgrid(y, x)
    Z = np.empty((steps, steps))

    for i in range(k):
        for j in range(steps):
            # construct vector with same x and all possible y to cover the plot space
            points = np.append(X[j], Y[j]).reshape(2, x.shape[0]).T
            Z[j] = multivariate_gaussian(points, mu[i], covar[i])
        plt.contour(X, Y, Z, 1)

    # plot the samples
    plt.plot(x_data, y_data, 'co', zorder=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Mixtures after {} steps".format(iteration + 1))


def calculate_log_likelihood(x, mu, covar, pi):
    likelihood = np.empty((k, x.shape[0]))
    for i in range(k):
        likelihood[i] = pi[i] * multivariate_gaussian(x, mu[i], covar[i])

    return np.sum(np.log(np.sum(likelihood, axis=0)))


EM()
