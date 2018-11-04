import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

training_data_1 = pd.read_csv("densEst1.txt", sep="\s{2}")
training_data_2 = pd.read_csv("densEst2.txt", sep="\s{2}")

training_data_1.columns = training_data_2.columns = ["x1", "x2"]

n_models = 2


def compute_prior():
    prior1 = training_data_1.shape[0] / (training_data_1.shape[0] + training_data_2.shape[0])
    return prior1, 1 - prior1


def estimate_parameters(data):
    covar = np.zeros((data.shape[1], data.shape[1]))

    N = data.shape[0]
    mu = np.sum(data, axis=0) / N
    for val in data:
        diff = val - mu
        covar += np.outer(diff, diff.T)

    covar_unbiased = covar / (N - 1)
    covar /= N

    return mu, covar, covar_unbiased


def visualize_likelihood(mu, covar, data):
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

    for i in range(n_models):
        for j in range(steps):
            # construct vector with same x and all possible y to cover the plot space
            points = np.append(X[j], Y[j]).reshape(2, x.shape[0]).T
            Z[j] = multivariate_gaussian(points, mu[i], covar[i])
        c_plot = plt.contour(X, Y, Z)
        plt.clabel(c_plot, inline=1, fontsize=10)


def multivariate_gaussian(data, mu, covar):
    """
    return likelihood for all given samples
    """
    out = np.empty(data.shape[0])
    denominator = np.sqrt((2 * math.pi) ** data.shape[1] * np.linalg.det(covar))

    # compute for each datapoint
    for i, x in enumerate(data):
        diff = x - mu
        out[i] = np.exp(-.5 * diff.T.dot(np.linalg.inv(covar)).dot(diff)) / denominator

    return out


def visualize_posterior(mu, covar, data, prior):
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

    for i in range(n_models):
        for j in range(steps):
            # construct vector with same x and all possible y to cover the plot space
            points = np.append(X[j], Y[j]).reshape(2, x.shape[0]).T
            Z[j] = multivariate_gaussian(points, mu[i], covar[i]) * prior[i]
        c_plot = plt.contour(X, Y, Z)
        plt.clabel(c_plot, inline=1, fontsize=10)

    for j in range(steps):
        # construct vector with same x and all possible y to cover the plot space
        points = np.append(X[j], Y[j]).reshape(2, x.shape[0]).T
        Z[j] = multivariate_gaussian(points, mu[0], covar[0]) * prior[0] - \
               multivariate_gaussian(points, mu[1], covar[1]) * prior[1]
    c_plot = plt.contour(X, Y, Z, levels=[0])
    plt.clabel(c_plot, inline=1, fontsize=10)


prior1, prior2 = compute_prior()

mu1, covar1, covar_unb1 = estimate_parameters(training_data_1.values)
mu2, covar2, covar_unb2 = estimate_parameters(training_data_2.values)

# print(mu1)
# print(covar1)
# print(covar_unb1)
# print(mu2)
# print(covar2)
# print(covar_unb2)

mus = np.append(mu1, mu2).reshape(n_models, mu1.shape[0])
sigmas = np.append(covar_unb1, covar_unb2).reshape(n_models, covar_unb1.shape[0], covar_unb1.shape[1])
data = np.append(training_data_1, training_data_2, axis=0)
priors = np.append(prior1, prior2).reshape(n_models, 1)

# plt.figure(0)
# visualize_likelihood(mus, sigmas, data)
#
# # plot the samples
# plt.plot(training_data_1.values[:, 0], training_data_1.values[:, 1], 'co', zorder=1, color="orange", label="estDens1")
# plt.plot(training_data_2.values[:, 0], training_data_2.values[:, 1], 'cx', zorder=1, label="estDens2")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("Estimate of Unbiased Gaussian Distributions")
# plt.legend()

plt.figure(1)
visualize_posterior(mus, sigmas, data, priors)
plt.plot(training_data_1.values[:, 0], training_data_1.values[:, 1], 'co', zorder=1, color="grey", label="estDens1")
plt.plot(training_data_2.values[:, 0], training_data_2.values[:, 1], 'cx', zorder=1, color="orange", label="estDens2")
plt.xlabel('x')
plt.ylabel('y')
plt.title("Posterior Distributions and Decision Boundary")
plt.legend()

plt.show()
