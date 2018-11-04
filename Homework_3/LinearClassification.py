import math

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

training_data = pd.read_csv("ldaData.txt", sep="\s{2}", header=None)


def lda(data):
    x1 = data[:50].values
    x2 = data[50:(50 + 44)].values
    x3 = data[(50 + 44):].values
    y = [0] * 50 + [1] * 44 + [2] * 43
    c = []
    for label in y:
        if label == 0:
            c.append('yellow')
        elif label == 1:
            c.append('blue')
        elif label == 2:
            c.append('red')

    plt.figure()
    plt.title("Data points with original classes")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(data.values[:, 0], data.values[:, 1], c=c)
    c1 = mpatches.Patch(color='yellow', label='Class 1')
    c2 = mpatches.Patch(color='blue', label='Class 2')
    c3 = mpatches.Patch(color='red', label='Class 3')
    plt.legend(handles=[c1, c2, c3], loc='lower right')
    plt.show()

    x = np.array([x1, x2, x3])
    n_samples = np.array([x1.shape[0], x2.shape[0], x3.shape[0]])

    means = np.concatenate((np.mean(x1, axis=0),
                            np.mean(x2, axis=0),
                            np.mean(x3, axis=0))
                           ).reshape(x.shape[0], data.values.shape[1])

    # class_mean = np.sum(n_samples[:, np.newaxis] * means, axis=0) / data.shape[0]
    total_mean = np.mean(data.values, axis=0)
    S_total = np.zeros(shape=(means.shape[1], means.shape[1]))
    S_within = np.zeros(shape=(means.shape[1], means.shape[1]))
    S_between = np.zeros(shape=(means.shape[1], means.shape[1]))

    # total scatter
    for val in data.values:
        diff = val - total_mean
        S_total += np.outer(diff, diff.T)

    # within class scatter
    for i, subset in enumerate(x):
        S_k = np.zeros(shape=(means.shape[1], means.shape[1]))
        for val in subset:
            diff = val - means[i]
            S_k += np.outer(diff, diff.T)

        # S_k = np.cov(subset, rowvar=False)
        S_within += S_k

    S_total /= data.values.shape[0]
    S_within /= data.values.shape[0]

    # between class scatter
    S_between = S_total - S_within  # Bishop page 191, equation (4.45)

    # eigendecomposition
    eigenval, eigenvec = np.linalg.eig(np.linalg.inv(S_within).dot(S_between))

    # select first c-1 eigenvectors
    idx = (-eigenval).argsort()
    W = eigenvec[idx][:, :means.shape[1] - 1].T

    # calculate projections and their mean / variance
    proj = []
    mean_proj = np.empty(shape=(means.shape[0], means.shape[1] - 1))
    # covar_proj = np.empty(shape=(means.shape[0], x1.shape[1], x1.shape[1]))
    var_proj = np.empty(shape=(means.shape[0], means.shape[1] - 1))

    for i, subset in enumerate(x):
        proj.append(W.dot(subset.T))
        mean_proj[i] = np.mean(proj[i], axis=1)
        var_proj[i] = np.sum((proj[i] - mean_proj[i]) ** 2) / (subset.shape[0] - 1)

    prior = n_samples / np.sum(n_samples)

    # gaussian maximum likelihood posterior
    proj_flat = np.concatenate([proj[0][0], proj[1][0], proj[2][0]])
    gaussian_posterior = np.empty(shape=(len(x), data.shape[0]))
    for i in range(mean_proj.shape[0]):
        gaussian_posterior[i] = np.log(gaussian(proj_flat, mean_proj[i], var_proj[i]) * prior[i])

    y_hat = gaussian_posterior.argmax(axis=0)
    plt.figure()
    plt.title("Data points with LDA classification")
    plt.xlabel("x1")
    plt.ylabel("x2")
    c = []
    for label in y_hat:
        if label == 0:
            c.append('yellow')
        elif label == 1:
            c.append('blue')
        elif label == 2:
            c.append('red')

    plt.legend(handles=[c1, c2, c3], loc='lower right')
    plt.scatter(data.values[:, 0], data.values[:, 1], c=c)
    plt.show()

    print('Number of misclassified samples:', np.count_nonzero(y - y_hat))


def gaussian(data, mu, var):
    out = np.empty(data.shape[0])
    denom = np.sqrt(2 * np.pi * var)
    for i, val in enumerate(data):
        out[i] = np.exp(-(val - mu) ** 2 / (2 * var)) / denom

    return out


def multivariate_gaussian(data, mu, covar):
    out = np.empty(data.shape[0])
    denom = np.sqrt((2 * math.pi) ** data.shape[1] * np.linalg.det(covar))

    # compute for each data point
    for i, x in enumerate(data):
        diff = x - mu
        out[i] = np.exp(-.5 * diff.dot(np.linalg.inv(covar)).dot(diff.T)) / denom

    return out


lda(training_data)
