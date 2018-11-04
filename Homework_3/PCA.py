import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("iris.txt", sep=",", header=None)
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


def normalize(x):
    return ((x - np.mean(x, axis=0)) / np.std(x, axis=0)).T


def PCA(x, n_eigen):
    C = x.dot(x.T) / x.shape[1]
    eigenvalues, eigenvectors = np.linalg.eig(C)
    eigenvalues_total = np.sum(eigenvalues)

    # eigenvalues are already sorted
    explained = np.sum(eigenvalues[:n_eigen + 1]) / eigenvalues_total

    B = eigenvectors[:, :n_eigen + 1]
    a = B.T.dot(x)

    return B, a, explained


def find_threshold(x):
    max_eigenvalue = x.shape[1]

    explained = np.empty(shape=(max_eigenvalue,))
    for n_eigen in range(max_eigenvalue):
        _, a, var = PCA(normalize(x), n_eigen)
        explained[n_eigen] = var

    plt.plot(np.arange(1, max_eigenvalue + 1), explained, label="marginal variance captured")
    plt.plot(np.arange(1, max_eigenvalue + 1), np.full(max_eigenvalue, .95), "--", label="threshold $\lambda=0.95$")
    plt.xticks(np.arange(1, max_eigenvalue + 1))
    plt.title("Threshold of marginal variance captured")
    plt.xlabel("# eigenvectors")
    plt.ylabel("marginal variance captured")
    plt.legend()
    plt.show()


def reverse_PCA(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    max_eigenvalue = x.shape[1]

    nrmse_total = np.empty(shape=(max_eigenvalue, x.shape[1]))

    for n_eigen in range(max_eigenvalue):
        B, a, _ = PCA(normalize(x), n_eigen)
        norm = np.amax(x, axis=0) - np.amin(x, axis=0)
        reverse = std * B.dot(a).T + mean
        nrmse_total[n_eigen] = nrmse(reverse, x, norm)

    print(nrmse_total)


def nrmse(y1, y2, norm):
    return np.sqrt(np.sum((y1 - y2) ** 2, axis=0) / y1.shape[0]) / norm


def plot_PCA_data(x, y):
    max_eigenvalue = x.shape[1]

    projection = None

    for n_eigen in range(max_eigenvalue):
        _, a, var = PCA(normalize(x), n_eigen)
        if var < .95: continue
        projection = a
        break

    # colors = [int(i % np.unique(y).shape[0]) for i in y]
    plt.scatter(projection[0], projection[1], c=y)
    plt.title("Data points after PCA")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


find_threshold(x)
plot_PCA_data(x, y)
reverse_PCA(x)
