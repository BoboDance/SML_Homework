import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

training_data = pd.read_csv("nonParamTrain.txt", sep="\s{2}")
test_data = pd.read_csv("nonParamTest.txt", sep="\s{2}")

training_data.columns = test_data.columns = ["value"]

x_min = -4
x_max = 8


def plot_histo():
    histo_size = [0.01, 0.5, 2.]

    for i, s in enumerate(histo_size):
        plt.figure(i)
        training_data.plot.hist(by="value", bins=math.ceil(training_data.max().value / s))
        plt.xlabel("x")
        plt.title("Histogram for bin size {}".format(s))
        plt.xlim(x_min, x_max)


def gaussian_kernel(x, data, sigma):
    numerator = np.sum(np.exp(-(x - data) ** 2 / (2 * sigma ** 2)))
    denominator = np.sqrt(2 * math.pi) * len(data) * sigma
    return numerator / denominator


def gaussian_KDE():
    sigmas = [.03, .2, .8]
    steps = (x_max - x_min) / 500
    x = np.arange(x_min, x_max, steps)
    # x = np.sort(test_data.values, axis=0)
    plt.figure()
    for sigma in sigmas:

        # get log-likelihood
        y = np.empty(training_data.values.shape[0])
        for i, val in enumerate(training_data.values):
            y[i] = gaussian_kernel(val, training_data.values, sigma)

        print("The train log−likelihood for $\sigma$={} is {}".format(str(sigma), str(np.sum(np.log(y)))))

        # get plots
        y = np.empty(x.shape)
        for i, val in enumerate(x):
            y[i] = gaussian_kernel(val, training_data.values, sigma)

        print("The test log−likelihood for $\sigma$={} is {}".format(str(sigma), str(np.sum(np.log(y)))))

        plt.plot(x, y, label="$\sigma=$" + str(sigma))
        plt.ylabel('Density')
        plt.xlabel('x')

    plt.legend()
    plt.show()


def knn():
    ks = [2, 8, 35]
    steps = (x_max - x_min) / 300
    x = np.arange(x_min, x_max, steps)
    # x = np.sort(test_data.values, axis=0)
    # x = np.sort(training_data.values, axis=0)
    # calculate pairwise distances
    x_dist = cdist(x.reshape(x.shape[0], 1),
                   training_data.values.reshape(training_data.values.shape[0], 1),
                   metric="minkowski")

    for k in ks:
        y = np.empty(x.shape)
        for i, val in enumerate(x_dist):
            # find nearest k points and take point with greatest distance as Volume size
            # this assumes the distance matrix was computed with two different vectors,
            # i.e. x_ij =/= 0; for i=j
	    # use k+1 for train data 	
            V = val[np.argpartition(val, range(k))[:(k)]][-1]
            # calculate density
            y[i] = \
                k / (training_data.values.shape[0] * V * 2)

        print("The log−likelihood for k={} is {}"
              .format(k, np.sum(np.log(y))))

        plt.plot(x, y, label="k={}".format(k))
        plt.ylabel('Density')
        plt.xlabel('x')

    plt.legend()
    plt.show()


# plot_histo()
# gaussian_KDE()
knn()
plt.show()
