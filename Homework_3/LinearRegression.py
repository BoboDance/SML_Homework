import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("linRegData.txt", sep="\s{2}", header=None)
training_data = data[:20]
test_data = data[20:]

n_samples_train = training_data.shape[0]
n_samples_test = test_data.shape[0]
n_samples_all = data.shape[0]

x_train = training_data[0].values.reshape(n_samples_train, 1)
y_train = training_data[1].values.reshape(n_samples_train, 1)

x_test = test_data[0].values.reshape(n_samples_test, 1)
y_test = test_data[1].values.reshape(n_samples_test, 1)

x_all = data[0].values.reshape(n_samples_all, 1)
y_all = data[1].values.reshape(n_samples_all, 1)

feature_type = "gaussian"
min_model_complexity = 15
max_model_complexity = 40

# feature_type = "poly"
# min_model_complexity = 0
# max_model_complexity = 21

ridge_param = 1e-6
variance = 0.02


def linear_regression(feature_type):
    train_rmse = np.zeros(shape=(max_model_complexity - min_model_complexity,))
    test_rmse = np.zeros(shape=(max_model_complexity - min_model_complexity,))
    theta = []

    for degree in range(min_model_complexity, max_model_complexity):
        if feature_type == "poly":
            # polynonimal features and bias
            phi = create_poly(x_train, degree)
            theta.append(np.dot(np.linalg.inv(
                np.dot(phi, phi.T) + ridge_param * np.identity(degree + 1)),
                np.dot(phi, y_train)))
            phi_test = create_poly(x_test, degree)

        elif feature_type == "gaussian":
            # only n gaussian features
            phi = create_gaussian(x_train, degree, variance)
            theta.append(np.dot(np.linalg.inv(
                np.dot(phi, phi.T) + ridge_param * np.identity(degree)),
                np.dot(phi, y_train)))
            phi_test = create_gaussian(x_test, degree, variance)

        # adjust index if not started with 0
        degree -= min_model_complexity

        y_hat_train = phi.T.dot(theta[degree])
        y_hat_test = phi_test.T.dot(theta[degree])

        train_rmse[degree] = rmse(y_train, y_hat_train)
        test_rmse[degree] = rmse(y_test, y_hat_test)

    plot_rmse(train_rmse, test_rmse)
    best_degree = np.argmin(test_rmse)
    var = variance if feature_type == "gaussian" else None
    plot_best(theta[best_degree], best_degree + min_model_complexity, feature_type, var)
    plt.show()

    return train_rmse, test_rmse


def bayesian_linear_regression():
    n_data_points = [10, 12, 16, 20, 50, 150]
    best_poly_degree = 12
    noise = 0.0025  # 1/beta

    for n in n_data_points:

        covar = np.identity(best_poly_degree + 1)
        x_points = x_all[:n]
        y_points = y_all[:n]

        phi = create_poly(x_points, best_poly_degree)

        mean_prior = np.linalg.inv(phi.dot(phi.T) + ridge_param * np.linalg.inv(covar)).dot(phi).dot(y_points)
        covar_prior = np.linalg.inv(phi.dot(phi.T)/noise + ridge_param/noise * np.linalg.inv(covar))

        phi_plot = create_poly(np.linspace(0, 2, 200), best_poly_degree)
        mean = mean_prior.T.dot(phi_plot)
        std = np.empty(shape=phi_plot.shape[1])
        for i, elem in enumerate(phi_plot.T):
            std[i] = np.sqrt(noise + elem.T.dot(covar_prior).dot(elem))

        plt.plot(x_points, y_points, '.', color="purple")
        y1 = mean.flatten() - std
        y2 = mean.flatten() + std
        plt.plot(np.linspace(0, 2, 200), mean.flatten(), color="orange", label="$\mu$")
        plt.fill_between(np.linspace(0, 2, 200), y1, y2, facecolor='blue', interpolate=True, alpha=.5, label="$\sigma$")

        plt.title("Mean and std for n={}".format(n))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


def rmse(y1, y2):
    return np.sqrt(np.sum((y1 - y2) ** 2) / y1.shape[0])


def plot_rmse(train_rmse, test_rmse):
    plt.figure(0)
    plt.plot(range(1 + min_model_complexity, max_model_complexity + 1), train_rmse, label="train")
    plt.plot(range(1 + min_model_complexity, max_model_complexity + 1), test_rmse, label="test")
    plt.xlabel("dimensions")
    plt.ylabel("RMSE")
    plt.title("Dimensionality and RMSE")
    plt.legend()


def plot_best(theta, degree, feature_type, var):
    steps = (np.max(x_all) - np.min(x_all)) / 500
    x = np.arange(np.min(x_all), np.max(x_all), steps)

    if feature_type == "poly":
        y_hat = theta.T.dot(create_poly(x, degree))
    elif feature_type == "gaussian":
        y_hat = theta.T.dot(create_gaussian(x, degree, var))

    pred = np.append(x, y_hat).reshape(2, x.shape[0])

    plt.figure(1)
    plt.plot(x_all, y_all, 'o', label="all data points")
    plt.plot(pred[0], pred[1], label="linear regression model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Best model with d={}".format(degree))


def create_poly(x, degree):
    # add one for bias
    degree += 1
    x = np.repeat(x, degree).reshape(x.shape[0], degree)
    out = np.empty(shape=x.shape).T

    # create polynomial in the form
    # [1, x, x^2, x^3,...,x^degree]
    # for each data sample
    for i, elem in enumerate(x.T):
        out[i] = elem ** i

    return out


def create_gaussian(x, n_features, var):
    means = np.linspace(0, 2, n_features)

    x = np.repeat(x, n_features).reshape(x.shape[0], n_features)
    out = np.empty(shape=x.shape).T

    # calc value based on n gaussians
    # for each data sample
    for i, elem in enumerate(x.T):
        out[i] = gaussian(elem, means[i], var)

    return out / np.sum(out, axis=0)


def gaussian(x, mean, var):
    denom = (2 * np.pi * var) ** .5
    num = np.exp(-(x - mean) ** 2 / (2 * var))
    return num / denom


def multivariate_gaussian(data, mu, covar):
    out = np.empty(data.shape[0])
    denom = np.sqrt((2 * math.pi) ** data.shape[1] * np.linalg.det(covar))

    # compute for each data point
    for i, x in enumerate(data):
        diff = x - mu
        out[i] = np.exp(-.5 * diff.dot(np.linalg.inv(covar)).dot(diff.T)) / denom

    return out


def visualize_gaussian(var):
    x = np.linspace(np.min(x_all), np.max(x_all), 200)
    phi = create_gaussian(x, 20, var)
    print(phi.shape)
    plt.figure(2)
    plt.plot(x, phi.T)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian Basis Functions")
    plt.show()


def visualize_bayes(x, y, std):
    plt.errorbar(x, y, std, linestyle='None', marker='^')


# linear_regression(feature_type=feature_type)
# visualize_gaussian(variance)
bayesian_linear_regression()
