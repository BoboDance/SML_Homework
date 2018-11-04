import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (15.0, 12.0)
plt.rcParams.update({'font.size': 18})

def target(x):
    s = np.sin(x)
    return s + np.square(s)


def kernel(x, z):
    # return theta_1 * np.exp(-.5 * theta_2 * np.abs(x - z) ** 2 / 2)
    return np.exp(-(x - z) ** 2)


def compute_c(x, noice):
    return kernel(x, x) + noice


def predict(x, X, Y, C_inv, noice):
    k = np.array([kernel(x, val) for val in X])
    mean = k.T.dot(C_inv).dot(Y)
    std = np.sqrt(compute_c(x, noice) - k.T.dot(C_inv).dot(k))
    return mean, std


def plot_gp(X, mean, std, iteration):
    y1 = mean - 2 * std
    y2 = mean + 2 * std

    plt.plot(X, target(X), "-", color="red", label="$sin(x)\cdot sin^2(x)$")
    plt.plot(X, mean, color="black", label="$\mu$")
    plt.fill_between(X, y1, y2, facecolor='lightblue', interpolate=True, alpha=.5, label="$2\sigma$")
    plt.title("$\mu$ and $\sigma$ for iteration={}".format(iteration))
    plt.xlabel("x")
    plt.ylabel("y")


def gpr():
    noice = 0.001 #noice
    step_size = 0.005
    X = np.arange(0, 2 * np.pi + step_size, step_size)
    iterations = 16

    std = np.array([np.sqrt(compute_c(x, noice)) for x in X])
    j = np.argmax(std)

    Xn = np.array([X[j]])
    Yn = np.array([target(Xn)])

    C = np.array(compute_c(X[j], noice)).reshape((1, 1))
    C_inv = np.linalg.solve(C, np.identity(C.shape[0]))

    for iteration in range(0, iterations):
        mean = np.zeros(X.shape[0])
        std = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            mean[i], std[i] = predict(x, Xn, Yn, C_inv, noice)

        if iteration + 1 in [1, 2, 4, 8, 16]:
            plot_gp(X, mean, std, iteration + 1)
            plt.plot(Xn, Yn, "o", c="blue", label="sampled")
            plt.legend()
            plt.show()

        j = np.argmax(std)
        Xn = np.append(Xn, X[j])
        Yn = np.append(Yn, target(Xn[-1]))

        # update C matrix
        x_new = Xn[-1]
        k = np.array([kernel(x_new, val) for val in Xn])
        c = kernel(x_new, x_new) + noice

        dim = C.shape[0]
        C_new = np.empty((dim + 1, dim + 1))
        C_new[:-1, :-1] = C
        C_new[-1, -1] = c
        C_new[:, -1] = k
        C_new[-1:] = k.T

        C = C_new
        C_inv = np.linalg.solve(C, np.identity(C.shape[0]))

gpr()