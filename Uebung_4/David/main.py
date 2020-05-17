import scipy.special as sc
import matplotlib.pyplot as plt
import numpy as np
import math


def get_k(x, n):
    epsilon = n + 1/2
    return 2 * epsilon - x**2


def numerov(k_, k_vor, k_vorvor, y_vor, y_vorvor, h):
    return (2 * (1 - (5 / 12) * h**2 * k_vor) * y_vor - (1 + (1 / 12) * h**2 * k_vorvor) * y_vorvor) / (1 + (1/12) * h**2 * k_)


def algo(h, iterations, n, a):
    k = []
    y = []
    k.append(get_k(0, n))
    k.append(get_k(h, n))
    if a%2 == 0:
        y.append(a)
        y.append(y[0] - h**2 * k[0] * y[0] / 2)
    else:
        y.append(0)
        y.append(a)
    x = 0
    for i in range(iterations):
        if i > 1:
            k.append(get_k(x, n))
            y.append(numerov(k[i], k[i - 1], k[i - 2], y[i - 1], y[i - 2], h))
        x += h
    return y


def ana(x, n):
    hermit = sc.eval_hermite(n, x)
    return hermit / (2**n * math.factorial(n) * math.sqrt(np.pi)) * np.exp(- (x**2 / 2))


def main():
    a = 1
    h = 0.01
    iterations = 300
    n = 0
    y_algo = algo(h, iterations, n, a)
    plt.plot(h * np.linspace(0, 300, 300), y_algo)
    plt.plot(h * np.linspace(0, 300, 300), ana(h * np.linspace(0, 300, 300), n))
    plt.show()


if __name__ == "__main__":
    main()