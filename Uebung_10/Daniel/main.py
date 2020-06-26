import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from typing import Callable, Iterable


def iteration(I, a, m, c):
    for j in range(m):
        Ij = np.mod(a * I[j] + c, m)
        I.append(Ij)
    normalizedi = np.array(I) / (m - 1)
    return I, normalizedi


def presence():
    a = 106
    m = 6075
    c = 1283
    I = [2]
    random, randomnor = iteration(I, a, m, c)
    for k in range(m - 1):
        plt.scatter(randomnor[k], randomnor[k + 1])
    plt.show()
    I1 = [2]
    I2 = [4]
    random1, randomnor1 = iteration(I1, a, m, c)
    random2, randomnor2 = iteration(I2, a, m, c)
    plt.plot(randomnor1, randomnor2)
    plt.show()
    plt.hist(randomnor, bins=100)


def homework():
    w = 0.5
    b = 2 / w ** 2
    x = np.random.random_sample([10000]) * 0.5
    plt.hist(x, bins=100)
    plt.show()
    probability = b * x

    def f(x):
        return np.sqrt(1 - x ** 2)

    def rejection(randomset):
        r = []
        for x in randomset:
            if x > 1:
                continue
            elif x < 0:
                continue
            else:
                if np.random.uniform(0, 1) < f(x):
                    r.append(x)
        return r

    N = [10, 100, 1000, 10000, 100000, 1000000]

    def getpi(N):
        for g in N:
            randomunity = np.random.random_sample(g)
            rand = rejection(randomunity)
            pi = 4 * len(rand) / g
            plt.scatter(g, np.abs(pi - np.pi))

    getpi(N)
    plt.title('Difference between pi and our calculated value')
    plt.xscale('log')
    plt.ylim(0.0005, 1)
    plt.yscale('log')
    plt.show()


def main(argv: list) -> int:
    # presence()
    homework()
    return 0


if __name__ == "__main__":
    import sys

    main(sys.argv)
