print("Hello world")
import numpy as np
from matplotlib import pyplot as plt
import math
import sys


def teila():
    nlist = [1, 5, 10, 20, 30, 50]
    a = 5
    x = np.linspace(0, 1, 1000)
    for n in nlist:
        def f(z):
            return (z ** n) / (x + a)

        plt.plot(x, f(x), label=n)
    plt.xlabel('x')
    plt.ylabel('integrand')
    plt.legend()
    plt.show()


def teilb():
    a = 5
    n0 = 2
    y0 = 8
    n1 = 1
    if n0 << n1:
        y = y0
        for n in range(n0, n1 - 1):
            y = 1 / (n + 1) - a * y
        print(y)
    if n0 >> n1:
        y = y0
        for n in range(n1, n0 - 1):
            y = 1 / (n + 1) - a * y
        print(y)


def teilc():
    a = 5
    y0 = np.log((1 + a) / a)
    n0 = 0
    n1 = 30
    y = y0
    for n in range(n0, n1 - 1):
        y = 1 / (n + 1) - a * y
    print(y)


def präsenz():
    c = np.linspace(0, 1 / 4, 10000)

    def lösunga(c):
        return (-1 + np.sqrt(1 - 4 * c)) / 2

    def lösungb(c):
        return (-1 - np.sqrt(1 - 4 * c)) / 2

    plt.plot(c, lösunga(c), label='+')
    plt.plot(c, lösungb(c), label='-')
    plt.show()


def main(argv: list) -> int:
    teilc()


if __name__ == "__main__":
    main(sys.argv)
