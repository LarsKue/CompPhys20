import sys
from typing import Callable, Union, Iterable
from scipy.special import eval_hermite
import numpy as np
from matplotlib import pyplot as plt
import math


def k(x, n):
    epsilon = n + 0.5
    return 2 * epsilon - x ** 2


def is_even(n):
    return n % 2 == 0


def step_numerov(h: Union[float, int], k0: Union[float, int], y0: Union[float, int],
                 k1: Union[float, int], y1: Union[float, int]):
    """
    Take a single numerov step:
    (1 + 1/12 h^2 k_n+1) y_n+1 = 2 (1 - 5/12 h^2 k_n) y_n - (1 + 1/12 h^2 k_n-1) y_n-1
    The error is of order h^6
    :param h: the time step
    :param k0: k_n-1 in the above equation
    :param y0: y_n-1 in the above equation
    :param k1: k_n in the above equation
    :param y1: y_n in the above equation
    :return: The left-hand-side of the above equation
    """
    factor_0 = -(1 + h ** 2 * k0 / 12)
    factor_1 = 2 * (1 - 5 * h ** 2 * k1 / 12)

    return factor_1 * y1 + factor_0 * y0


def solve_numerov(f: Callable, h: Union[float, int], n_steps: int, y0: Union[float, int], y1: Union[float, int], *args, **kwargs):
    """
    Solve the ODE y'' + f(x) y(x) = 0
    :param f: a function that returns the factor in the above equation for a given x
    :param h: the time step
    :param n_steps: the number of steps to be taken
    :param y0: first starting value for y
    :param y1: second starting value for y
    :param args: positional arguments for f
    :param kwargs: keyword arguments for f
    :return: (t, y) tuples
    """
    yield 0, y0
    yield h, y1

    result = None
    for i in range(n_steps):

        k0 = f(h * i, *args, **kwargs)
        k1 = f(h * (i + 1), *args, **kwargs)

        if result is not None:

            y0 = y1
            y1 = result / (1 + h ** 2 * k1 / 12)

            yield h * (i + 1), y1

        # (1 + 1/12 h^2 k2) y2
        result = step_numerov(h, k0, y0, k1, y1)


@np.vectorize
def analytical(x, n):
    return eval_hermite(n, x) / math.sqrt(2 ** n * math.factorial(n) * math.sqrt(np.pi)) * math.exp(- x ** 2 / 2)


def main(argv: list) -> int:

    h = 0.01
    n_steps = 300
    n_order = 0

    a = 1
    if is_even(n_order):
        y0 = a
        y1 = y0 - h ** 2 * k(0, n_order) * y0 / 2
    else:
        y0 = 0
        y1 = a

    # y0 = analytical(0, n_order)
    # y1 = analytical(h, n_order)

    t, y = zip(*list(solve_numerov(k, h, n_steps, y0, y1, n_order)))
    t = np.array(t)
    y = np.array(y)

    y_analytical = analytical(t, n_order)

    plt.figure(figsize=(10, 8))
    plt.plot(t, y, label="Numerov")
    plt.plot(t, y_analytical, label="analytical")

    factor = np.mean(np.divide(y_analytical, y))
    print("Correction Factor: {:.2e}".format(factor))
    plt.plot(t, factor * y, label="corrected")

    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
