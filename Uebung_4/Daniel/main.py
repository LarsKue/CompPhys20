import sys
from typing import Callable, Union, Iterable
from scipy.special import eval_hermite
import numpy as np
from matplotlib import pyplot as plt
import math


def minima_position(pos):
    # search for minima in the distance data by checking out if the previous value is bigger and the following value is bigger
    minima = []
    runter = pos[0] > pos[1]
    for n in range(len(pos) - 1):
        if runter and pos[n] < pos[n + 1]:
            minima.append(n)
        runter = pos[n] > pos[n + 1]
    return minima


def k(x, n):
    # k-function for the presence task
    epsilon = n + 0.5
    return 2 * epsilon - x ** 2


def k2(x, epsilon):
    # k-function for the homework
    return epsilon - x


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


def solve_numerov(f: Callable, h: Union[float, int], n_steps: int, y0: Union[float, int], y1: Union[float, int], *args,
                  **kwargs):
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


def presence(argv: list) -> int:
    h = 0.01
    n_steps = 300
    n_order = 0
    a = 1

    # check out if n is even or not and set the starting values to the corresponding values
    if is_even(n_order):
        y0 = a
        y1 = y0 - h ** 2 * k(0, n_order) * y0 / 2
    else:
        y0 = 0
        y1 = a

    # y0 = analytical(0, n_order)
    # y1 = analytical(h, n_order)

    # integration with the numerov scheme
    t, y = zip(*list(solve_numerov(k, h, n_steps, y0, y1, n_order)))
    t = np.array(t)
    y = np.array(y)

    # analytical solution
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
    plt.savefig("presence.pdf")
    plt.show()

    return 0


""" As you can see, if you refactor the solution you got for the numerov scheme, it is almost the same as the analytical
solution. Even for the relative big step size of 0.01, you have to zoom in a lot to see the error."""


def taska():
    # settings, similar to the presence task
    h = 0.01
    n_steps = 1000
    e_order = [1, 3]
    a = 1

    for j in e_order:
        if is_even(j):
            y0 = a
            y1 = y0 - h ** 2 * k2(0, j) * y0 / 2
        else:
            y0 = 0
            y1 = a

        t, y = zip(*list(solve_numerov(k2, h, n_steps, y0, y1, j)))
        t = np.array(t)
        y = np.array(y)

        plt.plot(t, y, label="e={}".format(j))

    plt.legend()
    plt.ylim(-3e6, 3e6)
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.savefig("neutrons.pdf")
    plt.show()


""" I ecided to plot the solution for n=1 and n=3. You can see that the divergence of e=1 is faster than e=3. But if
you set the y-limits symetrically, you can see that e=1 diverges towards infinity and e=3 towards minus infinity. 
The length x is z/z0 with z0 = h²/2m² (h is the reduced planck factor)
The energy e is p²/h²
"""


def taskb():
    # settings
    h = 0.01
    n_steps = 1000
    # all e values to check if it has a bound solution
    e_all = np.linspace(0, 10, 10000)
    a = 1
    y0 = 0
    y1 = a
    y10 = []

    # calculation of the numerov integration for every n and storation of the last value
    for j in e_all:
        t, y = zip(*list(solve_numerov(k2, h, n_steps, y0, y1, j)))
        y10.append(y[len(y) - 1])

    # find the minimas of the absolute values and find out their e's
    yinf = np.abs(np.array(y10))
    factor = e_all[len(e_all) - 1] / len(e_all)
    minima = factor * np.array(minima_position(yinf))

    # print the first three bound solutions
    for i in minima[:3]:
        t, y = zip(*list(solve_numerov(k2, h, n_steps, y0, y1, i)))
        t = np.array(t)
        y = np.array(y)
        plt.plot(t, y, label="e={}".format(i))
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.savefig("bound_solutions.pdf")
    plt.show()


""" For better accuracy, we decided to find the bound solutions with an accuracy of 0.001. The first bound solution 
seems to diverge, but much slower than what we saw in the first task. This is due to the limited accuracy of n. The other
 two solutions seem to go towards zero."""


def main():
    presence(sys.argv)
    taska()
    taskb()
    return 0


if __name__ == "__main__":
    main()
