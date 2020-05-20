
"""
Submission for Uebung 4 for Computational Physics 2020
Group Members:
Daniel Kreuzberger
Lars Kuehmichel
David Weinand
"""

import sys
from typing import Callable, Union
from scipy.special import eval_hermite
import numpy as np
from matplotlib import pyplot as plt
import math

"""
How to use: simply run this file with
python main.py
We used python 3.8.1 for the development of this code.

In order to understand the code, it's probably best to start with
the main function at the bottom of this file, and only look at other
functions to specifically understand the method we employed to solve
a single part of the tasks.
"""


def is_even(n):
    return n % 2 == 0


def numerov_init(a, n_order, h, k):
    """
    Get initial Numerov Values
    :param a: Factoring Parameter
    :param n_order: The order of the factoring function k
    :param h: The initial time step
    :param k: The factoring function which defines the ODE
    :return: The first two solutions for the ODE, at times t = 0 and t = h
    """
    if is_even(n_order):
        y0 = a
        y1 = y0 - h ** 2 * k(0, n_order) * y0 / 2
    else:
        y0 = 0
        y1 = a

    return y0, y1


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


def solve_and_plot_numerov(k: Callable[[float, int], float], n_order: int, h=0.01, n_steps=1000,
                           analytical: Callable[[float, int], float] = None):
    """
    :param k: A factor function which defines the ODE
    :param n_order: The order of the factor function
    :param h: The time step
    :param n_steps: The number of steps to be taken
    :param analytical: Optional Analytical Solution
    :return: None
    """

    # we choose a = 1 for initial conditions
    y0, y1 = numerov_init(1, n_order, h, k)

    # analytical starting points would yield perfectly scaled results
    # y0 = analytical(0, n_order)
    # y1 = analytical(h, n_order)

    # turn generator values into two lists
    t, y = zip(*list(solve_numerov(k, h, n_steps, y0, y1, n_order)))
    t = np.array(t)
    y = np.array(y)

    # plot the numerical solution
    plt.plot(t, y, label=f"Order {n_order}")

    if analytical is not None:
        # get analytical solution values
        y_analytical = analytical(t, n_order)

        # plot the analytical solution
        plt.plot(t, y_analytical, label="analytical")

        # find out to which factor the numerical solution is scaled
        factor = np.mean(np.divide(y_analytical, y))
        print("Correction Factor: {:.2e}".format(factor))

        # plot a properly scaled version of the numerical solution
        plt.plot(t, factor * y, label="corrected")

    # plot setup
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()


def find_eigenvalue(k: Callable[[float, float], float], nmin: float, nmax: float, precision=1e-3, h=0.01, n_steps=1000):
    """
    :param k: A factor function which defines the ODE
    :param nmin: Define the lower bound in which to check for the Eigenvalue
    :param nmax: Define the upper bound
    :param precision: Define the precision with which to check for an eigenvalue
    :param h: The time step for the convergence solution
    :param n_steps: The number of steps for the convergence solution
    :return: The value for n which is closest to the Eigenvalue (i.e. yields a function closest to converging)
    """
    if nmax < nmin:
        # swap
        nmin, nmax = nmax, nmin

    result = None
    n = nmin
    while n < nmax:
        # we loop over n with a certain precision
        print("\rprogress: {:.2f}%".format(100 * (n - nmin) / (nmax - nmin)), end="")
        last_value = None
        # initialize parameters and solve the ODE
        y0, y1 = numerov_init(1, n, h, k)
        for _, last_value in solve_numerov(k, h, n_steps, y0, y1, n):
            # we don't want to do anything with the values in between
            # and only want to know the final y value
            pass

        # find the eigenvalue by keeping track of the n which yielded a minimal final value (convergence)
        last_value = abs(last_value)
        if result is None or last_value < result[1]:
            result = (n, last_value)

        n += precision

    return result[0]


def main(argv: list) -> int:
    # Attendance Task
    def k(x, n):
        # factor function for the attendance task
        epsilon = n + 0.5
        return 2 * epsilon - x ** 2

    @np.vectorize
    def analytical(x, n):
        # analytical solution for the Attendance Task
        return eval_hermite(n, x) / math.sqrt(2 ** n * math.factorial(n) * math.sqrt(np.pi)) * math.exp(- x ** 2 / 2)

    plt.figure(figsize=(10, 8))
    solve_and_plot_numerov(k, 0, analytical=analytical)
    plt.show()

    # Homework
    def k(x, n):
        epsilon = n + 0.5
        return epsilon - x

    # the orders to plot, we chose 1 for positive, 2 for negative asymptotic behavior.
    orders = [1, 2]
    # comment this in to have a single plot window
    # plt.figure(figsize=(10, 8))
    for n_order in orders:
        # comment this in to have separate plot windows
        plt.figure(figsize=(10, 8))
        solve_and_plot_numerov(k, n_order, h=0.01, n_steps=5000)

    # single plot ylim to compare orders of different divergence speed in one plot
    # plt.ylim((-60000, 60000))
    plt.show()

    # from the previous plots we know there is one eigenvalue between 1 and 2, one between 3 and 4
    # and one between 9 and 10, because the divergence switches signs in between those points
    limits = [(1, 2), (3, 4), (9, 10)]

    for lower, upper in limits:
        # go up to t = 30
        ev = find_eigenvalue(k, lower, upper, precision=1e-3, h=1e-3, n_steps=30000)
        print("\nEigenvalue: {:.2f}".format(ev))

        # show a plot again to confirm the convergence
        # note these still look like they are far from converging, because we only calculate the ev to
        # within two decimal digits, but plot the solution for a large range of t
        plt.figure(figsize=(10, 8))
        solve_and_plot_numerov(k, ev, h=1e-3, n_steps=30000)
        plt.savefig("ev_{:.2f}.png".format(ev))
        plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
