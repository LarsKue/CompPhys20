
import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import random
from typing import Union


# the function required by task b and c
def get_iteration(n1: int, a: Union[float, int], n0: int, y0: Union[int, float]):
    def get_next(_a, n, y):
        # get y_n+1 from y_n
        return 1 / (n + 1) - a * y

    def get_prev(_a, n, y):
        # get y_n - 1 from y_n
        return (1 / n - y) / a

    while n0 < n1:
        y0 = get_next(a, n0, y0)
        n0 += 1
        # print("n = {}, y = {}".format(n0, y0))

    while n0 > n1:
        y0 = get_prev(a, n0, y0)
        n0 -= 1
        # print("n = {}, y = {}".format(n0, y0))

    return y0


def task_a():
    a = 5
    n_list = [1, 5, 10, 20, 30, 50]
    x = np.linspace(0, 1, 10000)

    plt.figure(figsize=(8, 6))

    for n in n_list:
        def f(_x):
            return _x ** n / (_x + a)

        plt.plot(x, f(x), label="n={:d}".format(n))
    plt.xlabel("x")
    plt.ylabel("Integrand")
    plt.legend()
    plt.title("Integrand of the series for a = 5 and different values for n")
    plt.show()


def task_b():
    # some examples
    a = 5
    n0 = 1
    y0 = 1
    n1 = 4

    # analytically determined
    y1 = 1 / 4 - 5 * (1 / 3 + 45 / 2)

    assert math.isclose(get_iteration(n1, a, n0, y0), y1)
    assert math.isclose(get_iteration(n0, a, n1, y1), y0)

    a = 1
    n0 = 0
    y0 = 0
    n1 = 3

    # analytically determined
    y1 = 1 / 3 + 1 / 2

    assert math.isclose(get_iteration(n1, a, n0, y0), y1)
    assert math.isclose(get_iteration(n0, a, n1, y1), y0)


def task_c():
    # from the bottom
    a = 5
    n_list = list(range(30 + 1))
    y0 = math.log((1 + a) / a)

    y_list = [get_iteration(n, a, n_list[0], y0) for n in n_list]

    # plot setup
    plt.figure(figsize=(8, 6))
    plt.plot(n_list, y_list)
    plt.xlabel("n")
    plt.ylabel("y")
    plt.title("The series diverges when iterating towards larger values for n")
    plt.show()

    # from the top
    n_list = list(reversed(range(30, 50 + 1)))

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)

    # choose arbitrary y0 at first
    lines, = plt.plot(n_list, [get_iteration(n, a, n_list[0], 0) for n in n_list])

    def new_plot(mouse_event):
        nonlocal lines, n_list
        # experiment with random starting values
        _y0 = random.randint(-10000, 10000)
        _y_list = [get_iteration(n, a, n_list[0], _y0) for n in n_list]

        # the series always converges for any y0
        # just change the label and ydata of the lines object
        lines.set_ydata(_y_list)
        lines.set_label("y0 = {}".format(_y0))
        # refresh the plot's legend
        ax.legend(loc="upper left")
        # redraw the plot
        plt.draw()

    ax.set_xlabel("n")
    ax.set_ylabel("y")
    ax.set_ylim((-10000, 10000))
    plt.title("The series converges when iterating towards smaller values for n")

    # add a button to redraw the plot with a new (random) starting value for the series
    ax_button = plt.axes([0.70, 0.05, 0.2, 0.075])
    button = Button(ax_button, "Click me!")
    button.on_clicked(new_plot)

    plt.show()


def praesenz():
    c = np.linspace(0, 1 / 2, 10000, dtype=np.complex64)

    plt.figure(figsize=(8, 6))

    def solution1(x):
        return (-1 + np.sqrt(1 - 4 * x)) / 2

    def solution2(x):
        return (-1 - np.sqrt(1 - 4 * x)) / 2

    y1 = solution1(c)
    y2 = solution2(c)

    # c for plotting must be real
    c = np.real(c)

    plt.plot(c, np.real(y1), label="Re(+)", alpha=0.7)
    plt.plot(c, np.imag(y1), label="Imag(+)", alpha=0.7)
    plt.plot(c, np.real(y2), label="Re(-)", alpha=0.7)
    plt.plot(c, np.imag(y2), label="Imag(-)", alpha=0.7)

    plt.xlabel("c")
    plt.ylabel(r"$\vert x \vert^2$")
    plt.legend()
    plt.show()


def main(argv: list) -> int:
    praesenz()
    # task_a()
    # task_b()
    # task_c()
    return 0


if __name__ == "__main__":
    main(sys.argv)
