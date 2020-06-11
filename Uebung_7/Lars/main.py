
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from typing import Callable, Iterable


def step_rk4(f: Callable[[float, float], float], t: float, h: float, y: float):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)

    return t + h, y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve_rk4(f: Callable[[float, float], float], ts: Iterable[float], y0: float):
    it = iter(ts)
    last_t = next(it)
    yield last_t, y0

    for current_t in it:
        h = current_t - last_t

        _, y0 = step_rk4(f, last_t, h, y0)

        yield current_t, y0

        last_t = current_t


def attendance():
    def f(tau, n):
        return n * (1 - n)

    t = np.linspace(0, 10, 1000)

    plt.figure(figsize=(10, 10))
    for n0 in [-1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10]:
        t, n = zip(*list(solve_rk4(f, t, n0)))

        plt.plot(t, n, label=n0)

    plt.xlabel("t")
    plt.ylabel("n")
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()


def homework():

    for alpha in [-0.1, -0.01, -0.001, 0, 0.001]:
    # for alpha in [-0.1]:
        beta = 1 / 7.3

        def f(tau, n):
            return alpha * n * (1 - beta * n) - n ** 2 / (1 + n ** 2)

        t = np.linspace(0, 10, 1000)

        n0 = np.linspace(-10, 100, 10000)

        plt.plot(n0, f(0, n0), label=alpha)
    plt.grid()
    plt.ylim(-1, 1)
    plt.legend()

    plt.show()


    # plt.figure(figsize=(10, 10))
    # for n0 in [0, 1, 2]:
    #     t, n = zip(*list(solve_rk4(f, t, n0)))
    #
    #     plt.plot(t, n, label=n0)
    #
    # plt.xlabel("t")
    # plt.ylabel("n")
    # plt.legend()
    # plt.show()


def main(argv: list) -> int:
    # attendance()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)