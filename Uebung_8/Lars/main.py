
import numpy as np
from scipy.linalg import eig
from copy import deepcopy
from matplotlib import pyplot as plt

from typing import Callable, Iterable


def jacobian(N, P, a, b, c, d):
    ls = [a[i] - 2 * N[i] - sum(b[i][j] * P[j] for j in range(3)) for i in range(3)]
    ks = [sum(c[i][j] * N[j] for j in range(3)) - d[i] for i in range(3)]

    return np.array([
        [ls[0], 0, 0, -b[0][0] * N[0], -b[0][1] * N[0], -b[0][2] * N[0]],
        [0, ls[1], 0, -b[1][0] * N[1], -b[1][1] * N[1], -b[1][2] * N[1]],
        [0, 0, ls[2], -b[2][0] * N[2], -b[2][1] * N[2], -b[2][2] * N[2]],
        [c[0][0] * P[0], c[0][1] * P[0], c[0][2] * P[0], ks[0], 0, 0],
        [c[1][0] * P[1], c[1][1] * P[1], c[1][2] * P[1], 0, ks[1], 0],
        [c[2][0] * P[2], c[2][1] * P[2], c[2][2] * P[2], 0, 0, ks[2]]
    ])


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


def homework():
    a = np.array([56, 12, 35])

    b = np.array([
        [20, 30, 5],
        [1, 3, 7],
        [4, 10, 20]
    ])

    c = np.array([
        [20, 30, 35],
        [3, 3, 3],
        [7, 8, 20]
    ])

    d = np.array([85, 9, 35])

    init_c = np.array([3, 3, 1, 1, -5, 0.1])

    # construct the matrix M = [[1, -a], [c, 0]]
    M1 = np.concatenate((-np.identity(3), -b.copy()), axis=1)
    M2 = np.concatenate((c.copy(), np.zeros((3, 3))), axis=1)
    M = np.concatenate((M1, M2), axis=0)

    sol = np.concatenate((-deepcopy(a), deepcopy(d)), axis=0)

    NP3 = np.linalg.solve(M, sol)

    print(NP3)

    Ns = [np.zeros(3),
          deepcopy(a),
          np.array([1, 1, 1])]

    Ps = [np.zeros(3),
          np.zeros(3),
          np.array([1, 1, 1])]

    np.set_printoptions(precision=2, suppress=True)

    print("===============================================")
    for fig_num, (N, P) in enumerate(zip(Ns, Ps)):
        # print(f"N = {N}, P = {P}")
        # print("Jacobian:")
        # print(jacobian(N, P, a, b, c, d))
        # print()

        ls, vs = eig(jacobian(N, P, a, b, c, d))

        # for i in range(len(ls)):
        #     print(f"Eigenvalue {ls[i]:.2f}, Eigenvector {vs[i]}")

        # n
        initial_state = sum(init_c[i] * vs[i] for i in range(len(vs)))

        # print(initial_state)

        # time evolution
        def f(t, state):

            def dNdt(_N, _P, i):
                return _N[i] * (a[i] - _N[i] - sum(b[i][j] * _P[j] for j in range(3)))

            def dPdt(_N, _P, i):
                return _P[i] * (sum(c[i][j] * _N[j] for j in range(3)) - d[i])

            _N = state[:3]
            _P = state[3:]

            return np.array([
                dNdt(_N, _P, 0),
                dNdt(_N, _P, 1),
                dNdt(_N, _P, 2),
                dPdt(_N, _P, 0),
                dPdt(_N, _P, 1),
                dPdt(_N, _P, 2)
            ])

        t = np.linspace(0, 0.3, 1000)

        _, time_evolution = zip(*list(solve_rk4(f, t, initial_state)))

        plt.figure(figsize=(12, 10))
        for i in range(6):
            plt.plot(t, [x[i].real for x in time_evolution], label=i+1)

        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Population")
        # plt.yscale("log")
        plt.savefig(f"figures/{fig_num}.png")
        plt.show()

        print("===============================================")


def main(argv: list) -> int:
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
