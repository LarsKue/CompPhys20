
import numpy as np
from scipy.linalg import eig
from copy import deepcopy


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


def main(argv: list) -> int:
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

    Ns = [np.zeros(3),
          deepcopy(a)]

    Ps = [np.zeros(3),
          np.zeros(3)]

    for N, P in zip(Ns, Ps):

        print(jacobian(N, P, a, b, c, d))

        ls, vs = eig(jacobian(N, P, a, b, c, d))

        for i in range(len(ls)):
            print(f"Eigenvalue {ls[i]}, Eigenvector {vs[i]}")

    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
