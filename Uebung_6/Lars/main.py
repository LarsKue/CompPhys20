
import numpy as np
from xalglib import smatrixtd
from scipy.linalg import eigh_tridiagonal


def is_symmetric(m, rtol=1e-5, atol=1e-8):
    return np.allclose(m, m.T, rtol=rtol, atol=atol)


def symmetric_to_tridiagonal(m):
    return smatrixtd(list(m), len(m), False)


def solve_eigenproblem(m):
    a, tau, d, e = symmetric_to_tridiagonal(m)
    return eigh_tridiagonal(d, e)


def delta(n, m):
    return n == m


def praesenz() -> None:
    # M = [
    #     [1, 2, 3, 4, 5],
    #     [2, 1, 2, 3, 4],
    #     [3, 2, 1, 2, 3],
    #     [4, 3, 2, 1, 2],
    #     [5, 4, 3, 2, 1]
    # ]

    M = [
        [1, 3, 1, 1],
        [2, 1, 3, 1],
        [1, 2, 1, 3],
        [1, 1, 2, 1]
    ]

    # tqli == scipy.linalg.eigh_tridiagonal
    ls, vs = solve_eigenproblem(M)

    np.set_printoptions(precision=2)

    for l, v in zip(ls, vs):
        print(f"lambda = {l:5.2f} for eigenvector v = {v}")


def homework():
    n = 0
    N = 15
    Q = np.array([[np.sqrt(i + 1) * delta(i, j - 1) + np.sqrt(i) * delta(i, j + 1) for j in range(N)] for i in range(N)]) / np.sqrt(2)

    print(Q)

    # Q^4
    Q = Q @ Q @ Q @ Q

    print(Q)

    l = 0.1

    h0 = (n + 0.5) * np.identity(N)




def main(argv: list) -> int:
    # praesenz()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
