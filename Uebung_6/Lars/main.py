
import numpy as np
from xalglib import smatrixtd
from scipy.linalg import eigh_tridiagonal
import math


def is_symmetric(m, rtol=1e-5, atol=1e-8):
    return np.allclose(m, m.T, rtol=rtol, atol=atol)


def symmetric_to_tridiagonal(m):
    return smatrixtd(m.tolist(), len(m), False)


def solve_eigenproblem(m):
    a, tau, d, e = symmetric_to_tridiagonal(m)
    return eigh_tridiagonal(d, e)


def delta(n, m):
    return n == m


def hamiltonian(n, l=0.1, N=30):
    Q = np.array([[np.sqrt(i + 1) * delta(i, j - 1) + np.sqrt(i) * delta(i, j + 1) for j in range(N)] for i in range(N)]) / np.sqrt(2)
    # Q^4
    Qt4 = Q @ Q @ Q @ Q

    h0 = (n + 0.5) * np.identity(N)

    h = h0 + l * Qt4

    return h


def test_eigenvalue(a, l):
    return math.isclose(np.linalg.det(a - l * np.identity(a.shape[0])), 0)


def homework():
    N = 30
    l = 0.1

    np.set_printoptions(precision=2)

    for n in range(1):
        h = hamiltonian(n, l, N)
        # print("Hamiltonian:\n", h)
        # print()

        ls, vs = solve_eigenproblem(h)

        for l, v in zip(ls, vs):
            # print(f"Eigenvalue l = {l:5.2f} for Eigenvector v = {v}")

            print(np.linalg.det(h - l * np.identity(N)))

            # assert test_eigenvalue(h, l)


        print()
        # only diagonal terms
        # h *= np.identity(N)
        # print(h)


def main(argv: list) -> int:
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
