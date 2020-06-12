
import numpy as np
from xalglib import smatrixtd
from scipy.linalg import eigh_tridiagonal
from matplotlib import pyplot as plt


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

    return h0 + l * Qt4


def homework():

    l = 0.1

    for n in range(10):
        Ns = np.arange(5, 50)
        eigenvalues = []
        for N in Ns:
            h = hamiltonian(n, l, N)

            ls, vs = solve_eigenproblem(h)

            ev = ls[0]

            eigenvalues.append(ev)

        # the eigenvalue visibly converges towards its analytical value
        plt.figure(figsize=(10, 8))
        plt.plot(Ns, eigenvalues)

        plt.xlabel("N")
        plt.ylabel("Eigenvalue")
        plt.title(f"Harmonic Oscillator Eigenvalue Convergence for n = {n}")
        plt.show()


def main(argv: list) -> int:
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
