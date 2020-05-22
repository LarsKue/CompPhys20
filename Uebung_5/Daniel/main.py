import numpy as np
import matplotlib as plt
import math
import sys


def LR(A):
    n = len(A)
    R = A
    L = np.identity(n)

    for i in range(n - 1):
        for k in range(i + 1, n):
            L[k][i] = R[k][i] / R[i][i]
            for j in range(i, n):
                R[k][j] = R[k][j] - L[k][i] * R[i][j]
    return L, R


def solution(L, b):
    n = len(b)
    x = np.zeros(n)
    # print(x)
    for i in range(n):
        x[i] = b[i] / L[i][i]
    return x


def solution_lars(L, R, b):
    # LRx = b
    n = b.shape[0]
    # we call this x for memory allocation purposes,
    # but in our first calculation this is y
    x = np.zeros(n)

    # first we solve Ly = b where y = Rx
    # L is a lower-left-triangular matrix
    for i in range(n):
        for j in range(i):
            # subtract from the solution vector to get the individual solution for this line
            b[i] -= L[i][j] * x[j]
        # still have to divide by the factor in the matrix
        x[i] = b[i] / L[i][i]

    # now we have the values for y in the variable called x, and want to solve for Rx = y
    # so the solution vector is now y
    b = x
    # and we want to find x
    x = np.zeros(n)

    # now we solve Rx = y in the same manner as above,
    # only R is an upper-right-triangular matrix
    for i in reversed(range(n)):
        for j in reversed(range(i, n)):
            b[i] -= R[i][j] * x[j]

        x[i] = b[i] / R[i][i]

    # x now contains the solution values for LRx = b
    return x


def presence():
    epsilon = 1e-6

    A = np.array([
        [epsilon, 0.5],
        [0.5, 0.5]
    ])

    b = np.array([0.5, 0.25])

    L, R = LR(A)

    print("Compare:")
    print(np.linalg.inv(L @ R) @ b)
    print("---------")

    x = solution_lars(L, R, b)
    print("solution:", x)

    print("rhs:", L @ R @ x)


def main():
    presence()
    # taska()
    # taskb()
    return 0


if __name__ == "__main__":
    main()
