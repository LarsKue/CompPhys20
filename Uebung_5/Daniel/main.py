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
    n = len(b)
    x = np.zeros(n)

    for i in range(n):
        for j in range(i):
            b[i] -= L[i][j] * x[j]
        x[i] = b[i] / L[i][i]

    b = x
    x = np.zeros(n)

    for i in reversed(range(n)):
        for j in reversed(range(i, n)):
            b[i] -= R[i][j] * x[j]

        x[i] = b[i] / R[i][i]

    return x


def presence():
    epsilon = 1e-6

    # A = np.array([
    #     [epsilon, 0.5],
    #     [0.5, 0.5]
    # ])
    #
    # b = np.array([0.5, 0.25])

    L = np.array([
        [1, 0, 0],
        [2, 1, 0],
        [3, 4, 1]
    ])

    R = np.array([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ])

    b = np.array([3, 10, 12])

    print("Compare:")
    print(np.linalg.inv(L @ R) @ b)
    print("---------")

    # L, R = LR(A)

    # print(R)

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
