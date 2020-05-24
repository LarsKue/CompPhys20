import numpy as np
import matplotlib as plt
import math
import sys


def LR(A):
    '''
    This is a function that does a LR-decomposition without pivoting
    :param A: A is the Matrix you want to decompose
    :return: L is the lower-left triangular matrix, R the upper right triangular matrix
    '''
    n = len(A)
    R = A
    L = np.identity(n)

    for i in range(n - 1):
        # since the first line of L and R stays the same, we do the iteration over the remaining n-1 lines
        for k in range(i + 1, n):
            L[k][i] = R[k][i] / R[i][i]
            # Calculation of the matrix L by iteration over the columns
            for j in range(i, n):
                R[k][j] = R[k][j] - L[k][i] * R[i][j]
                # calculation of the matrix R by iteration over the columns
    return L, R


def solution(L,R,b):
    n=len(b)
    m=len(b)-1
    y=np.zeros((n))
    x = np.zeros((n))
    for i in range(n):
        for k in range(i):
            b[i] -= L[i][k]*y[k]
        y[i]=(b[i])/L[i][i]
    for i in reversed(range(n)):
        for k in reversed(range(i,n)):
            y[i] -= R[i][k]*x[k]
        x[i]= (y[i])/R[i][i]
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
    A = np.array([[epsilon,0.5],[0.5,0.5]])
    b = np.array([[0.5],[0.25]])
    L,R=LR(A)
    x = solution(L,R,b)
    print(A@x)





    print("Compare:")
    print(np.linalg.inv(L @ R) @ b)
    print("---------")
    print(R)

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
