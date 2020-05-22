import numpy as np
import matplotlib as plt
import math
import sys


def LR(A):
    n = len(A)
    R = A
    L = np.identity(n)

    for i in range(n-1):
        for k in range(i+1,n):
            L[k][i]= R[k][i] / R[i][i]
            for j in range(i,n):
                R[k][j] = R[k][j]-L[k][i]*R[i][j]
    return L,R


def solution(L,b):
    n=len(b)
    x=np.zeros((n,1))
    print(x)
    for i in range(n):
        rest=0
        x[i]=(b[i][0])/L[i][i]
    return(x)


def presence():
    A = np.array([[1e-6,0.5],[0.5,0.5]])
    b = np.array([[0.5],[0.25]])
    L,R=LR(A)
    print(R)
    x=solution(L,b)
    print(x)



def main():
    presence()
    #taska()
    #taskb()
    return 0


if __name__ == "__main__":
    main()