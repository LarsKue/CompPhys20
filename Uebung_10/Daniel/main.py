import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from typing import Callable, Iterable


def presence():
    a=106
    m=6075
    c=1283
    I=[2]
    def iteration(I,a,m,c):
        for j in range(m):
            Ij=np.mod(a*I[j]+c,m)
            I.append(Ij)
        normalizedi=np.array(I)/(m-1)
        return I,normalizedi
    random,randomnor = iteration(I,a,m,c)
    for k in range(m-1):
        plt.scatter(randomnor[k],randomnor[k+1])
    plt.show()
    I1=[2]
    I2=[4]
    random1,randomnor1= iteration(I1,a,m,c)
    random2,randomnor2= iteration(I2,a,m,c)
    plt.plot(randomnor1,randomnor2)
    plt.show()




def main(argv: list) -> int:
    presence()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)