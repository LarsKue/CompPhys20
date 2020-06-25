import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from typing import Callable, Iterable
from mpl_toolkits.mplot3d import Axes3D


def polynomial(l,b,s,r):
    return l**3+(1+b+s)*l**2+b*(s+r)*l+2*s*b*(r-1)


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


def presence():
    lamb= np.linspace(-12,3,2000)
    sigma=10
    b =8/3
    r = [0.1,0.8,1,1.5]
    for x in r:
        plt.plot(lamb,polynomial(lamb,b,sigma,x), label=x)
    plt.legend()
    plt.show()


def homework():
    ts = np.linspace(0, 10, 1000)
    b = 8/3
    sigma = 20
    e = 0.1
    r = [0.5, 1.17, 1.3456, 25, 29]


    for x in r:
        def attractor(t, state):
            return np.array([-sigma * (state[0] - state[1]), x * state[0] - state[1] - state[0] * state[2],
                             state[0] * state[1] - b * state[2]])


        if x >= 1:
            initial= np.array([np.sqrt(b*(x-1))+e,np.sqrt(b*(x-1))+e,x-1+e])
        else:
            initial=np.array([e,e,e])


        _, time_evolution = zip(*list(solve_rk4(attractor,ts,initial)))
        xpos=[]
        ypos=[]
        zpos=[]
        for f in time_evolution:
            xpos.append(f[0])
            ypos.append(f[1])
            zpos.append(f[2])
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xpos,ypos,zpos, alpha=0.5)
        plt.show()


def main(argv: list) -> int:
    #presence()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
