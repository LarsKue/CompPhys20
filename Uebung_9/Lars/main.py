
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vec3 import Vec3

from typing import Callable, Iterable


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


def attendance() -> None:

    def P(lam, b, sig, r):
        return lam ** 3 + (1 + b + sig) * lam ** 2 + b * (sig + r) * lam + 2 * sig * b * (r - 1)

    sig = 10
    b = 8 / 3

    l = np.linspace(-12, 3, 100)
    r = np.linspace(0, 1.8, 100)

    l, r = np.meshgrid(l, r)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    zdata = P(l, b, sig, r)
    ax.plot_surface(l, r, zdata, alpha=0.5)
    ax.contour(l, r, zdata, levels=[0])
    # ax.plot_wireframe(l, r, np.zeros_like(l), color="grey", alpha=0.3)
    ax.set_xlabel("l")
    ax.set_ylabel("r")
    ax.set_zlabel(r"$P(\lambda)$")

    plt.show()


def homework() -> None:
    sigma = 10
    b = 8 / 3

    r = 1.5

    def lorenz_attractor(t, v):
        xp = sigma * (v.x - v.y)
        yp = r * v.x - v.y - v.x * v.z
        zp = v.x * v.y - b * v.z

        return Vec3(xp, yp, zp)

    a0 = np.sqrt(b * (r - 1))
    v0 = Vec3(a0, a0, r - 1)
    v0 = Vec3(1, 1, 1)
    t = np.linspace(0, 1, 1000)

    _, vs = zip(*list(solve_rk4(lorenz_attractor, t, v0)))

    x = [v.x for v in vs]
    y = [v.y for v in vs]
    z = [v.z for v in vs]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def main(argv: list) -> int:
    # attendance()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
