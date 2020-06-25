
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


def get_lorenz_attractor(r, sigma, b):
    def lorenz_attractor(t, v):
        xp = sigma * (v.x - v.y)
        yp = r * v.x - v.y - v.x * v.z
        zp = v.x * v.y - b * v.z

        return Vec3(xp, yp, zp)

    return lorenz_attractor


def init_conditions(r, b, epsilon=0.0):
    if r > 1:
        a0 = np.sqrt(b * (r - 1))
        v0 = Vec3(a0 + epsilon, a0 + epsilon, r - 1 + epsilon)
    else:
        v0 = Vec3(epsilon, epsilon, epsilon)

    return v0


def homework1(sigma, b) -> None:
    rs = [0.5, 1.17, 1.3456, 25.0, 29.0]

    for r in rs:
        lorenz_attractor = get_lorenz_attractor(r, sigma, b)

        v0 = init_conditions(r, b, epsilon=1e-1)

        # v0 = Vec3(1, 1, 1)
        t = np.linspace(0, 1, 10000)

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


def homework2(sigma, b) -> None:
    r = 26.5
    lorenz_attractor = get_lorenz_attractor(r, sigma, b)

    # v0 = init_conditions(r, b)
    v0 = Vec3(1, 1, 1)

    k = 10000

    # any higher than 1.5 and the system diverges
    t = np.linspace(0, 1.5, k)

    _, vs = zip(*list(solve_rk4(lorenz_attractor, t, v0)))

    # z[i]
    z = [v.z for v in vs]
    # z[k + 1]
    zkpo = [z[i + 1] for i in range(len(z) - 1)]

    plt.plot(z[:-1], zkpo)
    plt.xlabel("$z_k$")
    plt.ylabel("$z_{k+1}$")
    plt.show()


def homework() -> None:
    sigma = 10
    b = 8 / 3
    # homework1(sigma, b)
    homework2(sigma, b)


def main(argv: list) -> int:
    # attendance()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
