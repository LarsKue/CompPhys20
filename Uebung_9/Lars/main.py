
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def attendance() -> None:

    def P(lam, b, sig, r):
        return lam ** 3 + (1 + b + sig) * lam ** 2 + b * (sig + r) * lam + 2 * sig * b * (r - 1)

    sig = 10
    b = 8 / 3

    l = np.linspace(-15, 5, 100)
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


def main(argv: list) -> int:
    attendance()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
