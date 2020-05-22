
import sys
import numpy as np


def praesenz() -> None:
    epsilon = 1e-16

    M = np.array([
        [epsilon, 0.5],
        [0.5, 0.5]
    ])

    y = (epsilon - 1) / (2 * epsilon + 1)
    x = 0.5 - (epsilon - 1) / (2 * epsilon + 1)

    v = np.array([x, y])

    print(f"x = {x:.3f}")
    print(f"y = {y:.3f}")

    b = M @ v

    print(f"M * v = {b}")


def main(argv: list) -> int:
    praesenz()
    return 0


if __name__ == "__main__":
    main(sys.argv)