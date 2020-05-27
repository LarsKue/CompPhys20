
import numpy as np


def is_symmetric(m, rtol=1e-5, atol=1e-8):
    return np.allclose(m, m.T, rtol=rtol, atol=atol)


def praesenz() -> None:
    M = np.array([
        [1, 2, 3, 4, 5],
        [2, 1, 2, 3, 4],
        [3, 2, 1, 2, 3],
        [4, 3, 2, 1, 2],
        [5, 4, 3, 2, 1]
    ])


def main(argv: list) -> int:
    praesenz()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
