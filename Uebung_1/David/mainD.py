import matplotlib.pyplot as plt
import numpy as np


def integrand_a(a, n, x):
    return x ** n / (x + a)


def aufgabe_a():
    a = 5
    n_sequence = {1, 5, 10, 20, 30, 50}
    x = np.linspace(0, 1)
    for n in n_sequence:
        plt.plot(x, integrand_a(a, n, x))
    plt.show()


def aufgabe_b(y0, a, n0, n1):
    def iteration(vorgaenger, a, n, n1):
        if n1 != n:
            return iteration(1 / n - a * vorgaenger, a, n + ((n1 - n0) / abs(n1 - n0)), n1)
        return vorgaenger

    return iteration(y0, a, n0 + ((n1 - n0) / abs(n1 - n0)), n1 + ((n1 - n0) / abs(n1 - n0)))



def main():
    aufgabe_a()
    a = 5
    print(aufgabe_b(np.log((1 + a) / a), 5, 0.0, 30.0))
    print(aufgabe_b(4, 5, 50.0, 30.0))


if __name__ == "__main__":
    main()
