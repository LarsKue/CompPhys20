import matplotlib.pyplot as plt
import numpy as np

def integrand_a(a, n, x):
    return x**n / (x+a)

def aufgabeA():
    a = 5
    n_sequence = {1, 5, 10, 20, 30, 50}
    x = np.linspace(0, 1)
    for n in n_sequence:
        plt.plot(x, integrand_a(a, n, x))
    plt.show()

def aufgabeB(y0, a, n0, n1):
    def iteration(vorgaenger, a, n, n1):
        if n1 != n0:
            iteration(1/n - a*vorgaenger, a, n+((n1-n0)/abs(n1-n0)), n1)

def main():
    aufgabeA()
    aufgabeB(10, )

if __name__ == "__main__":
    main()