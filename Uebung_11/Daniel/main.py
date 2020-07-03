import numpy as np
from matplotlib import pyplot as plt
import random
import math


def integrate_mc(f, n, xmin=0.0, xmax=1.0):
    """
    Monte-Carlo Integrator Function
    :param f: The function to integrate
    :param n: The number of random samples to take
    :param xmin: Lower integration bound
    :param xmax: Upper integration bound
    :return: The approximate integral of f over the range of xmin - xmax

    Example: Integrate sin(x) over x from 0 to pi with 10000 samples
    >>> import numpy as np
    >>> print(f"{integrate_mc(np.sin, 10000, 0.0, np.pi):.1f}")
    2.0
    """
    # note we consciously chose not to add the parameter x0
    # if you need a specific seed, simply call random.seed(x0) before calling this function
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return (xmax - xmin) * sum(f(random.uniform(xmin, xmax)) for _ in range(n)) / n


def integrate_mc(f, n, xmin=0.0, xmax=1.0):
    """
    Monte-Carlo Integrator Function
    :param f: The function to integrate
    :param n: The number of random samples to take
    :param xmin: Lower integration bound
    :param xmax: Upper integration bound
    :return: The approximate integral of f over the range of xmin - xmax

    Example: Integrate sin(x) over x from 0 to pi with 10000 samples
    >>> import numpy as np
    >>> print(f"{integrate_mc(np.sin, 10000, 0.0, np.pi):.1f}")
    2.0
    """
    # note we consciously chose not to add the parameter x0
    # if you need a specific seed, simply call random.seed(x0) before calling this function
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return (xmax - xmin) * sum(f(random.uniform(xmin, xmax)) for _ in range(n)) / n


def integrate_mc_ndim(f, n, bounds):
    """
    n-dimensional Monte-Carlo Integrator Function
    :param f: The function to integrate
    :param n: The number of random samples to take
    :param bounds: Integration bounds, in order of dimension.
    :return: The approximate integral of f within the specified bounds

    Example: Integrate f(x, y) = sin(xy) over x from 0 to pi and over y from 0 to 0.9 with 100000 samples
    >>> import numpy as np
    >>> def f(x, y): return np.sin(x * y)
    >>> bounds = ((0, np.pi), (0, 0.9))
    >>> print(f"{integrate_mc_ndim(f, 100_000, bounds):.1f}")
    1.4
    """
    volume = math.prod(b[1] - b[0] for b in bounds)

    return volume * sum(f(*[random.uniform(b[0], b[1]) for b in bounds]) for _ in range(n)) / n


def rel_error(uv, ev):
    """
    Calculate the relative error in an uncertain value to an exact value
    :param uv: uncertain value
    :param ev: exact value
    :return: the relative error between the two
    """
    return abs(uv - ev) / max(abs(uv), abs(ev))


def attendance() -> None:

    def compute_print_and_get_relative_error(f, n, ai):
        nm = integrate_mc(f, n)
        re = rel_error(nm, ai)
        print(f"n: {n:8d}   analytical: {ai:9f}   numerical: {nm:9f}   difference: {ai - nm:9f}   relative error: {100 * re:5.2f}%")
        return re

    # use a seed for reproducability
    random.seed(0)

    # choose one
    # cleaner plot
    # ns = [10, 100, 1000, 10_000, 100_000, 1000_000, 10_000_000]
    # more data
    ns = np.logspace(start=1, stop=5, num=30, dtype=int).tolist()
    names = ["$x^2$", "$x^3$", r"$\sin(x)$", r"$\exp(x)$"]
    fs = [lambda x: x ** 2, lambda x: x ** 3, lambda x: np.sin(x), lambda x: np.exp(x)]
    gs = [lambda x: x ** 1, lambda x: x ** (3/2), lambda x: x, lambda x: x**2]
    # these were calculated by hand
    analytical_integrals = [1/3, 0.25, 1 - np.cos(1), np.exp(1) - 1]

    # comment this out if you want one figure per function
    plt.figure(figsize=(10, 10))
    for name, f, ai in zip(names, fs, analytical_integrals):
        rel_errs = [compute_print_and_get_relative_error(f, n, ai) for n in ns]

        # comment these out if you want one figure for all functions
        # plt.figure(figsize=(10, 10))
        # plt.title(name)
        # plt.xscale("log")
        # plt.yscale("log")

        # plot data (do not comment this out)
        plt.plot(ns, rel_errs, linewidth=0, marker="x", label=name)

    # comment these out if you want one figure per function
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    # show the plot (do not comment this out)
    plt.show()



    n=1000
    random.seed(0)
    seeds=np.random.randint(0,1000,int(n))


    def variance(f,seeds,ai,n):
        x=[]
        for i in range(len(seeds)):
            random.seed(seeds[i])
            x.append(integrate_mc(f,n))
        return np.var(x)


    plt.figure(figsize=[10,10])

    for name, f, ai in zip(names, fs, analytical_integrals):
        var = [variance(f, seeds, ai,n) for n in ns]
        plt.scatter(ns,var,label=name)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('number of samples N')
    plt.ylabel('variance')
    plt.ylim(4e-7,3e-2)
    plt.legend()
    plt.savefig("variance.pdf")
    plt.show()


def homework():
    def function(x1,x2):
        return 1/np.pi*np.exp(-x1**2-x2**2)

    random.seed(0)

    mc = integrate_mc_ndim(function,10000,[[-5.0,5.0],[-5.0,5.0]])
    print(mc)


#     def importanceI(x1,x2)


def main(argv: list) -> int:
    attendance()
    # homework()
    return 0


def _test():
    import doctest
    doctest.testmod()


if __name__ == "__main__":
    _test()

    import sys
    main(sys.argv)
