
import random
import numpy as np
from matplotlib import pyplot as plt

from random_number_generator import RandomNumberGenerator


def attendance():
    a = 106
    m = 6075
    c = 1283

    rng1 = RandomNumberGenerator(a, m, c, random.randint(0, 10000000))
    rng2 = RandomNumberGenerator(a, m, c, 312)

    for _ in range(10):
        # we test the generator against itself and the python random library
        r1 = rng1.sample_normalized()
        r2 = rng2.sample_normalized()
        r3 = random.random()

        print(f"{r1}, {r2}, {r3}")

    # rs = [random.uniform(0, m - 1) for _ in range(10)]
    rs = [rng1.sample() for _ in range(20)]

    ripo = [rs[i + 1] for i in range(len(rs) - 1)]

    # the numbers always (sort of) go clockwise in the plot
    plt.plot(rs[:-1], ripo)
    plt.xlabel("$I_j$")
    plt.ylabel("$I_{j+1}$")
    plt.show()

    roll_results = [sum(rng1.sample_die() for _ in range(10)) for _ in range(100_000)]
    random_results = [sum(random.randint(1, 6) for _ in range(10)) for _ in range(100_000)]

    # our random number generator often shows 1-3 peaks in the center of the distribution
    # whereas random.randint does not exhibit this property
    # our rng also does not curve out smoothly at the "edges" of the distribution
    plt.figure(figsize=(8, 8))
    plt.hist(roll_results, bins=list(range(10, 60 + 1)))
    plt.xlabel("Roll Result")
    plt.ylabel("Frequency")

    plt.figure(figsize=(8, 8))
    plt.hist(random_results, bins=list(range(10, 60 + 1)))
    plt.xlabel("Roll Result")
    plt.ylabel("Frequency")
    plt.show()


def homework1():

    a = 0.5

    # from normalization requirement
    b = 2 / a ** 2

    def p(x):
        return b * x

    def rejection_method(random_numbers):
        for rand_num in random_numbers:
            if rand_num > a:
                continue
            elif rand_num < 0:
                continue
            else:
                if random.uniform(0, a) < rand_num:
                    yield rand_num

    # a good fit is achieved with n >= 1e6
    n = 1000_000
    r = [random.uniform(0, 1) for _ in range(n)]

    n_bins = 300

    plt.hist(r, bins=n_bins)
    plt.title("Set of uniformly distributed random numbers")
    plt.show()

    r = list(rejection_method(r))

    x = np.linspace(0, a, 1000)

    # overplot equation 2.1 scaled to set and bin size
    plt.plot(x, n * p(x) / p(n_bins), label="fit")
    plt.hist(r, bins=n_bins, label="set")
    plt.legend()
    plt.title("Set of random numbers, distributed according to $p(x)$")
    plt.show()


def homework2():

    def f(x):
        return np.sqrt(1 - x ** 2)

    def rejection_method(random_numbers):
        for rand_num in random_numbers:
            if random.uniform(0, 1) <



def homework():
    homework1()
    homework2()


def main(argv: list) -> int:
    # attendance()
    homework()
    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
