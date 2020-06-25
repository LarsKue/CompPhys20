
import random

from matplotlib import pyplot as plt


class RandomNumberGenerator:
    def __init__(self, a, m, c, seed=0):
        self.a = a
        self.m = m
        self.c = c
        self.number = seed

    def sample(self):
        rv = self.number
        self.number = (self.a * self.number + self.c) % self.m
        return rv

    def sample_normalized(self):
        return self.sample() / (self.m - 1)

    def sample_die(self):
        return int(self.sample_normalized() * 6) + 1


def main(argv: list) -> int:

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

    return 0


if __name__ == "__main__":
    import sys
    main(sys.argv)
