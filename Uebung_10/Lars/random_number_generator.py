
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