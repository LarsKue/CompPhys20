import numpy as np
import matplotlib as plt
from vec3 import Vec3
import sys


def step(r1, r2, v1, v2, time):
    r1_new = r1 + time * v1
    r2_new = r2 + time * v2
    v1_new = v1 - time * kraft(r1, r2)
    v2_new = v2 - time * kraft(r2, r1)
    return r1_new, r2_new, v1_new, v2_new


def kraft(r1, r2):
    r = r1 - r2
    return r


def main():
    x = 0
    n = 5
    time = 0.01
    R = 1
    v = np.sqrt(1 / 2 * R)
    r1 = Vec3(R, 0, 0)
    r2 = Vec3(-R, 0, 0)
    v1 = Vec3(0, -v, 0)
    v2 = Vec3(0, v, 0)
    for x in range(0, n):
        r1, r2, v1, v2 = step(r1, r2, v1, v2, time)
        x = x + 1
        return r1, r2, v1, v2
    print(r1, r2, v1, v2)


if __name__ == "__main__":
    main()
