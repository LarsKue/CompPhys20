from vec3 import Vec3
import matplotlib.pyplot as plt
import math


def leapfrog(r1, r2, v1, v2, time_step):
    r1_neu = r1 + v1 * time_step + kraft(r1, r2) * (time_step ** 2) / 2
    r2_neu = r2 + v2 * time_step + kraft(r2, r1) * (time_step ** 2) / 2
    v1_neu = v1 + time_step * (kraft(r1_neu, r2_neu) + kraft(r1, r2)) / 2
    v2_neu = v2 + time_step * (kraft(r2_neu, r1_neu) + kraft(r2, r1)) / 2
    return r1_neu, r2_neu, v1_neu, v2_neu


def step(r1, r2, v1, v2, time_step):
    v1_neu = v1 - time_step * kraft(r1, r2)
    v2_neu = v2 - time_step * kraft(r2, r1)
    r1_neu = r1 + time_step * v1
    r2_neu = r2 + time_step * v2
    return r1_neu, r2_neu, v1_neu, v2_neu


def kraft(r1, r2):
    r = r1 - r2
    return 1 / r.abs_sq() * r.unit()


def runge(r, v):
    r = r.unit()
    v = v.unit()
    lenz = v.cross(r.cross(v)).abs_sq()
    return math.sqrt(lenz)


def main():
    # G = 1
    # M1 = 1
    # M2 = 1
    R = 1
    v = math.sqrt(1 / (4 * R))
    r1 = Vec3(-R, 0, 0)
    r2 = Vec3(R, 0, 0)
    v1 = Vec3(0, -v, 0)
    v2 = Vec3(0, v, 0)
    time_step = 0.01
    r1_data, r2_data = [], []
    n_steps = 100000
    for i in range(n_steps + 1):
        r1_data.append(r1)
        r2_data.append(r2)
        r1, r2, v1, v2 = leapfrog(r1, r2, v1, v2, time_step)
        # print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * time_step, n_steps * time_step,
        #                                                                      100 * i / n_steps),
        #       end="")
        print(runge(r1, v1))
        print(runge(r2, v2))
    x1_data = [p.x for p in r1_data]
    y1_data = [p.y for p in r1_data]
    x2_data = [p.x for p in r2_data]
    y2_data = [p.y for p in r2_data]
    plt.figure(figsize=(8, 8))
    plt.plot(x1_data, y1_data)
    plt.plot(x2_data, y2_data)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


if __name__ == "__main__":
    main()
