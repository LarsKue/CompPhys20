from vec3 import Vec3
import matplotlib.pyplot as plt
import math
import numpy as np


def step1(r1, r2, v1, v2, time_step):
    r1_neu = r1 + time_step * v1
    r2_neu = r2 + time_step * v2
    v1_neu = v1 - time_step * kraft(r1, r2)
    v2_neu = v2 - time_step * kraft(r2, r1)
    return r1_neu, r2_neu, v1_neu, v2_neu


def step(r1, r2, v1, v2, time_step):
    v1_neu = v1 - time_step * kraft(r1, r2)
    v2_neu = v2 - time_step * kraft(r2, r1)
    r1_neu = r1 + time_step * v1
    r2_neu = r2 + time_step * v2
    return r1_neu, r2_neu, v1_neu, v2_neu


def leapfrog(R, v, time_step, n_steps):
    r1 = Vec3(-R, 0, 0)
    r2 = Vec3(R, 0, 0)
    v1 = Vec3(0, -v, 0)
    v2 = Vec3(0, v, 0)
    r1_neu, r2_neu, v1, v2 = step(r1, r2, v1, v2,(time_step/2))
    r1_data, r2_data = [], []
    runge1, runge2 = [], []
    eccentr1, eccentr2 = [], []
    r1_data.append(r1)
    r2_data.append(r2)
    rlv1 = Vec3.cross(v1, Vec3.cross(r1, v1)) - r1
    rlv2 = Vec3.cross(v2, Vec3.cross(r2, v2)) - r2
    ecc1 = math.sqrt(rlv1.abs_sq())
    ecc2 = math.sqrt(rlv2.abs_sq())
    runge1.append(rlv1)
    runge2.append(rlv2)
    eccentr1.append(ecc1)
    eccentr2.append(ecc2)
    for i in range(n_steps-1):
        r1, r2, v1, v2 = step1(r1, r2, v1, v2, time_step)
        r1_data.append(r1)
        r2_data.append(r2)
        rlv1 = Vec3.cross(v1, Vec3.cross(r1, v1)) - r1
        rlv2 = Vec3.cross(v2, Vec3.cross(r2, v2)) - r2
        ecc1 = math.sqrt(rlv1.abs_sq())
        ecc2 = math.sqrt(rlv2.abs_sq())
        runge1.append(rlv1)
        runge2.append(rlv2)
        eccentr1.append(ecc1)
        eccentr2.append(ecc2)
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * time_step, n_steps * time_step,
                                                                             100 * i / n_steps),
              end="")
    r1, r2, v1_neu, v2_neu = step(r1, r2, v1, v2, (time_step / 2))
    r1_data.append(r1)
    r2_data.append(r2)
    rlv1 = Vec3.cross(v1, Vec3.cross(r1, v1)) - r1
    rlv2 = Vec3.cross(v2, Vec3.cross(r2, v2)) - r2
    ecc1 = math.sqrt(rlv1.abs_sq())
    ecc2 = math.sqrt(rlv2.abs_sq())
    runge1.append(rlv1)
    runge2.append(rlv2)
    eccentr1.append(ecc1)
    eccentr2.append(ecc2)
    return r1, r2, v1, v2


def kraft(r1, r2):
    r = r1 - r2
    return 1 / r.abs_sq() * r.unit()


def calculate(R, v, time_step, n_steps):
    # G = 1
    # M1 = 1
    # M2 = 1
    v_low = v / np.sqrt(2)
    v_high = v * 2
    v_third = v / 3
    r1 = Vec3(-R, 0, 0)
    r2 = Vec3(R, 0, 0)
    v1 = Vec3(0, -v, 0)
    v2 = Vec3(0, v, 0)
    r1_data, r2_data = [], []
    runge1, runge2 = [], []
    eccentr1, eccentr2 = [], []
    for i in range(n_steps + 1):
        r1_data.append(r1)
        r2_data.append(r2)
        rlv1 = Vec3.cross(v1, Vec3.cross(r1, v1)) - r1
        rlv2 = Vec3.cross(v2, Vec3.cross(r2, v2)) - r2
        ecc1 = math.sqrt(rlv1.abs_sq())
        ecc2 = math.sqrt(rlv2.abs_sq())
        runge1.append(rlv1)
        runge2.append(rlv2)
        eccentr1.append(ecc1)
        eccentr2.append(ecc2)
        r1, r2, v1, v2 = step(r1, r2, v1, v2, time_step)
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * time_step, n_steps * time_step,
                                                                             100 * i / n_steps),
              end="")
    # return r1_data, r2_data, eccentr1, eccentr2
    return r1, r2, v1, v2


def plot(R, v, time_step, n_steps):
    r1_data, r2_data, eccentr1, eccentr2 = calculate(R, v, time_step, n_steps)
    x1_data = [p.x for p in r1_data]
    y1_data = [p.y for p in r1_data]
    x2_data = [p.x for p in r2_data]
    y2_data = [p.y for p in r2_data]
    plt.figure(1, figsize=(8, 8))
    plt.plot(x1_data, y1_data)
    plt.plot(x2_data, y2_data)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    # plt.savefig('velocity_third.pdf')
    plt.show()
    plt.figure(2, figsize=(8, 8))
    plt.plot(np.linspace(0, len(eccentr1), len(eccentr1), ), eccentr1)
    plt.plot(np.linspace(0, len(eccentr2), len(eccentr2), ), eccentr2)
    # plt.savefig('eccentricity_third.pdf')
    plt.show()


def exercise_a(R, v, time_step, n_steps):
    energy0 = (v ** 2 / 2) - 1 / R
    energy1 = []
    energy2 = []
    energyerr1 = []
    energyerr2 = []
    time = []
    i = 0
    for i in range(0, 5):
        r1, r2, v1, v2 = calculate(R, v, time_step, n_steps)
        en1 = (v1.abs_sq() / 2) - 1 / math.sqrt(r1.abs_sq())
        en2 = (v2.abs_sq() / 2) - 1 / math.sqrt(r2.abs_sq())
        err1 = abs(en1 - energy0) / abs(energy0)
        err2 = abs(en2 - energy0) / abs(energy0)
        energy1.append(en1)
        energy2.append(en2)
        energyerr1.append(err1)
        energyerr2.append(err2)
        time.append(time_step)
        time_step = time_step / 10
        n_steps = n_steps * 10
        i = i + 1
    plt.figure(figsize=(8, 8))
    plt.loglog(time, energyerr1)
    plt.loglog(time, energyerr2)
    plt.xlabel('timestep')
    plt.ylabel('energy-error')
    plt.savefig('enerr_v005.pdf')
    print(time)
    plt.show()


def exercise_b(R, v, time_step, n_steps):
    energy0 = (v ** 2 / 2) - 1 / R
    energy1 = []
    energy2 = []
    energyerr1 = []
    energyerr2 = []
    time = []
    i = 0
    for i in range(0, 5):
        r1, r2, v1, v2 = leapfrog(R, v, time_step, n_steps)
        en1 = (v1.abs_sq() / 2) - 1 / math.sqrt(r1.abs_sq())
        en2 = (v2.abs_sq() / 2) - 1 / math.sqrt(r2.abs_sq())
        err1 = abs(en1 - energy0) / abs(energy0)
        err2 = abs(en2 - energy0) / abs(energy0)
        energy1.append(en1)
        energy2.append(en2)
        energyerr1.append(err1)
        energyerr2.append(err2)
        time.append(time_step)
        time_step = time_step / 10
        n_steps = n_steps * 10
        i = i + 1
    plt.figure(figsize=(8, 8))
    plt.loglog(time, energyerr1)
    plt.loglog(time, energyerr2)
    plt.xlabel('timestep')
    plt.ylabel('energy-error')
    plt.savefig('enerr_v0b.pdf')
    print(time)
    plt.show()


R = 1
v = math.sqrt(1 / (4 * R))
time_step = 0.1
n_steps = 10

if __name__ == "__main__":
    exercise_b(R, v, time_step, n_steps)
