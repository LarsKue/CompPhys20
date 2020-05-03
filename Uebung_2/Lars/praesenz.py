
import sys
import numpy as np
import math
from matplotlib import animation, pyplot as plt
from typing import Iterable, Union, Callable
from vec3 import Vec3
from particle import Particle
import random


class NParticleSimulation:
    def __init__(self, particles: Iterable[Particle]):
        self.particles = list(particles)

    def step(self, h: Union[float, int], method: Callable):
        # get the list of updated particles and then overwrite the old particles
        self.particles = [method(p, h, self.particles) for p in self.particles]

    def __len__(self):
        return len(self.particles)


def plot_animation(positions, time_step, linetrace=True):
    n_particles = len(positions)
    if n_particles == 0:
        return
    n_steps = len(positions[0])
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    dots, = ax.plot([], [], lw=0, marker="o")
    # for linetracing
    lines = [ax.plot([], [])[0] for _ in range(n_particles)]
    patches = lines + [dots]
    # if plot_lines:
    #     for particle in positions:
    #         ax.plot([position.x for position in particle], [position.y for position in particle])

    def init():
        dots.set_data([], [])
        if linetrace:
            for line in lines:
                line.set_data([], [])
        return patches

    def animate(j):
        print("\ranimation progress: frame {:d} / {:d} ({:.2f}%)".format(j, n_steps, 100 * j / n_steps), end="")
        x = [particle[j].x for particle in positions]
        y = [particle[j].y for particle in positions]
        dots.set_data(x, y)
        if linetrace:
            for k in range(len(positions)):
                particle = positions[k]
                line = lines[k]
                line.set_data([p.x for p in particle[:j]], [p.y for p in particle[:j]])
        return patches

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=time_step, blit=True)
    print()
    anim.save("particle_animation.mp4", fps=60, extra_args=["-vcodec", "libx264"])

    plt.grid()
    plt.show()


def main(argv: list) -> int:
    # simulation parameters
    n_steps = 24000
    h = 0.001

    # adjust this for different velocities of the particles
    Particle.G = 1
    M = 1
    R = 1.0

    # analytically, this velocity makes for a circular orbit of both particles around each other
    v = math.sqrt(M * Particle.G / (4 * R))

    # various velocity tests
    # v *= math.sqrt(2)
    # v /= 3

    # # create the particles
    p1 = Particle(Vec3(R, 0.0, 0.0), Vec3(0.0, -v, 0.0), M)
    p2 = Particle(Vec3(-R, 0.0, 0.0), Vec3(0.0, v, 0.0), M)

    # can calculate systems with more than 2 bodies too
    # p3 = Particle(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), 5
    # particles = [Particle(Vec3(random.normalvariate(0.0, R), random.normalvariate(0.0, R), 0.0),
    #                       Vec3(random.normalvariate(0.0, v), random.normalvariate(0.0, v), 0.0), M) for _ in range(200)]

    n_body_system = NParticleSimulation([p1, p2])

    # n_body_system = NParticlesSimulation([p1, p2, p3])
    # n_body_system = NParticlesSimulation(particles)

    def explicit_euler(p, h, particles):
        return p.step_explicit_euler(h, particles)

    # list of positions of the particles at different timesteps
    # dimensions are number of particles, number of time steps
    positions = [[particle.position] for particle in n_body_system.particles]

    t = np.linspace(0, n_steps * h, n_steps)
    last_t = t[0]
    for current_t in t:
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(current_t, t[-1], 100 * current_t / t[-1]), end="")
        h = current_t - last_t

        # update the positions
        n_body_system.step(h, explicit_euler)

        # get the positions for plotting
        for i in range(len(positions)):
            positions[i].append(n_body_system.particles[i].position)

        last_t = current_t
    print()

    # plot an animation, this may take a long time, so comment this out if you want fast (but unvisualized) results
    plot_animation(positions, h, linetrace=True)

    return 0


if __name__ == "__main__":
    main(sys.argv)
