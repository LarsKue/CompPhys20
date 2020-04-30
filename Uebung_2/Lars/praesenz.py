
import sys
import numpy as np
import math
from matplotlib import animation, pyplot as plt
from typing import Iterable, Union, Callable
from vec3 import Vec3
from particle import Particle


class NParticlesSimulation:
    def __init__(self, particles: Iterable[Particle]):
        self.particles = list(particles)

    def step(self, h: Union[float, int], method: Callable):
        # get the list of updated particles and then overwrite the old particles
        self.particles = [method(p, h, self.particles) for p in self.particles]

    def __len__(self):
        return len(self.particles)


def main(argv: list) -> int:
    # adjust this for different velocities of the particles
    Particle.G = 4
    M = 1
    R = 1.0
    # analytically, this velocity makes for a circular orbit of both particles around each other
    v = math.sqrt(M * Particle.G / (4 * R))
    # create the particles
    p1 = Particle(Vec3(R, 0.0, 0.0), Vec3(0.0, -v, 0.0), M)
    p2 = Particle(Vec3(-R, 0.0, 0.0), Vec3(0.0, v, 0.0), M)
    # p3 = Particle(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), 5)

    n_body_system = NParticlesSimulation([p1, p2])

    def explicit_euler(p, h, particles):
        return p.step_explicit_euler(h, particles)

    n = 10000
    h = 0.01
    t = np.linspace(0, n * h, n)
    positions = [[particle.position] for particle in n_body_system.particles]

    last_t = t[0]
    for current_t in t:
        print("\rprogress: t = {:.2f} / {:.2f}".format(current_t, t[-1]), end="")
        h = current_t - last_t

        # update the positions
        n_body_system.step(h, explicit_euler)

        # get the positions for plotting
        for i in range(len(positions)):
            positions[i].append(n_body_system.particles[i].position)

        last_t = current_t

    def plot_animation(linetrace=True):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(xlim=(-3, 3), ylim=(-3, 3))
        dots, = ax.plot([], [], lw=0, marker="o")
        # for linetracing
        lines = [ax.plot([], [])[0] for _ in range(len(n_body_system))]
        print(len(lines))
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
            x = [particle[j].x for particle in positions]
            y = [particle[j].y for particle in positions]
            dots.set_data(x, y)
            if linetrace:
                for k in range(len(positions)):
                    particle = positions[k]
                    line = lines[k]
                    line.set_data([p.x for p in particle[:j]], [p.y for p in particle[:j]])
            return patches

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n, interval=h, blit=True)

        anim.save("particle_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])

        plt.show()

    # plot an animation, this may take a long time, so comment this out if you want fast (but unvisualized) results
    plot_animation()

    return 0


if __name__ == "__main__":
    main(sys.argv)
