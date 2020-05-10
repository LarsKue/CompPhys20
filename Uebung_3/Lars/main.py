import sys
from particle import Particle
from n_particle_system import NParticleSystem
from vec3 import Vec3

from matplotlib import animation, pyplot as plt
from typing import List, Union


def plot_animation(positions: List[List[Vec3]], time_step: Union[float, int], linetrace: bool = True, show: bool = True,
                   save_as: str = None):
    """
    Plot and optionally save a real-time animation of the positions of multiple particles in a system
    :param positions: List of Frames and Positions. positions[i] is a frame, positions[i][j] a position
    :param time_step: the time step used in the simulation by which the positions were obtained
    :param linetrace: If true, the particles paths will be linetraced
    :param show: If true, the program will pause execution and the plot will be shown once generated
    :param save_as: Optional string to save a video of the animation
    """
    # find number of time steps and particles, exit if there is nothing to do
    n_steps = len(positions)
    if n_steps == 0:
        return
    n_particles = len(positions[0])
    if n_particles == 0:
        return

    # plot artists setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)

    # the particles are represented by big dots
    dots, = ax.plot([], [], lw=0, marker="o")

    # this is used for linetracing
    lines = [ax.plot([], [])[0] for _ in range(n_particles)]
    patches = lines + [dots]

    def init():
        dots.set_data([], [])
        if linetrace:
            for line in lines:
                line.set_data([], [])
        return patches

    def animate(i):
        # data for frame i
        print("\ranimation progress: frame {:d} / {:d} ({:.2f}%)".format(i, n_steps, 100 * i / n_steps), end="")
        # particle positions
        x = [p.x for p in positions[i]]
        y = [p.y for p in positions[i]]

        dots.set_data(x, y)

        if linetrace:
            # line tracing is done by setting the data of all positions up to the current one
            for j in range(n_particles):
                x = [particles[j].x for particles in positions[:i + 1]]
                y = [particles[j].y for particles in positions[:i + 1]]
                lines[j].set_data(x, y)

        return patches

    plt.xlim((-3, 3))
    plt.ylim((-3, 3))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Particle Motion")
    plt.grid()

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=1000 * time_step,
                                   repeat_delay=2000, blit=True)

    if save_as:
        anim.save(save_as, fps=1 / time_step, extra_args=["-vcodec", "libx264"])

    if show:
        plt.show()
    else:
        plt.close(fig)


def run_and_plot(n_body_system, n_steps, h, video_filename):
    positions = [[p.position for p in n_body_system.particles]]
    for i in range(n_steps):
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * h, n_steps * h, 100 * i / n_steps),
              end="")
        n_body_system.step_rk4(h)
        positions.append([p.position for p in n_body_system.particles])

    print("\rcalculation progress: done.")

    plot_animation(positions, h, save_as=video_filename)


def task_a():
    p1 = Particle(Vec3(-0.97000436, 0.24308753, 0.0), Vec3(-0.46620368, -0.43236573, 0.0), 1.0)
    p2 = Particle(Vec3(0.97000436, -0.24308753, 0.0), Vec3(-0.46620368, -0.43236573, 0.0), 1.0)
    p3 = Particle(Vec3(0.0, 0.0, 0.0), Vec3(0.93240737, 0.86473146, 0.0), 1.0)

    n_body_system = NParticleSystem([p1, p2, p3])

    n_steps = 5000
    h = 0.01

    run_and_plot(n_body_system, n_steps, h, "particle_animation.mp4")


def task_b():
    m1 = 3
    m2 = 4
    m3 = 5
    l1 = 3
    l2 = 4
    # l3 is defined by l1 and l2

    # first place the particles
    p1 = Particle(Vec3(0, 0, 0), mass=m2)
    p2 = Particle(Vec3(l1, 0, 0), mass=m3)
    p3 = Particle(Vec3(l1, l2, 0), mass=m1)

    n_body_system = NParticleSystem([p1, p2, p3])

    # then shift the system's center of mass to the origin
    n_body_system.shift_origin(n_body_system.center_of_mass())

    # now simulate
    n_steps = 5000
    h = 0.001

    run_and_plot(n_body_system, n_steps, h, "particle_animation.mp4")


def main(argv: list) -> int:
    # task_a()
    task_b()
    return 0


if __name__ == "__main__":
    main(sys.argv)
