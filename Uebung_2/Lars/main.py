import sys
import os
import numpy as np
import math
from matplotlib import animation, pyplot as plt
from vec3 import Vec3
from particle import Particle
from n_particle_simulation import NParticleSimulation
import random  # for random particle generation

"""
Submission for Uebung 2 for Computational Physics 2020
Group Members:
Daniel Kreuzberger
Lars Kuehmichel
David Weinand
"""


"""
How to use:
Simply run this file as-is or scroll down to the main function at the bottom of this file
to make parameter changes. All results are already on git, so you can play around with this as you like without
fear of overwriting anything.

Homework comments:
Clearly the integration error is dependent on the time resolution.
This is especially visible for the circular orbit, where the 
eccentricity only stays very close to 1 when the time step is
smaller than or equal to 0.01 (explicit euler)
or 0.1 (kdk leapfrog)

The integration method also makes a big difference here, the system is much more stable
with kdk leapfrog since it has a higher time resolution in the velocity and uses
an implicit-ish method to calculate the position.

You can clearly see a line in the log-log-plots for the energy error,
showing how important it is to select a proper time step. For the leapfrog
integrator, reducing the timestep by 3 orders of magnitude yielded an energy error
roughly 10 orders of magnitude smaller.

You can view the plots yourself by looking at the .png files in this folder. We only made energy plots
for the circular orbits, since any other kind of motion does not make much sense to analyze this way.
However, we plotted the squared eccentricity (which is linearly proportional to the energy error)
for all tests we performed, so if need be you can view those in the videos (more on that later).

The results are consistent with what you would expect, for small time steps and better integration methods,
the eccentricity stays very close to 1 (alternating in a sinusodial fashion), meaning the orbit stays very much
circular. For larger time steps, especially with the explicit euler method, the eccentricity sways a lot more,
meaning the error in energy is high and the orbit no longer approximately circular, which is also what you can see
in the animations.

We provided you with a bunch of cool videos and animations for all the tests we performed.
Most notable are
videos_kdk_leapfrog/h=0.100s.v0.mp4
and
videos_explicit_euler/h=0.100s.v0.mp4

where you can see how much more stable leapfrog is even with a larger time step.
We also performed some tests using implicit euler integration, you can see those in their respective videos


All the videos are named after their time step and initial velocity parameters.

They are also all "real-time" meaning those with large time steps have a smaller number of frames per second,
but this shows you exactly when and where a data point is recorded.
"""


def plot_animation(positions, sq_eccentricities, time_step, linetrace=True, video_filename="particle_animation.mp4",
                   show=True):
    n_particles = len(positions)
    if n_particles == 0:
        return
    n_steps = len(positions[0])
    fig = plt.figure(figsize=(16, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlim((-3, 3))
    ax1.set_ylim((-3, 3))
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlim((-0.1, time_step * n_steps + 0.1))

    # this needs to be set manually, unfortunately. Use the static plot in the end to check which values to set this to
    ax2.set_ylim((0, 10))  # lower and higher velocity
    # ax2.set_ylim((0.989, 1.011))  # circular orbit
    ax2.set_xlabel("t")
    ax2.set_ylabel("$e^2$")

    dots, = ax1.plot([], [], lw=0, marker="o")
    eccs = [ax2.plot([], [])[0] for _ in range(n_particles)]

    # for linetracing
    lines = [ax1.plot([], [])[0] for _ in range(n_particles)]
    patches = lines + [dots] + eccs

    def init():
        dots.set_data([], [])
        for ecc in eccs:
            ecc.set_data([], [])
        if linetrace:
            for line in lines:
                line.set_data([], [])
        return patches

    def animate(j):
        print("\ranimation progress: frame {:d} / {:d} ({:.2f}%)".format(j, n_steps, 100 * j / n_steps), end="")
        x = [particle[j].x for particle in positions]
        y = [particle[j].y for particle in positions]
        dots.set_data(x, y)
        for k in range(len(sq_eccentricities)):
            eccs[k].set_data([k * time_step for k in range(j)], [sq_eccentricities[k][:j]])
        if linetrace:
            for k in range(len(positions)):
                particle = positions[k]
                line = lines[k]
                line.set_data([p.x for p in particle[:j + 1]], [p.y for p in particle[:j + 1]])
        return patches

    ax1.grid()
    ax1.set_title("Particle Motion")
    ax2.grid()
    ax2.set_title("Eccentricities")

    anim1 = animation.FuncAnimation(fig, animate, init_func=init, frames=n_steps, interval=1000 * time_step,
                                    repeat_delay=2000, blit=True)
    print()

    # comment this out if you don't want to overwrite videos
    anim1.save(video_filename, fps=1 / time_step, extra_args=["-vcodec", "libx264"])
    if show:
        fig.show()
    else:
        plt.close(fig)


def run_simulation(n_steps, h, M, R, v, video_filename, show=True):
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

    # list of positions of the particles at different timesteps
    # dimensions are number of particles, number of time steps
    positions = [[particle.position] for particle in n_body_system.particles]
    ecc1 = []
    ecc2 = []

    t = np.linspace(0, n_steps * h, n_steps)
    for current_t in t:
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(current_t, t[-1], 100 * current_t / t[-1]),
              end="")

        # update the positions
        # n_body_system.step_explicit_euler(h)

        # tolerance may need manual tuning depending on time step
        # n_body_system.step_implicit_euler(h, sq_tolerance=1e-3)

        # n_body_system.step_kdk_leapfrog(h)

        n_body_system.step_rk4(h)

        # record the eccentricities for plotting later
        ecc1.append(n_body_system.particles[0].eccentricity(n_body_system.particles[1]))
        ecc2.append(n_body_system.particles[1].eccentricity(n_body_system.particles[0]))

        # get the positions for plotting
        for i in range(len(positions)):
            positions[i].append(n_body_system.particles[i].position)

        last_t = current_t
    print()

    # plot an animation, this may take a long time, so comment this in if you want fancy animated results
    plot_animation(positions, [[e.abs_sq() for e in ecc1], [e.abs_sq() for e in ecc2]], h, linetrace=True,
                   video_filename=video_filename, show=show)

    # statically plot the eccentricities
    # this is useful if you set show to True and want to examine a single simulation
    fig = plt.figure(figsize=(10, 7))
    plt.plot(t, [p.abs_sq() for p in ecc1], label="Particle 1")
    plt.plot(t, [p.abs_sq() for p in ecc2], label="Particle 2")
    plt.xlabel("t")
    plt.ylabel("e^2")
    plt.title("Squared Eccentricities")
    plt.legend(loc="lower left")
    plt.savefig("e_sq.png")

    """
    In our example, the error in energy simplifies to
    delta E = 4 * |e^2 - 1|
    where e is the eccentricity.
    
    Two orbits are complete after 8*pi seconds have passed.
    """

    # idx = int(8 * math.pi / h)
    #
    # # ecc1 and ecc2 are roughly equal for all simulations
    # delta_E = 4 * abs(ecc1[idx].abs_sq() - 1)
    delta_E = 0

    if show:
        plt.show()
    else:
        plt.close(fig)

    return delta_E


def main(argv: list) -> int:
    # simulation parameters
    ns = [50, 500, 5000, 50000]
    hs = [1, 0.1, 0.01, 0.001]

    # adjust this for different velocities of the particles
    Particle.G = 1
    M = 1
    R = 1.0

    # analytically, this velocity makes for a circular orbit of both particles around each other
    v0 = math.sqrt(M * Particle.G / (4 * R))
    # use these for animation generation
    # vs = [v0, v0 * math.sqrt(2), v0 / 3]
    # v_names = ["v0", "sqrt(2)v0", "v0 div 3"]

    # use these to generate Energy-only plots
    vs = [v0]
    v_names = ["v0"]

    directory = "videos/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        directory = input("Directory 'videos/' already exists. Please input the name of a replacement videos path or press Enter to overwrite.")

        if not directory:
            directory = "videos/"

        if not directory.endswith("/"):
            directory += "/"

    delta_energy = []
    for n_steps, h in zip(ns, hs):
        for v, v_name in zip(vs, v_names):
            # generate a bunch of animations for the above simulation parameters and save them in the directory videos/
            video_filename = directory + "h={:.3f}s.".format(h) + v_name + ".mp4"
            current_delta_energy = run_simulation(n_steps, h, M, R, v, video_filename, show=False)
            delta_energy.append(current_delta_energy)

    plt.figure(figsize=(8, 8))
    plt.plot(hs, delta_energy, lw=0, marker="o")
    plt.xlabel("h")
    plt.ylabel(r"$\Delta E$")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Energy Error vs Time Resolution")
    plt.savefig("deltaE.png")
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv)
