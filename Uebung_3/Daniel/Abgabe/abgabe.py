import numpy as np
import sys
import math
from particle import Particle
from n_particle_simulation import NParticleSimulation
from vec3 import Vec3
from matplotlib import animation, pyplot as plt
from typing import List, Union, Callable


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

    #  print("\ranimation progress: done.")

    if show:
        print("showing figure")
        plt.show()
    else:
        plt.close(fig)


def distance(v1, v2):
    return math.sqrt(Vec3.abs_sq(v1 - v2))


def energie_pot(pos_alt, pos_neu, massen):
    pot = 0
    for x in range(3):
        for y in range(3):
            # If Abgfrage, damit keine potentielle Energie doppelt berechnet wird
            if x < y:
                # die potentielle Energie wird nach der Formel: m1 * m2 * (1/r1 - 1/r2) berechnet
                pot += massen[x] * massen[y] * ((1 / distance(pos_alt[x], pos_alt[y])) - (1 / distance(pos_neu[x], pos_neu[y])))
    return pot


def energie_kin(massen, geschw):
    kin = 0
    for x in range(3):
         # kinetische Energie = 1/2 * m * v²
        kin += 1/2 * massen[x] * Vec3.abs_sq(geschw[x])
    return kin


def minima_suchen(pos, h):
    # in dieser Funktion wird nach Minima gesucht in den übergebenen Distanzen der einzelnen Körper gesucht
    minima = []
    runter = pos[0] > pos[1]
    for x in range(len(pos) - 1):
        if runter is True and pos[x] < pos[x + 1]:
            minima.append(x * h)
        runter = pos[x] > pos[x + 1]
    return minima


def minima_der_minima(minima):
    # in dieser Funktion werden die ersten 5 Näherungen der Körper aus allen Minima der Distanzen herausgesucht
    mini_minima = []
    for c in range(5):
        for x in range(len(minima)):
            if minima[x] == min(minima):
                mini_minima.append(minima[x])
                minima[x] = max(minima)
                break
    return mini_minima


def rk4_step(y0, x0, f, h, *args, **kwargs):
    """ Simple python implementation for one RK4 step.
        Inputs:
            y_0    - M x 1 numpy array specifying all variables of the ODE at the current time step
            x_0    - current time step
            f      - function that calculates the derivates of all variables of the ODE
            h      - time step size
            f_args - Dictionary of additional arguments to be passed to the function f
        Output:
            yp1 - M x 1 numpy array of variables at time step x0 + h
            xp1 - time step x0+h
    """
    k1 = h * f(y0, x0, *args, **kwargs)
    k2 = h * f(y0 + k1 / 2., x0 + h / 2., *args, **kwargs)
    k3 = h * f(y0 + k2 / 2., x0 + h / 2., *args, **kwargs)
    k4 = h * f(y0 + k3, x0 + h, *args, **kwargs)
    xp1 = x0 + h
    yp1 = y0 + 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return yp1, xp1


def rk4(y0, x0, f, h, n, *args, **kwargs):
    """ Simple implementation of RK4
        Inputs:
             y_0    - M x 1 numpy array specifying all variables of the ODE at the current time step
            x_0    - current time step
            f      - function that calculates the derivates of all variables of the ODE
            h      - time step size
            n      - number of steps
            f_args - Dictionary of additional arguments to be passed to the function f
        Output:
            yn - N+1 x M numpy array with the results of the integration for every time step (includes y0)
            xn - N+1 x 1 numpy array with the time step value (includes start x0)
    """
    yn = np.zeros((n + 1, y0.shape[0]))
    xn = np.zeros(n + 1)
    yn[0, :] = y0
    xn[0] = x0
    for m in np.arange(1, n + 1, 1):
        yn[m, :], xn[m] = rk4_step(yn[m - 1, :], xn[m - 1], f, h, *args, **kwargs)
    return yn, xn


def exact(x, t):
    return np.exp(-t * x)


# Be advised that the integration can take a while for large values of n (e.g >=10^5).
def deviation(y, _, r):
    return -r * y


def praesenz(y0: Union[float, int], x0: Union[float, int], x1: Union[float, int], f: Callable, f_exact: Callable):
    r = 1
    hs = [1, 0.1, 0.01, 0.001, 0.0001]

    plt.figure(figsize=(8, 8))
    for h in hs:
        n = int((x1 - x0) / h)
        x = np.linspace(x0, x1, n)
        # solve the equation
        yn, xn = rk4(np.array([y0]), np.array([x0]), f, h, n, r)
        yn = yn.reshape(yn.shape[0],)
        err = np.mean(np.abs(yn - f_exact(xn, r)) / f_exact(xn, r))
        plt.plot(xn, yn, label='stepsize={}, mean error={}'.format(h, err))
    # plot the exact solution
    n_max = int((x1 - x0) / hs[-1])
    x = np.linspace(x0, x1, n_max)
    plt.plot(x, f_exact(x, r), label="exact")
    plt.legend(loc="upper right")
    plt.title('rk4 with different step sizes')
    plt.savefig('rk4step.pdf')
    plt.show()


'''
In the presence task, we implemented the given rk4 algorithm. As you can see in the plot, the mean relative error is 
getting smaller for smaller step sizes. You can also see that for small step sizes, the curve is very close to the exact
curve.
'''


def task_a():
    p1 = Particle(Vec3(-0.97000436, 0.24308753, 0.0), Vec3(-0.46620368, -0.43236573, 0.0), 1.0)
    p2 = Particle(Vec3(0.97000436, -0.24308753, 0.0), Vec3(-0.46620368, -0.43236573, 0.0), 1.0)
    p3 = Particle(Vec3(0.0, 0.0, 0.0), Vec3(0.93240737, 0.96473146, 0.0), 1.0)

    n_body_system = NParticleSimulation([p1, p2, p3])

    n_steps = 5000
    h = 0.0001

    positions = [[p.position for p in n_body_system.particles]]
    for i in range(n_steps):
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * h, n_steps * h, 100 * i / n_steps),
              end="")
        n_body_system.step_rk4(h)
        positions.append([p.position for p in n_body_system.particles])

    print("\rcalculation progress: done.")

    plot_animation(positions, h, save_as="particle_animation.mp4")


'''
For this task we implemented our own rk4 algorithm. With this algorithm, we can simulate every n particle system.
For smaller step sizes the curves behave like a lying eight. If you take bigger step sizes,
you can see that there is a growing integration error, which leads to a slightly incorrect shape of the curves.
(If the animation don't work on your computer, please check out if it works if this part is commented out
(it is in the animate subprogram:
if save_as:
    anim.save(save_as, fps=1 / time_step, extra_args=["-vcodec", "libx264"])

    print("\ranimation progress: done.")
This problem occured for some group members
'''


def task_b():
    p1 = Particle(Vec3(-1, 1, 0.0), Vec3(0, 0, 0.0), 5.0)
    p2 = Particle(Vec3(3, 1, 0.0), Vec3(0, 0, 0.0), 4.0)
    p3 = Particle(Vec3(-1.0, -2.0, 0.0), Vec3(0, 0, 0.0), 3.0)

    # This particle setup was calculated by hand
    n_body_system = NParticleSimulation([p1, p2, p3])

    n_steps = 100000
    h = 0.0001

    # in unserem Beispile ist e_kin_alt natürlich 0
    e_kin_alt = energie_kin([p.mass for p in n_body_system.particles], [p.velocity for p in n_body_system.particles])

    dist1 = [distance(p1.position, p2.position)]
    dist2 = [distance(p2.position, p3.position)]
    dist3 = [distance(p1.position, p3.position)]

    positions = [[p.position for p in n_body_system.particles]]

    # Berechnung des totalen Energiefehlers mit Hilfe der beiden Energie Funktionen
    energy_tot = [energie_pot(positions[0], positions[0], [p1.mass, p2.mass, p3.mass]) + energie_kin(
        [p.mass for p in n_body_system.particles], [p.velocity for p in n_body_system.particles]) - e_kin_alt]


    for i in range(n_steps):
        print("\rcalculation progress: t = {:.2f} / {:.2f} ({:.2f}%)".format(i * h, n_steps * h, 100 * i / n_steps),
              end="")
        n_body_system.step_rk4(h)

        positions.append([p.position for p in n_body_system.particles])
        dist1.append(distance(p1.position, p2.position))
        dist2.append(distance(p2.position, p3.position))
        dist3.append(distance(p1.position, p3.position))

        energy_tot.append(
            energie_pot(positions[0], positions[i + 1], [p.mass for p in n_body_system.particles]) + energie_kin(
                [p.mass for p in n_body_system.particles],
                [p.velocity for p in n_body_system.particles]) - e_kin_alt)

    minima = (minima_der_minima(minima_suchen(dist1, h) + minima_suchen(dist2, h) + minima_suchen(dist3, h)))

    with open('mini{}.txt'.format(h), 'w+') as schreiberling:
        schreiberling.writelines("%s\n" % element for element in minima)

    print("\rcalculation progress: done.")

    xax = np.linspace(0, h * n_steps, int(n_steps + 1))

    plt.figure(1, figsize=(8, 8))

    plt.plot(xax, dist1, label='distance part 1, part 2')
    plt.plot(xax, dist2, label='distance part 2, part 3')
    plt.plot(xax, dist3, label='distance part 1, part 3')

    plt.yscale('log')
    plt.xlabel('time')
    plt.ylabel('distance')
    plt.legend()

    plt.savefig('distances{}.png'.format(h))

    plt.figure(2, figsize=(8, 8))

    plt.plot(xax, energy_tot)

    plt.yscale('log')
    plt.xlabel('timestep')
    plt.ylabel('error of the total energy of the system')
    plt.title('development of the total energy over time')

    plt.savefig('energyerror{}.png'.format(h))

    plot_animation(positions, h, save_as="particle_animation{}.mp4".format(h))


'''
For a step size smaller than 0.001, you can see the first five points of minimum separation. The program is 
not perfect, which you can see from the energy error. Every time any two particles come close to each other, the 
error in energy gets higher. This eventually leads to the two particles slinging each other with a big gain 
of energy. This cannot be prevented through lowering the time step. Using a different force evaluation function
which does not diverge for small distances would solve this, however. A smaller step size only pushes this effect to
a later time. In the file name of every plot, you can read the integration step, so you can easily compare the 
effect of a change
'''


def main(argv: list) -> int:
    # praesenz(1, 0, 10, deviation, exact)
    # task_a()
    task_b()
    return 0


if __name__ == "__main__":
    main(sys.argv)
