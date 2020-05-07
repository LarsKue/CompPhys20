
from copy import copy
from typing import Iterable, Union
from particle import Particle


class NParticleSimulation:
    def __init__(self, particles: Iterable[Particle]):
        self.particles = list(particles)

    def step_explicit_euler(self, h: Union[int, float]):
        """
        For every Particle:
        x_n+1 = x_n + h * v_n
        v_n+1 = v_n + h * a_n

        :param h: The time step
        """
        self.particles = [p.step_explicit_euler(h, self.particles) for p in self.particles]

    def step_implicit_euler(self, h: Union[int, float], sq_tolerance: Union[int, float] = 1e-5):
        """
        For every Particle:
        x_n+1 = x_n + h * v_n+1
        v_n+1 = v_n + h * a_n+1
        Since a_n+1 is a function of x_n+1 we need to iterate on
        the position until a certain tolerance level is reached

        :param h: The time step
        :param sq_tolerance: Exit when position of all particles changes less than this between steps
                Be careful about setting this too low, the program might not halt if you do
        """
        # copy particles from last step
        last_particles = [copy(p) for p in self.particles]

        while True:
            # create the current particles by updating the last particles
            current_particles = [p.step_implicit_euler(h, op, last_particles) for p, op in
                                 zip(last_particles, self.particles)]

            # check if we're done
            done = True
            for lp, cp in zip(last_particles, current_particles):
                if (lp.position - cp.position).abs_sq() > sq_tolerance:
                    done = False
                    break

            # overwrite last particles
            last_particles = current_particles

            if done:
                break

        # overwrite the original particles
        self.particles = last_particles

    def step_kdk_leapfrog(self, h: Union[int, float]):
        """
        We apply the kick-drift-kick method
        For every Particle:
        x_n+1 = x_n + h * v_n+1/2
        v_n+1 = v_n+1/2 + h / 2 * a_n+1
        v_n+1/2 = v_n + h / 2 * a_n

        :param h: The time step
        """
        # half step for velocity and full step for position
        self.particles = [p.step_velocity_kdk_leapfrog(h, self.particles).step_position_kdk_leapfrog(h, self.particles) for p in self.particles]
        # second half step for velocity (requires updated positions)
        self.particles = [p.step_velocity_kdk_leapfrog(h, self.particles) for p in self.particles]

    def step_midpoint(self, h: Union[float, int]):
        """
        For every Particle:
        x_n+1 = x_n + h * v_n+1/2
        v_n+1 = v_n + h * a_n+1/2
        :param h: The time step
        """
        raise NotImplemented

    def step_rk4(self, h: Union[float, int]):
        """
        For every Particle:
        x_n+1 = x_n + h / 6 * (k1 + 2k2 + 2k3 + k4)
        v_n+1 = v_n + h / 6 * (l1 + 2l2 + 2l3 + l4)
        :param h: The time step
        """
        raise NotImplemented

    def __len__(self):
        return len(self.particles)