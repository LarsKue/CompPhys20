
from copy import copy
from typing import Iterable, Union
from particle import Particle
from vec3 import Vec3


class NParticleSystem:
    def __init__(self, particles: Iterable[Particle]):
        self.particles = list(particles)

    def center_of_mass(self) -> Vec3:
        result = Vec3()
        total_mass = 0.0
        for p in self.particles:
            result += p.mass * p.position
            total_mass += p.mass

        return result / total_mass

    def shift_origin(self, new_origin: Vec3):
        for p in self.particles:
            p.position -= new_origin

    def evaluate_interaction_forces(self):
        for p1 in self.particles:
            for p2 in self.particles:
                if p1 is p2:
                    # skip self-interaction
                    continue
                f = p1.attraction_with(p2)
                p1.force += f
                p2.force -= f

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
        x_n+1 = x_n + h * (k1 + 2k2 + 2k3 + k4) / 6
        v_n+1 = v_n + h * (l1 + 2l2 + 2l3 + l4) / 6
        for definitions of l and k see below
        :param h:
        :return:
        """

        iter_particles = [copy(p) for p in self.particles]

        k1s = [p.velocity for p in self.particles]
        l1s = [p.total_interaction_force(iter_particles) for p in iter_particles]

        for p, k1, l1 in zip(iter_particles, k1s, l1s):
            p.position += h * k1 / 2
            p.velocity += h * l1 / 2

        k2s = [p.velocity for p in iter_particles]
        l2s = [p.total_interaction_force(iter_particles) for p in iter_particles]

        for p, op, k2, l2 in zip(iter_particles, self.particles, k2s, l2s):
            p.position = op.position + h * k2 / 2
            p.velocity = op.velocity + h * l2 / 2

        k3s = [p.velocity for p in iter_particles]
        l3s = [p.total_interaction_force(iter_particles) for p in iter_particles]

        for p, op, k3, l3 in zip(iter_particles, self.particles, k3s, l3s):
            p.position = op.position + h * k3
            p.velocity = op.velocity + h * l3

        k4s = [p.velocity for p in iter_particles]
        l4s = [p.total_interaction_force(iter_particles) for p in iter_particles]

        # print("k1 =", k1s)
        # print("k2 =", k2s)
        # print("k3 =", k3s)
        # print("k4 =", k4s)
        # print("l1 =", l1s)
        # print("l2 =", l2s)
        # print("l3 =", l3s)
        # print("l4 =", l4s)

        for p, k1, k2, k3, k4, l1, l2, l3, l4 in zip(self.particles, k1s, k2s, k3s, k4s, l1s, l2s, l3s, l4s):
            p.position += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            p.velocity += h * (l1 + 2 * l1 + 2 * l3 + l4) / 6

    def __len__(self):
        return len(self.particles)