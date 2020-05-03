
import math
from scipy import constants as consts
from typing import Union, Iterable
from vec3 import Vec3


class ParticleMeta(type):
    """
    Meta class to capture one single value of physical constants
    which are used across all instances of the Particle class
    """
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.__G = 1

    @property
    def G(cls):
        return cls.__G

    @G.setter
    def G(cls, value):
        cls.__G = value


class Particle(metaclass=ParticleMeta):
    def __init__(self, position: Vec3 = Vec3(), velocity: Vec3 = Vec3(), mass: Union[float, int] = 1.0):
        self.position = position
        self.velocity = velocity
        self.mass = float(mass)

    def attraction_with(self, other: "Particle") -> float:
        # F = (G mM / r^2) e_r
        direction = other.position - self.position
        return Particle.G * self.mass * other.mass / direction.abs_sq() * direction.unit()

    def runge_lenz_laplace(self, other: "Particle") -> Vec3:
        # TODO: Check with group members
        r = self.position - other.position
        p = self.gamma() * self.mass * self.velocity
        L = r.cross(p)
        return self.velocity.cross(L) / (Particle.G * self.mass * other.mass) - r.unit()

    def gamma(self):
        # lorentz factor
        return 1 / math.sqrt(1 - self.velocity.abs_sq() / consts.c ** 2)

    def step_explicit_euler(self, h: Union[float, int], particles: Iterable = None) -> "Particle":
        """
        x_n+1 = x_n + h * v_n
        v_n+1 = v_n + h * a_n
        :param h: The time step
        :param particles: All particles in the space this particle resides in (possibly including itself)
        """
        if particles is None:
            # no other interacting particles ==> a = 0
            # ==> only update the particle's position based on its velocity
            return Particle(self.position + h * self.velocity, self.velocity, self.mass)

        # accumulate total interaction force with all particles
        total_force = Vec3()
        for p in particles:
            if p is self:
                # skip "self-interaction"
                continue
            total_force += self.attraction_with(p)

        # calculate the explicit euler trajectory of the particle based on the total force
        return Particle(self.position + h * self.velocity, self.velocity + (h / self.mass) * total_force, self.mass)

    def step_leapfrog(self, h: Union[float, int], particles: Iterable = None) -> "Particle":
        # TODO: Homework
        raise NotImplemented

    def step_implicit_euler(self, h: Union[float, int], particles: Iterable = None) -> "Particle":
        """
        x_n+1 = x_n + h * v_n+1
        v_n+1 = v_n + h * a_n+1
        :param h: The time step
        :param particles: All particles in the space this particle resides in (possibly including itself)
        """
        raise NotImplemented

    def step_midpoint(self, h: Union[float, int], particles: Iterable = None) -> "Particle":
        """
        x_n+1 = x_n + h * v_n+1/2
        v_n+1 = v_n + h * a_n+1/2
        :param h: The time step
        :param particles: All particles in the space this particle resides in (possibly including itself)
        """
        raise NotImplemented

    def step_rk4(self, h: Union[float, int], particles: Iterable = None) -> "Particle":
        """
        x_n+1 = x_n + h / 6 * (k1 + 2k2 + 2k3 + k4)
        v_n+1 = v_n + h / 6 * (l1 + 2l2 + 2l3 + l4)
        :param h: The time step
        :param particles: All particles in the space this particle resides in (possibly including itself)
        """
        raise NotImplemented
