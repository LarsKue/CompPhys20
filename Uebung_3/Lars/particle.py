import math
from scipy import constants as consts
from typing import Union, Iterable
from vec3 import Vec3
from copy import copy


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
        self.force = Vec3()
        self.mass = float(mass)

    def attraction_with(self, other: "Particle") -> float:
        """
        F = (G mM / r^2) e_r
        :return: The gravitational attraction acting on self between the two particles
        """
        direction = other.position - self.position
        return Particle.G * self.mass * other.mass / direction.abs_sq() * direction.unit()

    def laplace_runge_lenz(self, other: "Particle") -> Vec3:
        """
        calculate and return the LRL vector for the rest system of self
        i.e. the system where other orbits around self
        """
        r = other.position - self.position
        v = other.velocity - self.velocity
        p = other.mass * v
        L = r.cross(p)
        return p.cross(L) - Particle.G * self.mass * other.mass * r.unit()

    def eccentricity(self, other: "Particle"):
        """
        see laplace_runge_lenz
        """
        return self.laplace_runge_lenz(other) / (Particle.G * self.mass * other.mass)

    def total_interaction_force(self, particles: Iterable["Particle"] = None) -> Vec3:
        """
        :param particles: All the particles of the system self resides in, possibly including itself
        :return: The total force the system enacts on self
        """
        # accumulate total interaction force with all particles
        result = Vec3()
        if particles:
            for p in particles:
                if p is self or math.isclose((p.position - self.position).abs_sq(), 0.0):
                    # skip "self-interaction" for the particle itself (same memory location)
                    # or if the particles are right on top of each other
                    # since usually this means it's the same particle
                    continue
                result += self.attraction_with(p)

        return result

    def step_explicit_euler(self, h: Union[float, int], particles: Iterable["Particle"] = None) -> "Particle":
        # this is a full step
        force = self.total_interaction_force(particles)
        velocity = self.velocity + (h / self.mass) * force
        position = self.position + h * velocity

        return Particle(position, velocity, self.mass)

    def step_implicit_euler(self, h: Union[float, int], original_self: "Particle",
                            particles: Iterable["Particle"] = None):
        # this is one iteration step
        force = self.total_interaction_force(particles)
        velocity = original_self.velocity + (h / self.mass) * force
        position = original_self.position + h * velocity
        return Particle(position, velocity, self.mass)

    def step_position_kdk_leapfrog(self, h: Union[float, int], particles: Iterable["Particle"] = None) -> "Particle":
        # this is the positional part of kdk leapfrog
        # we can directly calculate x_n+1
        return Particle(self.position + h * self.velocity, self.velocity, self.mass)

    def step_velocity_kdk_leapfrog(self, h: Union[float, int], particles: Iterable["Particle"] = None):
        # this is the velocity part of kdk leapfrog
        # we can directly calculate v_n+1/2 (or v_n+1)
        return Particle(self.position, self.velocity + (h / 2) * self.total_interaction_force(particles), self.mass)

    def step_rk4_1(self, h: Union[float, int], k1: Vec3) -> "Particle":
        return Particle(self.position + h * k1 / 2, self.velocity )

    def __copy__(self):
        return Particle(copy(self.position), copy(self.velocity), copy(self.mass))

    def __repr__(self):
        return "Particle { position: " + repr(self.position) + ", velocity: " + repr(self.velocity) + " }"
