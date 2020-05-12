import math
from copy import copy
from typing import Union


class Vec3:
    def __init__(self, x: Union[float, int] = 0.0, y: Union[float, int] = 0.0, z: Union[float, int] = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: Union["Vec3", int, float]) -> Union["Vec3", float]:
        if isinstance(other, (int, float)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        if isinstance(other, Vec3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        return NotImplemented

    def __truediv__(self, other: Union[int, float]) -> "Vec3":
        return Vec3(self.x / other, self.y / other, self.z / other)

    def __rmul__(self, other: Union[int, float]):
        return self * other

    def abs_sq(self) -> float:
        """
        :return: Absolute Value of self squared
        """
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def unit(self):
        """
        :return: The unit vector of self
        """
        return Vec3(self.x, self.y, self.z) / math.sqrt(self.abs_sq())

    def cross(self, other: "Vec3") -> "Vec3":
        """
        :return: The cross-product (or vector-product) between self and other
        """
        return Vec3(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x)

    def __repr__(self):
        return "Vec3 { x: " + repr(self.x) + ", y: " + repr(self.y) + ", z: " + repr(self.z) + " }"

    def __copy__(self):
        return Vec3(copy(self.x), copy(self.y), copy(self.z))
