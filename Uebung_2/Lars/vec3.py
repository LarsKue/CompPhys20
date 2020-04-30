
import math
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
        return self.x ** 2 + self.y ** 2 + self.z ** 2

    def unit(self):
        return Vec3(self.x, self.y, self.z) / math.sqrt(self.abs_sq())

    def __repr__(self):
        return "Vec3 { x: " + repr(self.x) + ", y: " + repr(self.y) + ", z: " + repr(self.z) + " }"