
import sys
import numpy as np
from typing import Iterable, Union
import math
from copy import deepcopy


def praesenz() -> None:
    epsilon = 1e-16

    M = np.array([
        [epsilon, 0.5],
        [0.5, 0.5]
    ])

    y = (epsilon - 1) / (2 * epsilon + 1)
    x = 0.5 - (epsilon - 1) / (2 * epsilon + 1)

    v = np.array([x, y])

    print(f"x = {x:.3f}")
    print(f"y = {y:.3f}")

    b = M @ v

    print(f"M * v = {b}")


def first_nonzero(l: Iterable[Union[int, float]]):
    # return index of first non-zero item
    for i, item in enumerate(l):
        if not math.isclose(item, 0):
            return i


class LinearEquationSystem:
    def __init__(self, matrix: np.ndarray, solution: np.ndarray):
        self.matrix = matrix.astype(float)
        self.solution = solution.astype(float)

    def rows(self):
        # number of rows in the matrix or the solution
        return len(self.matrix)

    def matrix_columns(self):
        # number of columns in the matrix
        return len(self.matrix[0])

    def row_add(self, i, j, factor=1):
        # add row j to row i
        self.matrix[i] += factor * self.matrix[j]
        self.solution[i] += factor * self.solution[j]

    def row_sub(self, i, j, factor=1):
        # subtract row j from row i
        self.matrix[i] -= factor * self.matrix[j]
        self.solution[i] -= factor * self.solution[j]

    def row_mul(self, i, factor):
        # multiply row i by a factor
        self.matrix[i] *= factor
        self.solution[i] *= factor

    def row_div(self, i, factor):
        # divide row i by a factor
        self.matrix[i] /= factor
        self.solution[i] /= factor

    def __first_nonzero(self, i):
        # get the index of the first nonzero element in row i
        for j, item in enumerate(self.matrix[i]):
            if not math.isclose(item, 0):
                return j

    def __normalize(self, i):
        # normalize a row to its first nonzero component
        # and return the column index of this component
        fnz = self.__first_nonzero(i)
        if fnz is None:
            raise RuntimeError("Row cannot be normalized as it contains only zeros.")
        self.row_div(i, self.matrix[i][fnz])
        return fnz

    def __normalize_and_isolate(self, i):
        # normalize a row to its first nonzero component
        # and zero all other values in that column
        try:
            fnz = self.__normalize(i)
        except RuntimeError:
            return
        for j in range(self.rows()):
            if i == j:
                # do not subtract row from itself
                continue
            factor = self.matrix[j][fnz]
            if not math.isclose(factor, 0):
                self.row_sub(j, i, factor)

    def solve_tridiagonal(self):
        """
        solve the linear equation system if self.matrix is tri-diagonal
        note that if self.matrix is not square, the initial solution provided by this
        is not of the correct dimension, you will have to add zeros to the end of the vector.
        """
        for i in range(self.rows()):
            self.__normalize_and_isolate(i)
        # return the solution for convenience
        return self.solution


def homework():
    M = np.array([
        [2, 3, 0, 0, 0, 0],
        [1, 4, 3, 0, 0, 0],
        [0, 3, 2, 1, 0, 0],
        [0, 0, 3, 2, 1, 0],
        [0, 0, 0, 1, 4, 2],
        [0, 0, 0, 0, 1, 1],
        # [0, 0, 0, 0, 0, 1],
    ])

    # print(f"Determinant: {np.linalg.det(M)}")

    x = np.array([1, 2, 3, 4, 5, 6])

    b = M @ x

    lgs = LinearEquationSystem(deepcopy(M), b)

    y = lgs.solve_tridiagonal()

    print(f"Compare Optional: {x} vs {y} (These will only be equivalent if M is square.)")
    # y = np.append(y, 0)
    # y = np.append(y, 0)
    print(f"Compare Definitive: {M @ y[:lgs.matrix_columns()]} vs {b}")
    print(lgs.matrix)


def main(argv: list) -> int:
    # praesenz()
    homework()
    return 0


if __name__ == "__main__":
    main(sys.argv)