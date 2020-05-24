
"""
Submission for Uebung 5 for Computational Physics 2020
Group Members:
Daniel Kreuzberger
Lars Kuehmichel
David Weinand
"""

import sys
import numpy as np
from typing import Iterable, Union
import math
from copy import deepcopy


"""
How to use:
This one is very self-explanatory. Run the file with
python main.py
and look at the homework function at the bottom of this file to see how to use the class
LinearEquationSystem
"""


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
        """
        Linear Equation System Mx = b
        :param matrix: M
        :param solution: b
        """
        # we want floating point types to avoid division errors
        self.matrix = matrix.astype(float)
        self.solution = solution.astype(float)

    def num_rows(self):
        # number of rows in the matrix or the solution
        return len(self.matrix)

    def num_matrix_columns(self):
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
        for j in range(self.num_rows()):
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
        is not of the correct dimension, you will have to add zeros to the end of the vector
        this method will overwrite self.matrix and self.solution
        """
        for i in range(self.num_rows()):
            self.__normalize_and_isolate(i)
        # return the solution for convenience
        return self.solution


def homework():
    """ Task 3 Testing """
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

    lgs = LinearEquationSystem(deepcopy(M), deepcopy(b))

    y = lgs.solve_tridiagonal()

    print(f"Compare Optional: {x} vs {y} (These will only be equivalent if M is square.)")
    # y = np.append(y, 0)
    # y = np.append(y, 0)
    print(f"Compare Definitive: {M @ y[:lgs.num_matrix_columns()]} vs {b}")
    # print(lgs.matrix)

    """ Task 4 """
    print("\nTask 4:")

    M = np.array([
        [3, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 3, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 3, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 3, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 3, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 3, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 3, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 3, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 3, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 3],
    ])

    b = np.array([0.2] * 10)

    lgs = LinearEquationSystem(deepcopy(M), deepcopy(b))

    # the solution is symmetrical, as is to be expected
    print(f"Solution: {lgs.solve_tridiagonal()}")

    """ Task 5 """
    print("\nTask 5:")

    y = M @ lgs.solution

    # these are identical
    print(f"Original RHS: {b}")
    print(f"New RHS: {y}")
    # the deviation is (within floating point rounding errors) zero
    print(f"Deviation: {y - b}")


def main(argv: list) -> int:
    # praesenz()
    homework()
    return 0


if __name__ == "__main__":
    main(sys.argv)
