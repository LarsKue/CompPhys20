import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable, Iterable


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

def solve_rk4(f: Callable, y0: Union[int, float], t: Iterable, *args, **kwargs):
    """
        Solve a differential equation of the form y' = f(t, y) using the runge-kutta method of order 4
        This implementation is specifically for scalar functions, but you could possibly use it
        for functions that yield objects which behave as or similar to scalars
        :param f: The right hand side of the equation
        :param y0: Starting Parameter, y(0) = y0
        :param t: The range in which to solve the equation
        :param args: Positional arguments for f
        :param kwargs: Keyword arguments for f
        :return: Tuples (t, y(t))
        """
    it = iter(t)
    last_t = next(it)
    yield last_t, y0
    for current_t in it:
        h = current_t - last_t

        # slope at current time: Euler Slope
        k1 = f(current_t, y0, *args, **kwargs)
        # slope at midpoint between current and next time point
        # the y value is linearly interpolated using the Euler-Slope
        k2 = f(current_t + h / 2, y0 + h * k1 / 2, *args, **kwargs)

        # again the slope at the midpoint
        # this time the y value is linearly interpolated with the above midpoint-Slope
        k3 = f(current_t + h / 2, y0 + h * k2 / 2, *args, **kwargs)

        # slope at the next time point
        # y value is linearly interpolated using the 2nd midpoint slope
        k4 = f(current_t + h, y0 + h * k2, *args, **kwargs)

        # update y with weighted interpolation using the above 4 slopes
        y0 += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        last_t = current_t

        yield current_t, y0


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

        plt.plot(xn, yn, label='stepsize={}, mean error={:.2e}'.format(h, err))

    n_max = int((x1 - x0) / hs[-1])
    x = np.linspace(x0, x1, n_max)
    plt.plot(x, f_exact(x, r), label="exact")
    plt.legend(loc="upper right")
    plt.title('rk4 with different step sizes')
    plt.savefig('rk4step.pdf')
    plt.show()


def main(argv: list) -> int:

    praesenz(1, 0, 10, deviation, exact)

    return 0


if __name__ == "__main__":
    main(sys.argv)
