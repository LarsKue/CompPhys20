import numpy as np
import matplotlib.pyplot as plt


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


# Be advised that the integration can take a while for large values of n (e.g >=10^5).
def deviation(y, _, r):
    return -r * y


def exact(x,t):
    return np.exp(-t*x)


y_0 = np.array([1])
x_0 = 0
h = 0.001
n = 1000
r = 1
x = np.linspace(x_0, n * h + x_0, n)


def main(y0, x0, f, h, n, *args, **kwargs):
    hs=[0.001,0.01,0.1,1]
    ns=[10000,1000,100,10]
    plt.figure(figsize=(8, 8))
    err=np.zeros(len(hs))
    for i in range(len(hs)):
        h=hs[i]
        n=ns[i]
        yn, xn = rk4(y0, x0, f, h, n, *args, **kwargs)
        err[i] = np.mean(np.abs(yn - exact(xn, r)) / exact(xn, r))
        plt.plot(xn, yn, label='stepsize={}, mean error={}'.format(h,err[i]))
    plt.plot(x, exact(x,r), label="analytical")
    plt.legend()
    plt.title('rk4 with different step sizes')
    plt.show()


if __name__ == "__main__":
    main(y_0, x_0, deviation, h, n, r)
