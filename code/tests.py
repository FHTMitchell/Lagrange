# tests.py

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from problem import SHM, RTBP, Lagrange
from driver import Driver
from solver import Newton
from pprint import pprint

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True


def test_driver(zeta=0):
    shm = SHM(1, {'zeta': zeta})
    driver = Driver(shm, 0.1, 3, 1)
    t, y = driver.run([1, 0], tf=10)
    shm.plot(t, y, data=dict(h0=0.01, L=3, K=1, mode='obr'))
    return

def test_rtbp(xx=None, vy=None):
    rtbp = RTBP(1)
    moon_sma = 384_400e3
    moon_period = 27 * 24 * 60**2.
    if vy is None:
        vy = 2.5e3
    v = [0, vy * moon_period/moon_sma, 0]

    leo = 7e6
    L1 = (1 - np.cbrt(rtbp.mu / 3)) * moon_sma
    if xx is None:
        xx = leo
    x = [xx/moon_sma, 0, 0]
    print(f"x = {x} \nv = {v}")
    driver = Driver(rtbp, h0=0.0001, K=2, L=1, t0=0, tf=3)
    t, y = driver.run(x + v)
    fig, ax = plt.subplots()
    ax.scatter([0, 1], [0, 0], marker='o', color='r')
    ax.plot(y[:,0], y[:,1], color='b')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.show()
    return

def test_lagrange(mu=None, tf=4):
    mapping = {'mu': mu} if mu is not None else {}
    lagrange = Lagrange(1, mapping, lagrange_point=1, legendre_order=2)
    driver = Driver(lagrange, h0=0.01, K=2, L=1, t0=0, tf=tf)
    t, y = driver.run([0, -1e-3, 0, 1e-3, 0, 0])
    assert all(v not in y for v in (np.nan, np.inf, -np.inf))

    lagrange.plot(t, y)
    return


def test_solver():
    """
    url = http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html
    """
    x0 = np.ones(3)

    def f(x):
        x1, x2, x3 = x

        y1 = 3*x1 - np.cos(x2*x3) - 3/2
        y2 = 4*x1**2 - 625*x2**2 + 2*x3 - 1
        y3 = 20*x3 + np.exp(-x1*x2) + 9

        return np.array([y1, y2, y3])

    solver = Newton(f, max_runs=10)

    ans = np.array(solver.solve(x0, 0.001))
    print(ans)


if __name__ == '__main__':
    test_lagrange(tf=10)
    pass