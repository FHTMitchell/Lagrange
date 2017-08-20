# main.py

from . import driver as _driver
from . import problem as _problem
from . import solver as _solver

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True

def lagrange(x0=-1e-2, v0=1e-3, dx=1e-5):
    """
    4 months of work... for this?!?!
    
    :param x0: 
    :param v0: 
    :param dx: 
    :return: 
    """

    problem = _problem.Lagrange(1, legendre_order=3)
    driver = _driver.Driver(problem, h0=0.001, K=2, L=1, tf=0.5)
    solver = _solver.Newton(driver.displacement)
    ans = solver.solve([x0, 0, 0, 0, v0, 0], dx=dx)

    t, y = driver.run(ans[-1])
    problem.plot(t, y)
    return ans


if __name__ == '__main__':
    ans = lagrange()
    print(ans[-1])