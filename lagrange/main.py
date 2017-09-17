# main.py

import lagrange

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True


def all_classes():
    d = {c.__name__: c for c in lagrange.bases.Base.subclasses()}
    d['Base'] = lagrange.bases.Base
    return d


def run_lagrange(x0=-1e-2, v0=1e-3, dx=1e-5, mu=None):
    """
    4 months of work... for this?!?!
    
    :param x0: 
    :param v0: 
    :param dx: 
    :return: 
    """
    mapping = {} if mu is None else {'mu': mu}

    problem = lagrange.problem.Lagrange(1, mapping, legendre_order=3)
    driver = lagrange.driver.Driver(problem, h0=0.001, K=2, L=1, tf=0.5)
    solver = lagrange.solver.Newton(driver.displacement)

    ans = solver.solve([x0, 0, 0, 0, v0, 0], dx=dx, ret_all_y=False)

    t, y = driver.run(ans)
    problem.plot(t, y)
    return ans


if __name__ == '__main__':
    ans = run_lagrange()
    print(ans[-1])
