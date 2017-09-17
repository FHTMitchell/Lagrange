# tests.py

import sys

import matplotlib.pyplot as plt
import numpy as np

import lagrange

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True

init = [-0.1, 0, 0, 0, 0.7698, 0]

test_halo = np.multiply([-1505994.497477462 + (lagrange.bodies.earth.sma/1e3),
                         207178.9433070129,
                         44653.24381740281,
                         0.04567564828317306,
                         0.06346749689164777,
                         0.05614178778548588], 1e3)  # m,m,m,m/s,m/s,m/s


def log(*strs, sep=' ', end='\n', file=sys.stdout, flush=False):
    if VERBOSE:
        print(*strs, sep=sep, end=end, flush=flush, file=file)


def get_driver(drivername, problem, **kwargs):
    if drivername == 'driver':
        driver = lagrange.driver.Driver(problem, **kwargs)
    elif drivername == 'scipy':
        driver = lagrange.driver.SciPy(problem, **kwargs)
    else:
        raise ValueError(f"drivername must be 'driver' or 'scipy', not "
                         f"{drivername!r}")
    return driver


def test_driver(zeta=0, drivername='driver', method_name='obr'):
    h0 = 0.01
    L = 1 if (drivername == 'scipy' or method_name == 'adams') else 1
    K = 8

    shm = lagrange.problem.SHM(L, {'zeta': zeta})
    corr_steps = 0 if method_name == 'adams' else 1
    driver = get_driver(drivername, shm, h0=h0, L=L, K=K, corr_steps=corr_steps,
                        method_name=method_name)

    t, y = driver.run([1, 0], tf=10)
    shm.plot(t, y, data=dict(h0=h0, L=L, K=K, mode=method_name))
    return


def test_rtbp(plot=True, steps=1e4, drivername='driver', method='obr'):
    mapping = {'mu': lagrange.problem.Lagrange.make_mu('sun', 'earth')}
    steps = int(steps)

    K = 4
    L = 1

    problem = lagrange.problem.RTBP(L, mapping)

    tf = 2*np.pi
    h0 = tf/steps
    kwargs = {'method_name': method,
              'corrector_steps': 1 if method == 'obr' else 0}

    driver = get_driver(drivername, problem, h0=h0, K=K, L=L, tf=tf, **kwargs)
    solver = lagrange.solver.Newton(driver.displacement, atol=1e-20, rtol=1e-8)

    log(*(repr(x) for x in (problem, driver, solver)), sep='\n\n')
    log(f"nmu = {problem.mu:5g}")

    halo = problem.convert_from_bodies(test_halo, 'sun', 'earth')
    l1 = problem.lagrange_points()[1]

    log(f"L1 = {l1[0]:.5f},\n halo = \n{halo},"
        f"\ntf = {tf:.5g} s.")

    t, y = driver.run(halo, tf=tf)
    if plot:
        problem.default_points['2d']['L1'] = l1[0:2]
        problem.plot(t, y, title=fr'$tf={tf:.2g}, h={h0:.2g}$')
    pass
    return


################################################################################
############################# LAGRANGE #########################################
################################################################################
def test_lagrange(plot=True, legendre_order=3, drivername='driver',
                  method='odeint', steps=464):
    """
    LAGRANGE
    """
    steps = int(steps)
    L = 1
    K = 4
    mapping = {'mu': lagrange.problem.Lagrange.make_mu('sun', 'earth')}

    problem = lagrange.problem.Lagrange(L, mapping,
                                        legendre_order=legendre_order)
    tf = problem.convert_from_barycentric(2*np.pi*480, 't')  # todo why O(10^2)?
    h0 = tf/steps
    driver = get_driver(drivername, problem, h0=h0, K=K, L=L, tf=tf,
                        method_name=method)

    solver = lagrange.solver.Root(driver.displacement, atol=1e-12)

    log(*(repr(x) for x in (problem, driver, solver)), sep='\n\n')

    # log('\nODEs = ')
    # log(*problem.make_ODEs(display='show'), sep='\n', end='\n\n')
    # log('Simplifying...')
    # log(*map(sp.simplify, problem.make_ODEs('show')), sep='\n', end='\n\n')
    log(*(f'{k}: {v:.6g}' for k, v in list(problem.c.items())[2:]),
        sep='\n', end='\n\n')
    # log('ODEs (With c values) = ')
    # log(*problem.ODEs[-1], sep='\n', end='\n\n')
    #
    log(f"gamma = {problem.gamma:.5g} \nmu = {problem.mu:5g}")

    log(f"init = \n{init}, \ntf = {tf:.5g}.")

    if solver.__class__.__name__ == 'Root':
        kwargs = {}
    elif solver.__class__.__name__ == 'Newton':
        kwargs = {'dx': 1e-4}
    else:
        raise ValueError(solver)

    ans = init
    # ans = solver.partsolve(0.2, [-0.1, 0, 0, 0, None, 0], **kwargs)
    # ans = solver.solve(init, **kwargs)


    t, y = driver.run(ans)

    if True:
        stack = np.column_stack((t, y))
        print(stack.shape)
        np.savetxt("../matlab/y.csv", stack, delimiter=", ")
        print('saved to file')

    if plot:
        lo = legendre_order
        title = fr'$t_f={tf:.1g},\ h={h0:.1g},\ L_n = L_{lo}$'
        problem.plot(t, y, show=False, title='y vs x: ' + title)
        # problem.plot(t, y,  mode='xy', title='xy vs t: ' + title)
        plt.show()

    return ans


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

        return np.asarray([y1, y2, y3])

    solver = lagrange.solver.Root(f, max_runs=10)

    ans = np.asarray(solver.solve(x0))
    log(ans)


def test_generate_sun_earth():
    mu = lagrange.problem.Lagrange.make_mu('sun', 'earth')
    lag = lagrange.problem.Lagrange(mapping={'mu': mu})
    log(lag.lagrange_points())


if __name__ == '__main__':
    # test_rtbp(plot=True, drivername='scipy')
    test_lagrange(plot=True, legendre_order=4, drivername='driver',
                  method='obr')
    # test_driver(drivername='scipy')
    # test_solver()
