# tests.py

import collections
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

import lagrange

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True

init0 = [-0.1, 0, 0, 0, 0.7698, 0]
init = [-0.05, 0, 0, 0, 0.35, 0]

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
        driver = lagrange.driver.SciPyODE(problem, **kwargs)
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
    solver = lagrange.solver.Newton(driver.displacement, atol=1e-2)

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

    ################# SOLVERS ##################################################
    solver = lagrange.solver.Newton(driver.cost_function, atol=0.001)
    ################# SOLVERS ##################################################

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

    x0 = init[0]
    vy0 = init[4]

    kwargs = {}
    # if solver._name.lower() == 'newton':
    #     kwargs['dx0'] = None

    print(f'Using the {solver._name} solver with tolerance {solver.atol:.1e}')
    vy, period = solver.solve(vars=(vy0, tf), consts=(x0,), **kwargs)
    tf0, y0 = problem.cost_init((vy, period), x0)
    t, y = driver.run(y0, tf=tf0)

    if False:
        stack = np.column_stack((t, y))
        print(stack.shape)
        np.savetxt("../matlab/y.csv", stack, delimiter=", ")
        print('saved to file')

    if VERBOSE:
        log('',
            '-'*70,
            'Start Initial values:',
            f'\ttau = {tf:.5g}',
            f'\ty0 = \n\t{y0}',
            'End Initial Values:',
            f'\ttau = {t[-1]-t[0]}',
            f'\ty0 = \n\t{y[0]}',
            'Final Values:',
            f'\tyf = \n\t{y[-1]}',
            sep='\n')

    if plot:
        lo = legendre_order
        title = fr'$\tau={period:.5g},\ h={h0:.1g},\ L_n = L_{lo}$' \
                '\n' \
                fr'$x_0={y0[0]:.4f},\ \dot{{y}}_0={y0[4]:.4f}$'
        problem.plot(t, y, show=False, title='y vs x: ' + title)
        # problem.plot(t, y,  mode='xy', title='xy vs t: ' + title)
        plt.show()

    return y


def test_solver(run=1):
    """
    url = http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html
    """

    def f1(x):

        x1, x2, x3 = x

        y1 = 3*x1 - np.cos(x2*x3) - 3/2
        y2 = 4*x1**2 - 625*x2**2 + 2*x3 - 1
        y3 = 20*x3 + np.exp(-x1*x2) + 9

        return np.asarray([y1, y2, y3])

    def f2(x):

        x1, x2, x3 = x

        y1 = x1**2 - 2*x1 + x2**2 - x3 + 1
        y2 = x1*x2**2 - x1 - 3*x2 + x2*x3 + 2
        y3 = x1*x3**2 - 3*x3 + x2*x3**2 + x1*x2

        return np.asarray([y1, y2, y3])

    if run == 1:
        x0 = np.ones(3)
        f = f1
    else:
        x0 = np.array([1, 2, 3])
        f = f2

    solver = lagrange.solver.Newton(f, max_runs=10, atol=1e-6)

    ans = np.asarray(solver.solve(x0, dx0=1e-10, all_out=True))
    log(ans)


def test_generate_sun_earth():
    mu = lagrange.problem.Lagrange.make_mu('sun', 'earth')
    lag = lagrange.problem.Lagrange(mapping={'mu': mu})
    log(lag.lagrange_points())


################################################################################
######################### EFFICIENCY TESTS #####################################
################################################################################

def test_find_num_calls(h0=0.05, plot=False, verbose=2, lk=None):
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def log2(*args, **kwargs):
        if verbose > 1:
            print(*args, **kwargs)

    if not lk:
        lk = {1: (1, 2, 3, 4), 2: (1, 2, 3), 3: (1, 2), 4: (1,)}

    Res = collections.namedtuple('Res', 'err, calls, time')
    y0 = np.array([1, 0])
    t0 = 0
    tf = 1000*2*np.pi

    res = {}

    for l in lk.keys():
        shm = lagrange.problem.SHM(L=l, mapping=dict(zeta=0))
        sympy_xf = shm.analytical_factory(y0)[0]
        log2(f'xlast = {sympy_xf(tf):.3f}')
        for k in lk[l]:
            title = f'h0 = {h0}, L = {l}, K = {k}'
            log2(f'RUNNING: {title}...')
            shm.calls = 0
            driver = lagrange.driver.Driver(shm, h0, k, l, t0, tf, 'obr')
            t, y = driver.run(y0)
            calls = shm.calls
            time = driver.time
            err, t_last, y_last = _err(t, y, h0, sympy_xf)
            res[(l, k)] = Res(err, calls, time)
            log2('*'*30, f'{title}:', res[(l, k)], '*'*30, sep='\n')
            if plot:
                shm.plot(t_last, y_last, y0=y0, title=f'${title}$')

    log('\n', '-'*80, '\n', '-'*80)
    log(f'h0 = {h0}')
    _print_res(res, verbose)

    return res


def _err(t, y, h0, sympy_xf):
    _last_period = slice(int(-2*np.pi/h0), None)
    y_last = y[_last_period]
    t_last = t[_last_period]
    anal_x = [sympy_xf(ti) for ti in t_last]
    diff = np.abs(y_last[:, 0] - anal_x)
    err = np.log10(np.mean(diff))
    return err, t_last, y_last


def _print_res(r, verbose=True):
    if not verbose:
        return
    for (l, k), (err, calls, time) in r.items():
        print(f'l = {l}, k = {k}:  err = 10^{err:.2f}, calls = {calls:,}'
              f' time={time:.2e} s')


def test_var_h0(max=0.5, min=0.05, steps=5, *, save=False):
    timer = lagrange.timers.Stopwatch()
    file = '../results/results.py'
    resdict = {}
    h0s = np.linspace(min, max, steps)
    for h0 in h0s:
        print('/'*80, f'h0 = {h0}', '/'*80, '\n', sep='\n')
        resdict[h0] = test_find_num_calls(h0, verbose=2)

    print()
    for i in range(4):
        print('/'*80)
    for h0, res in resdict.items():
        print('*'*80)
        print(f'h0 = {h0}')
        _print_res(res)
    print(f'\n Total time elapsed: {timer()}')
    if save:
        with open(file, 'w') as f:
            f.write("import collections\n"
                    "from numpy import nan, inf\n"
                    "Res = collections.namedtuple('Res', 'err, calls, time')\n"
                    f"resdict = {resdict}")
    return resdict


def test_compare_ints(save=False):
    L = 3
    K = 2
    t0 = 0
    tf = 2000*np.pi
    mapping = {'zeta': 0}
    y0 = np.array([1, 0])

    shm_obr = lagrange.problem.SHM(L, mapping)
    shm_other = lagrange.problem.SHM(1, mapping)
    sympy_xf = shm_obr.analytical_factory(y0)[0]
    Res = collections.namedtuple('Res', 'err, calls')

    names = ('obr', 'rk', 'adams')
    resdict = {name: {} for name in names}

    timer = lagrange.timers.Stopwatch()
    for h in np.linspace(0.05, 0.5, 10):
        if VERBOSE:
            print('', '/'*80, f'h = {h}', '/'*80, sep='\n')

        driver_obr = lagrange.driver.Driver(shm_obr, h, K, L, t0, tf, 'obr')
        driver_ode45 = lagrange.driver.SciPyODE(shm_other, h, K, t0, tf,
                                                'ode45')
        driver_adams = lagrange.driver.SciPyODE(shm_other, h, K, t0, tf,
                                                method_name='adams')

        drivers = (driver_obr, driver_ode45, driver_adams)[::-1]
        names = names[::-1]

        for name, driver in zip(names, drivers):
            if name == 'adams':
                continue
            if VERBOSE:
                print(f'Running {name} @ h = {h}')
            shm_other.calls = 0
            shm_obr.calls = 0
            t, y = driver.run(y0)
            calls = max((shm_other.calls, shm_obr.calls))
            err, tl, yl = _err(t, y, h, sympy_xf)
            resdict[name][h] = Res(err, calls)

            if VERBOSE:
                print(f'Time elapsed: {timer()}')

    if save:
        with open('../results/ints.py', 'w') as f:
            f.write("import collections\n"
                    "from numpy import nan, inf\n"
                    "Res = collections.namedtuple('Res', 'err, calls')\n"
                    f"resdict = {resdict}")

    if VERBOSE:
        pprint(resdict)
        print()
        print(f'Time elapsed: {timer()}')

    return resdict




if __name__ == '__main__':
    # test_rtbp(plot=True, drivername='scipy')
    # test_lagrange(plot=True, legendre_order=4, drivername='driver', method='obr')
    # test_driver(drivername='scipy')
    # test_solver(2)
    # test_find_num_calls(0.5, plot=1, verbose=2)
    # test_var_h0(save=0, steps=10)
    test_compare_ints(True)
