# driver.py


import collections
from typing import *

import numpy as np
from scipy import integrate

from . import method as _method, problem as _problem, stepper as _stepper
from .bases import Base
from .timers import Timer

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True

calls = 0


def drivers():
    return {c.__name__.lower(): c for c in DriverBase.subclasses()}


def static_stepsize_control_factory(h: float) -> Callable:
    def static_stepsize_control(t, y, yd):
        return h

    return static_stepsize_control


valid_methods = {'obr', 'adams'}
valid_problems = {'rtbp', 'shm'}


class DriverBase(Base):
    _calls = 0
    problem: _problem.Problem
    run: Callable
    t0: float

    def run(self):
        pass
        # self._calls += 1
        # if VERBOSE:
        #     print(f'calls: {self._calls}')

    def cost_function(self, vars, *consts):
        period, y0 = self.problem.cost_init(vars, *consts)
        t, y = self.run(y0, tf=period - self.t0)
        out = self.problem.cost_exit(t, y)
        return out

class Driver(DriverBase):
    def __init__(self,
                 problem: _problem.Problem,
                 h0: float,
                 K: int,
                 L: int = 1,
                 t0: float = 0.,
                 tf: float = None,
                 method_name: str = 'obr',
                 corrector_steps: int = 1,
                 stepsize_control: Callable = None,
                 explicit_free_params: Mapping[str, float] = None,
                 implicit_free_params: Mapping[str, float] = None):

        """
        
        :param problem: 
        :param h0: 
        :param K: 
        :param L: 
        :param method_name: 
        :param corrector_steps: 
        :param stepsize_control: 
        :return: 
        """

        assert isinstance(problem, _problem.Problem), repr(problem)
        assert h0 > 0, repr(h0)
        assert isinstance(K, int) and K >= 1, repr(K)
        assert isinstance(L, int) and L >= 1, repr(L)
        assert problem.L == L, (L, problem.L)
        assert isinstance(method_name, str), repr(method_name)
        assert isinstance(corrector_steps, int) and corrector_steps >= 0, repr(
            corrector_steps)

        self.problem = problem
        self.h0 = h0
        self.K = K
        self.L = L
        self.t0 = t0
        self.tf = tf
        self.method_name = method_name.lower().strip()
        self.corrector_steps = corrector_steps
        self.stepsize_control = stepsize_control if \
            stepsize_control is not None \
            else static_stepsize_control_factory(h0)

        self.explicit_free_params = explicit_free_params or {}
        self.implicit_free_params = implicit_free_params or {}

        self.Method = _method.methods()[self.method_name]
        self.explicit = self.Method(K, L, 'explicit', self.explicit_free_params)
        if corrector_steps > 0:
            self.implicit = self.Method(K, L, 'implicit',
                                        self.implicit_free_params)
        else:
            self.implicit = None

        self.stepper = _stepper.Stepper(self.stepsize_control, self.explicit,
                                        self.implicit)
        self.time = 0

    def __repr__(self):
        return self._make_repr('problem', 'h0', 'L', 'K')

    # noinspection PyMethodOverriding
    def run(self, y0: np.ndarray, *, t0=None, tf=None
            ) -> Tuple[np.ndarray, np.ndarray]:

        if t0 is None:
            t0 = self.t0
        if tf is None:
            tf = self.tf
        assert tf is not None, (tf, self.tf)

        y0 = np.array(y0, copy=True, ndmin=1)
        assert len(y0.shape) == 1

        y = [y0]
        yd = collections.deque([self.problem.vector_derivs(t0, y0)],
                               maxlen=self.K)

        if VERBOSE:
            print('Running driver...')

        t = t0
        k = 1
        counter = 0
        ts = [t0]
        if VERBOSE:
            total_iters = int((tf - t0)/self.h0)
        self.time = 0
        timer = Timer(5)

        while t < tf:
            h1, t, y1 = self.stepper.predict(t, y, yd, k)
            for _ in range(self.corrector_steps):
                y1d = self.problem.vector_derivs(t, y1)
                y2 = self.stepper.correct(yd, y1d, k, h1)
                # do anything fancy with y2 and y1
                y1 = y2

            ts.append(t)
            y.append(y1)
            yd.append(self.problem.vector_derivs(t, y[-1]))
            if k < self.K:  # Use maximum number of previous step up to K
                k += 1
            counter += 1
            if VERBOSE and timer.check():
                print(f'Time elapsed: {timer()}, Iterations: {counter:,} '
                      f'(~{100.*counter/total_iters:.1f}%).')

        if VERBOSE:

            print('Finished iterating.')
            print(f'y has length {len(y):,} and final value')
            print(f' yf = [{", ".join(format(x, ".3g") for x in y[-1])}].')
            print(f'Total time elapsed: {timer()}.')
            print('_'*30, end='\n\n')

        self.time = timer.toc

        super().run()
        return np.asarray(ts), np.asarray(y)


class SciPyODE(DriverBase):
    def __init__(self, problem: _problem.Problem, h0: float = 0., K: int = None,
                 t0: float = 0.,
                 tf: float = None, method_name: str = 'odeint', **kwargs):
        self.problem = problem
        self.h0 = h0
        self.t0 = t0
        self.tf = tf
        self.K = K
        self.method = method_name.lower().strip()

        assert problem.L == 1, problem.L

    def f_odeint(self, y, t):  # args must be swapped for odeint
        return self.problem.vector_derivs(t, y)[1]

    def f_ode(self, t, y):
        return self.problem.vector_derivs(t, y)[1]

    def __repr__(self):
        return self._make_repr('problem', 'h0')

    # noinspection PyMethodOverriding
    def run(self, y0: np.ndarray, *, t0=None, tf=None
            ) -> Tuple[np.ndarray, np.ndarray]:

        y0 = np.array(y0)

        if t0 is None:
            t0 = self.t0
        if tf is None:
            tf = self.tf
            assert tf is not None, (tf, self.tf)
        t = np.arange(t0, tf, self.h0)

        if self.method in ('ode45', 'adams'):
            if VERBOSE:
                print(f'\nUsing {self.method}')

            ode = integrate.ode(self.f_ode)

            if self.method == 'ode45':
                ode.set_integrator('dopri5', max_step=self.h0,
                                   first_step=self.h0,
                                   verbosity=VERBOSE, nsteps=self.K)
            elif self.method == 'adams':
                ode.set_integrator('lsoda')
            else:
                raise Exception

            ode.set_initial_value(y0, t0)
            y = np.empty((len(t), len(y0)))

            for index, ti in enumerate(t):
                y[index, :] = ode.integrate(ti)

        elif self.method == 'odeint':
            if VERBOSE:
                print('\nUsing odeint')
                print(f'y0 = \n {y0}\ntf = {tf:.5e}')
            y = integrate.odeint(self.f_odeint, y0, t, rtol=1e-12, atol=1e-12)

        else:
            raise ValueError(f'method must be "ode45", "adams" or "odeint"')

        super().run()
        return t, y
