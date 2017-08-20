# driver.py


import collections
import numpy as np

from . import method as _method
from . import stepper as _stepper
from . import problem as _problem
from .timers import Timer

from typing import *

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True

def static_stepsize_control_factory(h: float) -> Callable:
    def static_stepsize_control(t, y, yd):
        return h
    return static_stepsize_control


valid_methods = {'obr', 'adams'}
valid_problems = {'rtbp', 'shm'}

class Driver:
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
        assert isinstance(method_name, str), repr(method_name)
        assert isinstance(corrector_steps, int) and corrector_steps >= 0, repr(corrector_steps)




        self.problem = problem
        self.h0 = h0
        self.K = K
        self.L = L
        self.t0 = t0
        self.tf = tf
        self.method_name = method_name.lower().strip()
        self.corrector_steps = corrector_steps
        self.stepsize_control = stepsize_control if stepsize_control is not None \
            else static_stepsize_control_factory(h0)

        self.explicit_free_params = explicit_free_params or {}
        self.implicit_free_params = implicit_free_params or {}

        self.Method = _method.methods()[self.method_name]
        self.explicit = self.Method(K, L, 'explicit', self.explicit_free_params)
        if corrector_steps > 0:
            self.implicit = self.Method(K, L, 'implicit', self.implicit_free_params)
        else:
            self.implicit = None

        self.stepper = _stepper.Stepper(self.stepsize_control, self.explicit, self.implicit)


    def run(self, y0: np.ndarray, *, t0=None, tf=None) -> Tuple[np.ndarray, np.ndarray]:

        if t0 is None:
            t0 = self.t0
        if tf is None:
            tf = self.tf
        assert tf is not None, (tf, self.tf)

        y0 = np.array(y0)
        # make sure y0 is an array
        if y0.shape == ():
            # This allows floats to be entered for 1 D
            y0 = y0[np.newaxis]
        assert len(y0.shape) == 1

        y = [y0]
        yd = collections.deque([self.problem.vector_derivs(t0, y0)], maxlen=self.K)

        if __verbose__:
            print('Running driver...')

        t = t0
        k = 1
        counter = 0
        ts = [t0]
        if __verbose__:
            total_iters = int((tf - t0)/self.h0)
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
            if __verbose__ and timer.check():
                print(f'Time elapsed: {timer()}, Iterations: {counter:,} '
                      f'(~{100.*counter/total_iters:.1f}%).')

        if __verbose__:
            print('Finished iterating.')
            print(f'y has length {len(y):,} and final value')
            print(f' yf = [{", ".join(format(x, ".3g") for x in y[-1])}].')
            print(f'Total time elapsed: {timer()}.')
            print('_'*30, end='\n\n')

        return np.array(ts), np.array(y)


    def displacement(self, y0: np.array, **kwargs):

        t, y = self.run(y0, **kwargs)
        return y[-1] - y[0]