# solver.py

import warnings
from typing import *

import numpy as np
from scipy import optimize

from .bases import Base
from .timers import Timer

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True


class Solver(Base):
    # ABC for later?
    def __init__(self, func: Callable, *,
                 atol: float = 1e-8, max_runs: int = 100):
        self.func = func
        self.atol = atol
        self.max_runs = max_runs

    def solve(self, y0: np.ndarray, *, func: Callable = None) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return self._make_repr('func', ('atol', '.1e'), 'max_runs')


class Newton(Solver):
    """
    Custom Solver
    """

    # noinspection PyMethodOverriding
    def solve(self, y0: np.ndarray, dx: float, *, func: Callable = None,
              ret_all_y: bool = False) -> np.ndarray:
        if func is None:
            func = self.func
        y0 = np.array(y0, ndmin=1)
        assert len(y0.shape) == 1

        size = len(y0)
        zero = np.zeros(size)
        J = np.empty((size, size))

        y = [y0]

        timer = Timer(5)
        for run in range(self.max_runs):
            y1 = func(y[-1])

            if np.isclose(y1, zero, atol=self.atol).all():
                if VERBOSE:
                    print(f'Solution reached. Time taken: {timer()}.\n')
                return y

            for index in range(size):
                dy = np.zeros(size)
                dy[index] = dx
                J[:, index] = (func(y[-1] + dy) - y1)/dx

            if VERBOSE:
                print(f"Completed run {run}. Time taken: {timer()}.")

            y.append(y[-1] - np.linalg.inv(J)@y1)

            if VERBOSE:
                print('*'*60, y[-1], '*'*60, sep='\n')
                print('_'*60)

            if np.any(np.isnan(y[-1])) or np.any(np.isinf(y[-1])):
                warnings.warn(f'Encountered invalid value: {y[-1]!s}')
                break

        if VERBOSE:
            print(f"max_runs {self.max_runs} reached. Time taken: {timer()}")

        return y if ret_all_y else y[-1]


class Root(Solver):
    """
    Scipy Solver - return only final value
    """

    def solve(self, args, consts=(), *, func: Callable = None,
              all_out: bool = False, **kwargs):

        if func is None:
            func = self.func

        if not isinstance(consts, (tuple)):
            raise ValueError(
                f'consts must be a tuple, not {type(consts).__name__}')

        if 'tol' not in kwargs:
            kwargs['tol'] = self.atol

        if 'options' not in kwargs:
            kwargs['options'] = {}
        if 'maxiter' not in kwargs['options']:
            kwargs['options']['maxiter'] = self.max_runs

        out = optimize.root(func, args, method='broyden1', **kwargs)

        return out if all_out else out.x
