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


def solvers():
    return {c.__name__.lower(): c for c in Solver.subclasses()}


class Solver(Base):

    def __init__(self, func: Callable, *,
                 atol: float = 1e-5, max_runs: int = 100):
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
    def solve(self, vars: np.ndarray, consts: Tuple = (),
              dx0: Union[float, np.ndarray] = None, *,
              func: Callable = None, all_out: bool = False) -> np.ndarray:

        # TODO: Newton is broken since small changed in period do nothing

        if func is None:
            func = self.func

        vars = np.array(vars, ndmin=1, dtype=float, copy=False)
        assert len(vars.shape) == 1

        if dx0 is None:
            dx0 = vars/1e3
        elif isinstance(dx0, (int, float)):
            dx0 = np.ones(vars.shape)*dx0
        dx = np.array(dx0)
        assert dx.shape == vars.shape

        N = len(vars)
        ins = [vars]

        timer = Timer(5)
        for run in range(self.max_runs):
            out1 = func(ins[-1], *consts)
            if run == 0:
                M = len(out1)
                zero = np.zeros(M)
                J = np.zeros((M, N))

            if np.isclose(np.linalg.norm(out1), 0, atol=self.atol):
                if VERBOSE:
                    print(f'Solution reached. Time taken: {timer()}.\n')
                break

            for index in range(N):
                dy = np.zeros(N)
                dy[index] = dx[index]
                out2 = func(ins[-1] + dy, *consts)  # lots of vars
                dout = out2 - out1  # for debugging
                diff = dout/dx[index]  #
                if np.all(diff == 0):
                    warnings.warn(f'Got a no-diff in Newton:\n{out2}')
                    return ins if all_out else ins[-1]
                    ## diff = np.random.random(diff.shape) * dx
                J[:, index] = diff


            if VERBOSE:
                print(f"Completed run {run}. Time taken: {timer()}.")
                print(f"Error was {np.linalg.norm(out1):.5g}")

            # x1 = x0 - J^{-1} @ y0

            if J.shape[0] == J.shape[1]:  # square
                Jinv = np.linalg.inv(J)
            else:
                Jinv = np.linalg.inv(J.T@J)@J.T

            ins_next = ins[-1] - Jinv@out1
            ins.append(ins_next)
            dx = ins[-1] - ins[-2]

            if VERBOSE:
                print('*'*60, ins[-1], '*'*60, sep='\n')
                print('_'*60)

            if np.any(np.isnan(ins[-1])) or np.any(np.isinf(ins[-1])):
                warnings.warn(f'Encountered invalid value: {ins[-1]!s}')
                break
        else:
            if VERBOSE:
                print(
                    f"max_runs {self.max_runs} reached. Time taken: {timer()}")

        return ins if all_out else ins[-1]


class Root(Solver):
    """
    Scipy Solver root - return only final value
    """

    def solve(self, vars, consts=(), *, func: Callable = None,
              all_out: bool = False, **kwargs):

        if func is None:
            func = self.func

        vars = np.asarray(vars, dtype=float)

        if not isinstance(consts, (tuple)):
            raise ValueError(
                f'consts must be a tuple, not {type(consts).__name__}')

        if 'tol' not in kwargs:
            kwargs['tol'] = self.atol

        if 'options' not in kwargs:
            kwargs['options'] = {}
        if 'maxiter' not in kwargs['options']:
            kwargs['options']['maxiter'] = self.max_runs

        out = optimize.root(func, vars, args=consts, **kwargs)

        return out if all_out else out.x


class Fsolve(Solver):
    """
    Scipy Solver  fsolve - return only final value
    """

    def solve(self, vars, consts=(), *, func: Callable = None,
              all_out: bool = False, **kwargs):

        if func is None:
            func = self.func

        vars = np.asarray(vars, dtype=float)

        if not isinstance(consts, (tuple)):
            raise ValueError(
                f'consts must be a tuple, not {type(consts).__name__}')

        if 'xtol' not in kwargs:
            kwargs['xtol'] = self.atol

        if 'maxfev' not in kwargs:
            kwargs['maxfev'] = self.max_runs*(len(vars) + 1)

        kwargs['full_output'] = True

        out = optimize.fsolve(func, vars, args=consts, **kwargs)

        return out if all_out else out[0]


class Broyden(Solver):
    """
    Scipy Solver  broyden1 - return only final value
    """

    def solve(self, vars, consts=(), *, func: Callable = None,
              all_out: bool = VERBOSE, **kwargs):

        if func is None:
            func = self.func
        func1 = lambda x: func(x, *consts)

        vars = np.asarray(vars, dtype=float)

        if not isinstance(consts, (tuple)):
            raise ValueError(
                f'consts must be a tuple, not {type(consts).__name__}')

        if 'f_tol' not in kwargs:
            kwargs['f_tol'] = self.atol

        if 'maxiter' not in kwargs:
            kwargs['maxiter'] = self.max_runs*(len(vars) + 1)

        kwargs['verbose'] = all_out

        out = optimize.broyden1(func1, vars, **kwargs)

        return out
