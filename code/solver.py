# solver.py

import numpy as np
from timers import Timer
from typing import *

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True

class Solver:
    # ABC for later?
    pass

class Newton(Solver):

    def __init__(self, func: Callable, *, rtol: float = 1e-5, atol: float = 1e-8,
                 max_runs: int = 100):
        self.func = func
        self.rtol = rtol
        self.atol = atol
        self.max_runs = max_runs

    def solve(self, y0: np.ndarray, dx: float) -> np.ndarray:
        y0 = np.array(y0)
        assert len(y0.shape) == 1

        size = len(y0)
        zero = np.zeros(size)
        J = np.empty((size, size))

        y = [y0]

        timer = Timer(5)
        for run in range(self.max_runs):
            y1 = self.func(y[-1])

            if np.isclose(y1, zero, rtol=self.rtol, atol=self.atol).all():
                if __verbose__:
                    print(f'Solution reached. Time taken: {timer()}.\n')
                return y

            for index in range(size):
                dy = np.zeros(size)
                dy[index] = dx
                J[:, index] = (self.func(y[-1] + dy) - y1)/dx

            if __verbose__:
                print(f"Completed run {run}. Time taken: {timer()}.")
                print('_'*70)

            y.append(y[-1] - np.linalg.inv(J)@y1)

        if __verbose__:
            print(f"max_runs {self.max_runs} reached. Time taken: {timer()}")
        return y