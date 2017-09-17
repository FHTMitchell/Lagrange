# method.py

from collections import deque
from typing import *

import numpy as np
import sympy as sp

from .bases import Base

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True


def methods():
    return {c.__name__.lower(): c for c in Method.subclasses()}


class Method(Base):
    def __init__(self, maxK, L=1, mode='explicit', free_params=None):
        """
        
        :param maxK: 
        :param L: 
        :param mode: 
        :param free_params: 
        """
        self.maxK = maxK
        self.L = L
        self.mode = mode
        self.free_params = free_params if free_params is not None else {}
        assert isinstance(L, int) and (0 < L), L
        assert isinstance(maxK, int) and (0 < maxK), maxK

    def __repr__(self):
        return self._make_repr('maxK', 'L', mode=repr(self.mode))

    def explicit(self, yd: deque, h: float, k: int) -> np.ndarray:
        raise NotImplementedError

    def implicit(self, yd: deque, ynd: np.ndarray, h: float,
                 k: int) -> np.ndarray:
        raise NotImplementedError


class Obr(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alphas = {k: self.coeffs(k) for k in range(1, self.maxK + 1)}

    def explicit(self, yd: deque, h: float, k: int):
        nexty = sum(
            h**i * sum(self.alphas[k][i][j] * yd[j][i] for j in range(0, k))
            for i in range(0, self.L + 1))

        return nexty

    def implicit(self, yd: deque, ynd: Union[list, np.ndarray], k: int,
                 h: float):
        explicit_part = sum(
            h**i * sum(self.alphas[k][i][j] * yd[j][i] for j in range(0, k))
            for i in range(0, self.L + 1))

        implicit_part = sum(h**i * self.alphas[k][i][k] * ynd[i]
                            for i in range(1, self.L + 1))

        # if np.any(np.isnan(explicit_part)):
        #     print('nan encountered')

        thisy = explicit_part + implicit_part

        return thisy

    def coeffs(self, K) -> List[List[float]]:

        L = self.L

        a, b, c = sp.symbols('a, b, c')

        if self.mode == 'explicit':
            if K == 1:
                if L == 1:
                    alpha = [[1, -1], [1]]
                if L == 2:
                    alpha = [[1, -1], [1], [1 / 2]]
                if L == 3:
                    alpha = [[1, -1], [1], [1 / 2], [1 / 6]]
                if L == 4:
                    alpha = [[1, -1], [1], [1 / 2], [1 / 6], [1 / 24]]
            if K == 2:
                if L == 1:
                    alpha = [[a, 1 - a, -1], [-0.5 * (1 - a), 0.5 * (3 + a)]]
                if L == 2:
                    alpha = [[a, 1 - a, -1], [0.5 * (3 + a), -0.5 * (1 - a)],
                             [(7 + a) / 12, -(a - 17) / 12]]
                if L == 3:
                    alpha = [[a, 1 - a, -1], [-0.5 * (13 - a), 0.5 * (15 + a)],
                             [-0.1 * (29 - a), -0.1 * (31 + a)],
                             [-(49 - a) / 120, (111 + a) / 120]]
            if K == 3:
                if L == 1:
                    alpha = [[a, b, 1 - a - b, -1],
                             [(5 + 4 * a - b) / 12, (-4 + 4 * a + 2 * b) / 3,
                              (23 + 4 * a + 5 * b) / 12]]
                if L == 2:  # TODO: This is wrong (for a = b = 0)
                    alpha = [[a, b, 1 - a - b, -1],
                             [(581 + 112 * a + 11 * b) / 240,
                              (38 + 16 * a + 8 * b) / 15,
                              (-949 + 112 * a + 101 * b) / 240],
                             [(173 + 16 * a + 3 * b) / 240, (27 + b) / 6,
                              (637 - 16 * a - 13 * b) / 240]]
            if K == 4:
                if L == 1:
                    alpha = [[a, b, c, 1 - a - b - c, -1],
                             [(-9 + 9 * a + c) / 24,
                              (37 + 27 * a + 8 * b - 5 * c) / 24,
                              (-59 + 27 * a + 32 * b + 19 * c) / 24,
                              (55 + 9 * a + 8 * b + 9 * c) / 24]
                             ]
        elif self.mode == 'implicit':
            if K == 1:
                if L == 1:
                    alpha = [[1, -1], [1 / 2, 1 / 2]]
                if L == 2:
                    alpha = [[1, -1], [1 / 2, 1 / 2], [1 / 12, -1 / 12]]
                if L == 3:
                    alpha = [[1, -1], [1 / 2, 1 / 2], [1 / 10, -1 / 10],
                             [1 / 120, 1 / 120]]
                if L == 4:
                    alpha = [[1, -1], [1 / 2, 1 / 2], [3 / 28, -3 / 28],
                             [1 / 84, 1 / 84], [1 / 1680, -1 / 1680]]
            if K == 2:
                if L == 1:
                    alpha = [[a, 1 - a, -1],
                             [-(1 - 5 * a) / 12, 2 * (1 + a) / 3, (5 - a) / 12]]
                if L == 2:
                    alpha = [[a, 1 - a, -1],
                             [(11 + 101 * a) / 240, 8 * (1 + a) / 15,
                              (101 + 11 * a) / 240],
                             [(3 + 13 * a) / 240, (1 - a) / 6,
                              -(13 + 3 * a) / 240]]
                if L == 3:
                    alpha = [[a, 1 - a, -1],
                             [-(421 - 5669 * a) / 13_440, 64 * (1 + a) / 105,
                              (5669 - 421 * a) / 13_440],
                             [-(47 - 303 * a) / 4480, (1 - a) / 8,
                              -(303 - 47 * a) / 4480],
                             [-(41 - 169 * a) / 40_320, 8 * (1 + a) / 315,
                              (169 - 41 * a) / 40_320]]
            if K == 3:
                if L == 1:
                    alpha = [
                        [a, b, 1 - a - b, -1],
                        [(1 + 8 * a - b) / 24,
                         (-5 + 32 * a + 13 * b) / 24,
                         (19 + 8 * a + 13 * b) / 24,
                         (9 - b) / 24]
                    ]
                if L == 2:
                    alpha = [
                        [a, b, 1 - a - b, -1],
                        [(397 + 7136 * a + 243 * b) / 18_144,
                         (89 + 640 * a + 327 * b) / 672,
                         (313 + 416 * a + 327 * b) / 672,
                         (6893 + 640 * a + 234 * b) / 18_144],
                        [(163 + 1376 * a + 93 * b) / 30_240,
                         (269 - 512 * a + 339 * b) / 3360,
                         (851 - 608 * a - 339 * b) / 3360,
                         (-1284 - 256 * a - 93 * b) / 30_240]
                    ]
            if K == 4:
                if L == 1:
                    alpha = [
                        [a, b, c, 1 - a - b - c, -1],
                        [(-19 + 243 * a - 8 * b + 11 * c) / 720,
                         (53 + 459 * a + 136 * b + 37 * c) / 360,
                         (-11 + 27 * a + 38 * b + 19 * c) / 30,
                         (323 + 189 * a + 136 * b + 173 * c) / 360,
                         (251 - 27 * a - 8 * b - 19 * c) / 720]
                    ]

        else:
            raise ValueError(
                f"mode must be 'implicit' or 'explicit', not {self.mode!r}.")

        try:
            alpha
        except NameError:
            msg = f"Have no coefficients for K = {K}, L = {L} and mode = " \
                  f"{self.mode!r}"
            raise ValueError(msg)

        free_params = self.free_params.copy()
        if free_params is not None:
            # replace str with sp.symbol
            for key, value in free_params.items():
                if isinstance(key, str):
                    del free_params[key]
                    free_params[sp.symbols(key)] = value

            # by default set free params to 0 (seems to work)
            if a not in free_params:
                free_params[a] = 0
            if b not in free_params:
                free_params[b] = 0
            if c not in free_params:
                free_params[c] = 0

            # replace free params (a,b,c) with values
            for rown, row in enumerate(alpha):
                for coln, expr in enumerate(row):
                    try:
                        alpha[rown][coln] = float(expr.subs(free_params))
                    except AttributeError:
                        pass
                    except TypeError:  # Should never reach here
                        msg = 'Missing one of param obr.a, obr.b or obr.c in ' \
                              'consts'
                        raise ValueError(msg)

        return alpha


class Adams(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.L == 1, self.L

    def nabla(self, yd: deque, n: int, i: int = -1) -> np.ndarray:
        """
        Returns the discrete nabla^n function: nabla^n(y[i])
        :param yd: deque[list[np.ndarray[float]]]
        :param n:  int
        :param i: int

        :return: float
        """
        # assert isinstance(i, Integral), repr(i)
        # assert isinstance(n, Integral), repr(n)

        # Take first element of f[i] (ie. the first derivative of y)
        if n == 0:
            return yd[i][1]
        elif n == 1:
            return yd[i][1] - yd[i - 1][1]
        elif n > 1:
            return self.nabla(yd, n - 1, i) - self.nabla(yd, n - 1, i - 1)
        else:
            raise ValueError(f'n must be 0 or greater, not {n!r}.')

    def gamma(self, j: int) -> float:
        """
        Return the adams constant gamma for a given integer j.

        :param j: int

        :return: float 
        """
        # assert isinstance(j, Integral), repr(j)
        if j == 0:
            return 1
        elif j >= 1:
            return 1 - sum(self.gamma(m) / (j + 1 - m) for m in range(j))
        else:
            raise ValueError(f'j must be 0 or greater, not {j!r}.')

    def explicit(self, yd: deque, h: float, k: int) -> np.ndarray:
        """
        Return y[n+k] given a list of y, a list of y_derivs (equal to f) and 
        other 
        parameters. Exact same signature as obr.make_obr().

        :param yd: deque[list[np.ndarray[float]]]
            A deque of the previous k derivatives y, seen as a list of 
            [y, dy/dt, ...] of length l. Each element of the list is an 
            ndarray of 
            length len(y) of floats representing the jth derivative of y[i].

        :param h: float 
            timestep

        :param k: int 
            Number of previous steps to use


        :return: np.ndarray
            The next iteration of y, ie. y[len(y) + 1]
        """
        # assert isinstance(h, float) and h > 0, repr(h)
        # assert isinstance(k, int) and k >= 1, repr(k)
        return yd[0][-1] + h*np.sum([self.gamma(j)*self.nabla(yd, n=j, i=-1)
                                     for j in range(k)], axis=0)

        # could replace y[-1] with yd[-1][0] and remove the y paramater
        # from this and obr.obr