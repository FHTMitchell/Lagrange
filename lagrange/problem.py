#problem.py

import os
import warnings
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from . import dsympy
from .timers import Timer
from typing import *
import warnings

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
__verbose__ = True

# sympy symbols
t, zeta, mu  = sp.symbols('t, zeta, mu')
x, y, z = (sp.Function(s)(t) for s in 'xyz')
vx, vy, vz = [sp.diff(s, t) for s in (x, y, z)]
rho = x ** 2 + y ** 2 + z ** 2


try:
    fig_path = os.path.join(__file__, '..', 'figs')
except NameError:
    warnings.warn('fig_path unable to be automatircally created. '
                 'Please set problem.fig_path')


def problems():
    return {cls.__name__.lower(): cls for cls in Problem.__subclasses__()}


class Problem:
    """
    Abstract Base Class. Will not run without further definiton.
    """

    ind_var: sp.symbol = t
    dep_vars: List[sp.Symbol] = NotImplemented

    ODEs: List[List[sp.Expr]] = NotImplemented
    default_mapping: Mapping[sp.Symbol, float] = {}

    def __init__(self, L: int, mapping: dict = None, *, auto_higher_order: bool = True):
        self.name = self.__class__.__name__
        if NotImplemented in (self.dep_vars, self.ODEs):
            raise NotImplementedError('Attempting to initialise ABC')

        assert all(isinstance(row, (list, tuple, np.ndarray)) for row in self.ODEs)
        assert all(all(isinstance(f, sp.Expr) for f in row) for row in self.ODEs)
        assert all(isinstance(x, sp.Function) for x in self.dep_vars), self.dep_vars
        assert isinstance(self.ind_var, sp.Symbol), repr(self.ind_var)

        self.norder = len(self.ODEs)   # The order of the system of ODEs
        self.ncoords = len(self.dep_vars)  # The number of coordinates in the ODEs
        assert not any(len(row) - self.ncoords for row in self.ODEs)
        self._nargs = (L + self.norder) * self.ncoords
        # The (private) number of arguments in funcs

        self.L = L
        self.mapping = self.default_mapping.copy()
        if mapping is None:
            mapping = {}
        for key, value in mapping.items():
            if isinstance(key, str):
                self.mapping[sp.symbols(key)] = value
            elif isinstance(key, sp.Symbol):
                self.mapping[key] = value
            else:
                msg = f'mapping key {key!r}  str or sympy.Symbol, not {type(key).__name__!r}'
                raise TypeError(msg)

        for key, value in self.mapping.items():
            setattr(self, str(key), value)

        if auto_higher_order:
            try:
                self.funcs
            except (NameError, AttributeError):
                self.funcs = []
                self.higher_orders()
            else:
                raise ValueError(f'{self.name}.funcs already implemented yet'
                                  ' auto_higher_order is True')
        else:
            self.funcs = NotImplemented

    def __repr__(self):
        return f'problem.{self.name}(L={self.L})'

    def higher_orders(self):
        timer = Timer()
        if __verbose__:
            print(f'Running {self.name}.higher_orders() to order {self.L + self.norder - 1}.')

        dparams = self.dparams()
        mapping = self.mapping.copy()

        sym_funcs = self.ODEs.copy()
        for _ in range(self.L-1):
            sym_funcs.append([sp.diff(f, t) for f in sym_funcs[-1]])

        for order, row in enumerate(sym_funcs, 1):
            # Get the names of each lambda from the arguments for dsympy.AutoFunc
            # zip() will automatically grab correct number
            names = dparams[(order*self.ncoords)+1:]

            self.funcs.append([dsympy.auto(f, mapping, params=dparams, name=n)
                               for f, n in zip(row, names)])

            if __verbose__:
                print(f'Lambdifying r^({order})(t)... (Time elapsed: {timer()}')

        if __verbose__:
            print(f'Done lamdifying. Time taken: {timer()}')


    def vector_derivs(self, t: float, y: np.array) -> List[np.array]:
        # possibility to include "optimisation funcs" argument and code?

        args = [None] * self._nargs

        for i, yi in enumerate(y):
            args[i] = yi

        for order in range(1, self.L + self.norder - 1):
            for coord in range(self.ncoords):
                index = ((order+1) * self.ncoords) + coord
                ans = self.funcs[order][coord](t, *args)
                args[index] = self.funcs[order][coord](t, *args)

        ydiffs = [y]
        for order in range(1, self.L + 1):
            ydiffs.append(np.array(args[order * self.ncoords:(order * self.ncoords) + len(y)]))

        return ydiffs

    __call__ = vector_derivs

    def dparams(self):
        vars = [str(v).split('(')[0] for v in self.dep_vars]
        d_by = str(self.ind_var)
        params = [d_by] + vars
        for i in range(1, self.norder + self.L):
            for var in vars:
                params.append(f'd{var}_d' + 'd'.join([d_by]*i))

        return params

    @staticmethod
    def plotfig(fig, file=None, show=True, filetype='.png', **kwargs):
        if file is not None:
            if filetype[0] != '.':
                filetype = '.' + filetype
            if '.' in file:
                filetype = ''
            fig.savefig(os.path.join(fig_path, file + filetype))
        if show:
            plt.show()

    def plot(self, t, y, points=None, mode='', file=None, **kwargs):
        raise NotImplementedError()



class SHM(Problem):

    dep_vars = [x]
    default_mapping = {zeta: 0}
    omega = 1
    F = 0
    zeta: float

    v = sp.diff(x, t)
    ODEs = [[v], [-2*zeta*v - x]]


    def analytical_factory(self, y0: np.ndarray) -> Callable:
        from sympy import sin, cos, exp
        omega = self.omega
        zeta = self.zeta  # does exist
        F = self.F
        x0, v0 = y0


        om0, om1, z, f, t, c1, c2, = sp.symbols(
            'omega0, omega1, zeta, F, t, c1, c2')

        mapping = {z: zeta, om0: omega, f: F,
                   om1: omega * abs(1 - zeta ** 2) ** 0.5}

        if omega != 1:
            raise NotImplementedError('omega must be 1.0')
        if F != 0:
            raise NotImplementedError('F must be constant 0.0')

        if zeta == 0:
            x = c1 * cos(t) + c2 * sin(t)
        elif 0 < zeta < 1:
            x = exp(-z * t) * (c1 * cos(om1 * t) + c2 * sin(om1 * t))
        elif zeta == 1:
            x = exp(-t) * (c1 + c2 * t)
        elif zeta > 1:
            x = exp(-z * t) * (c1 * exp(t * om1) + c2 * exp(-t * om1))
        else:
            raise ValueError(
                f'zeta must be greater than or equal to 0, not {zeta}')

        v = sp.diff(x, t)

        xm = (x - x0).subs(mapping).subs(t, 0)
        vm = (v - v0).subs(mapping).subs(t, 0)

        if __verbose__: print(f'x: {xm}, v: {vm}')

        foo = sp.linsolve([xm, vm], c1, c2)
        c1_val, c2_val = (float(v) for v in list(foo)[0])

        if __verbose__: print(f"c1: {c1_val:.4f}, c2: {c2_val:.4f}")

        mapping.update({c1: c1_val, c2: c2_val})

        xm = x.subs(mapping)
        vm = v.subs(mapping)

        if __verbose__: print(f'x: {xm}, \nv: {vm}')

        xf = sp.lambdify(t, xm)
        vf = sp.lambdify(t, vm)

        return xf, vf


    def plot(self, t, y, mode='xve', file=None, data=None, **kwargs):

        fig, axs = plt.subplots(len(mode), sharex=True)
        nplt = 0

        maxerr = errdiff = None
        if 'e' in mode:
            analytic_xf, analytic_vf = self.analytical_factory(y[0, :])
            analytic_x = np.array([analytic_xf(ti) for ti in t])
            analytic_v = np.array([analytic_vf(ti) for ti in t])
            diffx = y[:, 0] - analytic_x
            diffv = y[:, 1] - analytic_v
            maxerr = np.max(np.abs(diffx))
            errdiff = y[2] - y[-1]

        if len(mode) == 1:
            axs = [axs]

        if 'x' in mode:
            axs[nplt].plot(t, y[:, 0], color='b', label='$x_{calc}$')
            axs[nplt].set_ylabel('$x(t)$')
            if 'e' in mode:
                axs[nplt].plot(t, analytic_x, color='k', label='$x_{true}$',
                               linestyle=':')
                axs[nplt].legend()
            nplt += 1

        if 'v' in mode:
            axs[nplt].plot(t, y[:, 1], color='g', label='$v_{calc}$')
            axs[nplt].set_ylabel('$v(t)$')
            if 'e' in mode:
                axs[nplt].plot(t, analytic_v, color='k', label='$v_{true}$',
                               linestyle=':')
                axs[nplt].legend()
            nplt += 1

        if 'e' in mode:
            if 'x' in mode:
                axs[nplt].plot(t, diffx, color='c', label=r'$\mathrm{err}(x)$')
            if 'v' in mode:
                axs[nplt].plot(t, diffv, color='m', label=r'$\mathrm{err}(v)$')
            axs[nplt].set_ylabel('$x_{calc}(t) - x_{true}(t)$')
            axs[nplt].legend()
            nplt += 1

        axs[-1].set_xlabel('$t$')
        axs[-1].set_xlim(0, t[-1])

        if 'title' in kwargs:
            title = kwargs['title']
        else:
            title = rf'Spring with $\zeta = {self.zeta:.2f}$'
            if 'data' is not None:
                for k, v in data.items():
                    if isinstance(v, float):
                        data[k] = format(v, '.2g')
                d = ', '.join(fr"${k}={v}$" for k, v in data.items())
                title += f'\n({d})'
        axs[0].set_title(title)

        self.plotfig(fig, file, **kwargs)
        return fig, axs



class RTBP(Problem):

    dep_vars = [x, y, z]
    r1 = sp.sqrt(x**2 + y**2 + z**2)
    r2 = sp.sqrt((x-1)**2 + y**2 + z**2)

    ODEs = [[vx, vy, vz],
            [2*vy + x - mu - (1-mu)*x*r1**-3 - mu*(x-1)*r2**-3,
             -2*vx + y - (1-mu)*y*r1**3 - mu*y*r2**-3,
             -(mu-1)*z*r1**-3 - mu*z*r2**-3]
            ]

    earth_mass = 5.9721986e24
    moon_mass = 7.3459e22
    default_mapping = {mu: moon_mass/(earth_mass + moon_mass)}

    R = 1  # dimensionless
    theta_dot = 1
    mu: float  # will be defined in self.__init__

    default_points = {'2d': [[0,0], [1,0]]}


    def plot(self, t, y, mode='2d', file=None, points=None, **kwargs):

        mode = mode.lower().strip()
        if points is None:
            if mode in self.default_points:
                points = self.default_points[mode]
            else:
                points = ()

        fig, ax = plt.subplots()
        if 'title' in kwargs:
            # noinspection PyStatementEffect
            ax.set_title(kwargs['title'])

        if mode == '2d':
            ax.plot(y[:, 0], y[:, 1], color='b', label='Path')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            fig.tight_layout()
            if 'title' not in kwargs:
                ax.set_title(r'$\mu = {self.mu:.2g}')
        else:
            raise ValueError(f'mode not supported: {mode!r}')


        if isinstance(points, dict):
            for k, v in points.items():
                ax.scatter([v[0]], [v[1]], color='r', marker='o', label=k)
            ax.legend()
        else:
            ax.scatter([e[0] for e in points], [e[1] for e in points], color='r',
                       marker='o')

        self.plotfig(fig, file, **kwargs)
        return fig, ax


    def lagrange_points(self):

        mu = self.mu

        xi = sp.symbols('xi', real=True)

        L4 = [1/2 - mu, np.sqrt(3)/2, 0]
        L5 = [L4[0], -L4[1], 0]

        if mu == self.default_mapping[sp.symbols('mu')]:
            soln = [-1.00512496490921, 0.836181649431693, 1.15625464982094]
        else:
            if __verbose__:
                print('Solving lagrange points')
            f = (1-mu)*abs(xi+mu)**-3*(xi+mu) + mu*abs(xi+mu-1)**-3*(xi+mu-1)-xi
            try:
                soln = list(map(float, sp.solve(f, xi, quick=True)))
            except TypeError:  # Bug in sympy converts some values to x + 0j
                warnings.warn('Error encountered in sympy.solve. Reverting to '
                              'linear approximation')
                soln = np.multiply(self.R,
                                   [1-(mu/3)**(1/3), 1+(mu/3)**(1/3), -1-(5/12)*mu])

        L3 = [soln[0], 0, 0]
        L1 = [soln[1], 0, 0]
        L2 = [soln[2], 0, 0]

        # noinspection PyTypeChecker
        return dict(enumerate(map(np.array, [L1, L2, L3, L4, L5]), 1))






class Lagrange(RTBP):
    # TODO: Largrange.ODEs is broken

    gamma: float
    c: List[float]
    ODEs = NotImplemented

    default_points = {'2d': [[0,0]]}

    def __init__(self, L: int = 1, mapping: dict = None, lagrange_point: int = 1,
                 legendre_order: int = 3, **kwargs):

        assert legendre_order >= 2, repr(legendre_order)
        assert 1 <= lagrange_point <= 5, repr(lagrange_point)
        if lagrange_point in (3,4,5):
            raise NotImplementedError('L3, L4 and L5 not implemented')

        self.legendre_order = legendre_order
        self.lagrange_point = lagrange_point

        if mapping is None:
            mapping = {}

        if 'mu' in mapping:
            self.mu = mapping['mu']
        elif mu in mapping:
            self.mu = mapping[mu]
        else:
            self.mu = self.default_mapping[mu]

        self.lagrange_coord = self.lagrange_points()[lagrange_point]
        self.gamma = 1 - self.lagrange_coord[0]  # TODO: needs checking
        self.c = c = self.legendre_coeffs(self.legendre_order)

        summation = sum(c[n] * rho**n * sp.legendre(n, x/rho)
                        for n in range(3, self.legendre_order+1))

        ax = 2*vy + (1+2*c[2])*x + sp.diff(summation, x)
        ay = -2*vx - (c[2]-1)*y + sp.diff(summation, y)
        az = -c[2]*z + sp.diff(summation, z)

        self.ODEs = [[vx, vy, vz], [ax, ay, az]]

        # noinspection PyArgumentList
        super().__init__(L, mapping, **kwargs)

    def legendre_coeffs(self, N):

        mu = self.mu
        gamma = self.gamma

        return [gamma**-3 * (mu+(-1)**n) * (1-mu) * (gamma/(1-gamma))**(n+1)
                for n in range(N+1)]

    def convert_coords(self, r, *, inv=False):

        r = np.array(r)
        assert r.shape == (3,)

        if not inv:
            newr = r / self.gamma
            newr[0] = (r[0] - 1 + self.mu + self.gamma)/self.gamma
        else:
            newr = r * self.gamma
            newr[0] = (r[0] * self.gamma) - self.gamma - self.mu + 1

        return newr

    def plot(self, t, y, mode='2d', file=None, points=None, **kwargs):

        mode = mode.lower().strip()

        if 'title' not in kwargs:
            if mode == '2d':
                title = fr'Motion at L{self.lagrange_point} ($\mu = {self.mu:.2g}$)'

            try:
                # noinspection PyUnboundLocalVariable
                kwargs['title'] = title
            except NameError:
                pass

        fig, ax = super().plot(t, y, mode, file, points, **kwargs)



        return fig, ax