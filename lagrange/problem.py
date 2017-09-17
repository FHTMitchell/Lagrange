# problem.py

import os
import warnings
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from . import dsympy
from .bases import Base
from .timers import Timer

__author__ = 'Mitchell, FHT'
__date__ = (2017, 8, 20)
VERBOSE = True
DEBUG = True

# sympy symbols
t, zeta, mu = sp.symbols('t, zeta, mu')
x, y, z = (sp.Function(s)(t) for s in 'xyz')
vx, vy, vz = [sp.diff(s, t) for s in (x, y, z)]
rho = sp.sqrt(x**2 + y**2 + z**2)  # Todo: Are we sure rho !-> sqrt(rho)

try:
    fig_path = os.path.join(__file__, '..', 'figs')
except NameError:
    warnings.warn('fig_path unable to be automatircally created. '
                  'Please set problem.fig_path before saving figures')


def problems():
    return {c.__name__.lower(): c for c in Problem.subclasses()}


class Problem(Base):
    """
    Abstract Base Class. Will not run without further definiton.
    """

    ind_var: sp.symbol = t
    dep_vars: List[sp.Symbol] = NotImplemented

    ODEs: List[List[sp.Expr]] = NotImplemented
    default_mapping: Mapping[sp.Symbol, float] = {}

    def __init__(self, L: int, mapping: dict = None, *,
                 auto_higher_order: bool = True):
        self.name = self.__class__.__name__
        if NotImplemented in (self.dep_vars, self.ODEs):
            raise NotImplementedError('Attempting to initialise ABC')

        assert all(
            isinstance(row, (list, tuple, np.ndarray)) for row in
            self.ODEs), self.ODEs
        assert all(
            all(isinstance(f, sp.Expr) for f in row) for row in
            self.ODEs), self.ODEs
        assert all(
            isinstance(x, sp.Function) for x in self.dep_vars), self.dep_vars
        assert isinstance(self.ind_var, sp.Symbol), repr(self.ind_var)

        self.norder = len(self.ODEs)  # The order of the system of ODEs
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
                msg = f'mapping key {key!r}  str or sympy.Symbol, not ' \
                      f'{type(key).__name__!r}'
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
        return self._make_repr('L', 'mapping')

    def higher_orders(self):
        timer = Timer()
        if VERBOSE:
            print(
                f'Running {self.name}.higher_orders() to order '
                f'{self.L + self.norder - 1}.')

        dparams = self.dparams()
        mapping = self.mapping.copy()

        sym_funcs = self.ODEs.copy()
        for _ in range(self.L - 1):
            sym_funcs.append([sp.diff(f, t) for f in sym_funcs[-1]])

        for order, row in enumerate(sym_funcs, 1):
            # Get the names of each lambda from the arguments for
            # dsympy.AutoFunc
            # zip() will automatically grab correct number
            names = dparams[(order*self.ncoords) + 1:]

            self.funcs.append([dsympy.auto(f, mapping, params=dparams, name=n)
                               for f, n in zip(row, names)])

            if VERBOSE:
                print(f'Lambdifying r^({order})(t)... (Time elapsed: {timer()}')

        if VERBOSE:
            print(f'Done lamdifying. Time taken: {timer()}')

    def vector_derivs(self, t: float, y: np.ndarray) -> List[np.ndarray]:
        # possibility to include "optimisation funcs" argument and code?

        # assert len(y) == self.

        args = [None] * self._nargs

        for i, yi in enumerate(y):
            args[i] = yi

        for order in range(1, self.L + self.norder - 1):
            for coord in range(self.ncoords):
                index = ((order + 1)*self.ncoords) + coord
                # ans = self.funcs[order][coord](t, *args)
                args[index] = self.funcs[order][coord](t, *args)

        ydiffs = [y]
        for order in range(1, self.L + 1):
            ydiffs.append(np.asarray(
                args[order*self.ncoords:(order*self.ncoords) + len(y)]))

        return ydiffs

    __call__ = vector_derivs  # not sure this works

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

    def cost_init(self, vars, *consts) -> Tuple[float, np.ndarray]:
        raise NotImplementedError()

    def cost_exit(self, t: float, y: np.ndarray):
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
                   om1: omega*abs(1 - zeta**2)**0.5}

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

        if VERBOSE:
            print(f'x: {xm}, v: {vm}')

        foo = sp.linsolve([xm, vm], c1, c2)
        c1_val, c2_val = (float(v) for v in list(foo)[0])

        if VERBOSE:
            print(f"c1: {c1_val:.4f}, c2: {c2_val:.4f}")

        mapping.update({c1: c1_val, c2: c2_val})

        xm = x.subs(mapping)
        vm = v.subs(mapping)

        if VERBOSE:
            print(f'x: {xm}, \nv: {vm}')

        xf = sp.lambdify(t, xm)
        vf = sp.lambdify(t, vm)

        return xf, vf

    def plot(self, t, y, mode='xve', file=None, data=None, **kwargs):

        fig, axs = plt.subplots(len(mode), sharex=True)
        nplt = 0

        maxerr = errdiff = None
        if 'e' in mode:
            analytic_xf, analytic_vf = self.analytical_factory(y[0, :])
            analytic_x = np.asarray([analytic_xf(ti) for ti in t])
            analytic_v = np.asarray([analytic_vf(ti) for ti in t])
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
                    if isinstance(v, str):
                        data[k] = repr(v)
                d = ', '.join(fr"${k}=${v}" for k, v in data.items())
                title += f'\n({d})'
        axs[0].set_title(title)

        self.plotfig(fig, file, **kwargs)
        return fig, axs


class RTBP(Problem):
    from . import bodies

    dep_vars = [x, y, z]
    r1 = sp.sqrt(x**2 + y**2 + z**2)
    r2 = sp.sqrt((x - 1)**2 + y**2 + z**2)

    ODEs = [[vx, vy, vz],
            [2*vy + x - mu - (1 - mu)*x*r1**-3 - mu*(x - 1)*r2**-3,
             -2*vx + y - (1 - mu)*y*r1**3 - mu*y*r2**-3,
             -(mu - 1)*z*r1**-3 - mu*z*r2**-3]
            ]

    default_mapping = {mu: bodies.moon.make_mu(bodies.earth)}

    mu: float  # will be defined in self.__init__

    default_points = {'2d': {'M1': [0, 0], 'M2': [1, 0]}}

    def __init__(self, L: int, mapping: dict = None, **kwargs):
        mapping = self._mapping_mu(mapping)
        super().__init__(L, mapping, **kwargs)

    @classmethod
    def _mapping_mu(cls, mapping):

        if mapping is None:
            mapping = {}

        assert isinstance(mapping, dict), repr(mapping)
        assert all(isinstance(key, (str, sp.Symbol)) for key in mapping), \
            repr(mapping)

        mapping = mapping.copy()

        if 'mu' in mapping:
            mapping[mu] = mapping['mu']
            del mapping['mu']

        if mu in mapping:
            if isinstance(mapping[mu], tuple):
                if len(mapping[mu]) == 2:
                    mapping[mu] = cls.make_mu(*mapping[mu])
                else:
                    raise ValueError("tuple mapping[mu] must be length 2, not "
                                     f"{len(mapping[mu])} ({mapping[mu]})")

        return mapping

    @classmethod
    def from_bodies(cls, L: int, body1: str, body2: str, mapping: dict = None,
                    *args, **kwargs):

        if mapping is None:
            mapping = {}
        mapping['mu'] = cls.make_mu(body1, body2)

        return cls(L, mapping, *args, **kwargs)

    def cost_init(self, vars, *consts) -> Tuple[float, np.ndarray]:
        """
        :param vars:  Union[Tuple[float, float], np.ndarray]
            (vy0, period0)
        :param consts: Union[Tuple[float], np.ndarray]
            (x0,)
        :return:
            period, y0
        """

        assert len(vars) == 2, vars
        assert len(consts) == 1, consts

        x0 = consts[0]
        y0 = 0

        return consts[1], np.array([consts[0], 0, 0, 0, vars[0], 0])

    def cost_exit(self, t, y):
        """

        :param t:
        :param y:
        :return:
            np.array([ dx, dy ])
        """

        return y[-1, 0:2] - y[0, 0:2]



    def plot(self, t, y, mode='2d', file=None, points=None, **kwargs):

        mode = mode.lower().strip()
        if points is None:
            points = self.default_points.get(mode, {})

        if mode == '2d':
            fig, ax = plt.subplots()
            ax.plot(y[:, 0], y[:, 1], color='b', label='r(t)')
            ax.scatter(y[0, 0], y[0, 1], color='g', label='r(t=0)', marker='+')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            # fig.tight_layout()
            ax.set_title(kwargs.get('title', fr'$\mu = {self.mu:.2g}$'))

        elif mode == 'xy':
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(t, y[:, 0], color='b', label='$x$')
            ax[1].plot(t, y[:, 1], color='g', label='$y$')
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$y$')
            ax[0].set_ylabel('$x$')
            ax[0].set_title(kwargs.get('title', fr'$\mu = {self.mu:.2g}$'))

        else:
            raise ValueError(f'mode not supported: {mode!r}')

        if mode in ('2d', '3d') and points:
            if isinstance(points, dict):
                for k, v in points.items():
                    ax.scatter([v[0]], [v[1]], marker='o', label=k)
                ax.legend()
            else:
                ax.scatter([e[0] for e in points], [e[1] for e in points],
                           marker='o')

        self.plotfig(fig, file, **kwargs)
        return fig, ax

    def lagrange_points(self):

        mu = self.mu

        xi = sp.symbols('xi', real=True)

        L4 = [1/2 - mu, np.sqrt(3)/2, 0]
        L5 = [L4[0], -L4[1], 0]

        # hardcoed values
        if mu == self.make_mu('earth', 'moon'):
            soln = [-1.00512496490921, 0.836181649431693, 1.15625464982094]
        elif mu == self.make_mu('sun', 'earth'):
            soln = [-1.00000125, 0.99002589, 1.01003482, ]
        else:
            if VERBOSE:
                print('Solving lagrange points')
            f = (1 - mu)*abs(xi + mu)**-3*(xi + mu) + mu*abs(
                xi + mu - 1)**-3*(xi + mu - 1) - xi
            try:
                soln = list(map(float, sp.solve(f, xi, quick=True)))
            except TypeError:  # Bug in sympy converts some values to x + 0j
                warnings.warn('Error encountered in sympy.solve. Reverting to '
                              'linear approximation')
                soln = [1 - (mu/3)**(1/3),
                        1 + (mu/3)**(1/3),
                        -1 - (5/12)*mu]

        soln.sort()  # L3 < L1 < L2
        L3 = [soln[0], 0, 0]
        L1 = [soln[1], 0, 0]
        L2 = [soln[2], 0, 0]

        # noinspection PyTypeChecker
        return dict(enumerate(map(np.asarray, [L1, L2, L3, L4, L5]), 1))

    @classmethod
    def make_mu(cls, m1: Union[str, float], m2: Union[str, float]):

        if isinstance(m1, str):
            m1 = cls.bodies.get_body(m1)
        if isinstance(m2, str):
            m2 = cls.bodies.get_body(m2)

        if isinstance(m1, cls.bodies.Body):
            m1 = m1.mass
        if isinstance(m2, cls.bodies.Body):
            m2 = m2.mass

        if m2 > m1:
            m2, m1 = m1, m2

        return m2/(m1 + m2)

    def convert(self, r: np.ndarray, R12: float, total_mass: float, *,
                inv: bool = False) -> np.ndarray:

        r = np.array(r, copy=True)
        assert r.shape == (6,)

        convert_array = [R12]*3 + [np.sqrt(self.bodies.G*total_mass/R12)]*3
        if not inv:
            r /= convert_array
        else:
            r *= convert_array

        return r

    def convert_from_bodies(self, r: np.ndarray, body1: str, body2: str) \
            -> np.ndarray:

        body1 = self.bodies.get_body(body1)
        body2 = self.bodies.get_body(body2)

        if body2.mass > body1.mass:
            body1, body2 = body2, body1

        r12 = body2.sma

        return self.convert(r, r12, body1.mass + body2.mass)




class Lagrange(RTBP):
    # TODO: Largrange.ODEs is broken (needs conversions)

    gamma: float
    c: List[float]
    ODEs = NotImplemented  # created on init

    default_points = {}

    def __init__(self, L: int = 1, mapping: Dict[str, float] = None,
                 lagrange_point: int = 1, legendre_order: int = 3, **kwargs):

        assert legendre_order >= 2, repr(legendre_order)
        assert isinstance(lagrange_point, int), lagrange_point
        assert 1 <= lagrange_point <= 5, repr(lagrange_point)
        if lagrange_point in (3, 4, 5):
            raise NotImplementedError('L3, L4 and L5 not implemented')

        self.legendre_order = legendre_order
        self.lagrange_point = lagrange_point
        self.default_points['2d'] = {f'L{lagrange_point}': [0, 0]}

        mapping = self._mapping_mu(mapping)

        if 'mu' in mapping:
            self.mu = mapping['mu']
        elif mu in mapping:
            self.mu = mapping[mu]
        else:
            self.mu = self.default_mapping[mu]

        self.lagrange_coord = self.lagrange_points()[lagrange_point]
        if lagrange_point == 1:
            self.gamma = 1 - self.lagrange_coord[0]  # TODO: needs checking
        else:
            self.gamma = self.lagrange_coord[0] - 1

        self.c_syms = sp.symbols([f'c{i}'
                                  for i in range(self.legendre_order + 1)])
        self.c = dict(zip(self.c_syms, self.legendre_coeffs()))

        self.ODEs = [[vx, vy, vz], self.make_ODEs(use_hardcoded=True)]
        # todo

        # noinspection PyArgumentList
        super().__init__(L, mapping, **kwargs)

    def make_ODEs(self, display=None, use_hardcoded=True):

        if use_hardcoded or DEBUG:  # debug
            c2 = self.c_syms[2]
            dax = 2*vy + (1 + 2*c2)*x
            day = -2*vx + (1 - c2)*y
            daz = -c2*z
            if self.legendre_order >= 3:
                c3 = self.c_syms[3]
                dax += 1.5*c3*(2*x**2 - y**2 - z**2)
                day += - 3*c3*x*y
                daz += - 3*c3*x*z
            if self.legendre_order >= 4:
                c4 = self.c_syms[4]
                dax += 2*c4*x*(2*x**2 - 3*y**2 - 3*z**2)
                day += - 1.5*c4*y*(4*x**2 - y**2 - z**2)
                daz += - 1.5*c4*z*(4*x**2 - y**2 - z**2)
            if self.legendre_order >= 5:
                raise ValueError('Hardcoded cannot be implemented for '
                                 'legendre_order > 4')

        if (not use_hardcoded) or DEBUG:
            summation = sum(self.c_syms[n]*rho**n*sp.legendre(n, x/rho)
                            for n in range(3, self.legendre_order + 1))

            ax = 2*vy + (1 + 2*self.c_syms[2])*x + sp.diff(summation, x)
            ay = -2*vx - (self.c_syms[2] - 1)*y + sp.diff(summation, y)
            az = -self.c_syms[2]*z + sp.diff(summation, z)

        if DEBUG:
            sax, say, saz = map(sp.simplify, (ax, ay, az))
            Ea = [sp.simplify(d - s) for d, s
                  in zip((dax, day, daz), (sax, say, saz))]
            if VERBOSE:
                print(f'legendre functions are correct if 0: {Ea}')

        if use_hardcoded:
            ax, ay, az = dax, day, daz

        if display == 'show':
            pass
        elif display == 'simple':
            ax, ay, az = map(sp.simplify, (ax, ay, az))
        elif display is None:  # NORMAL BEHAVIOR
            ax, ay, az = (expr.simplify().subs(self.c) for expr in (ax, ay, az))
        else:
            raise ValueError(f'display value not valid: {display!r}')

        return ax, ay, az

    def legendre_coeffs(self):

        mu = self.mu
        gamma = self.gamma
        N = self.legendre_order

        return [gamma**-3*(mu + (-1)**n*(1 - mu)*
                           (gamma/(1 - gamma))**(n + 1))
                for n in range(N + 1)]

    def convert(self, r: np.ndarray, R12: float, total_mass: float, *,
                inv: bool = False) -> np.ndarray:
        # Todo: Lagrange.convert: Units -> Lagrange centred dimensionless

        r = super().convert(r, R12, total_mass, inv=inv)
        return self.convert_from_barycentric(r, dim=None)

    def convert_from_barycentric(self, value: np.ndarray, dim: str = None,
                                 inv: bool = False,
                                 barycentric=False) -> np.ndarray:

        if not dim:  # assume vector of form [x,y,z,vx,vy,vz]
            r = np.array(value)
            assert r.shape == (6,), (r, r.shape)
            newr = np.empty(6)
            newr[0] = self.convert_from_barycentric(r[0], 'x', inv)
            newr[1:3] = self.convert_from_barycentric(r[1:3], 'y', inv)
            newr[3:] = self.convert_from_barycentric(r[3:], 'v', inv)
            return newr

        if not inv:
            if dim == 'x':  # x ps
                if not barycentric:
                    return (value - 1 + self.gamma)/self.gamma
                else:
                    return (value - 1 + self.mu + self.gamma)/self.gamma
            if dim in 'yz':  # y, z pos
                return value/self.gamma
            if 'v' in dim:  # velocity
                return self.gamma**(-5/2)*value
            if dim == 't':  # time
                return value*self.gamma**(3/2)
        else:
            if dim == 'x':
                if not barycentric:
                    return value*self.gamma - self.gamma + 1
                else:
                    return value*self.gamma - self.gamma - self.mu + 1
            if dim in 'yz':
                return value*self.gamma
            if 'v' in dim:
                return self.gamma**(5/2)*value
            if dim == 't':
                return value/self.gamma**(3/2)

        raise ValueError(f'dim must only contain "vxyzt", not {dim!r}')


    def plot(self, t, y, mode='2d', file=None, points=None, **kwargs):

        mode = mode.lower().strip()  # "2D " -> "2d"

        if 'title' not in kwargs:
            if mode == '2d':
                title = fr'Motion at L{self.lagrange_point} ($\mu = ' \
                        fr'{self.mu:.2g}$)'

            try:
                # noinspection PyUnboundLocalVariable
                kwargs['title'] = title
            except NameError:
                pass

        fig, ax = super().plot(t, y, mode, file, points, **kwargs)

        return fig, ax